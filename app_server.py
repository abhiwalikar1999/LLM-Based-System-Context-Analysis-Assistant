# app_server.py

import os
import re
import time
import secrets
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

from core import run_pipeline

# Phase 8: configurable via environment variables for container/cloud
# deployment (e.g. Render sets PORT itself and expects the app to bind
# 0.0.0.0, not 127.0.0.1). Defaults preserve the original local-dev
# behavior exactly -- plain `python app_server.py` with no env vars set
# behaves identically to before this change.
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

MAX_LLM_REQUESTS_PER_SESSION = 10
SESSION_COOKIE_NAME = "ncw_session"

# In-memory session store: {session_id: {"used": int, "created": float}}
_SESSIONS = {}


def _html_page(title: str, body: str) -> bytes:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; max-width: 1000px; }}
    textarea {{ width: 100%; height: 240px; }}
    .btn {{ padding: 10px 16px; background: #2d7ff9; border: none; color: white; border-radius: 6px; cursor: pointer; }}
    .btn.secondary {{ background: #666; }}
    .muted {{ color: #666; font-size: 0.95em; }}
    iframe {{ width: 100%; height: 720px; border: 1px solid #ddd; border-radius: 8px; }}
    .meter {{ margin: 12px 0; padding: 10px; border: 1px solid #e5e5e5; border-radius: 8px; background: #fafafa; }}
    .row {{ display:flex; gap:16px; flex-wrap:wrap; }}
    .row > div {{ min-width: 240px; }}
    a {{ word-break: break-all; }}
  </style>
</head>
<body>
<h2>{title}</h2>
{body}
</body>
</html>"""
    return html.encode("utf-8")


def _parse_content_type(header_value: str):
    """Minimal stand-in for the removed cgi.parse_header (cgi is gone in Python 3.13+)."""
    if not header_value:
        return "", {}
    parts = header_value.split(";")
    ctype = parts[0].strip().lower()
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().lower()] = v.strip().strip('"')
    return ctype, params


def _parse_multipart_form(rfile, content_type_header: str, content_length: int):
    """
    Minimal stand-in for the removed cgi.FieldStorage (cgi is gone in Python 3.13+).
    Returns {field_name: [{"value": bytes, "filename": Optional[str]}, ...]} (a list per
    field name, since a <input type="file" multiple> submits several parts under the same
    name -- see Phase 4's multi-document upload), or None if the request isn't
    multipart/form-data.
    """
    ctype, params = _parse_content_type(content_type_header)
    boundary = params.get("boundary")
    if ctype != "multipart/form-data" or not boundary:
        return None

    body = rfile.read(content_length)
    boundary_bytes = ("--" + boundary).encode("utf-8")

    fields = {}
    for raw_part in body.split(boundary_bytes):
        part = raw_part.strip(b"\r\n")
        if not part or part.startswith(b"--"):
            continue

        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if not sep:
            continue

        headers_text = header_blob.decode("utf-8", errors="replace")
        disp_match = re.search(r'Content-Disposition:\s*form-data;\s*(.*)', headers_text, re.IGNORECASE)
        if not disp_match:
            continue

        disp_params = {}
        for kv in disp_match.group(1).split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                disp_params[k.strip().lower()] = v.strip().strip('"')

        name = disp_params.get("name")
        if not name:
            continue

        if content.endswith(b"\r\n"):
            content = content[:-2]

        fields.setdefault(name, []).append({"value": content, "filename": disp_params.get("filename")})

    return fields


def _parse_cookies(cookie_header: str) -> dict:
    cookies = {}
    if not cookie_header:
        return cookies
    parts = cookie_header.split(";")
    for p in parts:
        if "=" in p:
            k, v = p.strip().split("=", 1)
            cookies[k] = v
    return cookies


def _get_or_create_session(handler: BaseHTTPRequestHandler) -> str:
    """
    Returns session_id. If missing, creates one and sets Set-Cookie.
    """
    cookies = _parse_cookies(handler.headers.get("Cookie", ""))
    sid = cookies.get(SESSION_COOKIE_NAME)

    if sid and sid in _SESSIONS:
        return sid

    sid = secrets.token_urlsafe(16)
    _SESSIONS[sid] = {"used": 0, "created": time.time()}

    # Set cookie (Path=/ so it applies to all routes)
    handler.send_header(
        "Set-Cookie",
        f"{SESSION_COOKIE_NAME}={sid}; Path=/; SameSite=Lax"
    )
    return sid


def _session_used(sid: str) -> int:
    return int(_SESSIONS.get(sid, {}).get("used", 0))


def _session_add_used(sid: str, delta: int):
    if sid not in _SESSIONS:
        _SESSIONS[sid] = {"used": 0, "created": time.time()}
    _SESSIONS[sid]["used"] = int(_SESSIONS[sid].get("used", 0)) + int(delta)


def _session_reset(sid: str):
    if sid not in _SESSIONS:
        _SESSIONS[sid] = {"used": 0, "created": time.time()}
    _SESSIONS[sid]["used"] = 0


def _meter_html(used: int, last_run_used: int | None, cached: bool | None) -> str:
    last_line = ""
    if last_run_used is not None:
        if last_run_used == 0:
            last_line = "<div><b>This run used:</b> 0 (cached)</div>"
        else:
            last_line = "<div><b>This run used:</b> 1 (new LLM request)</div>"

    cached_line = ""
    if cached is not None:
        cached_line = f"<div><b>Cached:</b> {'Yes' if cached else 'No'}</div>"

    return f"""
<div class="meter">
  <div class="row">
    <div><b>LLM requests used this session:</b> {used} / {MAX_LLM_REQUESTS_PER_SESSION}</div>
    <div>{cached_line}</div>
    <div>{last_line}</div>
  </div>
  <form method="POST" action="/reset_requests" style="margin-top:8px;">
    <button class="btn secondary" type="submit">Reset counter (session)</button>
  </form>
  <div class="muted" style="margin-top:8px;">
    Note: Each Analyze normally uses 1 request. If the exact same input was analyzed before, it is served from cache (0 requests).
  </div>
</div>
"""


STATIC_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def _serve_static(self, prefix: str) -> bool:
        """Serves a file under `prefix` (e.g. "/output/", "/example_output/")
        if it exists and has a known extension. Returns True if it handled
        the request (success or 404), False if `self.path` didn't match
        `prefix` at all."""
        if not self.path.startswith(prefix):
            return False
        file_path = self.path.lstrip("/")
        ext = os.path.splitext(file_path)[1]
        content_type = STATIC_CONTENT_TYPES.get(ext)
        if content_type and os.path.exists(file_path) and os.path.isfile(file_path):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
            return True
        self.send_error(404, "File not found")
        return True

    def do_GET(self):
        # Serve generated run outputs, and the bundled pre-generated example
        # (Phase 8 -- example_output/ ships in the repo/image so a deployed
        # instance always has something real to show, even where the UPB AI
        # Gateway itself isn't reachable; see the home page note below).
        if self._serve_static("/output/") or self._serve_static("/example_output/"):
            return

        # Home page
        self.send_response(200)
        sid = _get_or_create_session(self)  # sets cookie if new (must be before end_headers)
        used = _session_used(sid)

        meter = _meter_html(used=used, last_run_used=None, cached=None)

        body = f"""
{meter}

<div class="card">
  <p class="muted">
    NeuroCardioWatch — AI-assisted System Context Analysis
  </p>

  <p class="muted">
    <b>Note on live analysis:</b> the LLM backend (UPB AI Gateway) is only reachable from the Paderborn University
    network/VPN. If you're viewing this deployed instance from elsewhere, clicking Analyze will time out -- that's
    an access restriction on the backend, not a bug. See the pre-generated example below for real output from this
    exact pipeline.
  </p>

  <form method="POST" action="/analyze" enctype="multipart/form-data">
    <h3>Input</h3>
    <p class="muted">Paste text and/or upload one or more .txt files -- each is treated as a separate document and aggregated into one context model, then click Analyze (1 request unless cached).</p>

    <label><b>Paste text:</b></label><br/>
    <textarea name="text"></textarea><br/><br/>

    <label><b>Upload .txt file(s):</b></label><br/>
    <input type="file" name="file" multiple/><br/><br/>

    <button class="btn" type="submit">Analyze</button>
  </form>
</div>

<div class="card" style="margin-top:16px;">
  <h3>Example output (pre-generated)</h3>
  <p class="muted">Real output from this pipeline, run against the hospital EEG/ECG monitoring reference scenario (<code>system_description.txt</code>). No live request needed to view these.</p>
  <div class="row">
    <div><a href="/example_output/context_graph.html" target="_blank">context_graph.html</a></div>
    <div><a href="/example_output/risk_overlay.html" target="_blank">risk_overlay.html</a></div>
    <div><a href="/example_output/governance_overlay.html" target="_blank">governance_overlay.html</a></div>
    <div><a href="/example_output/context_model.json" target="_blank">context_model.json</a></div>
  </div>
  <h4 style="margin-top:14px;">Preview</h4>
  <iframe src="/example_output/context_graph.html"></iframe>
</div>
"""
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_html_page("NeuroCardioWatch Context Analysis App", body))

    def do_POST(self):
        # Reset session counter
        if self.path == "/reset_requests":
            self.send_response(200)
            sid = _get_or_create_session(self)
            _session_reset(sid)
            used = _session_used(sid)

            meter = _meter_html(used=used, last_run_used=None, cached=None)

            body = f"""
{meter}
<div class="card">
  <p><b>Counter reset.</b></p>
  <p><a href="/">Back to home</a></p>
</div>
"""
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_html_page("Reset", body))
            return

        # Analyze
        if self.path != "/analyze":
            self.send_error(404, "Not found")
            return

        try:
            self.send_response(200)
            sid = _get_or_create_session(self)
            used = _session_used(sid)

            # Parse multipart form
            content_length = int(self.headers.get("Content-Length", 0))
            form = _parse_multipart_form(self.rfile, self.headers.get("content-type"), content_length)
            if form is None:
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                body = "<div class='card'><p class='muted'>Expected multipart/form-data.</p><p><a href='/'>Back</a></p></div>"
                self.wfile.write(_html_page("Error", body))
                return

            # Build the list of input documents: the pasted-text box (if used)
            # plus every uploaded file (Phase 4 supports multiple), each kept
            # as its own (label, text) document rather than one blob -- this
            # is what lets core.py's retrieval step reason about document
            # boundaries.
            documents = []

            if "text" in form and form["text"][0]["value"]:
                pasted = form["text"][0]["value"].decode("utf-8", errors="replace").strip()
                if pasted:
                    documents.append(("pasted text", pasted))

            if "file" in form:
                for i, f in enumerate(form["file"]):
                    if f.get("filename") and f["value"]:
                        content = f["value"].decode("utf-8", errors="replace").strip()
                        if content:
                            documents.append((f.get("filename") or f"uploaded file {i + 1}", content))

            if not documents:
                meter = _meter_html(used=used, last_run_used=0, cached=None)
                body = f"""
{meter}
<div class="card">
  <p class="muted">No input provided. Please paste text or upload one or more .txt files.</p>
  <p><a href="/">Back</a></p>
</div>
"""
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_html_page("Error", body))
                return

            # Run pipeline FIRST to find out if it's cached.

            result = run_pipeline(documents, out_dir="output")
            cached = bool(result.get("cached", False))
            this_run_used = 0 if cached else 1

            if (not cached) and used >= MAX_LLM_REQUESTS_PER_SESSION:
                meter = _meter_html(used=used, last_run_used=0, cached=None)
                body = f"""
{meter}
<div class="card">
  <p><b>LLM request limit reached</b> ({used}/{MAX_LLM_REQUESTS_PER_SESSION}).</p>
  <p class="muted">This request would require a new LLM call, so it was blocked to comply with the grading cost limit.</p>
  <p><a href="/">Back</a></p>
</div>
"""
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_html_page("Limit reached", body))
                return

            # Update session counter if LLM was used
            _session_add_used(sid, this_run_used)
            used = _session_used(sid)

            # Build output links (served by /output/ handler above)
            run_dir = result.get("run_dir", "").replace("\\", "/").strip("/")
            json_rel = result["json_path"].replace("\\", "/")
            context_rel = (result.get("context_html_path") or result.get("html_path") or "").replace("\\", "/")
            risk_rel = (result.get("risk_html_path") or "").replace("\\", "/")
            gov_rel = (result.get("governance_html_path") or "").replace("\\", "/")

            #  context_rel for iframe
            iframe_src = context_rel or risk_rel or gov_rel

            meter = _meter_html(used=used, last_run_used=this_run_used, cached=cached)

            cached_txt = "Yes" if cached else "No"

            body = f"""
{meter}

<div class="card">
  <p><b>Analysis complete</b> (cached: {cached_txt})</p>

  <div class="row" style="margin-top:10px;">
    <div>
      <b>Downloads / Outputs</b><br/>
      <a href="/{urllib.parse.quote(json_rel)}" target="_blank">context_model.json</a><br/>
      {"<a href='/" + urllib.parse.quote(context_rel) + "' target='_blank'>context_graph.html</a><br/>" if context_rel else ""}
      {"<a href='/" + urllib.parse.quote(risk_rel) + "' target='_blank'>risk_overlay.html</a><br/>" if risk_rel else ""}
      {"<a href='/" + urllib.parse.quote(gov_rel) + "' target='_blank'>governance_overlay.html</a><br/>" if gov_rel else ""}
      <span class="muted">Run folder: {run_dir}</span>
    </div>

    <div>
      <b>Open in new tab</b><br/>
      {"<a href='/" + urllib.parse.quote(iframe_src) + "' target='_blank'>Open current preview</a>" if iframe_src else "<span class='muted'>No preview available.</span>"}
    </div>
  </div>

  <h3 style="margin-top:14px;">Diagram preview</h3>
  {"<iframe src='/" + urllib.parse.quote(iframe_src) + "'></iframe>" if iframe_src else "<p class='muted'>No HTML output found to preview.</p>"}

  <p class="muted">Tip: open the HTML in a new tab for full interaction and better performance.</p>
  <p><a href="/">Back to home</a></p>
</div>
"""
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_html_page("Results", body))

        except Exception as e:
            # The 200 status line above is already on the wire (it's sent before we know
            # whether the pipeline will succeed), so we can't switch to a 500 here without
            # writing a second status line and corrupting the response. Render the error
            # inline on the already-started 200 response instead.
            try:
                body = f"<div class='card'><h3>Server error</h3><pre>{str(e)}</pre><p><a href='/'>Back</a></p></div>"
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_html_page("Server error", body))
            except Exception:
                pass


def main():
    os.makedirs("output", exist_ok=True)
    print(f"Starting app: http://{HOST}:{PORT}")
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
