# Phase 8: single-container deployment.
#
# HOST/PORT default to 0.0.0.0/8000 here (vs. app_server.py's own
# 127.0.0.1/8000 default for plain local `python app_server.py` runs) --
# a container must bind 0.0.0.0 to be reachable from outside itself, and
# cloud platforms like Render override PORT at runtime anyway.
#
# The UPB_AI_GATEWAY_KEY is intentionally NOT baked in here -- it's a
# secret, passed at `docker run`/deploy time via -e UPB_AI_GATEWAY_KEY=...
# or the host platform's environment-variable settings, same as local use.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer) so `docker build` doesn't
# re-resolve/re-download packages on every code change, only when
# requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
EXPOSE 8000

CMD ["python", "app_server.py"]
