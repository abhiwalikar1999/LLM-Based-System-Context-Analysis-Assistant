# llm_engine.py
import os
import json
from openai import OpenAI

BASE_URL = "https://ai-gateway.uni-paderborn.de/v1"
MODEL_NAME = "gwdg.qwen3-30b-a3b-instruct-2507"

SYSTEM_PROMPT = """
You are an AI that extracts system context information from user text.

Return ONLY valid JSON in this exact format:

{
  "entities": [
    {"id": "id1", "label": "name", "type": "component|actor|external_system|requirement"}
  ],
  "relationships": [
    {"source_id": "id1", "target_id": "id2", "label": "interaction"}
  ],
  "risks": [
    {"id": "r1", "description": "text", "severity": "low|medium|high",
     "related_nodes": ["id1"]}
  ],
  "assumptions": [
    {"id": "a1", "description": "text", "related_nodes": ["id1"]}
  ]
}

Rules:
- Output JSON only. No extra text.
- Use ONLY these types: component, actor, external_system, requirement.
- Use snake_case ids (e.g., eeg_ecg_sensors, analytics_backend, ehr).
- Relationship labels must be short verbs in snake_case.
- Prefer directional data/control flow from source_id -> target_id.
- Do not invent components not supported by the text.
"""

def _read_key_from_file(path: str = "UPB_AI_GATEWAY_KEY.txt") -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def _get_key() -> str:
    key = os.getenv("UPB_AI_GATEWAY_KEY", "").strip()
    if not key:
        key = _read_key_from_file()
    if not key:
        raise RuntimeError(
            "Missing API key. Set environment variable UPB_AI_GATEWAY_KEY "
            "or create file UPB_AI_GATEWAY_KEY.txt containing the key."
        )
    return key

def _extract_json_object(text: str) -> str:
    t = (text or "").strip()
    # Remove markdown fences if any
    if t.startswith("```"):
        brace = t.find("{")
        if brace != -1:
            t = t[brace:]
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON found in model response. Raw response:\n{t[:500]}")
    return t[start:end + 1]

def ask_chatgpt(text: str) -> dict:
    key = _get_key()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0
    )

    result_text = response.choices[0].message.content
    json_text = _extract_json_object(result_text)
    return json.loads(json_text)
