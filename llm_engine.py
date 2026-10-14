# llm_engine.py
import os
from typing import List, Literal

from openai import OpenAI
from pydantic import BaseModel, model_validator

BASE_URL = "https://ai-gateway.uni-paderborn.de/v1"
MODEL_NAME = "gwdg.qwen3-30b-a3b-instruct-2507"

SYSTEM_PROMPT = """
You are an AI that extracts system context information from user text.

Return ONLY valid JSON in this exact format:

{
  "entities": [
    {"id": "id1", "label": "name", "type": "component|actor|external_system|requirement",
     "evidence": "the sentence(s) from the input text that support this entity"}
  ],
  "relationships": [
    {"source_id": "id1", "target_id": "id2", "label": "interaction",
     "evidence": "the sentence(s) from the input text that support this relationship"}
  ],
  "risks": [
    {"id": "r1", "description": "text", "severity": "low|medium|high",
     "related_nodes": ["id1"],
     "evidence": "the sentence(s) from the input text that support this risk"}
  ],
  "assumptions": [
    {"id": "a1", "description": "text", "related_nodes": ["id1"],
     "evidence": "the sentence(s) from the input text that support this assumption"}
  ]
}

Rules:
- Output JSON only. No extra text.
- Use ONLY these types: component, actor, external_system, requirement.
- Use snake_case ids (e.g., eeg_ecg_sensors, analytics_backend, ehr).
- Relationship labels must be short verbs in snake_case.
- Prefer directional data/control flow from source_id -> target_id.
- Do not invent components not supported by the text.
- "evidence" must be copied verbatim or near-verbatim from the input text, not paraphrased or invented. If no specific sentence supports an item, use the shortest exact substring of the input that is most relevant.
"""

MAX_ATTEMPTS = 3


# --- Structured output schema -------------------------------------------
#
# Mirrors the JSON shape described in SYSTEM_PROMPT exactly, so this is a
# pure mechanism change (Phase 2), not a prompt change: the model still sees
# the same instructions, but the API now enforces the response shape
# server-side (strict JSON-schema mode -- confirmed supported by the UPB AI
# Gateway / gwdg backend) instead of us hunting for a JSON object in free
# text. See docs/PROGRESS.md for why this matters and what it replaced.
#
# JSON-schema enforcement guarantees required fields and enum values, but it
# CANNOT express cross-field referential integrity (e.g. "every id used in
# relationships/related_nodes must actually exist in entities") -- that's a
# real class of error schema validation alone won't catch, so it's checked
# explicitly below via a model_validator.

class Entity(BaseModel):
    id: str
    label: str
    type: Literal["component", "actor", "external_system", "requirement"]
    evidence: str = ""


class Relationship(BaseModel):
    source_id: str
    target_id: str
    label: str
    evidence: str = ""


class Risk(BaseModel):
    id: str
    description: str
    severity: Literal["low", "medium", "high"]
    related_nodes: List[str] = []
    evidence: str = ""


class Assumption(BaseModel):
    id: str
    description: str
    related_nodes: List[str] = []
    evidence: str = ""


class ContextExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    risks: List[Risk]
    assumptions: List[Assumption]

    @model_validator(mode="after")
    def check_referential_integrity(self) -> "ContextExtraction":
        known_ids = {e.id for e in self.entities}
        missing = set()
        for rel in self.relationships:
            if rel.source_id not in known_ids:
                missing.add(rel.source_id)
            if rel.target_id not in known_ids:
                missing.add(rel.target_id)
        for item in (*self.risks, *self.assumptions):
            for node_id in item.related_nodes:
                if node_id not in known_ids:
                    missing.add(node_id)
        if missing:
            raise ValueError(
                f"relationships/related_nodes reference id(s) not present in entities: {sorted(missing)}"
            )
        return self


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


def ask_chatgpt(text: str) -> dict:
    """
    Extracts entities/relationships/risks/assumptions from `text` using
    strict JSON-schema structured output (ContextExtraction), with a
    retry-and-repair loop: on a refusal, API error, or a failed
    referential-integrity check, the model's own prior (invalid) output and
    the specific error are fed back so it can correct itself, up to
    MAX_ATTEMPTS total tries.
    """
    key = _get_key()
    client = OpenAI(api_key=key, base_url=BASE_URL)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    last_error: Exception | None = None
    last_raw_content: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=messages,
                response_format=ContextExtraction,
                temperature=0,
            )
            message = response.choices[0].message
            last_raw_content = message.content

            if message.refusal:
                raise ValueError(f"Model refused to answer: {message.refusal}")

            return message.parsed.model_dump()

        except Exception as e:
            last_error = e
            if attempt < MAX_ATTEMPTS:
                if last_raw_content:
                    messages.append({"role": "assistant", "content": last_raw_content})
                messages.append({
                    "role": "user",
                    "content": f"That output was invalid: {e}. Please correct it and resend the full, corrected JSON extraction.",
                })
            continue

    raise RuntimeError(
        f"Structured extraction failed after {MAX_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error
