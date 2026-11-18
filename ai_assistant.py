"""
Optional LLM explanation layer for ClaimGuard.

The trained Random Forest remains the sole source of the numeric score and
priority. This module sends a minimized, identifier-free claim summary to the
configured OpenAI Responses API endpoint and returns advisory
review guidance. It never changes a prediction or decides claim outcomes.
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from model_service import Assessment, _pretty

DEFAULT_MODEL = "gpt-6-luna"
SENSITIVE_KEYS = {
    "customer_id", "policy_number", "claim_number", "name", "first_name",
    "last_name", "email", "phone", "address", "ssn", "date_of_birth",
    # Avoid exporting demographic/protected attributes to the LLM provider.
    "age", "gender", "marital_status", "education_level", "occupation",
}


def _safe_claim(values: dict[str, Any]) -> dict[str, Any]:
    """Keep only scalar model features; exclude common direct identifiers."""
    result = {}
    for key, value in values.items():
        if key.lower() in SENSITIVE_KEYS or any(
            token in key.lower() for token in ("email", "phone", "address", "name")
        ):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            result[_pretty(key)] = value
    return result


class AIConfigurationError(RuntimeError):
    pass


def generate_investigation_brief(
    values: dict[str, Any], assessment: Assessment, timeout: int = 25
) -> str:
    """
    Generate a concise, non-determinative investigator brief.

    Environment:
      OPENAI_API_KEY (required)
      OPENAI_BASE_URL (optional, API base URL)
      OPENAI_MODEL (optional)
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AIConfigurationError(
            "AI assistant is not configured. Set OPENAI_API_KEY in your environment."
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    if base_url.endswith("/responses"):
        endpoint = base_url
    elif base_url.endswith("/v1"):
        endpoint = base_url + "/responses"
    else:
        endpoint = base_url + "/v1/responses"

    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    instructions = (
        "You are ClaimGuard's claim-review writing assistant. Your job is to "
        "explain the supplied model result and help a human understand which "
        "source documents may be worth checking. Write as a clear, calm, "
        "experienced colleague speaking to a non-technical reader. Use short "
        "natural sentences, not form-like labels or repetitive templates. "
        "The Random Forest score and priority supplied in the input are the "
        "only model result. Repeat the score accurately as a model screening "
        "score. It is not a probability, a confidence score, or proof of fraud. "
        "Never create, estimate, reinterpret, or rank with another score. "
        "Never accuse the claimant, infer intent, recommend denial, or make "
        "a claim decision. Do not use protected or demographic traits. "
        "Use only supplied claim fields and copy values and units exactly. "
        "Be conservative: do not turn an unusual value, missing document, or "
        "possible difference into a fraud signal. A missing police report by "
        "itself is not evidence of wrongdoing. No injury or hospital visit "
        "does not by itself contradict a serious accident or property damage. "
        "Policy duration and policy tenure are different fields; do not call "
        "them inconsistent just because their numbers differ. If policy "
        "status is inactive, say only that the policy's effective dates and "
        "coverage for the incident date need to be checked; do not assume the "
        "claim is uncovered because the incident date may not be supplied. "
        "Only describe a mismatch when the supplied fields directly refer to "
        "the same thing and clearly disagree. If uncertain, say that the "
        "information cannot be confirmed from the form alone. "
        "Mention no more than three useful checks. Omit weak or routine "
        "observations rather than filling a quota. For each check, explain "
        "in one short paragraph: what the form says, why it needs confirmation, "
        "and which actual record could confirm it. Do not invent documents "
        "or facts. If no meaningful issue is supported, say that no clear "
        "inconsistency is apparent from the entered details. "
        "Format the answer in simple Markdown with exactly these sections: "
        "'## Claim screening summary' (two short sentences: the supplied score and "
        "priority, plus a reminder that this is only a screening result); "
        "'## Details worth checking' (zero to three concise bullets written "
        "as natural sentences, not repeated sublabels); and "
        "'## Human review' (one short sentence explaining that the reviewer "
        "must check original records and policy terms before deciding). "
        "Do not repeat the score in multiple sections. Do not use phrases "
        "such as 'fraud risk percentage', 'fraud probability', 'suspicious "
        "claim', or 'fraud detected'."
    )
    user_input = json.dumps(
        {
            "model_risk_score_percent": round(assessment.score * 100, 1),
            "model_priority": assessment.title,
            "flagged_for_human_review": assessment.flagged,
            "claim_features": _safe_claim(values),
        },
        ensure_ascii=False,
        default=str,
    )
    payload = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 700,
        "store": False,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"AI provider returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"AI provider could not be reached: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError("AI provider returned an invalid response.") from exc

    # Responses API returns output message items; collect their text parts.
    text_parts = []
    for item in data.get("output", []):
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                value = part.get("text")
                if isinstance(value, str) and value.strip():
                    text_parts.append(value.strip())
    text = "\n".join(text_parts)
    if not text.strip():
        raise RuntimeError("AI provider response did not contain an answer.")
    return text.strip()
