import os
from typing import Dict, Any, Tuple

from openai import OpenAI
import google.generativeai as genai

# --- OpenAI setup -----------------------------------------------------------

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


def generate_ai_summary_openai(
    patient_id: str,
    latest_metrics: Dict[str, Any],
    trend_notes: str,
) -> Tuple[str, str]:
    """
    Returns (summary_md, risk_level) using OpenAI.
    latest_metrics: e.g. {"wound_area_cm2": 4.3, "redness_score": 0.7, ...}
    trend_notes: plain-text description of timeline / trend.
    """
    client = _get_openai_client()

    system_prompt = (
        "You are an assistant for educational wound-healing monitoring. "
        "You explain wound status in simple, non-alarming language and NEVER "
        "give definitive medical diagnoses. You always remind users to consult "
        "a healthcare professional for any concerns."
    )

    user_prompt = f"""
Patient ID: {patient_id}

Latest computed wound metrics (JSON-like):
{latest_metrics}

Trend notes:
{trend_notes}

Tasks:
1. Briefly summarize how the wound seems to be healing.
2. List 3–5 bullet points in plain language.
3. Provide a qualitative risk level: one of LOW, MEDIUM, or HIGH concern.
4. Add a final line reminding them to contact a clinician for any worries.

Respond in Markdown. At the very end, add a line starting with:
RISK_LEVEL:
and then the risk level in ALL CAPS. Example: RISK_LEVEL: MEDIUM
"""

    response = client.chat.completions.create(
        model="gpt-5.1-mini",  # or any other chat model you like
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    # Extract risk level from the last line
    risk_level = "UNKNOWN"
    for line in reversed(content.splitlines()):
        line = line.strip()
        if line.upper().startswith("RISK_LEVEL:"):
            risk_level = line.split(":", 1)[1].strip().upper()
            # remove that line from the markdown
            content = "\n".join(content.splitlines()[:-1]).strip()
            break

    return content, risk_level


# --- Gemini setup -----------------------------------------------------------

def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    genai.configure(api_key=api_key)


def generate_ai_summary_gemini(
    patient_id: str,
    latest_metrics: Dict[str, Any],
    trend_notes: str,
) -> Tuple[str, str]:
    """
    Same interface as the OpenAI version, but using Gemini.
    """
    _configure_gemini()

    system_prompt = (
        "You are an assistant for educational wound-healing monitoring. "
        "Explain wound status simply, avoid strong diagnostic language, and "
        "always advise consulting clinicians for decisions."
    )

    prompt = f"""{system_prompt}

Patient ID: {patient_id}

Latest computed wound metrics:
{latest_metrics}

Trend notes:
{trend_notes}

Tasks:
1. Briefly summarize healing status.
2. Give 3–5 clear bullet points.
3. Add a final line with 'RISK_LEVEL: LOW/MEDIUM/HIGH'.
"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    content = resp.text.strip()

    risk_level = "UNKNOWN"
    for line in reversed(content.splitlines()):
        line = line.strip()
        if line.upper().startswith("RISK_LEVEL:"):
            risk_level = line.split(":", 1)[1].strip().upper()
            content = "\n".join(content.splitlines()[:-1]).strip()
            break

    return content, risk_level
