"""
Pathway pipeline for MediTrack - Wound Healing Monitor

- Ingests Aparavi-enriched wound documents from:
    data/processed/aparavi_results/*.json

- Uses Pathway's streaming engine + LLM xPack to:
    * Maintain a live table of wound events
    * Generate short clinical-style summaries

- Writes live output to:
    data/outputs/wound_events.jsonl

This file ALSO exposes a lightweight `publish_wound_event` function
used by the Streamlit app. For this hackathon version, that function
is just a logging stub so that Pathway usage is centered around the
Aparavi -> Pathway streaming pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pathway as pw
from dotenv import load_dotenv

# LLM xPack imports (OpenAI used here; you can swap to others if needed)
from pathway.xpacks.llm.llms import OpenAIChat
from pathway.xpacks.llm.splitters import TokenCountSplitter

load_dotenv()

# --------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------

BASE_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path(
    os.getenv("PROCESSED_DATA_DIR", BASE_DATA_DIR / "processed")
)
APARAVI_RESULTS_DIR = PROCESSED_DATA_DIR / "aparavi_results"

OUTPUT_DIR = Path(os.getenv("PATHWAY_OUTPUT_DIR", BASE_DATA_DIR / "outputs"))
OUTPUT_FILE = OUTPUT_DIR / "wound_events.jsonl"

APARAVI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Schema for Aparavi-enriched JSON
# Adjust field names if your Aparavi pipeline uses different keys.
# --------------------------------------------------------------------


class WoundDocSchema(pw.Schema):
    # IDs
    patient_id: str
    doc_id: str

    # Wound metadata (tune to your JSON)
    wound_stage: str
    redness_score: float
    area_cm2: float
    infection_risk_flag: bool

    # Timestamp of capture / analysis
    timestamp: str  # we keep as string for now (ISO or similar)

    # Free-text description / OCR summary from Aparavi
    text_summary: str


# --------------------------------------------------------------------
# Public function: publish_wound_event
# (used by Streamlit app; here we keep it as a logging stub)
# --------------------------------------------------------------------


def publish_wound_event(
    patient_id: str,
    metrics: Dict[str, Any],
    risk_level: str,
) -> None:
    """
    Lightweight stub to keep Streamlit happy.

    In a more advanced version, this could:
      - Push events into Kafka / Pulsar / Redis Stream
      - Be consumed by a Pathway connector (e.g. Kafka connector)
      - Merge CV-derived metrics with Aparavi-enriched documents

    For now, we just print to stdout so judges can see calls.
    """
    print(
        "[Pathway publish_wound_event] "
        f"patient_id={patient_id}, risk_level={risk_level}, metrics={metrics}"
    )


# --------------------------------------------------------------------
# Pathway pipeline
# --------------------------------------------------------------------


def build_pipeline() -> pw.Table:
    """
    Build the Pathway streaming pipeline:

    1. Read Aparavi-enriched JSON files from APARAVI_RESULTS_DIR
    2. Perform light feature engineering
    3. Use OpenAI LLM via Pathway xPack to generate short summaries
    4. Emit a joined live table ready to be written to OUTPUT_FILE
    """

    # 1) Streaming read of Aparavi JSON
    docs = pw.io.fs.read(
        path=str(APARAVI_RESULTS_DIR),
        format="json",
        schema=WoundDocSchema,
        mode="streaming",  # new files are picked up incrementally
        object_pattern="*.json",
    )

    # 2) Basic selection (you can add more fields if your schema has them)
    events = docs.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        timestamp=pw.this.timestamp,
        text_summary=pw.this.text_summary,
    )

    # 3) Prepare text for LLM (chunk if needed)
    splitter = TokenCountSplitter(
        min_tokens=80,
        max_tokens=256,
        encoding="cl100k_base",
    )

    chunked = events.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        timestamp=pw.this.timestamp,
        chunks=splitter(pw.this.text_summary),
    ).flatten(pw.this.chunks)

    chunked = chunked.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        timestamp=pw.this.timestamp,
        chunk_text=pw.this.chunks[0],
    )

    # 4) LLM model (OpenAI; requires OPENAI_API_KEY)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print(
            "[Pathway] WARNING: OPENAI_API_KEY not set. "
            "Summaries will fall back to echoing truncated text."
        )

    if openai_api_key:
        model = OpenAIChat(
            model="gpt-4o-mini",
            api_key=openai_api_key,
        )
    else:
        model = None

    @pw.udf
    def build_prompt(chunk_text: str, wound_stage: str) -> str:
        return (
            "You are an AI clinical assistant summarizing wound-healing status for a "
            "clinician and patient. Using the following machine-extracted notes, write "
            "a concise 2–3 sentence summary that covers: healing progression, "
            "inflammation/redness, infection risk, and any suggested follow-up. "
            "Avoid giving direct medical advice; use educational language.\n\n"
            f"WOUND STAGE: {wound_stage}\n"
            f"NOTES:\n{chunk_text}"
        )

    if model is not None:
        summarized_chunks = chunked.select(
            patient_id=pw.this.patient_id,
            doc_id=pw.this.doc_id,
            wound_stage=pw.this.wound_stage,
            redness_score=pw.this.redness_score,
            area_cm2=pw.this.area_cm2,
            infection_risk_flag=pw.this.infection_risk_flag,
            timestamp=pw.this.timestamp,
            llm_output=model(build_prompt(pw.this.chunk_text, pw.this.wound_stage)),
        )
    else:
        # Fallback: no LLM; just truncate chunk_text as a pseudo-summary
        @pw.udf
        def truncate_text(text: str) -> str:
            return (text[:400] + "…") if len(text) > 400 else text

        summarized_chunks = chunked.select(
            patient_id=pw.this.patient_id,
            doc_id=pw.this.doc_id,
            wound_stage=pw.this.wound_stage,
            redness_score=pw.this.redness_score,
            area_cm2=pw.this.area_cm2,
            infection_risk_flag=pw.this.infection_risk_flag,
            timestamp=pw.this.timestamp,
            llm_output=truncate_text(pw.this.chunk_text),
        )

    # 5) Aggregate multiple chunks per document into a single summary
    summarized = summarized_chunks.groupby(
        pw.this.patient_id, pw.this.doc_id
    ).reduce(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        timestamp=pw.this.timestamp,
        summary=pw.reducers.concat(pw.this.llm_output, separator="\n"),
    )

    return summarized


def run_pipeline() -> None:
    """
    Build the pipeline and write resulting table to JSONL
    so the Streamlit app can read `data/outputs/wound_events.jsonl`.
    """
    table = build_pipeline()

    pw.io.fs.write(
        table=table,
        filename=str(OUTPUT_FILE),
        format="json",
    )

    print(
        f"[Pathway] Streaming pipeline started.\n"
        f"Reading Aparavi JSON from: {APARAVI_RESULTS_DIR}\n"
        f"Writing live wound events to: {OUTPUT_FILE}\n"
        "Leave this process running while you use the Streamlit UI."
    )

    pw.run()


if __name__ == "__main__":
    run_pipeline()
