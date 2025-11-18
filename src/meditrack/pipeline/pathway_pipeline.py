import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pathway as pw
from dotenv import load_dotenv
from pathway.xpacks.llm.llms import OpenAIChat
from pathway.xpacks.llm.splitters import TokenCountSplitter

load_dotenv()

PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
APARAVI_RESULTS_DIR = PROCESSED_DATA_DIR / "aparavi_results"
APARAVI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PATHWAY_OUTPUT_DIR = Path(os.getenv("PATHWAY_OUTPUT_DIR", "data/outputs"))
PATHWAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Schema definitions --------- #

class WoundDocSchema(pw.Schema):
    # Shape should match whatever your Aparavi pipeline emits.
    # Adjust these field names to your JSON keys.
    patient_id: str
    doc_id: str
    wound_stage: str
    redness_score: float
    area_cm2: float
    infection_risk_flag: bool
    timestamp: str  # we'll convert to datetime later
    text_summary: str  # free-text description from Aparavi / OCR


# --------- Pathway pipeline --------- #

def build_pipeline() -> None:
    # 1) Read Aparavi-enriched wound events (JSONL files) in streaming mode.
    docs = pw.io.fs.read(
        path=str(APARAVI_RESULTS_DIR),
        format="json",
        schema=WoundDocSchema,
        mode="streaming",  # keeps updating as new files arrive :contentReference[oaicite:9]{index=9}
        object_pattern="*.json",
    )

    # 2) Basic type fixes / feature engineering.
    events = docs.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        # convert ISO string to timestamp if needed
        ts=pw.this.timestamp,
        text_summary=pw.this.text_summary,
    )

    # 3) Optional: chunk long text before sending to LLM.
    splitter = TokenCountSplitter(min_tokens=80, max_tokens=256, encoding="cl100k_base")
    chunked = events.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        chunk=splitter(pw.this.text_summary),
    ).flatten(pw.this.chunk)
    chunked = chunked.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        chunk_text=pw.this.chunk[0],  # text
    )

    # 4) LLM wrapper using OpenAI (or Gemini via LiteLLM, etc.). :contentReference[oaicite:10]{index=10}
    model = OpenAIChat(
        model="gpt-4o-mini",  # or another allowed model
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    @pw.udf
    def make_prompt(text: str) -> str:
        return (
            "You are a clinical assistant summarizing wound-healing progress for doctors "
            "and patients. Given the following machine-extracted notes, write a concise "
            "2-3 sentence summary describing healing status, risk level, and next steps.\n\n"
            f"NOTES:\n{text}"
        )

    enriched = chunked.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        chunk_text=pw.this.chunk_text,
        prompt=make_prompt(pw.this.chunk_text),
    )

    # Apply LLM to each chunk
    summarized_chunks = enriched.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        llm_output=model(pw.this.prompt),
    )

    # 5) Aggregate multiple chunks per doc into a single summary.
    #    For simplicity, just concatenate.
    grouped = summarized_chunks.groupby(
        pw.this.patient_id, pw.this.doc_id
    ).reduce(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        combined_summary=pw.reducers.concat(pw.this.llm_output, separator="\n"),
    )

    # 6) Join back with numeric metrics from original events.
    joined = grouped.join(
        events,
        left_on=pw.this.doc_id,
        right_on=events.doc_id,
        how="left",
    ).select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=events.wound_stage,
        redness_score=events.redness_score,
        area_cm2=events.area_cm2,
        infection_risk_flag=events.infection_risk_flag,
        timestamp=events.ts,
        summary=pw.this.combined_summary,
    )

    # 7) Write to file system where Streamlit can read it.
    pw.io.fs.write(
        table=joined,
        filename=str(PATHWAY_OUTPUT_DIR / "wound_events.jsonl"),
        format="json",
    )

    # When run as a script, just call pw.run() below.


if __name__ == "__main__":
    build_pipeline()
    pw.run()
