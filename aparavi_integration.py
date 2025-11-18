import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from aparavi_dtc_sdk import AparaviClient  # from official SDK
# https://aparavi.com/documentation-aparavi/data-toolchain-for-ai-documentation/python-sdk/python-sdk-quickstart/ :contentReference[oaicite:3]{index=3}

load_dotenv()

APARAVI_BASE_URL = os.getenv("APARAVI_BASE_URL")
APARAVI_API_KEY = os.getenv("APARAVI_API_KEY")

# Directory where Aparavi-safe files will be written (e.g., redacted JSON, text, etc.)
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_client() -> Optional[AparaviClient]:
    """Return an AparaviClient if creds are set, else None (for offline/demo mode)."""
    if not APARAVI_BASE_URL or not APARAVI_API_KEY:
        return None

    return AparaviClient(
        base_url=APARAVI_BASE_URL,
        api_key=APARAVI_API_KEY,
    )


def run_aparavi_pipeline_on_wound_files(
    file_glob: str,
    pipeline_config: str = "aparavi_pipeline_config.json",
    output_subdir: str = "aparavi_results",
) -> Path:
    """
    Send wound-related files to Aparavi for PII/PHI detection, redaction, and enrichment.

    - file_glob: glob pattern for images/docs you want to send (e.g., 'data/sample_wounds/*.jpg')
    - pipeline_config: either a JSON pipeline config file or a Predefined pipeline name.
    - Returns: directory where Aparavi wrote results (you'll read them from Pathway).
    """
    client = _get_client()
    out_dir = PROCESSED_DATA_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    if client is None:
        # Fallback for when you don't have Aparavi credentials yet.
        # We'll just mirror the input files into out_dir.
        from glob import glob
        import shutil

        for f in glob(file_glob):
            src = Path(f)
            dst = out_dir / src.name
            shutil.copy2(src, dst)

        print(
            "[Aparavi] No credentials set – running in NOOP mode. "
            f"Copied raw files into {out_dir}"
        )
        return out_dir

    # For hackathon: use execute_pipeline_workflow with your pipeline config.
    # Docs: execute_pipeline_workflow(pipeline='pipeline.json', file_glob='./*.png') :contentReference[oaicite:4]{index=4}
    result = client.execute_pipeline_workflow(
        pipeline=pipeline_config,  # can also be a PredefinedPipeline enum
        file_glob=file_glob,
    )

    # The exact shape of `result` depends on your pipeline. Typical patterns:
    # - A URL to a result object
    # - A local path where files were written (if you mounted volumes)
    # For safety, we just log it and rely on your pipeline to write into `out_dir`.
    print("[Aparavi] Pipeline result:", result)
    print("[Aparavi] Ensure your Aparavi pipeline writes outputs into", out_dir)

    return out_dir


def get_aparavi_enriched_docs(
    aparavi_output_dir: Path,
    pattern: str = "*.json",
) -> List[Path]:
    """
    Helper to list Aparavi-enriched artifacts (e.g. JSON with redacted text & metadata)
    that Pathway will ingest as part of the live index.
    """
    return sorted(aparavi_output_dir.glob(pattern))
