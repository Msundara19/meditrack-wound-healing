"""
Pathway pipeline integration for MediTrack.

For the hackathon demo:
- `publish_wound_event` is called by the Streamlit app whenever analysis runs.
- If the `pathway` package is installed and configured, you can extend this
  to push events into a real streaming pipeline.
- If not, it simply logs events locally (safe no-op for now).
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv


def _load_env_if_needed() -> None:
    secret_path = "/etc/secrets/imp.env"
    if os.path.exists(secret_path):
        load_dotenv(secret_path)
    else:
        if os.path.exists("imp.env"):
            load_dotenv("imp.env")
        else:
            load_dotenv()


def _get_pathway_license_key() -> str | None:
    _load_env_if_needed()
    return os.getenv("PATHWAY_LICENSE_KEY")


def publish_wound_event(
    patient_id: str,
    metrics: Dict[str, Any],
    risk_level: str,
) -> None:
    """
    Entry point called from Streamlit whenever an analysis is run.

    For now:
    - Checks for PATHWAY_LICENSE_KEY
    - If `pathway` is installed, this is where you'd push into a real pipeline
    - Otherwise, logs the event to a local JSONL file.

    This keeps the integration safe & demo-friendly, while showing how
    Pathway would conceptually fit into the architecture.
    """
    license_key = _get_pathway_license_key()
    if not license_key:
        # No license configured, just log a warning and continue
        print("[Pathway] PATHWAY_LICENSE_KEY not set; logging event locally.")
    else:
        print("[Pathway] License key detected (not validated here).")

    try:
        import pathway  # noqa: F401
        pathway_installed = True
    except ImportError:
        pathway_installed = False

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "patient_id": patient_id,
        "risk_level": risk_level,
        "metrics": metrics,
    }

    # In a real integration, you'd define a Pathway schema and stream here.
    if pathway_installed:
        # Placeholder: this is where a real Pathway sink would go.
        print("[Pathway] (Placeholder) Would push event into Pathway stream:")
        print(json.dumps(event)[:300] + "...")
    else:
        # Fallback: append to a local JSONL file for debugging/demo purposes.
        os.makedirs("pathway_logs", exist_ok=True)
        log_path = os.path.join("pathway_logs", "wound_events.jsonl")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        print(f"[Pathway] Logged wound event to {log_path}")
