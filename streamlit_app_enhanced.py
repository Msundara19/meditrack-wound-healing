"""
MediTrack - Enhanced Streamlit Dashboard
Real-time wound monitoring with Pathway streaming and Aparavi PHI protection
"""

from load_env import load_environment
load_environment()

from meditrack.llm.ai_client import (
    generate_ai_summary_groq,
    generate_ai_summary_gemini,
)

# Try to import Pathway publisher (safe even if not installed)
try:
    from meditrack.pipeline.pathway_pipeline import publish_wound_event
except Exception:
    publish_wound_event = None

import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from PIL import Image

# Page config
st.set_page_config(
    page_title="MediTrack - Wound Healing Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .sidebar-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }
    .badge-pathway {
        background-color: #4caf50;
        color: white;
    }
    .badge-aparavi {
        background-color: #2196f3;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = "DEMO-001"
    if "wound_history" not in st.session_state:
        st.session_state.wound_history = []
    if "aparavi_enabled" not in st.session_state:
        st.session_state.aparavi_enabled = True
    if "pathway_streaming" not in st.session_state:
        st.session_state.pathway_streaming = False
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "Groq"      # DEFAULT
    if "ai_live" not in st.session_state:
        st.session_state.ai_live = True


def render_header():
    st.markdown(
        '<div class="main-header">🏥 MediTrack - Wound Healing Monitor</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<span class="sidebar-badge badge-pathway">⚡ Pathway Streaming</span>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<span class="sidebar-badge badge-aparavi">🔒 Aparavi PHI Protection</span>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<span class="sidebar-badge" style="background-color: #9c27b0; color: white;">🤖 AI Analysis</span>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<span class="sidebar-badge" style="background-color: #ff5722; color: white;">📊 Real-time Tracking</span>',
            unsafe_allow_html=True,
        )
    st.divider()


def render_sidebar():
    with st.sidebar:
        st.image(
            "https://via.placeholder.com/200x80/1f77b4/ffffff?text=MediTrack",
            use_container_width=True,
        )

        st.header("⚙️ Settings")

        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            help="Enter patient identifier (demo mode)",
        )
        st.session_state.patient_id = patient_id

        st.divider()
        st.subheader("🔧 Features")

        aparavi = st.checkbox(
            "Aparavi PHI Protection",
            value=st.session_state.aparavi_enabled,
            help="Automatically detect and redact PHI/PII from LLM prompts",
        )
        st.session_state.aparavi_enabled = aparavi

        pathway = st.checkbox(
            "Pathway Streaming",
            value=st.session_state.pathway_streaming,
            help="Enable real-time streaming event publishing",
        )
        st.session_state.pathway_streaming = pathway

        st.divider()
        st.subheader("🤖 AI Engine")

        st.session_state.ai_provider = st.radio(
            "Provider",
            ["Groq", "Gemini"],
            index=0 if st.session_state.ai_provider == "Groq" else 1,
        )
        st.session_state.ai_live = st.checkbox(
            "Use live LLM (needs API key)",
            value=st.session_state.ai_live,
        )

        st.divider()
        st.subheader("📡 System Status")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV Pipeline", "🟢 Active")
        with col2:
            st.metric("LLM Engine", "🟢 Ready")

        if pathway:
            if publish_wound_event is not None:
                st.success("⚡ Streaming ACTIVE (events will be published)")
            else:
                st.warning("⚠️ Pathway not fully configured (no publisher)")
        else:
            st.info("⏸️ Batch Mode")

        if aparavi:
            st.success("🔒 PHI Protection ENABLED (prompts filtered)")
        else:
            st.warning("⚠️ PHI Protection OFF (raw prompts to LLM)")

        st.divider()
        st.subheader("🏆 Hack With Chicago 2.0")
        st.markdown(
            """
        **Track:** Open Innovation (Healthcare AI)  
        **Date:** November 17, 2025  
        **Team:** IIT - AI Vision Lab
        """
        )

        st.divider()
        with st.expander("⚠️ Medical Disclaimer"):
            st.warning(
                """
            **Educational Prototype Only**
            
            This is NOT a medical device:
            - ❌ Not FDA approved
            - ❌ Not for diagnosis
            - ✅ Always consult healthcare professionals
            """
            )


def upload_and_process_image():
    st.header("📸 Upload Wound Image")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of the wound",
        )

    with col2:
        st.info(
            """
        **Image Guidelines:**
        - Good lighting
        - Clear wound view
        - Include reference object
        - Avoid glare
        """
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)

        if st.session_state.aparavi_enabled:
            with col2:
                st.subheader("🔒 PHI Detection (simulated)")
                with st.spinner("Scanning for PHI..."):
                    phi_detections = simulate_phi_detection(img_array)
                    if phi_detections > 0:
                        st.error(
                            f"⚠️ {phi_detections} PHI element(s) detected and redacted"
                        )
                        redacted_img = apply_redaction(img_array)
                        st.image(redacted_img, use_container_width=True)
                    else:
                        st.success("✅ No PHI detected (simulated)")
                        st.image(image, use_container_width=True, caption="Clean Image")
        else:
            with col2:
                st.subheader("🔒 PHI Detection")
                st.info("PHI protection disabled in settings.")

        with col3:
            st.subheader("🔬 Wound Segmentation (simulated)")
            with st.spinner("Processing..."):
                segmented = simulate_segmentation(img_array)
                st.image(segmented, use_container_width=True)

        st.divider()

        st.subheader("🤖 AI Analysis Settings")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.session_state.ai_provider = st.radio(
                "LLM Provider",
                ["Groq", "Gemini"],
                index=0 if st.session_state.ai_provider == "Groq" else 1,
                horizontal=True,
                key="ai_provider_main",
            )
        with c2:
            st.session_state.ai_live = st.checkbox(
                "Use live LLM",
                value=st.session_state.ai_live,
                key="ai_live_main",
            )

        if st.button("🚀 Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                process_wound_analysis(img_array)


def process_wound_analysis(image: np.ndarray):
    metrics = extract_wound_metrics(image)

    st.header("📊 Wound Analysis Results")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Wound Area",
            f"{metrics['area']:.2f} cm²",
            delta=f"{metrics['area_change']:.1f}%",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "Redness Index",
            f"{metrics['redness']:.0f}%",
            delta=f"{metrics['redness_change']:+.0f}%",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Granulation",
            f"{metrics['granulation']:.0f}%",
            delta=f"{metrics['granulation_change']:+.0f}%",
        )
    with col4:
        st.metric(
            "Edge Quality",
            f"{metrics['edge_quality']:.2f}",
            delta="Good" if metrics["edge_quality"] > 0.7 else "Monitor",
        )

    st.divider()

    provider = st.session_state.get("ai_provider", "Groq")
    use_live = st.session_state.get("ai_live", True)
    analysis = generate_llm_analysis(metrics, provider=provider, use_live=use_live)
    display_ai_analysis(analysis)
    save_to_history(metrics, analysis)

    # ---- Pathway streaming hook ----
    if st.session_state.get("pathway_streaming", False):
        if publish_wound_event is not None:
            try:
                publish_wound_event(
                    patient_id=st.session_state.patient_id,
                    metrics=metrics,
                    risk_level=analysis["risk_level"],
                )
                st.info("📡 Wound event published to Pathway stream (simulated).")
            except Exception as e:
                st.warning(f"Pathway streaming failed: {e}")
        else:
            st.warning("Pathway streaming enabled, but no publisher is configured.")


def generate_llm_analysis(
    metrics: dict,
    provider: str = "Groq",
    use_live: bool = True
) -> dict:
    """
    Combine simple heuristic risk logic with Groq/Gemini LLM explanation.
    Explicitly handles the case where no clear wound is visible.
    """

    area_change = metrics["area_change"]
    redness = metrics["redness"]
    granulation = metrics["granulation"]
    area = metrics["area"]
    area_fraction = metrics.get("area_fraction", area / 12.0)

    # ---- Heuristic risk buckets ----
    # Case 1: No obvious wound (very small or whole-image area AND low redness)
    if ((area_fraction < 0.12) or (area_fraction > 0.8)) and redness < 35:
        risk_level = "low"
        area_trend = "no clear wound area detected"
    # Case 2: clearly improving, good granulation, low redness
    elif area_change < -10 and redness < 45 and granulation > 60:
        risk_level = "low"
        area_trend = "improving rapidly"
    # Case 3: worsening or very inflamed
    elif area_change > 15 or (redness > 70 and area_fraction > 0.15):
        risk_level = "high"
        area_trend = "worsening"
    else:
        risk_level = "medium"
        area_trend = (
            "improving"
            if area_change < 0
            else "stable"
            if abs(area_change) <= 5
            else "worsening slightly"
        )

    base_summary = (
        f"The wound shows {area_trend} progress with a "
        f"{abs(area_change):.1f}% change in area. "
        f"Granulation tissue is at {granulation:.0f}%, "
        f"and redness is {redness:.0f}%. {metrics['redness_context']}"
    )

    recommendations = [
        "Continue normal skin care and hygiene." if risk_level == "low"
        else "Monitor closely for infection signs and follow local wound care advice.",
        "Keep the area clean and dry.",
        "Take progress photos if you notice any new changes.",
        "Seek prompt clinical review if pain, discharge, or fever occur.",
    ]

    consult_doctor = risk_level != "low"
    trend = area_trend
    summary_text = base_summary

    # ---- Live LLM call (Groq or Gemini) ----
    if use_live:
        try:
            patient_id = st.session_state.get("patient_id", "DEMO-001")
            latest_metrics = {
                "area_cm2": metrics["area"],
                "area_change_pct": metrics["area_change"],
                "redness_pct": metrics["redness"],
                "granulation_pct": metrics["granulation"],
                "edge_quality": metrics["edge_quality"],
                "healing_score": metrics["healing_score"],
                "area_fraction": metrics.get("area_fraction"),
            }
            trend_notes = (
                f"Wound area changed by {metrics['area_change']:.1f}% compared to the "
                f"previous measurement. Redness is {metrics['redness']:.1f}% and "
                f"granulation is {metrics['granulation']:.1f}%."
            )

            use_aparavi_phi = st.session_state.get("aparavi_enabled", False)

            if provider == "Groq":
                summary_md, llm_risk = generate_ai_summary_groq(
                    patient_id=patient_id,
                    latest_metrics=latest_metrics,
                    trend_notes=trend_notes,
                    use_aparavi=use_aparavi_phi,
                )
            else:
                summary_md, llm_risk = generate_ai_summary_gemini(
                    patient_id=patient_id,
                    latest_metrics=latest_metrics,
                    trend_notes=trend_notes,
                    use_aparavi=use_aparavi_phi,
                )

            if summary_md:
                summary_text = summary_md
            if llm_risk and llm_risk != "UNKNOWN":
                risk_level = llm_risk.lower()

        except Exception as e:
            st.warning(
                f"Live LLM analysis failed, using offline summary instead. Details: {e}"
            )

    return {
        "summary": summary_text,
        "risk_level": risk_level,
        "recommendations": recommendations,
        "consult_doctor": consult_doctor,
        "trend": trend,
    }


def display_ai_analysis(analysis: dict):
    st.header("🤖 AI Insights")

    risk_colors = {"low": "alert-low", "medium": "alert-medium", "high": "alert-high"}
    risk_icons = {"low": "✅", "medium": "⚠️", "high": "🚨"}

    risk_key = analysis["risk_level"]
    if risk_key not in risk_colors:
        risk_key = "medium"

    st.markdown(
        f"""
    <div class="alert-box {risk_colors[risk_key]}">
        <h3>{risk_icons[risk_key]} Risk Level: {risk_key.upper()}</h3>
        <p><strong>Summary:</strong> {analysis['summary']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Detailed AI Summary")
    st.markdown(analysis["summary"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Recommendations")
        for i, rec in enumerate(analysis["recommendations"], 1):
            st.write(f"{i}. {rec}")
    with col2:
        st.subheader("🔔 Action Items")
        if analysis["consult_doctor"]:
            st.error("⚠️ Contact your healthcare provider")
        else:
            st.success("✅ Continue current care")
        st.info(f"📈 Trend: {analysis['trend'].title()}")


def render_historical_trends():
    st.header("📈 Healing Progress Timeline")
    if len(st.session_state.wound_history) < 2:
        st.info("Upload at least 2 images to see trend analysis")
        return

    df = pd.DataFrame(st.session_state.wound_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["area"],
            mode="lines+markers",
            name="Wound Area",
            line=dict(color="#f44336", width=3),
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title="Wound Area Over Time",
        xaxis_title="Date",
        yaxis_title="Area (cm²)",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_red = go.Figure()
        fig_red.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["redness"],
                mode="lines+markers",
                name="Redness",
                line=dict(color="#ff5722", width=2),
            )
        )
        fig_red.update_layout(
            title="Redness Index",
            yaxis_title="Redness (%)",
            height=300,
        )
        st.plotly_chart(fig_red, use_container_width=True)
    with col2:
        fig_gran = go.Figure()
        fig_gran.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["granulation"],
                mode="lines+markers",
                name="Granulation",
                line=dict(color="#4caf50", width=2),
            )
        )
        fig_gran.update_layout(
            title="Granulation Tissue",
            yaxis_title="Granulation (%)",
            height=300,
        )
        st.plotly_chart(fig_gran, use_container_width=True)


def render_metrics_dashboard():
    st.header("📊 Detailed Metrics")
    if not st.session_state.wound_history:
        st.info("No data available yet. Upload an image to get started.")
        return

    latest = st.session_state.wound_history[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Morphological")
        st.write(f"**Area:** {latest['area']:.2f} cm²")
        st.write(f"**Perimeter:** {latest['perimeter']:.2f} cm")
        st.write(f"**Aspect Ratio:** {latest['aspect_ratio']:.2f}")
    with col2:
        st.subheader("Tissue Analysis")
        st.write(f"**Granulation:** {latest['granulation']:.0f}%")
        st.write(f"**Epithelialization:** {latest['epithelialization']:.0f}%")
        st.write(f"**Necrotic:** {latest['necrotic']:.0f}%")
    with col3:
        st.subheader("Healing Indicators")
        st.write(f"**Redness Index:** {latest['redness']:.0f}%")
        st.write(f"**Edge Quality:** {latest['edge_quality']:.2f}")
        st.write(f"**Overall Score:** {latest['healing_score']:.0f}/100")


# ---- Helper functions (simulated CV & PHI) ----

def simulate_phi_detection(image: np.ndarray) -> int:
    return int(np.random.choice([0, 0, 0, 1, 2]))


def apply_redaction(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    img[0:int(h * 0.1), :] = cv2.GaussianBlur(
        img[0:int(h * 0.1), :], (51, 51), 30
    )
    return img


def simulate_segmentation(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    overlay = image.copy()
    overlay[mask == 0] = overlay[mask == 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    return overlay.astype(np.uint8)


def extract_wound_metrics(image: np.ndarray) -> dict:
    """
    Extract wound metrics from the image.

    This is a lightweight, demo-friendly version that:
    - Estimates wound area from a simple threshold mask
    - Estimates redness from red-vs-green in the wound region
    - Varies granulation etc. based on texture / brightness

    It is NOT medically accurate, but it makes the AI output
    react meaningfully to different images.
    """
    # Ensure RGB uint8
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]

    # --- Rough wound mask: darker region assumed to be wound  ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    wound_pixels = mask > 0
    wound_area_px = int(wound_pixels.sum())

    # If mask failed, fall back to "whole image"
    if wound_area_px == 0:
        wound_pixels = np.ones_like(gray, dtype=bool)
        wound_area_px = wound_pixels.sum()

    # Area as % of image, mapped to ~0–12 cm² range
    area_fraction = wound_area_px / float(h * w)          # 0–1
    base_area = area_fraction * 12.0                      # arbitrary scaling for demo

    # --- Area change compared to previous measurement ---
    if st.session_state.wound_history:
        prev_area = st.session_state.wound_history[-1]["area"]
        area_change = ((base_area - prev_area) / prev_area) * 100
    else:
        area_change = 0.0

    # --- Redness: red channel vs green inside wound mask ---
    img_float = image.astype(np.float32)
    red = img_float[:, :, 0]
    green = img_float[:, :, 1]

    red_wound = red[wound_pixels]
    green_wound = green[wound_pixels]

    red_diff = np.clip(red_wound.mean() - green_wound.mean(), -80, 80)
    redness = np.interp(red_diff, [-80, 80], [20, 90])  # 20–90%

    # --- Granulation: use brightness + a bit of noise ---
    brightness = gray[wound_pixels].mean()
    granulation = np.interp(brightness, [40, 200], [40, 85])  # %
    granulation += np.random.uniform(-5, 5)
    granulation = float(np.clip(granulation, 0, 100))

    # --- Edge quality: how "sharp" boundary is (roughly) ---
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges[wound_pixels].mean() / 255.0
    edge_quality = float(np.clip(0.4 + edge_density, 0.0, 1.0))

    # --- Healing score: simple composite metric ---
    area_score = np.clip(100 - area_fraction * 200, 0, 100)
    redness_score = np.clip(100 - redness, 0, 100)
    gran_score = granulation
    healing_score = float(
        0.4 * gran_score + 0.3 * area_score + 0.3 * redness_score
    )

    return {
        "area": float(base_area),
        "area_change": float(area_change),
        "area_fraction": float(area_fraction),
        "perimeter": float(base_area * 2.5),
        "aspect_ratio": 1.2,
        "redness": float(redness),
        "redness_change": -3 if area_change < 0 else 2,
        "redness_context": (
            "Redness is within normal healing range."
            if redness < 50
            else "Elevated redness detected - monitor for infection."
        ),
        "granulation": granulation,
        "granulation_change": 5,
        "epithelialization": 20,
        "necrotic": 5,
        "edge_quality": edge_quality,
        "healing_score": healing_score,
    }


def save_to_history(metrics: dict, analysis: dict):
    entry = {
        **metrics,
        "timestamp": datetime.now().isoformat(),
        "patient_id": st.session_state.patient_id,
        "risk_level": analysis["risk_level"],
    }
    st.session_state.wound_history.append(entry)
    if len(st.session_state.wound_history) > 30:
        st.session_state.wound_history = st.session_state.wound_history[-30:]


def main():
    initialize_session_state()
    render_sidebar()
    render_header()

    tab1, tab2, tab3 = st.tabs(
        ["📸 New Analysis", "📈 Progress Tracking", "📊 Metrics"]
    )
    with tab1:
        upload_and_process_image()
    with tab2:
        render_historical_trends()
    with tab3:
        render_metrics_dashboard()

    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ for Hack With Chicago 2.0 | 
        <a href='https://github.com/Msundara19/meditrack-wound-healing'>GitHub</a> | 
        IIT - AI Vision Lab</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
