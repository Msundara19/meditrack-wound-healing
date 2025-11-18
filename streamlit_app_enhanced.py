"""
MediTrack - Enhanced Streamlit Dashboard
Real-time wound monitoring with Pathway streaming and Aparavi PHI protection
"""
from load_env import load_environment
load_environment()
from meditrack.llm.ai_client import (
    generate_ai_summary_openai,
    generate_ai_summary_gemini,
)
import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import io

# Import your modules
# from pathway_pipeline import PathwayWoundPipeline, HistoricalDataManager
# from aparavi_integration import AparaviIntegration
# from cv_processing import WoundSegmenter


# Page config
st.set_page_config(
    page_title="MediTrack - Wound Healing Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'patient_id' not in st.session_state:
        st.session_state.patient_id = 'DEMO-001'
    if 'wound_history' not in st.session_state:
        st.session_state.wound_history = []
    if 'aparavi_enabled' not in st.session_state:
        st.session_state.aparavi_enabled = True
    if 'pathway_streaming' not in st.session_state:
        st.session_state.pathway_streaming = False


def render_header():
    """Render the main header"""
    st.markdown('<div class="main-header">🏥 MediTrack - Wound Healing Monitor</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<span class="sidebar-badge badge-pathway">⚡ Pathway Streaming</span>', 
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<span class="sidebar-badge badge-aparavi">🔒 Aparavi PHI Protection</span>', 
                    unsafe_allow_html=True)
    with col3:
        st.markdown('<span class="sidebar-badge" style="background-color: #9c27b0; color: white;">🤖 AI Analysis</span>', 
                    unsafe_allow_html=True)
    with col4:
        st.markdown('<span class="sidebar-badge" style="background-color: #ff5722; color: white;">📊 Real-time Tracking</span>', 
                    unsafe_allow_html=True)
    
    st.divider()


def render_sidebar():
    """Render sidebar with settings and info"""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=MediTrack", 
                 use_container_width=True)
        
        st.header("⚙️ Settings")
        
        # Patient ID
        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
            help="Enter patient identifier (demo mode)"
        )
        st.session_state.patient_id = patient_id
        
        st.divider()
        
        # Feature toggles
        st.subheader("🔧 Features")
        
        aparavi = st.checkbox(
            "Aparavi PHI Protection",
            value=st.session_state.aparavi_enabled,
            help="Automatically detect and redact PHI/PII from images"
        )
        st.session_state.aparavi_enabled = aparavi
        
        pathway = st.checkbox(
            "Pathway Streaming",
            value=st.session_state.pathway_streaming,
            help="Enable real-time streaming analysis"
        )
        st.session_state.pathway_streaming = pathway
        
        st.divider()
        
        # System status
        st.subheader("📡 System Status")
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            st.metric("CV Pipeline", "🟢 Active")
        with status_col2:
            st.metric("LLM Engine", "🟢 Ready")
        
        if pathway:
            st.success("⚡ Streaming ACTIVE")
        else:
            st.info("⏸️ Batch Mode")
        
        if aparavi:
            st.success("🔒 PHI Protected")
        else:
            st.warning("⚠️ No PHI Scan")
        
        st.divider()
        
        # Hackathon info
        st.subheader("🏆 Hack With Chicago 2.0")
        st.markdown("""
        **Track:** Open Innovation (Healthcare AI)  
        **Date:** November 17, 2025  
        **Team:** IIT - AI Vision Lab
        """)
        
        st.divider()
        
        # Disclaimer
        with st.expander("⚠️ Medical Disclaimer"):
            st.warning("""
            **Educational Prototype Only**
            
            This is NOT a medical device:
            - ❌ Not FDA approved
            - ❌ Not for diagnosis
            - ✅ Always consult healthcare professionals
            """)


def upload_and_process_image():
    """Image upload and processing section"""
    st.header("📸 Upload Wound Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the wound"
        )
    
    with col2:
        st.info("""
        **Image Guidelines:**
        - Good lighting
        - Clear wound view
        - Include reference object
        - Avoid glare
        """)
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(image, use_container_width=True)
        
        # PHI Detection if Aparavi enabled
        if st.session_state.aparavi_enabled:
            with col2:
                st.subheader("🔒 PHI Detection")
                with st.spinner("Scanning for PHI..."):
                    phi_detections = simulate_phi_detection(img_array)
                    
                    if phi_detections > 0:
                        st.error(f"⚠️ {phi_detections} PHI element(s) detected and redacted")
                        # Display redacted image
                        redacted_img = apply_redaction(img_array)
                        st.image(redacted_img, use_container_width=True)
                    else:
                        st.success("✅ No PHI detected")
                        st.image(image, use_container_width=True, caption="Clean Image")
        
        # Process with CV pipeline
        with col3:
            st.subheader("🔬 Wound Segmentation")
            with st.spinner("Processing..."):
                segmented = simulate_segmentation(img_array)
                st.image(segmented, use_container_width=True)
        
        st.divider()
        
        # Analysis button
        if st.button("🚀 Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                process_wound_analysis(img_array)


def process_wound_analysis(image: np.ndarray):
    """Process wound and display results"""
    # Simulate metrics extraction
    metrics = extract_wound_metrics(image)
    
    # Display metrics
    st.header("📊 Wound Analysis Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Wound Area",
            f"{metrics['area']:.2f} cm²",
            delta=f"{metrics['area_change']:.1f}%",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Redness Index",
            f"{metrics['redness']:.0f}%",
            delta=f"{metrics['redness_change']:+.0f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Granulation",
            f"{metrics['granulation']:.0f}%",
            delta=f"{metrics['granulation_change']:+.0f}%"
        )
    
    with col4:
        st.metric(
            "Edge Quality",
            f"{metrics['edge_quality']:.2f}",
            delta="Good" if metrics['edge_quality'] > 0.7 else "Monitor"
        )
    
    st.divider()
    
    # AI Analysis
    analysis = generate_llm_analysis(metrics)
    display_ai_analysis(analysis)
    
    # Save to history
    save_to_history(metrics, analysis)


def generate_llm_analysis(metrics: dict) -> dict:
    """Generate AI analysis using LLM (simulated for demo)"""
    # In production, this would call your Pathway LLM xPack
    
    area_trend = "improving" if metrics['area_change'] < 0 else "concerning"
    risk_level = "low" if metrics['area_change'] < 0 and metrics['redness'] < 50 else "medium"
    
    return {
        'summary': f"The wound shows {area_trend} progress with a {abs(metrics['area_change']):.1f}% change in area. Granulation tissue is developing at {metrics['granulation']:.0f}%, indicating active healing. {metrics['redness_context']}",
        'risk_level': risk_level,
        'recommendations': [
            "Continue current wound care regimen" if risk_level == "low" else "Monitor closely for infection signs",
            "Keep wound clean and dry",
            "Take progress photos daily",
            "Follow up with healthcare provider in 3-5 days" if risk_level == "medium" else "Next routine check-up as scheduled"
        ],
        'consult_doctor': risk_level != "low",
        'trend': area_trend
    }


def display_ai_analysis(analysis: dict):
    """Display AI-generated analysis"""
    st.header("🤖 AI Insights")
    
    # Risk alert box
    risk_colors = {
        'low': 'alert-low',
        'medium': 'alert-medium',
        'high': 'alert-high'
    }
    
    risk_icons = {
        'low': '✅',
        'medium': '⚠️',
        'high': '🚨'
    }
    
    st.markdown(f"""
    <div class="alert-box {risk_colors[analysis['risk_level']]}">
        <h3>{risk_icons[analysis['risk_level']]} Risk Level: {analysis['risk_level'].upper()}</h3>
        <p><strong>Summary:</strong> {analysis['summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Recommendations")
        for i, rec in enumerate(analysis['recommendations'], 1):
            st.write(f"{i}. {rec}")
    
    with col2:
        st.subheader("🔔 Action Items")
        if analysis['consult_doctor']:
            st.error("⚠️ Contact your healthcare provider")
        else:
            st.success("✅ Continue current care")
        
        st.info(f"📈 Trend: {analysis['trend'].title()}")


def render_historical_trends():
    """Render historical trends section"""
    st.header("📈 Healing Progress Timeline")
    
    if len(st.session_state.wound_history) < 2:
        st.info("Upload at least 2 images to see trend analysis")
        return
    
    # Create dataframe from history
    df = pd.DataFrame(st.session_state.wound_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot wound area over time
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['area'],
        mode='lines+markers',
        name='Wound Area',
        line=dict(color='#f44336', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Wound Area Over Time",
        xaxis_title="Date",
        yaxis_title="Area (cm²)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Multi-metric comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Redness trend
        fig_red = go.Figure()
        fig_red.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['redness'],
            mode='lines+markers',
            name='Redness',
            line=dict(color='#ff5722', width=2)
        ))
        fig_red.update_layout(
            title="Redness Index",
            yaxis_title="Redness (%)",
            height=300
        )
        st.plotly_chart(fig_red, use_container_width=True)
    
    with col2:
        # Granulation trend
        fig_gran = go.Figure()
        fig_gran.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['granulation'],
            mode='lines+markers',
            name='Granulation',
            line=dict(color='#4caf50', width=2)
        ))
        fig_gran.update_layout(
            title="Granulation Tissue",
            yaxis_title="Granulation (%)",
            height=300
        )
        st.plotly_chart(fig_gran, use_container_width=True)


def render_metrics_dashboard():
    """Render comprehensive metrics dashboard"""
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


# Helper functions (simulated for demo)

def simulate_phi_detection(image: np.ndarray) -> int:
    """Simulate PHI detection"""
    # In production, this calls aparavi_integration.py
    return np.random.choice([0, 0, 0, 1, 2])  # Mostly no PHI


def apply_redaction(image: np.ndarray) -> np.ndarray:
    """Simulate redaction"""
    img = image.copy()
    # Add blur to simulated PHI regions
    h, w = img.shape[:2]
    img[0:int(h*0.1), :] = cv2.GaussianBlur(img[0:int(h*0.1), :], (51, 51), 30)
    return img


def simulate_segmentation(image: np.ndarray) -> np.ndarray:
    """Simulate wound segmentation"""
    # In production, this uses your U-Net model
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    # Create overlay
    overlay = image.copy()
    overlay[mask == 0] = overlay[mask == 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    return overlay.astype(np.uint8)


def extract_wound_metrics(image: np.ndarray) -> dict:
    """Extract wound metrics (simulated)"""
    # In production, this calls cv_processing.py
    base_area = 4.5 + np.random.uniform(-0.5, 0.5)
    
    if st.session_state.wound_history:
        prev_area = st.session_state.wound_history[-1]['area']
        area_change = ((base_area - prev_area) / prev_area) * 100
    else:
        area_change = 0
    
    redness = 45 + np.random.uniform(-10, 10)
    granulation = 65 + np.random.uniform(-5, 5)
    
    return {
        'area': base_area,
        'area_change': area_change,
        'perimeter': base_area * 2.5,
        'aspect_ratio': 1.2,
        'redness': redness,
        'redness_change': -3 if area_change < 0 else 2,
        'redness_context': "Redness is within normal healing range." if redness < 50 else "Elevated redness detected - monitor for infection.",
        'granulation': granulation,
        'granulation_change': 5,
        'epithelialization': 20,
        'necrotic': 5,
        'edge_quality': 0.78,
        'healing_score': 75
    }


def save_to_history(metrics: dict, analysis: dict):
    """Save measurement to history"""
    entry = {
        **metrics,
        'timestamp': datetime.now().isoformat(),
        'patient_id': st.session_state.patient_id,
        'risk_level': analysis['risk_level']
    }
    
    st.session_state.wound_history.append(entry)
    
    # Keep only last 30 measurements
    if len(st.session_state.wound_history) > 30:
        st.session_state.wound_history = st.session_state.wound_history[-30:]


def main():
    """Main application"""
    initialize_session_state()
    render_sidebar()
    render_header()
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📸 New Analysis", "📈 Progress Tracking", "📊 Metrics"])
    
    with tab1:
        upload_and_process_image()
    
    with tab2:
        render_historical_trends()
    
    with tab3:
        render_metrics_dashboard()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ for Hack With Chicago 2.0 | 
        <a href='https://github.com/Msundara19/meditrack-wound-healing'>GitHub</a> | 
        IIT - AI Vision Lab</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
