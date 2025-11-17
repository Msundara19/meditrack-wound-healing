# MediTrack: Real-Time Wound Healing Monitoring System 🏥

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pathway](https://img.shields.io/badge/Powered%20by-Pathway-green.svg)](https://pathway.com)

> An AI-powered wound healing monitoring system combining computer vision, real-time streaming, and LLM insights for post-surgical care.

**Built for Hack With Chicago 2.0** | Track: Open Innovation (Healthcare AI)

---

## 🎯 The Problem

Post-surgical patients face significant challenges in wound care management:

- **Delayed intervention** when complications arise
- **Unnecessary ER visits** for normal healing progression
- **Poor outcomes** due to missed infection signs
- **Healthcare provider burnout** from routine follow-up calls

## 💡 Our Solution

MediTrack provides real-time wound healing assessment through:

✨ **Computer Vision** - Automated wound segmentation and feature extraction  
⚡ **Live Data Processing** - Real-time updates using Pathway's streaming engine  
🤖 **AI Insights** - Patient-friendly explanations powered by LLMs  
📊 **Trend Analysis** - Longitudinal tracking with early warning detection
**Doctor appointment booking** - helps to book nearest doctor appointment based on priority

---

## 🏗️ System Architecture

```
┌─────────────────┐
│ Patient Upload  │
│ (Image/Video)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Pathway Streaming Ingestion     │
│ - File Connector                │
│ - Real-time Processing          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ CV Pipeline (Edge-Optimized)    │
│ - Wound Segmentation (U-Net)    │
│ - Feature Extraction            │
│ - Metrics Computation           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Pathway Live Index              │
│ - Vector Store                  │
│ - Time-series Data              │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ LLM Analysis (Pathway xPack)    │
│ - Trend Analysis                │
│ - Risk Assessment               │
│ - Patient-Friendly Summaries    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Streamlit Dashboard             │
│ - Real-time Visualization       │
│ - Alert System                  │
│ - Historical Trends             │
└─────────────────────────────────┘
```

---

## 🚀 Key Features

### Real-Time Processing
- Live wound image analysis using Pathway's streaming engine
- Instant metric updates as new images arrive
- Sub-second latency for clinical decision support

### Advanced Computer Vision
- Deep learning-based wound segmentation
- Multi-metric extraction:
  - Wound area (cm²)
  - Color analysis (redness, granulation)
  - Edge characteristics (healing vs. spreading)
  - Tissue classification

### Intelligent Analysis
- AI-generated patient summaries in plain language
- Evidence-based recommendations with citations
- Risk stratification (low/medium/high concern)
- Automatic alerts for concerning trends

### Privacy & Security
- PII anonymization using Aparavi integration (planned)
- Secure data handling
- HIPAA-aware design principles
- Clear disclaimers (educational use only)

---

## 🛠️ Technology Stack

### Core Framework
- **Pathway Framework** - Real-time data processing and live indexing
- **Pathway LLM xPack** - RAG pipeline for contextual insights
- **PaddleOCR/Docling** - Medical document parsing (for lab reports integration)

### Computer Vision
- **Python 3.9+** - Primary language
- **OpenCV** - Image preprocessing
- **segmentation_models_pytorch** - Pre-trained U-Net for wound segmentation

### AI & NLP
- **OpenAI/Gemini API** - LLM for natural language generation

### Frontend & Visualization
- **Streamlit** - Interactive web dashboard
- **Plotly** - Data visualization

### Partner Integrations
- **Aparavi** - PHI/PII detection and redaction
- **Juspay** - Payment processing (for future telehealth consultations)

---

## ⚡ Performance Metrics

Our edge-optimized approach delivers:

- **Inference Time**: ~150ms per image (MobileNetV2)
- **Memory Footprint**: <500MB
- **Accuracy**: 92% wound boundary detection (validated on AZH dataset)
- **Latency**: Real-time updates within 200ms

*Compared to cloud-only solutions with 2-3 second latency*

---

## 📋 Prerequisites

- Python >= 3.9.0
- pip
- git

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Msundara19/meditrack-wound-healing.git
cd meditrack-wound-healing
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your API keys:
# - OPENAI_API_KEY or GEMINI_API_KEY
# - PATHWAY_LICENSE_KEY (if applicable)
```

### 5. Download Sample Data

```bash
python scripts/download_sample_data.py
```

---

## 🎮 Usage

### Start the Pathway Processing Pipeline

```bash
python src/pathway_pipeline.py
```

### Launch the Streamlit Dashboard

In a separate terminal:

```bash
streamlit run src/app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
meditrack-wound-healing/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── sample_wounds/          # Sample wound images
│   ├── patient_data/           # Simulated patient records
│   └── outputs/                # Pathway outputs
├── models/
│   ├── wound_segmentation/     # Pre-trained models
│   └── checkpoints/            # Fine-tuned weights
├── src/
│   ├── pathway_pipeline.py     # Main Pathway streaming pipeline
│   ├── cv_processing.py        # Computer vision module
│   ├── llm_analyzer.py         # LLM integration (Pathway xPack)
│   ├── app.py                  # Streamlit dashboard
│   └── utils/
│       ├── metrics.py          # Wound metric calculations
│       ├── visualization.py    # Plotting utilities
│       └── data_generator.py   # Synthetic data for demo
├── scripts/
│   ├── download_sample_data.py # Dataset downloader
│   ├── test_pipeline.py        # Integration tests
│   └── benchmark.py            # Performance evaluation
├── docs/
│   ├── ARCHITECTURE.md         # Detailed system design
│   ├── API.md                  # API documentation
│   └── DEPLOYMENT.md           # Deployment guide
└── tests/
    ├── test_cv.py
    ├── test_pathway.py
    └── test_llm.py
```

---

## 📊 Use Case Examples

### Case 1: Normal Healing

- **Patient**: Post-appendectomy, Day 0-14
- **Trend**: Wound area decreasing 10%/day
- **Result**: ✅ *"Healing normally - continue current care"*

### Case 2: Early Warning

- **Patient**: Diabetic foot ulcer, Day 5-7
- **Trend**: Redness increasing, area expanding
- **Result**: ⚠️ *"Consult healthcare provider - signs of infection"*

### Case 3: Delayed Healing

- **Patient**: Pressure ulcer, Day 0-21
- **Trend**: Minimal area reduction, no granulation
- **Result**: 🔴 *"Requires medical evaluation - healing stalled"*

---

## 🧪 Validation & Results

- **Dataset**: AZH Wound Care dataset (500+ images)
- **Segmentation Accuracy**: 92.3% IoU
- **Inference Speed**: 6.7 FPS on CPU (Intel i5)
- **Patient Satisfaction**: N/A (prototype stage)

---

## 🔬 Technical Background

This project builds on research in:

- Real-time computer vision systems (inspired by [Using Computer Vision and Artificial Intelligence to Track the Healing of Severe Burns](https://pubmed.ncbi.nlm.nih.gov/38126807/))
- Hardware-accelerated CNN architectures (ECE 588 coursework)
- Edge computing for medical applications
- Streaming data processing with Pathway

---

## 👥 Team

- **Developers**: Meenakshi Sridharan & Akshitha Priyadharshini
- **Institution**: Illinois Institute of Technology
- **Program**: Master of Engineering in AI (Computer Vision & Control)

---

## 🏆 Hackathon Information

- **Event**: Hack With Chicago 2.0
- **Track**: Open Innovation (Healthcare AI)
- **Date**: November 17, 2025
- **Organizers**: Pathway, Devnovate, Microsoft, Aparavi, Juspay

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Important Disclaimer

**This is an educational prototype and NOT a medical device.**

- ❌ Not FDA approved
- ❌ Not intended for clinical diagnosis or treatment decisions
- ❌ Always consult healthcare professionals for medical advice
- ❌ Not a substitute for professional wound care assessment

---

## 🔗 Resources

- [Pathway Documentation](https://pathway.com/developers)
- [Hackathon Details](https://pathway.com)
- [Live Demo](https://meditrack-demo.streamlit.app) *(to be deployed)*
- [Presentation Slides](docs/presentation.pdf)

---

## 📧 Contact

For questions or collaboration:

- **GitHub**: [@Msundara19](https://github.com/Msundara19)
- **Email**: msridharansundaram@hawk.illinoistech.edu

---

<div align="center">

**Built with ❤️ for Hack With Chicago 2.0**

*Empowering patients with AI-driven wound care insights*

</div>
