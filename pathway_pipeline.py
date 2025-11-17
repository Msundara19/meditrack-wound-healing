"""
MediTrack - Pathway Streaming Pipeline
Real-time wound image processing with live indexing
"""

import pathway as pw
from pathway.xpacks.llm import llms, embedders
from pathway.xpacks.llm.vector_store import VectorStoreServer
import cv2
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional
import os
from pathlib import Path

# Import your existing CV processing module
# from cv_processing import WoundSegmenter, extract_wound_metrics


class PathwayWoundPipeline:
    """
    Real-time streaming pipeline for wound image analysis using Pathway
    """
    
    def __init__(
        self,
        input_dir: str = "./data/uploads",
        output_dir: str = "./data/outputs",
        model_path: str = "./models/wound_segmentation",
        openai_api_key: Optional[str] = None
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = Path(model_path)
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM for analysis
        self.llm = llms.OpenAIChat(
            model="gpt-4",
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )
        
        # Initialize embedder for vector store
        self.embedder = embedders.OpenAIEmbedder(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        
    def setup_input_stream(self):
        """
        Set up Pathway input connector for file monitoring
        """
        # Monitor the uploads directory for new images
        input_table = pw.io.fs.read(
            self.input_dir,
            format="binary",
            mode="streaming",
            with_metadata=True
        )
        
        return input_table
    
    def process_wound_image(self, data: Dict) -> Dict:
        """
        Process a single wound image through the CV pipeline
        
        Args:
            data: Dictionary containing image binary data and metadata
            
        Returns:
            Dictionary with wound metrics and segmentation results
        """
        try:
            # Decode image from binary
            img_array = np.frombuffer(data['data'], dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Failed to decode image"}
            
            # Run wound segmentation (using your existing model)
            # segmenter = WoundSegmenter(self.model_path)
            # mask = segmenter.segment(image)
            
            # For demo purposes, using placeholder processing
            # Replace with your actual CV processing
            metrics = self._extract_metrics(image)
            
            # Add metadata
            metrics['timestamp'] = datetime.now().isoformat()
            metrics['filename'] = data.get('path', 'unknown')
            metrics['image_shape'] = image.shape
            
            return metrics
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_metrics(self, image: np.ndarray) -> Dict:
        """
        Extract wound healing metrics from image
        
        Replace this with your actual cv_processing.py methods
        """
        # Placeholder metrics - replace with actual CV processing
        height, width = image.shape[:2]
        
        # Simulate wound detection
        metrics = {
            "wound_area_cm2": 4.5,  # Replace with actual calculation
            "wound_perimeter_cm": 8.2,
            "redness_index": 0.45,  # 0-1 scale
            "granulation_percentage": 65.0,
            "epithelialization_percentage": 20.0,
            "edge_regularity": 0.78,  # 0-1, higher is more regular
            "color_analysis": {
                "red_intensity": 145.2,
                "pink_percentage": 35.5,
                "white_percentage": 15.2
            },
            "image_dimensions": {
                "width": width,
                "height": height
            }
        }
        
        return metrics
    
    def analyze_with_llm(self, metrics: Dict, historical_data: List[Dict]) -> Dict:
        """
        Use LLM to analyze wound metrics and generate patient-friendly insights
        """
        # Build context from historical data
        context = self._build_analysis_context(metrics, historical_data)
        
        prompt = f"""You are a medical AI assistant analyzing wound healing progress.

Current Metrics:
{json.dumps(metrics, indent=2)}

Historical Context:
{json.dumps(historical_data[-5:], indent=2) if historical_data else "No historical data"}

Based on the wound metrics and trends, provide:
1. A patient-friendly summary (2-3 sentences in plain English)
2. Risk assessment (low/medium/high concern)
3. Specific recommendations
4. Whether medical consultation is advised

Respond in JSON format:
{{
    "summary": "Patient-friendly explanation here",
    "risk_level": "low/medium/high",
    "trend": "improving/stable/concerning",
    "recommendations": ["recommendation 1", "recommendation 2"],
    "consult_doctor": true/false,
    "key_observations": ["observation 1", "observation 2"]
}}

Remember: This is for educational purposes only, not medical advice.
"""
        
        try:
            response = self.llm(prompt)
            analysis = json.loads(response)
            analysis['generated_at'] = datetime.now().isoformat()
            return analysis
        except Exception as e:
            return {
                "error": str(e),
                "summary": "Unable to generate analysis",
                "risk_level": "unknown"
            }
    
    def _build_analysis_context(self, current: Dict, historical: List[Dict]) -> str:
        """Build context string for LLM analysis"""
        if not historical:
            return "First measurement - no trend data available."
        
        # Calculate trends
        if len(historical) >= 2:
            prev = historical[-1]
            area_change = ((current.get('wound_area_cm2', 0) - 
                           prev.get('wound_area_cm2', 0)) / 
                          prev.get('wound_area_cm2', 1)) * 100
            
            context = f"Wound area change: {area_change:+.1f}% since last measurement. "
            
            if area_change < -5:
                context += "Wound is shrinking (positive sign). "
            elif area_change > 5:
                context += "Wound is expanding (concerning). "
            else:
                context += "Wound size is stable. "
                
            return context
        
        return "Limited historical data available."
    
    def create_vector_store(self):
        """
        Create Pathway vector store for medical knowledge base
        """
        # Load medical knowledge documents
        docs_table = pw.io.fs.read(
            "./data/medical_knowledge",
            format="plaintext",
            mode="static"
        )
        
        # Create vector store with embeddings
        vector_store = VectorStoreServer(
            docs_table,
            embedder=self.embedder
        )
        
        return vector_store
    
    def run_pipeline(self):
        """
        Main pipeline execution with Pathway streaming
        """
        print("🚀 Starting MediTrack Pathway Pipeline...")
        
        # Set up input stream
        input_stream = self.setup_input_stream()
        
        # Process images through CV pipeline
        processed = input_stream.select(
            pw.this.data,
            pw.this.path,
            metrics=pw.apply(self.process_wound_image, pw.this)
        )
        
        # Store processed data
        pw.io.jsonlines.write(
            processed,
            f"{self.output_dir}/wound_metrics.jsonl"
        )
        
        print(f"📊 Monitoring: {self.input_dir}")
        print(f"💾 Outputs: {self.output_dir}")
        print("✅ Pipeline running. Upload images to start processing...")
        
        # Run the streaming pipeline
        pw.run()


class HistoricalDataManager:
    """
    Manage historical wound data for trend analysis
    """
    
    def __init__(self, db_path: str = "./data/wound_history.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_history(self, patient_id: str) -> List[Dict]:
        """Load historical data for a patient"""
        if not self.db_path.exists():
            return []
        
        try:
            with open(self.db_path, 'r') as f:
                all_data = json.load(f)
                return all_data.get(patient_id, [])
        except Exception as e:
            print(f"Error loading history: {e}")
            return []
    
    def save_measurement(self, patient_id: str, metrics: Dict):
        """Save new measurement to history"""
        # Load existing data
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}
        
        # Add new measurement
        if patient_id not in all_data:
            all_data[patient_id] = []
        
        all_data[patient_id].append(metrics)
        
        # Save back to file
        with open(self.db_path, 'w') as f:
            json.dump(all_data, f, indent=2)
    
    def get_trend_analysis(self, patient_id: str) -> Dict:
        """Analyze trends from historical data"""
        history = self.load_history(patient_id)
        
        if len(history) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        areas = [m.get('wound_area_cm2', 0) for m in history]
        
        trend = {
            "num_measurements": len(history),
            "duration_days": self._calculate_duration(history),
            "area_change_percent": ((areas[-1] - areas[0]) / areas[0] * 100) if areas[0] > 0 else 0,
            "average_daily_change": self._calculate_daily_change(history, areas),
            "improving": areas[-1] < areas[0]
        }
        
        return trend
    
    def _calculate_duration(self, history: List[Dict]) -> float:
        """Calculate duration in days between first and last measurement"""
        if len(history) < 2:
            return 0
        
        first = datetime.fromisoformat(history[0]['timestamp'])
        last = datetime.fromisoformat(history[-1]['timestamp'])
        return (last - first).total_seconds() / 86400
    
    def _calculate_daily_change(self, history: List[Dict], areas: List[float]) -> float:
        """Calculate average daily change in wound area"""
        duration = self._calculate_duration(history)
        if duration == 0:
            return 0
        
        total_change = areas[-1] - areas[0]
        return total_change / duration


def main():
    """
    Main entry point for the Pathway pipeline
    """
    # Initialize pipeline
    pipeline = PathwayWoundPipeline(
        input_dir="./data/uploads",
        output_dir="./data/pathway_outputs",
        model_path="./models/wound_segmentation"
    )
    
    # Run streaming pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
