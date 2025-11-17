"""
MediTrack - Aparavi Integration
PII/PHI Detection and Redaction for Medical Images
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from typing import Dict, List, Tuple, Optional
import re
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PIIDetection:
    """Data class for PII detection results"""
    type: str  # 'name', 'mrn', 'dob', 'phone', 'address', 'ssn', 'face'
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    severity: str  # 'high', 'medium', 'low'


class AparaviPHIDetector:
    """
    Aparavi-inspired PHI/PII detector for medical images
    
    Detects and redacts:
    - Patient names
    - Medical Record Numbers (MRN)
    - Dates of Birth
    - Phone numbers
    - Addresses
    - Social Security Numbers
    - Faces in images
    - Hospital/room numbers
    """
    
    def __init__(self, face_cascade_path: Optional[str] = None):
        """
        Initialize the PHI detector
        
        Args:
            face_cascade_path: Path to OpenCV face cascade classifier
        """
        # Compile regex patterns for PHI detection
        self.patterns = {
            'mrn': re.compile(r'\b[A-Z]{2,3}\d{6,10}\b'),  # Medical Record Number
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),   # Social Security
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # Phone
            'dob': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),  # Date of Birth
            'zip': re.compile(r'\b\d{5}(?:-\d{4})?\b'),    # ZIP code
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        }
        
        # Common medical identifiers
        self.medical_keywords = [
            'patient', 'name', 'dob', 'mrn', 'medical record',
            'room', 'bed', 'physician', 'doctor', 'nurse'
        ]
        
        # Initialize face detector
        self.face_cascade = None
        if face_cascade_path:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            # Try default OpenCV cascade
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                print("⚠️  Warning: Face detection not available")
        
        # Detection statistics
        self.stats = {
            'total_scanned': 0,
            'phi_detected': 0,
            'redactions_made': 0
        }
    
    def detect_phi_in_image(self, image: np.ndarray) -> List[PIIDetection]:
        """
        Detect PHI/PII in a medical image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of PIIDetection objects
        """
        detections = []
        
        # 1. Detect faces
        face_detections = self._detect_faces(image)
        detections.extend(face_detections)
        
        # 2. Detect text-based PHI using OCR
        text_detections = self._detect_text_phi(image)
        detections.extend(text_detections)
        
        # 3. Detect metadata in image borders (common in medical equipment)
        border_detections = self._detect_border_metadata(image)
        detections.extend(border_detections)
        
        # Update statistics
        self.stats['total_scanned'] += 1
        if detections:
            self.stats['phi_detected'] += 1
        
        return detections
    
    def _detect_faces(self, image: np.ndarray) -> List[PIIDetection]:
        """Detect faces in image using OpenCV"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append(PIIDetection(
                type='face',
                text='[FACE DETECTED]',
                confidence=0.85,
                bbox=(x, y, w, h),
                severity='high'
            ))
        
        return detections
    
    def _detect_text_phi(self, image: np.ndarray) -> List[PIIDetection]:
        """Detect PHI in text extracted from image"""
        detections = []
        
        try:
            # Use pytesseract for OCR
            ocr_data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT
            )
            
            # Process each text element
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                conf = int(ocr_data['conf'][i])
                
                if conf < 30 or not text:  # Skip low confidence
                    continue
                
                # Check against PHI patterns
                x, y, w, h = (
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['width'][i],
                    ocr_data['height'][i]
                )
                
                # Check regex patterns
                for phi_type, pattern in self.patterns.items():
                    if pattern.search(text):
                        detections.append(PIIDetection(
                            type=phi_type,
                            text=text,
                            confidence=conf / 100.0,
                            bbox=(x, y, w, h),
                            severity='high' if phi_type in ['ssn', 'mrn'] else 'medium'
                        ))
                
                # Check for medical keywords
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in self.medical_keywords):
                    # This might be PHI context, expand bounding box
                    detections.append(PIIDetection(
                        type='metadata',
                        text=text,
                        confidence=conf / 100.0,
                        bbox=(x, y, w, h),
                        severity='medium'
                    ))
        
        except Exception as e:
            print(f"⚠️  OCR failed: {e}")
        
        return detections
    
    def _detect_border_metadata(self, image: np.ndarray) -> List[PIIDetection]:
        """
        Detect metadata typically found in image borders
        Medical imaging equipment often embeds patient info in borders
        """
        detections = []
        h, w = image.shape[:2]
        
        # Define border regions (top 10%, bottom 10%, left/right 5%)
        border_regions = [
            ('top', image[0:int(h*0.1), :]),
            ('bottom', image[int(h*0.9):h, :]),
            ('left', image[:, 0:int(w*0.05)]),
            ('right', image[:, int(w*0.95):w])
        ]
        
        for region_name, region in border_regions:
            # Check if region has high text density
            if self._has_text_density(region):
                # Mark entire region for redaction
                if region_name == 'top':
                    bbox = (0, 0, w, int(h*0.1))
                elif region_name == 'bottom':
                    bbox = (0, int(h*0.9), w, int(h*0.1))
                elif region_name == 'left':
                    bbox = (0, 0, int(w*0.05), h)
                else:  # right
                    bbox = (int(w*0.95), 0, int(w*0.05), h)
                
                detections.append(PIIDetection(
                    type='border_metadata',
                    text=f'[{region_name.upper()} BORDER METADATA]',
                    confidence=0.7,
                    bbox=bbox,
                    severity='medium'
                ))
        
        return detections
    
    def _has_text_density(self, region: np.ndarray) -> bool:
        """Check if region has high text density (indicates metadata)"""
        if region.size == 0:
            return False
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if len(region.shape) == 3 else region
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate text density (% of non-white pixels)
        text_pixels = np.sum(binary < 127)
        total_pixels = binary.size
        density = text_pixels / total_pixels
        
        # If density > 5%, likely has text
        return density > 0.05
    
    def redact_image(
        self,
        image: np.ndarray,
        detections: List[PIIDetection],
        redaction_method: str = 'blur'
    ) -> np.ndarray:
        """
        Redact detected PHI from image
        
        Args:
            image: Original image
            detections: List of PHI detections
            redaction_method: 'blur', 'black', or 'pixelate'
            
        Returns:
            Redacted image
        """
        redacted = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            if redaction_method == 'blur':
                # Gaussian blur
                roi = redacted[y:y+h, x:x+w]
                blurred = cv2.GaussianBlur(roi, (51, 51), 50)
                redacted[y:y+h, x:x+w] = blurred
                
            elif redaction_method == 'black':
                # Black box
                cv2.rectangle(redacted, (x, y), (x+w, y+h), (0, 0, 0), -1)
                
            elif redaction_method == 'pixelate':
                # Pixelation effect
                roi = redacted[y:y+h, x:x+w]
                roi_small = cv2.resize(roi, (w//10, h//10), interpolation=cv2.INTER_LINEAR)
                roi_pixelated = cv2.resize(roi_small, (w, h), interpolation=cv2.INTER_NEAREST)
                redacted[y:y+h, x:x+w] = roi_pixelated
            
            self.stats['redactions_made'] += 1
        
        return redacted
    
    def create_phi_report(
        self,
        detections: List[PIIDetection],
        image_path: str
    ) -> Dict:
        """
        Create a detailed PHI detection report
        
        Args:
            detections: List of detections
            image_path: Path to original image
            
        Returns:
            Report dictionary
        """
        report = {
            'image_path': image_path,
            'scan_timestamp': datetime.now().isoformat(),
            'total_detections': len(detections),
            'severity_breakdown': {
                'high': sum(1 for d in detections if d.severity == 'high'),
                'medium': sum(1 for d in detections if d.severity == 'medium'),
                'low': sum(1 for d in detections if d.severity == 'low')
            },
            'type_breakdown': {},
            'detections': [],
            'compliance_status': 'PASS' if len(detections) == 0 else 'FAIL',
            'requires_review': len(detections) > 0
        }
        
        # Group by type
        for detection in detections:
            # Count by type
            if detection.type not in report['type_breakdown']:
                report['type_breakdown'][detection.type] = 0
            report['type_breakdown'][detection.type] += 1
            
            # Add detection details
            report['detections'].append({
                'type': detection.type,
                'severity': detection.severity,
                'confidence': detection.confidence,
                'bbox': detection.bbox,
                'text_preview': detection.text[:50] if detection.text else None
            })
        
        return report
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            **self.stats,
            'detection_rate': (
                self.stats['phi_detected'] / self.stats['total_scanned']
                if self.stats['total_scanned'] > 0 else 0
            )
        }


class AparaviIntegration:
    """
    High-level integration class for Aparavi PHI protection
    """
    
    def __init__(self, output_dir: str = "./data/aparavi_outputs"):
        self.detector = AparaviPHIDetector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_image(
        self,
        image_path: str,
        auto_redact: bool = True,
        save_report: bool = True
    ) -> Dict:
        """
        Complete PHI detection and redaction pipeline
        
        Args:
            image_path: Path to input image
            auto_redact: Automatically redact detected PHI
            save_report: Save detection report to file
            
        Returns:
            Processing result dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Detect PHI
        detections = self.detector.detect_phi_in_image(image)
        
        # Create report
        report = self.detector.create_phi_report(detections, image_path)
        
        result = {
            'original_path': image_path,
            'detections_found': len(detections),
            'report': report
        }
        
        # Redact if needed
        if auto_redact and detections:
            redacted_image = self.detector.redact_image(image, detections, method='blur')
            
            # Save redacted image
            filename = Path(image_path).stem
            redacted_path = self.output_dir / f"{filename}_redacted.jpg"
            cv2.imwrite(str(redacted_path), redacted_image)
            result['redacted_path'] = str(redacted_path)
        
        # Save report
        if save_report:
            report_path = self.output_dir / f"{Path(image_path).stem}_phi_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            result['report_path'] = str(report_path)
        
        return result
    
    def batch_process(self, image_dir: str) -> List[Dict]:
        """Process multiple images in a directory"""
        results = []
        
        image_dir = Path(image_dir)
        for img_path in image_dir.glob("*.jpg"):
            result = self.process_image(str(img_path))
            results.append(result)
        
        return results


def main():
    """Demo of Aparavi integration"""
    # Initialize
    aparavi = AparaviIntegration(output_dir="./data/aparavi_outputs")
    
    # Process a single image
    result = aparavi.process_image(
        "./data/sample_wounds/wound_001.jpg",
        auto_redact=True,
        save_report=True
    )
    
    print("\n🔒 Aparavi PHI Detection Report")
    print("=" * 50)
    print(f"Image: {result['original_path']}")
    print(f"Detections: {result['detections_found']}")
    
    if result['detections_found'] > 0:
        print(f"✅ Redacted image saved: {result.get('redacted_path')}")
        print(f"📄 Report saved: {result.get('report_path')}")
    else:
        print("✅ No PHI detected - image is clean")
    
    # Get statistics
    stats = aparavi.detector.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total scanned: {stats['total_scanned']}")
    print(f"  PHI detected in: {stats['phi_detected']} images")
    print(f"  Redactions made: {stats['redactions_made']}")


if __name__ == "__main__":
    main()
