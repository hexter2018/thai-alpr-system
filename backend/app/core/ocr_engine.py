"""
OCR Engine for Thai License Plate Recognition
Supports PaddleOCR (primary) and Tesseract (fallback)
"""
import re
import logging
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
from pathlib import Path

from ..utils.thai_provinces import (
    normalize_province_name,
    validate_plate_format,
    format_plate_text,
    extract_plate_components
)

logger = logging.getLogger(__name__)


class ThaiOCREngine:
    """
    Thai License Plate OCR Engine
    Optimized for Thai character recognition with post-processing
    """
    
    def __init__(
        self,
        engine: str = "paddleocr",
        use_gpu: bool = True,
        lang: str = "th"
    ):
        """
        Initialize OCR engine
        
        Args:
            engine: "paddleocr" or "tesseract"
            use_gpu: Enable GPU acceleration
            lang: Language code ("th" for Thai)
        """
        self.engine_type = engine.lower()
        self.use_gpu = use_gpu
        self.lang = lang
        
        if self.engine_type == "paddleocr":
            self._init_paddle_ocr()
        elif self.engine_type == "tesseract":
            self._init_tesseract()
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")
        
        logger.info(f"OCR Engine initialized: {self.engine_type}, GPU: {self.use_gpu}")
    
    def _init_paddle_ocr(self):
        """Initialize PaddleOCR with Thai language support"""
        try:
            from paddleocr import PaddleOCR

            paddle_lang = self._resolve_paddle_lang(self.lang)
            
            self.ocr = PaddleOCR(
                use_angle_cls=True,      # Enable text angle classification
                lang=paddle_lang,        # Thai language
                use_gpu=self.use_gpu,    # GPU acceleration
                show_log=False,          # Suppress logs
                
                # Detection parameters (optimized for license plates)
                det_db_thresh=0.3,       # Binary threshold for detection
                det_db_box_thresh=0.5,   # Box threshold
                det_db_unclip_ratio=1.6, # Unclip ratio for text region
                
                # Recognition parameters
                rec_batch_num=6,         # Batch size for recognition
                drop_score=0.3,          # Drop results below this score
                
                # Performance
                use_mp=True,             # Enable multiprocessing
                total_process_num=2,     # Number of processes
            )
            
            logger.info("PaddleOCR initialized successfully")
            
        except ImportError:
            logger.error("PaddleOCR not installed. Install: pip install paddleocr paddlepaddle-gpu")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    @staticmethod
    def _resolve_paddle_lang(lang: str) -> str:
        """Map requested language to a PaddleOCR supported language code."""
        lang_code = (lang or "").lower().strip()

        # PaddleOCR does not expose `th` directly; `latin` is the closest
        # multilingual recognizer and avoids startup crashes.
        if lang_code in {"th", "thai"}:
            logger.warning(
                "PaddleOCR does not support '%s' directly. Falling back to 'en'.",
                lang,
            )
            return "en"

        return lang_code or "en"
    
    def _init_tesseract(self):
        """Initialize Tesseract OCR (fallback option)"""
        try:
            import pytesseract
            
            self.ocr = pytesseract
            
            # Verify Tesseract installation
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract initialized: version {version}")
            
        except ImportError:
            logger.error("pytesseract not installed. Install: pip install pytesseract")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract: {e}")
            raise
    
    def preprocess_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR accuracy
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to optimal height (improves OCR accuracy)
        target_height = 64
        aspect_ratio = gray.shape[1] / gray.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter (reduces noise while preserving edges)
        filtered = cv2.bilateralFilter(resized, 11, 17, 17)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Invert if text is white on black background
        mean_val = np.mean(morphed)
        if mean_val < 127:
            morphed = cv2.bitwise_not(morphed)
        
        return morphed
    
    def recognize_plate(
        self,
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, any]:
        """
        Recognize license plate text from image
        
        Args:
            image: Plate image (cropped)
            preprocess: Apply preprocessing
        
        Returns:
            Dictionary with detection results
        """
        try:
            # Preprocess
            if preprocess:
                processed_image = self.preprocess_plate_image(image)
            else:
                processed_image = image
            
            # Run OCR
            if self.engine_type == "paddleocr":
                results = self._paddle_recognize(processed_image)
            else:
                results = self._tesseract_recognize(processed_image)
            
            # Post-process results
            processed_results = self._post_process_results(results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"OCR recognition failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "province": None,
                "raw_output": None,
                "success": False,
                "error": str(e)
            }
    
    def _paddle_recognize(self, image: np.ndarray) -> List[Tuple]:
        """Run PaddleOCR recognition"""
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return []
        
        # Extract text and confidence
        detections = []
        for line in result[0]:
            text = line[1][0]  # Recognized text
            confidence = line[1][1]  # Confidence score
            bbox = line[0]  # Bounding box
            
            detections.append((text, confidence, bbox))
        
        return detections
    
    def _tesseract_recognize(self, image: np.ndarray) -> List[Tuple]:
        """Run Tesseract recognition"""
        import pytesseract
        
        # Configuration for Thai + License plate
        config = '--psm 7 --oem 3 -l tha'  # PSM 7: single line, OEM 3: default
        
        # Get detailed data
        data = pytesseract.image_to_data(
            image,
            lang='tha',
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text with confidence
        detections = []
        text_parts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if int(conf) > 0:  # Valid detection
                text_parts.append(data['text'][i])
                confidences.append(int(conf) / 100.0)
        
        if text_parts:
            full_text = ''.join(text_parts)
            avg_confidence = np.mean(confidences)
            detections.append((full_text, avg_confidence, None))
        
        return detections
    
    def _post_process_results(self, detections: List[Tuple]) -> Dict[str, any]:
        """
        Post-process OCR results
        Apply corrections, extract province, calculate confidence
        """
        if not detections:
            return {
                "text": "",
                "confidence": 0.0,
                "province": None,
                "raw_output": None,
                "success": False
            }
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        # Get best detection
        best_text, best_confidence, bbox = sorted_detections[0]
        
        # Format plate text
        formatted_text = format_plate_text(best_text)
        
        # Validate format
        is_valid = validate_plate_format(formatted_text)
        
        # Extract components
        components = extract_plate_components(formatted_text)
        
        # Try to extract province from nearby text
        province = self._extract_province_from_detections(detections)
        
        # Apply OCR corrections
        corrected_text = self._apply_ocr_corrections(formatted_text)
        
        return {
            "text": corrected_text,
            "confidence": float(best_confidence),
            "province": province,
            "components": components,
            "is_valid_format": is_valid,
            "raw_output": {
                "original_text": best_text,
                "all_detections": [(t, float(c)) for t, c, _ in detections]
            },
            "success": True
        }
    
    def _extract_province_from_detections(self, detections: List[Tuple]) -> Optional[str]:
        """
        Try to extract province name from OCR detections
        Often appears near the plate number
        """
        for text, confidence, _ in detections:
            if confidence < 0.3:  # Skip low confidence
                continue
            
            # Try to normalize as province
            province = normalize_province_name(text)
            if province:
                return province
        
        return None
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """
        Apply common OCR error corrections
        Handle character confusion (e.g., 0 vs O, 1 vs I)
        """
        corrections = {
            # Thai character corrections
            'O': '0',  # O (letter) -> 0 (number)
            'o': '0',
            'I': '1',  # I (letter) -> 1 (number)
            'l': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            
            # Remove common artifacts
            '.': '',
            ',': '',
            ' ': '',
            '|': '1',
        }
        
        corrected = text
        for wrong, correct in corrections.items():
            corrected = corrected.replace(wrong, correct)
        
        return corrected
    
    def batch_recognize(
        self,
        images: List[np.ndarray],
        preprocess: bool = True
    ) -> List[Dict[str, any]]:
        """
        Batch recognition for multiple plates
        
        Args:
            images: List of plate images
            preprocess: Apply preprocessing
        
        Returns:
            List of recognition results
        """
        results = []
        
        for image in images:
            result = self.recognize_plate(image, preprocess=preprocess)
            results.append(result)
        
        return results


def get_ocr_engine(
    engine: str = "paddleocr",
    use_gpu: bool = True,
    lang: str = "th"
) -> ThaiOCREngine:
    """
    Factory function to get OCR engine instance
    
    Args:
        engine: OCR engine type
        use_gpu: Enable GPU
        lang: Language
    
    Returns:
        OCR engine instance
    """
    return ThaiOCREngine(engine=engine, use_gpu=use_gpu, lang=lang)