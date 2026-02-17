"""
OCR Post-Processing Utilities
"""
import re
from typing import Optional, Tuple
from .thai_provinces import (
    normalize_province_name,
    validate_plate_format,
    format_plate_text
)


def clean_ocr_output(text: str) -> str:
    """
    Clean OCR output text
    
    Args:
        text: Raw OCR text
    
    Returns:
        Cleaned text
    """
    # Remove whitespace
    cleaned = text.strip()
    
    # Remove common OCR artifacts
    cleaned = cleaned.replace(" ", "")
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("|", "1")
    cleaned = cleaned.replace("O", "0")
    cleaned = cleaned.replace("o", "0")
    
    # Standardize separators
    cleaned = cleaned.replace("—", "-")
    cleaned = cleaned.replace("–", "-")
    
    return cleaned


def correct_common_mistakes(text: str) -> str:
    """
    Correct common OCR mistakes for Thai plates
    
    Args:
        text: Plate text
    
    Returns:
        Corrected text
    """
    corrections = {
        # Number/letter confusion
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6',
        
        # Common Thai character mistakes
        'ว': 'ง',  # Similar looking
        'ท': 'ง',
    }
    
    result = text
    for wrong, correct in corrections.items():
        result = result.replace(wrong, correct)
    
    return result


def extract_plate_and_province(
    ocr_text: str,
    ocr_metadata: Optional[dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract plate number and province from OCR output
    
    Args:
        ocr_text: OCR detected text
        ocr_metadata: Additional OCR metadata
    
    Returns:
        (plate_text, province_name)
    """
    cleaned = clean_ocr_output(ocr_text)
    
    # Try to extract province from text
    province = None
    plate = cleaned
    
    # Pattern 1: Plate + Province (e.g., "กก1234กรุงเทพมหานคร")
    thai_pattern = r'([ก-ฮ]{2,3}\d{4})(.*)'
    match = re.match(thai_pattern, cleaned)
    
    if match:
        plate = match.group(1)
        potential_province = match.group(2)
        
        if potential_province:
            province = normalize_province_name(potential_province)
    
    # Pattern 2: Number format (e.g., "30-1234")
    number_pattern = r'(\d{1,2}-\d{4})'
    match = re.match(number_pattern, cleaned)
    
    if match:
        plate = match.group(1)
    
    # Format plate
    plate = format_plate_text(plate)
    
    # Validate
    if not validate_plate_format(plate):
        plate = correct_common_mistakes(plate)
    
    return plate, province


def calculate_ocr_confidence(
    raw_confidence: float,
    is_valid_format: bool,
    province_found: bool
) -> float:
    """
    Calculate adjusted OCR confidence
    
    Args:
        raw_confidence: Raw OCR confidence
        is_valid_format: Valid plate format
        province_found: Province detected
    
    Returns:
        Adjusted confidence (0.0 - 1.0)
    """
    confidence = raw_confidence
    
    # Penalty for invalid format
    if not is_valid_format:
        confidence *= 0.8
    
    # Bonus for province detection
    if province_found:
        confidence = min(1.0, confidence * 1.1)
    
    return confidence


def format_plate_for_display(plate: str, province: Optional[str] = None) -> str:
    """
    Format plate for display
    
    Args:
        plate: Plate text
        province: Province name
    
    Returns:
        Formatted string
    """
    if province:
        return f"{plate} {province}"
    return plate


def validate_and_correct_plate(
    ocr_text: str,
    confidence: float
) -> dict:
    """
    Validate and correct plate text
    
    Args:
        ocr_text: Raw OCR text
        confidence: OCR confidence
    
    Returns:
        Validation result dict
    """
    plate, province = extract_plate_and_province(ocr_text)
    is_valid = validate_plate_format(plate) if plate else False
    
    adjusted_confidence = calculate_ocr_confidence(
        confidence,
        is_valid,
        province is not None
    )
    
    return {
        "original_text": ocr_text,
        "plate": plate,
        "province": province,
        "is_valid_format": is_valid,
        "raw_confidence": confidence,
        "adjusted_confidence": adjusted_confidence,
        "display_text": format_plate_for_display(plate, province)
    }