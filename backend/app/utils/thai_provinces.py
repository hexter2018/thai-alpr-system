"""
Thai Provinces Master List
Complete list of all 77 Thai provinces with common OCR variations
"""
from typing import Dict, List, Set, Optional
import re

# Official 77 Thai Provinces
THAI_PROVINCES = [
    "กรุงเทพมหานคร", "สมุทรปราการ", "นนทบุรี", "ปทุมธานี", "พระนครศรีอยุธยา",
    "อ่างทอง", "ลพบุรี", "สิงห์บุรี", "ชัยนาท", "สระบุรี",
    "ชลบุรี", "ระยอง", "จันทบุรี", "ตราด", "ฉะเชิงเทรา",
    "ปราจีนบุรี", "นครนายก", "สระแก้ว", "นครราชสีมา", "บุรีรัมย์",
    "สุรินทร์", "ศรีสะเกษ", "อุบลราชธานี", "ยโสธร", "ชัยภูมิ",
    "อำนาจเจริญ", "บึงกาฬ", "หนองบัวลำภู", "ขอนแก่น", "อุดรธานี",
    "เลย", "หนองคาย", "มหาสารคาม", "ร้อยเอ็ด", "กาฬสินธุ์",
    "สกลนคร", "นครพนม", "มุกดาหาร", "เชียงใหม่", "ลำพูน",
    "ลำปาง", "อุตรดิตถ์", "แพร่", "น่าน", "พะเยา",
    "เชียงราย", "แม่ฮ่องสอน", "นครสวรรค์", "อุทัยธานี", "กำแพงเพชร",
    "ตาก", "สุโขทัย", "พิษณุโลก", "พิจิตร", "เพชรบูรณ์",
    "ราชบุรี", "กาญจนบุรี", "สุพรรณบุรี", "นครปฐม", "สมุทรสาคร",
    "สมุทรสงคราม", "เพชรบุรี", "ประจวบคีรีขันธ์", "นครศรีธรรมราช", "กระบี่",
    "พังงา", "ภูเก็ต", "สุราษฎร์ธานี", "ระนอง", "ชุมพร",
    "สงขลา", "สตูล", "ตรัง", "พัทลุง", "ปัตตานี",
    "ยะลา", "นราธิวาส"
]

# Common OCR misreads and variations
OCR_PROVINCE_VARIATIONS: Dict[str, List[str]] = {
    "กรุงเทพมหานคร": ["กทม", "กทม.", "กรุงเทพ", "กรุงเทพฯ", "Bangkok"],
    "สมุทรปราการ": ["สป", "ส.ป.", "สมุทรปราการ์"],
    "นนทบุรี": ["นบ", "น.บ.", "นนท์"],
    "ปทุมธานี": ["ปท", "ป.ท."],
    "พระนครศรีอยุธยา": ["พนศ", "อยุธยา", "พ.น.ศ."],
    "ชลบุรี": ["ชบ", "ช.บ."],
    "ระยอง": ["ระยอง", "รย", "ร.ย."],
    "เชียงใหม่": ["ชม", "ช.ม.", "เชียงใหม่"],
    "นครราชสีมา": ["นม", "น.ม.", "โคราช"],
    "สงขลา": ["สข", "ส.ข."],
    "ภูเก็ต": ["ภก", "ภ.ก.", "Phuket"],
    "นครศรีธรรมราช": ["นศ", "น.ศ."],
    "อุบลราชธานี": ["อบ", "อ.บ."],
    "ขอนแก่น": ["ขก", "ข.ก."],
    "อุดรธานี": ["อด", "อ.ด."],
}

# Province abbreviations (2-letter codes used on plates)
PROVINCE_ABBREVIATIONS: Dict[str, str] = {
    "กท": "กรุงเทพมหานคร",
    "สป": "สมุทรปราการ",
    "นบ": "นนทบุรี",
    "ปท": "ปทุมธานี",
    "พศ": "พระนครศรีอยุธยา",
    "อท": "อ่างทอง",
    "ลบ": "ลพบุรี",
    "สห": "สิงห์บุรี",
    "ชน": "ชัยนาท",
    "สบ": "สระบุรี",
    "ชบ": "ชลบุรี",
    "รย": "ระยอง",
    "จบ": "จันทบุรี",
    "ตร": "ตราด",
    "ฉช": "ฉะเชิงเทรา",
    "ปจ": "ปราจีนบุรี",
    "นย": "นครนายก",
    "สก": "สระแก้ว",
    "นม": "นครราชสีมา",
    "บร": "บุรีรัมย์",
    "สร": "สุรินทร์",
    "ศก": "ศรีสะเกษ",
    "อบ": "อุบลราชธานี",
    "ยส": "ยโสธร",
    "ชย": "ชัยภูมิ",
    "อจ": "อำนาจเจริญ",
    "บก": "บึงกาฬ",
    "นภ": "หนองบัวลำภู",
    "ขก": "ขอนแก่น",
    "อด": "อุดรธานี",
    "ลย": "เลย",
    "นค": "หนองคาย",
    "มค": "มหาสารคาม",
    "รอ": "ร้อยเอ็ด",
    "กส": "กาฬสินธุ์",
    "สน": "สกลนคร",
    "นพ": "นครพนม",
    "มห": "มุกดาหาร",
    "ชม": "เชียงใหม่",
    "ลพ": "ลำพูน",
    "ลป": "ลำปาง",
    "อต": "อุตรดิตถ์",
    "พร": "แพร่",
    "นน": "น่าน",
    "พย": "พะเยา",
    "ชร": "เชียงราย",
    "มส": "แม่ฮ่องสอน",
    "นว": "นครสวรรค์",
    "อน": "อุทัยธานี",
    "กพ": "กำแพงเพชร",
    "ตก": "ตาก",
    "สท": "สุโขทัย",
    "พล": "พิษณุโลก",
    "พจ": "พิจิตร",
    "พช": "เพชรบูรณ์",
    "รบ": "ราชบุรี",
    "กจ": "กาญจนบุรี",
    "สพ": "สุพรรณบุรี",
    "นฐ": "นครปฐม",
    "สค": "สมุทรสาคร",
    "สง": "สมุทรสงคราม",
    "พบ": "เพชรบุรี",
    "ปข": "ประจวบคีรีขันธ์",
    "นศ": "นครศรีธรรมราช",
    "กบ": "กระบี่",
    "พง": "พังงา",
    "ภก": "ภูเก็ต",
    "สฎ": "สุราษฎร์ธานี",
    "รน": "ระนอง",
    "ชพ": "ชุมพร",
    "สข": "สงขลา",
    "สต": "สตูล",
    "ตง": "ตรัง",
    "พท": "พัทลุง",
    "ปน": "ปัตตานี",
    "ยล": "ยะลา",
    "นธ": "นราธิวาส",
}

# Reverse mapping
FULL_NAME_TO_ABBREV = {v: k for k, v in PROVINCE_ABBREVIATIONS.items()}


def normalize_province_name(province_text: str) -> Optional[str]:
    """
    Normalize OCR-detected province text to official province name
    Handles abbreviations, variations, and common OCR errors
    
    Args:
        province_text: Raw OCR output (e.g., "กทม", "กรุงเทพ", "Bangkok")
    
    Returns:
        Official province name or None if not found
    """
    if not province_text:
        return None
    
    # Clean input
    cleaned = province_text.strip()
    
    # Check exact match first
    if cleaned in THAI_PROVINCES:
        return cleaned
    
    # Check abbreviations
    if cleaned in PROVINCE_ABBREVIATIONS:
        return PROVINCE_ABBREVIATIONS[cleaned]
    
    # Check variations
    for official_name, variations in OCR_PROVINCE_VARIATIONS.items():
        if cleaned in variations:
            return official_name
    
    # Fuzzy matching (for OCR errors)
    best_match = find_closest_province(cleaned)
    return best_match


def find_closest_province(text: str, threshold: float = 0.7) -> Optional[str]:
    """
    Find closest matching province using similarity scoring
    Useful for correcting OCR errors
    
    Args:
        text: Input text to match
        threshold: Minimum similarity score (0.0 - 1.0)
    
    Returns:
        Best matching province name or None
    """
    from difflib import SequenceMatcher
    
    best_score = 0.0
    best_match = None
    
    for province in THAI_PROVINCES:
        score = SequenceMatcher(None, text, province).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_match = province
    
    return best_match


def validate_plate_format(plate_text: str) -> bool:
    """
    Validate Thai license plate format
    
    Common formats:
    - กก1234 (2 Thai chars + 4 digits)
    - 1กก1234 (1 digit + 2 Thai chars + 4 digits)
    - 12กก1234 (2 digits + 2 Thai chars + 4 digits)
    - กกก1234 (3 Thai chars + 4 digits) - rare
    
    Args:
        plate_text: License plate text
    
    Returns:
        True if valid format
    """
    if not plate_text:
        return False
    
    patterns = [
        r'^[ก-ฮ]{2}\d{4}$',           # กก1234
        r'^[ก-ฮ]{2}-?\d{4}$',         # กก-1234
        r'^\d{1}[ก-ฮ]{2}\d{4}$',      # 1กก1234
        r'^\d{2}[ก-ฮ]{2}\d{4}$',      # 12กก1234
        r'^[ก-ฮ]{3}\d{4}$',           # กกก1234 (rare)
        r'^\d{1,2}-\d{4}$',           # 30-1234 (motorcycle)
    ]
    
    for pattern in patterns:
        if re.match(pattern, plate_text):
            return True
    
    return False


def format_plate_text(plate_text: str) -> str:
    """
    Format plate text for consistency
    Remove spaces, standardize separators
    
    Args:
        plate_text: Raw plate text
    
    Returns:
        Formatted plate text
    """
    # Remove spaces and common OCR artifacts
    cleaned = plate_text.strip().replace(" ", "").replace(".", "").replace(",", "")
    
    # Standardize hyphen
    cleaned = cleaned.replace("—", "-").replace("–", "-")
    
    return cleaned


def extract_plate_components(plate_text: str) -> Dict[str, Optional[str]]:
    """
    Extract components from license plate text
    
    Args:
        plate_text: License plate text
    
    Returns:
        Dictionary with 'prefix', 'number', 'format'
    """
    formatted = format_plate_text(plate_text)
    
    # Pattern 1: กก1234 or กก-1234
    match = re.match(r'^([ก-ฮ]{2,3})-?(\d{4})$', formatted)
    if match:
        return {
            "prefix": match.group(1),
            "number": match.group(2),
            "format": "thai_standard"
        }
    
    # Pattern 2: 1กก1234
    match = re.match(r'^(\d{1,2})([ก-ฮ]{2})(\d{4})$', formatted)
    if match:
        return {
            "prefix": match.group(2),
            "number": f"{match.group(1)}{match.group(3)}",
            "format": "mixed"
        }
    
    # Pattern 3: 30-1234 (motorcycle)
    match = re.match(r'^(\d{1,2})-(\d{4})$', formatted)
    if match:
        return {
            "prefix": match.group(1),
            "number": match.group(2),
            "format": "motorcycle"
        }
    
    return {
        "prefix": None,
        "number": None,
        "format": "unknown"
    }


def get_all_province_names() -> List[str]:
    """Get list of all official province names"""
    return THAI_PROVINCES.copy()


def get_all_abbreviations() -> Dict[str, str]:
    """Get all province abbreviations"""
    return PROVINCE_ABBREVIATIONS.copy()


def is_valid_province(province_name: str) -> bool:
    """Check if province name is valid"""
    return province_name in THAI_PROVINCES