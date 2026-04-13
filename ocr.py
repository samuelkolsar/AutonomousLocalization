"""
ocr.py: Road sign detection and OCR pipeline.

Detects road signs using YOLO, reads text with EasyOCR, and parses
city name + distance pairs (e.g. "TARTU 82") from the detected text.
Includes fuzzy correction of OCR output against known Estonian place names.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
from rapidfuzz import process, fuzz

from place_names import ESTONIAN_PLACE_NAMES

# Configuration
YOLO_MODEL_PATH       = "models/sign_detector_v11.pt"
MIN_OCR_CONFIDENCE    = 0.25
FUZZY_MATCH_THRESHOLD = 65    # partial_ratio threshold for fuzzy correction
ROW_TOLERANCE         = 1.5   # vertical alignment tolerance (multiples of text height)

# Distance range for valid road sign distances (km)
MIN_DISTANCE_KM = 5
MAX_DISTANCE_KM = 400

# Regex patterns
CITY_RE    = re.compile(r'^[A-ZÕÄÖÜ][a-zõäöüA-ZÕÄÖÜ\-]{2,}$')
NUMBER_RE  = re.compile(r'^\d{1,4}$')
INLINE_RE  = re.compile(r'([A-ZÕÄÖÜ][a-zõäöüA-ZÕÄÖÜ\-]{2,})\s+(\d{1,4})')
E_ROUTE_RE = re.compile(r'\bE\s*(\d{2,4})\b', re.IGNORECASE)

# Lazy-loaded singletons
_yolo_model = None
_ocr_reader = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print(f"Loading YOLO model from {YOLO_MODEL_PATH} ...")
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        print("Loading EasyOCR reader (et + en) ...")
        _ocr_reader = easyocr.Reader(['et', 'en'], gpu=False)
    return _ocr_reader


# Preprocessing
def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Upscale small crops and apply CLAHE contrast enhancement."""
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    if min(h, w) < 80:
        scale = 80 / min(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # CLAHE on the luminance channel to improve contrast
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# Detection and OCR
def detect_signs(image_path: str) -> list[np.ndarray]:
    """Run YOLO on the image and return cropped sign regions."""
    model = _get_yolo()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    crops = []
    for result in model(image_path, verbose=False):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    print(f"  YOLO detected {len(crops)} sign region(s)")
    return crops


def run_ocr(crop: np.ndarray) -> list[dict]:
    """Run EasyOCR on a preprocessed sign crop. Returns list of {text, conf, box}."""
    preprocessed = _preprocess_crop(crop)
    reader = _get_ocr()
    hits = []
    for box, text, conf in reader.readtext(preprocessed):
        text = text.strip()
        if text and conf >= MIN_OCR_CONFIDENCE:
            hits.append({"text": text, "conf": conf, "box": box})
    return hits


# Fuzzy matching
def fuzzy_correct_city(text: str) -> str:
    """
    Map OCR text to the nearest known Estonian place name using fuzzy matching.
    Uses partial_ratio for substring tolerance (e.g. "Junja" -> "Luunja"),
    with a secondary full-string check to reject false substring matches
    (e.g. "Jagala" should NOT match "Valga" despite sharing "alga").
    """
    if not CITY_RE.match(text):
        return text

    match, score, _ = process.extractOne(
        text, ESTONIAN_PLACE_NAMES, scorer=fuzz.partial_ratio
    )
    if score < FUZZY_MATCH_THRESHOLD:
        return text
    if fuzz.ratio(text, match) < 60:
        return text
    return match


# Bounding box helpers
def _box_center_y(box) -> float:
    ys = [pt[1] for pt in box]
    return (min(ys) + max(ys)) / 2.0

def _box_center_x(box) -> float:
    xs = [pt[0] for pt in box]
    return (min(xs) + max(xs)) / 2.0

def _box_height(box) -> float:
    ys = [pt[1] for pt in box]
    return max(ys) - min(ys)


# City–distance parsing
def _collect_route_numbers(hits: list[dict]) -> set[int]:
    """
    Identify European route codes (E263, E265, etc.) so they are not
    mistaken for distances. Handles cases where "E" and the number are
    split across separate OCR tokens.
    """
    route_numbers = set()

    # Scan individual hits and joined text for E-route patterns
    joined = " ".join(h['text'] for h in hits)
    for text in [joined] + [h['text'] for h in hits]:
        for m in E_ROUTE_RE.finditer(text):
            route_numbers.add(int(m.group(1)))

    # If a standalone "E" token exists, treat all 2-3 digit numbers as routes
    if any(h['text'].strip() in ('E', 'e') for h in hits):
        for h in hits:
            if re.match(r'^\d{2,3}$', h['text'].strip()):
                route_numbers.add(int(h['text'].strip()))

    return route_numbers


def parse_city_distances(hits: list[dict]) -> list[dict]:
    """
    Extract {city, distance_km, ocr_confidence} pairs from OCR hits.
    Uses three strategies:
      1. Inline: "TARTU 82" on a single OCR line
      2. Spatial: pair city and number tokens by vertical alignment
      3. Sequential: pair in reading order as fallback
    """
    route_numbers = _collect_route_numbers(hits)

    def valid_distance(n):
        return MIN_DISTANCE_KM <= n <= MAX_DISTANCE_KM and n not in route_numbers

    parsed = []
    used = set()

    # Pass 1: inline matches (city + number on same line)
    for i, hit in enumerate(hits):
        for m in INLINE_RE.finditer(hit['text']):
            dist = int(m.group(2))
            if not valid_distance(dist):
                continue
            city = fuzzy_correct_city(m.group(1).strip().title())
            if city not in ESTONIAN_PLACE_NAMES:
                continue
            parsed.append({'city': city, 'distance_km': dist,
                           'ocr_confidence': round(hit['conf'], 3)})
            used.add(i)

    remaining = [(i, h) for i, h in enumerate(hits) if i not in used]

    # Pass 2: spatial pairing (same row, number to the right of city)
    cities  = [(i, h) for i, h in remaining if CITY_RE.match(h['text'])]
    numbers = [(i, h) for i, h in remaining if NUMBER_RE.match(h['text'])]

    if cities and numbers:
        paired_nums = set()
        for ci, city_hit in cities:
            cy = _box_center_y(city_hit['box'])
            ch = _box_height(city_hit['box'])
            cx = _box_center_x(city_hit['box'])

            best_ni, best_num, best_dist = None, None, float('inf')
            for ni, num_hit in numbers:
                if ni in paired_nums:
                    continue
                if _box_center_x(num_hit['box']) <= cx:
                    continue  # number must be to the right
                diff = abs(cy - _box_center_y(num_hit['box']))
                if diff < ch * ROW_TOLERANCE and diff < best_dist:
                    best_ni, best_num, best_dist = ni, num_hit, diff

            if best_num:
                dist = int(best_num['text'])
                if not valid_distance(dist):
                    continue
                city = fuzzy_correct_city(city_hit['text'].strip().title())
                if city not in ESTONIAN_PLACE_NAMES:
                    continue
                paired_nums.add(best_ni)
                used.update([ci, best_ni])
                conf = round((city_hit['conf'] + best_num['conf']) / 2, 3)
                parsed.append({'city': city, 'distance_km': dist,
                               'ocr_confidence': conf})

    # Pass 3: sequential fallback (pair in reading order)
    remaining = [(i, h) for i, h in enumerate(hits) if i not in used]
    city_tokens  = [(i, h) for i, h in remaining if CITY_RE.match(h['text'])]
    number_tokens = [(i, h) for i, h in remaining if NUMBER_RE.match(h['text'])]

    for (ci, city_hit), (ni, num_hit) in zip(city_tokens, number_tokens):
        if ci >= ni:
            continue  # city must come before number
        dist = int(num_hit['text'])
        if not valid_distance(dist):
            continue
        city = fuzzy_correct_city(city_hit['text'].strip().title())
        if city not in ESTONIAN_PLACE_NAMES:
            continue
        conf = round((city_hit['conf'] + num_hit['conf']) / 2, 3)
        parsed.append({'city': city, 'distance_km': dist,
                       'ocr_confidence': conf})

    return parsed


# Public entry point
def process_image(image_path: str) -> list[dict]:
    """
    Full pipeline: detect signs with YOLO, run OCR, parse city–distance pairs.
    Returns list of {city, distance_km, ocr_confidence}.
    """
    print(f"\n[OCR] Processing {Path(image_path).name} ...")
    crops = detect_signs(image_path)

    if not crops:
        print("  No signs detected.")
        return []

    all_destinations = []
    for i, crop in enumerate(crops):
        hits = run_ocr(crop)
        print(f"  Sign #{i+1} raw hits: {[(h['text'], round(h['conf'], 2)) for h in hits]}")
        parsed = parse_city_distances(hits)
        print(f"  Sign #{i+1}: {[(p['city'], p['distance_km']) for p in parsed]}")
        all_destinations.extend(parsed)

    return all_destinations
