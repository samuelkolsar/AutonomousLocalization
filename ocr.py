"""
ocr.py — Sign detection + OCR pipeline

Extracted from OCR_Pipeline.ipynb.
Takes an image path, returns a list of {city, distance_km, ocr_confidence} dicts.

Improvements: Estonian + English EasyOCR, crop preprocessing (upscale + CLAHE),
optional Tesseract fallback, low-confidence filter, common misreading corrections.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
from rapidfuzz import process, fuzz

# Optional Tesseract (pip install pytesseract; system: install Tesseract + est language pack)
try:
    import pytesseract
    from pytesseract import Output
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False
    pytesseract = None

_tesseract_warned = False

# ── Constants ─────────────────────────────────────────────────────────────────

YOLO_MODEL_PATH = "minu_mudel_best.pt"   # path to your trained model

# Reject OCR lines below this confidence (reduces garbage feeding the parser)
MIN_OCR_CONFIDENCE = 0.25

from place_names import ESTONIAN_PLACE_NAMES

# Fuzzy match: correct OCR output to nearest known place. 65 allows single-char errors (e.g. Junja→Luunja).
FUZZY_MATCH_THRESHOLD = 65
ROW_TOLERANCE         = 1.5

CITY_RE    = re.compile(r'^[A-ZÕÄÖÜ][a-zõäöüA-ZÕÄÖÜ\-]{2,}$')   # 3+ chars minimum
NUMBER_RE  = re.compile(r'^\d{1,4}$')
INLINE_RE  = re.compile(r'([A-ZÕÄÖÜ][a-zõäöüA-ZÕÄÖÜ\-]{2,})\s+(\d{1,4})')
# Matches European route codes like E263, E 265, e265 — these are road numbers, not distances
E_ROUTE_RE = re.compile(r'\bE\s*(\d{2,4})\b', re.IGNORECASE)

# ── Module-level singletons (loaded once) ─────────────────────────────────────

_yolo_model  = None
_ocr_reader  = None


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


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_crop_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Improve crop for OCR: upscale small crops, then CLAHE on luminance.
    Helps with small or low-contrast sign text.
    """
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    min_side = min(h, w)
    # Upscale if crop is very small (text too tiny for OCR)
    if min_side < 80:
        scale = 80 / min_side
        new_w, new_h = int(w * scale), int(h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = crop.shape[:2]
    # CLAHE on luminance (preserve color for EasyOCR, but improve contrast)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_signs(image_path: str) -> list[np.ndarray]:
    """
    Run YOLO on the image, return a list of cropped sign regions (numpy arrays).
    """
    model = _get_yolo()
    img   = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model(image_path, verbose=False)
    crops   = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

    print(f"  YOLO detected {len(crops)} sign region(s)")
    return crops


# ── OCR ───────────────────────────────────────────────────────────────────────

def _run_tesseract(crop: np.ndarray) -> list[dict]:
    """Run Tesseract on crop; return same format as EasyOCR hits. Optional fallback."""
    if not _TESSERACT_AVAILABLE:
        return []
    try:
        data = pytesseract.image_to_data(crop, lang="est+eng", output_type=Output.DICT)
    except Exception:
        return []
    hits = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = data["conf"][i]
        if conf < 0:
            continue
        conf_01 = conf / 100.0
        if conf_01 < MIN_OCR_CONFIDENCE:
            continue
        left = data["left"][i]
        top = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        box = [[left, top], [left + w, top], [left + w, top + h], [left, top + h]]
        hits.append({"text": text, "conf": conf_01, "box": np.array(box)})
    return hits


def run_ocr(crop: np.ndarray) -> list[dict]:
    """
    Run OCR on a single cropped sign: preprocess, then EasyOCR + optional Tesseract.
    Merges results; low-confidence lines are dropped.
    """
    preprocessed = _preprocess_crop_for_ocr(crop)
    reader = _get_ocr()
    results = reader.readtext(preprocessed)
    hits = []
    for (box, text, conf) in results:
        text = text.strip()
        if not text or conf < MIN_OCR_CONFIDENCE:
            continue
        hits.append({"text": text, "conf": conf, "box": box})
    # Optional: add Tesseract results (same crop, preprocessed)
    if _TESSERACT_AVAILABLE:
        tesseract_hits = _run_tesseract(preprocessed)
        hits = hits + tesseract_hits
    return hits


def fuzzy_correct_city(text: str) -> str:
    """
    Map OCR output to nearest known place name by similarity (no hardcoded misreadings).
    Uses partial_ratio so single-character errors (e.g. Junja→Luunja) can still match.
    """
    if not CITY_RE.match(text):
        return text
    match, score, _ = process.extractOne(
        text, ESTONIAN_PLACE_NAMES, scorer=fuzz.partial_ratio
    )
    return match if score >= FUZZY_MATCH_THRESHOLD else text


MIN_DISTANCE_KM = 5
MAX_DISTANCE_KM = 400

def _box_vertical_center(box) -> float:
    ys = [pt[1] for pt in box]
    return (min(ys) + max(ys)) / 2.0

def _box_horizontal_center(box) -> float:
    xs = [pt[0] for pt in box]
    return (min(xs) + max(xs)) / 2.0

def _box_height(box) -> float:
    ys = [pt[1] for pt in box]
    return max(ys) - min(ys)

def _valid_distance(distance_km: int) -> bool:
    return MIN_DISTANCE_KM <= distance_km <= MAX_DISTANCE_KM


def parse_city_distances(hits: list[dict]) -> list[dict]:
    """
    Extract city + distance pairs using three passes:
      Pass 1 — inline  : 'TARTU 82' on a single OCR line
      Pass 2 — spatial : pair city and number by vertical alignment
      Pass 3 — sequential fallback
    """
    # Collect all numbers that appear as European route codes (E263, E265, etc.)
    # across the full set of hits so they are never mistaken for distances.
    # Strategy 1: scan each hit individually.
    # Strategy 2: scan joined text to catch "E" and "263" split across separate tokens.
    # Strategy 3: if any hit is a lone "E"/"e" token, mark ALL 2-3 digit numbers as
    #             route numbers (OCR sometimes drops the E leaving a bare digit string).
    route_numbers: set[int] = set()
    joined_text = " ".join(h['text'] for h in hits)
    for m in E_ROUTE_RE.finditer(joined_text):
        route_numbers.add(int(m.group(1)))
    for hit in hits:
        for m in E_ROUTE_RE.finditer(hit['text']):
            route_numbers.add(int(m.group(1)))
    # If a standalone "E"/"e" letter token is present (route marker without number),
    # treat any 2–3 digit number in the crop as a route code, not a distance.
    has_lone_e = any(h['text'].strip() in ('E', 'e') for h in hits)
    if has_lone_e:
        for hit in hits:
            if re.match(r'^\d{2,3}$', hit['text'].strip()):
                route_numbers.add(int(hit['text'].strip()))

    def _valid_distance_not_route(n: int) -> bool:
        return _valid_distance(n) and n not in route_numbers

    parsed      = []
    used_indices = set()

    # Pass 1 — inline: city and number on the same OCR line (city always left of number)
    for i, hit in enumerate(hits):
        for m in INLINE_RE.finditer(hit['text']):
            distance_km = int(m.group(2))
            if not _valid_distance_not_route(distance_km):
                continue
            city = fuzzy_correct_city(m.group(1).strip().title())
            parsed.append({
                'city': city,
                'distance_km': distance_km,
                'ocr_confidence': round(hit['conf'], 3)
            })
            used_indices.add(i)

    remaining = [(i, h) for i, h in enumerate(hits) if i not in used_indices]

    # Pass 2 — spatial
    cities  = [(i, h) for i, h in remaining if CITY_RE.match(h['text'])]
    numbers = [(i, h) for i, h in remaining if NUMBER_RE.match(h['text'])]

    if cities and numbers:
        paired = set()
        for ci, city_hit in cities:
            cy  = _box_vertical_center(city_hit['box'])
            ch  = _box_height(city_hit['box'])
            cx  = _box_horizontal_center(city_hit['box'])
            best_dist, best_ni, best_num = float('inf'), None, None
            for ni, num_hit in numbers:
                if ni in paired:
                    continue
                # Number must be to the right of the city (city name comes first)
                if _box_horizontal_center(num_hit['box']) <= cx:
                    continue
                diff = abs(cy - _box_vertical_center(num_hit['box']))
                if diff < ch * ROW_TOLERANCE and diff < best_dist:
                    best_dist, best_ni, best_num = diff, ni, num_hit
            if best_num:
                distance_km = int(best_num['text'])
                if not _valid_distance_not_route(distance_km):
                    continue
                paired.add(best_ni)
                used_indices.update([ci, best_ni])
                city = fuzzy_correct_city(city_hit['text'].strip().title())
                parsed.append({
                    'city': city,
                    'distance_km': distance_km,
                    'ocr_confidence': round((city_hit['conf'] + best_num['conf']) / 2, 3)
                })

    # Pass 3 — sequential fallback: city token must appear before number token in
    # the hits list (EasyOCR returns hits left-to-right, so earlier index = further left)
    remaining_after = [(i, h) for i, h in enumerate(hits) if i not in used_indices]
    city_tokens     = [(i, h) for i, h in remaining_after if CITY_RE.match(h['text'])]
    number_tokens   = [(i, h) for i, h in remaining_after if NUMBER_RE.match(h['text'])]
    if city_tokens and number_tokens:
        for (ci, city_hit), (ni, num_hit) in zip(city_tokens, number_tokens):
            if ci >= ni:  # city must come before number in reading order
                continue
            distance_km = int(num_hit['text'])
            if not _valid_distance_not_route(distance_km):
                continue
            parsed.append({
                'city': fuzzy_correct_city(city_hit['text'].strip().title()),
                'distance_km': distance_km,
                'ocr_confidence': round((city_hit['conf'] + num_hit['conf']) / 2, 3)
            })

    return parsed


# ── Public entry point ────────────────────────────────────────────────────────

def process_image(image_path: str) -> list[dict]:
    """
    Full pipeline: image → YOLO detection → OCR → parsed destinations.
    Returns list of {city, distance_km, ocr_confidence}.
    """
    global _tesseract_warned
    print(f"\n[OCR] Processing {Path(image_path).name} ...")
    if not _TESSERACT_AVAILABLE and not _tesseract_warned:
        _tesseract_warned = True
        print("  (Tesseract not available — install tesseract-ocr + est language pack for optional second reader)")
    crops = detect_signs(image_path)

    if not crops:
        print("  No signs detected.")
        return []

    all_destinations = []
    for i, crop in enumerate(crops):
        hits   = run_ocr(crop)
        print(f"  Sign #{i+1} raw hits: {[(h['text'], round(h['conf'],2)) for h in hits]}")
        parsed = parse_city_distances(hits)
        print(f"  Sign #{i+1}: {[(p['city'], p['distance_km']) for p in parsed]}")
        all_destinations.extend(parsed)

    return all_destinations
