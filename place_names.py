"""
place_names.py — Estonian settlement names for OCR fuzzy matching.

Covers cities, towns, and villages that appear on Estonian road signs.
Used by ocr.py for fuzzy-correcting OCR output to known place names.
"""

ESTONIAN_PLACE_NAMES = [
    # ── Major cities ──────────────────────────────────────────────────────────
    "Tallinn", "Tartu", "Narva", "Pärnu", "Kohtla-Järve", "Viljandi",
    "Rakvere", "Maardu", "Sillamäe", "Kuressaare",

    # ── Towns (5 000–10 000) ──────────────────────────────────────────────────
    "Võru", "Valga", "Jõhvi", "Haapsalu", "Keila", "Paide", "Tapa",
    "Põlva", "Jõgeva", "Saue", "Märjamaa", "Türi", "Elva", "Rapla",
    "Kärdla", "Kiviõli", "Mustvee", "Kallaste", "Põltsamaa", "Kunda",
    "Paldiski", "Luunja", "Otepää", "Antsla", "Kanepi", "Värska",
    "Tõrva", "Võhma", "Sindi", "Kilingi-Nõmme", "Karksi-Nuia",
    "Abja-Paluoja", "Suure-Jaani", "Püssi",

    # ── Tallinn–Narva corridor (E20) ──────────────────────────────────────────
    "Aegviidu", "Aravete", "Ambla", "Järva-Jaani", "Mäo",
    "Kadrina", "Haljala", "Vinni", "Iisaku", "Aseri",

    # ── Tallinn–Tartu corridor ────────────────────────────────────────────────
    "Kose", "Ardu", "Arisvere", "Imavere", "Laeva",
    "Tabivere", "Ülenurme", "Nõo", "Ropka",

    # ── Tallinn–Pärnu corridor ────────────────────────────────────────────────
    "Laagri", "Saku", "Kohila", "Kehtna", "Lihula",
    "Tori", "Pärnu-Jaagupi", "Are", "Vändra",

    # ── Around Tallinn ────────────────────────────────────────────────────────
    "Saue", "Keila", "Paldiski", "Loksa", "Kuusalu",
    "Maardu", "Jüri", "Rae", "Saku", "Harkujärve",
    "Tabasalu", "Viimsi", "Pirita",

    # ── West Estonia / islands ────────────────────────────────────────────────
    "Haapsalu", "Taebla", "Märjamaa", "Lihula", "Virtsu",
    "Kärdla", "Emmaste", "Kõrgessaare",
    "Kuressaare", "Orissaare", "Leisi", "Mustjala", "Muhu",

    # ── South Estonia ─────────────────────────────────────────────────────────
    "Võru", "Antsla", "Võhandu", "Rõuge", "Mõniste",
    "Valga", "Taheva", "Sangaste", "Tõrva", "Helme",
    "Põlva", "Räpina", "Värska", "Võnnu",
    "Otepää", "Puka", "Rõngu", "Elva", "Nõo",
    "Kanepi", "Põlva",

    # ── Central Estonia ───────────────────────────────────────────────────────
    "Paide", "Türi", "Järva-Jaani", "Ambla", "Albu",
    "Imavere", "Võhma", "Suure-Jaani", "Kõpu",
    "Viljandi", "Abja-Paluoja", "Halliste", "Karksi-Nuia",
    "Põltsamaa", "Jõgeva", "Palamuse", "Tabivere",

    # ── East Estonia ──────────────────────────────────────────────────────────
    "Jõhvi", "Kohtla-Järve", "Sillamäe", "Narva", "Kiviõli",
    "Kunda", "Rakvere", "Tapa", "Iisaku", "Avinurme",
    "Mustvee", "Kallaste", "Alajõe",

    # ── North coast ───────────────────────────────────────────────────────────
    "Loksa", "Kuusalu", "Haljala", "Vihula", "Käsmu",
    "Võsu", "Kunda",

    # ── Pärnu region ─────────────────────────────────────────────────────────
    "Pärnu", "Sindi", "Tori", "Vändra", "Are",
    "Kilingi-Nõmme", "Mõisaküla", "Lavassaare",
]

# Deduplicate while preserving order
_seen = set()
_deduped = []
for _name in ESTONIAN_PLACE_NAMES:
    if _name not in _seen:
        _seen.add(_name)
        _deduped.append(_name)
ESTONIAN_PLACE_NAMES = _deduped
