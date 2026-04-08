"""
place_names.py — List of Estonian settlement names for OCR fuzzy matching.

Used by ocr.py to correct misread sign text to the nearest known place name.
"""

ESTONIAN_PLACE_NAMES = list(dict.fromkeys([
    # Major cities
    "Tallinn", "Tartu", "Narva", "Pärnu", "Kohtla-Järve", "Viljandi",
    "Rakvere", "Maardu", "Sillamäe", "Kuressaare",

    # Towns
    "Võru", "Valga", "Jõhvi", "Haapsalu", "Keila", "Paide", "Tapa",
    "Põlva", "Jõgeva", "Saue", "Märjamaa", "Türi", "Elva", "Rapla",
    "Kärdla", "Kiviõli", "Mustvee", "Kallaste", "Põltsamaa", "Kunda",
    "Paldiski", "Luunja", "Otepää", "Antsla", "Kanepi", "Värska",
    "Tõrva", "Võhma", "Sindi", "Kilingi-Nõmme", "Karksi-Nuia",
    "Abja-Paluoja", "Suure-Jaani", "Püssi",

    # Tallinn–Narva corridor (E20)
    "Aegviidu", "Aravete", "Ambla", "Järva-Jaani", "Mäo",
    "Kadrina", "Haljala", "Vinni", "Iisaku", "Aseri",

    # Tallinn–Tartu corridor
    "Ardu", "Arisvere", "Imavere", "Laeva",
    "Tabivere", "Ülenurme", "Nõo", "Ropka",

    # Tallinn–Pärnu corridor
    "Laagri", "Saku", "Kohila", "Kehtna", "Lihula",
    "Tori", "Pärnu-Jaagupi", "Are", "Vändra",

    # Around Tallinn
    "Loksa", "Kuusalu", "Jüri", "Rae", "Harkujärve",
    "Tabasalu", "Viimsi", "Pirita",

    # West Estonia and islands
    "Taebla", "Virtsu", "Emmaste", "Kõrgessaare",
    "Orissaare", "Leisi", "Mustjala", "Muhu",

    # South Estonia
    "Võhandu", "Rõuge", "Mõniste", "Luhamaa",
    "Taheva", "Sangaste", "Helme",
    "Räpina", "Võnnu", "Puka", "Rõngu",

    # Central Estonia
    "Albu", "Kõpu", "Halliste", "Palamuse",

    # East Estonia
    "Avinurme", "Alajõe",

    # North coast
    "Vihula", "Käsmu", "Võsu",

    # Pärnu region
    "Mõisaküla", "Lavassaare",
]))
