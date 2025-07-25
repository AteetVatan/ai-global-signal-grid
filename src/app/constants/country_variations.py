# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

ONE_WAY_COUNTRY_VARIATIONS = {
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "america": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "england": "United Kingdom",
    "scotland": "United Kingdom",
    "wales": "United Kingdom",
    "northern ireland": "United Kingdom",
    "britain": "United Kingdom",
    "russia": "Russian Federation",
    "russian federation": "Russian Federation",
    "persia": "Iran",
    "syria": "Syrian Arab Republic",
    "venezuela": "Venezuela, Bolivarian Republic of",
    "bolivia": "Bolivia, Plurinational State of",
    "tanzania": "Tanzania, United Republic of",
    "south korea": "Korea, Republic of",
    "republic of korea": "Korea, Republic of",
    "north korea": "Korea, Democratic People's Republic of",
    "dprk": "Korea, Democratic People's Republic of",
    "roc": "Korea, Republic of",
    "taiwan": "Taiwan, Province of China",
    "republic of china": "Taiwan, Province of China",
    "palestine": "Palestine, State of",
    "occupied palestinian territories": "Palestine, State of",
    "west bank": "Palestine, State of",
    "gaza": "Palestine, State of",
    "burma": "Myanmar",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "czech": "Czechia",
    "Czechia": "czech",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "swaziland": "Eswatini",
    "macedonia": "North Macedonia",
    "laos": "Lao People's Democratic Republic",
    "moldova": "Moldova, Republic of",
    "brunei": "Brunei Darussalam",
    "cape verde": "Cabo Verde",
    "vatican": "Holy See",
    "east timor": "Timor-Leste",
    "democratic republic of congo": "Congo, The Democratic Republic of the",
    "republic of congo": "Congo",
    "congo-brazzaville": "Congo",
    "congo-kinshasa": "Congo, The Democratic Republic of the",
    "china": "China",
    "people's republic of china": "China",
    "prc": "China",
    "bharat": "India",
    "hindustan": "India",
    "hind": "India",
    "turkey": "Türkiye",
    "saudi arabia": "Saudi Arabia",
    "uae": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "holland": "Netherlands",
    "slovak republic": "Slovakia",
}

def build_bidirectional_map(original_map: dict) -> dict:
    bidirectional = {}

    for k, v in original_map.items():
        k_l = k.lower().strip()
        v_l = v.lower().strip()

        # Primary direction
        if k_l not in bidirectional:
            bidirectional[k_l] = v_l

        # Safe reverse direction (don't overwrite canonical names)
        if v_l not in bidirectional:
            bidirectional[v_l] = k_l

    return bidirectional

COUNTRY_VARIATIONS = build_bidirectional_map(ONE_WAY_COUNTRY_VARIATIONS)
