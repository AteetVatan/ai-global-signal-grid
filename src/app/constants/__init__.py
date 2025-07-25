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

from .country_variations import COUNTRY_VARIATIONS
from .country_v2 import CountryV2Manager
from .iso_language import ISO_REGION_LANGUAGES
from .iso_to_nllb import ISO_TO_NLLB_MERGED
from .google_translate_variants import GOOGLE_TRANSLATE_VARIANTS
from .domains import DOMAIN_CATEGORIES  

__all__ = [
    CountryV2Manager,
    COUNTRY_VARIATIONS,
    ISO_REGION_LANGUAGES,
    ISO_TO_NLLB_MERGED,
    GOOGLE_TRANSLATE_VARIANTS,
    DOMAIN_CATEGORIES
]
