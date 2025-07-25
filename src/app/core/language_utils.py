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

import pycountry


class LanguageUtils:
    """Utility class for validating ISO 639-1 language codes."""

    @staticmethod
    def is_valid_iso639_1(code: str) -> bool:
        try:
            return pycountry.languages.get(alpha_2=code.lower()) is not None
        except Exception:
            return False

    # get language code from language name
    @staticmethod
    def get_language_code(language_name: str) -> str:
        try:
            # Exact match
            lang = pycountry.languages.get(name=language_name)
            if lang and hasattr(lang, "alpha_2"):
                return lang.alpha_2

            # Fuzzy fallback
            matches = pycountry.languages.search_fuzzy(language_name)
            if matches and hasattr(matches[0], "alpha_2"):
                return matches[0].alpha_2
        except LookupError:
            return None
        except Exception as e:
            print(f"[get_language_code] Error for '{language_name}': {e}")
            return None
