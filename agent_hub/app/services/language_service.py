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

"""
MASX AI LanguageService: Resolves official ISO-639-1 languages for geopolitical entities.

Supports:
- ISO country codes (via RESTCountries API + pycountry)
- Non-state or stateless regions (via internal override map)
"""

import re
import requests
import pycountry
from typing import List, Dict, Optional
from functools import lru_cache
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from ..constants import ISO_REGION_LANGUAGES, CountryV2Manager
from ..core.singleton import ThreadSafeRateLimiter

# Constants
_ALPHA2_PATTERN = re.compile(r"^[A-Za-z]{2}$")
_RESTCOUNTRIES_URL = "https://restcountries.com/v3.1/alpha/{code}"
_REQUEST_TIMEOUT = 2  # seconds


class LanguageServiceError(Exception):
    """Custom exception for language service failures."""

    def __init__(self, message: str):
        super().__init__(f"[LanguageServiceError] {message}")


class LanguageService:
    _language_cache: Dict[str, List[str]] = {}
    _country_v2_manager = CountryV2Manager()

    @classmethod
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def get_languages_for_country_code(cls, country_code: str) -> List[str]:
        """
        Fetches official ISO-639-1 language codes for a given country code.
        Tries both local COUNTRY_V2 manager and remote RestCountries API.

        Returns:
            List of unique ISO-639-1 codes like ['en', 'de']
        """
        code = country_code.strip().upper()
        lang_set = set()

        # Check cache
        if code in cls._language_cache:
            return cls._language_cache[code]

        # Validate
        if not _ALPHA2_PATTERN.match(code):
            raise LanguageServiceError(f"Invalid country code: '{code}'")

        # --------- Local lookup from COUNTRY_V2 ----------
        try:
            local_country = cls._country_v2_manager.get_country(code)
            if local_country:
                lang_set.update(local_country.get_language_codes_iso6391())
        except Exception as e:
            # Do not raise; just log if needed
            print(f"[Local fallback failed] {code}: {e}")

        # --------- Remote API fallback -----------
        rate_limiter = ThreadSafeRateLimiter.get_instance(max_calls_per_sec=1)
        rate_limiter.acquire()

        try:
            response = requests.get(
                _RESTCOUNTRIES_URL.format(code=code),
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": "MASX-AI/1.0"},
            )
            response.raise_for_status()
            remote_data = response.json()[0]
            lang_tags = remote_data.get("languages", {})

            for lang_tag in lang_tags.keys():
                lang = pycountry.languages.get(alpha_3=lang_tag)
                if lang and hasattr(lang, "alpha_2"):
                    lang_set.add(lang.alpha_2)

        except requests.RequestException as re:
            print(f"[Remote API Error] {code}: {re}")
        except (ValueError, KeyError, IndexError) as e:
            print(f"[Remote JSON Error] {code}: {e}")

        # --------- Final Result ---------
        result = sorted(lang_set) or ["en"]  # fallback to English if empty
        cls._language_cache[code] = result
        return result

    @classmethod
    def get_languages_for_entity(cls, entity: str) -> List[str]:
        """
        Resolves languages for an entity: uses override map or country code.

        Args:
            entity: e.g., "Iran", "Palestine", "Balochistan"

        Returns:
            List of ISO 639-1 codes
        """
        languages = []
        entity_key = entity.strip().lower()
        entity_title = entity.strip().title()

        # Check static override
        if entity_key in ISO_REGION_LANGUAGES:
            languages = ISO_REGION_LANGUAGES[entity_key]

        # Try resolving via country code
        try:
            country = pycountry.countries.search_fuzzy(entity_title)[0]
            if country and hasattr(country, "alpha_2"):
                from_country = cls.get_languages_for_country_code(country.alpha_2)
                languages.extend(from_country)
        except LookupError:
            pass

        return languages

    @classmethod
    def get_languages_for_entities(cls, entities: List[str]) -> List[str]:
        """
        Resolves all unique languages across a list of entities.

        Args:
            entities: List of geopolitical entities

        Returns:
            Sorted list of ISO 639-1 language codes
        """
        all_langs = set()
        for entity in entities:
            langs = cls.get_languages_for_entity(entity)
            all_langs.update(langs)
        return sorted(all_langs)
