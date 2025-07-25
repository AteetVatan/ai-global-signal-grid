# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                       │
# │  Project: MASX AI – Strategic Agentic AI System               │
# │  All rights reserved.                                         │
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

import json
from typing import Dict, List, Optional, Any


class CountryV2:
    def __init__(self, country_data: Dict[str, Any]):
        self._data = country_data

    # ---------------- Basic Properties ----------------
    @property
    def name(self) -> str:
        return self._data.get("name", "")

    @property
    def native_name(self) -> str:
        return self._data.get("nativeName", "")

    @property
    def alpha2_code(self) -> str:
        return self._data.get("alpha2Code", "")

    @property
    def alpha3_code(self) -> str:
        return self._data.get("alpha3Code", "")

    @property
    def numeric_code(self) -> str:
        return self._data.get("numericCode", "")

    @property
    def cioc(self) -> str:
        return self._data.get("cioc", "")

    @property
    def capital(self) -> str:
        return self._data.get("capital", "")

    @property
    def region(self) -> str:
        return self._data.get("region", "")

    @property
    def subregion(self) -> str:
        return self._data.get("subregion", "")

    @property
    def population(self) -> int:
        return self._data.get("population", 0)

    @property
    def area(self) -> float:
        return self._data.get("area", 0.0)

    @property
    def gini(self) -> float:
        return self._data.get("gini", 0.0)

    @property
    def demonym(self) -> str:
        return self._data.get("demonym", "")

    @property
    def flag_url(self) -> str:
        return self._data.get("flag", "")

    # ---------------- Array Fields ----------------

    @property
    def alt_spellings(self) -> List[str]:
        return self._data.get("altSpellings", [])

    @property
    def borders(self) -> List[str]:
        return self._data.get("borders", [])

    @property
    def calling_codes(self) -> List[str]:
        return self._data.get("callingCodes", [])

    @property
    def timezones(self) -> List[str]:
        return self._data.get("timezones", [])

    @property
    def top_level_domains(self) -> List[str]:
        return self._data.get("topLevelDomain", [])

    @property
    def latlng(self) -> List[float]:
        return self._data.get("latlng", [])

    # ---------------- Translations ----------------

    @property
    def translations(self) -> Dict[str, str]:
        return self._data.get("translations", {})

    def get_name_in_language(self, lang_code: str) -> Optional[str]:
        return self.translations.get(lang_code)

    # ---------------- Nested Objects ----------------

    @property
    def currencies(self) -> List[Dict[str, Any]]:
        return self._data.get("currencies", [])

    @property
    def languages(self) -> List[Dict[str, Any]]:
        return self._data.get("languages", [])

    @property
    def regional_blocs(self) -> List[Dict[str, Any]]:
        return self._data.get("regionalBlocs", [])

    # ---------------- Helper Methods ----------------

    def get_currency_names(self) -> List[str]:
        return [c.get("name") for c in self.currencies if "name" in c]

    def get_currency_codes(self) -> List[str]:
        return [c.get("code") for c in self.currencies if "code" in c]

    def get_language_names(self) -> List[str]:
        return [l.get("name") for l in self.languages if "name" in l]

    def get_language_codes_iso6391(self) -> List[str]:
        """Returns all ISO-639-1 language codes (e.g., ['ps', 'uz'])"""
        return [l.get("iso639_1") for l in self.languages if "iso639_1" in l]
    
    def get_language_codes_iso6392(self) -> List[str]:
        """Returns all ISO-639-2 language codes (e.g., ['ps', 'uz'])"""
        return [l.get("iso639_2") for l in self.languages if "iso639_2" in l]  

    def get_primary_language_code(self) -> Optional[str]:
        return self.get_language_codes()[0] if self.get_language_codes() else None

    def get_regional_bloc_acronyms(self) -> List[str]:
        return [b.get("acronym") for b in self.regional_blocs if "acronym" in b]

    def get_coordinates(self) -> Optional[tuple]:
        if len(self.latlng) == 2:
            return tuple(self.latlng)
        return None

    def to_dict(self) -> Dict:
        return self._data.copy()



class CountryV2Manager:
    _filepath = "src/app/constants/countriesV2.json"
    
    def __init__(self):
        self._filepath = self._filepath
        self._countries: Dict[str, CountryV2] = {}
        self._load_countries()

    def _load_countries(self):
        with open(self._filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            self._countries = {}

            for entry in raw_data:
                code = entry.get("alpha2Code")
                if code:
                    self._countries[code.upper()] = CountryV2(entry)

    def get_country(self, alpha2_code: str) -> Optional[CountryV2]:
        return self._countries.get(alpha2_code.upper())

    def list_all_countries(self) -> List[CountryV2]:
        return list(self._countries.values())

    def filter_by_language(self, language_code: str) -> List[CountryV2]:
        return [c for c in self._countries.values() if language_code in c.languages]

    def filter_by_region(self, region: str) -> List[CountryV2]:
        return [c for c in self._countries.values() if c.region == region]

    def reload(self):
        self._load_countries()

    
 







