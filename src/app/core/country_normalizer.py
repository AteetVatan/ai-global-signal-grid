from typing import Optional, List
import pycountry
from country_converter import CountryConverter
from ..constants import COUNTRY_VARIATIONS


class CountryNormalizer:
    def __init__(self):
        self.cc = CountryConverter()
        self.valid_names = self._build_country_name_set()

    def _build_country_name_set(self) -> set[str]:
        """Build lowercase whitelist of all valid country names from pycountry."""
        names = set()
        for country in pycountry.countries:
            names.add(country.name.lower())
            if hasattr(country, "official_name"):
                names.add(country.official_name.lower())
            if hasattr(country, "common_name"):
                names.add(country.common_name.lower())
        return names

    def normalize(self, name: str) -> Optional[str]:
        """Normalize and validate a country name deterministically (temp 0)."""
        cleaned = name.strip().lower()

        #Step 1: Exact match from pycountry whitelist
        if cleaned in self.valid_names:
            try:
                return pycountry.countries.lookup(cleaned).name
            except LookupError:
                pass

        #Step 2: Coco conversion
        converted = self.cc.convert(cleaned, to='name_short', not_found=None)
        if converted and isinstance(converted, str) and converted.lower() != "not found":
              if converted.lower() in self.valid_names:
                return converted

        #Step 3: Fuzzy match
        try:
            match = pycountry.countries.search_fuzzy(name)
            if match:
                return match[0].name
        except LookupError:
            pass
        
        #step 4 :
        normalized = name.lower().strip()             
        if normalized in COUNTRY_VARIATIONS:
            try:
                pycountry.countries.search_fuzzy(COUNTRY_VARIATIONS[normalized])
                return COUNTRY_VARIATIONS[normalized]
            except LookupError:
                pass
            

        return None

    def is_country(self, name: str) -> bool:
        """Return True if the name can be normalized to a known country."""
        return self.normalize(name) is not None
    
    def get_country_name(self, name: str) -> Optional[str]:
        """Return the normalized country name if it exists, otherwise return None."""
        return self.normalize(name)
    
    def get_country_names(self, names: List[str]) -> List[str]:
        """Return a list of normalized country names if they exist, otherwise return None."""
        return [self.normalize(name) for name in names if self.normalize(name) is not None]