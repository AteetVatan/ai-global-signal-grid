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

        # Step 1: Exact match from pycountry whitelist
        if cleaned in self.valid_names:
            try:
                return pycountry.countries.lookup(cleaned).name
            except LookupError:
                pass

        # Step 2: Coco conversion
        coco_converted = self.get_coco_country_name(cleaned)
        if coco_converted:
            return coco_converted

        # Step 3: Fuzzy match
        try:
            match = pycountry.countries.search_fuzzy(name)
            if match:
                return match[0].name
        except LookupError:
            pass

        # step 4 :
        normalized = name.lower().strip()
        if normalized in COUNTRY_VARIATIONS:
            try:
                pycountry.countries.search_fuzzy(COUNTRY_VARIATIONS[normalized])
                return COUNTRY_VARIATIONS[normalized]
            except LookupError:
                pass

        
        #step 5: try to find the country in the pycountry

        return None
    
    def country_name_to_alpha2(self, country_name: str) -> str:
        """
        Convert a country name to its ISO 3166-1 alpha-2 code.

        Args:
            name (str): Country name (e.g., 'India', 'United States')

        Returns:
            str: Alpha-2 code (e.g., 'IN', 'US') or empty string if not found.
        """
        try:
            country = pycountry.countries.get(name=country_name)
            if country:
                return country.alpha_2
            # Fallback: fuzzy search
            matches = pycountry.countries.search_fuzzy(country_name)
            return matches[0].alpha_2 if matches else None
        except LookupError:
            return ""
    
    def get_coco_country_name(self, name: str, return_all: bool = False) -> Optional[str]:
        """Get the country name from the coco converter."""
        coco_converted = self.cc.convert(name, to="name_short", not_found=None)
        if (
            coco_converted
            and isinstance(coco_converted, str)
            and coco_converted.lower() != "not found"
        ):
            if return_all:
                return coco_converted
            elif coco_converted.lower() in self.valid_names:
                return coco_converted            
          
        return None
    

    def is_country(self, name: str) -> bool:
        """Return True if the name can be normalized to a known country."""
        return self.normalize(name) is not None

    def get_country_name(self, name: str) -> Optional[str]:
        """Return the normalized country name if it exists, otherwise return None."""
        return self.normalize(name)

    def get_country_names(self, names: List[str]) -> List[str]:
        """Return a list of normalized country names if they exist, otherwise return None."""
        return [
            self.normalize(name) for name in names if self.normalize(name) is not None
        ]
