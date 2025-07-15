import pycountry


class LanguageUtils:
    """Utility class for validating ISO 639-1 language codes."""

    @staticmethod
    def is_valid_iso639_1(code: str) -> bool:
        try:
            return pycountry.languages.get(alpha_2=code.lower()) is not None
        except Exception:
            return False
