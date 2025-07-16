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
        return pycountry.languages.get(name=language_name).alpha_2
