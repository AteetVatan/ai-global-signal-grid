"""
Enum for supported spaCy model names used in MASX AI.
Add more models here as needed.
"""

from enum import Enum


class SpaCyModelName(str, Enum):
    EN_CORE_WEB_SM = "en_core_web_sm"
    EN_CORE_WEB_MD = "en_core_web_md"
    EN_CORE_WEB_LG = "en_core_web_lg"
    XX_ENT_WIKI_SM = "xx_ent_wiki_sm"       # Multilingual model
    DE_CORE_NEWS_SM = "de_core_news_sm"     # German
    ES_CORE_NEWS_SM = "es_core_news_sm"     # Spanish
    FR_CORE_NEWS_SM = "fr_core_news_sm"     # French

    @classmethod
    def list_all(cls):
        """Returns all model names as a list of strings."""
        return [member.value for member in cls]
