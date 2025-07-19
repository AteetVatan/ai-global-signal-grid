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

from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Optional, Dict
from .language_utils import LanguageUtils

# class QueryState(BaseModel):
#     query: str
#     query_translated: list[dict[str, str]]
#     entities: List[str]
#     language: List[str]
#     country_language: Optional[Dict[str, List[str]]] = {}
#     rss_urls: Optional[List[str]] = []
#     feed_entries: Optional[List[FeedEntry]] = []


class FeedEntry(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    seendate: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    sourcecountry: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None


class QueryTranslated(BaseModel):
    language: str = (
        Field(
            ...,
            min_length=2,
            max_length=2,
            description="2-letter ISO-639-1 language code (e.g., 'en', 'de')",
        ),
    )
    query_translated: str = Field(..., min_length=1, description="Translated query")

    @field_validator("language")
    @classmethod
    def check_iso_639_1(cls, v):
        if not LanguageUtils.is_valid_iso639_1(v):
            raise ValueError(f"'{v}' is not a valid ISO 639-1 language code.")
        return v.lower()

    @field_validator("query_translated", mode="before")
    @classmethod
    def strip_and_validate_non_empty(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Must be a non-empty, non-whitespace string")
        return v.strip()


class QueryState(BaseModel):
    query: str = Field(..., min_length=1, description="Query")
    list_query_translated: Optional[List[QueryTranslated]] = Field(
        None, description="Translated query"
    )
    entities: Optional[List[str]] = Field(
        None, description="List of named entities involved"
    )
    language: Optional[List[str]] = Field(None, description="List of languages")
    entity_languages: Optional[Dict[str, List[str]]] = Field(
        None, description="List of entity languages"
    )
    rss_urls: Optional[List[str]] = Field(None, description="List of RSS URLs")
    google_feed_entries: Optional[List[FeedEntry]] = Field(
        [], description="List of google feed entries"
    )
    gdelt_feed_entries: Optional[List[FeedEntry]] = Field(
        [], description="List of gdeltfeed entries"
    )
    
