import json
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
import pycountry
import langcodes

from fuzzywuzzy import process
from typing import List, TypedDict, Optional
import country_converter as coco
from transformers import pipeline
import os
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict, Dict, Any
from pydantic import BaseModel
from typing import List, Dict, Any

# from langgraph.graph.schema import transform_schema
# state_schema = transform_schema(FlashpointState)


ALL_FEED_URLS = []


class FeedEntry(BaseModel):
    url: Optional[str] = None
    title: Optional[str] = None
    seendate: Optional[str] = None
    domain: Optional[Dict[str, str]] = {}
    language: Optional[str] = None
    sourcecountry: Optional[str] = None
    description: Optional[str] = None


class QueryState(BaseModel):
    query: str
    query_translated: list[dict[str, str]]
    entities: List[str]
    language: List[str]
    country_language: Optional[Dict[str, List[str]]] = {}
    rss_urls: Optional[List[str]] = []
    feed_entries: Optional[List[FeedEntry]] = []


class FlashpointState(BaseModel):
    flashpoint: Dict[str, Any]
    all_queries: Optional[List[QueryState]] = []
    domains: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    error: Optional[str] = None  # <- This flag will break the chain


# Single instance for reuse
llm_mistral = ChatOpenAI(
    model_name="mistral-small",  # or mistral-medium / mistral-large
    openai_api_base="https://api.mistral.ai/v1",
    openai_api_key=os.getenv("MISTRAL_API_KEY"),
    temperature=0,
)

import time
import random


def call_mistral_chat(
    prompt: str, system_prompt: str = "You are a helpful assistant", retries=3, delay=2
) -> str:
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt),
    ]
    for attempt in range(retries):
        try:
            response = llm_mistral.invoke(messages)
            return response.content
        except Exception as e:
            wait_time = delay * (2**attempt) + random.uniform(0.1, 0.5)
            print(
                f"[Rate Limited] Retry {attempt+1}/{retries} after {wait_time:.2f}s..."
            )
            time.sleep(wait_time)
    return ""


# ---------- INPUT ----------
with open("result.json", "r") as file:
    input_data = json.load(file)

# Load NER pipeline once
ner = pipeline(
    "ner", model="Babelscape/wikineural-multilingual-ner", aggregation_strategy="simple"
)


# ---------- DOMAIN INFERENCE ----------
def infer_domains(title: str, description: str) -> List[str]:
    prompt = f"""
Given the following flashpoint, identify which of the following high-level categories it falls under. Return a comma-separated list.

Categories:
- Geopolitical
- Military / Conflict / Strategic Alliances
- Economic / Trade / Sanctions
- Cultural / Identity Clashes
- Religious Tensions / Ideological Movements
- Technological Competition
- Cybersecurity Threats / Disinformation Ops
- Environmental Flashpoints / Resource Crises
- Civilizational / Ethnonationalist Narratives
- AI Governance / Tech Ethics Conflicts
- Migration / Demographic Pressures
- Sovereignty / Border / Legal Disputes

Flashpoint:
Title: {title}
Description: {description}

Respond with comma-separated categories only.
"""
    response = call_mistral_chat(
        prompt, system_prompt="You are a geopolitical taxonomy expert."
    )
    return [d.strip() for d in response.split(",") if d.strip()]


# ---------- QUERY EXPANSION ----------
def build_query_expansion_prompt(title, description, entities, domains):
    return f"""You are a geopolitical news analyst. Your task is to generate 50 natural-language search queries that people might use to find news related to a global flashpoint.

Use diverse perspectives: military, economic, cultural, religious, tech, environmental, migration, ideological, legal, civilizational. Include synonyms, abbreviations, slang, location names, and domain-specific terms.

Each query should be short (max 6 words), realistic, and phrased the way a journalist, civilian, policymaker, activist, or intelligence analyst might search.

Respond with a JSON array of strings.

Flashpoint:
Title: {title}
Description: {description}
Entities: {entities}
Domains: {domains}
"""


def check_for_errors_node(state: FlashpointState) -> str:
    if state.error:
        print(f"Aborting chain due to error: {state.error}")
        return END  # Abort chain early
    return "detect_entities_sub_queries"


from pydantic import ValidationError


def safe_flatten_queries(queries):
    flat_queries = []
    for q in queries:
        if isinstance(q, str):
            flat_queries.append(q.strip())
        elif isinstance(q, list) and q and isinstance(q[0], str):
            flat_queries.append(q[0].strip())
    return flat_queries


def expand_queries_node(state: FlashpointState):
    flashpoint = state.flashpoint
    domains = infer_domains(flashpoint["title"], flashpoint["description"])
    prompt = build_query_expansion_prompt(
        flashpoint["title"],
        flashpoint["description"],
        flashpoint["entities"],
        ", ".join(domains),
    )

    try:
        queries = json.loads(
            call_mistral_chat(
                prompt, system_prompt="You are a geopolitical news analyst."
            )
        )
    except Exception as e:
        print(f"Failed to parse queries: {e}")
        queries = []

    valid_queries = []

    for q in safe_flatten_queries(queries):
        try:
            query_state = QueryState(
                query=q,
                language=["en"],
                query_translated=[{"language": "en", "query": q}],
                entities=[],
            )
            valid_queries.append(query_state)
        except ValidationError as ve:
            error = f"Validation error for query '{q}': {ve}"
            print(error)
            state.error = error
            return state

    state.all_queries = valid_queries

    state.domains = domains
    return state


import spacy

import importlib
import subprocess
import sys


def ensure_spacy_model(model_name: str):
    """
    Ensure a spaCy model is installed and return the loaded nlp object.

    Args:
        model_name: e.g. "en_core_web_sm"

    Returns:
        The loaded spaCy Language pipeline.
    """
    try:
        # Try to import the model package
        importlib.import_module(model_name)
    except ImportError:
        # If that fails, download it via the spacy CLI
        print(f"Model {model_name} not found. Downloading...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name], check=True
        )
    # Finally, load and return the pipeline
    import spacy

    return spacy.load(model_name)


nlp = ensure_spacy_model("en_core_web_sm")


def extract_countries(text: str):
    doc = nlp(text)
    countries = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ == "GPE":
            country = detect_country_entity(ent.text)
            if country and country.alpha_2 not in seen:
                seen.add(country.alpha_2)
                countries.append(country)
    return countries


def detect_entities_sub_queries(state: FlashpointState):
    queries = state.all_queries
    for query in queries:
        countries = extract_countries(query.query)
        for country in countries:
            query.entities.append(country.alpha_2)
    return state


# ---------- LANGUAGE RESOLUTION ----------
def resolve_languages_node(state: FlashpointState):
    queries = state.all_queries

    for query in queries:
        entities = query.entities
        for entity in entities:
            country = detect_country_entity(entity)
            if country:
                official_langs = get_official_languages(country.alpha_2)
                query.language.extend(official_langs)
                query.language = list(set(query.language))
                query.country_language[country.alpha_2] = official_langs

    return state


# detect_country_entity.py

import re
from functools import lru_cache
from typing import Optional

from rapidfuzz import process, fuzz

# Constants
_MIN_FUZZY_LEN = 3
_SCORE_CUTOFF = 75
_ACRONYM_RE = re.compile(r"^[A-Z]{2,}$")


@lru_cache(maxsize=128)
def detect_country_entity(name: str) -> Optional[pycountry.db.Country]:
    """
    Resolve an input string to a pycountry Country, or None if no valid match.
    1) Exact alias/code via country_converter
    2) Pycountry lookup on converted name
    3) Reject pure acronyms (e.g. NATO)
    4) Fuzzy match on common names with RapidFuzz
    """
    key = name.strip()
    if not key:
        return None

    # 1) country_converter handles lots of aliases and codes
    converted = coco.convert(names=key, to="name_short", not_found=None)
    if converted and converted.lower() != "not found":
        try:
            return pycountry.countries.lookup(converted)
        except LookupError:
            pass

    # 2) Reject acronyms to avoid spurious hits
    if _ACRONYM_RE.fullmatch(key):
        return None

    # 3) Controlled fuzzy match on official country names
    if len(key) >= _MIN_FUZZY_LEN:
        choices = {c.name: c for c in pycountry.countries}
        match = process.extractOne(
            key, choices.keys(), scorer=fuzz.WRatio, score_cutoff=_SCORE_CUTOFF
        )
        if match:
            matched_name, score, _ = match
            return choices[matched_name]

    return None


# official_languages.py

import re
from functools import lru_cache
from typing import List
import requests

# Constants
_ALPHA2_PATTERN = re.compile(r"^[A-Za-z]{2}$")
_RESTCOUNTRIES_URL = "https://restcountries.com/v3.1/alpha/{code}"
_REQUEST_TIMEOUT = 2  # seconds


class LanguageServiceError(Exception):
    """Custom exception for language service failures."""


@lru_cache(maxsize=256)
def get_official_languages(country_code: str) -> List[str]:
    """
    Fetches the list of official languages for a given ISO-3166 alpha-2 country code.

    Args:
        country_code: Two-letter country code (e.g., "US", "DE", "BR").

    Returns:
        A sorted list of ISO-639-1 language codes.
        Defaults to ["en"] if detection fails or no official languages are listed.

    Raises:
        LanguageServiceError: On invalid input or upstream/service errors.
    """
    code = country_code.strip().upper()
    if not _ALPHA2_PATTERN.match(code):
        raise LanguageServiceError(f"Invalid country code: '{country_code}'")

    try:
        resp = requests.get(
            _RESTCOUNTRIES_URL.format(code=code),
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "MASX-AI/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        # Data is a list of country objects; take the first
        languages = data[0].get("languages", {})
        # languages: { "eng": "English", "fra": "French", ... }
        iso6391 = []
        for lang_tag in languages.keys():
            # convert 3-letter to 2-letter if possible
            if len(lang_tag) == 3:
                # ISO-639-1 mapping (could use pycountry here)
                try:
                    import pycountry

                    lang = pycountry.languages.get(alpha_3=lang_tag)
                    if lang and hasattr(lang, "alpha_2"):
                        iso6391.append(lang.alpha_2)
                except Exception:
                    continue
        iso6391 = sorted(set(iso6391))
        return iso6391 or ["en"]
    except requests.RequestException as re:
        raise LanguageServiceError(f"Network error for {code}: {re}")
    except (ValueError, KeyError, IndexError) as e:
        raise LanguageServiceError(f"Unexpected response structure for {code}: {e}")


# ---------- NER-AUGMENTED TRANSLATION ----------
from deep_translator import GoogleTranslator


def mistral_translate(original_query: str, target_language: str) -> str:
    # natural, context-aware translations for news search queries.
    prompt = f"""Translate the following search query naturally for a news search in {target_language}:
 "{original_query}"
 Respond ONLY in the following JSON format:
 {{"translation": "..."}}"""

    response = call_mistral_chat(
        prompt,
        system_prompt="You are a news translator. Only return the translated query. No explanation, no commentary.",
    )

    non_latin_iso_639_1_codes = {
        "he",
        "hi",
        "ar",
        "ur",
        "fa",
        "ps",
        "pa",
        "ta",
        "te",
        "kn",
        "ml",
        "bn",
        "gu",
        "mr",
        "as",
        "or",
        "ne",
        "si",
        "my",
        "zh",
        "he",
        "th",
        "km",
        "lo",
        "bo",
    }
    #  if target_language in non_latin_iso_639_1_codes:
    #      chk = "non-latin"

    non_latin = target_language in non_latin_iso_639_1_codes

    try:
        result = json.loads(response.strip())["translation"]
        if non_latin:
            result = google_translate(result, target_language)
        return result

    except Exception as e:
        return google_translate(original_query, target_language)


def google_translate(original_query: str, target_language: str) -> str:
    try:
        return GoogleTranslator(source="auto", target=target_language).translate(
            original_query
        )
    except Exception as ge:
        print("Google Translate also failed:", ge)
        return original_query


def named_entities(text):
    return {ent["word"] for ent in ner(text)}


def validate_entity_preservation(original_query, translated_query):
    original_ents = named_entities(original_query)
    translated_ents = named_entities(translated_query)
    missing = original_ents - translated_ents
    return {
        "original_entities": list(original_ents),
        "translated_entities": list(translated_ents),
        "missing_entities": list(missing),
    }


def translate_queries_node(state: FlashpointState):
    queries = state.all_queries
    for query in queries:
        query.query_translated = []
        for lang in query.language:
            if lang == "en" or lang == "uk":
                query.query_translated.append({"language": lang, "query": query.query})
            else:
                query.query_translated.append(
                    {"language": lang, "query": mistral_translate(query.query, lang)}
                )

    return state


# ---------- BUILD GOOGLE RSS URLS ----------
def build_rss_urls_node(state: FlashpointState):
    def build_url(query, lang, country_code):
        query_encoded = "+".join(query.split())
        ceid = f"{country_code.upper()}:{lang.upper()}"
        return f"https://news.google.com/rss/search?q={query_encoded}&hl={lang.lower()}&ceid={ceid}"

    rss_urls = []
    queries = state.all_queries
    for query in queries:
        for query_translated in query.query_translated:
            # query.country_language
            lang = query_translated["language"]
            # find country of the language
            country = next(
                (
                    country
                    for country, langs in query.country_language.items()
                    if lang in langs
                ),
                None,
            )

            if country:
                entity = country
            elif len(query.entities) > 0:
                entity = query.entities[0]
            else:
                entity = "US"

            url = build_url(
                query_translated["query"], query_translated["language"], entity
            )
            rss_urls.append(url)
        query.rss_urls = list(set(rss_urls))

    # state.rss_urls = rss_urls
    return state


import feedparser
from concurrent.futures import ThreadPoolExecutor


def check_rss_feeds(state: FlashpointState, min_results: int = 2) -> FlashpointState:
    state = process_queries(state, min_results)
    return state

    # def check_feed(url: str) -> tuple:
    #     try:
    #         feed = feedparser.parse(url)
    #         return url, len(feed.entries) >= min_results
    #     except Exception as e:
    #         print(f"Feed check failed for {url}: {e}")
    #         return url, False

    # for query in state.all_queries:
    #     query.rss_urls
    #     #ThreadPoolExecutor for each query.rss_urls seperately

    # with ThreadPoolExecutor(max_workers=10) as executor:
    #     results = list(executor.map(check_feed, state.rss_urls))

    # # Filter only valid URLs
    # state.rss_urls = [url for url, is_valid in results if is_valid]

    # return state


from datetime import datetime, timedelta
from dateutil import parser


def date_parser(date_str):
    try:
        dt = parser.parse(date_str)
        return dt
    except Exception as e:
        return None


def fetch_rss(url, min_results):
    try:
        feed = feedparser.parse(url)
        if len(feed.entries) >= min_results:
            return url, feed.entries
        else:
            return url, None

    except Exception as e:
        return url, None


def process_queries(state, min_results: int = 2):

    if os.cpu_count() >= 8:
        max_worker = 10
    elif os.cpu_count() >= 16:
        max_worker = 20
    else:
        max_worker = 3

    for query in state.all_queries:
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            raw_results = list(
                executor.map(lambda url: fetch_rss(url, min_results), query.rss_urls)
            )

            # Extract only non-empty feed entries
            feeds = [(url, entries) for url, entries in raw_results if entries]

            feed_entries = []
            # process feed entries
            for url, entries in feeds:
                for entry in entries:

                    if entry.published:
                        seendate = date_parser(entry.published)
                        if seendate < datetime.now(seendate.tzinfo) - timedelta(days=1):
                            # Entry is not within the last 24 hours
                            continue

                    feed = FeedEntry()

                    feed.url = entry.link
                    feed.title = entry.title
                    feed.seendate = entry.published
                    feed.domain["title"] = entry.source.title
                    feed.domain["href"] = entry.source.href
                    feed.description = ""
                    feed_entries.append(feed)
                    ALL_FEED_URLS.append(entry.link)

            query.feed_entries = feed_entries
            print(f"Feed entries for query: {feed_entries}")
    return state


# ---------- DEFINE LANGGRAPH PIPELINE ----------


# state_schema = transform_schema(FlashpointState)
workflow = StateGraph(FlashpointState)
workflow.add_node("expand_queries", RunnableLambda(expand_queries_node))
workflow.add_node(
    "detect_entities_sub_queries", RunnableLambda(detect_entities_sub_queries)
)
workflow.add_node("resolve_languages", RunnableLambda(resolve_languages_node))
workflow.add_node("translate_queries", RunnableLambda(translate_queries_node))
workflow.add_node("build_rss_urls", RunnableLambda(build_rss_urls_node))
workflow.add_node("check_rss_feeds", RunnableLambda(check_rss_feeds))
workflow.set_entry_point("expand_queries")
workflow.add_edge("expand_queries", "detect_entities_sub_queries")
workflow.add_edge("detect_entities_sub_queries", "resolve_languages")
workflow.add_edge("resolve_languages", "translate_queries")
workflow.add_edge("translate_queries", "build_rss_urls")
workflow.add_edge("build_rss_urls", "check_rss_feeds")
workflow.add_edge("check_rss_feeds", END)

pipeline = workflow.compile()

# -----+---- EXECUTION ----------
if __name__ == "__main__":

    all_results = []
    for item in input_data:
        print(f"▶ Processing: {item['title']}")
        result: FlashpointState = pipeline.invoke({"flashpoint": item})
        all_results.append(result)
        # print(json.dumps(result["rss_urls"], indent=2, ensure_ascii=False))
        print("Done\n")

    # write all_results to a json file

    # save the result to a json file
    with open("result.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
