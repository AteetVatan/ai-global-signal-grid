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

import os
import feedparser
from datetime import datetime, timedelta
from dateutil import parser as dtparser
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs

from ..core import QueryState, FeedEntry, LanguageUtils
from ..config.logging_config import get_logger
from ..core import DateUtils
from ..config.settings import get_settings

logger = get_logger("FeedParserService")


class FeedParserService:
    def __init__(self, min_results: int = 2):
        self.min_results = min_results
        self.recent_days = 1
        self.settings = get_settings()
        if self.settings.debug:
            self.recent_days = 10
        self.max_workers = self._decide_workers()

    def _decide_workers(self) -> int:
        cores = os.cpu_count()
        if cores >= 16:
            return 20
        elif cores >= 8:
            return 10
        return 3

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        try:
            return dtparser.parse(date_str)
        except Exception as e:
            logger.warning(f"Date parse failed: {e}")
            return None

    def _fetch_rss(self, url: str) -> Tuple[str, Optional[List[dict]]]:
        try:
            feed = feedparser.parse(url)
            if len(feed.entries) >= self.min_results:
                return url, feed.entries
        except Exception as e:
            logger.warning(f"Feed fetch failed: {url} — {e}")
        return url, None
    
    
    def process_gdelt_feed_entries(self, entries:  Dict[str, List[Dict]]) -> List[FeedEntry]:                
        # Convert articles to FeedEntry
        try:
            feed_entries = []
            for key, articles in entries.items():
                for article in articles:
                    seen_dt = self._parse_date(article.get("seen_date", ""))
                    if seen_dt and seen_dt < datetime.now(seen_dt.tzinfo) - timedelta(
                        days=self.recent_days
                    ):  # from last 24 hour
                        continue  # Skip old entries
                    
                    # url=article.get("url", "")
                    # title=article.get("title")
                    # seendate=DateUtils.convert_iso_to_date(article.get("seen_date", ""))
                    # image=article.get("image", "")
                    # domain=article.get("domain", "")
                    # description=article.get("title","")
                    # language=LanguageUtils.get_language_code(article.get("language", ""))
                    # sourcecountry=article.get("country", "")
                    
                    feed_entry = FeedEntry(
                        url=article.get("url", ""),
                        title=article.get("title", ""),
                        seendate=DateUtils.convert_iso_to_date(article.get("seen_date", "")),
                        domain=article.get("domain", ""),
                        description=article.get("title",""),
                        language=LanguageUtils.get_language_code(article.get("language", "")),
                        sourcecountry=article.get("country", ""),
                        image=article.get("image", "")
                    )
                    feed_entries.append(feed_entry)               
        except Exception as e:
            logger.error(f"Error processing GDELT feed entries: {e}")
            return []
                
        return feed_entries       
        

    def process_google_feed_entries(self, entries: List[dict], language_code: str) -> List[FeedEntry]:
        valid_entries = []
        for entry in entries:
            seen_dt = self._parse_date(entry.get("published", ""))

            if seen_dt and seen_dt < datetime.now(seen_dt.tzinfo) - timedelta(
                days=self.recent_days
            ):  # from last 24 hour
                continue  # Skip old entries

            feed_entry = FeedEntry(
                url=entry.get("link", ""),
                title=entry.get("title", ""),
                seendate=DateUtils.convert_rfc822_to_iso_date(entry.get("published", "")),
                domain=entry.get("domain", ""),
                description=entry.get("summary", ""),   
                language=language_code,
                sourcecountry=entry.get("country", ""),
                image=entry.get("image", "")
            )
            valid_entries.append(feed_entry)

        return valid_entries


    def run(self, query_state: QueryState) -> QueryState:
        """
        Main entry point for LangGraph node. Enriches each query with valid recent RSS entries.
        - For each RSS URL, fetch and process entries in parallel.
        """
        if not query_state.rss_urls:
            logger.warning(f"[{query_state.query}] No RSS URLs found.")
            return query_state

        #extract language code from url
        language_code = self.__extract_language_code(query_state.rss_urls[0])
        # Fetch all feeds in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results: List[Tuple[str, List[dict]]] = list(executor.map(self._fetch_rss, query_state.rss_urls))

        feed_entries = []
        for url, entries in results:
            if entries:
                processed = self.process_google_feed_entries(entries, language_code)
                feed_entries.extend(processed)

        query_state.google_feed_entries = feed_entries
        logger.info(f"[{query_state.query}] Parsed {len(feed_entries)} recent feed entries from {len(query_state.rss_urls)} feeds.")

        return query_state
    
    def __extract_language_code(self, url: str) -> str:
        parsed_url = urlparse(url)
        query = parse_qs(parsed_url.query)

        # Priority 1: hl (e.g., "en-US" → "en")
        if "hl" in query:
            return query["hl"][0].split("-")[0]

        # Priority 2: ceid (e.g., "US:en" → "en")
        if "ceid" in query and ":" in query["ceid"][0]:
            return query["ceid"][0].split(":")[1]

