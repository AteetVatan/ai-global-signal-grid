import os
import feedparser
from datetime import datetime, timedelta
from dateutil import parser as dtparser
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

from ..core.querystate import QueryState, FeedEntry
from ..config.logging_config import get_logger

logger = get_logger("FeedParserService")


class FeedParserService:
    def __init__(self, min_results: int = 2):
        self.min_results = min_results
        self.recent_days = 1
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

    def _process_feed_entries(self, entries: List[dict]) -> List[FeedEntry]:
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
                seendate=entry.get("published", ""),
                domain={
                    "title": entry.get("source", {}).get("title", ""),
                    "href": entry.get("source", {}).get("href", ""),
                },
                description=entry.get("summary", ""),
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

        # Fetch all feeds in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results: List[Tuple[str, List[dict]]] = list(executor.map(self._fetch_rss, query_state.rss_urls))

        feed_entries = []
        for url, entries in results:
            if entries:
                processed = self._process_feed_entries(entries)
                feed_entries.extend(processed)

        query_state.google_feed_entries = feed_entries
        logger.info(f"[{query_state.query}] Parsed {len(feed_entries)} recent feed entries from {len(query_state.rss_urls)} feeds.")

        return query_state
