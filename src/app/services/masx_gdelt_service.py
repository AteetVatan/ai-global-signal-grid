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
import requests
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import random
import time
from ..config.settings import get_settings
from ..config.logging_config import get_logger
from ..core.utils import retry_with_backoff, safe_json_loads
from .gdeltdoc import GdeltDoc, Filters, RateLimitError
from threading import Lock


class MasxGdeltService:
    """
    Service class to interact with the MASX GDELT API.
    """

    _instance = None
    _lock: Lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MasxGdeltService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.API_KEY = self.settings.gdelt_api_key
        self.BASE_URL = self.settings.GDELT_API_URL
        # self._set_base_urls()

        self.ENDPOINT = "/api/articles"

        self.headers = {"x-api-key": self.API_KEY, "Content-Type": "application/json"}
        self.max_workers = 1  # due to the api gdelt rate limit
        # self._decide_workers()
        self.gdelt_doc = GdeltDoc()

    # def _set_base_urls(self):
    #     self.base_urls =[]
    #     self.base_urls.append(self.settings.GDELT_API_URL)
    #     self.base_urls.append(self.settings.GDELT_API_URL_1)

    # @property
    # def base_url(self):
    #     #randomly select a base url
    #     return random.choice(self.base_urls)

    def _decide_workers(self) -> int:
        cores = os.cpu_count()
        if cores >= 16:
            return 20
        elif cores >= 8:
            return 10
        return 3

    def test_connectivity(self) -> bool:
        """
        Test connectivity to the GDELT API endpoint.
        Returns True if connection is successful, False otherwise.
        """
        try:
            # Simple HEAD request to test connectivity
            response = requests.head(
                self.BASE_URL,
                headers=self.headers,
                timeout=(5, 10),  # Short timeout for connectivity test
            )
            self.logger.info(
                f"GDELT API connectivity test successful: {response.status_code}"
            )
            return True
        except requests.Timeout:
            self.logger.error("GDELT API connectivity test failed: Timeout")
            return False
        except requests.ConnectionError as e:
            self.logger.error(
                f"GDELT API connectivity test failed: Connection error - {e}"
            )
            return False
        except Exception as e:
            self.logger.error(f"GDELT API connectivity test failed: {e}")
            return False

    def extract_gdelt_article_data(self, articles: List[Dict]) -> List[Dict]:
        """
        Extract simplified, structured data from the article response.
        """
        return [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "image": a.get("socialimage"),
                "seen_date": a.get("seendate"),
                "domain": a.get("domain"),
                "language": a.get("language"),
                "country": a.get("sourcecountry"),
            }
            for a in articles
        ]

    @retry_with_backoff(max_attempts=5, base_delay=5)
    def fetch_gdelt_articles_http(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        country: str,
        maxrecords: int = 250,
    ) -> List[Dict]:
        """
        Sends a POST request to fetch articles from MASX GDELT API.
        """
        # Beauty
        payload = {
            k: v
            for k, v in {
                "keyword": keyword,
                "start_date": start_date,
                "end_date": end_date,
                "country": country,
                "maxrecords": maxrecords,
            }.items()
            if v
        }

        try:
            # Add timeout to prevent hanging
            response = requests.post(
                self.BASE_URL + self.ENDPOINT,
                json=payload,
                headers=self.headers,
                timeout=(10, 30),  # (connect_timeout, read_timeout)
            )
            # time.sleep(2)
            response.raise_for_status()
            return safe_json_loads(response.text)
        except requests.Timeout as e:
            self.logger.error(
                f"MASX GDELT API timeout for [{keyword} – {country}]: {e}"
            )
            raise e
        except requests.ConnectionError as e:
            self.logger.error(
                f"MASX GDELT API connection error for [{keyword} – {country}]: {e}"
            )
            raise e
        except requests.HTTPError as e:
            if response.status_code == 500:
                self.logger.error(
                    f"MASX GDELT API 500 error for [{keyword} – {country}]: {e}"
                )
                # time.sleep(10) # TODO: remove this
            else:
                self.logger.error(f"HTTP error: {e} | Status: {response.status_code}")
            raise e
        except requests.RequestException as e:
            self.logger.error(f"MASX GDELT API error [{keyword} – {country}]: {e}")
            # time.sleep(2)
            raise e

    @retry_with_backoff(max_attempts=5, base_delay=5, jitter=True)
    def fetch_gdelt_articles(
        self,
        keyword: str,
        start_date: str,
        end_date: str,
        country: str,
        maxrecords: int = 250,
    ) -> List[Dict]:
        """
        Sends a request to fetch articles from GDELT API.
        """
        try:
            f = Filters(
                keyword=keyword,
                start_date=start_date,
                end_date=end_date,
                country=country,
                max_records=maxrecords,
            )
            articles = self.gdelt_doc.article_search(f)
            time.sleep(random.uniform(0, 5))
            return articles

        except RateLimitError as e:
            self.logger.warning(
                f"[GDELT Rate Limit] keyword={keyword}, country={country}: {e}",
                exc_info=True,
            )
            time.sleep(60)
            raise

        except Exception as e:
            self.logger.error(
                f"[GDELT Fetch Failed] keyword={keyword}, country={country}: {e}",
                exc_info=True,
            )
            raise

    def _fetch_one_combo(self, combo: Dict) -> Tuple[Dict, Optional[List[Dict]]]:
        """
        Internal helper to fetch one keyword-country-date combo.
        """
        try:
            keyword = combo.get("keyword", "")
            country = combo.get("country", "")
            start_date = combo.get("start_date", "")
            end_date = combo.get("end_date", "")
            maxrecords = combo.get("maxrecords", 250)
            is_valid, errors = self.validate_search_query(combo)
            if not is_valid:
                self.logger.error(f"Invalid search query: {errors}")
                return combo, []

            articles = self.fetch_gdelt_articles(
                keyword, start_date, end_date, country, maxrecords
            )
            return combo, articles
        except Exception as e:
            self.logger.error(
                f"[Combo Failed] keyword={keyword}, country={country}, date={start_date}–{end_date} | Error: {e}",
                exc_info=True,
            )
            return combo, []

    def fetch_articles_batch_threaded(
        self, combos: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Fetch multiple keyword-country-date combinations using threading.
        Returns a dictionary keyed by '<keyword>_<country>' → list of articles.
        """
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_combo = {
                executor.submit(self._fetch_one_combo, combo): combo for combo in combos
            }

            for future in as_completed(future_to_combo):
                combo = future_to_combo[future]  # Ensure combo is always available
                keyword = combo.get("keyword", "")
                country = combo.get("country", "")
                key = f"{keyword}_{country}"

                try:
                    _, articles = future.result()
                    if articles:
                        simplified = self.extract_gdelt_article_data(articles)
                        results[key] = simplified
                        self.logger.info(
                            f"[{key}] - {len(simplified)} articles fetched."
                        )
                    else:
                        results[key] = []
                        self.logger.warning(f"[{key}] - No articles fetched.")
                except Exception as e:
                    self.logger.error(
                        f"[{key}] - Exception during fetch: {e}", exc_info=True
                    )
                    results[key] = []

        return results

    def validate_search_query(self, data: dict) -> tuple[bool, list[str]]:
        errors = []

        required_fields = ["keyword", "start_date", "end_date", "maxrecords"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing field: {field}")

        if errors:
            return False, errors

        # Keyword validation
        if not isinstance(data["keyword"], str):
            errors.append("Keyword must be a string.")
        else:
            keywords = [k.strip() for k in data["keyword"].split(",") if k.strip()]
            if not keywords:
                errors.append(
                    "Keyword list must contain at least one non-empty keyword."
                )

        # Date parsing and validation
        try:
            start_date = datetime.strptime(data["start_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("start_date must be in YYYY-MM-DD format.")
            start_date = None

        try:
            end_date = datetime.strptime(data["end_date"], "%Y-%m-%d")
        except ValueError:
            errors.append("end_date must be in YYYY-MM-DD format.")
            end_date = None

        if start_date and end_date and end_date < start_date:
            errors.append("end_date cannot be earlier than start_date.")

        # Country check
        # if not isinstance(data["country"], str) or not data["country"].strip():
        #     errors.append("Country must be a non-empty string.")

        # Maxrecords check
        if not isinstance(data["maxrecords"], int) or not (
            1 <= data["maxrecords"] <= 1000
        ):
            errors.append("maxrecords must be an integer between 1 and 1000.")

        return len(errors) == 0, errors
