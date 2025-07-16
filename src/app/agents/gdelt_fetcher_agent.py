"""
Event Fetcher Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Fetching events from GDELT 2.0 API
- Processing GDELT event data
- Rate limiting and error handling
- Filtering events by relevance
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from itertools import combinations

from .base import BaseAgent, AgentResult
from ..services.data_sources import DataSourcesService
from ..core.state import AgentState
from ..core import QueryState, QueryTranslated, FeedEntry
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger
from ..services import MasxGdeltService
from ..core.country_normalizer import CountryNormalizer
from ..core.language_utils import LanguageUtils

class GdeltFetcherAgent(BaseAgent):
    """
    Event Fetcher Agent for retrieving events from GDELT 2.0.

    Responsibilities:
    - Fetch events from GDELT API
    - Process and filter event data
    - Handle rate limiting and errors
    - Apply relevance filters
    """

    def __init__(self):
        """Initialize the GDELT Fetcher agent."""  
        super().__init__(
            name="GdeltFetcherAgent",
            description="Extracts RSS feeds from Google News",
        )
        #self.feed_parser_service = FeedParserService()
        self.logger = get_agent_logger("GdeltFetcherAgent")
        self.masx_gdelt_service = MasxGdeltService()
        self.country_normalizer = CountryNormalizer()
        
        
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute Google RSS Feeder Agent.
        Args: input_data: Dictionary containing 'queries' field
        Returns: AgentResult: Result with RSS URLs and feed entries
        """
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise AgentException("Invalid input: missing queries")

            queries = [
                QueryState.model_validate(q) for q in input_data.get("queries", [])
            ]

            result:AgentResult = self.fetch_events(queries)

            return result
        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Gdelt Fetcher Agent failed: {str(e)}",
                metadata={"exception_type": type(e).__name__},
            )

    def fetch_events(self, queries: List[QueryState], max_events: Optional[int] = 1000) -> AgentResult:
        """
        Fetch events from GDELT based on queries.

        Args:
            queries: List of GDELT query configurations
            max_events: Maximum number of events to fetch per query

        Returns:
            AgentResult: Contains fetched GDELT events
        """
        try:
            self.logger.info("Fetching GDELT events", query_count=len(queries), max_events=max_events)

            combo_set = set()  # for deduplication (keyword, country)
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            for query in queries:
                
                maxrecords = 250

                countries = self.country_normalizer.get_country_names_from_pycountry(query.entities)
                if not countries:
                    continue

                combo_list = []

                for r in range(1, len(countries) + 1):
                    for keyword_combo in combinations(countries, r):
                        keyword_str = ", ".join(keyword_combo)

                        for country in countries:
                            combo_key = (keyword_str, country)
                            if combo_key in combo_set:
                                continue
                            combo_set.add(combo_key)

                            combo = {
                                "start_date": start_date,
                                "end_date": end_date,
                                "maxrecords": maxrecords,
                                "keyword": keyword_str,
                                "country": country
                            }
                            combo_list.append(combo)

                if not combo_list:
                    query.gdelt_feed_entries = []
                    continue

                # Fetch in batch (threaded)
                results = self.masx_gdelt_service.fetch_articles_batch_threaded(combo_list)

                # Convert articles to FeedEntry
                feed_entries = []
                for key, articles in results.items():
                    for article in articles:
                        feed_entries.append(FeedEntry(
                            url=article.get("url", ""),
                            title=article.get("title"),
                            seendate=article.get("published"),
                            domain={
                                "title": article.get("source", {}).get("title", ""),
                                "href": article.get("source", {}).get("href", ""),
                            },
                            description=article.get("title"),
                            language=LanguageUtils.get_language_code(article.get("language", "")),
                            country=article.get("sourcecountry", "")
                        ))

                query.gdelt_feed_entries = feed_entries

            return AgentResult(
                success=True,
                data={"queries": queries, "timestamp": datetime.utcnow().isoformat()},
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            return AgentResult(
                success=False,
                error=f"Gdelt Fetcher Agent failed: {str(e)}",
                metadata={"exception_type": type(e).__name__},
            )



    def _calculate_relevance_score(self, event: Dict[str, Any]) -> float:
        """Calculate relevance score for an event."""
        # Simple scoring based on presence of key fields
        score = 0.0

        if event.get("title"):
            score += 0.3
        if event.get("url"):
            score += 0.2
        if event.get("source"):
            score += 0.2
        if event.get("published_date"):
            score += 0.2
        if event.get("keywords"):
            score += 0.1

        return min(score, 1.0)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data for google rss agent.
        Args: input_data: Input data to validate
        Returns: bool: True if input is valid

          input_data = {
                "queries": flashpoint.queries, # list[QueryState]
            }

        """
        if not isinstance(input_data, dict):
            return False
        queries = input_data.get("queries", [])
        if not isinstance(queries, list):
            return False
        for query in queries:
            if not isinstance(query, QueryState):
                return False
        return True
