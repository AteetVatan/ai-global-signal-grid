"""
Event Fetcher Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Fetching events from GDELT 2.0 API
- Processing GDELT event data
- Rate limiting and error handling
- Filtering events by relevance
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseAgent, AgentResult
from ..services.data_sources import DataSourcesService
from ..core.state import AgentState
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger


class EventFetcher(BaseAgent):
    """
    Event Fetcher Agent for retrieving events from GDELT 2.0.

    Responsibilities:
    - Fetch events from GDELT API
    - Process and filter event data
    - Handle rate limiting and errors
    - Apply relevance filters
    """

    def __init__(self):
        """Initialize the Event Fetcher agent."""
        super().__init__("EventFetcher")
        self.data_sources_service = DataSourcesService()
        self.logger = get_agent_logger("EventFetcher")

    async def fetch_events(
        self, queries: List[Dict[str, Any]], max_events: Optional[int] = 100
    ) -> AgentResult:
        """
        Fetch events from GDELT based on queries.

        Args:
            queries: List of GDELT query configurations
            max_events: Maximum number of events to fetch per query

        Returns:
            AgentResult: Contains fetched GDELT events
        """
        try:
            self.logger.info(
                "Fetching GDELT events", query_count=len(queries), max_events=max_events
            )

            all_events = []

            for query in queries:
                keywords = query.get("keywords", [])
                themes = query.get("themes", [])
                locations = query.get("locations", [])

                # Fetch events for this query
                events = await self._fetch_events_for_query(
                    keywords, themes, locations, max_events
                )
                all_events.extend(events)

                self.logger.info(
                    f"Fetched {len(events)} events for keywords: {keywords}"
                )

            result = {
                "events": all_events,
                "total_count": len(all_events),
                "queries_executed": len(queries),
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.logger.info(
                "Event fetching completed",
                total_events=len(all_events),
                queries_executed=len(queries),
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow(),
                    "query_count": len(queries),
                },
            )

        except Exception as e:
            self.logger.error(f"Event fetching failed: {e}")
            raise AgentException(f"Event fetching failed: {str(e)}")

    async def _fetch_events_for_query(
        self,
        keywords: List[str],
        themes: List[str],
        locations: List[str],
        max_events: int,
    ) -> List[Dict[str, Any]]:
        """Fetch events for a specific query configuration."""
        try:
            # Use the data sources service to fetch from GDELT
            events = await self.data_sources_service.fetch_gdelt_events(
                keywords=keywords,
                themes=themes,
                locations=locations,
                max_results=max_events,
            )

            # Convert to standard format
            formatted_events = []
            for event in events:
                formatted_events.append(
                    {
                        "title": event.get("title", ""),
                        "url": event.get("url", ""),
                        "source": event.get("source", ""),
                        "published_date": event.get("published_date", ""),
                        "language": event.get("language", ""),
                        "keywords": keywords,
                        "themes": themes,
                        "locations": locations,
                    }
                )

            return formatted_events

        except Exception as e:
            self.logger.warning(f"Failed to fetch events for keywords {keywords}: {e}")
            return []

    async def fetch_trending_events(
        self, time_range: Optional[str] = "24h", max_events: Optional[int] = 50
    ) -> AgentResult:
        """
        Fetch trending events from GDELT.

        Args:
            time_range: Time range for events (e.g., "24h", "7d")
            max_events: Maximum number of events to fetch

        Returns:
            AgentResult: Contains trending events
        """
        try:
            self.logger.info(
                "Fetching trending events", time_range=time_range, max_events=max_events
            )

            # Fetch trending events using default parameters
            events = await self._fetch_events_for_query(
                keywords=["trending", "breaking"],
                themes=[],
                locations=[],
                max_events=max_events,
            )

            result = {
                "events": events,
                "time_range": time_range,
                "total_count": len(events),
                "timestamp": datetime.utcnow().isoformat(),
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow(),
                    "time_range": time_range,
                },
            )

        except Exception as e:
            self.logger.error(f"Trending events fetching failed: {e}")
            raise AgentException(f"Trending events fetching failed: {str(e)}")

    async def filter_events_by_relevance(
        self, events: List[Dict[str, Any]], relevance_threshold: Optional[float] = 0.5
    ) -> AgentResult:
        """
        Filter events by relevance score.

        Args:
            events: List of events to filter
            relevance_threshold: Minimum relevance score

        Returns:
            AgentResult: Contains filtered events
        """
        try:
            self.logger.info(
                f"Filtering {len(events)} events by relevance",
                threshold=relevance_threshold,
            )

            relevant_events = []
            irrelevant_events = []

            for event in events:
                # Simple relevance scoring (in production, use ML model)
                relevance_score = self._calculate_relevance_score(event)

                if relevance_score >= relevance_threshold:
                    event["relevance_score"] = relevance_score
                    relevant_events.append(event)
                else:
                    irrelevant_events.append(event)

            result = {
                "relevant_events": relevant_events,
                "irrelevant_events": irrelevant_events,
                "filtering_stats": {
                    "total": len(events),
                    "relevant": len(relevant_events),
                    "irrelevant": len(irrelevant_events),
                    "threshold": relevance_threshold,
                },
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Event filtering failed: {e}")
            raise AgentException(f"Event filtering failed: {str(e)}")

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
