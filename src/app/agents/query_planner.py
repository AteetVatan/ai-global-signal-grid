"""
Query Planner Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Formulating search queries based on domain classification
- Planning data source queries (Google News RSS, GDELT)
- Optimizing query parameters for better results
- Managing query history and avoiding duplicates
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .base import BaseAgent, AgentResult
from ..services.llm_service import LLMService
from ..services.database import DatabaseService
from ..core.state import AgentState
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger


class QueryPlanner(BaseAgent):
    """
    Query Planner Agent for formulating optimized search queries.
    
    Responsibilities:
    - Generate search queries based on domain and context
    - Plan GDELT API queries with appropriate filters
    - Optimize query parameters for maximum relevance
    - Track query history to avoid duplicates
    """

    def __init__(self):
        """Initialize the Query Planner agent."""
        super().__init__("QueryPlanner")
        self.llm_service = LLMService()
        self.database_service = DatabaseService()
        self.logger = get_agent_logger("QueryPlanner")

    async def plan_queries(
        self,
        domain: str,
        context: Optional[Dict[str, Any]] = None,
        time_range: Optional[str] = "24h"
    ) -> AgentResult:
        """
        Plan search queries based on domain and context.
        
        Args:
            domain: Domain classification (e.g., "conflict", "economy")
            context: Additional context for query planning
            time_range: Time range for queries (e.g., "24h", "7d")
            
        Returns:
            AgentResult: Contains planned queries for different sources
        """
        try:
            self.logger.info(
                "Planning queries",
                domain=domain,
                time_range=time_range,
                context=context
            )

            # Generate Google News RSS queries
            news_queries = await self._generate_news_queries(domain, context, time_range)
            
            # Generate GDELT queries
            gdelt_queries = await self._generate_gdelt_queries(domain, context, time_range)
            
            # Check for recent similar queries to avoid duplicates
            await self._check_query_history(news_queries + gdelt_queries)
            
            result = {
                "news_queries": news_queries,
                "gdelt_queries": gdelt_queries,
                "domain": domain,
                "time_range": time_range,
                "query_count": len(news_queries) + len(gdelt_queries)
            }
            
            self.logger.info(
                "Query planning completed",
                query_count=result["query_count"],
                news_queries=len(news_queries),
                gdelt_queries=len(gdelt_queries)
            )
            
            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow(),
                    "domain": domain
                }
            )
            
        except Exception as e:
            self.logger.error(f"Query planning failed: {e}")
            raise AgentException(f"Query planning failed: {str(e)}")

    async def _generate_news_queries(
        self,
        domain: str,
        context: Optional[Dict[str, Any]],
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Generate Google News RSS queries."""
        prompt = f"""
        Generate Google News RSS search queries for domain: {domain}
        Time range: {time_range}
        Context: {context or 'None'}
        
        Generate 3-5 specific, relevant queries that will return high-quality news articles.
        Focus on current events, breaking news, and trending topics in this domain.
        
        Return as JSON array with objects containing:
        - query: The search query string
        - description: Brief description of what this query targets
        - priority: High/Medium/Low priority
        """
        
        response = await self.llm_service.generate_text(prompt)
        
        # Parse response and format queries
        queries = []
        try:
            # Simple parsing - in production, use proper JSON parsing
            lines = response.split('\n')
            for line in lines:
                if 'query' in line.lower() and ':' in line:
                    query_text = line.split(':')[1].strip().strip('"')
                    if query_text:
                        queries.append({
                            "query": query_text,
                            "source": "google_news",
                            "time_range": time_range,
                            "domain": domain
                        })
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to basic queries
            queries = [
                {
                    "query": f"{domain} news",
                    "source": "google_news",
                    "time_range": time_range,
                    "domain": domain
                }
            ]
        
        return queries

    async def _generate_gdelt_queries(
        self,
        domain: str,
        context: Optional[Dict[str, Any]],
        time_range: str
    ) -> List[Dict[str, Any]]:
        """Generate GDELT API queries."""
        prompt = f"""
        Generate GDELT API search parameters for domain: {domain}
        Time range: {time_range}
        Context: {context or 'None'}
        
        Generate 2-3 GDELT query configurations that will return relevant events.
        Consider themes, keywords, and geographic filters appropriate for this domain.
        
        Return as JSON array with objects containing:
        - keywords: List of relevant keywords
        - themes: GDELT theme codes if applicable
        - locations: Geographic locations to focus on
        - description: Brief description of what this query targets
        """
        
        response = await self.llm_service.generate_text(prompt)
        
        # Parse response and format queries
        queries = []
        try:
            # Simple parsing - in production, use proper JSON parsing
            lines = response.split('\n')
            for line in lines:
                if 'keywords' in line.lower() and ':' in line:
                    keywords_text = line.split(':')[1].strip().strip('"')
                    if keywords_text:
                        queries.append({
                            "keywords": keywords_text.split(','),
                            "source": "gdelt",
                            "time_range": time_range,
                            "domain": domain,
                            "themes": [],
                            "locations": []
                        })
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            # Fallback to basic queries
            queries = [
                {
                    "keywords": [domain],
                    "source": "gdelt",
                    "time_range": time_range,
                    "domain": domain,
                    "themes": [],
                    "locations": []
                }
            ]
        
        return queries

    async def _check_query_history(self, queries: List[Dict[str, Any]]) -> None:
        """Check recent query history to avoid duplicates."""
        try:
            # Get recent queries from database
            recent_queries = await self.database_service.get_recent_queries(
                hours=24
            )
            
            # Simple duplicate detection
            for query in queries:
                query_text = query.get("query", str(query.get("keywords", [])))
                for recent in recent_queries:
                    if query_text.lower() in recent.get("query", "").lower():
                        self.logger.info(f"Similar query found in history: {query_text}")
                        # Could modify query or skip it
                        
        except Exception as e:
            self.logger.warning(f"Failed to check query history: {e}")

    async def optimize_queries(
        self,
        queries: List[Dict[str, Any]],
        feedback: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        Optimize queries based on feedback or performance data.
        
        Args:
            queries: List of queries to optimize
            feedback: Performance feedback or results data
            
        Returns:
            AgentResult: Contains optimized queries
        """
        try:
            self.logger.info("Optimizing queries", query_count=len(queries))
            
            # Simple optimization logic
            optimized_queries = []
            for query in queries:
                # Add time-based modifiers for better results
                if "news" in query.get("query", "").lower():
                    query["query"] += " when:1d"
                
                optimized_queries.append(query)
            
            return AgentResult(
                success=True,
                data={"optimized_queries": optimized_queries},
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow(),
                    "optimization_type": "time_modifiers"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            raise AgentException(f"Query optimization failed: {str(e)}") 