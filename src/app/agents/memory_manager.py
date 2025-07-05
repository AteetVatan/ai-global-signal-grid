"""
Memory Manager Agent

This agent handles all interactions with the Supabase Postgres database and pgvector long-term memory.
It is responsible for storing new data and retrieving relevant historical data for other agents.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid

from ..core.state import MASXState
from ..core.exceptions import AgentException, DatabaseException
from ..services.database import DatabaseService
from ..services.embedding import EmbeddingService
from .base import BaseAgent


class MemoryManager(BaseAgent):
    """
    Memory Manager Agent for handling database operations and vector memory.
    
    This agent:
    - Stores hotspots, articles, and embeddings in Supabase
    - Retrieves historical data using vector similarity search
    - Manages long-term memory for context and continuity
    - Provides data persistence and retrieval services
    """
    
    def __init__(self, database_service: Optional[DatabaseService] = None,
                 embedding_service: Optional[EmbeddingService] = None):
        """Initialize the Memory Manager agent."""
        super().__init__("memory_manager")
        self.database_service = database_service or DatabaseService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.logger = logging.getLogger(__name__)
        
    def store_hotspots(self, hotspots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store hotspots in the database with embeddings.
        
        Args:
            hotspots: List of hotspot dictionaries to store
            
        Returns:
            Dict containing storage results
        """
        try:
            self.logger.info(f"Storing {len(hotspots)} hotspots in database")
            
            stored_hotspots = []
            failed_hotspots = []
            
            for hotspot in hotspots:
                try:
                    # Generate embedding for hotspot
                    hotspot_text = f"{hotspot.get('title', '')} {hotspot.get('summary', '')}"
                    embedding = self.embedding_service.generate_embedding(hotspot_text)
                    
                    # Prepare hotspot data for storage
                    hotspot_data = {
                        "id": hotspot.get("id", str(uuid.uuid4())),
                        "title": hotspot.get("title", ""),
                        "summary": hotspot.get("summary", ""),
                        "article_urls": json.dumps(hotspot.get("article_urls", [])),
                        "entities": json.dumps(hotspot.get("entities", [])),
                        "confidence_score": hotspot.get("confidence_score", 0.0),
                        "embedding": embedding,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Store in database
                    stored_id = self.database_service.insert_hotspot(hotspot_data)
                    
                    stored_hotspots.append({
                        "id": stored_id,
                        "title": hotspot.get("title", ""),
                        "stored_at": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to store hotspot {hotspot.get('id', 'unknown')}: {str(e)}")
                    failed_hotspots.append({
                        "hotspot": hotspot,
                        "error": str(e)
                    })
            
            return {
                "stored_hotspots": stored_hotspots,
                "failed_hotspots": failed_hotspots,
                "total_stored": len(stored_hotspots),
                "total_failed": len(failed_hotspots)
            }
            
        except Exception as e:
            self.logger.error(f"Error during hotspot storage: {str(e)}")
            raise AgentException(f"Hotspot storage failed: {str(e)}")
    
    def store_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store articles in the database with embeddings.
        
        Args:
            articles: List of article dictionaries to store
            
        Returns:
            Dict containing storage results
        """
        try:
            self.logger.info(f"Storing {len(articles)} articles in database")
            
            stored_articles = []
            failed_articles = []
            
            for article in articles:
                try:
                    # Generate embedding for article
                    article_text = f"{article.get('title', '')} {article.get('content', '')}"
                    embedding = self.embedding_service.generate_embedding(article_text)
                    
                    # Prepare article data for storage
                    article_data = {
                        "id": article.get("id", str(uuid.uuid4())),
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "content": article.get("content", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "language": article.get("language", "en"),
                        "embedding": embedding,
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Store in database
                    stored_id = self.database_service.insert_article(article_data)
                    
                    stored_articles.append({
                        "id": stored_id,
                        "url": article.get("url", ""),
                        "stored_at": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to store article {article.get('url', 'unknown')}: {str(e)}")
                    failed_articles.append({
                        "article": article,
                        "error": str(e)
                    })
            
            return {
                "stored_articles": stored_articles,
                "failed_articles": failed_articles,
                "total_stored": len(stored_articles),
                "total_failed": len(failed_articles)
            }
            
        except Exception as e:
            self.logger.error(f"Error during article storage: {str(e)}")
            raise AgentException(f"Article storage failed: {str(e)}")
    
    def search_similar_events(self, query_text: str, limit: int = 5, 
                            similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar events using vector similarity.
        
        Args:
            query_text: Text to search for similar events
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar events
        """
        try:
            self.logger.info(f"Searching for events similar to: {query_text[:100]}...")
            
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query_text)
            
            # Search database for similar events
            similar_events = self.database_service.search_similar_hotspots(
                query_embedding, limit=limit, similarity_threshold=similarity_threshold
            )
            
            self.logger.info(f"Found {len(similar_events)} similar events")
            return similar_events
            
        except Exception as e:
            self.logger.error(f"Error during similar event search: {str(e)}")
            raise AgentException(f"Similar event search failed: {str(e)}")
    
    def get_recent_hotspots(self, days: int = 7, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent hotspots from the database.
        
        Args:
            days: Number of days to look back
            limit: Maximum number of results to return
            
        Returns:
            List of recent hotspots
        """
        try:
            self.logger.info(f"Retrieving hotspots from last {days} days")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_hotspots = self.database_service.get_hotspots_since(
                cutoff_date.isoformat(), limit=limit
            )
            
            self.logger.info(f"Retrieved {len(recent_hotspots)} recent hotspots")
            return recent_hotspots
            
        except Exception as e:
            self.logger.error(f"Error retrieving recent hotspots: {str(e)}")
            raise AgentException(f"Recent hotspots retrieval failed: {str(e)}")
    
    def get_context_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get a summary of recent context for other agents.
        
        Args:
            days: Number of days to include in context
            
        Returns:
            Dict containing context summary
        """
        try:
            self.logger.info(f"Generating context summary for last {days} days")
            
            # Get recent hotspots
            recent_hotspots = self.get_recent_hotspots(days=days, limit=100)
            
            # Extract key themes and entities
            themes = self._extract_themes(recent_hotspots)
            entities = self._extract_entities(recent_hotspots)
            
            # Calculate trends
            trends = self._calculate_trends(recent_hotspots)
            
            context_summary = {
                "period_days": days,
                "total_hotspots": len(recent_hotspots),
                "key_themes": themes,
                "key_entities": entities,
                "trends": trends,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            self.logger.info("Context summary generated successfully")
            return context_summary
            
        except Exception as e:
            self.logger.error(f"Error generating context summary: {str(e)}")
            raise AgentException(f"Context summary generation failed: {str(e)}")
    
    def _extract_themes(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key themes from hotspots."""
        try:
            themes = {}
            
            for hotspot in hotspots:
                title = hotspot.get("title", "").lower()
                summary = hotspot.get("summary", "").lower()
                
                # Simple theme extraction (could be enhanced with LLM)
                theme_keywords = [
                    "conflict", "crisis", "election", "protest", "economic", 
                    "military", "diplomatic", "health", "environment", "technology"
                ]
                
                for keyword in theme_keywords:
                    if keyword in title or keyword in summary:
                        themes[keyword] = themes.get(keyword, 0) + 1
            
            # Return top themes
            sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
            return [{"theme": theme, "count": count} for theme, count in sorted_themes[:10]]
            
        except Exception as e:
            self.logger.warning(f"Failed to extract themes: {str(e)}")
            return []
    
    def _extract_entities(self, hotspots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract key entities from hotspots."""
        try:
            entities = {}
            
            for hotspot in hotspots:
                hotspot_entities = hotspot.get("entities", [])
                if isinstance(hotspot_entities, str):
                    try:
                        hotspot_entities = json.loads(hotspot_entities)
                    except:
                        hotspot_entities = []
                
                for entity in hotspot_entities:
                    if isinstance(entity, dict):
                        entity_name = entity.get("name", "")
                        entity_type = entity.get("type", "unknown")
                        
                        if entity_name:
                            key = f"{entity_name} ({entity_type})"
                            entities[key] = entities.get(key, 0) + 1
            
            # Return top entities
            sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
            return [{"entity": entity, "count": count} for entity, count in sorted_entities[:20]]
            
        except Exception as e:
            self.logger.warning(f"Failed to extract entities: {str(e)}")
            return []
    
    def _calculate_trends(self, hotspots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trends from recent hotspots."""
        try:
            if not hotspots:
                return {"trend": "no_data", "confidence": 0.0}
            
            # Group by date
            daily_counts = {}
            for hotspot in hotspots:
                created_at = hotspot.get("created_at", "")
                if created_at:
                    try:
                        date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).date()
                        daily_counts[date] = daily_counts.get(date, 0) + 1
                    except:
                        continue
            
            if len(daily_counts) < 2:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            # Calculate trend
            dates = sorted(daily_counts.keys())
            counts = [daily_counts[date] for date in dates]
            
            # Simple trend calculation
            if len(counts) >= 2:
                recent_avg = sum(counts[-3:]) / min(3, len(counts))
                earlier_avg = sum(counts[:-3]) / max(1, len(counts) - 3)
                
                if recent_avg > earlier_avg * 1.2:
                    trend = "increasing"
                    confidence = min(0.9, (recent_avg - earlier_avg) / earlier_avg)
                elif recent_avg < earlier_avg * 0.8:
                    trend = "decreasing"
                    confidence = min(0.9, (earlier_avg - recent_avg) / earlier_avg)
                else:
                    trend = "stable"
                    confidence = 0.5
            else:
                trend = "stable"
                confidence = 0.5
            
            return {
                "trend": trend,
                "confidence": round(confidence, 2),
                "daily_counts": daily_counts
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate trends: {str(e)}")
            return {"trend": "error", "confidence": 0.0}
    
    def store_execution_log(self, workflow_id: str, execution_data: Dict[str, Any]) -> str:
        """
        Store execution log in the database.
        
        Args:
            workflow_id: ID of the workflow execution
            execution_data: Data about the execution
            
        Returns:
            ID of the stored log entry
        """
        try:
            self.logger.info(f"Storing execution log for workflow: {workflow_id}")
            
            log_data = {
                "workflow_id": workflow_id,
                "execution_data": json.dumps(execution_data),
                "created_at": datetime.utcnow().isoformat()
            }
            
            log_id = self.database_service.insert_execution_log(log_data)
            
            self.logger.info(f"Execution log stored with ID: {log_id}")
            return log_id
            
        except Exception as e:
            self.logger.error(f"Error storing execution log: {str(e)}")
            raise AgentException(f"Execution log storage failed: {str(e)}")
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, Any]:
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dict containing cleanup results
        """
        try:
            self.logger.info(f"Cleaning up data older than {days_to_keep} days")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean up old hotspots
            deleted_hotspots = self.database_service.delete_hotspots_before(
                cutoff_date.isoformat()
            )
            
            # Clean up old articles
            deleted_articles = self.database_service.delete_articles_before(
                cutoff_date.isoformat()
            )
            
            # Clean up old logs
            deleted_logs = self.database_service.delete_logs_before(
                cutoff_date.isoformat()
            )
            
            cleanup_results = {
                "deleted_hotspots": deleted_hotspots,
                "deleted_articles": deleted_articles,
                "deleted_logs": deleted_logs,
                "cutoff_date": cutoff_date.isoformat(),
                "cleanup_timestamp": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Cleanup completed: {deleted_hotspots} hotspots, "
                           f"{deleted_articles} articles, {deleted_logs} logs deleted")
            
            return cleanup_results
            
        except Exception as e:
            self.logger.error(f"Error during data cleanup: {str(e)}")
            raise AgentException(f"Data cleanup failed: {str(e)}")
    
    def execute(self, state: MASXState) -> MASXState:
        """Execute the memory management workflow."""
        try:
            self.logger.info("Starting memory manager execution")
            
            # Get data from state
            hotspots = state.workflow.get("hotspots", [])
            articles = state.workflow.get("articles", [])
            workflow_id = state.workflow.get("workflow_id", str(uuid.uuid4()))
            
            memory_results = {}
            
            # Store hotspots if available
            if hotspots:
                memory_results["hotspots"] = self.store_hotspots(hotspots)
            
            # Store articles if available
            if articles:
                memory_results["articles"] = self.store_articles(articles)
            
            # Get context summary for future reference
            context_summary = self.get_context_summary(days=7)
            memory_results["context_summary"] = context_summary
            
            # Store execution log
            execution_data = {
                "workflow_id": workflow_id,
                "hotspots_count": len(hotspots),
                "articles_count": len(articles),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            log_id = self.store_execution_log(workflow_id, execution_data)
            memory_results["execution_log_id"] = log_id
            
            # Update state
            state.agents[self.name] = {
                "status": "completed",
                "output": {
                    "memory_results": memory_results,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            self.logger.info("Memory manager execution completed successfully")
            return state
            
        except Exception as e:
            self.logger.error(f"Memory manager execution failed: {str(e)}")
            state.agents[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            return state 