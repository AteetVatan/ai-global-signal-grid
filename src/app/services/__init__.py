"""
Services for Global Signal Grid (MASX) Agentic AI System.

This package contains service layer components including:
- Database service with Supabase integration
- Translation service for multilingual support
- Embedding service for vector operations
- Data sources integration (Google News RSS, GDELT)
- Data processing and ETL pipelines
- Real-time streaming and WebSocket support
- Advanced analytics and reporting
- External API integrations

Usage:
    from app.services import DatabaseService, TranslationService, EmbeddingService
"""

from .database import DatabaseService
from .translation import TranslationService
from .embedding import EmbeddingService
from .llm_service import LLMService
from .data_sources import DataSourcesService, Article, GDELTEvent
from .data_processing import DataProcessingPipeline, ProcessedArticle, TrendAnalysis
from .streaming import StreamingService, WebSocketManager, StreamEvent, StreamFilter
from .analytics import DataAnalyticsService, AnalyticsReport, RiskAssessment
from .token_tracker import TokenCostTracker, get_token_tracker
from .web_search import WebSearchService, create_web_search_service
from .flashpoint_detection import FlashpointDetectionService, Flashpoint, create_flashpoint_detection_service

__all__ = [
    "DatabaseService",
    "TranslationService", 
    "EmbeddingService",
    "LLMService",
    "DataSourcesService",
    "Article",
    "GDELTEvent",
    "DataProcessingPipeline",
    "ProcessedArticle",
    "TrendAnalysis",
    "StreamingService",
    "WebSocketManager",
    "StreamEvent",
    "StreamFilter",
    "DataAnalyticsService",
    "AnalyticsReport",
    "RiskAssessment",
    "TokenCostTracker",
    "get_token_tracker",
    "WebSearchService",
    "create_web_search_service",
    "FlashpointDetectionService",
    "Flashpoint",
    "create_flashpoint_detection_service",
]
