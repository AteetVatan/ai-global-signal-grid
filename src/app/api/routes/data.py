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

"""
Data retrieval endpoints for Global Signal Grid (MASX) Agentic AI System.

Provides endpoints for:
- Hotspot data retrieval
- Article data retrieval
- Analytics and statistics
- Search and filtering
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ...config.logging_config import get_api_logger
from ...services import DatabaseService

router = APIRouter()
logger = get_api_logger("DataRoutes")


class FlashpointResponse(BaseModel):
    """Flashpoint response model."""

    id: str
    title: str
    summary: str
    domains: List[str]
    entities: Dict[str, List[str]]
    articles: List[str]
    confidence_score: float
    created_at: str
    updated_at: str


class FeedResponse(BaseModel):
    """Feed response model."""

    id: str
    url: str
    title: str
    content: str
    source: str
    language: str
    published_at: Optional[str]
    entities: Dict[str, List[str]]
    created_at: str


class AnalyticsResponse(BaseModel):
    """Analytics response model."""

    total_hotspots: int
    total_articles: int
    domains_distribution: Dict[str, int]
    sources_distribution: Dict[str, int]
    languages_distribution: Dict[str, int]
    time_series: List[Dict[str, Any]]


@router.get("/flashpoints", response_model=List[FlashpointResponse])
async def get_flashpoints(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    domains: Optional[str] = Query(None, description="Comma-separated list of domains"),
    run_id: Optional[str] = Query(None, description="Filter by run ID"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
):
    """
    Get flashpoints with optional filtering.

    Args:
        limit: Maximum number of hotspots to return
        offset: Number of hotspots to skip
        domains: Filter by domains (comma-separated)
        run_id: Filter by run ID
        sort_by: Field to sort by
        sort_order: Sort order

    Returns:
        List of hotspot records
    """
    logger.info("Flashpoints retrieval requested")

    try:
        async with DatabaseService() as db:
            # Parse domains filter
            domain_list = None
            if domains:
                domain_list = [d.strip() for d in domains.split(",")]

            hotspots = await db.get_hotspots(
                limit=limit, offset=offset, domains=domain_list, run_id=run_id
            )

            # Convert to response format
            response_data = []
            for hotspot in hotspots:
                response_data.append(
                    FlashpointResponse(
                        id=hotspot.id or "",
                        title=hotspot.title,
                        summary=hotspot.summary,
                        domains=hotspot.domains,
                        entities=hotspot.entities,
                        articles=hotspot.articles,
                        confidence_score=hotspot.confidence_score,
                        created_at=(
                            hotspot.created_at.isoformat() if hotspot.created_at else ""
                        ),
                        updated_at=(
                            hotspot.updated_at.isoformat() if hotspot.updated_at else ""
                        ),
                    )
                )

            logger.info(f"Flashpoints retrieved: {len(response_data)} records")
            return response_data

    except Exception as e:
        logger.error(f"Flashpoints retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Flashpoints retrieval failed: {str(e)}"
        )


@router.get("/flashpoints/{flashpoint_id}", response_model=FlashpointResponse)
async def get_flashpoint(flashpoint_id: str):
    """
    Get a specific flashpoint by ID.

    Args:
        flashpoint_id: Flashpoint ID

    Returns:
        Flashpoint record
    """
    logger.info(f"Flashpoint retrieval requested: {flashpoint_id}")       

    try:
        async with DatabaseService() as db:
            flashpoint = await db.get_flashpoint(flashpoint_id)

            if not flashpoint:
                raise HTTPException(
                    status_code=404, detail=f"Flashpoint {flashpoint_id} not found"
                )

            response = FlashpointResponse(
                id=flashpoint.id or "",
                title=flashpoint.title,
                summary=flashpoint.summary,
                domains=flashpoint.domains,
                entities=flashpoint.entities,
                articles=flashpoint.articles,
                confidence_score=flashpoint.confidence_score,
                created_at=flashpoint.created_at.isoformat() if flashpoint.created_at else "",
                updated_at=flashpoint.updated_at.isoformat() if flashpoint.updated_at else "",
            )

            logger.info(f"Flashpoint retrieved: {flashpoint_id}")
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hotspot retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Hotspot retrieval failed: {str(e)}"
        )


@router.get("/feeds", response_model=List[FeedResponse])
async def get_feeds(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    language: Optional[str] = Query(None, description="Filter by language"),
    source: Optional[str] = Query(None, description="Filter by source"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
):
    """
    Get feeds with optional filtering.

    Args:
        limit: Maximum number of feeds to return
        offset: Number of feeds to skip
        language: Filter by language
        source: Filter by source
        sort_by: Field to sort by
        sort_order: Sort order

    Returns:
        List of feed records
    """
    logger.info("Feeds retrieval requested")

    try:
        async with DatabaseService() as db:
            feeds = await db.get_feeds(
                limit=limit, offset=offset, language=language, source=source
            )

            # Convert to response format
            response_data = []
            for feed in feeds:
                response_data.append(
                    FeedResponse(
                        id=feed.id or "",
                        url=feed.url,
                        title=feed.title,
                        content=feed.content,
                        source=feed.source,
                        language=feed.language,
                        published_at=(
                            feed.published_at.isoformat()
                            if feed.published_at
                            else None
                        ),
                        entities=feed.entities,
                        created_at=(
                            feed.created_at.isoformat() if feed.created_at else ""
                        ),
                    )
                )

            logger.info(f"Feeds retrieved: {len(response_data)} records")
            return response_data

    except Exception as e:
        logger.error(f"Feeds retrieval failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Feeds retrieval failed: {str(e)}"
        )


# @router.get("/search")
# async def search_data(
#     query: str = Query(..., description="Search query"),
#     search_type: str = Query("hotspots", description="Search type (hotspots/articles)"),
#     limit: int = Query(10, ge=1, le=100),
#     similarity_threshold: float = Query(0.7, ge=0.0, le=1.0),
# ):
#     """
#     Search data using vector similarity.

#     Args:
#         query: Search query text
#         search_type: Type of data to search
#         limit: Maximum number of results
#         similarity_threshold: Minimum similarity score

#     Returns:
#         Search results
#     """
#     logger.info(f"Data search requested: {query}")

#     try:
#         from ...services import EmbeddingService

#         async with EmbeddingService() as embedder:
#             # Generate embedding for query
#             embedding_result = await embedder.embed_text(query)

#             if search_type == "hotspots":
#                 async with DatabaseService() as db:
#                     similar_hotspots = await db.search_hotspots_by_similarity(
#                         embedding=embedding_result.embedding,
#                         limit=limit,
#                         similarity_threshold=similarity_threshold,
#                     )

#                     results = []
#                     for hotspot in similar_hotspots:
#                         results.append(
#                             {
#                                 "id": hotspot.id,
#                                 "title": hotspot.title,
#                                 "summary": hotspot.summary,
#                                 "similarity_score": hotspot.confidence_score,
#                                 "domains": hotspot.domains,
#                                 "created_at": (
#                                     hotspot.created_at.isoformat()
#                                     if hotspot.created_at
#                                     else None
#                                 ),
#                             }
#                         )

#                     logger.info(f"Search completed: {len(results)} results")
#                     return {
#                         "query": query,
#                         "search_type": search_type,
#                         "results": results,
#                         "total": len(results),
#                     }
#             else:
#                 raise HTTPException(
#                     status_code=400, detail=f"Unsupported search type: {search_type}"
#                 )

#     except Exception as e:
#         logger.error(f"Data search failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Data search failed: {str(e)}")


# @router.get("/analytics", response_model=AnalyticsResponse)
# async def get_analytics(
#     days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
# ):
#     """
#     Get analytics and statistics.

#     Args:
#         days: Number of days to analyze

#     Returns:
#         Analytics data
#     """
#     logger.info("Analytics requested")

#     try:
#         async with DatabaseService() as db:
#             # Get database statistics
#             stats = await db.get_database_stats()

#             # Get recent hotspots and articles
#             from datetime import datetime, timedelta

#             cutoff_date = datetime.utcnow() - timedelta(days=days)

#             recent_hotspots = await db.get_hotspots(limit=1000)
#             recent_articles = await db.get_articles(limit=1000)

#             # Calculate distributions
#             domains_distribution = {}
#             sources_distribution = {}
#             languages_distribution = {}

#             for hotspot in recent_hotspots:
#                 for domain in hotspot.domains:
#                     domains_distribution[domain] = (
#                         domains_distribution.get(domain, 0) + 1
#                     )

#             for article in recent_articles:
#                 sources_distribution[article.source] = (
#                     sources_distribution.get(article.source, 0) + 1
#                 )
#                 languages_distribution[article.language] = (
#                     languages_distribution.get(article.language, 0) + 1
#                 )

#             # Generate time series data (mock for now)
#             time_series = []
#             for i in range(days):
#                 date = datetime.utcnow() - timedelta(days=i)
#                 time_series.append(
#                     {
#                         "date": date.date().isoformat(),
#                         "hotspots_count": len(
#                             [
#                                 h
#                                 for h in recent_hotspots
#                                 if h.created_at and h.created_at.date() == date.date()
#                             ]
#                         ),
#                         "articles_count": len(
#                             [
#                                 a
#                                 for a in recent_articles
#                                 if a.created_at and a.created_at.date() == date.date()
#                             ]
#                         ),
#                     }
#                 )

#             analytics = AnalyticsResponse(
#                 total_hotspots=stats.get("hotspots_count", 0),
#                 total_articles=stats.get("articles_count", 0),
#                 domains_distribution=domains_distribution,
#                 sources_distribution=sources_distribution,
#                 languages_distribution=languages_distribution,
#                 time_series=time_series,
#             )

#             logger.info("Analytics generated successfully")
#             return analytics

#     except Exception as e:
#         logger.error(f"Analytics generation failed: {e}")
#         raise HTTPException(
#             status_code=500, detail=f"Analytics generation failed: {str(e)}"
#         )


# @router.get("/stats")
# async def get_data_stats():
#     """
#     Get data statistics.

#     Returns:
#         Data statistics
#     """
#     logger.info("Data statistics requested")

#     try:
#         async with DatabaseService() as db:
#             stats = await db.get_database_stats()

#             logger.info("Data statistics retrieved")
#             return stats

#     except Exception as e:
#         logger.error(f"Data statistics retrieval failed: {e}")
#         raise HTTPException(
#             status_code=500, detail=f"Data statistics retrieval failed: {str(e)}"
#         )


@router.get("/export")
async def export_data(
    data_type: str = Query(..., description="Data type to export (flashpoints/feeds)"),
    format: str = Query("json", description="Export format (json/csv)"),
    limit: int = Query(1000, ge=1, le=10000),
):
    """
    Export data in various formats.

    Args:
        data_type: Type of data to export
        format: Export format
        limit: Maximum number of records

    Returns:
        Exported data
    """
    logger.info(f"Data export requested: {data_type} in {format} format")

    try:
        if data_type == "flashpoints":
            async with DatabaseService() as db:
                flashpoints = await db.get_flashpoints(limit=limit)

                if format == "json":
                    return {
                        "type": "flashpoints",
                        "format": "json",
                        "count": len(flashpoints),
                        "data": [
                            {
                                "id": f.id,
                                "title": f.title,
                                "summary": f.summary,
                                "domains": f.domains,
                                "entities": f.entities,
                                "articles": f.articles,
                                "confidence_score": f.confidence_score,
                                "created_at": (
                                    f.created_at.isoformat() if f.created_at else None
                                ),
                            }
                            for f in flashpoints
                        ],
                    }
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported format: {format}"
                    )

        elif data_type == "feeds":
            async with DatabaseService() as db:
                feeds = await db.get_feeds(limit=limit)

                if format == "json":
                    return {
                        "type": "feeds",
                        "format": "json",
                        "count": len(feeds),
                        "data": [
                            {
                                "id": f.id,
                                "url": f.url,
                                "title": f.title,
                                "content": f.content,
                                "source": f.source,
                                "language": f.language,
                                "published_at": (
                                    f.published_at.isoformat()
                                    if f.published_at
                                    else None
                                ),
                                "entities": f.entities,
                                "created_at": (
                                    f.created_at.isoformat() if f.created_at else None
                                ),
                            }
                            for f in feeds
                        ],
                    }
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported format: {format}"
                    )
        else:
            raise HTTPException(
                status_code=400, detail=f"Unsupported data type: {data_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}")
