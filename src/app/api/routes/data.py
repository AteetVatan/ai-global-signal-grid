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


class HotspotResponse(BaseModel):
    """Hotspot response model."""
    id: str
    title: str
    summary: str
    domains: List[str]
    entities: Dict[str, List[str]]
    articles: List[str]
    confidence_score: float
    created_at: str
    updated_at: str


class ArticleResponse(BaseModel):
    """Article response model."""
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


@router.get("/hotspots", response_model=List[HotspotResponse])
async def get_hotspots(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    domains: Optional[str] = Query(None, description="Comma-separated list of domains"),
    run_id: Optional[str] = Query(None, description="Filter by run ID"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)")
):
    """
    Get hotspots with optional filtering.
    
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
    logger.info("Hotspots retrieval requested")
    
    try:
        async with DatabaseService() as db:
            # Parse domains filter
            domain_list = None
            if domains:
                domain_list = [d.strip() for d in domains.split(",")]
            
            hotspots = await db.get_hotspots(
                limit=limit,
                offset=offset,
                domains=domain_list,
                run_id=run_id
            )
            
            # Convert to response format
            response_data = []
            for hotspot in hotspots:
                response_data.append(HotspotResponse(
                    id=hotspot.id or "",
                    title=hotspot.title,
                    summary=hotspot.summary,
                    domains=hotspot.domains,
                    entities=hotspot.entities,
                    articles=hotspot.articles,
                    confidence_score=hotspot.confidence_score,
                    created_at=hotspot.created_at.isoformat() if hotspot.created_at else "",
                    updated_at=hotspot.updated_at.isoformat() if hotspot.updated_at else ""
                ))
            
            logger.info(f"Hotspots retrieved: {len(response_data)} records")
            return response_data
            
    except Exception as e:
        logger.error(f"Hotspots retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hotspots retrieval failed: {str(e)}")


@router.get("/hotspots/{hotspot_id}", response_model=HotspotResponse)
async def get_hotspot(hotspot_id: str):
    """
    Get a specific hotspot by ID.
    
    Args:
        hotspot_id: Hotspot ID
        
    Returns:
        Hotspot record
    """
    logger.info(f"Hotspot retrieval requested: {hotspot_id}")
    
    try:
        async with DatabaseService() as db:
            hotspot = await db.get_hotspot(hotspot_id)
            
            if not hotspot:
                raise HTTPException(status_code=404, detail=f"Hotspot {hotspot_id} not found")
            
            response = HotspotResponse(
                id=hotspot.id or "",
                title=hotspot.title,
                summary=hotspot.summary,
                domains=hotspot.domains,
                entities=hotspot.entities,
                articles=hotspot.articles,
                confidence_score=hotspot.confidence_score,
                created_at=hotspot.created_at.isoformat() if hotspot.created_at else "",
                updated_at=hotspot.updated_at.isoformat() if hotspot.updated_at else ""
            )
            
            logger.info(f"Hotspot retrieved: {hotspot_id}")
            return response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hotspot retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hotspot retrieval failed: {str(e)}")


@router.get("/articles", response_model=List[ArticleResponse])
async def get_articles(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    language: Optional[str] = Query(None, description="Filter by language"),
    source: Optional[str] = Query(None, description="Filter by source"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)")
):
    """
    Get articles with optional filtering.
    
    Args:
        limit: Maximum number of articles to return
        offset: Number of articles to skip
        language: Filter by language
        source: Filter by source
        sort_by: Field to sort by
        sort_order: Sort order
        
    Returns:
        List of article records
    """
    logger.info("Articles retrieval requested")
    
    try:
        async with DatabaseService() as db:
            articles = await db.get_articles(
                limit=limit,
                offset=offset,
                language=language,
                source=source
            )
            
            # Convert to response format
            response_data = []
            for article in articles:
                response_data.append(ArticleResponse(
                    id=article.id or "",
                    url=article.url,
                    title=article.title,
                    content=article.content,
                    source=article.source,
                    language=article.language,
                    published_at=article.published_at.isoformat() if article.published_at else None,
                    entities=article.entities,
                    created_at=article.created_at.isoformat() if article.created_at else ""
                ))
            
            logger.info(f"Articles retrieved: {len(response_data)} records")
            return response_data
            
    except Exception as e:
        logger.error(f"Articles retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Articles retrieval failed: {str(e)}")


@router.get("/search")
async def search_data(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("hotspots", description="Search type (hotspots/articles)"),
    limit: int = Query(10, ge=1, le=100),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0)
):
    """
    Search data using vector similarity.
    
    Args:
        query: Search query text
        search_type: Type of data to search
        limit: Maximum number of results
        similarity_threshold: Minimum similarity score
        
    Returns:
        Search results
    """
    logger.info(f"Data search requested: {query}")
    
    try:
        from ...services import EmbeddingService
        
        async with EmbeddingService() as embedder:
            # Generate embedding for query
            embedding_result = await embedder.embed_text(query)
            
            if search_type == "hotspots":
                async with DatabaseService() as db:
                    similar_hotspots = await db.search_hotspots_by_similarity(
                        embedding=embedding_result.embedding,
                        limit=limit,
                        similarity_threshold=similarity_threshold
                    )
                    
                    results = []
                    for hotspot in similar_hotspots:
                        results.append({
                            "id": hotspot.id,
                            "title": hotspot.title,
                            "summary": hotspot.summary,
                            "similarity_score": hotspot.confidence_score,
                            "domains": hotspot.domains,
                            "created_at": hotspot.created_at.isoformat() if hotspot.created_at else None
                        })
                    
                    logger.info(f"Search completed: {len(results)} results")
                    return {
                        "query": query,
                        "search_type": search_type,
                        "results": results,
                        "total": len(results)
                    }
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported search type: {search_type}")
                
    except Exception as e:
        logger.error(f"Data search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data search failed: {str(e)}")


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get analytics and statistics.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Analytics data
    """
    logger.info("Analytics requested")
    
    try:
        async with DatabaseService() as db:
            # Get database statistics
            stats = await db.get_database_stats()
            
            # Get recent hotspots and articles
            from datetime import datetime, timedelta
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            recent_hotspots = await db.get_hotspots(limit=1000)
            recent_articles = await db.get_articles(limit=1000)
            
            # Calculate distributions
            domains_distribution = {}
            sources_distribution = {}
            languages_distribution = {}
            
            for hotspot in recent_hotspots:
                for domain in hotspot.domains:
                    domains_distribution[domain] = domains_distribution.get(domain, 0) + 1
            
            for article in recent_articles:
                sources_distribution[article.source] = sources_distribution.get(article.source, 0) + 1
                languages_distribution[article.language] = languages_distribution.get(article.language, 0) + 1
            
            # Generate time series data (mock for now)
            time_series = []
            for i in range(days):
                date = datetime.utcnow() - timedelta(days=i)
                time_series.append({
                    "date": date.date().isoformat(),
                    "hotspots_count": len([h for h in recent_hotspots if h.created_at and h.created_at.date() == date.date()]),
                    "articles_count": len([a for a in recent_articles if a.created_at and a.created_at.date() == date.date()])
                })
            
            analytics = AnalyticsResponse(
                total_hotspots=stats.get("hotspots_count", 0),
                total_articles=stats.get("articles_count", 0),
                domains_distribution=domains_distribution,
                sources_distribution=sources_distribution,
                languages_distribution=languages_distribution,
                time_series=time_series
            )
            
            logger.info("Analytics generated successfully")
            return analytics
            
    except Exception as e:
        logger.error(f"Analytics generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


@router.get("/stats")
async def get_data_stats():
    """
    Get data statistics.
    
    Returns:
        Data statistics
    """
    logger.info("Data statistics requested")
    
    try:
        async with DatabaseService() as db:
            stats = await db.get_database_stats()
            
            logger.info("Data statistics retrieved")
            return stats
            
    except Exception as e:
        logger.error(f"Data statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data statistics retrieval failed: {str(e)}")


@router.get("/export")
async def export_data(
    data_type: str = Query(..., description="Data type to export (hotspots/articles)"),
    format: str = Query("json", description="Export format (json/csv)"),
    limit: int = Query(1000, ge=1, le=10000)
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
        if data_type == "hotspots":
            async with DatabaseService() as db:
                hotspots = await db.get_hotspots(limit=limit)
                
                if format == "json":
                    return {
                        "type": "hotspots",
                        "format": "json",
                        "count": len(hotspots),
                        "data": [
                            {
                                "id": h.id,
                                "title": h.title,
                                "summary": h.summary,
                                "domains": h.domains,
                                "entities": h.entities,
                                "articles": h.articles,
                                "confidence_score": h.confidence_score,
                                "created_at": h.created_at.isoformat() if h.created_at else None
                            }
                            for h in hotspots
                        ]
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
                    
        elif data_type == "articles":
            async with DatabaseService() as db:
                articles = await db.get_articles(limit=limit)
                
                if format == "json":
                    return {
                        "type": "articles",
                        "format": "json",
                        "count": len(articles),
                        "data": [
                            {
                                "id": a.id,
                                "url": a.url,
                                "title": a.title,
                                "content": a.content,
                                "source": a.source,
                                "language": a.language,
                                "published_at": a.published_at.isoformat() if a.published_at else None,
                                "entities": a.entities,
                                "created_at": a.created_at.isoformat() if a.created_at else None
                            }
                            for a in articles
                        ]
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported data type: {data_type}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data export failed: {str(e)}") 