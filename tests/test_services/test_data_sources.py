"""
Unit tests for Data Sources Service.

Tests:
- Google News RSS fetching
- GDELT API integration
- Rate limiting and caching
- Error handling and retry logic
- Data validation and processing
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import json

from app.services.data_sources import (
    DataSourcesService, 
    Article, 
    GDELTEvent, 
    DataSourceConfig,
    DataSourceType
)
from app.core.exceptions import DataSourceError


class TestDataSourcesService:
    """Test cases for DataSourcesService."""
    
    @pytest.fixture
    def service(self):
        """Create a DataSourcesService instance for testing."""
        return DataSourcesService()
    
    @pytest.fixture
    def mock_article(self):
        """Create a mock article for testing."""
        return Article(
            id="test_article_1",
            title="Test Article Title",
            description="This is a test article description",
            url="https://example.com/test-article",
            source="Test Source",
            published_at=datetime.now(),
            language="en",
            country="US",
            category="geopolitics",
            keywords=["test", "article", "geopolitics"]
        )
    
    @pytest.fixture
    def mock_gdelt_event(self):
        """Create a mock GDELT event for testing."""
        return GDELTEvent(
            event_id="test_event_1",
            event_time=datetime.now(),
            actor1_name="Test Actor 1",
            actor2_name="Test Actor 2",
            event_code="1234",
            avg_tone=0.5,
            num_mentions=10,
            source_url="https://example.com/test-event"
        )
    
    def test_service_initialization(self, service):
        """Test service initialization."""
        assert service is not None
        assert hasattr(service, 'sources')
        assert hasattr(service, 'cache')
        assert hasattr(service, 'rate_limiters')
        
        # Check that Google News sources are configured
        google_news_sources = [name for name, config in service.sources.items() 
                             if config.source_type == DataSourceType.GOOGLE_NEWS_RSS]
        assert len(google_news_sources) > 0
    
    @pytest.mark.asyncio
    async def test_context_manager(self, service):
        """Test async context manager."""
        async with service as s:
            assert s.session is not None
            assert s.session.closed is False
        
        assert service.session is None or service.session.closed is True
    
    @pytest.mark.asyncio
    async def test_make_request_success(self, service):
        """Test successful HTTP request."""
        mock_response = AsyncMock()
        mock_response.raise_for_status.return_value = None
        mock_response.text.return_value = "test response"
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch.object(service, 'session', mock_session):
            result = await service._make_request("https://example.com")
            assert result == "test response"
    
    @pytest.mark.asyncio
    async def test_make_request_failure(self, service):
        """Test HTTP request failure."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Connection error")
        
        with patch.object(service, 'session', mock_session):
            with pytest.raises(DataSourceError):
                await service._make_request("https://example.com")
    
    def test_rate_limit_check(self, service):
        """Test rate limiting functionality."""
        source_name = "test_source"
        
        # First request should pass
        assert service._check_rate_limit(source_name) is True
        
        # Immediate second request should fail
        assert service._check_rate_limit(source_name) is False
        
        # Check that rate limiters are tracked
        assert source_name in service.rate_limiters
    
    @pytest.mark.asyncio
    async def test_fetch_google_news_rss_success(self, service):
        """Test successful Google News RSS fetching."""
        # Mock RSS content
        mock_rss_content = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News</title>
                <item>
                    <title>Test Article 1</title>
                    <description>Test description 1</description>
                    <link>https://example.com/article1</link>
                    <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
                </item>
                <item>
                    <title>Test Article 2</title>
                    <description>Test description 2</description>
                    <link>https://example.com/article2</link>
                    <pubDate>Mon, 01 Jan 2024 13:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        # Mock the HTTP request
        with patch.object(service, '_make_request', return_value=mock_rss_content):
            with patch.object(service, '_check_rate_limit', return_value=True):
                articles = await service.fetch_google_news_rss("google_news_0")
                
                assert len(articles) == 2
                assert articles[0].title == "Test Article 1"
                assert articles[0].url == "https://example.com/article1"
                assert articles[1].title == "Test Article 2"
                assert articles[1].url == "https://example.com/article2"
    
    @pytest.mark.asyncio
    async def test_fetch_google_news_rss_rate_limit(self, service):
        """Test rate limiting in Google News RSS fetching."""
        with patch.object(service, '_check_rate_limit', return_value=False):
            with pytest.raises(DataSourceError, match="Rate limit exceeded"):
                await service.fetch_google_news_rss("google_news_0")
    
    @pytest.mark.asyncio
    async def test_fetch_google_news_rss_caching(self, service):
        """Test caching in Google News RSS fetching."""
        mock_rss_content = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>Google News</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                </item>
            </channel>
        </rss>
        """
        
        with patch.object(service, '_make_request', return_value=mock_rss_content):
            with patch.object(service, '_check_rate_limit', return_value=True):
                # First call should make HTTP request
                articles1 = await service.fetch_google_news_rss("google_news_0")
                
                # Second call should use cache
                articles2 = await service.fetch_google_news_rss("google_news_0")
                
                assert len(articles1) == len(articles2)
                assert articles1[0].title == articles2[0].title
    
    @pytest.mark.asyncio
    async def test_fetch_gdelt_events_success(self, service):
        """Test successful GDELT events fetching."""
        # Mock GDELT API response
        mock_gdelt_response = {
            "articles": [
                {
                    "seendate": "20240101120000",
                    "url": "https://example.com/gdelt1",
                    "nummentions": 5,
                    "avgtone": 0.3,
                    "mentions": []
                },
                {
                    "seendate": "20240101130000",
                    "url": "https://example.com/gdelt2",
                    "nummentions": 10,
                    "avgtone": -0.2,
                    "mentions": []
                }
            ]
        }
        
        # Mock the HTTP request
        with patch.object(service, '_make_request', return_value=json.dumps(mock_gdelt_response)):
            with patch.object(service, '_check_rate_limit', return_value=True):
                # Mock GDELT source configuration
                service.sources["gdelt"] = DataSourceConfig(
                    name="GDELT API",
                    source_type=DataSourceType.GDELT_API,
                    url="https://api.gdeltproject.org/api/v2",
                    api_key="test_key",
                    rate_limit=30,
                    cache_ttl=600
                )
                
                events = await service.fetch_gdelt_events("test query")
                
                assert len(events) == 2
                assert events[0].source_url == "https://example.com/gdelt1"
                assert events[0].num_mentions == 5
                assert events[1].source_url == "https://example.com/gdelt2"
                assert events[1].avg_tone == -0.2
    
    @pytest.mark.asyncio
    async def test_fetch_gdelt_events_no_api_key(self, service):
        """Test GDELT events fetching without API key."""
        with pytest.raises(DataSourceError, match="GDELT API key not configured"):
            await service.fetch_gdelt_events("test query")
    
    @pytest.mark.asyncio
    async def test_fetch_all_sources(self, service):
        """Test fetching from all sources."""
        mock_articles = [
            Article(
                id="test_1",
                title="Test Article 1",
                url="https://example.com/1",
                source="Test Source",
                published_at=datetime.now()
            ),
            Article(
                id="test_2",
                title="Test Article 2",
                url="https://example.com/2",
                source="Test Source",
                published_at=datetime.now()
            )
        ]
        
        # Mock individual source fetching
        with patch.object(service, 'fetch_google_news_rss', return_value=mock_articles):
            with patch.object(service, '_check_rate_limit', return_value=True):
                articles = await service.fetch_all_sources()
                
                # Should return unique articles (no duplicates)
                assert len(articles) == 2
                assert articles[0].id == "test_1"
                assert articles[1].id == "test_2"
    
    @pytest.mark.asyncio
    async def test_fetch_all_sources_with_old_articles(self, service):
        """Test fetching with age filtering."""
        old_article = Article(
            id="old_article",
            title="Old Article",
            url="https://example.com/old",
            source="Test Source",
            published_at=datetime.now() - timedelta(days=2)
        )
        
        new_article = Article(
            id="new_article",
            title="New Article",
            url="https://example.com/new",
            source="Test Source",
            published_at=datetime.now()
        )
        
        with patch.object(service, 'fetch_google_news_rss', return_value=[old_article, new_article]):
            with patch.object(service, '_check_rate_limit', return_value=True):
                articles = await service.fetch_all_sources(max_age_hours=24)
                
                # Should only return new article
                assert len(articles) == 1
                assert articles[0].id == "new_article"
    
    @pytest.mark.asyncio
    async def test_stream_articles(self, service):
        """Test article streaming functionality."""
        mock_articles = [
            Article(
                id="stream_1",
                title="Stream Article 1",
                url="https://example.com/stream1",
                source="Test Source",
                published_at=datetime.now()
            )
        ]
        
        with patch.object(service, 'fetch_all_sources', return_value=mock_articles):
            # Test streaming with short interval
            stream_count = 0
            async for articles in service.stream_articles(interval_seconds=0.1):
                stream_count += 1
                assert len(articles) == 1
                assert articles[0].id == "stream_1"
                
                if stream_count >= 2:  # Limit to 2 iterations for testing
                    break
    
    def test_get_source_status(self, service):
        """Test source status reporting."""
        status = service.get_source_status()
        
        assert isinstance(status, dict)
        assert len(status) > 0
        
        # Check structure of status data
        for source_name, source_status in status.items():
            assert "name" in source_status
            assert "type" in source_status
            assert "enabled" in source_status
            assert "rate_limit" in source_status
            assert "cache_ttl" in source_status
    
    def test_clear_cache(self, service):
        """Test cache clearing functionality."""
        # Add some test data to cache
        service.cache["test_key"] = "test_value"
        assert len(service.cache) > 0
        
        service.clear_cache()
        assert len(service.cache) == 0
    
    def test_add_custom_source(self, service):
        """Test adding custom data sources."""
        initial_count = len(service.sources)
        
        custom_config = DataSourceConfig(
            name="custom_source",
            source_type=DataSourceType.CUSTOM_RSS,
            url="https://example.com/custom-rss",
            rate_limit=50,
            cache_ttl=300
        )
        
        service.add_custom_source(custom_config)
        
        assert len(service.sources) == initial_count + 1
        assert "custom_source" in service.sources
        assert service.sources["custom_source"].source_type == DataSourceType.CUSTOM_RSS


class TestArticle:
    """Test cases for Article model."""
    
    def test_article_creation(self):
        """Test Article model creation."""
        article = Article(
            title="Test Article",
            url="https://example.com/test",
            source="Test Source",
            published_at=datetime.now()
        )
        
        assert article.title == "Test Article"
        assert article.url == "https://example.com/test"
        assert article.source == "Test Source"
        assert article.id is not None  # Should be auto-generated
    
    def test_article_id_generation(self):
        """Test automatic ID generation."""
        article1 = Article(
            title="Test Article 1",
            url="https://example.com/test1",
            source="Test Source",
            published_at=datetime.now()
        )
        
        article2 = Article(
            title="Test Article 2",
            url="https://example.com/test2",
            source="Test Source",
            published_at=datetime.now()
        )
        
        # IDs should be different
        assert article1.id != article2.id
        assert len(article1.id) == 16  # SHA256 hash truncated to 16 chars
    
    def test_article_validation(self):
        """Test Article model validation."""
        # Should raise validation error for invalid URL
        with pytest.raises(ValueError):
            Article(
                title="Test Article",
                url="invalid-url",
                source="Test Source",
                published_at=datetime.now()
            )


class TestGDELTEvent:
    """Test cases for GDELTEvent model."""
    
    def test_gdelt_event_creation(self):
        """Test GDELTEvent model creation."""
        event = GDELTEvent(
            event_id="test_event",
            event_time=datetime.now(),
            actor1_name="Actor 1",
            actor2_name="Actor 2",
            avg_tone=0.5,
            num_mentions=10
        )
        
        assert event.event_id == "test_event"
        assert event.actor1_name == "Actor 1"
        assert event.actor2_name == "Actor 2"
        assert event.avg_tone == 0.5
        assert event.num_mentions == 10 