"""
Unit tests for DatabaseService.

Tests database operations including:
- Connection management
- CRUD operations
- Vector similarity search
- Health checks
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from app.services.database import (
    DatabaseService,
    HotspotRecord,
    ArticleRecord,
    WorkflowLogRecord
)
from app.core.exceptions import DatabaseException, ConfigurationException


class TestDatabaseService:
    """Test cases for DatabaseService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('app.services.database.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                supabase_url="https://test.supabase.co",
                supabase_key="test-key",
                database_url="postgresql://test:test@localhost/test",
                database_max_connections=5,
                database_min_connections=1
            )
            self.db_service = DatabaseService()
    
    def test_initialization(self):
        """Test database service initialization."""
        assert self.db_service.settings is not None
        assert self.db_service.logger is not None
        assert self.db_service.client is None
        assert self.db_service.pool is None
    
    def test_initialization_missing_config(self):
        """Test initialization with missing configuration."""
        with patch('app.services.database.get_settings') as mock_settings:
            mock_settings.return_value = Mock(
                supabase_url=None,
                supabase_key=None
            )
            
            with pytest.raises(ConfigurationException, match="Supabase URL and key are required"):
                DatabaseService()
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful database connection."""
        with patch('app.services.database.create_client') as mock_create_client:
            with patch('app.services.database.asyncpg.create_pool') as mock_create_pool:
                mock_client = Mock()
                mock_pool = AsyncMock()
                mock_create_client.return_value = mock_client
                mock_create_pool.return_value = mock_pool
                
                await self.db_service.connect()
                
                assert self.db_service.client == mock_client
                assert self.db_service.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test database connection failure."""
        with patch('app.services.database.create_client', side_effect=Exception("Connection failed")):
            with pytest.raises(DatabaseException, match="Database connection failed"):
                await self.db_service.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test database disconnection."""
        self.db_service.pool = AsyncMock()
        self.db_service.client = Mock()
        
        await self.db_service.disconnect()
        
        self.db_service.pool.close.assert_called_once()
        assert self.db_service.pool is None
        assert self.db_service.client is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch.object(self.db_service, 'connect') as mock_connect:
            with patch.object(self.db_service, 'disconnect') as mock_disconnect:
                async with self.db_service as db:
                    assert db == self.db_service
                
                mock_connect.assert_called_once()
                mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_hotspot_success(self):
        """Test successful hotspot storage."""
        hotspot = HotspotRecord(
            title="Test Hotspot",
            summary="Test summary",
            domains=["Geopolitical"],
            entities={"people": ["John Doe"]},
            articles=["http://example.com/article1"]
        )
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.insert.return_value.execute.return_value.data = [
                {"id": "test-id"}
            ]
            
            result = await self.db_service.store_hotspot(hotspot)
            
            assert result == "test-id"
            mock_client.table.assert_called_with("hotspots")
    
    @pytest.mark.asyncio
    async def test_store_hotspot_failure(self):
        """Test hotspot storage failure."""
        hotspot = HotspotRecord(title="Test Hotspot")
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.insert.return_value.execute.return_value.data = []
            
            with pytest.raises(DatabaseException, match="Failed to store hotspot"):
                await self.db_service.store_hotspot(hotspot)
    
    @pytest.mark.asyncio
    async def test_get_hotspot_success(self):
        """Test successful hotspot retrieval."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
                {
                    "id": "test-id",
                    "title": "Test Hotspot",
                    "summary": "Test summary",
                    "domains": ["Geopolitical"],
                    "entities": {"people": ["John Doe"]},
                    "articles": ["http://example.com/article1"],
                    "confidence_score": 0.8,
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00"
                }
            ]
            
            result = await self.db_service.get_hotspot("test-id")
            
            assert result is not None
            assert result.id == "test-id"
            assert result.title == "Test Hotspot"
    
    @pytest.mark.asyncio
    async def test_get_hotspot_not_found(self):
        """Test hotspot retrieval when not found."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
            
            result = await self.db_service.get_hotspot("test-id")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_hotspots_with_filters(self):
        """Test hotspots retrieval with filters."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.overlaps.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = [
                {
                    "id": "test-id",
                    "title": "Test Hotspot",
                    "domains": ["Geopolitical"]
                }
            ]
            
            result = await self.db_service.get_hotspots(
                limit=10,
                offset=0,
                domains=["Geopolitical"],
                run_id="test-run"
            )
            
            assert len(result) == 1
            assert result[0].id == "test-id"
    
    @pytest.mark.asyncio
    async def test_search_hotspots_by_similarity(self):
        """Test vector similarity search."""
        embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        with patch.object(self.db_service, 'pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.fetch.return_value = [
                Mock(
                    id="test-id",
                    title="Test Hotspot",
                    similarity=0.2
                )
            ]
            
            result = await self.db_service.search_hotspots_by_similarity(
                embedding=embedding,
                limit=5,
                similarity_threshold=0.7
            )
            
            assert len(result) == 1
            assert result[0].id == "test-id"
            assert result[0].confidence_score == 0.8  # 1 - similarity
    
    @pytest.mark.asyncio
    async def test_update_hotspot_success(self):
        """Test successful hotspot update."""
        updates = {"title": "Updated Title"}
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = [
                {"id": "test-id"}
            ]
            
            result = await self.db_service.update_hotspot("test-id", updates)
            
            assert result is True
            mock_client.table.assert_called_with("hotspots")
    
    @pytest.mark.asyncio
    async def test_update_hotspot_not_found(self):
        """Test hotspot update when not found."""
        updates = {"title": "Updated Title"}
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value.data = []
            
            result = await self.db_service.update_hotspot("test-id", updates)
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_hotspot_success(self):
        """Test successful hotspot deletion."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = [
                {"id": "test-id"}
            ]
            
            result = await self.db_service.delete_hotspot("test-id")
            
            assert result is True
            mock_client.table.assert_called_with("hotspots")
    
    @pytest.mark.asyncio
    async def test_store_article_success(self):
        """Test successful article storage."""
        article = ArticleRecord(
            url="http://example.com/article",
            title="Test Article",
            content="Test content",
            source="Test Source",
            language="en"
        )
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.insert.return_value.execute.return_value.data = [
                {"id": "test-id"}
            ]
            
            result = await self.db_service.store_article(article)
            
            assert result == "test-id"
            mock_client.table.assert_called_with("articles")
    
    @pytest.mark.asyncio
    async def test_get_articles_with_filters(self):
        """Test articles retrieval with filters."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = [
                {
                    "id": "test-id",
                    "url": "http://example.com/article",
                    "title": "Test Article",
                    "source": "Test Source",
                    "language": "en"
                }
            ]
            
            result = await self.db_service.get_articles(
                limit=10,
                offset=0,
                language="en",
                source="Test Source"
            )
            
            assert len(result) == 1
            assert result[0].id == "test-id"
    
    @pytest.mark.asyncio
    async def test_store_workflow_log(self):
        """Test workflow log storage."""
        log = WorkflowLogRecord(
            run_id="test-run",
            agent="test-agent",
            action="test-action",
            status="success"
        )
        
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.insert.return_value.execute.return_value.data = [
                {"id": "test-id"}
            ]
            
            result = await self.db_service.store_workflow_log(log)
            
            assert result == "test-id"
            mock_client.table.assert_called_with("workflow_logs")
    
    @pytest.mark.asyncio
    async def test_get_workflow_logs(self):
        """Test workflow logs retrieval."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = [
                {
                    "id": "test-id",
                    "run_id": "test-run",
                    "agent": "test-agent",
                    "action": "test-action",
                    "status": "success"
                }
            ]
            
            result = await self.db_service.get_workflow_logs(
                run_id="test-run",
                agent="test-agent",
                status="success"
            )
            
            assert len(result) == 1
            assert result[0].run_id == "test-run"
    
    @pytest.mark.asyncio
    async def test_get_database_stats(self):
        """Test database statistics retrieval."""
        with patch.object(self.db_service, 'pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.fetchval.side_effect = [100, 500, 50, 25]  # Mock counts
            
            result = await self.db_service.get_database_stats()
            
            assert "hotspots_count" in result
            assert "articles_count" in result
            assert "workflow_logs_count" in result
            assert "recent_hotspots_24h" in result
            assert "recent_logs_24h" in result
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self):
        """Test old data cleanup."""
        with patch.object(self.db_service, 'pool') as mock_pool:
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            mock_conn.execute.return_value = "DELETE 10"
            
            result = await self.db_service.cleanup_old_data(days=30)
            
            assert "deleted_logs" in result
            assert "deleted_articles" in result
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful health check."""
        with patch.object(self.db_service, 'client') as mock_client:
            with patch.object(self.db_service, 'pool') as mock_pool:
                mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value.data = [
                    {"id": "test"}
                ]
                mock_conn = AsyncMock()
                mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
                mock_conn.fetchval.return_value = 1
                
                result = await self.db_service.health_check()
                
                assert result["status"] == "healthy"
                assert "connections" in result
                assert result["connections"]["supabase"] == "connected"
                assert result["connections"]["postgres"] == "connected"
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check with failures."""
        with patch.object(self.db_service, 'client') as mock_client:
            mock_client.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Connection failed")
            
            result = await self.db_service.health_check()
            
            assert result["status"] == "unhealthy"
            assert result["connections"]["supabase"] == "error: Connection failed"


class TestDatabaseRecords:
    """Test cases for database record models."""
    
    def test_hotspot_record_defaults(self):
        """Test HotspotRecord default values."""
        hotspot = HotspotRecord()
        
        assert hotspot.id is None
        assert hotspot.title == ""
        assert hotspot.summary == ""
        assert hotspot.domains == []
        assert hotspot.entities == {}
        assert hotspot.articles == []
        assert hotspot.confidence_score == 0.0
        assert hotspot.created_at is not None
        assert hotspot.updated_at is not None
    
    def test_article_record_defaults(self):
        """Test ArticleRecord default values."""
        article = ArticleRecord()
        
        assert article.id is None
        assert article.url == ""
        assert article.title == ""
        assert article.content == ""
        assert article.source == ""
        assert article.language == "en"
        assert article.entities == {}
        assert article.created_at is not None
        assert article.updated_at is not None
    
    def test_workflow_log_record_defaults(self):
        """Test WorkflowLogRecord default values."""
        log = WorkflowLogRecord()
        
        assert log.id is None
        assert log.run_id == ""
        assert log.agent == ""
        assert log.action == ""
        assert log.status == "success"
        assert log.input_data == {}
        assert log.output_data == {}
        assert log.execution_time == 0.0
        assert log.created_at is not None 