"""
Tests for the newly implemented agents: FactChecker, Validator, MemoryManager, LoggingAuditor.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.agents import FactChecker, Validator, MemoryManager, LoggingAuditor
from app.core.state import MASXState


class TestFactChecker:
    """Test FactChecker agent."""

    def test_initialization(self):
        """Test FactChecker initialization."""
        with patch("app.agents.fact_checker.LLMService"):
            with patch("app.agents.fact_checker.DatabaseService"):
                agent = FactChecker()
                assert agent.name == "fact_checker"
                assert hasattr(agent, "verify_facts")
                assert hasattr(agent, "cross_reference_with_memory")
                assert hasattr(agent, "execute")

    def test_verify_facts_method_exists(self):
        """Test that verify_facts method exists."""
        with patch("app.agents.fact_checker.LLMService"):
            with patch("app.agents.fact_checker.DatabaseService"):
                agent = FactChecker()
                assert callable(agent.verify_facts)

    def test_cross_reference_with_memory_method_exists(self):
        """Test that cross_reference_with_memory method exists."""
        with patch("app.agents.fact_checker.LLMService"):
            with patch("app.agents.fact_checker.DatabaseService"):
                agent = FactChecker()
                assert callable(agent.cross_reference_with_memory)


class TestValidator:
    """Test Validator agent."""

    def test_initialization(self):
        """Test Validator initialization."""
        agent = Validator()
        assert agent.name == "validator"
        assert hasattr(agent, "validate_hotspots")
        assert hasattr(agent, "validate_articles")
        assert hasattr(agent, "validate_urls")
        assert hasattr(agent, "check_business_rules")
        assert hasattr(agent, "execute")

    def test_validate_hotspots_method_exists(self):
        """Test that validate_hotspots method exists."""
        agent = Validator()
        assert callable(agent.validate_hotspots)

    def test_validate_articles_method_exists(self):
        """Test that validate_articles method exists."""
        agent = Validator()
        assert callable(agent.validate_articles)

    def test_validate_urls_method_exists(self):
        """Test that validate_urls method exists."""
        agent = Validator()
        assert callable(agent.validate_urls)

    def test_check_business_rules_method_exists(self):
        """Test that check_business_rules method exists."""
        agent = Validator()
        assert callable(agent.check_business_rules)


class TestMemoryManager:
    """Test MemoryManager agent."""

    def test_initialization(self):
        """Test MemoryManager initialization."""
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                agent = MemoryManager()
                assert agent.name == "memory_manager"
                assert hasattr(agent, "store_hotspots")
                assert hasattr(agent, "store_articles")
                assert hasattr(agent, "search_similar_events")
                assert hasattr(agent, "get_context_summary")
                assert hasattr(agent, "execute")

    def test_store_hotspots_method_exists(self):
        """Test that store_hotspots method exists."""
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                agent = MemoryManager()
                assert callable(agent.store_hotspots)

    def test_store_articles_method_exists(self):
        """Test that store_articles method exists."""
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                agent = MemoryManager()
                assert callable(agent.store_articles)

    def test_search_similar_events_method_exists(self):
        """Test that search_similar_events method exists."""
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                agent = MemoryManager()
                assert callable(agent.search_similar_events)

    def test_get_context_summary_method_exists(self):
        """Test that get_context_summary method exists."""
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                agent = MemoryManager()
                assert callable(agent.get_context_summary)


class TestLoggingAuditor:
    """Test LoggingAuditor agent."""

    def test_initialization(self):
        """Test LoggingAuditor initialization."""
        with patch("app.agents.logging_auditor.DatabaseService"):
            agent = LoggingAuditor()
            assert agent.name == "logging_auditor"
            assert hasattr(agent, "log_agent_execution")
            assert hasattr(agent, "validate_content_policy")
            assert hasattr(agent, "monitor_agent_outputs")
            assert hasattr(agent, "generate_audit_report")
            assert hasattr(agent, "execute")

    def test_log_agent_execution_method_exists(self):
        """Test that log_agent_execution method exists."""
        with patch("app.agents.logging_auditor.DatabaseService"):
            agent = LoggingAuditor()
            assert callable(agent.log_agent_execution)

    def test_validate_content_policy_method_exists(self):
        """Test that validate_content_policy method exists."""
        with patch("app.agents.logging_auditor.DatabaseService"):
            agent = LoggingAuditor()
            assert callable(agent.validate_content_policy)

    def test_monitor_agent_outputs_method_exists(self):
        """Test that monitor_agent_outputs method exists."""
        with patch("app.agents.logging_auditor.DatabaseService"):
            agent = LoggingAuditor()
            assert callable(agent.monitor_agent_outputs)

    def test_generate_audit_report_method_exists(self):
        """Test that generate_audit_report method exists."""
        with patch("app.agents.logging_auditor.DatabaseService"):
            agent = LoggingAuditor()
            assert callable(agent.generate_audit_report)


class TestAgentIntegration:
    """Test that all agents can work together."""

    def test_all_agents_importable(self):
        """Test that all new agents can be imported."""
        from app.agents import FactChecker, Validator, MemoryManager, LoggingAuditor

        # Test that all agents can be instantiated
        with patch("app.agents.fact_checker.LLMService"):
            with patch("app.agents.fact_checker.DatabaseService"):
                fact_checker = FactChecker()
                assert fact_checker.name == "fact_checker"

        validator = Validator()
        assert validator.name == "validator"

        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                memory_manager = MemoryManager()
                assert memory_manager.name == "memory_manager"

        with patch("app.agents.logging_auditor.DatabaseService"):
            logging_auditor = LoggingAuditor()
            assert logging_auditor.name == "logging_auditor"

    def test_agent_execute_methods(self):
        """Test that all agents have execute methods that return MASXState."""
        # Create a basic state
        state = MASXState(
            workflow_id="test-run",
            workflow={
                "workflow_id": "test-workflow",
                "start_time": datetime.utcnow().isoformat(),
                "hotspots": [],
                "articles": [],
            },
        )

        # Test FactChecker execute
        with patch("app.agents.fact_checker.LLMService"):
            with patch("app.agents.fact_checker.DatabaseService"):
                fact_checker = FactChecker()
                result_state = fact_checker.execute(state)
                assert isinstance(result_state, MASXState)
                assert "fact_checker" in result_state.agents

        # Test Validator execute
        validator = Validator()
        result_state = validator.execute(state)
        assert isinstance(result_state, MASXState)
        assert "validator" in result_state.agents

        # Test MemoryManager execute
        with patch("app.agents.memory_manager.DatabaseService"):
            with patch("app.agents.memory_manager.EmbeddingService"):
                memory_manager = MemoryManager()
                result_state = memory_manager.execute(state)
                assert isinstance(result_state, MASXState)
                assert "memory_manager" in result_state.agents

        # Test LoggingAuditor execute
        with patch("app.agents.logging_auditor.DatabaseService"):
            logging_auditor = LoggingAuditor()
            result_state = logging_auditor.execute(state)
            assert isinstance(result_state, MASXState)
            assert "logging_auditor" in result_state.agents
