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
Unit tests for MASXOrchestrator.

Tests workflow orchestration, agent coordination, state management,
and error handling capabilities.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.workflows.orchestrator import MASXOrchestrator
from app.core.state import MASXState, WorkflowState, AgentState
from app.core.exceptions import WorkflowException
from app.agents.base import AgentResult


class TestMASXOrchestrator:
    """Test cases for MASXOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("app.workflows.orchestrator.get_settings"):
            with patch("app.workflows.orchestrator.get_workflow_logger"):
                self.orchestrator = MASXOrchestrator()

    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.settings is not None
        assert self.orchestrator.logger is not None
        assert isinstance(self.orchestrator.agents, dict)
        assert isinstance(self.orchestrator.workflows, dict)

    def test_agent_initialization_with_import_error(self):
        """Test agent initialization when some agents are not available."""
        with patch("app.workflows.orchestrator.get_workflow_logger") as mock_logger:
            orchestrator = MASXOrchestrator()
            # Should handle import errors gracefully
            assert isinstance(orchestrator.agents, dict)

    def test_create_workflow_graph_daily(self):
        """Test daily workflow graph creation."""
        workflow = self.orchestrator.create_workflow_graph("daily")
        assert workflow is not None
        # Check that workflow has expected nodes
        assert hasattr(workflow, "nodes")

    def test_create_workflow_graph_detection(self):
        """Test detection workflow graph creation."""
        workflow = self.orchestrator.create_workflow_graph("detection")
        assert workflow is not None

    def test_create_workflow_graph_trigger(self):
        """Test trigger workflow graph creation."""
        workflow = self.orchestrator.create_workflow_graph("trigger")
        assert workflow is not None

    def test_create_workflow_graph_invalid_type(self):
        """Test workflow graph creation with invalid type."""
        with pytest.raises(WorkflowException, match="Unknown workflow type"):
            self.orchestrator.create_workflow_graph("invalid_type")

    def test_start_workflow(self):
        """Test workflow initialization."""
        state = MASXState()
        result_state = self.orchestrator._start_workflow(state)

        assert result_state.workflow_id is not None
        assert result_state.timestamp is not None
        assert result_state.workflow is not None
        assert result_state.workflow.current_step == "start"

    def test_run_domain_classifier_success(self):
        """Test successful domain classifier execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True,
            data={"domains": ["Geopolitical", "Economic"]},
            execution_time=1.0,
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["domain_classifier"] = mock_agent

        # Test state
        state = MASXState()
        state.metadata = {"title": "Test", "description": "Test description"}

        result_state = self.orchestrator._run_domain_classifier(state)

        assert result_state.workflow.current_step == "domain_classification"
        assert "domain_classifier" in result_state.agents
        assert result_state.metadata.get("domains") == ["Geopolitical", "Economic"]

    def test_run_domain_classifier_agent_not_available(self):
        """Test domain classifier execution when agent is not available."""
        state = MASXState()
        state.metadata = {"title": "Test", "description": "Test description"}

        result_state = self.orchestrator._run_domain_classifier(state)

        assert len(result_state.errors) > 0
        assert "DomainClassifier agent not available" in result_state.errors[0]

    def test_run_domain_classifier_execution_error(self):
        """Test domain classifier execution with error."""
        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.run.side_effect = Exception("Test error")

        self.orchestrator.agents["domain_classifier"] = mock_agent

        state = MASXState()
        state.metadata = {"title": "Test", "description": "Test description"}

        result_state = self.orchestrator._run_domain_classifier(state)

        assert len(result_state.errors) > 0
        assert "Domain classification failed" in result_state.errors[0]

    def test_run_query_planner_success(self):
        """Test successful query planner execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True, data={"queries": ["query1", "query2"]}, execution_time=1.0
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["query_planner"] = mock_agent

        # Test state
        state = MASXState()
        state.metadata = {"domains": ["Geopolitical"], "context": {}}

        result_state = self.orchestrator._run_query_planner(state)

        assert result_state.workflow.current_step == "query_planning"
        assert "query_planner" in result_state.agents
        assert result_state.metadata.get("queries") == ["query1", "query2"]

    def test_run_data_fetchers_parallel_execution(self):
        """Test parallel data fetching execution."""
        # Mock agents
        mock_news_agent = Mock()
        mock_news_agent.run.return_value = AgentResult(
            success=True,
            data={"articles": ["article1", "article2"]},
            execution_time=1.0,
        )
        mock_news_agent.state = AgentState(status="completed")

        mock_event_agent = Mock()
        mock_event_agent.run.return_value = AgentResult(
            success=True, data={"events": ["event1", "event2"]}, execution_time=1.0
        )
        mock_event_agent.state = AgentState(status="completed")

        self.orchestrator.agents["news_fetcher"] = mock_news_agent
        self.orchestrator.agents["event_fetcher"] = mock_event_agent

        # Test state
        state = MASXState()
        state.metadata = {"queries": ["query1", "query2"]}

        with patch("asyncio.run") as mock_asyncio_run:
            mock_asyncio_run.return_value = [
                mock_news_agent.state,
                mock_event_agent.state,
            ]

            result_state = self.orchestrator._run_data_fetchers(state)

            assert result_state.workflow.current_step == "data_fetching"
            assert "news_fetcher" in result_state.agents
            assert "event_fetcher" in result_state.agents

    def test_run_data_fetchers_no_agents(self):
        """Test data fetching when no agents are available."""
        state = MASXState()
        state.metadata = {"queries": ["query1", "query2"]}

        result_state = self.orchestrator._run_data_fetchers(state)

        assert result_state.workflow.current_step == "data_fetching"
        assert len(result_state.agents) == 0

    def test_run_merge_deduplicator_success(self):
        """Test successful merge deduplicator execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True,
            data={"merged_articles": ["article1", "article2"]},
            execution_time=1.0,
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["merge_deduplicator"] = mock_agent

        # Test state with previous agent outputs
        state = MASXState()
        state.agents = {
            "news_fetcher": {"output": {"articles": ["article1"]}},
            "event_fetcher": {"output": {"articles": ["article2"]}},
        }

        result_state = self.orchestrator._run_merge_deduplicator(state)

        assert result_state.workflow.current_step == "merge_deduplication"
        assert "merge_deduplicator" in result_state.agents

    def test_run_language_processing_success(self):
        """Test successful language processing execution."""
        # Mock agents
        mock_resolver = Mock()
        mock_resolver.run.return_value = AgentResult(
            success=True, data={"languages": ["en", "es"]}, execution_time=1.0
        )
        mock_resolver.state = AgentState(status="completed")

        mock_translator = Mock()
        mock_translator.run.return_value = AgentResult(
            success=True,
            data={"translated": ["translated1", "translated2"]},
            execution_time=1.0,
        )
        mock_translator.state = AgentState(status="completed")

        self.orchestrator.agents["language_resolver"] = mock_resolver
        self.orchestrator.agents["translator"] = mock_translator

        # Test state
        state = MASXState()
        state.agents = {"merge_deduplicator": {"output": {"articles": ["article1"]}}}

        result_state = self.orchestrator._run_language_processing(state)

        assert result_state.workflow.current_step == "language_processing"
        assert "language_resolver" in result_state.agents
        assert "translator" in result_state.agents

    def test_run_entity_extractor_success(self):
        """Test successful entity extraction execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True,
            data={"entities": {"people": ["John"], "orgs": ["UN"]}},
            execution_time=1.0,
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["entity_extractor"] = mock_agent

        # Test state
        state = MASXState()
        state.agents = {"merge_deduplicator": {"output": {"articles": ["article1"]}}}

        result_state = self.orchestrator._run_entity_extractor(state)

        assert result_state.workflow.current_step == "entity_extraction"
        assert "entity_extractor" in result_state.agents

    def test_run_event_analyzer_success(self):
        """Test successful event analysis execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True,
            data={"hotspots": [{"title": "Event1", "articles": ["article1"]}]},
            execution_time=1.0,
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["event_analyzer"] = mock_agent

        # Test state
        state = MASXState()
        state.agents = {
            "merge_deduplicator": {"output": {"articles": ["article1"]}},
            "entity_extractor": {"output": {"entities": {"people": ["John"]}}},
        }

        result_state = self.orchestrator._run_event_analyzer(state)

        assert result_state.workflow.current_step == "event_analysis"
        assert "event_analyzer" in result_state.agents

    def test_run_fact_checker_success(self):
        """Test successful fact checking execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True, data={"verified": True}, execution_time=1.0
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["fact_checker"] = mock_agent

        # Test state
        state = MASXState()
        state.agents = {"event_analyzer": {"output": {"hotspots": []}}}

        result_state = self.orchestrator._run_fact_checker(state)

        assert result_state.workflow.current_step == "fact_checking"
        assert "fact_checker" in result_state.agents

    def test_run_validator_success(self):
        """Test successful validation execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True, data={"valid": True}, execution_time=1.0
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["validator"] = mock_agent

        # Test state
        state = MASXState()
        state.agents = {"fact_checker": {"output": {"verified": True}}}

        result_state = self.orchestrator._run_validator(state)

        assert result_state.workflow.current_step == "validation"
        assert "validator" in result_state.agents

    def test_run_memory_manager_success(self):
        """Test successful memory storage execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True, data={"stored": True}, execution_time=1.0
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["memory_manager"] = mock_agent

        # Test state
        state = MASXState()
        state.workflow_id = "test-run-123"
        state.agents = {"validator": {"output": {"valid": True}}}

        result_state = self.orchestrator._run_memory_manager(state)

        assert result_state.workflow.current_step == "memory_storage"
        assert "memory_manager" in result_state.agents

    def test_end_workflow_success(self):
        """Test successful workflow completion."""
        state = MASXState()
        state.errors = []

        result_state = self.orchestrator._end_workflow(state)

        assert result_state.workflow.completed is True
        assert result_state.workflow.failed is False
        assert result_state.workflow.current_step == "end"

    def test_end_workflow_with_errors(self):
        """Test workflow completion with errors."""
        state = MASXState()
        state.errors = ["Error 1", "Error 2"]

        result_state = self.orchestrator._end_workflow(state)

        assert result_state.workflow.completed is True
        assert result_state.workflow.failed is True
        assert result_state.workflow.current_step == "end"

    def test_run_workflow_success(self):
        """Test successful workflow execution."""
        with patch.object(self.orchestrator, "create_workflow_graph") as mock_create:
            mock_workflow = Mock()
            mock_workflow.compile.return_value = Mock()
            mock_workflow.compile.return_value.invoke.return_value = MASXState(
                workflow_id="test-run",
                workflow=WorkflowState(completed=True, failed=False),
            )
            mock_create.return_value = mock_workflow

            result = self.orchestrator.run_workflow("daily")

            assert result.workflow_id == "test-run"
            assert result.workflow.completed is True
            assert result.workflow.failed is False

    def test_run_workflow_execution_error(self):
        """Test workflow execution with error."""
        with patch.object(self.orchestrator, "create_workflow_graph") as mock_create:
            mock_workflow = Mock()
            mock_workflow.compile.side_effect = Exception("Workflow error")
            mock_create.return_value = mock_workflow

            with pytest.raises(WorkflowException, match="Workflow execution failed"):
                self.orchestrator.run_workflow("daily")

    def test_run_daily_workflow(self):
        """Test daily workflow execution."""
        with patch.object(self.orchestrator, "run_workflow") as mock_run:
            mock_run.return_value = MASXState()

            result = self.orchestrator.run_daily_workflow()

            mock_run.assert_called_once_with("daily", None)
            assert result is not None

    def test_run_detection_workflow(self):
        """Test detection workflow execution."""
        with patch.object(self.orchestrator, "run_workflow") as mock_run:
            mock_run.return_value = MASXState()

            result = self.orchestrator.run_detection_workflow()

            mock_run.assert_called_once_with("detection", None)
            assert result is not None

    def test_run_trigger_workflow(self):
        """Test trigger workflow execution."""
        with patch.object(self.orchestrator, "run_workflow") as mock_run:
            mock_run.return_value = MASXState()

            result = self.orchestrator.run_trigger_workflow()

            mock_run.assert_called_once_with("trigger", None)
            assert result is not None

    def test_run_agent_async(self):
        """Test async agent execution."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.run.return_value = AgentResult(
            success=True, data={"result": "test"}, execution_time=1.0
        )
        mock_agent.state = AgentState(status="completed")

        self.orchestrator.agents["test_agent"] = mock_agent

        # Test async execution
        async def test_async():
            result = await self.orchestrator._run_agent_async(
                "test_agent", {"input": "test"}
            )
            return result

        import asyncio

        result = asyncio.run(test_async())

        assert result.status == "completed"
        mock_agent.run.assert_called_once_with({"input": "test"})

    def test_run_agent_async_agent_not_available(self):
        """Test async agent execution when agent is not available."""

        async def test_async():
            with pytest.raises(
                WorkflowException, match="Agent test_agent not available"
            ):
                await self.orchestrator._run_agent_async(
                    "test_agent", {"input": "test"}
                )

        import asyncio

        asyncio.run(test_async())

    def test_detection_workflow_steps(self):
        """Test detection workflow step implementations."""
        state = MASXState()

        # Test detect anomaly step
        result_state = self.orchestrator._detect_anomaly(state)
        assert result_state.workflow.current_step == "detect_anomaly"

        # Test classify anomaly step
        result_state = self.orchestrator._classify_anomaly(state)
        assert result_state.workflow.current_step == "classify_anomaly"

        # Test resolve anomaly step
        result_state = self.orchestrator._resolve_anomaly(state)
        assert result_state.workflow.current_step == "resolve_anomaly"

        # Test visualize result step
        result_state = self.orchestrator._visualize_result(state)
        assert result_state.workflow.current_step == "visualize_result"

    def test_trigger_workflow_steps(self):
        """Test trigger workflow step implementations."""
        state = MASXState()

        # Test trigger workflow step
        result_state = self.orchestrator._trigger_workflow(state)
        assert result_state.workflow.current_step == "trigger_workflow"

        # Test delegate tasks step
        result_state = self.orchestrator._delegate_tasks(state)
        assert result_state.workflow.current_step == "delegate_tasks"

        # Test fetch results step
        result_state = self.orchestrator._fetch_results(state)
        assert result_state.workflow.current_step == "fetch_results"

        # Test update plan step
        result_state = self.orchestrator._update_plan(state)
        assert result_state.workflow.current_step == "update_plan"

        # Test re-execute step
        result_state = self.orchestrator._re_execute(state)
        assert result_state.workflow.current_step == "re_execute"

    def test_should_continue(self):
        """Test continuation logic for trigger workflow."""
        state = MASXState()

        # Test default behavior (should complete)
        result = self.orchestrator._should_continue(state)
        assert result == "complete"

        # Test with custom logic (would need implementation)
        # This is a placeholder test for when the logic is implemented
        assert result in ["continue", "complete"]
