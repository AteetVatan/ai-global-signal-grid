"""
Main orchestrator for Global Signal Grid (MASX) Agentic AI System.

Coordinates multi-agent workflows using LangGraph with:
- Agent execution coordination and state management
- Parallel processing where possible
- Error handling and recovery mechanisms
- Workflow monitoring and logging

Usage: from app.workflows.orchestrator import MASXOrchestrator
    orchestrator = MASXOrchestrator()
    result = orchestrator.run_daily_workflow()
"""

import asyncio
from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from copy import deepcopy
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_core.runnables import RunnableLambda
from ..agents.base import AgentResult
from ..core.state import MASXState, AgentState, WorkflowState

from ..core.exceptions import WorkflowException
from ..core.utils import generate_workflow_id, measure_execution_time
from ..config.logging_config import get_workflow_logger, log_workflow_step
from ..config.settings import get_settings
from ..core.flashpoint import FlashpointDataset, FlashpointItem
from ..core.querystate import QueryState


class MASXOrchestrator:
    """
    Main orchestrator for coordinating multi-agent workflows.
    Manages:
    - Agent execution and coordination
    - State transitions and validation
    - Parallel processing where applicable
    - Error handling and recovery
    - Workflow monitoring and logging
    """

    def __init__(self):
        """Initialize the MASX orchestrator."""
        self.settings = get_settings()
        self.logger = get_workflow_logger("MASXOrchestrator")
        self.agents = {}
        self.workflows = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all available agents."""
        try:
            from ..agents import (
                GoogleRssAgent,
                FlashpointLLMAgent,
                DomainClassifier,
                QueryPlanner,
                LanguageAgent,
                Translator,
                # NewsFetcher,
                # EventFetcher,
                # MergeDeduplicator,
                # LanguageResolver,
                # Translator,
                # EntityExtractor,
                # EventAnalyzer,
                # FactChecker,
                # Validator,
                # MemoryManager,
                # LoggingAuditor,
            )

            # Initialize agents
            self.agents = {
                "google_rss_agent": GoogleRssAgent(),
                "flashpoint_llm_agent": FlashpointLLMAgent(),
                "domain_classifier": DomainClassifier(),
                "query_planner": QueryPlanner(),
                "language_agent": LanguageAgent(),
                "translation_agent": Translator(),
                # "news_fetcher": NewsFetcher(),
                # "event_fetcher": EventFetcher(),
                # "merge_deduplicator": MergeDeduplicator(),
                # "language_resolver": LanguageResolver(),
                # "translator": Translator(),
                # "entity_extractor": EntityExtractor(),
                # "event_analyzer": EventAnalyzer(),
                # "fact_checker": FactChecker(),
                # "validator": Validator(),
                # "memory_manager": MemoryManager(),
                # "logging_auditor": LoggingAuditor(),
            }

            self.logger.info(
                "Agents initialized",
                agent_count=len(self.agents),
                agent_names=list(self.agents.keys()),
            )

        except ImportError as e:
            self.logger.warning(f"Some agents not available: {e}")
            # Initialize with available agents only
            self.agents = {}

    def create_workflow_graph(self, workflow_type: str = "daily") -> StateGraph:
        """
        Create a LangGraph workflow based on type.
        Args: workflow_type: Type of workflow ('daily', 'detection', 'trigger')
        Returns: StateGraph: Configured workflow graph
        """
        if workflow_type == "daily":  # normal signal processing flow
            return self._create_daily_workflow()
        elif workflow_type == "detection":  # anomaly detection
            return self._create_detection_workflow()
        elif workflow_type == "trigger":  # re-execution loop until termination
            return self._create_trigger_workflow()
        else:
            raise WorkflowException(f"Unknown workflow type: {workflow_type}")

    def _create_daily_workflow_bck(self) -> StateGraph:
        """Create the main daily workflow graph."""
        workflow = StateGraph(MASXState)

        # Add nodes for each workflow step
        workflow.add_node("start", self._start_workflow)
        workflow.add_node("flashpoint_detection", self._run_flashpoint_detection)
        # define a subgraph for each flashpoint     below
        workflow.add_node("domain_classification", self._run_domain_classifier)
        # workflow.add_node("query_planning", self._run_query_planner)
        # workflow.add_node("data_fetching", self._run_data_fetchers)
        # workflow.add_node("merge_deduplication", self._run_merge_deduplicator)
        # workflow.add_node("language_processing", self._run_language_processing)
        # workflow.add_node("entity_extraction", self._run_entity_extractor)
        # workflow.add_node("event_analysis", self._run_event_analyzer)
        # workflow.add_node("fact_checking", self._run_fact_checker)
        # workflow.add_node("validation", self._run_validator)
        # workflow.add_node("memory_storage", self._run_memory_manager)
        workflow.add_node("end", self._end_workflow)

        # Define workflow edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "flashpoint_detection")
        workflow.add_edge("flashpoint_detection", "domain_classification")
        # workflow.add_edge("domain_classification", "query_planning")
        # workflow.add_edge("query_planning", "data_fetching")
        # workflow.add_edge("data_fetching", "merge_deduplication")
        # workflow.add_edge("merge_deduplication", "language_processing")
        # workflow.add_edge("language_processing", "entity_extraction")
        # workflow.add_edge("entity_extraction", "event_analysis")
        # workflow.add_edge("event_analysis", "fact_checking")
        # workflow.add_edge("fact_checking", "validation")
        # workflow.add_edge("validation", "memory_storage")
        # workflow.add_edge("memory_storage", "end")
        workflow.add_edge("end", END)

        return workflow

    def _create_daily_workflow(self) -> StateGraph:
        """Create the main daily workflow graph."""

        # Google RSS Subgraph for each flashpoint ---
        per_flashpoint = StateGraph(MASXState)
        per_flashpoint.add_node("domain_classification", self._run_domain_classifier)
        per_flashpoint.add_node("query_planning", self._run_query_planner)
        per_flashpoint.add_node("language_agent", self._run_language_agent)
        per_flashpoint.add_node("translation_agent", self._run_translation_agent)
        per_flashpoint.add_node("google_rss_agent", self._run_google_rss_agent)
        per_flashpoint.add_node("data_fetching", self._run_data_fetchers)
        # ---now here

        # per_flashpoint.add_node("data_fetching", self._run_data_fetchers)
        # per_flashpoint.add_node("merge_deduplication", self._run_merge_deduplicator)
        # per_flashpoint.add_node("language_processing", self._run_language_processing)
        # per_flashpoint.add_node("entity_extraction", self._run_entity_extractor)
        # per_flashpoint.add_node("event_analysis", self._run_event_analyzer)
        # per_flashpoint.add_node("fact_checking", self._run_fact_checker)
        # per_flashpoint.add_node("validation", self._run_validator)
        # per_flashpoint.add_node("memory_storage", self._run_memory_manager)

        # edges

        per_flashpoint.set_entry_point("domain_classification")
        per_flashpoint.add_edge("domain_classification", "query_planning")
        per_flashpoint.add_edge("query_planning", "language_agent")
        per_flashpoint.add_edge("language_agent", "translation_agent")
        per_flashpoint.add_edge("translation_agent", "google_rss_agent")
        per_flashpoint.add_edge("google_rss_agent", "data_fetching")

        # per_flashpoint.add_edge("query_planning", "data_fetching")
        # per_flashpoint.add_edge("data_fetching", "merge_deduplication")
        # per_flashpoint.add_edge("merge_deduplication", "language_processing")

        # per_flashpoint.add_edge("language_processing", "language_agent")
        # per_flashpoint.add_edge("language_agent", "event_analysis")

        # per_flashpoint.add_edge("event_analysis", "fact_checking")
        # per_flashpoint.add_edge("fact_checking", "validation")
        # per_flashpoint.add_edge("validation", "memory_storage")
        # per_flashpoint.set_finish_point("memory_storage")

        # Compile the subgraph into a single node that can be mapped -----
        per_flashpoint_subgraph = per_flashpoint.compile()

        workflow = StateGraph(MASXState)
        # --- Main workflow nodes ---
        workflow.add_node("start", self._start_workflow)
        workflow.add_node("flashpoint_detection", self._run_flashpoint_detection)
        workflow.add_node("process_one_fp", per_flashpoint_subgraph)
        workflow.add_node("end", self._end_workflow)

        # each flashpoint to parallel processing
        def fan_out_flashpoints(state: MASXState):
            # ateet
            flashpointdataset = state.data["all_flashpoints"]
            state_list = []
            for flashpoint in flashpointdataset:
                # Initialize state
                new_state = MASXState(
                    workflow_id="sub_" + generate_workflow_id(),
                    workflow=WorkflowState(),  # generating a new workflowstate prevent th run_id bug.
                    metadata=deepcopy(state.metadata),
                    data={
                        "parent_workflow_id": state.workflow_id,
                        "current_flashpoint": flashpoint,
                        "all_flashpoints": flashpointdataset,
                    },
                )
                state_list.append(new_state)
            return [Send("process_one_fp", state) for state in state_list]

        workflow.add_edge(START, "start")
        workflow.add_edge("start", "flashpoint_detection")
        workflow.add_conditional_edges("flashpoint_detection", fan_out_flashpoints)
        workflow.add_edge("process_one_fp", "end")
        workflow.add_edge("end", END)
        return workflow

    def _create_detection_workflow(self) -> StateGraph:
        """Create detection workflow for anomaly handling."""
        workflow = StateGraph(MASXState)

        workflow.add_node("detect", self._detect_anomaly)
        workflow.add_node("classify", self._classify_anomaly)
        workflow.add_node("resolve", self._resolve_anomaly)
        workflow.add_node("visualize", self._visualize_result)

        workflow.set_entry_point("detect")
        workflow.add_edge("detect", "classify")
        workflow.add_edge("classify", "resolve")
        workflow.add_edge("resolve", "visualize")
        workflow.add_edge("visualize", END)

        return workflow

    def _create_trigger_workflow(self) -> StateGraph:
        """Create trigger workflow for iterative refinement."""
        workflow = StateGraph(MASXState)

        workflow.add_node("trigger", self._trigger_workflow)
        workflow.add_node("delegate", self._delegate_tasks)
        workflow.add_node("fetch_results", self._fetch_results)
        workflow.add_node("update_plan", self._update_plan)
        workflow.add_node("re_execute", self._re_execute)

        workflow.set_entry_point("trigger")
        workflow.add_edge("trigger", "delegate")
        workflow.add_edge("delegate", "fetch_results")
        workflow.add_edge("fetch_results", "update_plan")
        workflow.add_conditional_edges(
            "update_plan",
            self._should_continue,
            {"continue": "re_execute", "complete": END},
        )
        workflow.add_edge("re_execute", "delegate")

        return workflow

    # Workflow step implementations
    def _start_workflow(self, state: MASXState) -> MASXState:
        """Initialize workflow state."""
        if not state.workflow_id:
            state.workflow_id = generate_workflow_id()
        state.timestamp = datetime.utcnow()
        state.workflow = WorkflowState(
            steps=[
                "start",
                "flashpoint_detection",
                "domain_classification",
                "query_planning",
                "data_fetching",
            ],
            current_step="start",
        )

        log_workflow_step(
            self.logger,
            "start",
            "workflow_initialization",
            output_data={"workflow_id": state.workflow_id},
            workflow_id=state.workflow_id,
        )

        return state

    def _run_flashpoint_detection(self, state: MASXState) -> MASXState:
        """Run flashpoint detection step using FlashpointLLMAgent."""
        try:
            if self.settings.debug:

                # dummy agent result
                result = AgentResult(success=True, data={"flashpoints": []})
                print(result.data)

                # read json debug_data/flashpoint.json
                with open(
                    "src/app/debug_data/flashpoint.json", "r"
                ) as f:  # check this path
                    result.data["flashpoints"] = json.load(f)

                if isinstance(result.data["flashpoints"][0], FlashpointItem):
                    result.data["flashpoints"] = [
                        fp.model_dump() for fp in result.data["flashpoints"]
                    ]
                # data validation
                state.data["all_flashpoints"] = FlashpointDataset.model_validate(
                    result.data["flashpoints"]
                )

                state.metadata["flashpoint_stats"] = {
                    "total_count": len(result.data.get("flashpoints", [])),
                    "iterations": result.data.get("iterations", 0),
                    "search_runs": result.data.get("search_runs", 0),
                    "llm_runs": result.data.get("llm_runs", 0),
                    "token_usage": result.data.get("token_usage", {}),
                }

                return state

            state.workflow.current_step = "flashpoint_detection"

            agent = self.agents.get("flashpoint_llm_agent")
            if not agent:
                raise WorkflowException("FlashpointLLMAgent not available")

            # Prepare input data for flashpoint detection
            input_data = {
                "max_iterations": self.settings.flashpoint_max_iterations,
                "target_flashpoints": self.settings.target_flashpoints,
            }

            # Run flashpoint detection
            result = agent.run(input_data, workflow_id=state.workflow_id)

            # Update state
            # state.agents["flashpoint_llm_agent"] = agent.state
            if result.success:

                state.data["all_flashpoints"] = FlashpointDataset.model_validate(
                    result.data["flashpoints"]
                )

                state.metadata["flashpoint_stats"] = {
                    "total_count": len(result.data.get("flashpoints", [])),
                    "iterations": result.data.get("iterations", 0),
                    "search_runs": result.data.get("search_runs", 0),
                    "llm_runs": result.data.get("llm_runs", 0),
                    "token_usage": result.data.get("token_usage", {}),
                }

            state.workflow.current_step = "flashpoint_detection"

            log_workflow_step(
                self.logger,
                "flashpoint_detection",
                "flashpoint_detection",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )

            self.logger.info(
                f"Flashpoint detection completed: {len(state.metadata.get('flashpoints', []))} flashpoints found",
                workflow_id=state.workflow_id,
                flashpoint_count=len(state.metadata.get("flashpoints", [])),
            )

        except Exception as e:
            state.errors.append(f"Flashpoint detection failed: {str(e)}")
            self.logger.error(
                f"Flashpoint detection error: {e}", workflow_id=state.workflow_id
            )

        return state

    def _run_domain_classifier(self, state: MASXState) -> MASXState:
        """Run domain classification step."""
        try:
            agent = self.agents.get("domain_classifier")
            if not agent:
                raise WorkflowException("DomainClassifier agent not available")

            if self.settings.debug and False:
                # dummy agent result
                flashpoint = FlashpointItem.model_validate(
                    state.data["current_flashpoint"]
                )
                domains = '["Military / Conflict / Strategic Alliances", "Economic / Trade / Sanctions", "Religious Tensions / Ideological Movements", "Sovereignty / Border / Legal Disputes", "Environmental Flashpoints / Resource Crises", "Civilizational / Ethnonationalist Narratives"]'
                flashpoint.domains = json.loads(domains)
                state.data["current_flashpoint"] = flashpoint.model_dump()
                state.workflow.current_step = "domain_classification"
                return state

            # Flashpoint validation
            flashpoint = FlashpointItem.model_validate(state.data["current_flashpoint"])
            # Prepare input data
            input_data = flashpoint.model_dump()
            # Run agent
            result = agent.run(input_data, workflow_id=state.workflow_id)

            # Update state
            state.agents["domain_classifier"] = agent.state
            if result.success:
                domains = result.data.get("domains", [])
                flashpoint.domains = domains
                state.data["current_flashpoint"] = flashpoint.model_dump()

            state.workflow.current_step = "domain_classification"

            log_workflow_step(
                self.logger,
                "domain_classification",
                "agent_execution",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )
        except Exception as e:
            state.errors.append(f"Domain classification failed: {str(e)}")
            self.logger.error(f"Domain classification error: {e}")

        return state

    def _run_query_planner(self, state: MASXState) -> MASXState:
        """Run query planning step."""
        try:
            agent = self.agents.get("query_planner")
            if not agent:
                raise WorkflowException("QueryPlanner agent not available")

            # Flashpoint validation
            flashpoint = FlashpointItem.model_validate(state.data["current_flashpoint"])
            # Prepare input data
            input_data = {
                "title": flashpoint.title,
                "description": flashpoint.description,
                "entities": flashpoint.entities,
                "domains": flashpoint.domains,
            }

            # Run agent
            result = agent.run(input_data, workflow_id=state.workflow_id)
            # Update state
            state.agents["query_planner"] = agent.state
            if result.success:
                result_query_states = result.data.get("query_states", [])
                query_states = [
                    QueryState.model_validate(q) for q in result_query_states
                ]
                flashpoint.queries = query_states
                state.data["current_flashpoint"] = flashpoint.model_dump()

            state.workflow.current_step = "query_planning"

            log_workflow_step(
                self.logger,
                "query_planning",
                "agent_execution",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )

        except Exception as e:
            state.errors.append(f"Query planning failed: {str(e)}")
            self.logger.error(f"Query planning error: {e}")

        return state

    def _run_language_agent(self, state: MASXState) -> MASXState:
        """Run language agent step.
        This agent is responsible for extracting the language from the entities.
        """
        try:
            agent = self.agents.get("language_agent")
            if not agent:
                raise WorkflowException("LanguageAgent agent not available")

            # Flashpoint validation
            flashpoint = FlashpointItem.model_validate(state.data["current_flashpoint"])
            # Prepare input data
            input_data = {
                "queries": flashpoint.queries,
            }
            # Run agent
            result = agent.run(input_data, workflow_id=state.workflow_id)
            # Update state
            state.agents["language_agent"] = agent.state

            if result.success:
                queries = result.data.get("queries", [])
                query_states = [QueryState.model_validate(q) for q in queries]
                # update the queries in the flashpoint
                flashpoint.queries = query_states
                state.data["current_flashpoint"] = flashpoint.model_dump()
            state.workflow.current_step = "language_agent"

            log_workflow_step(
                self.logger,
                "language_agent",
                "agent_execution",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )

        except Exception as e:
            state.errors.append(f"LanguageAgent failed: {str(e)}")
            self.logger.error(f"LanguageAgent error: {e}")

        return state

    def _run_translation_agent(self, state: MASXState) -> MASXState:
        """
        Run translation step.
        This agent is responsible for translating the queries to the target languages
        (The languages extracted from the entities by the language agent).
        """
        try:
            agent = self.agents.get("translation_agent")
            if not agent:
                raise WorkflowException("TranslationAgent agent not available")

            # Flashpoint validation
            flashpoint = FlashpointItem.model_validate(state.data["current_flashpoint"])
            # Prepare input data
            input_data = {
                "queries": flashpoint.queries,
            }
            # Run agent
            result = agent.run(input_data, workflow_id=state.workflow_id)
            # Update state
            state.agents["translation_agent"] = agent.state

            if result.success:
                queries = result.data.get("queries", [])
                query_states = [QueryState.model_validate(q) for q in queries]
                # update the queries in the flashpoint
                flashpoint.queries = query_states
                state.data["current_flashpoint"] = flashpoint.model_dump()
            state.workflow.current_step = "translation_agent"

            log_workflow_step(
                self.logger,
                "translation_agent",
                "agent_execution",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )

        except Exception as e:
            state.errors.append(f"TranslationAgent failed: {str(e)}")
            self.logger.error(f"TranslationAgent error: {e}")

        return state

    def _run_google_rss_agent(self, state: MASXState) -> MASXState:
        """Run Google RSS Agent.
        This agent is responsible for fetching the news and events from the google rss feed.
        """
        try:
            agent = self.agents.get("google_rss_agent")
            if not agent:
                raise WorkflowException("GoogleRSSAgent agent not available")

            # Flashpoint validation
            flashpoint = FlashpointItem.model_validate(state.data["current_flashpoint"])
            # Prepare input data
            input_data = {
                "queries": flashpoint.queries,
            }
            # Run agent
            result = agent.run(input_data, workflow_id=state.workflow_id)
            # Update state
            state.agents["google_rss_agent"] = agent.state

            if result.success:
                queries = result.data.get("queries", [])
                query_states = [QueryState.model_validate(q) for q in queries]
                flashpoint.queries = query_states
                state.data["current_flashpoint"] = flashpoint.model_dump()
            state.workflow.current_step = "google_rss_agent"

            log_workflow_step(
                self.logger,
                "google_rss_agent",
                "agent_execution",
                input_data=input_data,
                output_data=result.data,
                workflow_id=state.workflow_id,
            )

        except Exception as e:
            state.errors.append(f"GoogleRSSAgent failed: {str(e)}")
            self.logger.error(f"GoogleRSSAgent error: {e}")

        return state

    def _run_data_fetchers(self, state: MASXState) -> MASXState:
        """Run data fetching step with parallel execution."""
        chk = state.data["current_flashpoint"]
        # try:
        #     queries = state.metadata.get("queries", [])

        #     # Run news fetcher and event fetcher in parallel
        #     tasks = []

        #     # News fetcher task
        #     if "news_fetcher" in self.agents:
        #         news_task = self._run_agent_async("news_fetcher", {"queries": queries})
        #         tasks.append(news_task)

        #     # Event fetcher task
        #     if "event_fetcher" in self.agents:
        #         event_task = self._run_agent_async(
        #             "event_fetcher", {"queries": queries}
        #         )
        #         tasks.append(event_task)

        #     # Wait for all tasks to complete
        #     if tasks:
        #         results = asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

        #         # Process results
        #         for i, result in enumerate(results):
        #             agent_name = ["news_fetcher", "event_fetcher"][i]
        #             if isinstance(result, Exception):
        #                 state.errors.append(f"{agent_name} failed: {str(result)}")
        #             else:
        #                 state.agents[agent_name] = result

        #     state.workflow.current_step = "data_fetching"

        # except Exception as e:
        #     state.errors.append(f"Data fetching failed: {str(e)}")
        #     self.logger.error(f"Data fetching error: {e}")

        return state

    async def _run_agent_async(
        self, agent_name: str, input_data: Dict[str, Any]
    ) -> AgentState:
        """Run an agent asynchronously."""
        agent = self.agents.get(agent_name)
        if not agent:
            raise WorkflowException(f"Agent {agent_name} not available")

        # For now, run synchronously (can be made truly async later)
        result = agent.run(input_data)
        return agent.state

    def _run_merge_deduplicator(self, state: MASXState) -> MASXState:
        """Run merge and deduplication step."""
        try:
            agent = self.agents.get("merge_deduplicator")
            if not agent:
                raise WorkflowException("MergeDeduplicator agent not available")

            # Prepare input data from previous steps
            input_data = {
                "news_data": state.agents.get("news_fetcher", {}).get("output"),
                "event_data": state.agents.get("event_fetcher", {}).get("output"),
            }

            result = agent.run(input_data, workflow_id=state.workflow_id)
            state.agents["merge_deduplicator"] = agent.state
            state.workflow.current_step = "merge_deduplication"

        except Exception as e:
            state.errors.append(f"Merge deduplication failed: {str(e)}")
            self.logger.error(f"Merge deduplication error: {e}")

        return state

    def _run_language_processing(self, state: MASXState) -> MASXState:
        """Run language processing step."""
        try:
            # Run language resolver
            if "language_resolver" in self.agents:
                agent = self.agents["language_resolver"]
                input_data = state.agents.get("merge_deduplicator", {}).get(
                    "output", {}
                )
                result = agent.run(input_data, workflow_id=state.workflow_id)
                state.agents["language_resolver"] = agent.state

            # Run translator if needed
            if "translator" in self.agents:
                agent = self.agents["translator"]
                input_data = state.agents.get("language_resolver", {}).get("output", {})
                result = agent.run(input_data, workflow_id=state.workflow_id)
                state.agents["translator"] = agent.state

            state.workflow.current_step = "language_processing"

        except Exception as e:
            state.errors.append(f"Language processing failed: {str(e)}")
            self.logger.error(f"Language processing error: {e}")

        return state

    def _run_event_analyzer(self, state: MASXState) -> MASXState:
        """Run event analysis step."""
        try:
            agent = self.agents.get("event_analyzer")
            if not agent:
                raise WorkflowException("EventAnalyzer agent not available")

            input_data = {
                "articles": state.agents.get("merge_deduplicator", {}).get(
                    "output", {}
                ),
                "entities": state.agents.get("entity_extractor", {}).get("output", {}),
            }

            result = agent.run(input_data, workflow_id=state.workflow_id)
            state.agents["event_analyzer"] = agent.state
            state.workflow.current_step = "event_analysis"

        except Exception as e:
            state.errors.append(f"Event analysis failed: {str(e)}")
            self.logger.error(f"Event analysis error: {e}")

        return state

    def _run_fact_checker(self, state: MASXState) -> MASXState:
        """Run fact checking step."""
        try:
            agent = self.agents.get("fact_checker")
            if not agent:
                raise WorkflowException("FactChecker agent not available")

            input_data = state.agents.get("event_analyzer", {}).get("output", {})
            result = agent.run(input_data, workflow_id=state.workflow_id)
            state.agents["fact_checker"] = agent.state
            state.workflow.current_step = "fact_checking"

        except Exception as e:
            state.errors.append(f"Fact checking failed: {str(e)}")
            self.logger.error(f"Fact checking error: {e}")

        return state

    def _run_validator(self, state: MASXState) -> MASXState:
        """Run validation step."""
        try:
            agent = self.agents.get("validator")
            if not agent:
                raise WorkflowException("Validator agent not available")

            input_data = state.agents.get("fact_checker", {}).get("output", {})
            result = agent.run(input_data, workflow_id=state.workflow_id)
            state.agents["validator"] = agent.state
            state.workflow.current_step = "validation"

        except Exception as e:
            state.errors.append(f"Validation failed: {str(e)}")
            self.logger.error(f"Validation error: {e}")

        return state

    def _run_memory_manager(self, state: MASXState) -> MASXState:
        """Run memory storage step."""
        try:
            agent = self.agents.get("memory_manager")
            if not agent:
                raise WorkflowException("MemoryManager agent not available")

            input_data = {
                "hotspots": state.agents.get("validator", {}).get("output", {}),
                "run_id": state.workflow_id,
            }

            result = agent.run(input_data, workflow_id=state.workflow_id)
            state.agents["memory_manager"] = agent.state
            state.workflow.current_step = "memory_storage"

        except Exception as e:
            state.errors.append(f"Memory storage failed: {str(e)}")
            self.logger.error(f"Memory storage error: {e}")

        return state

    def _end_workflow(self, state: MASXState) -> MASXState:
        """Finalize workflow execution."""
        state.workflow.completed = True
        state.workflow.current_step = "end"

        # Check for errors
        if state.errors:
            state.workflow.failed = True
            self.logger.error(f"Workflow completed with errors: {state.errors}")
        else:
            self.logger.info("Workflow completed successfully")

        log_workflow_step(
            self.logger,
            "end",
            "workflow_completion",
            output_data={
                "completed": state.workflow.completed,
                "failed": state.workflow.failed,
                "error_count": len(state.errors),
            },
            run_id=state.workflow_id,
        )

        return state

    # Detection workflow steps
    def _detect_anomaly(self, state: MASXState) -> MASXState:
        """Detect anomalies in the workflow."""
        # Implementation for anomaly detection
        return state

    def _classify_anomaly(self, state: MASXState) -> MASXState:
        """Classify detected anomalies."""
        # Implementation for anomaly classification
        return state

    def _resolve_anomaly(self, state: MASXState) -> MASXState:
        """Resolve detected anomalies."""
        # Implementation for anomaly resolution
        return state

    def _visualize_result(self, state: MASXState) -> MASXState:
        """Visualize workflow results."""
        # Implementation for result visualization
        return state

    # Trigger workflow steps
    def _trigger_workflow(self, state: MASXState) -> MASXState:
        """Trigger workflow execution."""
        # Implementation for workflow triggering
        return state

    def _delegate_tasks(self, state: MASXState) -> MASXState:
        """Delegate tasks to agents."""
        # Implementation for task delegation
        return state

    def _fetch_results(self, state: MASXState) -> MASXState:
        """Fetch results from agents."""
        # Implementation for result fetching
        return state

    def _update_plan(self, state: MASXState) -> MASXState:
        """Update execution plan."""
        # Implementation for plan updating
        return state

    def _re_execute(self, state: MASXState) -> MASXState:
        """Re-execute workflow steps."""
        # Implementation for re-execution
        return state

    def _should_continue(self, state: MASXState) -> str:
        """Determine if workflow should continue."""
        # Implementation for continuation logic
        return "complete"

    def run_workflow(
        self, workflow_type: str = "daily", input_data: Optional[Dict[str, Any]] = None
    ) -> MASXState:
        """
        Run a complete workflow.

        Args:
            workflow_type: Type of workflow to run
            input_data: Optional input data for the workflow

        Returns:
            MASXState: Final workflow state
        """
        with measure_execution_time(f"{workflow_type}_workflow"):
            try:
                # Create workflow graph
                workflow_graph = self.create_workflow_graph(workflow_type)
                compiled_workflow = workflow_graph.compile()

                # Initialize state
                initial_state = MASXState(
                    workflow_id=generate_workflow_id(),
                    workflow=WorkflowState(),
                    metadata=input_data or {},
                    data={},
                )

                # Run workflow
                final_state = compiled_workflow.invoke(initial_state)

                self.logger.info(
                    f"{workflow_type} workflow completed",
                    run_id=final_state.workflow_id,
                    success=final_state.workflow.completed
                    and not final_state.workflow.failed,
                    error_count=len(final_state.errors),
                )

                return final_state

            except Exception as e:
                self.logger.error(f"Workflow execution failed: {e}")
                raise WorkflowException(f"Workflow execution failed: {str(e)}")

    def run_daily_workflow(
        self, input_data: Optional[Dict[str, Any]] = None
    ) -> MASXState:
        """Run the daily workflow."""
        return self.run_workflow("daily", input_data)

    def run_detection_workflow(
        self, input_data: Optional[Dict[str, Any]] = None
    ) -> MASXState:
        """Run the detection workflow."""
        return self.run_workflow("detection", input_data)

    def run_trigger_workflow(
        self, input_data: Optional[Dict[str, Any]] = None
    ) -> MASXState:
        """Run the trigger workflow."""
        return self.run_workflow("trigger", input_data)
