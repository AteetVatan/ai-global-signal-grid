"""
Flashpoint LLM Agent for Global Signal Grid (MASX) Agentic AI System.

Implements iterative flashpoint detection using LLM reasoning and web search.
Detects global geopolitical flashpoints through multi-domain analysis with
entity tracking and deduplication.

Usage: from app.agents.flashpoint_llm_agent import FlashpointLLMAgent
    agent = FlashpointLLMAgent()
    result = agent.run({"max_iterations": 5, "target_flashpoints": 10})
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, ValidationError, RootModel
from copy import deepcopy

from .base import BaseAgent, AgentResult
from ..core.state import AgentState
from ..core.exceptions import AgentException
from ..services.llm_service import LLMService
from ..services.web_search import WebSearchService
from ..services.flashpoint_detection import FlashpointDetectionService, Flashpoint
from ..services.token_tracker import get_token_tracker
from ..config.logging_config import get_logger


class FlashpointList(RootModel[List[Flashpoint]]):
    """Pydantic model for validating flashpoint lists."""


class FlashpointLLMAgent(BaseAgent):
    """
    Agent for detecting global geopolitical flashpoints using LLM reasoning.
    
    Features:
    - Iterative search and reasoning cycles
    - Entity tracking and deduplication
    - Multi-domain geopolitical analysis
    - JSON validation and error handling
    - Token cost tracking
    """

    # System prompt for flashpoint detection
    SYSTEM_PROMPT = (
        "You are a global strategic signal analyst.\n"
        "Your mission is to detect the top 10 most active or unstable global regions in the last 24 hours, "
        "based on multi-domain tension signals.\n\n"
        "Include ALL of the following domains:\n"
        "- Geopolitical, Military, Economic, Cultural, Religious, Tech, Cybersecurity, Environmental, Demographics, Sovereignty.\n\n"
        "Each output must include:\n"
        "- title: short phrase (e.g., 'US–China Chip War')\n"
        "- description: one factual sentence (≤200 chars)\n"
        "- entities: list of involved countries, organizations, regions, or non-state actors\n\n"
        "Output: JSON list of 10 dictionaries.\n"
        "NO extra text, bullets, or explanation. JUST clean JSON.\n"
        'Example: [{"title": "X", "description": "Y", "entities": ["Israel", "Iran"]}]'
    )

    USER_PROMPT = (
        "List 10 top global flashpoints from the past 24 hours with title, description, and involved entities "
        "(countries, regions, organizations, or non-state actors).\n\n"
        "Return only valid JSON."
    )

    def __init__(self):
        """Initialize FlashpointLLMAgent with required services."""
        super().__init__(
            name="flashpoint_llm_agent",
            description="Agent for detecting global geopolitical flashpoints using LLM reasoning"
        )
        
        # Initialize services
        self.llm_service = LLMService()
        self.web_search = WebSearchService()
        self.flashpoint_service = FlashpointDetectionService()
        self.token_tracker = get_token_tracker()
        
        # Get entity tracker
        self.entity_tracker = self.flashpoint_service.get_entity_tracker()

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute flashpoint detection workflow.
        
        Args:
            input_data: Input configuration and context
                - max_iterations: Maximum search-reason cycles (default: 5)
                - target_flashpoints: Target number of flashpoints (default: 10)
                - max_context_length: Max tokens for context (default: 15000)
                - context: Optional existing context
                
        Returns:
            AgentResult: Standardized result containing flashpoints and statistics
        """
        try:
            self.logger.info(
                "Starting flashpoint detection",
                input_data=input_data
            )
            
            # Extract configuration
            config = self._extract_config(input_data)
            
            # Execute flashpoint detection
            result = self._execute_flashpoint_detection(config)
            
            self.logger.info(
                "Flashpoint detection completed successfully",
                flashpoints_count=len(result.get("flashpoints", [])),
                iterations=result.get("iterations", 0)
            )
            
            return AgentResult(
                success=True,
                data=result,
                error=None,
                metadata={
                    "flashpoints_count": len(result.get("flashpoints", [])),
                    "iterations": result.get("iterations", 0),
                    "search_runs": result.get("search_runs", 0),
                    "llm_runs": result.get("llm_runs", 0)
                }
            )
            
        except Exception as e:
            # Handle errors
            error_msg = f"Flashpoint detection failed: {str(e)}"
            self.logger.error(
                "Flashpoint detection error",
                error=str(e),
                exc_info=True
            )
            
            return AgentResult(
                success=False,
                data={},
                error=error_msg,
                metadata={"exception_type": type(e).__name__}
            )

    def _extract_config(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate configuration from input data."""
        return {
            "max_iterations": input_data.get("max_iterations", 5),
            "target_flashpoints": input_data.get("target_flashpoints", 10),
            "max_context_length": input_data.get("max_context_length", 15000),
            "context": input_data.get("context", ""),
            "existing_flashpoints": input_data.get("existing_flashpoints", [])
        }

    def _execute_flashpoint_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the main flashpoint detection workflow.
        
        Args:
            config: Configuration parameters
            
        Returns:
            Dict containing flashpoints and statistics
        """
        accumulated_flashpoints = []
        iterations = 0
        
        while (iterations < config["max_iterations"] and 
               len(accumulated_flashpoints) < config["target_flashpoints"]):
            
            iterations += 1
            self.logger.info(
                f"Starting iteration {iterations}",
                current_flashpoints=len(accumulated_flashpoints),
                target=config["target_flashpoints"]
            )
            
            # Step 1: Fetch context
            context = self._fetch_context()
            if not context:
                self.logger.warning(
                    f"No context found in iteration {iterations}"
                )
                continue
            
            # Step 2: Generate flashpoints using LLM
            new_flashpoints = self._generate_flashpoints(context)
            if not new_flashpoints:
                self.logger.warning(
                    f"No flashpoints generated in iteration {iterations}"
                )
                continue
            
            # Step 3: Process and deduplicate flashpoints
            accumulated_flashpoints = self._process_flashpoints(
                new_flashpoints, accumulated_flashpoints
            )
            
            # Step 4: Add new flashpoints to accumulated list
            # for flashpoint in processed_flashpoints:
            #      if len(accumulated_flashpoints) >= config["target_flashpoints"]:
            #          break
            #      accumulated_flashpoints.append(flashpoint)
            
            self.logger.info(
                f"Iteration {iterations} completed",
                new_flashpoints=len(new_flashpoints),
                total_flashpoints=len(accumulated_flashpoints)
            )
        
        # Prepare final result
        result = {
            "flashpoints": [fp.model_dump() for fp in accumulated_flashpoints],
            "iterations": iterations,
            "search_runs": self.entity_tracker.search_run,
            "llm_runs": self.entity_tracker.llm_run,
            "token_usage": self.token_tracker.get_summary(),
            "entity_stats": self.entity_tracker.get_stats(),
            "geographic_distribution": self.flashpoint_service.get_geographic_distribution(
                accumulated_flashpoints
            )
        }
        
        return result

    def _fetch_context(self) -> str:
        """
        Fetch news context for flashpoint detection.
        
        Returns:
            str: Aggregated news context
        """
        self.entity_tracker.search_run += 1
        
        # Build search query with exclusions
        exclude_terms = self.entity_tracker.get_exclude_query()
        query = "global tension news last 24 hours " + exclude_terms
        
        self.logger.info(
            "[SearchAgent] Fetching news context...",
            search_run=self.entity_tracker.search_run,
            query=query
        )
        
        try:
            context = self.web_search.gather_context(query)
            return context
        except Exception as e:
            self.logger.error(
                "Context fetching failed",
                error=str(e)
            )
            return ""

    def _generate_flashpoints(self, context: str) -> List[Flashpoint]:
        """
        Generate flashpoints using LLM reasoning.
        
        Args:
            context: News context to analyze
            
        Returns:
            List of generated flashpoints
        """
        self.entity_tracker.llm_run += 1
        
        self.logger.info(
            "[LLMAgent] Generating flashpoints...",
            llm_run=self.entity_tracker.llm_run,
            context_length=len(context)
        )
        
        try:
            
            full_prompt = self.SYSTEM_PROMPT + self.USER_PROMPT + context
            
            # Count total tokens
            context_token_count = self.token_tracker.count_tokens(full_prompt)
            # het llm inputtoke
            llm_input_token = self.llm_service.max_tokens
            
            if context_token_count > llm_input_token:
                # summarize context
                pass
            
            self.logger.info(
                "Context token count",
                context_token_count=context_token_count
            )
            #if context token count is greater than 15000, truncate context
            
            
            # Generate response using LLM service
            response = self.llm_service.generate_text(
                prompt=self.USER_PROMPT + f"\n\nContext:\n{context[:15000]}",
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.0
            )
            
            # Validate JSON response
            validated_flashpoints = self._validate_json_response(response)
            
            if validated_flashpoints:
                self.logger.info(
                    "Flashpoints generated successfully",
                    flashpoints_count=len(validated_flashpoints)
                )
                return validated_flashpoints
            else:
                self.logger.warning(
                    "Invalid JSON response from LLM",
                    response_preview=response[:200]
                )
                return []
                
        except Exception as e:
            self.logger.error(
                "Flashpoint generation failed",
                error=str(e)
            )
            return []

    def _validate_json_response(self, response: str) -> Optional[List[Flashpoint]]:
        """
        Validate and parse JSON response from LLM.
        
        Args:
            response: Raw LLM response
            
        Returns:
            List of validated flashpoints or None if invalid
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            
            # Parse JSON
            data = json.loads(cleaned_response.strip())
            
            # Validate with Pydantic
            flashpoint_list = FlashpointList.model_validate(data)
            return flashpoint_list.root
            
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.warning(
                "JSON validation failed",
                error=str(e),
                response_preview=response[:200]
            )
            return None

    def _process_flashpoints(self, new_flashpoints: List[Flashpoint], 
                           accumulated_flashpoints: List[Flashpoint]) -> List[Flashpoint]:
        """
        Process and deduplicate flashpoints.
        
        Args:
            new_flashpoints: Newly generated flashpoints
            existing_flashpoints: Previously accumulated flashpoints
            
        Returns:
            List of processed flashpoints
        """
        processed = []
        
        existing_flashpoints = deepcopy(accumulated_flashpoints)
        
        for flashpoint in new_flashpoints:
            overlap_found = False
            
            # Check for overlap with existing flashpoints
            for existing in existing_flashpoints:
                if set(flashpoint.entities) & set(existing.entities):
                    # Merge overlapping flashpoints
                    existing.title += f" / {flashpoint.title}"
                    existing.description += f" {flashpoint.description}"
                    existing.entities = list(set(existing.entities + flashpoint.entities))
                    
                    # Update entity tracker
                    # check if flashpoint.entities is a country
                    geo_entities = self.get_geo_entities(flashpoint.entities)
                    self.entity_tracker.update_seen_entities(geo_entities)
                    
                    overlap_found = True
                    break
            
            # Add new flashpoint if no overlap and valid
            if not overlap_found and self.flashpoint_service.validate_flashpoint(flashpoint):
                geo_entities = self.get_geo_entities(flashpoint.entities)
                self.entity_tracker.add(flashpoint.entities, geo_entities)
                existing_flashpoints.append(flashpoint)
                
        processed = existing_flashpoints
        
        self.logger.info(
            "Flashpoint processing completed",
            new_flashpoints=len(new_flashpoints),
            processed_flashpoints=len(processed),
            duplicates_filtered=len(new_flashpoints) - len(processed)
        )
        
        return processed
    
    def get_geo_entities(self, entities: List[str]):
        geo_entities = []
        for entity in entities:
            if self.flashpoint_service.is_country(entity):
                geo_entities.append(entity)
        return geo_entities

    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive agent statistics.
        
        Returns:
            Dict containing agent statistics
        """
        return {
            "agent_name": self.name,
            "state": self.state.model_dump() if self.state else None,
            "entity_tracker": self.entity_tracker.get_stats(),
            "token_usage": self.token_tracker.get_summary(),
            "service_stats": self.flashpoint_service.get_service_stats()
        }

    def reset(self):
        """Reset agent state for new session."""
        self.entity_tracker.reset()
        self.flashpoint_service.reset()
        self._state = None
        
        self.logger.info("FlashpointLLMAgent reset") 