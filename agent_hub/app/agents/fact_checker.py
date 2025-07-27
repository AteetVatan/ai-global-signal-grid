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
Fact Checker Agent

This agent verifies critical facts and consistency across sources for identified events.
It acts as a failsafe to reduce misinformation by cross-verifying details.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.state import MASXState
from ..core.exceptions import AgentException, ValidationError
from ..services.llm_service import LLMService
from ..services.database import DatabaseService
from .base import BaseAgent


class FactChecker(BaseAgent):
    """
    Fact Checker Agent for verifying critical facts and consistency.

    This agent:
    - Cross-verifies details across multiple sources
    - Identifies factual discrepancies and inconsistencies
    - Provides confidence scores for claims
    - Flags potentially unreliable sources
    """

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        database_service: Optional[DatabaseService] = None,
    ):
        """Initialize the Fact Checker agent."""
        super().__init__("fact_checker")
        self.llm_service = LLMService.get_instance()  # singleton()
        self.database_service = database_service or DatabaseService()
        self.logger = logging.getLogger(__name__)

    def verify_facts(
        self, events: List[Dict[str, Any]], articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify facts across multiple sources for given events.

        Args:
            events: List of events to verify
            articles: List of articles containing source information

        Returns:
            Dict containing verification results
        """
        try:
            self.logger.info(f"Starting fact verification for {len(events)} events")

            verification_results = []

            for event in events:
                event_id = event.get("id", "unknown")
                self.logger.info(f"Verifying facts for event: {event_id}")

                # Extract claims from event
                claims = self._extract_claims(event)

                # Find related articles for this event
                related_articles = self._find_related_articles(event, articles)

                # Verify each claim
                verified_claims = []
                for claim in claims:
                    verification = self._verify_single_claim(claim, related_articles)
                    verified_claims.append(verification)

                # Check for inconsistencies
                inconsistencies = self._detect_inconsistencies(verified_claims)

                # Generate confidence score
                confidence_score = self._calculate_confidence(verified_claims)

                verification_results.append(
                    {
                        "event_id": event_id,
                        "verified_claims": verified_claims,
                        "inconsistencies": inconsistencies,
                        "confidence_score": confidence_score,
                        "verification_timestamp": datetime.utcnow().isoformat(),
                    }
                )

            return {
                "verified_events": verification_results,
                "total_events_verified": len(verification_results),
                "verification_status": "completed",
            }

        except Exception as e:
            self.logger.error(f"Error during fact verification: {str(e)}")
            raise AgentException(f"Fact verification failed: {str(e)}")

    def _extract_claims(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract factual claims from an event."""
        try:
            # Use LLM to extract claims from event summary
            prompt = f"""
            Extract factual claims from the following event summary. 
            Focus on specific, verifiable statements about:
            - Numbers (casualties, dates, amounts)
            - Locations
            - People and organizations
            - Actions and events
            
            Event: {event.get('summary', '')}
            
            Return a JSON list of claims with fields:
            - claim: the factual statement
            - type: number/location/person/action
            - confidence: high/medium/low
            """

            response = self.llm_service.generate_response(
                prompt=prompt, temperature=0.0, max_tokens=1000
            )

            claims = json.loads(response)
            self.logger.info(f"Extracted {len(claims)} claims from event")
            return claims

        except Exception as e:
            self.logger.warning(f"Failed to extract claims: {str(e)}")
            return []

    def _find_related_articles(
        self, event: Dict[str, Any], articles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find articles related to the event."""
        try:
            event_urls = event.get("article_urls", [])
            related = []

            for article in articles:
                if article.get("url") in event_urls:
                    related.append(article)

            self.logger.info(f"Found {len(related)} related articles for event")
            return related

        except Exception as e:
            self.logger.warning(f"Failed to find related articles: {str(e)}")
            return []

    def _verify_single_claim(
        self, claim: Dict[str, Any], articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify a single claim against multiple sources."""
        try:
            claim_text = claim.get("claim", "")
            claim_type = claim.get("type", "unknown")

            # Use LLM to verify claim against articles
            prompt = f"""
            Verify the following claim against the provided articles:
            
            Claim: {claim_text}
            Claim Type: {claim_type}
            
            Articles:
            {json.dumps([{"title": a.get("title", ""), "content": a.get("content", "")} for a in articles], indent=2)}
            
            Analyze the claim and return a JSON response with:
            - verified: true/false/uncertain
            - supporting_sources: list of article indices that support the claim
            - contradicting_sources: list of article indices that contradict the claim
            - confidence: high/medium/low
            - reasoning: brief explanation of verification result
            """

            response = self.llm_service.generate_response(
                prompt=prompt, temperature=0.0, max_tokens=500
            )

            verification = json.loads(response)
            verification["original_claim"] = claim

            return verification

        except Exception as e:
            self.logger.warning(f"Failed to verify claim: {str(e)}")
            return {
                "verified": "uncertain",
                "supporting_sources": [],
                "contradicting_sources": [],
                "confidence": "low",
                "reasoning": f"Verification failed: {str(e)}",
                "original_claim": claim,
            }

    def _detect_inconsistencies(
        self, verified_claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect inconsistencies between verified claims."""
        try:
            inconsistencies = []

            for i, claim1 in enumerate(verified_claims):
                for j, claim2 in enumerate(verified_claims[i + 1 :], i + 1):
                    if self._claims_contradict(claim1, claim2):
                        inconsistencies.append(
                            {
                                "claim1": claim1.get("original_claim", {}).get(
                                    "claim", ""
                                ),
                                "claim2": claim2.get("original_claim", {}).get(
                                    "claim", ""
                                ),
                                "reason": "Contradicting information detected",
                            }
                        )

            self.logger.info(f"Detected {len(inconsistencies)} inconsistencies")
            return inconsistencies

        except Exception as e:
            self.logger.warning(f"Failed to detect inconsistencies: {str(e)}")
            return []

    def _claims_contradict(
        self, claim1: Dict[str, Any], claim2: Dict[str, Any]
    ) -> bool:
        """Check if two claims contradict each other."""
        try:
            # Simple contradiction detection based on verification results
            verified1 = claim1.get("verified", "uncertain")
            verified2 = claim2.get("verified", "uncertain")

            # If one is verified true and the other is verified false
            if (verified1 == "true" and verified2 == "false") or (
                verified1 == "false" and verified2 == "true"
            ):
                return True

            # Check for contradicting sources
            supporting1 = set(claim1.get("supporting_sources", []))
            contradicting2 = set(claim2.get("contradicting_sources", []))

            if supporting1.intersection(contradicting2):
                return True

            return False

        except Exception as e:
            self.logger.warning(f"Failed to check claim contradiction: {str(e)}")
            return False

    def _calculate_confidence(self, verified_claims: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the event."""
        try:
            if not verified_claims:
                return 0.0

            total_claims = len(verified_claims)
            verified_count = sum(
                1 for c in verified_claims if c.get("verified") == "true"
            )
            uncertain_count = sum(
                1 for c in verified_claims if c.get("verified") == "uncertain"
            )

            # Calculate confidence based on verification ratio
            confidence = (verified_count / total_claims) * 0.8 + (
                uncertain_count / total_claims
            ) * 0.4

            return round(confidence, 2)

        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence: {str(e)}")
            return 0.0

    def cross_reference_with_memory(
        self, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Cross-reference events with historical data in memory."""
        try:
            self.logger.info("Cross-referencing events with historical data")

            cross_reference_results = []

            for event in events:
                event_summary = event.get("summary", "")

                # Search for similar events in database
                similar_events = self.database_service.search_similar_events(
                    event_summary, limit=5
                )

                cross_reference_results.append(
                    {
                        "event_id": event.get("id", "unknown"),
                        "similar_historical_events": len(similar_events),
                        "historical_consistency": self._check_historical_consistency(
                            event, similar_events
                        ),
                    }
                )

            return {
                "cross_reference_results": cross_reference_results,
                "total_events_checked": len(cross_reference_results),
            }

        except Exception as e:
            self.logger.error(f"Error during cross-referencing: {str(e)}")
            raise AgentException(f"Cross-referencing failed: {str(e)}")

    def _check_historical_consistency(
        self, current_event: Dict[str, Any], historical_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check consistency with historical events."""
        try:
            if not historical_events:
                return {"consistent": True, "reason": "No historical data available"}

            # Use LLM to check consistency
            prompt = f"""
            Check if the current event is consistent with historical events:
            
            Current Event: {current_event.get('summary', '')}
            
            Historical Events:
            {json.dumps([e.get('summary', '') for e in historical_events], indent=2)}
            
            Return JSON with:
            - consistent: true/false
            - reason: explanation of consistency check
            - historical_pattern: any pattern detected
            """

            response = self.llm_service.generate_response(
                prompt=prompt, temperature=0.0, max_tokens=300
            )

            return json.loads(response)

        except Exception as e:
            self.logger.warning(f"Failed to check historical consistency: {str(e)}")
            return {"consistent": True, "reason": "Consistency check failed"}

    def execute(self, state: MASXState) -> MASXState:
        """Execute the fact checking workflow."""
        try:
            self.logger.info("Starting fact checker execution")

            # Get events and articles from state
            events = state.workflow.get("events", [])
            articles = state.workflow.get("articles", [])

            if not events:
                self.logger.warning("No events to verify")
                return state

            # Verify facts
            verification_results = self.verify_facts(events, articles)

            # Cross-reference with memory
            cross_reference_results = self.cross_reference_with_memory(events)

            # Update state
            state.agents[self.name] = {
                "status": "completed",
                "output": {
                    "verification_results": verification_results,
                    "cross_reference_results": cross_reference_results,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            self.logger.info("Fact checker execution completed successfully")
            return state

        except Exception as e:
            self.logger.error(f"Fact checker execution failed: {str(e)}")
            state.agents[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            return state
