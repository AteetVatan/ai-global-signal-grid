"""
Entity Extractor Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Extracting entities (people, organizations, locations, etc.) from text
- Structuring extracted entities for downstream analysis
- Handling errors in entity extraction
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseAgent, AgentResult
from ..core.state import AgentState
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger


class EntityExtractor(BaseAgent):
    """
    Entity Extractor Agent for extracting entities from text content.

    Responsibilities:
    - Extract entities (people, organizations, locations, etc.)
    - Structure extracted entities for downstream analysis
    - Handle errors in entity extraction
    """

    def __init__(self):
        """Initialize the Entity Extractor agent."""
        super().__init__("EntityExtractor")
        self.logger = get_agent_logger("EntityExtractor")

    async def extract_entities(self, items: List[Dict[str, Any]]) -> AgentResult:
        """
        Extract entities from a list of content items.

        Args:
            items: List of content items to analyze

        Returns:
            AgentResult: Contains extracted entities
        """
        try:
            self.logger.info(f"Extracting entities from {len(items)} items")

            extracted_results = []

            for item in items:
                text = item.get("content", item.get("title", ""))
                entities = self._extract_entities_from_text(text)

                extracted_results.append(
                    {
                        "item_id": item.get("id", ""),
                        "entities": entities,
                        "original_item": item,
                    }
                )

            result = {
                "extracted_entities": extracted_results,
                "total_items": len(items),
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.logger.info("Entity extraction completed", total_items=len(items))

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            raise AgentException(f"Entity extraction failed: {str(e)}")

    def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Stub entity extraction from text (replace with real NER in production)."""
        # In production, use spaCy or another NER library
        # Here, return a stub list
        if not text:
            return []

        # Simple stub: extract capitalized words as entities
        words = text.split()
        entities = []
        for word in words:
            if word.istitle() and len(word) > 2:
                entities.append({"text": word, "type": "UNKNOWN"})
        return entities
