"""
Entity Extractor Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Extracting entities (people, organizations, locations, etc.) from text
- Structuring extracted entities for downstream analysis
- Handling errors in entity extraction
"""

from typing import Dict, List, Any
from datetime import datetime

from .base import BaseAgent, AgentResult
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger
from ..core.singleton.spacy_singleton import SpacySingleton
from ..core.enums.spacy_model_name import SpaCyModelName
from ..core.querystate import QueryState
from ..core.country_normalizer import CountryNormalizer
from ..services import LanguageService


class LanguageAgent(BaseAgent):
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
        #self.nlp_eng = SpacySingleton.get(SpaCyModelName.EN_CORE_WEB_SM)
        self.country_normalizer = CountryNormalizer()
        
    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Synchronous entrypoint to extract entities (required by BaseAgent).
        
        Args:
            input_data: Dictionary with a list under 'items' key
        
        Returns:
            AgentResult with extracted entities
        """
        try:
            queries : List[QueryState] = input_data.get("queries", [])
            if not isinstance(queries, list):
                raise AgentException("Missing or invalid 'queries' list for entity extraction")

            self.logger.info(f"Extracting entities from {len(queries)} queries")

            for query in queries:
                entities = query.entities
                #get all languages associated with the entities
                languages = self._get_languages_from_entities(entities)
                query.language = languages

            self.logger.info("Language extraction completed", total_items=len(queries))

            result = {
                "queries": queries
            }
        
            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow()
                },
            )        
            

        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return AgentResult(
                success=False,
                error=f"Entity extraction failed: {str(e)}",
                metadata={"agent": self.name}
            )

    def _get_languages_from_entities(self, entities: List[str]) -> List[str]:
        """Get all languages associated with the entities."""
        try:
            language_list = []
            for entity in entities:
            #get the language of the entity
                country = self.country_normalizer.normalize(entity)
                #get the alpha2 code of the country
                if not country:
                    country = self.country_normalizer.get_coco_country_name(entity, return_all=True)
                
                
                if country:
                    alpha2 = self.country_normalizer.country_name_to_alpha2(country)            
                    if alpha2:
                        #get the language of the country
                        language = LanguageService.get_languages_for_country_code(alpha2)                
                        language_list.extend(language)
                        continue
                        
                #get the language of the entity
                language = LanguageService.get_languages_for_entity(entity)                
                language_list.extend(language)            
        except Exception as e:
            self.logger.error(f"Error getting languages from entities: {e}")
            return []
                

    
        #remove duplicates use dict key to remove duplicates
        language_list = list(dict.fromkeys(language_list))        
        #sort the languages
        language_list.sort()        
        #return the languages
        return language_list
    
    
    
    
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
    
    def __ensure_spacy_model(model_name: str):
        """
        Ensure a spaCy model is installed and return the loaded nlp object.
        
        Args:
            model_name: e.g. "en_core_web_sm"
        
        Returns:
            The loaded spaCy Language pipeline.
        """
        try:
            # Try to import the model package
            importlib.import_module(model_name)
        except ImportError:
            # If that fails, download it via the spacy CLI
            print(f"Model {model_name} not found. Downloading...")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True
            )
        # Finally, load and return the pipeline
        import spacy
        return spacy.load(model_name)
