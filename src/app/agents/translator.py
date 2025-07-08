"""
Translator Agent for Global Signal Grid (MASX) Agentic AI System.

This agent is responsible for:
- Translating content between languages
- Managing translation quality
- Caching translations
- Handling translation errors
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseAgent, AgentResult
from ..services.translation import TranslationService
from ..core.state import AgentState
from ..core.exceptions import AgentException
from ..config.logging_config import get_agent_logger


class Translator(BaseAgent):
    """
    Translator Agent for translating content between languages.

    Responsibilities:
    - Translate content between languages
    - Manage translation quality
    - Cache translations for performance
    - Handle translation errors
    """

    def __init__(self):
        """Initialize the Translator agent."""
        super().__init__("Translator")
        self.translation_service = TranslationService()
        self.logger = get_agent_logger("Translator")

    async def translate_content(
        self, items: List[Dict[str, Any]], target_language: str = "en"
    ) -> AgentResult:
        """
        Translate content items to target language.

        Args:
            items: List of items to translate
            target_language: Target language for translation

        Returns:
            AgentResult: Contains translated content
        """
        try:
            self.logger.info(f"Translating {len(items)} items to {target_language}")

            translated_items = []
            failed_translations = []

            for item in items:
                try:
                    source_language = item.get("source_language", "auto")
                    content = item.get("content", item.get("title", ""))

                    if not content:
                        continue

                    # Translate the content
                    translated_text = self.translation_service.translate(
                        text=content,
                        source_lang=source_language,
                        target_lang=target_language,
                    )

                    translated_item = {
                        **item,
                        "translated_content": translated_text,
                        "translation_status": "success",
                        "target_language": target_language,
                    }
                    translated_items.append(translated_item)

                except Exception as e:
                    self.logger.warning(f"Translation failed for item: {e}")
                    failed_item = {
                        **item,
                        "translation_status": "failed",
                        "error": str(e),
                    }
                    failed_translations.append(failed_item)

            result = {
                "translated_items": translated_items,
                "failed_translations": failed_translations,
                "translation_stats": {
                    "total": len(items),
                    "successful": len(translated_items),
                    "failed": len(failed_translations),
                },
                "target_language": target_language,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.logger.info(
                "Translation completed",
                successful=len(translated_items),
                failed=len(failed_translations),
            )

            return AgentResult(
                success=True,
                data=result,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.utcnow(),
                    "target_language": target_language,
                },
            )

        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            raise AgentException(f"Translation failed: {str(e)}")

    async def batch_translate(
        self,
        texts: List[str],
        source_language: str = "auto",
        target_language: str = "en",
    ) -> AgentResult:
        """
        Translate a batch of texts.

        Args:
            texts: List of texts to translate
            source_language: Source language
            target_language: Target language

        Returns:
            AgentResult: Contains translated texts
        """
        try:
            self.logger.info(
                f"Batch translating {len(texts)} texts",
                source_language=source_language,
                target_language=target_language,
            )

            translated_texts = []

            for text in texts:
                try:
                    translated_text = self.translation_service.translate(
                        text=text,
                        source_lang=source_language,
                        target_lang=target_language,
                    )
                    translated_texts.append(translated_text)
                except Exception as e:
                    self.logger.warning(f"Failed to translate text: {e}")
                    translated_texts.append(text)  # Keep original on failure

            result = {
                "translated_texts": translated_texts,
                "original_texts": texts,
                "source_language": source_language,
                "target_language": target_language,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Batch translation failed: {e}")
            raise AgentException(f"Batch translation failed: {str(e)}")

    async def validate_translation(
        self,
        original_text: str,
        translated_text: str,
        source_language: str,
        target_language: str,
    ) -> AgentResult:
        """
        Validate translation quality.

        Args:
            original_text: Original text
            translated_text: Translated text
            source_language: Source language
            target_language: Target language

        Returns:
            AgentResult: Contains validation results
        """
        try:
            self.logger.info("Validating translation quality")

            # Simple validation checks
            validation_results = {
                "length_ratio": len(translated_text) / max(len(original_text), 1),
                "has_content": bool(translated_text.strip()),
                "language_appropriate": True,  # Placeholder
                "quality_score": 0.9,  # Placeholder
            }

            # Determine if translation is acceptable
            is_acceptable = (
                validation_results["length_ratio"] > 0.1
                and validation_results["has_content"]
                and validation_results["quality_score"] > 0.7
            )

            result = {
                "validation_results": validation_results,
                "is_acceptable": is_acceptable,
                "original_text": original_text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "timestamp": datetime.utcnow().isoformat(),
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Translation validation failed: {e}")
            raise AgentException(f"Translation validation failed: {str(e)}")

    async def get_supported_languages(self) -> AgentResult:
        """
        Get list of supported languages.

        Returns:
            AgentResult: Contains supported languages
        """
        try:
            self.logger.info("Getting supported languages")

            # Common supported languages
            supported_languages = [
                {"code": "en", "name": "English"},
                {"code": "es", "name": "Spanish"},
                {"code": "fr", "name": "French"},
                {"code": "de", "name": "German"},
                {"code": "zh", "name": "Chinese"},
                {"code": "ja", "name": "Japanese"},
                {"code": "ko", "name": "Korean"},
                {"code": "ar", "name": "Arabic"},
                {"code": "ru", "name": "Russian"},
                {"code": "pt", "name": "Portuguese"},
                {"code": "it", "name": "Italian"},
                {"code": "nl", "name": "Dutch"},
            ]

            result = {
                "supported_languages": supported_languages,
                "total_count": len(supported_languages),
                "timestamp": datetime.utcnow().isoformat(),
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Failed to get supported languages: {e}")
            raise AgentException(f"Failed to get supported languages: {str(e)}")

    async def detect_language(self, text: str) -> AgentResult:
        """
        Detect the language of text.

        Args:
            text: Text to analyze

        Returns:
            AgentResult: Contains language detection results
        """
        try:
            self.logger.info("Detecting language of text")

            # Simple language detection (in production, use proper service)
            detected_language = "en"  # Placeholder

            result = {
                "detected_language": detected_language,
                "confidence": 0.9,  # Placeholder
                "text_length": len(text),
                "timestamp": datetime.utcnow().isoformat(),
            }

            return AgentResult(
                success=True,
                data=result,
                metadata={"agent": self.name, "timestamp": datetime.utcnow()},
            )

        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            raise AgentException(f"Language detection failed: {str(e)}")
