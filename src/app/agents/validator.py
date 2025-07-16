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
Validator Agent

This agent ensures the final data output conforms to the expected schema and quality standards.
It validates JSON structure, URL validity, and business rules compliance.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError as PydanticValidationError
from pydantic import HttpUrl, validator

from ..core.state import MASXState
from ..core.exceptions import AgentException, ValidationError
from .base import BaseAgent


class HotspotSchema(BaseModel):
    """Schema for hotspot validation."""

    id: str
    title: str
    summary: str
    article_urls: List[HttpUrl]
    entities: List[Dict[str, Any]]
    confidence_score: float
    created_at: str

    @validator("confidence_score")
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v

    @validator("article_urls")
    def validate_urls(cls, v):
        if not v:
            raise ValueError("At least one article URL is required")
        return v


class ArticleSchema(BaseModel):
    """Schema for article validation."""

    url: HttpUrl
    title: str
    content: str
    source: str
    published_at: str
    language: str = "en"

    @validator("title")
    def validate_title(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Title must be at least 5 characters long")
        return v.strip()

    @validator("content")
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        return v.strip()


class Validator(BaseAgent):
    """
    Validator Agent for ensuring data quality and schema compliance.

    This agent:
    - Validates JSON structure against Pydantic schemas
    - Checks URL validity and accessibility
    - Ensures business rules compliance
    - Validates data types and formats
    - Performs consistency checks
    """

    def __init__(self, timeout: int = 10):
        """Initialize the Validator agent."""
        super().__init__("validator")
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def validate_hotspots(self, hotspots: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate hotspot data against schema.

        Args:
            hotspots: List of hotspot dictionaries to validate

        Returns:
            Dict containing validation results
        """
        try:
            self.logger.info(f"Validating {len(hotspots)} hotspots")

            validation_results = []
            valid_hotspots = []
            invalid_hotspots = []

            for i, hotspot in enumerate(hotspots):
                try:
                    # Validate against schema
                    validated_hotspot = HotspotSchema(**hotspot)
                    valid_hotspots.append(validated_hotspot.dict())

                    validation_results.append(
                        {
                            "index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "valid": True,
                            "errors": [],
                        }
                    )

                except PydanticValidationError as e:
                    errors = [
                        {"field": error["loc"][0], "message": error["msg"]}
                        for error in e.errors()
                    ]

                    invalid_hotspots.append(
                        {"index": i, "hotspot": hotspot, "errors": errors}
                    )

                    validation_results.append(
                        {
                            "index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "valid": False,
                            "errors": errors,
                        }
                    )

            return {
                "valid_hotspots": valid_hotspots,
                "invalid_hotspots": invalid_hotspots,
                "validation_results": validation_results,
                "total_hotspots": len(hotspots),
                "valid_count": len(valid_hotspots),
                "invalid_count": len(invalid_hotspots),
            }

        except Exception as e:
            self.logger.error(f"Error during hotspot validation: {str(e)}")
            raise AgentException(f"Hotspot validation failed: {str(e)}")

    def validate_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate article data against schema.

        Args:
            articles: List of article dictionaries to validate

        Returns:
            Dict containing validation results
        """
        try:
            self.logger.info(f"Validating {len(articles)} articles")

            validation_results = []
            valid_articles = []
            invalid_articles = []

            for i, article in enumerate(articles):
                try:
                    # Validate against schema
                    validated_article = ArticleSchema(**article)
                    valid_articles.append(validated_article.dict())

                    validation_results.append(
                        {
                            "index": i,
                            "url": article.get("url", ""),
                            "valid": True,
                            "errors": [],
                        }
                    )

                except PydanticValidationError as e:
                    errors = [
                        {"field": error["loc"][0], "message": error["msg"]}
                        for error in e.errors()
                    ]

                    invalid_articles.append(
                        {"index": i, "article": article, "errors": errors}
                    )

                    validation_results.append(
                        {
                            "index": i,
                            "url": article.get("url", ""),
                            "valid": False,
                            "errors": errors,
                        }
                    )

            return {
                "valid_articles": valid_articles,
                "invalid_articles": invalid_articles,
                "validation_results": validation_results,
                "total_articles": len(articles),
                "valid_count": len(valid_articles),
                "invalid_count": len(invalid_articles),
            }

        except Exception as e:
            self.logger.error(f"Error during article validation: {str(e)}")
            raise AgentException(f"Article validation failed: {str(e)}")

    def validate_urls(self, urls: List[str]) -> Dict[str, Any]:
        """
        Validate URL accessibility and format.

        Args:
            urls: List of URLs to validate

        Returns:
            Dict containing URL validation results
        """
        try:
            self.logger.info(f"Validating {len(urls)} URLs")

            validation_results = []
            accessible_urls = []
            inaccessible_urls = []

            for i, url in enumerate(urls):
                try:
                    # Check URL format
                    parsed_url = urlparse(url)
                    if not parsed_url.scheme or not parsed_url.netloc:
                        raise ValueError("Invalid URL format")

                    # Check URL accessibility
                    response = requests.head(
                        url, timeout=self.timeout, allow_redirects=True
                    )

                    if response.status_code < 400:
                        accessible_urls.append(url)
                        validation_results.append(
                            {
                                "index": i,
                                "url": url,
                                "accessible": True,
                                "status_code": response.status_code,
                                "error": None,
                            }
                        )
                    else:
                        inaccessible_urls.append(url)
                        validation_results.append(
                            {
                                "index": i,
                                "url": url,
                                "accessible": False,
                                "status_code": response.status_code,
                                "error": f"HTTP {response.status_code}",
                            }
                        )

                except Exception as e:
                    inaccessible_urls.append(url)
                    validation_results.append(
                        {
                            "index": i,
                            "url": url,
                            "accessible": False,
                            "status_code": None,
                            "error": str(e),
                        }
                    )

            return {
                "accessible_urls": accessible_urls,
                "inaccessible_urls": inaccessible_urls,
                "validation_results": validation_results,
                "total_urls": len(urls),
                "accessible_count": len(accessible_urls),
                "inaccessible_count": len(inaccessible_urls),
            }

        except Exception as e:
            self.logger.error(f"Error during URL validation: {str(e)}")
            raise AgentException(f"URL validation failed: {str(e)}")

    def check_business_rules(
        self, hotspots: List[Dict[str, Any]], articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check business rules compliance.

        Args:
            hotspots: List of validated hotspots
            articles: List of validated articles

        Returns:
            Dict containing business rule validation results
        """
        try:
            self.logger.info("Checking business rules compliance")

            rule_violations = []

            # Rule 1: No empty hotspots
            for i, hotspot in enumerate(hotspots):
                if not hotspot.get("article_urls"):
                    rule_violations.append(
                        {
                            "rule": "no_empty_hotspots",
                            "hotspot_index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "message": "Hotspot has no associated articles",
                        }
                    )

            # Rule 2: No duplicate URLs across hotspots
            all_urls = []
            for hotspot in hotspots:
                all_urls.extend(hotspot.get("article_urls", []))

            duplicate_urls = [url for url in set(all_urls) if all_urls.count(url) > 1]
            if duplicate_urls:
                rule_violations.append(
                    {
                        "rule": "no_duplicate_urls",
                        "duplicate_urls": duplicate_urls,
                        "message": f"Found {len(duplicate_urls)} duplicate URLs across hotspots",
                    }
                )

            # Rule 3: All URLs must be accessible
            url_validation = self.validate_urls(all_urls)
            if url_validation["inaccessible_count"] > 0:
                rule_violations.append(
                    {
                        "rule": "all_urls_accessible",
                        "inaccessible_urls": url_validation["inaccessible_urls"],
                        "message": f"Found {url_validation['inaccessible_count']} inaccessible URLs",
                    }
                )

            # Rule 4: Minimum confidence threshold
            min_confidence = 0.3
            low_confidence_hotspots = []
            for i, hotspot in enumerate(hotspots):
                if hotspot.get("confidence_score", 0) < min_confidence:
                    low_confidence_hotspots.append(
                        {
                            "index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "confidence": hotspot.get("confidence_score", 0),
                        }
                    )

            if low_confidence_hotspots:
                rule_violations.append(
                    {
                        "rule": "minimum_confidence_threshold",
                        "low_confidence_hotspots": low_confidence_hotspots,
                        "message": f"Found {len(low_confidence_hotspots)} hotspots below confidence threshold",
                    }
                )

            # Rule 5: Date format consistency
            for i, hotspot in enumerate(hotspots):
                created_at = hotspot.get("created_at", "")
                try:
                    datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    rule_violations.append(
                        {
                            "rule": "valid_date_format",
                            "hotspot_index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "message": f"Invalid date format: {created_at}",
                        }
                    )

            return {
                "rule_violations": rule_violations,
                "total_violations": len(rule_violations),
                "compliant": len(rule_violations) == 0,
            }

        except Exception as e:
            self.logger.error(f"Error during business rule validation: {str(e)}")
            raise AgentException(f"Business rule validation failed: {str(e)}")

    def validate_data_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data types and formats.

        Args:
            data: Data dictionary to validate

        Returns:
            Dict containing type validation results
        """
        try:
            self.logger.info("Validating data types and formats")

            type_errors = []

            # Check required fields exist and have correct types
            required_fields = {
                "hotspots": list,
                "articles": list,
                "workflow_id": str,
                "timestamp": str,
            }

            for field, expected_type in required_fields.items():
                if field not in data:
                    type_errors.append(
                        {
                            "field": field,
                            "error": "Missing required field",
                            "expected_type": expected_type.__name__,
                        }
                    )
                elif not isinstance(data[field], expected_type):
                    type_errors.append(
                        {
                            "field": field,
                            "error": f"Expected {expected_type.__name__}, got {type(data[field]).__name__}",
                            "expected_type": expected_type.__name__,
                            "actual_type": type(data[field]).__name__,
                        }
                    )

            # Validate nested structures
            if "hotspots" in data and isinstance(data["hotspots"], list):
                for i, hotspot in enumerate(data["hotspots"]):
                    if not isinstance(hotspot, dict):
                        type_errors.append(
                            {
                                "field": f"hotspots[{i}]",
                                "error": "Expected dict, got list",
                                "expected_type": "dict",
                                "actual_type": type(hotspot).__name__,
                            }
                        )

            if "articles" in data and isinstance(data["articles"], list):
                for i, article in enumerate(data["articles"]):
                    if not isinstance(article, dict):
                        type_errors.append(
                            {
                                "field": f"articles[{i}]",
                                "error": "Expected dict, got list",
                                "expected_type": "dict",
                                "actual_type": type(article).__name__,
                            }
                        )

            return {
                "type_errors": type_errors,
                "total_errors": len(type_errors),
                "valid": len(type_errors) == 0,
            }

        except Exception as e:
            self.logger.error(f"Error during data type validation: {str(e)}")
            raise AgentException(f"Data type validation failed: {str(e)}")

    def perform_consistency_check(
        self, hotspots: List[Dict[str, Any]], articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform consistency checks across the dataset.

        Args:
            hotspots: List of hotspots
            articles: List of articles

        Returns:
            Dict containing consistency check results
        """
        try:
            self.logger.info("Performing consistency checks")

            consistency_issues = []

            # Check 1: All hotspot URLs exist in articles
            all_article_urls = {article.get("url", "") for article in articles}
            for i, hotspot in enumerate(hotspots):
                hotspot_urls = set(hotspot.get("article_urls", []))
                missing_urls = hotspot_urls - all_article_urls
                if missing_urls:
                    consistency_issues.append(
                        {
                            "type": "missing_article_urls",
                            "hotspot_index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "missing_urls": list(missing_urls),
                        }
                    )

            # Check 2: Entity consistency
            for i, hotspot in enumerate(hotspots):
                entities = hotspot.get("entities", [])
                if entities and not isinstance(entities, list):
                    consistency_issues.append(
                        {
                            "type": "invalid_entity_format",
                            "hotspot_index": i,
                            "hotspot_id": hotspot.get("id", f"hotspot_{i}"),
                            "message": "Entities must be a list",
                        }
                    )

            # Check 3: Language consistency
            languages = {article.get("language", "en") for article in articles}
            if len(languages) > 1:
                consistency_issues.append(
                    {
                        "type": "mixed_languages",
                        "languages": list(languages),
                        "message": "Articles contain mixed languages",
                    }
                )

            return {
                "consistency_issues": consistency_issues,
                "total_issues": len(consistency_issues),
                "consistent": len(consistency_issues) == 0,
            }

        except Exception as e:
            self.logger.error(f"Error during consistency check: {str(e)}")
            raise AgentException(f"Consistency check failed: {str(e)}")

    def execute(self, state: MASXState) -> MASXState:
        """Execute the validation workflow."""
        try:
            self.logger.info("Starting validator execution")

            # Get data from state
            hotspots = state.workflow.get("hotspots", [])
            articles = state.workflow.get("articles", [])

            validation_results = {}

            # Validate hotspots
            if hotspots:
                validation_results["hotspots"] = self.validate_hotspots(hotspots)

            # Validate articles
            if articles:
                validation_results["articles"] = self.validate_articles(articles)

            # Check business rules
            if hotspots and articles:
                validation_results["business_rules"] = self.check_business_rules(
                    validation_results.get("hotspots", {}).get("valid_hotspots", []),
                    validation_results.get("articles", {}).get("valid_articles", []),
                )

            # Validate data types
            workflow_data = {
                "hotspots": hotspots,
                "articles": articles,
                "workflow_id": state.workflow.get("workflow_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            }
            validation_results["data_types"] = self.validate_data_types(workflow_data)

            # Perform consistency checks
            if hotspots and articles:
                validation_results["consistency"] = self.perform_consistency_check(
                    validation_results.get("hotspots", {}).get("valid_hotspots", []),
                    validation_results.get("articles", {}).get("valid_articles", []),
                )

            # Determine overall validation status
            overall_valid = all(
                result.get("valid", True) if isinstance(result, dict) else True
                for result in validation_results.values()
            )

            # Update state
            state.agents[self.name] = {
                "status": "completed",
                "output": {
                    "validation_results": validation_results,
                    "overall_valid": overall_valid,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

            self.logger.info("Validator execution completed successfully")
            return state

        except Exception as e:
            self.logger.error(f"Validator execution failed: {str(e)}")
            state.agents[self.name] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
            return state
