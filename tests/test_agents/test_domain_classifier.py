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
Unit tests for DomainClassifier agent.

Tests the domain classification functionality including:
- Agent initialization and capabilities
- Input validation and preprocessing
- Domain classification logic
- Error handling and edge cases
- LLM service integration (mocked)
"""

import pytest
from unittest.mock import Mock, patch

from app.agents.domain_classifier import DomainClassifier
from app.agents.base import AgentResult


class TestDomainClassifier:
    """Test cases for DomainClassifier agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DomainClassifier()

        assert agent.name == "DomainClassifier"
        assert "geopolitical" in agent.description.lower()
        assert len(agent.DOMAIN_CATEGORIES) == 12

    def test_capabilities(self):
        """Test agent capabilities."""
        agent = DomainClassifier()
        capabilities = agent.get_capabilities()

        assert capabilities["name"] == "DomainClassifier"
        assert capabilities["type"] == "DomainClassifier"
        assert capabilities["max_domains"] == 12
        assert capabilities["supports_uncategorized"] is True
        assert "Geopolitical" in capabilities["domains"]

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        agent = DomainClassifier()

        # Valid input with title
        assert agent.validate_input({"title": "Test title"}) is True

        # Valid input with description
        assert agent.validate_input({"description": "Test description"}) is True

        # Valid input with both
        assert (
            agent.validate_input(
                {"title": "Test title", "description": "Test description"}
            )
            is True
        )

    def test_validate_input_invalid(self):
        """Test input validation with invalid data."""
        agent = DomainClassifier()

        # Invalid input types
        assert agent.validate_input(None) is False
        assert agent.validate_input("string") is False
        assert agent.validate_input([]) is False

        # Empty content
        assert agent.validate_input({}) is False
        assert agent.validate_input({"title": "", "description": ""}) is False
        assert agent.validate_input({"title": "   ", "description": "   "}) is False

    def test_build_classification_prompt(self):
        """Test prompt building."""
        agent = DomainClassifier()

        title = "Test Title"
        description = "Test Description"

        prompt = agent._build_classification_prompt(title, description)

        assert title in prompt
        assert description in prompt
        assert "Categories:" in prompt
        assert "Geopolitical" in prompt
        assert "Respond with comma-separated categories" in prompt

    def test_parse_domains_single(self):
        """Test parsing single domain from response."""
        agent = DomainClassifier()

        response = "Geopolitical"
        domains = agent._parse_domains(response)

        assert domains == ["Geopolitical"]

    def test_parse_domains_multiple(self):
        """Test parsing multiple domains from response."""
        agent = DomainClassifier()

        response = "Geopolitical, Military / Conflict / Strategic Alliances"
        domains = agent._parse_domains(response)

        assert "Geopolitical" in domains
        assert "Military / Conflict / Strategic Alliances" in domains
        assert len(domains) == 2

    def test_parse_domains_uncategorized(self):
        """Test parsing uncategorized response."""
        agent = DomainClassifier()

        response = "Uncategorized"
        domains = agent._parse_domains(response)

        assert domains == ["Uncategorized"]

    def test_parse_domains_case_insensitive(self):
        """Test parsing domains with different cases."""
        agent = DomainClassifier()

        response = "geopolitical, MILITARY / CONFLICT / STRATEGIC ALLIANCES"
        domains = agent._parse_domains(response)

        assert "Geopolitical" in domains
        assert "Military / Conflict / Strategic Alliances" in domains

    def test_validate_domains_valid(self):
        """Test domain validation with valid domains."""
        agent = DomainClassifier()

        domains = ["Geopolitical", "Military / Conflict / Strategic Alliances"]
        validated = agent._validate_domains(domains)

        assert validated == domains

    def test_validate_domains_invalid(self):
        """Test domain validation with invalid domains."""
        agent = DomainClassifier()

        domains = [
            "Geopolitical",
            "Invalid Domain",
            "Military / Conflict / Strategic Alliances",
        ]
        validated = agent._validate_domains(domains)

        assert "Geopolitical" in validated
        assert "Military / Conflict / Strategic Alliances" in validated
        assert "Invalid Domain" not in validated
        assert len(validated) == 2

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        agent = DomainClassifier()

        # Single domain - high confidence
        assert agent._calculate_confidence(["Geopolitical"]) == 0.9

        # Two domains - medium confidence
        assert (
            agent._calculate_confidence(
                ["Geopolitical", "Military / Conflict / Strategic Alliances"]
            )
            == 0.7
        )

        # Three domains - lower confidence
        assert (
            agent._calculate_confidence(
                [
                    "Geopolitical",
                    "Military / Conflict / Strategic Alliances",
                    "Economic / Trade / Sanctions",
                ]
            )
            == 0.5
        )

        # Many domains - low confidence
        assert (
            agent._calculate_confidence(
                [
                    "Geopolitical",
                    "Military / Conflict / Strategic Alliances",
                    "Economic / Trade / Sanctions",
                    "Cultural / Identity Clashes",
                ]
            )
            == 0.3
        )

        # No domains - zero confidence
        assert agent._calculate_confidence([]) == 0.0

    @patch("app.agents.domain_classifier.LLMService")
    def test_execute_success(self, mock_llm_service):
        """Test successful execution."""
        # Mock LLM service
        mock_llm = Mock()
        mock_llm.generate_text.return_value = (
            "Geopolitical, Military / Conflict / Strategic Alliances"
        )
        mock_llm_service.return_value = mock_llm

        agent = DomainClassifier()

        input_data = {"title": "Test Title", "description": "Test Description"}

        result = agent.execute(input_data)

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert "domains" in result.data
        assert "Geopolitical" in result.data["domains"]
        assert "Military / Conflict / Strategic Alliances" in result.data["domains"]
        assert result.data["confidence"] == 0.7
        assert "input_length" in result.metadata

    @patch("app.agents.domain_classifier.LLMService")
    def test_execute_llm_error(self, mock_llm_service):
        """Test execution with LLM error."""
        # Mock LLM service to raise exception
        mock_llm = Mock()
        mock_llm.generate_text.side_effect = Exception("LLM API error")
        mock_llm_service.return_value = mock_llm

        agent = DomainClassifier()

        input_data = {"title": "Test Title", "description": "Test Description"}

        result = agent.execute(input_data)

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "LLM API error" in result.error
        assert result.metadata["exception_type"] == "Exception"

    def test_execute_invalid_input(self):
        """Test execution with invalid input."""
        agent = DomainClassifier()

        input_data = {}  # Invalid input

        result = agent.execute(input_data)

        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "missing title or description" in result.error

    @patch("app.agents.domain_classifier.LLMService")
    def test_run_with_logging(self, mock_llm_service):
        """Test running agent with full logging and state management."""
        # Mock LLM service
        mock_llm = Mock()
        mock_llm.generate_text.return_value = "Geopolitical"
        mock_llm_service.return_value = mock_llm

        agent = DomainClassifier()

        input_data = {"title": "Test Title", "description": "Test Description"}

        result = agent.run(input_data, run_id="test-run-123")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert agent.state is not None
        assert agent.state.name == "DomainClassifier"
        assert agent.state.status == "success"
        assert agent.state.input == input_data
