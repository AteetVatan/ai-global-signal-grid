"""
LLM service for Global Signal Grid (MASX) Agentic AI System.
Provides unified interface for LLM interactions with:
- Multiple provider support (OpenAI, Mistral)
- Retry logic with exponential backoff
- Token counting and cost tracking
- Structured output validation
Usage: from app.services.llm_service import LLMService
    llm_service = LLMService()
    result = llm_service.generate_text(prompt, temperature=0.0)
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

import tiktoken
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config.settings import get_settings
from ..core.exceptions import ExternalServiceException, ConfigurationException
from ..core.utils import retry_with_backoff, safe_json_loads
from ..config.logging_config import get_logger


class LLMService:
    """
    Service for LLM interactions with multiple provider support.
    Handles:
    - OpenAI and Mistral API integration
    - Retry logic and error handling
    - Token counting and cost tracking
    - Structured output generation
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize LLM service.
        Args: provider: LLM provider to use ('openai', 'mistral', or None for auto-detect)
        """
        self.settings = get_settings()
        self.logger = get_logger(__name__)

        # Determine provider
        if provider:
            self.provider = provider
        else:
            self.provider = self.settings.primary_llm_provider

        # Initialize provider-specific clients
        self._init_provider()

        # Token counting
        self.tokenizer = self.__safe_encoding_for_model()

        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    def _init_provider(self):
        """Initialize the selected LLM provider."""
        if self.provider == "openai":
            if not self.settings.has_openai_config:
                raise ConfigurationException("OpenAI not configured")

            config = self.settings.get_llm_config("openai")
            # self.client = OpenAI(api_key=config["api_key"]) #old way
            self.client = ChatOpenAI(
                model_name=config["model"],
                openai_api_key=config["api_key"],
                temperature=config["temperature"],
            )
            self.model = config["model"]
            self.temperature = config["temperature"]
            self.max_tokens = config["max_tokens"]

        elif self.provider == "mistral":
            if not self.settings.has_mistral_config:
                raise ConfigurationException("Mistral not configured")

            config = self.settings.get_llm_config("mistral")
            self.client = ChatOpenAI(
                model_name=config["model"],
                openai_api_base=config["api_base"],
                openai_api_key=config["api_key"],
                temperature=config["temperature"],
            )
            self.model = config["model"]
            self.temperature = config["temperature"]
            self.max_tokens = config["max_tokens"]
            self.token_ref = config["token_ref"]

        else:
            raise ConfigurationException(f"Unsupported LLM provider: {self.provider}")

    @retry_with_backoff(max_attempts=3, base_delay=1.0)
    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text using the configured LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Generated text response
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(
                    prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            elif self.provider == "mistral":
                return self._generate_mistral(
                    prompt, system_prompt, temperature, max_tokens, **kwargs
                )
            else:
                raise ConfigurationException(f"Unsupported provider: {self.provider}")

        except Exception as e:
            raise ExternalServiceException(
                f"LLM generation failed: {str(e)}", context={"provider": self.provider}
            )

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        )

        # Track usage
        self._track_usage(
            response.usage.prompt_tokens, response.usage.completion_tokens
        )

        return response.choices[0].message.content

    def _generate_mistral(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text using Mistral API."""
        messages = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.client.invoke(messages)

        # Note: Mistral doesn't provide token usage in the same way
        # We'll estimate based on text length
        estimated_tokens = len(prompt.split()) + len(response.content.split())
        self._track_usage(estimated_tokens // 2, estimated_tokens // 2)

        return response.content

    def generate_structured_output(
        self,
        prompt: str,
        output_schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate structured output in JSON format.

        Args:
            prompt: User prompt
            output_schema: Expected output schema
            system_prompt: Optional system prompt
            temperature: Generation temperature (0.0 for deterministic)
            **kwargs: Additional parameters

        Returns:
            Dict[str, Any]: Structured output matching the schema
        """
        # Add JSON formatting instructions to prompt
        json_prompt = f"""
        {prompt}

        Respond with ONLY valid JSON that matches this schema:
        {json.dumps(output_schema, indent=2)}

        Do not include any explanation or additional text.
        """

        response = self.generate_text(
            json_prompt, system_prompt=system_prompt, temperature=temperature, **kwargs
        )

        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]

            parsed = json.loads(cleaned_response.strip())
            return parsed

        except json.JSONDecodeError as e:
            raise ExternalServiceException(
                f"Failed to parse JSON response: {str(e)}",
                context={"response": response},
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            int: Number of tokens
        """
        return len(self.tokenizer.encode(text))

    def _track_usage(self, input_tokens: int, output_tokens: int):
        """Track token usage and calculate costs.
        embede actual cost of openAI and Mistral.
        """
        self.total_tokens += input_tokens + output_tokens

        # Calculate cost (approximate for Mistral)
        if self.provider == "openai":
            input_cost = (input_tokens / 1000) * 0.01  # $0.01 per 1K input tokens
            output_cost = (output_tokens / 1000) * 0.03  # $0.03 per 1K output tokens
            self.total_cost += input_cost + output_cost

        self.logger.debug(
            "Token usage tracked",
            provider=self.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=self.total_cost,
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        Returns:  Dict[str, Any]: Usage statistics
        """
        return {
            "provider": self.provider,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
        }

    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.total_tokens = 0
        self.total_cost = 0.0

    def __safe_encoding_for_model(self):
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            # fallback to default tokenizer
            if self.token_ref:
                return tiktoken.get_encoding(self.token_ref)
            else:
                return tiktoken.get_encoding(self.model)
