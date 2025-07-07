"""
Settings configuration.

Type safe configuration management using Pydantic Settings,
handling all environment variables and system configuration with proper validation.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Settings can be configured via environment variables or .env file.
    Type validation and defaults are handled automatically.
    """

    # Pydantic Settings to load .env file
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Environment Configuration
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    #HOTSPOT_QUERY
    hotspot_query: str = Field(default="global tension news last 24 hours", description="Hotspot query")

    # LLM API Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(
        default="gpt-4-turbo-preview", description="OpenAI model name"
    )
    openai_temperature: float = Field(default=0.0, description="OpenAI temperature")
    openai_max_tokens: int = Field(default=4000, description="OpenAI max tokens")

    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")
    mistral_model: str = Field(
        default="mistral-small", description="Mistral model name"
    )
    mistral_api_base: str = Field(
        default="https://api.mistral.ai/v1", description="Mistral API base URL"
    )
    mistral_temperature: float = Field(default=0.0, description="Mistral temperature")
    mistral_max_tokens: int = Field(default=4000, description="Mistral max tokens")
    mistral_token_ref: str = Field(default="mistral-small", description="Mistral token ref")

    # Database Configuration (Supabase)
    supabase_url: Optional[str] = Field(
        default=None, description="Supabase project URL"
    )
    supabase_anon_key: Optional[str] = Field(
        default=None, description="Supabase anonymous key"
    )
    supabase_service_role_key: Optional[str] = Field(
        default=None, description="Supabase service role key"
    )
    supabase_db_password: Optional[str] = Field(
        default=None, description="Supabase database password"
    )
    supabase_db_url: Optional[str] = Field(
        default=None, description="Supabase database URL"
    )
    database_max_connections: int = Field(
        default=10, description="Maximum number of database connections"
    )
    database_min_connections: int = Field(
        default=1, description="Minimum number of database connections"
    )

    # External API Keys
    google_search_api_key: Optional[str] = Field(
        default=None, description="Google Custom Search API key"
    )
    google_cx: Optional[str] = Field(
        default=None, description="Google Custom Search Engine ID"
    )
    
    google_search_base_url: Optional[str] = Field(
        default=None, description="Google search base URL"
    )        
        
    gdelt_api_key: Optional[str] = Field(default=None, description="GDELT API key")
    google_translate_api_key: Optional[str] = Field(
        default=None, description="Google Translate API key"
    )
    deepl_api_key: Optional[str] = Field(default=None, description="DeepL API key")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port number")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_secret_key: str = Field(
        default="change_this_in_production", description="API secret key for security"
    )

    # Scheduling Configuration
    daily_run_time: str = Field(default="00:00", description="Daily run time (HH:MM)")
    timezone: str = Field(default="UTC", description="System timezone")
    enable_scheduler: bool = Field(
        default=True, description="Enable internal scheduler"
    )

    # Feature Flags
    use_gdelt: bool = Field(default=True, description="Enable GDELT integration")
    use_translator: bool = Field(
        default=True, description="Enable translation services"
    )
    use_embeddings: bool = Field(
        default=True, description="Enable embedding generation"
    )
    use_fact_checking: bool = Field(default=True, description="Enable fact checking")
    use_parallel_processing: bool = Field(
        default=True, description="Enable parallel processing"
    )

    # Performance Configuration
    max_concurrent_requests: int = Field(
        default=10, description="Maximum concurrent requests"
    )
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: int = Field(default=2, description="Retry delay in seconds")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_memory_usage: float = Field(
        default=0.8, description="Maximum memory usage (0.0-1.0)"
    )

    # Security Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"], description="Allowed CORS origins"
    )
    enable_content_filtering: bool = Field(
        default=True, description="Enable content filtering"
    )
    max_content_length: int = Field(default=10000, description="Maximum content length")

    # Monitoring and Logging
    log_format: str = Field(default="json", description="Log format")
    log_file: str = Field(default="logs/masx.log", description="Log file path")
    log_rotation: str = Field(default="daily", description="Log rotation policy")
    log_retention: int = Field(default=30, description="Log retention in days")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")
    health_check_endpoint: str = Field(
        default="/health", description="Health check endpoint"
    )

    # Development Configuration
    enable_hot_reload: bool = Field(
        default=True, description="Enable hot reload in development"
    )
    enable_debug_toolbar: bool = Field(
        default=False, description="Enable debug toolbar"
    )
    enable_sql_logging: bool = Field(
        default=False, description="Enable SQL query logging"
    )
    test_database_url: Optional[str] = Field(
        default=None, description="Test database URL"
    )
    mock_external_apis: bool = Field(
        default=False, description="Mock external APIs in testing"
    )

    # Validators
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level setting."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @validator("openai_temperature", "mistral_temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature setting."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @validator("max_memory_usage")
    def validate_memory_usage(cls, v: float) -> float:
        """Validate memory usage setting."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Memory usage must be between 0.0 and 1.0")
        return v

    @validator("daily_run_time")
    def validate_run_time(cls, v: str) -> str:
        """Validate daily run time format."""
        try:
            hour, minute = map(int, v.split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError
        except (ValueError, AttributeError):
            raise ValueError("Daily run time must be in HH:MM format")
        return v

    # Computed Properties
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def has_openai_config(self) -> bool:
        """Check if OpenAI is properly configured."""
        return bool(self.openai_api_key)

    @property
    def has_mistral_config(self) -> bool:
        """Check if Mistral is properly configured."""
        return bool(self.mistral_api_key)

    @property
    def has_supabase_config(self) -> bool:
        """Check if Supabase is properly configured."""
        return all(
            [self.supabase_url, self.supabase_anon_key, self.supabase_service_role_key]
        )

    @property
    def primary_llm_provider(self) -> str:
        """Determine the primary LLM provider."""
        if self.has_mistral_config:
            return "mistral"
        elif self.has_openai_config:
            return "openai"
        else:
            raise ValueError("No LLM provider configured")

    def get_llm_config(self, provider: str = None) -> dict:
        """Get LLM configuration for specified provider."""
        if provider is None:
            provider = self.primary_llm_provider

        if provider == "openai":
            return {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": self.openai_temperature,
                "max_tokens": self.openai_max_tokens,
            }
        elif provider == "mistral":
            return {
                "api_key": self.mistral_api_key,
                "model": self.mistral_model,
                "api_base": self.mistral_api_base,
                "temperature": self.mistral_temperature,
                "max_tokens": self.mistral_max_tokens,
                "token_ref": self.mistral_token_ref,
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


@lru_cache()  # Least Recently Used
def get_settings() -> Settings:
    """
    Cached to avoid reloading settings on every call.
    Settings are loaded once and reused throughout the application lifecycle.
    """
    return Settings()


# Convenience function for getting settings in other modules
def get_config() -> Settings:
    """Alias for get_settings() for backward compatibility."""
    return get_settings()
