"""Configuration management for LLM service and model refresh."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ModelRefreshConfig:
    """Configuration for model refresh operations."""

    # Feature toggles
    enable_model_refresh: bool = False
    enable_api_discovery: bool = True
    enable_price_scraping: bool = False

    # Safety settings
    require_manual_approval_threshold: float = 0.20  # 20% price change
    auto_update_threshold: float = 0.05  # 5% price change
    max_models_per_provider: int = 50

    # Rate limiting
    api_request_delay_seconds: float = 1.0
    max_retries: int = 3
    timeout_seconds: int = 30

    # Backup settings
    backup_directory: str = "/tmp/llm_model_backups"
    backup_retention_count: int = 10
    auto_backup_before_refresh: bool = True

    # Provider API credentials
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    google_project_id: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # Database connection
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "postgres"
    database_user: str = "postgres"
    database_password: str = "postgres"
    database_schema: str = "llmring"

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_environment(cls) -> "ModelRefreshConfig":
        """Create configuration from environment variables."""
        return cls(
            # Feature toggles
            enable_model_refresh=_get_bool_env("ENABLE_MODEL_REFRESH", False),
            enable_api_discovery=_get_bool_env("ENABLE_API_DISCOVERY", True),
            enable_price_scraping=_get_bool_env("ENABLE_PRICE_SCRAPING", False),
            # Safety settings
            require_manual_approval_threshold=_get_float_env(
                "MANUAL_APPROVAL_THRESHOLD", 0.20
            ),
            auto_update_threshold=_get_float_env("AUTO_UPDATE_THRESHOLD", 0.05),
            max_models_per_provider=_get_int_env("MAX_MODELS_PER_PROVIDER", 50),
            # Rate limiting
            api_request_delay_seconds=_get_float_env("API_REQUEST_DELAY", 1.0),
            max_retries=_get_int_env("MAX_RETRIES", 3),
            timeout_seconds=_get_int_env("TIMEOUT_SECONDS", 30),
            # Backup settings
            backup_directory=os.getenv("BACKUP_DIRECTORY", "/tmp/llm_model_backups"),
            backup_retention_count=_get_int_env("BACKUP_RETENTION_COUNT", 10),
            auto_backup_before_refresh=_get_bool_env(
                "AUTO_BACKUP_BEFORE_REFRESH", True
            ),
            # Provider API credentials
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_project_id=os.getenv("GOOGLE_PROJECT_ID"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            # Database connection
            database_host=os.getenv("DATABASE_HOST", "localhost"),
            database_port=_get_int_env("DATABASE_PORT", 5432),
            database_name=os.getenv("DATABASE_NAME", "postgres"),
            database_user=os.getenv("DATABASE_USER", "postgres"),
            database_password=os.getenv("DATABASE_PASSWORD", "postgres"),
            database_schema=os.getenv("DATABASE_SCHEMA", "llmring"),
            # Logging
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE"),
        )

    def get_database_connection_params(self) -> Dict[str, Any]:
        """Get database connection parameters as a dictionary."""
        return {
            "host": self.database_host,
            "port": self.database_port,
            "database": self.database_name,
            "user": self.database_user,
            "password": self.database_password,
        }

    def get_provider_credentials(self) -> Dict[str, Dict[str, Any]]:
        """Get provider API credentials as a dictionary."""
        return {
            "openai": {
                "api_key": self.openai_api_key,
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
            },
            "google": {
                "api_key": self.google_api_key,
                "project_id": self.google_project_id,
            },
            "ollama": {
                "base_url": self.ollama_base_url,
            },
        }

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.enable_model_refresh:
            if not self.enable_api_discovery and not self.enable_price_scraping:
                issues.append("Model refresh enabled but no discovery methods enabled")

        if self.enable_api_discovery:
            credentials = self.get_provider_credentials()
            if not any(
                [
                    credentials["openai"]["api_key"],
                    credentials["anthropic"]["api_key"],
                    credentials["google"]["api_key"],
                    # Ollama doesn't require API key
                ]
            ):
                issues.append(
                    "API discovery enabled but no provider credentials configured"
                )

        if self.auto_update_threshold >= self.require_manual_approval_threshold:
            issues.append(
                "Auto-update threshold must be less than manual approval threshold"
            )

        if self.backup_retention_count < 1:
            issues.append("Backup retention count must be at least 1")

        if self.max_models_per_provider < 1:
            issues.append("Max models per provider must be at least 1")

        return issues


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    else:
        return default


def _get_int_env(key: str, default: int) -> int:
    """Get integer environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# Global configuration instance
config = ModelRefreshConfig.from_environment()
