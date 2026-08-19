"""Registry client for fetching model metadata from GitHub Pages. Provides caching, version management, and drift detection for provider models."""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from cachetools import TTLCache
from pydantic import BaseModel, Field, field_validator

from llmring.exceptions import RegistryStaleError
from llmring.validation import InputValidator

logger = logging.getLogger(__name__)


def _parse_iso_datetime(value: str) -> datetime:
    """Parse ISO 8601 datetime string, handling 'Z' suffix for UTC.

    Python 3.10's fromisoformat() doesn't handle 'Z' (added in 3.11).
    """
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


class RegistryModel(BaseModel):
    """Model information from the registry."""

    provider: str = Field(..., description="Provider name")
    model_name: str = Field(..., description="Model identifier")
    display_name: str = Field(..., description="Human-friendly name")
    description: Optional[str] = Field(None, description="Model description")
    model_aliases: Optional[List[str]] = Field(None, description="Alternative names for the model")

    # Token limits
    max_input_tokens: Optional[int] = Field(None, description="Max input tokens")
    max_output_tokens: Optional[int] = Field(None, description="Max output tokens")

    # Pricing (dollars per million tokens)
    dollars_per_million_tokens_input: Optional[float] = Field(
        None, description="USD per million input tokens"
    )
    dollars_per_million_tokens_output: Optional[float] = Field(
        None, description="USD per million output tokens"
    )

    # Capabilities
    supports_vision: bool = Field(False, description="Supports image input")
    supports_function_calling: bool = Field(False, description="Supports functions")
    supports_json_mode: bool = Field(False, description="Supports JSON output")
    supports_parallel_tool_calls: bool = Field(False, description="Supports parallel tools")
    supports_temperature: bool = Field(True, description="Supports temperature parameter")
    supports_streaming: bool = Field(True, description="Supports streaming responses")
    supports_caching: bool = Field(False, description="Supports prompt caching")

    # Reasoning model support
    is_reasoning_model: bool = Field(
        False,
        description="Model uses reasoning tokens for internal thinking before generating output",
    )
    min_recommended_reasoning_tokens: Optional[int] = Field(
        None,
        description="Typical number of reasoning tokens this model uses before generating output. "
        "For reasoning models, max_completion_tokens should be at least this value + desired output tokens.",
    )
    supports_thinking: bool = Field(False, description="Model supports dedicated thinking tokens")
    dollars_per_million_tokens_output_thinking: Optional[float] = Field(
        None, description="USD per million thinking tokens"
    )

    # API routing hints
    api_endpoint: Optional[
        Literal["chat", "responses", "assistants", "generateContent", "embeddings"]
    ] = Field(
        None,
        description="Preferred API endpoint (chat/responses/assistants/generateContent/embeddings)",
    )

    @field_validator("api_endpoint")
    @classmethod
    def validate_api_endpoint(cls, v):
        """Validate api_endpoint is one of the allowed values."""
        if v is not None and v not in [
            "chat",
            "responses",
            "assistants",
            "generateContent",
            "embeddings",
        ]:
            raise ValueError(
                "Invalid api_endpoint: "
                f"{v}. Must be 'chat', 'responses', 'assistants', 'generateContent', or 'embeddings'"
            )
        return v

    # Status
    is_active: bool = Field(True, description="Model is currently available")
    added_date: Optional[datetime] = Field(None, description="When model was added")
    deprecated_date: Optional[datetime] = Field(None, description="When model was deprecated")

    # Cache pricing (provider-specific)
    dollars_per_million_tokens_cached_input: Optional[float] = Field(
        None, description="USD per million cached input tokens"
    )
    dollars_per_million_tokens_cache_write_5m: Optional[float] = Field(
        None, description="USD per million cache write tokens (5 minute TTL)"
    )
    dollars_per_million_tokens_cache_write_1h: Optional[float] = Field(
        None, description="USD per million cache write tokens (1 hour TTL)"
    )
    dollars_per_million_tokens_cache_read: Optional[float] = Field(
        None, description="USD per million cache read tokens"
    )
    cache_storage_cost_per_million_tokens_per_hour: Optional[float] = Field(
        None, description="USD per million cache storage tokens per hour"
    )

    # Long context pricing
    supports_long_context_pricing: bool = Field(
        False, description="Model offers separate long-context pricing tiers"
    )
    long_context_threshold_tokens: Optional[int] = Field(
        None, description="Token threshold where long-context pricing applies"
    )
    dollars_per_million_tokens_input_long_context: Optional[float] = Field(
        None, description="USD per million input tokens beyond the long-context threshold"
    )
    dollars_per_million_tokens_output_long_context: Optional[float] = Field(
        None, description="USD per million output tokens beyond the long-context threshold"
    )


class RegistryVersion(BaseModel):
    """Registry version information."""

    provider: str = Field(..., description="Provider name")
    version: int = Field(..., description="Version number")
    updated_at: datetime = Field(..., description="Last update time")
    models: List[RegistryModel] = Field(default_factory=list, description="Models in this version")


class RegistrySourceInfo(BaseModel):
    """Provenance for one provider's slice of the registry."""

    provider: str
    version: Optional[int] = None
    updated_at: Optional[datetime] = None
    url: str
    from_cache: bool = False

    @property
    def age(self) -> Optional[timedelta]:
        if self.updated_at is None:
            return None
        updated = self.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - updated

    @property
    def age_days(self) -> Optional[float]:
        age = self.age
        return None if age is None else age.total_seconds() / 86400.0


class RegistryClient:
    """Client for fetching model information from the registry."""

    DEFAULT_REGISTRY_URL = "https://llmring.github.io/registry"
    CACHE_DIR = Path.home() / ".cache" / "llmring" / "registry"
    CACHE_DURATION_HOURS = 24

    def __init__(self, registry_url: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize the registry client.

        Args:
            registry_url: Base URL for the registry (defaults to GitHub Pages)
            cache_dir: Directory for caching registry data

        Raises:
            ValueError: If registry URL is invalid or potentially unsafe
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL

        # Validate registry URL for security
        InputValidator.validate_registry_url(self.registry_url)

        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache with TTL (24 hours)
        cache_ttl = self.CACHE_DURATION_HOURS * 3600  # Convert to seconds
        self._cache: TTLCache = TTLCache(maxsize=100, ttl=cache_ttl)
        self._pinned_version: Optional[int] = None
        # Provenance for whatever payload we last parsed, per provider.
        self._source_info: Dict[str, RegistrySourceInfo] = {}

    async def fetch_models(
        self, provider: str, version: Optional[int] = None
    ) -> List[RegistryModel]:
        """
        Fetch models for a provider, either current or a specific version.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            version: Optional specific version to fetch. If None, fetches current.

        Returns:
            List of models
        """
        # If version specified, fetch that version
        if version is not None:
            version_data = await self.fetch_version(provider, version)
            return version_data.models

        # Otherwise fetch current
        return await self.fetch_current_models(provider)

    async def fetch_current_models(
        self, provider: str, force_refresh: bool = False
    ) -> List[RegistryModel]:
        """
        Fetch current models for a provider.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            force_refresh: Bypass both the in-memory and on-disk caches and read
                from the origin. Use when the answer must reflect what is
                published right now rather than what was published up to 24h ago
                - see assert_fresh for why age alone is not enough.

        Returns:
            List of current models
        """
        url = f"{self.registry_url}/{provider}/models.json"
        cache_key = f"current_{provider}"

        # Check in-memory cache
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{provider}_current.json"
        if not force_refresh and self._is_cache_valid(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
                models = self._parse_models_dict(data.get("models", {}))
                self._record_source_info(provider, data, url, from_cache=True)
                self._cache[cache_key] = models
                return models

        # Fetch from registry with retry logic
        try:
            # Handle file:// URLs (no retry needed for local files)
            if url.startswith("file://"):
                file_path = Path(url[7:])  # Remove file:// prefix
                if not file_path.exists():
                    raise FileNotFoundError(f"Registry file not found: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)
            else:
                # Handle HTTP/HTTPS URLs
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()

            # Parse models from dictionary
            models = self._parse_models_dict(data.get("models", {}))
            self._record_source_info(provider, data, url, from_cache=False)

            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)

            self._cache[cache_key] = models
            return models

        except Exception as e:
            # If fetch fails, try to use stale cache
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self._record_source_info(provider, data, url, from_cache=True)
                    return self._parse_models_dict(data.get("models", {}))
            raise Exception(f"Failed to fetch registry for {provider}: {e}")

    def _record_source_info(
        self, provider: str, data: Dict[str, Any], url: str, *, from_cache: bool
    ) -> None:
        updated_at = None
        raw = data.get("updated_at")
        if raw:
            try:
                updated_at = _parse_iso_datetime(raw)
            except (ValueError, TypeError):
                logger.debug("Unparseable registry updated_at for %s: %r", provider, raw)
        version = data.get("version")
        self._source_info[provider] = RegistrySourceInfo(
            provider=provider,
            version=int(version) if isinstance(version, int) else None,
            updated_at=updated_at,
            url=url,
            from_cache=from_cache,
        )

    def invalidate_cache(self, provider: Optional[str] = None) -> None:
        """Drop cached registry payloads so the next read comes from the origin.

        Call after publishing a registry update. Without this the process, and
        any other process on the machine, keeps serving the previous payload
        for up to CACHE_DURATION_HOURS.
        """
        providers = [provider] if provider else ["openai", "anthropic", "google"]
        for prov in providers:
            self._cache.pop(f"current_{prov}", None)
            self._source_info.pop(prov, None)
            cache_file = self.cache_dir / f"{prov}_current.json"
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:  # pragma: no cover
                logger.warning("Could not remove registry cache %s: %s", cache_file, e)

    async def get_source_info(
        self, provider: str, force_refresh: bool = False
    ) -> RegistrySourceInfo:
        """Return provenance (version, updated_at, age) for a provider's registry data.

        Fetches the provider slice if it has not been loaded yet. Pass
        force_refresh to describe what is published now rather than what is
        cached locally.
        """
        if force_refresh or provider not in self._source_info:
            await self.fetch_current_models(provider, force_refresh=force_refresh)
        return self._source_info[provider]

    async def assert_fresh(
        self, provider: str, max_age_days: float, force_refresh: bool = True
    ) -> RegistrySourceInfo:
        """Raise RegistryStaleError if this provider's registry data is too old.

        Prices are the reason this matters: a registry published before a model
        existed cannot price that model, and the call then records no cost at
        all. Assert freshness before committing to a metered batch.

        An unknown ``updated_at`` is treated as stale - it cannot be shown to be
        fresh, and silently passing would defeat the check.

        ``force_refresh`` defaults to True, and that default is load-bearing.
        Age alone cannot detect a cache that is merely a VERSION behind: a
        payload published hours ago is "fresh" by age while already superseded,
        and it will keep being served for up to 24h. Validating against the
        origin is what makes this check version-aware. Pass force_refresh=False
        only to describe the local cache deliberately.
        """
        info = await self.get_source_info(provider, force_refresh=force_refresh)
        age_days = info.age_days
        if age_days is None:
            raise RegistryStaleError(
                f"Registry for '{provider}' has no usable updated_at timestamp, "
                "so its age cannot be verified.",
                provider=provider,
                updated_at=None,
                age_days=float("inf"),
            )
        if age_days > max_age_days:
            raise RegistryStaleError(
                f"Registry for '{provider}' is {age_days:.1f} days old "
                f"(updated {info.updated_at:%Y-%m-%d}, version {info.version}), "
                f"which exceeds the {max_age_days} day limit. Model prices may be "
                "missing or wrong; refresh the registry before relying on costs.",
                provider=provider,
                updated_at=info.updated_at,
                age_days=age_days,
            )
        return info

    async def fetch_version(self, provider: str, version: int) -> RegistryVersion:
        """
        Fetch a specific version of the registry for a provider.

        Args:
            provider: Provider name
            version: Version number

        Returns:
            Registry version with models
        """
        url = f"{self.registry_url}/{provider}/v/{version}/models.json"
        cache_key = f"{provider}_v{version}"

        # Check in-memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check file cache
        cache_file = self.cache_dir / f"{provider}_v{version}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                version_info = RegistryVersion(
                    provider=provider,
                    version=version,
                    updated_at=_parse_iso_datetime(
                        data.get("updated_at", datetime.now(timezone.utc).isoformat())
                    ),
                    models=[RegistryModel(**m) for m in data.get("models", [])],
                )
                self._cache[cache_key] = version_info
                return version_info

        # Fetch from registry
        try:
            # Handle file:// URLs
            if url.startswith("file://"):
                file_path = Path(url[7:])  # Remove file:// prefix
                if not file_path.exists():
                    raise FileNotFoundError(f"Registry file not found: {file_path}")
                with open(file_path, "r") as f:
                    data = json.load(f)

                version_info = RegistryVersion(
                    provider=provider,
                    version=version,
                    updated_at=_parse_iso_datetime(
                        data.get("updated_at", datetime.now(timezone.utc).isoformat())
                    ),
                    models=[RegistryModel(**m) for m in data.get("models", [])],
                )

                # Save to cache for parity with HTTP branch
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)

                self._cache[cache_key] = version_info
                return version_info
            else:
                # Handle HTTP/HTTPS URLs
                async with httpx.AsyncClient() as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()

                # Create version info
                version_info = RegistryVersion(
                    provider=provider,
                    version=version,
                    updated_at=_parse_iso_datetime(
                        data.get("updated_at", datetime.now(timezone.utc).isoformat())
                    ),
                    models=[RegistryModel(**m) for m in data.get("models", [])],
                )

                # Save to cache
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)

                self._cache[cache_key] = version_info
                return version_info

        except Exception as e:
            raise Exception(f"Failed to fetch registry version {version} for {provider}: {e}")

    async def get_current_version(self, provider: str) -> int:
        """
        Get the current version number for a provider.

        Args:
            provider: Provider name

        Returns:
            Current version number
        """
        # Fetch current models which should include version info
        url = f"{self.registry_url}/{provider}/models.json"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                return data.get("version", 1)
        except Exception:
            # Default to version 1 if not specified
            return 1

    async def check_drift(self, provider: str, pinned_version: int) -> Dict[str, Any]:
        """
        Check for drift between pinned version and current version.

        Args:
            provider: Provider name
            pinned_version: Version that was pinned in lockfile

        Returns:
            Dictionary with drift information
        """
        current_version = await self.get_current_version(provider)

        if current_version == pinned_version:
            return {
                "has_drift": False,
                "pinned_version": pinned_version,
                "current_version": current_version,
                "versions_behind": 0,
            }

        # Fetch both versions to compare
        pinned = await self.fetch_version(provider, pinned_version)
        current_models = await self.fetch_current_models(provider)

        # Find differences
        pinned_model_names = {m.model_name for m in pinned.models}
        current_model_names = {m.model_name for m in current_models}

        added_models = current_model_names - pinned_model_names
        removed_models = pinned_model_names - current_model_names

        # Check for price changes in common models
        price_changes = []
        for current_model in current_models:
            if current_model.model_name in pinned_model_names:
                pinned_model = next(
                    m for m in pinned.models if m.model_name == current_model.model_name
                )
                if (
                    current_model.dollars_per_million_tokens_input
                    != pinned_model.dollars_per_million_tokens_input
                    or current_model.dollars_per_million_tokens_output
                    != pinned_model.dollars_per_million_tokens_output
                ):
                    price_changes.append(
                        {
                            "model": current_model.model_name,
                            "old_input_cost": pinned_model.dollars_per_million_tokens_input,
                            "new_input_cost": current_model.dollars_per_million_tokens_input,
                            "old_output_cost": pinned_model.dollars_per_million_tokens_output,
                            "new_output_cost": current_model.dollars_per_million_tokens_output,
                        }
                    )

        return {
            "has_drift": True,
            "pinned_version": pinned_version,
            "current_version": current_version,
            "versions_behind": current_version - pinned_version,
            "added_models": list(added_models),
            "removed_models": list(removed_models),
            "price_changes": price_changes,
        }

    def _parse_models_dict(self, models_dict: Dict[str, Any]) -> List[RegistryModel]:
        """Parse models from dictionary format to list of RegistryModel objects."""
        # Validate that we received a dictionary, not an array
        if not isinstance(models_dict, dict):
            raise ValueError(
                f"Registry models must be a dictionary with 'provider:model' keys, got {type(models_dict)}"
            )

        models = []
        for model_key, model_data in models_dict.items():
            try:
                # Validate key format for O(1) lookup compliance
                if ":" not in model_key:
                    print(
                        f"Warning: Model key '{model_key}' doesn't follow 'provider:model' format"
                    )
                    continue

                provider, model_name = model_key.split(":", 1)

                # Validate model_data is a dictionary
                if not isinstance(model_data, dict):
                    print(f"Warning: Model data for '{model_key}' is not a dictionary")
                    continue

                # Add provider if not present
                if "provider" not in model_data:
                    model_data["provider"] = provider

                # Validate required fields
                if "model_name" not in model_data:
                    print(f"Warning: Model '{model_key}' missing required field 'model_name'")
                    continue

                if not model_data.get("display_name"):
                    print(f"Warning: Model '{model_key}' missing required field 'display_name'")
                    continue

                model = RegistryModel(**model_data)
                models.append(model)
            except Exception as e:
                # Log but don't fail on individual model parsing errors
                print(f"Warning: Failed to parse model {model_key}: {e}")
                continue

        if not models and models_dict:
            print(f"Warning: No valid models parsed from {len(models_dict)} entries")

        return models

    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if a cache file is still valid."""
        if not cache_file.exists():
            return False

        # Check age
        age_hours = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(cache_file.stat().st_mtime, tz=timezone.utc)
        ).total_seconds() / 3600
        return age_hours < self.CACHE_DURATION_HOURS

    def clear_cache(self):
        """Clear all cached registry data."""
        self._cache.clear()

        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    async def validate_model(
        self, provider: str, model_name: str, version: Optional[int] = None
    ) -> bool:
        """
        Validate that a model exists in the registry.

        Args:
            provider: Provider name
            model_name: Model name to validate
            version: Optional specific version to validate against. If None, uses current.

        Returns:
            True if model exists and is active
        """
        try:
            # Use pinned version if set as an attribute (set by service)
            if version is None and hasattr(self, "_pinned_version"):
                version = self._pinned_version

            models = await self.fetch_models(provider, version)
            for model in models:
                if model.model_name == model_name and model.is_active:
                    return True
            return False
        except Exception as e:
            # Fail-open per design: treat model as valid if registry is unavailable
            # but emit an explicit warning so callers can distinguish outage from success.
            import logging

            logging.getLogger(__name__).warning(
                "Registry validate_model fail-open: provider=%s model=%s error=%s",
                provider,
                model_name,
                str(e),
            )
            return True
