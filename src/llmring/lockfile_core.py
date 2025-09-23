"""
Lockfile management for LLMRing.

The lockfile (llmring.lock) is the authoritative configuration source for:
- Alias to model bindings
- Pinned registry versions per provider
- Profiles (prod/staging/dev)
- Optional constraints
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
from pydantic import BaseModel, Field


class AliasBinding(BaseModel):
    """Represents an alias to model binding."""

    alias: str = Field(..., description="Alias name (e.g., 'summarizer')")
    provider: str = Field(..., description="Provider name (e.g., 'openai')")
    model: str = Field(..., description="Model name (e.g., 'gpt-4')")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optional constraints")

    @property
    def model_ref(self) -> str:
        """Get the full model reference (provider:model)."""
        return f"{self.provider}:{self.model}"

    @classmethod
    def from_model_ref(
        cls, alias: str, model_ref: str, constraints: Optional[Dict] = None
    ) -> "AliasBinding":
        """Create from a model reference string like 'openai:gpt-4'."""
        if ":" not in model_ref:
            raise ValueError(
                f"Invalid model reference: {model_ref}. Expected format: provider:model"
            )
        provider, model = model_ref.split(":", 1)
        return cls(alias=alias, provider=provider, model=model, constraints=constraints)


class ProfileConfig(BaseModel):
    """Configuration for a specific profile."""

    name: str = Field(..., description="Profile name (e.g., 'prod', 'staging', 'dev')")
    bindings: List[AliasBinding] = Field(default_factory=list, description="Alias bindings")
    registry_versions: Dict[str, int] = Field(
        default_factory=dict, description="Pinned registry versions per provider"
    )

    def get_binding(self, alias: str) -> Optional[AliasBinding]:
        """Get binding for a specific alias."""
        for binding in self.bindings:
            if binding.alias == alias:
                return binding
        return None

    def set_binding(self, alias: str, model_ref: str, constraints: Optional[Dict] = None):
        """Set or update a binding."""
        # Remove existing binding if present
        self.bindings = [b for b in self.bindings if b.alias != alias]
        # Add new binding
        binding = AliasBinding.from_model_ref(alias, model_ref, constraints)
        self.bindings.append(binding)

    def remove_binding(self, alias: str) -> bool:
        """Remove a binding. Returns True if removed, False if not found."""
        original_count = len(self.bindings)
        self.bindings = [b for b in self.bindings if b.alias != alias]
        return len(self.bindings) < original_count


class Lockfile(BaseModel):
    """Represents the complete lockfile configuration."""

    version: str = Field(default="1.0", description="Lockfile format version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    profiles: Dict[str, ProfileConfig] = Field(
        default_factory=dict, description="Profile configurations"
    )

    default_profile: str = Field(default="default", description="Default profile name")

    # Metadata field for intelligent creation rationale and other info
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata including rationale from intelligent creation"
    )

    @classmethod
    def create_default(cls) -> "Lockfile":
        """Create a default lockfile with sensible defaults based on available API keys.

        Note: This creates a basic lockfile. For intelligent recommendations based on
        the registry, use create_default_async() instead.
        """
        lockfile = cls()

        # Create default profile
        default_profile = ProfileConfig(name="default")

        # Create basic defaults without hardcoded models
        defaults = cls._suggest_defaults_basic()
        for alias, model_ref in defaults.items():
            default_profile.set_binding(alias, model_ref)

        lockfile.profiles["default"] = default_profile

        # Create additional profiles
        lockfile.profiles["prod"] = ProfileConfig(name="prod")
        lockfile.profiles["staging"] = ProfileConfig(name="staging")
        lockfile.profiles["dev"] = ProfileConfig(name="dev")

        return lockfile

    @classmethod
    async def create_default_async(cls, registry_client=None) -> "Lockfile":
        """Create a default lockfile with intelligent defaults from registry.

        Args:
            registry_client: Optional RegistryClient instance. If not provided,
                           creates a new one.

        Returns:
            Lockfile with intelligent alias bindings based on registry data.
        """
        from llmring.registry import RegistryClient

        if registry_client is None:
            registry_client = RegistryClient()

        lockfile = cls()

        # Create default profile
        default_profile = ProfileConfig(name="default")

        # Get registry-based suggestions
        defaults = await cls._suggest_defaults_from_registry(registry_client)
        for alias, model_ref in defaults.items():
            default_profile.set_binding(alias, model_ref)

        lockfile.profiles["default"] = default_profile

        # Create additional profiles
        lockfile.profiles["prod"] = ProfileConfig(name="prod")
        lockfile.profiles["staging"] = ProfileConfig(name="staging")
        lockfile.profiles["dev"] = ProfileConfig(name="dev")

        return lockfile

    @staticmethod
    def _suggest_defaults_basic() -> Dict[str, str]:
        """Suggest basic default bindings without hardcoded models.

        Returns minimal defaults that don't rely on specific model IDs.
        For intelligent suggestions, use _suggest_defaults_from_registry().
        """
        defaults = {}

        # Return empty defaults - let user explicitly configure
        # or use intelligent creation with registry data
        # No hardcoded models per source-of-truth principles

        return defaults

    @staticmethod
    async def _suggest_defaults_from_registry(registry_client) -> Dict[str, str]:
        """Suggest default bindings based on registry data and available API keys.

        Args:
            registry_client: RegistryClient instance for querying model data

        Returns:
            Dictionary of alias to model bindings based on capabilities
        """
        defaults = {}

        # Check available providers and query registry for each
        if os.environ.get("OPENAI_API_KEY"):
            try:
                models = await registry_client.fetch_current_models("openai")
                active_models = [m for m in models if m.is_active]

                if active_models:
                    # Find long_context: highest max_input_tokens
                    long_context = max(active_models, key=lambda m: m.max_input_tokens or 0)
                    if long_context.max_input_tokens and long_context.max_input_tokens > 0:
                        defaults["long_context"] = f"openai:{long_context.model_name}"

                    # Find low_cost: lowest input price
                    models_with_price = [
                        m for m in active_models if m.dollars_per_million_tokens_input
                    ]
                    if models_with_price:
                        low_cost = min(
                            models_with_price,
                            key=lambda m: m.dollars_per_million_tokens_input or float("inf"),
                        )
                        defaults["low_cost"] = f"openai:{low_cost.model_name}"
                        defaults["fast"] = f"openai:{low_cost.model_name}"

                    # Find json_mode: supports_json_mode=True
                    json_models = [m for m in active_models if m.supports_json_mode]
                    if json_models:
                        # Prefer one with good balance of capability and cost
                        json_model = json_models[0]  # Simple selection for now
                        defaults["json_mode"] = f"openai:{json_model.model_name}"

                    # Add mcp_agent if not set by Anthropic (prefer most capable OpenAI model)
                    if "mcp_agent" not in defaults and active_models:
                        # Find the most capable model (usually the one with highest tokens)
                        capable_model = max(active_models, key=lambda m: m.max_input_tokens or 0)
                        defaults["mcp_agent"] = f"openai:{capable_model.model_name}"

            except Exception as e:
                # Log but don't fail - registry might be unavailable
                import logging

                logging.warning(f"Could not fetch OpenAI models from registry: {e}")

        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                models = await registry_client.fetch_current_models("anthropic")
                active_models = [m for m in models if m.is_active]

                if active_models:
                    # Find deep: balance of capability (prefer higher tokens) and reasonable cost
                    models_with_tokens = [m for m in active_models if m.max_input_tokens]
                    if models_with_tokens:
                        # Sort by max_input_tokens descending
                        sorted_models = sorted(
                            models_with_tokens,
                            key=lambda m: m.max_input_tokens or 0,
                            reverse=True,
                        )
                        # Take the most capable model
                        if sorted_models:
                            defaults["deep"] = f"anthropic:{sorted_models[0].model_name}"
                            # MCP agent should use the most capable model for complex reasoning
                            defaults["mcp_agent"] = f"anthropic:{sorted_models[0].model_name}"
                        # Take a mid-tier model for balanced
                        if len(sorted_models) > 1:
                            mid_idx = len(sorted_models) // 2
                            defaults["balanced"] = f"anthropic:{sorted_models[mid_idx].model_name}"

                    # Find low_cost if not already set
                    if "low_cost" not in defaults:
                        models_with_price = [
                            m for m in active_models if m.dollars_per_million_tokens_input
                        ]
                        if models_with_price:
                            low_cost = min(
                                models_with_price,
                                key=lambda m: m.dollars_per_million_tokens_input or float("inf"),
                            )
                            defaults["low_cost"] = f"anthropic:{low_cost.model_name}"

                    # PDF reader - prefer models with good context length
                    if "pdf_reader" not in defaults and models_with_tokens:
                        pdf_model = max(models_with_tokens, key=lambda m: m.max_input_tokens or 0)
                        defaults["pdf_reader"] = f"anthropic:{pdf_model.model_name}"

            except Exception as e:
                import logging

                logging.warning(f"Could not fetch Anthropic models from registry: {e}")

        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            try:
                models = await registry_client.fetch_current_models("google")
                active_models = [m for m in models if m.is_active]

                if active_models:
                    # Find vision: supports_vision=True
                    vision_models = [m for m in active_models if m.supports_vision]
                    if vision_models:
                        # Prefer one with good context length
                        vision_models_sorted = sorted(
                            vision_models,
                            key=lambda m: m.max_input_tokens or 0,
                            reverse=True,
                        )
                        if vision_models_sorted:
                            defaults["vision"] = f"google:{vision_models_sorted[0].model_name}"
                            defaults["multimodal"] = f"google:{vision_models_sorted[0].model_name}"

                    # Long context if not set
                    if "long_context" not in defaults:
                        models_with_tokens = [m for m in active_models if m.max_input_tokens]
                        if models_with_tokens:
                            long_context = max(
                                models_with_tokens,
                                key=lambda m: m.max_input_tokens or 0,
                            )
                            defaults["long_context"] = f"google:{long_context.model_name}"

                    # PDF reader if not set
                    if "pdf_reader" not in defaults:
                        models_with_tokens = [m for m in active_models if m.max_input_tokens]
                        if models_with_tokens:
                            pdf_model = max(
                                models_with_tokens,
                                key=lambda m: m.max_input_tokens or 0,
                            )
                            defaults["pdf_reader"] = f"google:{pdf_model.model_name}"

            except Exception as e:
                import logging

                logging.warning(f"Could not fetch Google models from registry: {e}")

        # Don't add hardcoded local option - let user configure explicitly
        # or use registry data if available

        return defaults

    def get_profile(self, name: Optional[str] = None) -> ProfileConfig:
        """Get a profile by name, or the default profile."""
        profile_name = name or self.default_profile

        if profile_name not in self.profiles:
            # Create profile on demand if it doesn't exist
            self.profiles[profile_name] = ProfileConfig(name=profile_name)

        return self.profiles[profile_name]

    def set_binding(
        self,
        alias: str,
        model_ref: str,
        profile: Optional[str] = None,
        constraints: Optional[Dict] = None,
    ):
        """Set a binding in the specified profile."""
        profile_config = self.get_profile(profile)
        profile_config.set_binding(alias, model_ref, constraints)
        self.updated_at = datetime.now(timezone.utc)

    def get_binding(self, alias: str, profile: Optional[str] = None) -> Optional[AliasBinding]:
        """Get a binding from the specified profile."""
        profile_config = self.get_profile(profile)
        return profile_config.get_binding(alias)

    def list_aliases(self, profile: Optional[str] = None) -> List[str]:
        """List all aliases in the specified profile."""
        profile_config = self.get_profile(profile)
        return [b.alias for b in profile_config.bindings]

    def resolve_alias(self, alias: str, profile: Optional[str] = None) -> Optional[str]:
        """Resolve an alias to a model reference (provider:model)."""
        binding = self.get_binding(alias, profile)
        return binding.model_ref if binding else None

    def save(self, path: Optional[Path] = None):
        """Save the lockfile to disk."""
        path = path or Path("llmring.lock")

        # Convert to dict for serialization
        data = self.model_dump(mode="json")

        # Convert datetime to ISO format strings
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()

        # Ensure metadata is properly serialized (it's already a dict)
        if "metadata" not in data:
            data["metadata"] = {}

        # Save as TOML or JSON based on preference
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Default to TOML
            with open(path, "w") as f:
                toml.dump(data, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Lockfile":
        """Load a lockfile from disk."""
        path = path or Path("llmring.lock")

        if not path.exists():
            # Return default if file doesn't exist
            return cls.create_default()

        # Load based on file extension
        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
        else:
            # Default to TOML
            with open(path, "r") as f:
                data = toml.load(f)

        # Convert ISO strings back to datetime
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Convert profiles dict to ProfileConfig objects
        if "profiles" in data:
            for profile_name, profile_data in data["profiles"].items():
                if isinstance(profile_data, dict) and "bindings" in profile_data:
                    # Ensure bindings are AliasBinding objects
                    bindings = []
                    for binding_data in profile_data["bindings"]:
                        if isinstance(binding_data, dict):
                            bindings.append(AliasBinding(**binding_data))
                    profile_data["bindings"] = bindings
                    data["profiles"][profile_name] = ProfileConfig(**profile_data)

        # Ensure backward compatibility - add empty metadata if not present
        if "metadata" not in data:
            data["metadata"] = {}

        return cls(**data)

    @classmethod
    def find_lockfile(cls, start_path: Optional[Path] = None) -> Optional[Path]:
        """Find a lockfile by searching up the directory tree."""
        current = Path(start_path or os.getcwd()).resolve()

        while current != current.parent:
            lockfile_path = current / "llmring.lock"
            if lockfile_path.exists():
                return lockfile_path

            # Also check for JSON variant
            lockfile_json = current / "llmring.lock.json"
            if lockfile_json.exists():
                return lockfile_json

            current = current.parent

        return None

    def calculate_digest(self) -> str:
        """
        Calculate SHA256 digest of the lockfile for receipts.

        Returns:
            Hex-encoded SHA256 digest
        """
        import hashlib

        # Get canonical representation
        data = self.model_dump(mode="json")

        # Convert datetimes to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()

        # Sort and serialize deterministically
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))

        # Calculate SHA256
        return hashlib.sha256(canonical.encode()).hexdigest()
