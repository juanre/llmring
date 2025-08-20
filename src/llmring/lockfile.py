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
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Optional constraints"
    )

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
    bindings: List[AliasBinding] = Field(
        default_factory=list, description="Alias bindings"
    )
    registry_versions: Dict[str, int] = Field(
        default_factory=dict, description="Pinned registry versions per provider"
    )

    def get_binding(self, alias: str) -> Optional[AliasBinding]:
        """Get binding for a specific alias."""
        for binding in self.bindings:
            if binding.alias == alias:
                return binding
        return None

    def set_binding(
        self, alias: str, model_ref: str, constraints: Optional[Dict] = None
    ):
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

    @classmethod
    def create_default(cls) -> "Lockfile":
        """Create a default lockfile with sensible defaults based on available API keys."""
        lockfile = cls()

        # Create default profile
        default_profile = ProfileConfig(name="default")

        # Auto-detect available providers and suggest defaults
        defaults = cls._suggest_defaults()
        for alias, model_ref in defaults.items():
            default_profile.set_binding(alias, model_ref)

        lockfile.profiles["default"] = default_profile

        # Create additional profiles
        lockfile.profiles["prod"] = ProfileConfig(name="prod")
        lockfile.profiles["staging"] = ProfileConfig(name="staging")
        lockfile.profiles["dev"] = ProfileConfig(name="dev")

        return lockfile

    @staticmethod
    def _suggest_defaults() -> Dict[str, str]:
        """Suggest default bindings based on available API keys."""
        defaults = {}

        # Check available providers
        if os.environ.get("OPENAI_API_KEY"):
            defaults["long_context"] = "openai:gpt-4-turbo-preview"
            defaults["low_cost"] = "openai:gpt-3.5-turbo"
            defaults["json_mode"] = "openai:gpt-4-turbo-preview"
            defaults["fast"] = "openai:gpt-3.5-turbo"

        if os.environ.get("ANTHROPIC_API_KEY"):
            defaults["deep"] = "anthropic:claude-3-opus-20240229"
            defaults["balanced"] = "anthropic:claude-3-sonnet-20240229"
            defaults["pdf_reader"] = (
                "anthropic:claude-3-sonnet-20240229"  # Claude is good at PDFs
            )
            if "low_cost" not in defaults:
                defaults["low_cost"] = "anthropic:claude-3-haiku-20240307"
            if "fast" not in defaults:
                defaults["fast"] = "anthropic:claude-3-haiku-20240307"

        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            defaults["vision"] = "google:gemini-1.5-pro"
            defaults["multimodal"] = "google:gemini-1.5-pro"
            if "long_context" not in defaults:
                defaults["long_context"] = "google:gemini-1.5-pro"
            if "pdf_reader" not in defaults:
                defaults["pdf_reader"] = (
                    "google:gemini-1.5-pro"  # Gemini also handles PDFs well
                )

        # Ollama is always available locally
        if not defaults:
            defaults["default"] = "ollama:llama3.3:latest"
            defaults["local"] = "ollama:llama3.3:latest"
        else:
            # Add a local option if Ollama is available
            defaults["local"] = "ollama:llama3.3:latest"

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

    def get_binding(
        self, alias: str, profile: Optional[str] = None
    ) -> Optional[AliasBinding]:
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
