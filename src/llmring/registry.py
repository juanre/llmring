"""
Registry client for fetching model information from GitHub Pages.

The registry is hosted at https://llmring.github.io/registry/
with per-provider model lists and versioned archives.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field


class RegistryModel(BaseModel):
    """Model information from the registry."""
    
    provider: str = Field(..., description="Provider name")
    model_name: str = Field(..., description="Model identifier")
    display_name: str = Field(..., description="Human-friendly name")
    description: Optional[str] = Field(None, description="Model description")
    
    # Capabilities
    max_context_tokens: Optional[int] = Field(None, description="Max context size")
    max_output_tokens: Optional[int] = Field(None, description="Max output size")
    supports_vision: bool = Field(False, description="Supports image input")
    supports_function_calling: bool = Field(False, description="Supports functions")
    supports_json_mode: bool = Field(False, description="Supports JSON output")
    supports_parallel_tool_calls: bool = Field(False, description="Supports parallel tools")
    
    # Pricing (per million tokens)
    cost_per_million_input_tokens: Optional[float] = Field(None, description="Input cost")
    cost_per_million_output_tokens: Optional[float] = Field(None, description="Output cost")
    
    # Status
    is_active: bool = Field(True, description="Model is currently available")
    added_date: Optional[datetime] = Field(None, description="When model was added")
    deprecated_date: Optional[datetime] = Field(None, description="When model was deprecated")


class RegistryVersion(BaseModel):
    """Registry version information."""
    
    provider: str = Field(..., description="Provider name")
    version: int = Field(..., description="Version number")
    updated_at: datetime = Field(..., description="Last update time")
    models: List[RegistryModel] = Field(default_factory=list, description="Models in this version")


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
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._cache: Dict[str, Any] = {}
    
    async def fetch_current_models(self, provider: str) -> List[RegistryModel]:
        """
        Fetch current models for a provider.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            List of current models
        """
        url = f"{self.registry_url}/{provider}/models.json"
        cache_key = f"current_{provider}"
        
        # Check in-memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{provider}_current.json"
        if self._is_cache_valid(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
                models = [RegistryModel(**m) for m in data.get("models", [])]
                self._cache[cache_key] = models
                return models
        
        # Fetch from registry
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Parse models
                models = [RegistryModel(**m) for m in data.get("models", [])]
                
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
                    return [RegistryModel(**m) for m in data.get("models", [])]
            raise Exception(f"Failed to fetch registry for {provider}: {e}")
    
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
                    updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
                    models=[RegistryModel(**m) for m in data.get("models", [])]
                )
                self._cache[cache_key] = version_info
                return version_info
        
        # Fetch from registry
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                
                # Create version info
                version_info = RegistryVersion(
                    provider=provider,
                    version=version,
                    updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
                    models=[RegistryModel(**m) for m in data.get("models", [])]
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
                "versions_behind": 0
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
                pinned_model = next(m for m in pinned.models if m.model_name == current_model.model_name)
                if (current_model.cost_per_million_input_tokens != pinned_model.cost_per_million_input_tokens or
                    current_model.cost_per_million_output_tokens != pinned_model.cost_per_million_output_tokens):
                    price_changes.append({
                        "model": current_model.model_name,
                        "old_input_cost": pinned_model.cost_per_million_input_tokens,
                        "new_input_cost": current_model.cost_per_million_input_tokens,
                        "old_output_cost": pinned_model.cost_per_million_output_tokens,
                        "new_output_cost": current_model.cost_per_million_output_tokens
                    })
        
        return {
            "has_drift": True,
            "pinned_version": pinned_version,
            "current_version": current_version,
            "versions_behind": current_version - pinned_version,
            "added_models": list(added_models),
            "removed_models": list(removed_models),
            "price_changes": price_changes
        }
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if a cache file is still valid."""
        if not cache_file.exists():
            return False
        
        # Check age
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        return age_hours < self.CACHE_DURATION_HOURS
    
    def clear_cache(self):
        """Clear all cached registry data."""
        self._cache.clear()
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
    
    async def validate_model(self, provider: str, model_name: str) -> bool:
        """
        Validate that a model exists in the current registry.
        
        Args:
            provider: Provider name
            model_name: Model name to validate
            
        Returns:
            True if model exists and is active
        """
        try:
            models = await self.fetch_current_models(provider)
            for model in models:
                if model.model_name == model_name and model.is_active:
                    return True
            return False
        except Exception:
            # If we can't fetch registry, assume model is valid
            return True