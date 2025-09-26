"""
MCP tools for conversational lockfile management.

Provides tools for managing lockfiles through natural conversation,
including adding/removing aliases, assessing models, and generating configurations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llmring.lockfile_core import AliasBinding, Lockfile, ProfileConfig
from llmring.registry import RegistryClient

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class LockfileManagerTools:
    """MCP tools for managing lockfiles conversationally."""

    def __init__(self, lockfile_path: Optional[Path] = None):
        """
        Initialize lockfile manager tools.

        Args:
            lockfile_path: Path to lockfile, defaults to llmring.lock
        """
        self.lockfile_path = lockfile_path or Path("llmring.lock")
        self.lockfile = None
        self.registry = RegistryClient()
        self.working_profile = "default"

        # Load existing or create new
        if self.lockfile_path.exists():
            self.lockfile = Lockfile.load(self.lockfile_path)
        else:
            self.lockfile = Lockfile()

    async def get_available_providers(self) -> Dict[str, Any]:
        """
        Check which providers have API keys configured.

        Returns:
            Dictionary with configured and unconfigured providers
        """
        provider_configs = {
            "openai": {
                "env_var": "OPENAI_API_KEY",
                "has_key": bool(os.environ.get("OPENAI_API_KEY"))
            },
            "anthropic": {
                "env_var": "ANTHROPIC_API_KEY",
                "has_key": bool(os.environ.get("ANTHROPIC_API_KEY"))
            },
            "google": {
                "env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                "has_key": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            },
            "ollama": {
                "env_var": None,
                "has_key": True  # Ollama doesn't require API key
            }
        }

        configured = []
        unconfigured = []

        for provider, config in provider_configs.items():
            if config["has_key"]:
                configured.append(provider)
            else:
                unconfigured.append(provider)

        return {
            "configured": configured,
            "unconfigured": unconfigured,
            "details": provider_configs
        }

    async def list_models(
        self,
        providers: Optional[List[str]] = None,
        include_inactive: bool = False
    ) -> Dict[str, Any]:
        """
        List all available models with their specifications.

        Args:
            providers: Optional list of providers to filter by
            include_inactive: Whether to include deprecated/inactive models

        Returns:
            Dict containing models list, count, and providers included
        """
        if providers is None:
            # Get all configured providers
            available = await self.get_available_providers()
            providers = available["configured"]

        all_models = []

        for provider in providers:
            try:
                models = await self.registry.fetch_current_models(provider)

                for model in models:
                    # Skip inactive unless requested
                    if not model.is_active and not include_inactive:
                        continue

                    model_info = {
                        "model_ref": f"{provider}:{model.model_name}",
                        "provider": provider,
                        "model_name": model.model_name,
                        "display_name": model.display_name,
                        "description": model.description,
                        "context_window": model.max_input_tokens,
                        "max_output": model.max_output_tokens,
                        "price_input": model.dollars_per_million_tokens_input,
                        "price_output": model.dollars_per_million_tokens_output,
                        "supports_vision": model.supports_vision,
                        "supports_functions": model.supports_function_calling,
                        "supports_json_mode": model.supports_json_mode,
                        "supports_parallel_tools": model.supports_parallel_tool_calls,
                        "is_active": model.is_active,
                        "active": model.is_active,  # Alias for consistency
                        "input_cost": model.dollars_per_million_tokens_input,  # Alias
                        "output_cost": model.dollars_per_million_tokens_output,  # Alias
                        "added_date": model.added_date.isoformat() if model.added_date else None,
                        "deprecated_date": model.deprecated_date.isoformat() if model.deprecated_date else None,
                        "capabilities": []  # Will be populated below
                    }

                    # Add capability indicators
                    if model.supports_vision:
                        model_info["capabilities"].append("vision")
                    if model.supports_function_calling:
                        model_info["capabilities"].append("function_calling")
                    if model.supports_json_mode:
                        model_info["capabilities"].append("json_output")
                    all_models.append(model_info)

            except Exception as e:
                logger.warning(f"Failed to fetch models for {provider}: {e}")
                continue

        return {
            "models": all_models,
            "total_count": len(all_models),
            "providers_included": providers
        }

    async def filter_models_by_requirements(
        self,
        min_context: Optional[int] = None,
        max_price_input: Optional[float] = None,
        max_price_output: Optional[float] = None,
        requires_vision: Optional[bool] = None,
        requires_functions: Optional[bool] = None,
        requires_json_mode: Optional[bool] = None,
        providers: Optional[List[str]] = None,
        include_inactive: bool = False
    ) -> Dict[str, Any]:
        """
        Filter models based on specific requirements.

        Args:
            min_context: Minimum context window size
            max_price_input: Maximum price per million input tokens
            max_price_output: Maximum price per million output tokens
            requires_vision: Must support vision
            requires_functions: Must support function calling
            requires_json_mode: Must support JSON mode
            providers: Limit to specific providers
            include_inactive: Include deprecated models

        Returns:
            Dict containing filtered models and applied filters
        """
        # Get all models
        models_result = await self.list_models(providers, include_inactive)
        all_models = models_result["models"]

        # Apply filters
        filtered = []
        for model in all_models:
            # Context window filter
            if min_context and model["context_window"]:
                if model["context_window"] < min_context:
                    continue

            # Price filters
            if max_price_input and model["price_input"]:
                if model["price_input"] > max_price_input:
                    continue

            if max_price_output and model["price_output"]:
                if model["price_output"] > max_price_output:
                    continue

            # Capability filters
            if requires_vision is not None and requires_vision:
                if not model["supports_vision"]:
                    continue

            if requires_functions is not None and requires_functions:
                if not model["supports_functions"]:
                    continue

            if requires_json_mode is not None and requires_json_mode:
                if not model["supports_json_mode"]:
                    continue

            filtered.append(model)

        applied_filters = []
        if min_context:
            applied_filters.append(f"min_context={min_context}")
        if max_price_input:
            applied_filters.append(f"max_input_cost=${max_price_input}")
        if max_price_output:
            applied_filters.append(f"max_output_cost=${max_price_output}")
        if requires_vision:
            applied_filters.append("vision_required")
        if requires_functions:
            applied_filters.append("functions_required")
        if requires_json_mode:
            applied_filters.append("json_mode_required")
        if providers:
            applied_filters.append(f"providers={','.join(providers)}")

        return {
            "models": filtered,
            "count": len(filtered),
            "applied_filters": applied_filters
        }

    async def get_model_details(
        self,
        models: List[str]
    ) -> Dict[str, Any]:
        """
        Get complete details for specific models.

        Args:
            models: List of model references (provider:model format)

        Returns:
            Dict containing detailed information for each model
        """
        detailed_models = []

        for model_ref in models:
            if ":" not in model_ref:
                detailed_models.append({
                    "error": f"Invalid model reference: {model_ref}",
                    "model": model_ref
                })
                continue

            provider, model_name = model_ref.split(":", 1)

            try:
                # Get all models for provider
                provider_models = await self.registry.fetch_current_models(provider)

                # Find the specific model
                for model in provider_models:
                    if model.model_name == model_name:
                        detailed_models.append({
                            "model_ref": model_ref,
                            "provider": provider,
                            "model_name": model.model_name,
                            "display_name": model.display_name,
                            "description": model.description,
                            "full_details": {
                                "display_name": model.display_name,
                                "description": model.description,
                                "context_window": model.max_input_tokens,
                                "max_output": model.max_output_tokens,
                                "dollars_per_million_tokens_input": model.dollars_per_million_tokens_input,
                                "dollars_per_million_tokens_output": model.dollars_per_million_tokens_output,
                                "supports_vision": model.supports_vision,
                                "supports_function_calling": model.supports_function_calling,
                                "supports_json_mode": model.supports_json_mode,
                                "supports_parallel_tool_calls": model.supports_parallel_tool_calls,
                                "active": model.is_active,
                                "knowledge_cutoff": getattr(model, "knowledge_cutoff", None),
                                "added_date": model.added_date.isoformat() if model.added_date else None,
                                "deprecated_date": model.deprecated_date.isoformat() if model.deprecated_date else None
                            }
                        })
                        break
                else:
                    detailed_models.append({
                        "error": f"Model {model_ref} not found",
                        "model": model_ref
                    })

            except Exception as e:
                detailed_models.append({
                    "error": f"Failed to fetch details: {str(e)}",
                    "model": model_ref
                })

        return {
            "models": detailed_models,
            "requested": models,
            "found": len([m for m in detailed_models if "error" not in m])
        }

    async def add_alias(
        self,
        alias: str,
        model: str,
        profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add or update an alias in the lockfile.

        Args:
            alias: Name of the alias to add
            model: Model reference (provider:model)
            profile: Profile to add to, defaults to current working profile

        Returns:
            Result with the alias configuration
        """
        profile = profile or self.working_profile

        # Add the binding
        self.lockfile.set_binding(alias, model, profile=profile)

        # Save
        self.lockfile.save(self.lockfile_path)

        return {
            "success": True,
            "alias": alias,
            "model": model,
            "profile": profile,
            "message": f"Added alias '{alias}' → {model} to profile '{profile}'"
        }

    async def remove_alias(self, alias: str, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Remove an alias from the lockfile.

        Args:
            alias: Name of the alias to remove
            profile: Profile to remove from, defaults to current working profile

        Returns:
            Result of the removal
        """
        profile = profile or self.working_profile
        profile_config = self.lockfile.get_profile(profile)

        if profile_config.remove_binding(alias):
            self.lockfile.save(self.lockfile_path)
            return {
                "success": True,
                "message": f"Removed alias '{alias}' from profile '{profile}'"
            }
        else:
            return {
                "success": False,
                "message": f"Alias '{alias}' not found in profile '{profile}'"
            }

    async def list_aliases(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        List all aliases in a profile.

        Args:
            profile: Profile to list, defaults to current working profile

        Returns:
            List of aliases with their configurations
        """
        profile = profile or self.working_profile
        profile_config = self.lockfile.get_profile(profile)

        aliases = []
        for binding in profile_config.bindings:
            aliases.append({
                "alias": binding.alias,
                "provider": binding.provider,
                "model": binding.model,
                "model_ref": binding.model_ref
            })

        return {
            "profile": profile,
            "aliases": aliases,
            "count": len(aliases)
        }

    async def assess_model(self, model_ref: str) -> Dict[str, Any]:
        """
        Assess a model's capabilities and costs.

        Args:
            model_ref: Model reference (provider:model or alias)

        Returns:
            Model assessment with capabilities, pricing, and recommendations
        """
        # Try to resolve as alias first
        if ":" not in model_ref:
            resolved = self.lockfile.resolve_alias(model_ref)
            if resolved:
                model_ref = resolved
            else:
                return {
                    "error": f"Invalid model reference: {model_ref}. Use format provider:model or a valid alias"
                }

        provider, model_name = model_ref.split(":", 1)

        try:
            # Fetch models from registry
            models = await self.registry.fetch_current_models(provider)

            # Find the specific model
            model_data = None
            for m in models:
                if m.model_name == model_name:
                    model_data = m
                    break

            if not model_data:
                return {
                    "error": f"Model {model_ref} not found in registry"
                }

            # Prepare comprehensive assessment
            assessment = {
                "model": model_ref,
                "provider": provider,
                "model_name": model_name,
                "display_name": model_data.display_name,
                "description": model_data.description,
                "active": model_data.is_active,
                "capabilities": {
                    "max_input_tokens": model_data.max_input_tokens,
                    "max_output_tokens": model_data.max_output_tokens,
                    "supports_vision": model_data.supports_vision,
                    "supports_functions": model_data.supports_function_calling,
                    "supports_json_mode": model_data.supports_json_mode,
                    "supports_parallel_tools": model_data.supports_parallel_tool_calls
                },
                "pricing": {
                    "input": model_data.dollars_per_million_tokens_input,
                    "output": model_data.dollars_per_million_tokens_output,
                    "input_cost_per_million": model_data.dollars_per_million_tokens_input,
                    "output_cost_per_million": model_data.dollars_per_million_tokens_output
                },
                "specifications": {
                    "context_window": model_data.max_input_tokens,
                    "max_output": model_data.max_output_tokens,
                    "knowledge_cutoff": getattr(model_data, "knowledge_cutoff", None)
                },
                "metadata": {
                    "added_date": model_data.added_date.isoformat() if model_data.added_date else None,
                    "deprecated_date": model_data.deprecated_date.isoformat() if model_data.deprecated_date else None,
                    "is_deprecated": model_data.deprecated_date is not None if model_data.deprecated_date else False
                },
                "status": {
                    "added_date": model_data.added_date.isoformat() if model_data.added_date else None,
                    "deprecated_date": model_data.deprecated_date.isoformat() if model_data.deprecated_date else None,
                    "is_deprecated": model_data.deprecated_date is not None if model_data.deprecated_date else False
                },
                "recommended_for": []
            }

            # Add recommendations based on capabilities
            if model_data.dollars_per_million_tokens_input and model_data.dollars_per_million_tokens_input < 1.0:
                assessment["recommended_for"].append("high-volume tasks")
                assessment["recommended_for"].append("quick responses")

            if model_data.max_input_tokens and model_data.max_input_tokens > 100000:
                assessment["recommended_for"].append("long documents")
                assessment["recommended_for"].append("extensive context")

            if model_data.supports_vision:
                assessment["recommended_for"].append("image analysis")
                assessment["recommended_for"].append("multimodal tasks")

            if model_data.supports_function_calling:
                assessment["recommended_for"].append("tool use")
                assessment["recommended_for"].append("structured outputs")

            return assessment

        except Exception as e:
            return {
                "error": f"Failed to assess model: {str(e)}"
            }

    async def save_lockfile(self) -> Dict[str, Any]:
        """Save the current lockfile state."""
        self.lockfile.save(self.lockfile_path)
        return {
            "success": True,
            "path": str(self.lockfile_path),
            "message": f"Lockfile saved to {self.lockfile_path}"
        }

    async def switch_profile(self, profile: str) -> Dict[str, Any]:
        """
        Switch the working profile.

        Args:
            profile: Name of the profile to switch to

        Returns:
            Result of the switch
        """
        self.working_profile = profile
        # Ensure profile exists
        self.lockfile.get_profile(profile)
        return {
            "success": True,
            "profile": profile,
            "message": f"Switched to profile '{profile}'"
        }

    async def analyze_costs(
        self,
        profile: Optional[str] = None,
        monthly_volume: Optional[Dict[str, int]] = None,
        hypothetical_models: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze estimated costs for current or hypothetical configuration.

        Args:
            profile: Profile to analyze (default: current working profile)
            monthly_volume: Expected monthly token usage with 'input_tokens' and 'output_tokens'
            hypothetical_models: Optional dict of {alias: model_ref} to analyze what-if scenarios

        Returns:
            Cost analysis with breakdown and recommendations
        """
        profile = profile or self.working_profile

        if not monthly_volume:
            # Default estimate: 1M input, 500K output per month
            monthly_volume = {
                "input_tokens": 1_000_000,
                "output_tokens": 500_000
            }

        # Determine which models to analyze
        models_to_analyze = {}
        if hypothetical_models:
            # Use hypothetical models for what-if analysis
            models_to_analyze = hypothetical_models
        else:
            # Use actual profile bindings
            prof = self.lockfile.get_profile(profile)
            for binding in prof.bindings:
                models_to_analyze[binding.alias] = binding.model_ref

        cost_breakdown = {}
        total_cost = 0.0
        analysis_type = "hypothetical" if hypothetical_models else "actual"

        # Calculate costs for each model in models_to_analyze
        for alias, model_ref in models_to_analyze.items():
            try:
                # Parse provider and model from model_ref
                if ":" in model_ref:
                    provider, model_name = model_ref.split(":", 1)
                else:
                    # Assume it's just a model name, try to find provider
                    provider = None
                    model_name = model_ref
                    # Try to find which provider has this model
                    for prov in ["openai", "anthropic", "google", "ollama", "groq", "cohere"]:
                        try:
                            models = await self.registry.fetch_current_models(prov)
                            if any(m.model_name == model_name for m in models):
                                provider = prov
                                break
                        except:
                            continue

                if not provider:
                    logger.warning(f"Could not find provider for model {model_name}")
                    continue

                models = await self.registry.fetch_current_models(provider)
                model = next((m for m in models if m.model_name == model_name), None)

                if model and model.dollars_per_million_tokens_input:
                    input_cost = (monthly_volume["input_tokens"] / 1_000_000) * model.dollars_per_million_tokens_input
                    output_cost = (monthly_volume["output_tokens"] / 1_000_000) * model.dollars_per_million_tokens_output
                    alias_cost = input_cost + output_cost

                    cost_breakdown[alias] = {
                        "model": model_ref,
                        "input_cost": round(input_cost, 2),
                        "output_cost": round(output_cost, 2),
                        "total_cost": round(alias_cost, 2),
                        "pricing": {
                            "prompt": model.dollars_per_million_tokens_input,
                            "completion": model.dollars_per_million_tokens_output
                        }
                    }
                    total_cost += alias_cost
            except Exception as e:
                logger.warning(f"Could not calculate cost for {alias}: {e}")

        recommendations = []
        if total_cost > 100:
            recommendations.append("Consider using more cost-effective models for high-volume aliases")
        if total_cost < 10:
            recommendations.append("You have room to use more capable models if needed")

        return {
            "profile": profile,
            "analysis_type": analysis_type,
            "monthly_volume": monthly_volume,
            "cost_breakdown": cost_breakdown,
            "total_monthly_cost": round(total_cost, 2),
            "recommendations": recommendations,
            "models_analyzed": len(models_to_analyze)
        }

    async def get_current_configuration(self) -> Dict[str, Any]:
        """
        Get the complete current lockfile configuration.

        Returns:
            Current lockfile configuration
        """
        profiles = {}
        for profile_name in self.lockfile.profiles:
            profile = self.lockfile.get_profile(profile_name)
            profiles[profile_name] = {
                "bindings": [
                    {
                        "alias": b.alias,
                        "provider": b.provider,
                        "model": b.model,
                        "model_ref": b.model_ref
                    }
                    for b in profile.bindings
                ]
            }

        return {
            "version": self.lockfile.version,
            "default_profile": self.lockfile.default_profile,
            "profiles": profiles,
            "metadata": self.lockfile.metadata or {},
            "lockfile_path": str(self.lockfile_path)
        }