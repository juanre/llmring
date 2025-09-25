"""
MCP tools for conversational lockfile management.

Provides tools for managing lockfiles through natural conversation,
including adding/removing aliases, assessing models, and generating configurations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmring.lockfile_core import AliasBinding, Lockfile, ProfileConfig
from llmring.registry import RegistryClient

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

    async def add_alias(
        self,
        alias: str,
        model: Optional[str] = None,
        use_case: Optional[str] = None,
        profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add or update an alias in the lockfile.

        Args:
            alias: Name of the alias to add
            model: Model reference (provider:model), or None to auto-select
            use_case: Description of what the alias will be used for
            profile: Profile to add to, defaults to current working profile

        Returns:
            Result with the alias configuration
        """
        profile = profile or self.working_profile

        # If no model specified, recommend one based on use case
        if not model:
            model = await self._recommend_model_for_use_case(alias, use_case)

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
            model_ref: Model reference (provider:model)

        Returns:
            Model assessment with capabilities, pricing, and recommendations
        """
        if ":" not in model_ref:
            return {
                "error": f"Invalid model reference: {model_ref}. Use format provider:model"
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

            # Prepare assessment
            assessment = {
                "model": model_ref,
                "display_name": model_data.display_name,
                "active": model_data.is_active,
                "capabilities": {
                    "max_input_tokens": model_data.max_input_tokens,
                    "max_output_tokens": model_data.max_output_tokens,
                    "supports_vision": model_data.supports_vision,
                    "supports_functions": model_data.supports_function_calling,
                    "supports_json_mode": model_data.supports_json_mode
                },
                "pricing": {
                    "input_cost_per_million": model_data.dollars_per_million_tokens_input,
                    "output_cost_per_million": model_data.dollars_per_million_tokens_output
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

    async def recommend_alias(self, use_case: str) -> Dict[str, Any]:
        """
        Recommend an alias configuration for a specific use case.

        Args:
            use_case: Description of what the user wants to do

        Returns:
            Recommended alias configuration
        """
        use_case_lower = use_case.lower()

        # Analyze use case keywords
        recommendations = []

        # Check for specific patterns
        if any(word in use_case_lower for word in ["fast", "quick", "simple", "cheap"]):
            model = await self._find_cheapest_model()
            if model:
                recommendations.append({
                    "alias": "fast",
                    "model": model,
                    "reason": "Cost-effective model for quick tasks"
                })

        if any(word in use_case_lower for word in ["complex", "reasoning", "analysis", "deep"]):
            model = await self._find_most_capable_model()
            if model:
                recommendations.append({
                    "alias": "deep",
                    "model": model,
                    "reason": "High-capability model for complex reasoning"
                })

        if any(word in use_case_lower for word in ["vision", "image", "picture", "screenshot"]):
            model = await self._find_vision_model()
            if model:
                recommendations.append({
                    "alias": "vision",
                    "model": model,
                    "reason": "Vision-capable model for image analysis"
                })

        if any(word in use_case_lower for word in ["code", "coding", "programming"]):
            model = await self._find_coding_model()
            if model:
                recommendations.append({
                    "alias": "coder",
                    "model": model,
                    "reason": "Optimized for code generation and analysis"
                })

        if any(word in use_case_lower for word in ["write", "writing", "content", "article"]):
            model = await self._find_writing_model()
            if model:
                recommendations.append({
                    "alias": "writer",
                    "model": model,
                    "reason": "Optimized for content creation"
                })

        # Default balanced option if no specific match
        if not recommendations:
            model = await self._find_balanced_model()
            if model:
                recommendations.append({
                    "alias": "balanced",
                    "model": model,
                    "reason": "Balanced performance and cost for general use"
                })

        return {
            "use_case": use_case,
            "recommendations": recommendations
        }

    async def analyze_costs(self, profile: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze potential costs for current alias configuration.

        Args:
            profile: Profile to analyze, defaults to current working profile

        Returns:
            Cost analysis for the profile
        """
        profile = profile or self.working_profile
        profile_config = self.lockfile.get_profile(profile)

        costs = []
        total_min_cost = 0
        total_max_cost = 0

        for binding in profile_config.bindings:
            try:
                # Fetch model data
                models = await self.registry.fetch_current_models(binding.provider)
                model_data = None
                for m in models:
                    if m.model_name == binding.model:
                        model_data = m
                        break

                if model_data and model_data.dollars_per_million_tokens_input:
                    # Estimate costs (assuming 1k input, 0.5k output per request)
                    input_cost = (1000 / 1_000_000) * model_data.dollars_per_million_tokens_input
                    output_cost = (500 / 1_000_000) * (model_data.dollars_per_million_tokens_output or 0)
                    per_request = input_cost + output_cost

                    costs.append({
                        "alias": binding.alias,
                        "model": binding.model_ref,
                        "cost_per_request": round(per_request, 6),
                        "cost_per_1k_requests": round(per_request * 1000, 2)
                    })

                    total_min_cost += per_request
                    total_max_cost += per_request * 10  # Assume up to 10x for heavy use

            except Exception:
                # Skip if can't fetch model data
                pass

        return {
            "profile": profile,
            "alias_costs": costs,
            "estimated_monthly_cost": {
                "light_use": round(total_min_cost * 100, 2),  # 100 requests
                "moderate_use": round(total_min_cost * 1000, 2),  # 1k requests
                "heavy_use": round(total_min_cost * 10000, 2)  # 10k requests
            }
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

    # Helper methods for finding models
    async def _find_cheapest_model(self) -> Optional[str]:
        """Find the cheapest available model."""
        cheapest = None
        min_cost = float('inf')

        for provider in ["openai", "anthropic", "google"]:
            try:
                models = await self.registry.fetch_current_models(provider)
                for model in models:
                    if model.is_active and model.dollars_per_million_tokens_input:
                        if model.dollars_per_million_tokens_input < min_cost:
                            min_cost = model.dollars_per_million_tokens_input
                            cheapest = f"{provider}:{model.model_name}"
            except Exception:
                pass

        return cheapest

    async def _find_most_capable_model(self) -> Optional[str]:
        """Find the most capable model based on context length."""
        most_capable = None
        max_tokens = 0

        for provider in ["anthropic", "openai", "google"]:
            try:
                models = await self.registry.fetch_current_models(provider)
                for model in models:
                    if model.is_active and model.max_input_tokens:
                        if model.max_input_tokens > max_tokens:
                            max_tokens = model.max_input_tokens
                            most_capable = f"{provider}:{model.model_name}"
            except Exception:
                pass

        return most_capable

    async def _find_vision_model(self) -> Optional[str]:
        """Find a vision-capable model."""
        for provider in ["openai", "google", "anthropic"]:
            try:
                models = await self.registry.fetch_current_models(provider)
                for model in models:
                    if model.is_active and model.supports_vision:
                        return f"{provider}:{model.model_name}"
            except Exception:
                pass

        return None

    async def _find_coding_model(self) -> Optional[str]:
        """Find a model optimized for coding."""
        # Prefer models with function calling and high capability
        return await self._find_most_capable_model()

    async def _find_writing_model(self) -> Optional[str]:
        """Find a model optimized for writing."""
        # Balance between capability and cost
        return await self._find_balanced_model()

    async def _find_balanced_model(self) -> Optional[str]:
        """Find a balanced model."""
        # Look for mid-tier models
        for provider in ["openai", "anthropic", "google"]:
            try:
                models = await self.registry.fetch_current_models(provider)
                # Sort by cost and pick middle option
                active_models = [m for m in models if m.is_active and m.dollars_per_million_tokens_input]
                if len(active_models) > 1:
                    active_models.sort(key=lambda m: m.dollars_per_million_tokens_input)
                    mid_idx = len(active_models) // 2
                    model = active_models[mid_idx]
                    return f"{provider}:{model.model_name}"
            except Exception:
                pass

        return None

    async def _recommend_model_for_use_case(self, alias: str, use_case: Optional[str]) -> str:
        """Recommend a model based on alias name and use case."""
        if use_case:
            result = await self.recommend_alias(use_case)
            if result["recommendations"]:
                return result["recommendations"][0]["model"]

        # Fallback based on alias name
        if "fast" in alias.lower():
            return await self._find_cheapest_model() or "openai:gpt-4o-mini"
        elif "deep" in alias.lower() or "complex" in alias.lower():
            return await self._find_most_capable_model() or "anthropic:claude-3-opus"
        elif "vision" in alias.lower():
            return await self._find_vision_model() or "openai:gpt-4o"
        else:
            return await self._find_balanced_model() or "openai:gpt-4o"


def get_lockfile_tools() -> List[Dict[str, Any]]:
    """Get tool definitions for MCP server registration."""
    return [
        {
            "name": "add_alias",
            "description": "Add or update an alias in the lockfile",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "Name of the alias to add"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model reference (provider:model), or omit to auto-select"
                    },
                    "use_case": {
                        "type": "string",
                        "description": "Description of what the alias will be used for"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile to add to (default: 'default')"
                    }
                },
                "required": ["alias"]
            }
        },
        {
            "name": "remove_alias",
            "description": "Remove an alias from the lockfile",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "alias": {
                        "type": "string",
                        "description": "Name of the alias to remove"
                    },
                    "profile": {
                        "type": "string",
                        "description": "Profile to remove from"
                    }
                },
                "required": ["alias"]
            }
        },
        {
            "name": "list_aliases",
            "description": "List all aliases in a profile",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Profile to list"
                    }
                }
            }
        },
        {
            "name": "assess_model",
            "description": "Assess a model's capabilities, costs, and recommended use cases",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_ref": {
                        "type": "string",
                        "description": "Model reference to assess (provider:model)"
                    }
                },
                "required": ["model_ref"]
            }
        },
        {
            "name": "recommend_alias",
            "description": "Get alias recommendations for a specific use case",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "use_case": {
                        "type": "string",
                        "description": "Description of what you want to do"
                    }
                },
                "required": ["use_case"]
            }
        },
        {
            "name": "analyze_costs",
            "description": "Analyze potential costs for current configuration",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Profile to analyze"
                    }
                }
            }
        },
        {
            "name": "save_lockfile",
            "description": "Save the current lockfile state",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "switch_profile",
            "description": "Switch to a different profile",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "profile": {
                        "type": "string",
                        "description": "Profile name to switch to"
                    }
                },
                "required": ["profile"]
            }
        }
    ]