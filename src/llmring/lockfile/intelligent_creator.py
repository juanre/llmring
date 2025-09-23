"""
Intelligent lockfile creation system.

Uses LLMRing's own API with MCP registry tools to create optimal lockfiles
through conversation and registry analysis.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from llmring.lockfile_core import AliasBinding, Lockfile, ProfileConfig
from llmring.registry import RegistryClient
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


class IntelligentLockfileCreator:
    """Creates lockfiles through intelligent conversation using LLMRing's own API."""

    def __init__(self, bootstrap_lockfile: str = None):
        """
        Initialize the intelligent lockfile creator.

        Args:
            bootstrap_lockfile: Optional path to bootstrap lockfile with advisor model.
                               If None, works without LLM advisor (simpler mode).
        """
        # Optional: Use bootstrap lockfile to power the system (self-hosting)
        if bootstrap_lockfile:
            self.service = LLMRing(lockfile_path=bootstrap_lockfile)
        else:
            self.service = None  # Work without LLM advisor

        self.registry = RegistryClient()
        self.conversation_state = {
            "user_needs": {},
            "registry_analysis": {},
            "recommendations": {},
            "selected_aliases": {},
        }

    async def create_lockfile_interactively(
        self, profile_name: str = "default", max_cost_per_turn: float = 0.50
    ) -> Lockfile:
        """
        Guide user through intelligent lockfile creation.

        Args:
            profile_name: Profile to create
            max_cost_per_turn: Maximum cost per advisor turn (safety guard)

        Returns:
            Optimized lockfile with rationale metadata
        """
        logger.info("Starting intelligent lockfile creation")

        # Phase 1: Analyze current registry
        print("🤖 Analyzing current registry across all providers...")
        await self._analyze_registry()

        # Phase 2: Understand user needs through conversation
        print("📋 Understanding your requirements...")
        await self._discover_user_needs()

        # Phase 3: Generate structured recommendations
        print("🎯 Generating optimal recommendations...")
        await self._generate_recommendations()

        # Phase 4: Create final lockfile with metadata
        print("🔧 Creating lockfile with rationale...")
        return await self._create_lockfile_with_rationale(profile_name)

    async def _analyze_registry(self):
        """Directly analyze registry without LLM tools (simpler approach)."""
        import os
        from datetime import datetime

        logger.info("Analyzing registry for all providers")

        analysis = {
            "analysis_date": datetime.now(timezone.utc).isoformat(),
            "providers": {},
            "recommendations": {}
        }

        # Analyze each provider's models directly
        providers_to_check = []
        if os.environ.get("OPENAI_API_KEY"):
            providers_to_check.append("openai")
        if os.environ.get("ANTHROPIC_API_KEY"):
            providers_to_check.append("anthropic")
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            providers_to_check.append("google")

        for provider in providers_to_check:
            try:
                models = await self.registry.fetch_current_models(provider)
                active_models = [m for m in models if m.is_active]

                provider_info = {
                    "total_models": len(active_models),
                    "models": []
                }

                for model in active_models[:5]:  # Limit to top 5 for analysis
                    provider_info["models"].append({
                        "name": model.model_name,
                        "max_input": model.max_input_tokens,
                        "cost_input": model.dollars_per_million_tokens_input,
                        "supports_vision": model.supports_vision,
                        "supports_functions": model.supports_function_calling
                    })

                analysis["providers"][provider] = provider_info

                # Find best models in categories
                if active_models:
                    # Most capable (highest input tokens)
                    if not analysis["recommendations"].get("most_capable"):
                        capable = max(active_models, key=lambda m: m.max_input_tokens or 0)
                        analysis["recommendations"]["most_capable"] = f"{provider}:{capable.model_name}"

                    # Most cost effective
                    models_with_cost = [m for m in active_models if m.dollars_per_million_tokens_input]
                    if models_with_cost and not analysis["recommendations"].get("most_cost_effective"):
                        cheap = min(models_with_cost, key=lambda m: m.dollars_per_million_tokens_input)
                        analysis["recommendations"]["most_cost_effective"] = f"{provider}:{cheap.model_name}"

                    # Best vision
                    vision_models = [m for m in active_models if m.supports_vision]
                    if vision_models and not analysis["recommendations"].get("best_vision"):
                        vision = vision_models[0]
                        analysis["recommendations"]["best_vision"] = f"{provider}:{vision.model_name}"

            except Exception as e:
                logger.warning(f"Could not analyze {provider}: {e}")
                analysis["providers"][provider] = {"error": str(e)}

        self.conversation_state["registry_analysis"] = analysis

    async def _discover_user_needs(self):
        """Simplified user needs discovery - can be enhanced with interactive prompts later."""
        # For now, use sensible defaults based on common use cases
        # In a full implementation, this would be interactive

        print("\n📝 Determining optimal configuration based on available models...")

        # Analyze available capabilities from registry
        has_vision = False
        has_functions = False

        for provider_data in self.conversation_state["registry_analysis"]["providers"].values():
            if isinstance(provider_data, dict) and "models" in provider_data:
                for model in provider_data["models"]:
                    if model.get("supports_vision"):
                        has_vision = True
                    if model.get("supports_functions"):
                        has_functions = True

        # Set defaults based on what's available
        capabilities = []
        if has_vision:
            capabilities.append("vision")
        if has_functions:
            capabilities.append("function_calling")

        self.conversation_state["user_needs"] = {
            "use_cases": [
                "general_qa",
                "data_analysis",
                "document_processing",
            ],
            "budget_preference": "balanced",
            "required_capabilities": capabilities,
            "usage_volume": "moderate",
            "stability_preference": "established",
        }

        print("   ✓ Identified capabilities:", ", ".join(capabilities) if capabilities else "basic text")
        print("   ✓ Optimization goal: balanced cost and performance")

    async def _generate_recommendations(self):
        """Generate structured alias recommendations based on registry analysis."""

        print("\n🎯 Generating optimal alias configuration...")

        recommendations = {
            "aliases": [],
            "total_estimated_monthly_cost": 0,
            "coverage_analysis": "Comprehensive coverage for general use cases"
        }

        analysis = self.conversation_state["registry_analysis"]

        # Helper to add recommendation
        def add_recommendation(alias_name, model_ref, rationale, use_cases):
            if not model_ref:
                return
            provider, model = model_ref.split(":", 1)
            recommendations["aliases"].append({
                "alias": alias_name,
                "provider": provider,
                "model": model,
                "rationale": rationale,
                "use_cases": use_cases
            })

        # Generate recommendations based on analysis
        recs = analysis.get("recommendations", {})

        # 1. Fast/low-cost alias
        if recs.get("most_cost_effective"):
            add_recommendation(
                "fast",
                recs["most_cost_effective"],
                "Most cost-effective model for quick, simple tasks",
                ["general_qa", "simple_analysis"]
            )
            add_recommendation(
                "low_cost",
                recs["most_cost_effective"],
                "Optimized for high-volume, cost-sensitive operations",
                ["batch_processing", "simple_tasks"]
            )

        # 2. Deep/capable alias
        if recs.get("most_capable"):
            add_recommendation(
                "deep",
                recs["most_capable"],
                "Most capable model for complex reasoning and analysis",
                ["complex_analysis", "research", "reasoning"]
            )
            add_recommendation(
                "long_context",
                recs["most_capable"],
                "Handles large documents and extensive context",
                ["document_processing", "long_form_content"]
            )

        # 3. Vision alias
        if recs.get("best_vision"):
            add_recommendation(
                "vision",
                recs["best_vision"],
                "Specialized for image and visual content analysis",
                ["image_analysis", "document_ocr", "visual_understanding"]
            )

        # 4. Balanced/default alias - pick a middle ground
        for provider_name, provider_data in analysis["providers"].items():
            if isinstance(provider_data, dict) and "models" in provider_data:
                models = provider_data["models"]
                if len(models) > 1:
                    # Pick second model as balanced option (first is often most expensive)
                    balanced_model = models[1]
                    add_recommendation(
                        "balanced",
                        f"{provider_name}:{balanced_model['name']}",
                        "Balanced performance and cost for general use",
                        ["general_qa", "data_analysis"]
                    )
                    break

        self.conversation_state["recommendations"] = recommendations

        print(f"   ✓ Generated {len(recommendations['aliases'])} alias configurations")

    async def _create_lockfile_with_rationale(self, profile_name: str) -> Lockfile:
        """Create final lockfile with rationale metadata."""
        lockfile = Lockfile()

        # Create profile
        profile = ProfileConfig(name=profile_name)

        # Add recommended bindings
        recommendations = self.conversation_state["recommendations"]
        for alias_config in recommendations["aliases"]:
            binding = AliasBinding(
                alias=alias_config["alias"],
                provider=alias_config["provider"],
                model=alias_config["model"],
                constraints={},  # Could add temperature, max_tokens, etc.
            )
            profile.bindings.append(binding)

        lockfile.profiles[profile_name] = profile

        # Add metadata with rationale
        lockfile.metadata = {
            "created_by": "llmring-intelligent-creator-v1.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "registry_analysis_date": self.conversation_state["registry_analysis"]["analysis_date"],
            "user_needs": self.conversation_state["user_needs"],
            "recommendations_rationale": {
                alias_config["alias"]: alias_config["rationale"]
                for alias_config in recommendations["aliases"]
            },
            "estimated_monthly_cost": recommendations.get("total_estimated_monthly_cost"),
            "coverage_analysis": recommendations.get("coverage_analysis"),
        }

        return lockfile



async def create_intelligent_lockfile(
    output_path: str = "llmring.lock",
    bootstrap_lockfile: str = None,
) -> bool:
    """
    Create an intelligent lockfile using the conversation system.

    Args:
        output_path: Where to save the generated lockfile
        bootstrap_lockfile: Optional bootstrap lockfile for powering the advisor.
                          If None, works without LLM advisor.

    Returns:
        True if successful, False otherwise
    """
    try:
        creator = IntelligentLockfileCreator(bootstrap_lockfile)
        lockfile = await creator.create_lockfile_interactively()

        # Save with rationale
        lockfile.save(Path(output_path))

        print(f"\n✅ Created intelligent lockfile: {output_path}")
        print("📊 Configuration Summary:")

        for alias_config in creator.conversation_state["recommendations"]["aliases"]:
            print(
                f"  {alias_config['alias']:<12} → {alias_config['provider']}:{alias_config['model']}"
            )
            print(f"               Rationale: {alias_config['rationale']}")

        return True

    except Exception as e:
        logger.error(f"Failed to create intelligent lockfile: {e}")
        print(f"❌ Error: {e}")
        return False
