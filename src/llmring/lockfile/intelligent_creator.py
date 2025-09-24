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
        self, profile_name: str = "default", requirements_text: str = None
    ) -> Lockfile:
        """
        Guide user through intelligent lockfile creation.

        Args:
            profile_name: Profile to create
            requirements_text: Optional requirements text (from file or CLI)

        Returns:
            Optimized lockfile with rationale metadata
        """
        logger.info("Starting intelligent lockfile creation")

        # Phase 1: Analyze current registry
        print("🤖 Analyzing current registry across all providers...")
        await self._analyze_registry()

        # Phase 2: Understand user needs through conversation or text
        await self._discover_user_needs(requirements_text)

        # Phase 3: Generate structured recommendations
        print("\n🎯 Generating optimal recommendations...")
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

    async def _discover_user_needs(self, requirements_text: str = None):
        """Discover user needs through interactive prompts or provided text."""

        print("\n📝 Understanding your requirements...")

        # If requirements provided as text, parse them
        if requirements_text:
            print(f"   Using provided requirements: {requirements_text[:100]}...")
            self._parse_requirements_text(requirements_text)
            return

        # Otherwise, ask the user interactively
        print("\nI'll ask you a few questions to create the optimal configuration.")
        print("Press Enter for default values.\n")

        # Question 1: Use cases
        use_cases_input = input("1. What will you primarily use LLMs for?\n   (e.g., 'coding, writing, analysis' or press Enter for general use): ").strip()
        if not use_cases_input:
            use_cases = ["general_qa", "data_analysis", "document_processing"]
            print("   → Using default: general question answering, data analysis, document processing")
        else:
            use_cases = [uc.strip() for uc in use_cases_input.split(",")]
            print(f"   → Selected use cases: {', '.join(use_cases)}")

        # Question 2: Budget preference
        print("\n2. What's your budget preference?")
        print("   a) low_cost - Minimize costs, OK with reduced capability")
        print("   b) balanced - Balance cost and performance (default)")
        print("   c) performance - Maximize capability, cost is secondary")
        budget_input = input("   Choice (a/b/c or Enter for balanced): ").strip().lower()

        budget_map = {
            'a': 'low_cost',
            'b': 'balanced',
            'c': 'performance',
            '': 'balanced'
        }
        budget_preference = budget_map.get(budget_input, 'balanced')
        print(f"   → Selected: {budget_preference}")

        # Question 3: Required capabilities
        print("\n3. Which capabilities do you need? (comma-separated)")

        # Show available capabilities from registry
        available_caps = set()
        for provider_data in self.conversation_state["registry_analysis"]["providers"].values():
            if isinstance(provider_data, dict) and "models" in provider_data:
                for model in provider_data["models"]:
                    if model.get("supports_vision"):
                        available_caps.add("vision")
                    if model.get("supports_functions"):
                        available_caps.add("function_calling")

        if available_caps:
            print(f"   Available: {', '.join(sorted(available_caps))}")

        caps_input = input("   Enter capabilities or press Enter for auto-detect: ").strip()
        if not caps_input:
            capabilities = list(available_caps)
            print(f"   → Auto-detected: {', '.join(capabilities) if capabilities else 'basic text only'}")
        else:
            capabilities = [cap.strip() for cap in caps_input.split(",")]
            print(f"   → Selected: {', '.join(capabilities)}")

        # Question 4: Usage volume
        print("\n4. Expected usage volume?")
        print("   a) low - Less than 100 requests/month")
        print("   b) moderate - 100-1000 requests/month (default)")
        print("   c) high - More than 1000 requests/month")
        volume_input = input("   Choice (a/b/c or Enter for moderate): ").strip().lower()

        volume_map = {
            'a': 'low',
            'b': 'moderate',
            'c': 'high',
            '': 'moderate'
        }
        usage_volume = volume_map.get(volume_input, 'moderate')
        print(f"   → Selected: {usage_volume}")

        # Question 5: Specific aliases needed
        print("\n5. Any specific aliases you need?")
        print("   Common aliases: fast, deep, vision, coder, writer")
        aliases_input = input("   Enter aliases or press Enter to auto-generate: ").strip()

        if aliases_input:
            requested_aliases = [a.strip() for a in aliases_input.split(",")]
            print(f"   → Will include: {', '.join(requested_aliases)}")
        else:
            requested_aliases = []
            print("   → Will auto-generate based on your needs")

        # Store user needs
        self.conversation_state["user_needs"] = {
            "use_cases": use_cases,
            "budget_preference": budget_preference,
            "required_capabilities": capabilities,
            "usage_volume": usage_volume,
            "stability_preference": "established",
            "requested_aliases": requested_aliases
        }

        print("\n✓ Requirements gathered successfully!")

    def _parse_requirements_text(self, text: str):
        """Parse requirements from provided text."""
        # Simple parsing - can be enhanced with more sophisticated NLP
        text_lower = text.lower()

        # Detect use cases
        use_cases = []
        if "coding" in text_lower or "code" in text_lower:
            use_cases.append("coding")
        if "writing" in text_lower or "content" in text_lower:
            use_cases.append("writing")
        if "analysis" in text_lower or "data" in text_lower:
            use_cases.append("data_analysis")
        if not use_cases:
            use_cases = ["general_qa"]

        # Detect budget preference
        if "cheap" in text_lower or "low cost" in text_lower or "budget" in text_lower:
            budget = "low_cost"
        elif "performance" in text_lower or "best" in text_lower or "powerful" in text_lower:
            budget = "performance"
        else:
            budget = "balanced"

        # Detect capabilities
        capabilities = []
        if "vision" in text_lower or "image" in text_lower:
            capabilities.append("vision")
        if "function" in text_lower or "tool" in text_lower:
            capabilities.append("function_calling")

        # Store parsed needs
        self.conversation_state["user_needs"] = {
            "use_cases": use_cases,
            "budget_preference": budget,
            "required_capabilities": capabilities,
            "usage_volume": "moderate",
            "stability_preference": "established",
            "requested_aliases": []
        }

    async def _generate_recommendations(self):
        """Generate structured alias recommendations based on registry analysis and user needs."""

        recommendations = {
            "aliases": [],
            "total_estimated_monthly_cost": 0,
            "coverage_analysis": "Customized configuration based on your requirements"
        }

        analysis = self.conversation_state["registry_analysis"]
        user_needs = self.conversation_state["user_needs"]
        budget_pref = user_needs.get("budget_preference", "balanced")
        requested_aliases = user_needs.get("requested_aliases", [])

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

        # Get recommendations from analysis
        recs = analysis.get("recommendations", {})

        # Build aliases based on user preferences
        if budget_pref == "low_cost":
            # Prioritize cost-effective models
            if recs.get("most_cost_effective"):
                add_recommendation(
                    "primary",
                    recs["most_cost_effective"],
                    "Primary model optimized for low cost as requested",
                    user_needs.get("use_cases", ["general_qa"])
                )
                add_recommendation(
                    "fast",
                    recs["most_cost_effective"],
                    "Fast, cost-effective model for simple tasks",
                    ["simple_tasks"]
                )
        elif budget_pref == "performance":
            # Prioritize high-capability models
            if recs.get("most_capable"):
                add_recommendation(
                    "primary",
                    recs["most_capable"],
                    "Primary model with maximum capability as requested",
                    user_needs.get("use_cases", ["general_qa"])
                )
                add_recommendation(
                    "deep",
                    recs["most_capable"],
                    "Deep reasoning model for complex tasks",
                    ["complex_analysis"]
                )
        else:  # balanced
            # Mix of capabilities
            if recs.get("most_cost_effective"):
                add_recommendation(
                    "fast",
                    recs["most_cost_effective"],
                    "Cost-effective model for simple tasks",
                    ["simple_tasks"]
                )
            if recs.get("most_capable"):
                add_recommendation(
                    "deep",
                    recs["most_capable"],
                    "High-capability model for complex tasks",
                    ["complex_analysis"]
                )

        # Add requested aliases
        for alias in requested_aliases:
            if alias == "vision" and recs.get("best_vision"):
                add_recommendation(
                    "vision",
                    recs["best_vision"],
                    "Vision model as specifically requested",
                    ["image_analysis", "visual_understanding"]
                )
            elif alias == "coder" and recs.get("most_capable"):
                add_recommendation(
                    "coder",
                    recs["most_capable"],
                    "Code generation and analysis model",
                    ["coding", "code_review"]
                )
            elif alias == "writer":
                # Use balanced model for writing
                model = recs.get("most_capable") or recs.get("most_cost_effective")
                if model:
                    add_recommendation(
                        "writer",
                        model,
                        "Content creation and writing model",
                        ["writing", "content_creation"]
                    )
            elif alias not in [a["alias"] for a in recommendations["aliases"]]:
                # Generic alias - use best available
                model = recs.get("most_capable") or recs.get("most_cost_effective")
                if model:
                    add_recommendation(
                        alias,
                        model,
                        f"Custom alias '{alias}' as requested",
                        ["custom"]
                    )

        # Add capability-based aliases
        if "vision" in user_needs.get("required_capabilities", []):
            if recs.get("best_vision") and "vision" not in [a["alias"] for a in recommendations["aliases"]]:
                add_recommendation(
                    "vision",
                    recs["best_vision"],
                    "Vision-capable model for image processing",
                    ["image_analysis"]
                )

        # Ensure we have at least a default/balanced option
        if not recommendations["aliases"]:
            # Fallback to any available model
            for provider_name, provider_data in analysis["providers"].items():
                if isinstance(provider_data, dict) and "models" in provider_data and provider_data["models"]:
                    model = provider_data["models"][0]
                    add_recommendation(
                        "default",
                        f"{provider_name}:{model['name']}",
                        "Default model based on available options",
                        ["general_qa"]
                    )
                    break

        self.conversation_state["recommendations"] = recommendations

        print(f"   ✓ Generated {len(recommendations['aliases'])} alias configurations")
        print(f"   ✓ Optimized for: {budget_pref} budget preference")

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
