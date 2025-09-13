"""
Intelligent lockfile creation system.

Uses LLMRing's own API with MCP registry tools to create optimal lockfiles
through conversation and registry analysis.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from llmring.lockfile import AliasBinding, Lockfile, ProfileConfig
from llmring.registry import RegistryClient
from llmring.schemas import LLMRequest, Message
from llmring.service import LLMRing

logger = logging.getLogger(__name__)


class IntelligentLockfileCreator:
    """Creates lockfiles through intelligent conversation using LLMRing's own API."""

    def __init__(self, bootstrap_lockfile: str = "bootstrap.llmring.lock"):
        """
        Initialize with bootstrap lockfile for self-hosting.

        Args:
            bootstrap_lockfile: Path to bootstrap lockfile with advisor model
        """
        # Use bootstrap lockfile to power the system (self-hosting)
        self.service = LLMRing(lockfile_path=bootstrap_lockfile)
        self.registry = RegistryClient()
        self.conversation_state = {
            "user_needs": {},
            "registry_analysis": {},
            "recommendations": {},
            "selected_aliases": {}
        }

    async def create_lockfile_interactively(
        self,
        profile_name: str = "default",
        max_cost_per_turn: float = 0.50
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
        """Use advisor LLM to analyze current registry state."""
        # Use our own API with "advisor" alias from bootstrap lockfile
        request = LLMRequest(
            model="advisor",  # Self-hosted: uses bootstrap lockfile
            messages=[Message(
                role="system",
                content="""You are a registry analyst for LLMRing. Analyze the current state
                of all provider registries to understand the model landscape.

                Use the available tools to:
                1. Get current models for each provider
                2. Identify the best models in key categories:
                   - Most capable (reasoning, complex tasks)
                   - Most cost-effective (good performance per dollar)
                   - Fastest response (low latency)
                   - Best specialized capabilities (vision, function calling, etc.)

                Respond with structured analysis in JSON format with your findings."""
            ), Message(
                role="user",
                content="Analyze the current registry and categorize the best available models"
            )],
            tools=self._get_registry_tools(),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "registry_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "analysis_date": {"type": "string"},
                            "providers": {
                                "type": "object",
                                "properties": {
                                    "openai": {"type": "object"},
                                    "anthropic": {"type": "object"},
                                    "google": {"type": "object"}
                                }
                            },
                            "recommendations": {
                                "type": "object",
                                "properties": {
                                    "most_capable": {"type": "string"},
                                    "most_cost_effective": {"type": "string"},
                                    "fastest": {"type": "string"},
                                    "best_vision": {"type": "string"}
                                }
                            }
                        },
                        "required": ["analysis_date", "providers", "recommendations"]
                    }
                },
                "strict": True
            }
        )

        response = await self.service.chat(request)
        self.conversation_state["registry_analysis"] = response.parsed

    async def _discover_user_needs(self):
        """Interactive conversation to understand user requirements."""
        # Multi-turn conversation using advisor
        conversation_messages = [Message(
            role="system",
            content=f"""You are an expert LLM configuration advisor. Help users create
            optimal lockfile configurations based on their specific needs.

            Current registry analysis: {json.dumps(self.conversation_state['registry_analysis'], indent=2)}

            Your job:
            1. Ask focused questions to understand their use cases
            2. Understand their budget preferences and usage patterns
            3. Identify required capabilities (vision, function calling, etc.)
            4. Determine optimal alias configuration for their workflow

            Ask 2-3 targeted questions total. Be conversational and helpful.
            End with a summary of their needs in structured format."""
        )]

        # Simulated conversation for now (in CLI implementation, this would be interactive)
        # For the design, assume user has mixed use cases with balanced budget
        user_responses = [
            "I need LLMs for data analysis, some creative writing, and general Q&A. Budget is important but I want good quality.",
            "I'd like vision capabilities for document processing, and I do expect moderate usage - maybe 100-200 requests per month.",
            "I prefer reliable, established models over cutting-edge experimental ones."
        ]

        for user_response in user_responses:
            conversation_messages.append(Message(role="user", content=user_response))

            request = LLMRequest(
                model="advisor",
                messages=conversation_messages,
                max_tokens=300,
                temperature=0.7
            )

            response = await self.service.chat(request)
            conversation_messages.append(Message(role="assistant", content=response.content))

        self.conversation_state["user_needs"] = {
            "use_cases": ["data_analysis", "creative_writing", "general_qa", "document_processing"],
            "budget_preference": "balanced",
            "required_capabilities": ["vision", "function_calling"],
            "usage_volume": "moderate",
            "stability_preference": "established"
        }

    async def _generate_recommendations(self):
        """Generate structured alias recommendations."""
        request = LLMRequest(
            model="advisor",
            messages=[Message(
                role="system",
                content=f"""You are creating optimal alias recommendations for a lockfile.

                Registry Analysis: {json.dumps(self.conversation_state['registry_analysis'], indent=2)}
                User Needs: {json.dumps(self.conversation_state['user_needs'], indent=2)}

                Create 5-6 semantic aliases that match the user's workflow.
                For each alias, recommend the optimal model from the registry analysis.

                Respond with structured recommendations."""
            ), Message(
                role="user",
                content="Generate optimal alias configuration based on the analysis and user needs"
            )],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "recommendations",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "aliases": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "alias": {"type": "string"},
                                        "provider": {"type": "string"},
                                        "model": {"type": "string"},
                                        "rationale": {"type": "string"},
                                        "use_cases": {"type": "array", "items": {"type": "string"}},
                                        "estimated_cost_per_request": {"type": "number"}
                                    },
                                    "required": ["alias", "provider", "model", "rationale"]
                                }
                            },
                            "total_estimated_monthly_cost": {"type": "number"},
                            "coverage_analysis": {"type": "string"}
                        },
                        "required": ["aliases"]
                    }
                },
                "strict": True
            }
        )

        response = await self.service.chat(request)
        self.conversation_state["recommendations"] = response.parsed

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
                constraints={}  # Could add temperature, max_tokens, etc.
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
            "coverage_analysis": recommendations.get("coverage_analysis")
        }

        return lockfile

    def _get_registry_tools(self) -> List[Dict[str, Any]]:
        """Get MCP tools for registry analysis."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_provider_models",
                    "description": "Get all active models for a provider with capabilities and pricing",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "provider": {
                                "type": "string",
                                "enum": ["openai", "anthropic", "google", "ollama"]
                            }
                        },
                        "required": ["provider"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "compare_models_by_cost",
                    "description": "Compare models by cost efficiency",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_refs": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["model_refs"]
                    }
                }
            }
        ]


async def create_intelligent_lockfile(
    output_path: str = "llmring.lock",
    bootstrap_lockfile: str = "bootstrap.llmring.lock"
) -> bool:
    """
    Create an intelligent lockfile using the conversation system.

    Args:
        output_path: Where to save the generated lockfile
        bootstrap_lockfile: Bootstrap lockfile for powering the advisor

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
            print(f"  {alias_config['alias']:<12} → {alias_config['provider']}:{alias_config['model']}")
            print(f"               Rationale: {alias_config['rationale']}")

        return True

    except Exception as e:
        logger.error(f"Failed to create intelligent lockfile: {e}")
        print(f"❌ Error: {e}")
        return False