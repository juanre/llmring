"""
Example FastAPI application with proper lifecycle management for LLM service.

Requirements:
    uv add --dev fastapi uvicorn

Run with:
    uvicorn examples.fastapi_app:app --reload

Test endpoints:
    http://localhost:8000/health
    http://localhost:8000/docs
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException
from llmring import LLMRing
from llmring.schemas import Message
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat completion request."""

    messages: list[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    id_at_origin: str = "anonymous"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    database: Dict[str, Any]
    providers: list[str]


# Global LLM service instance
llmring: LLMRing = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llmring

    # Startup
    print("🚀 Starting LLM Service API...")

    # Initialize LLM service with database
    db_url = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres"
    )
    llmring = LLMRing(
        db_connection_string=db_url,
        origin="llm-api",
        enable_db_logging=True,
    )

    # Initialize database connection and apply migrations
    if llmring.db:
        await llmring.db.initialize()
        result = await llmring.db.apply_migrations()
        print(f"✅ Database connected and migrations applied: {result}")

        # Get initial pool stats
        pool_stats = await llmring.db.get_pool_stats()
        print(f"📊 Connection pool initialized: {pool_stats}")

    print("✅ LLM Service ready")

    yield

    # Shutdown
    print("🛑 Shutting down LLM Service API...")

    # Close LLM service (which closes database connections)
    await llmring.close()

    print("✅ Cleanup complete")


# Create FastAPI app with lifecycle management
app = FastAPI(
    title="LLM Service API",
    description="Production-ready LLM service with async database support",
    version="1.0.0",
    lifespan=lifespan,
)


async def get_llmring() -> LLMRing:
    """Dependency to get LLM service instance."""
    if not llmring:
        raise HTTPException(500, "LLM service not initialized")
    return llmring


@app.get("/health", response_model=HealthResponse)
async def health_check(service: LLMRing = Depends(get_llmring)):
    """Health check endpoint with database status."""
    try:
        # Get database health
        db_health = {"status": "disabled"}
        if service.db:
            db_health = await service.db.health_check()

        # Get available providers
        providers = []
        for provider_name in ["openai", "anthropic", "google", "ollama"]:
            provider = service._get_provider(provider_name)
            if provider:
                providers.append(provider_name)

        return HealthResponse(
            status="healthy" if db_health.get("status") == "healthy" else "degraded",
            database=db_health,
            providers=providers,
        )
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")


@app.get("/metrics")
async def get_metrics(service: LLMRing = Depends(get_llmring)):
    """Get service metrics including database performance."""
    if not service.db:
        return {"error": "Database not configured"}

    try:
        # Get pool stats
        pool_stats = await service.db.get_pool_stats()

        # Get query metrics if monitoring is enabled
        query_metrics = await service.db.get_query_metrics()

        # Get slow queries
        slow_queries = await service.db.get_slow_queries()

        return {
            "pool": pool_stats,
            "queries": query_metrics,
            "slow_queries": len(slow_queries) if slow_queries else 0,
            "slow_query_samples": slow_queries[:5] if slow_queries else [],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get metrics: {str(e)}")


@app.post("/chat")
async def chat_completion(
    request: ChatRequest, service: LLMRing = Depends(get_llmring)
):
    """Chat completion endpoint."""
    try:
        # Convert messages
        messages = [
            Message(role=msg["role"], content=msg["content"])
            for msg in request.messages
        ]

        # Call LLM
        response = await service.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            id_at_origin=request.id_at_origin,
        )

        return {
            "content": response.content,
            "model": response.model,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
            "tool_calls": response.tool_calls,
        }

    except Exception as e:
        raise HTTPException(500, f"Chat completion failed: {str(e)}")


@app.get("/models")
async def list_models(
    provider: str = None, service: LLMRing = Depends(get_llmring)
):
    """List available models."""
    try:
        if service.db:
            models = await service.get_models_from_db(provider=provider)
            return {
                "models": [
                    {
                        "provider": m.provider,
                        "model": m.model_name,
                        "display_name": m.display_name,
                        "max_context": m.max_context,
                        "supports_vision": m.supports_vision,
                        "supports_tools": m.supports_function_calling,
                    }
                    for m in models
                ]
            }
        else:
            return {"error": "Database not configured"}
    except Exception as e:
        raise HTTPException(500, f"Failed to list models: {str(e)}")


@app.get("/usage/{user_id}")
async def get_usage_stats(
    user_id: str, days: int = 30, service: LLMRing = Depends(get_llmring)
):
    """Get usage statistics for a user."""
    try:
        stats = await service.get_usage_stats(user_id, days=days)
        if not stats:
            return {"message": "No usage data found"}

        return {
            "user_id": user_id,
            "period_days": days,
            "stats": {
                "total_calls": stats.total_calls,
                "total_tokens": stats.total_tokens,
                "total_cost": stats.total_cost,
                "avg_response_time_ms": stats.avg_response_time_ms,
                "success_rate": stats.success_rate,
                "most_used_model": stats.most_used_model,
                "providers_used": stats.providers_used,
                "models_used": stats.models_used,
            },
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get usage stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run with proper async server
    uvicorn.run(
        "fastapi_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
