"""
Example FastAPI application for LLM service.

Requirements:
    uv add --dev fastapi uvicorn

Run with:
    uvicorn examples.fastapi_app:app --reload

Test endpoints:
    http://localhost:8000/health
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from llmring import LLMRing
from llmring.api.types import ChatRequest, ChatResponse, ServiceHealth
from llmring.schemas import LLMRequest, Message

# Global LLM service instance
llmring: LLMRing = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global llmring

    # Startup
    print("🚀 Starting LLM Service API...")

    # Initialize LLM service
    llmring = LLMRing(origin="llm-api")

    print("✅ LLM Service ready")

    yield

    # Shutdown
    print("🛑 Shutting down LLM Service API...")

    # Close LLM service
    await llmring.close()

    print("✅ Cleanup complete")


# Create FastAPI app with lifecycle management
app = FastAPI(
    title="LLM Service API",
    description="Lightweight LLM service for routing requests to providers",
    version="1.0.0",
    lifespan=lifespan,
)


async def get_llmring() -> LLMRing:
    """Dependency to get LLM service instance."""
    if not llmring:
        raise HTTPException(500, "LLM service not initialized")
    return llmring


@app.get("/health", response_model=ServiceHealth)
async def health_check(service: LLMRing = Depends(get_llmring)):
    """Health check endpoint."""
    try:
        # Get available providers
        providers = []
        for provider_name in ["openai", "anthropic", "google", "ollama"]:
            try:
                provider = service.get_provider(provider_name)
                if provider:
                    providers.append(provider_name)
            except Exception:
                pass

        return ServiceHealth(
            status="healthy" if providers else "unhealthy",
            providers=providers,
        )
    except Exception as e:
        raise HTTPException(500, f"Health check failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, service: LLMRing = Depends(get_llmring)):
    """Chat completion endpoint."""
    try:
        # Convert messages
        messages = [Message(role=msg["role"], content=msg["content"]) for msg in request.messages]

        # Create LLM request
        llm_request = LLMRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
        )

        # Call LLM
        response = await service.chat(llm_request)

        return ChatResponse(
            content=response.content,
            model=response.model,
            usage=response.usage,
            finish_reason=response.finish_reason,
            tool_calls=response.tool_calls,
        )

    except Exception as e:
        raise HTTPException(500, f"Chat completion failed: {str(e)}")


@app.get("/models")
async def list_models(provider: str = None, service: LLMRing = Depends(get_llmring)):
    """List available models."""
    try:
        models = service.get_available_models()

        if provider:
            # Filter by provider
            models = {k: v for k, v in models.items() if k == provider}

        return {"models": models}
    except Exception as e:
        raise HTTPException(500, f"Failed to list models: {str(e)}")


@app.get("/providers")
async def list_providers(service: LLMRing = Depends(get_llmring)):
    """List configured providers."""
    try:
        providers_info = []

        for provider_name in ["openai", "anthropic", "google", "ollama"]:
            try:
                provider = service.get_provider(provider_name)
                configured = provider is not None
            except Exception:
                configured = False

            providers_info.append(
                {
                    "provider": provider_name,
                    "configured": configured,
                    "models": (
                        service.get_available_models().get(provider_name, []) if configured else []
                    ),
                }
            )

        return {"providers": providers_info}
    except Exception as e:
        raise HTTPException(500, f"Failed to list providers: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    # Run with proper async server
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
