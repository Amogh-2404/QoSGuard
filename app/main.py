"""
QoSGuard FastAPI Application
Main entry point for the network anomaly detection and QoS recommendation API.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import structlog

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.models.ml_models import ModelRegistry


logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting QoSGuard API server...")
    
    # Initialize model registry
    model_registry = ModelRegistry()
    await model_registry.load_models()
    app.state.model_registry = model_registry
    
    logger.info("QoSGuard API server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down QoSGuard API server...")


# Create FastAPI application
app = FastAPI(
    title="QoSGuard API",
    description="Real-time network anomaly detection and QoS recommendation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Setup logging
setup_logging()

# Get settings
settings = get_settings()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": "QoSGuard API",
        "version": "1.0.0",
        "status": "healthy",
        "docs": "/docs"
    }


@app.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring."""
    try:
        # Check if models are loaded
        model_registry = getattr(app.state, 'model_registry', None)
        models_loaded = model_registry.is_loaded() if model_registry else False
        
        return {
            "status": "healthy",
            "models_loaded": models_loaded,
            "timestamp": asyncio.get_event_loop().time()
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service unhealthy")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our custom logging
    )
