"""
main.py
Punto de entrada de la aplicación FastAPI.
Monta routers, inicializa modelos y conecta DB.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.databases import router as databases_router
from backend.api.feedback import router as feedback_router
from backend.api.ingest import router as ingest_router
from backend.api.models import router as models_router
from backend.api.query import router as query_router
from backend.deps import (
    check_db_health,
    check_models_health,
    get_settings,
    shutdown,
    startup,
)
from backend.settings import get_rag_config


# ==========================================
# Lifespan
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida de la aplicación.
    
    - Startup: Conecta a DB, inicializa modelos
    - Shutdown: Cierra conexiones, limpia recursos
    """
    # Startup
    try:
        await startup()
        yield
    finally:
        # Shutdown
        await shutdown()


# ==========================================
# Crear Aplicación
# ==========================================

def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI.
    
    Returns:
        FastAPI: Aplicación configurada
    """
    settings = get_settings()
    
    app = FastAPI(
        title="Paideia API",
        description="""
        Sistema RAG educativo con grafos de conocimiento.
        
        ## Funcionalidades
        
        - **Query**: Consultas con contexto personalizado
        - **Ingest**: Ingesta de documentos, URLs y media
        - **Feedback**: Sistema de retroalimentación
        
        ## Arquitectura
        
        - GraphRAG con SurrealDB
        - Agentes de retrieval, razonamiento y reflexión
        - Perfiles de estudiantes adaptativos
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # En producción, especificar orígenes
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Montar routers
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(feedback_router, prefix="/api/v1")
    app.include_router(databases_router, prefix="/api/v1")
    app.include_router(models_router, prefix="/api/v1")
    
    # Registrar handlers de excepciones
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handler global de excepciones."""
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(exc),
                "path": str(request.url),
            },
        )
    
    return app


# ==========================================
# Endpoints de Health
# ==========================================

app = create_app()


@app.get(
    "/",
    tags=["Root"],
    summary="Root",
    description="Endpoint raíz",
)
async def root() -> Dict[str, str]:
    """
    Endpoint raíz de la API.
    """
    return {
        "name": "Paideia API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Verifica el estado de la aplicación",
)
async def health_check() -> Dict[str, Any]:
    """
    Health check básico.
    """
    return {
        "status": "healthy",
        "service": "paideia-api",
    }


@app.get(
    "/health/detailed",
    tags=["Health"],
    summary="Detailed Health Check",
    description="Health check detallado con estado de dependencias",
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    Health check detallado que verifica todas las dependencias.
    """
    db_health = await check_db_health()
    models_health = await check_models_health()
    
    overall_status = "healthy"
    if db_health.get("status") != "healthy":
        overall_status = "degraded"
    if models_health.get("llm", {}).get("status") != "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "components": {
            "database": db_health,
            "models": models_health,
        },
        "config": {
            "graph_rag_enabled": get_rag_config().get("enable_graph_rag", True),
            "cache_enabled": get_rag_config().get("enable_cache", True),
        },
    }


@app.get(
    "/config",
    tags=["Config"],
    summary="Configuration",
    description="Obtiene configuración actual (sin secretos)",
)
async def get_config() -> Dict[str, Any]:
    """
    Obtiene configuración actual de la aplicación.
    
    No incluye secretos como API keys.
    """
    settings = get_settings()
    rag_config = get_rag_config()
    
    return {
        "llm": {
            "provider": settings.llm_provider.value,
            "model": settings.openai_model if settings.llm_provider.value == "openai" else settings.google_model,
        },
        "embedding": {
            "provider": settings.embedding_provider.value,
        },
        "rag": {
            "chunk_size": rag_config["chunk_size"],
            "chunk_overlap": rag_config["chunk_overlap"],
            "max_context_tokens": rag_config["max_context_tokens"],
            "similarity_threshold": rag_config["similarity_threshold"],
            "graph_rag_enabled": rag_config["enable_graph_rag"],
            "cache_enabled": rag_config["enable_cache"],
        },
    }


# ==========================================
# Entry Point
# ==========================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
