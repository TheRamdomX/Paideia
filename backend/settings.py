"""
settings.py
Configuración centralizada.
(modelos, chunk size, thresholds, flags RAG/Graph)
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Proveedores de LLM soportados."""
    OPENAI = "openai"
    GOOGLE = "google"


class EmbeddingProvider(str, Enum):
    """Proveedores de embeddings soportados."""
    OPENAI = "openai"
    GOOGLE = "google"


class Settings(BaseSettings):
    """Configuración principal de la aplicación."""
    
    # ==========================================
    # LLM Provider Configuration
    # ==========================================
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Proveedor de LLM a utilizar"
    )
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, description="API Key de OpenAI")
    openai_model: str = Field(default="gpt-4o-mini", description="Modelo de OpenAI")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modelo de embeddings de OpenAI"
    )
    
    # Google (Gemini/Gemma)
    google_api_key: Optional[str] = Field(default=None, description="API Key de Google AI")
    google_model: str = Field(default="gemma-3-27b-it", description="Modelo de Google")
    google_embedding_model: str = Field(
        default="gemini-embedding-001",
        description="Modelo de embeddings de Google"
    )
    
    # ==========================================
    # Embedding Provider Configuration
    # ==========================================
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.OPENAI,
        description="Proveedor de embeddings a utilizar"
    )
    
    # ==========================================
    # SurrealDB Configuration
    # ==========================================
    surreal_url: str = Field(
        default="ws://localhost:8000/rpc",
        description="URL de conexión a SurrealDB"
    )
    surreal_namespace: str = Field(default="paideia", description="Namespace de SurrealDB")
    surreal_database: str = Field(default="education", description="Base de datos de SurrealDB")
    surreal_user: str = Field(default="root", description="Usuario de SurrealDB")
    surreal_pass: str = Field(default="root", description="Contraseña de SurrealDB")
    
    # ==========================================
    # RAG Configuration
    # ==========================================
    chunk_size: int = Field(default=512, description="Tamaño de chunk en tokens")
    chunk_overlap: int = Field(default=50, description="Solapamiento entre chunks")
    max_context_tokens: int = Field(default=4096, description="Máximo de tokens de contexto")
    similarity_threshold: float = Field(
        default=0.7,
        description="Umbral mínimo de similitud para retrieval"
    )
    
    # ==========================================
    # Model Parameters
    # ==========================================
    llm_temperature: float = Field(default=0.7, description="Temperatura del LLM")
    llm_max_tokens: int = Field(default=2048, description="Máximo de tokens de salida")
    
    # ==========================================
    # Feature Flags
    # ==========================================
    enable_graph_rag: bool = Field(default=True, description="Habilitar GraphRAG")
    enable_cache: bool = Field(default=True, description="Habilitar caché de respuestas")
    enable_feedback: bool = Field(default=True, description="Habilitar sistema de feedback")
    
    # ==========================================
    # User API Keys Configuration
    # ==========================================
    allow_user_api_keys: bool = Field(
        default=True, 
        description="Permitir que usuarios envíen sus propias API keys"
    )
    require_api_key: bool = Field(
        default=False,
        description="Requerir API key (del backend o usuario) para funcionar"
    )
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


def load_settings() -> None:
    """
    Carga las variables de entorno desde el archivo .env.
    Debe llamarse al inicio de la aplicación.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Intenta cargar desde el directorio actual
        load_dotenv()


@lru_cache()
def get_settings() -> Settings:
    """
    Obtiene la configuración de la aplicación (singleton cacheado).
    
    Returns:
        Settings: Instancia de configuración
    """
    load_settings()
    return Settings()


def get_model_config() -> dict:
    """
    Devuelve la configuración del modelo LLM activo.
    
    Returns:
        dict: Configuración del modelo con keys:
            - provider: Proveedor activo
            - model: Nombre del modelo
            - api_key: API key del proveedor
            - temperature: Temperatura
            - max_tokens: Máximo de tokens
    """
    settings = get_settings()
    
    if settings.llm_provider == LLMProvider.OPENAI:
        return {
            "provider": LLMProvider.OPENAI,
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
    elif settings.llm_provider == LLMProvider.GOOGLE:
        return {
            "provider": LLMProvider.GOOGLE,
            "model": settings.google_model,
            "api_key": settings.google_api_key,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }
    else:
        raise ValueError(f"Proveedor LLM no soportado: {settings.llm_provider}")


def get_model_config_with_user_keys(
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
) -> dict:
    """
    Devuelve la configuración del modelo LLM.
    
    MODO EXCLUSIVO:
    - Si el usuario envía API key → se usa SOLO la del usuario (ignora servidor)
    - Si NO envía API key → se usa SOLO la del servidor
    
    Args:
        user_openai_key: API key de OpenAI del usuario
        user_google_key: API key de Google del usuario  
        preferred_provider: Proveedor preferido ('openai' o 'google')
        user_model: Modelo específico a usar (ej: 'gpt-4', 'gemini-pro')
        
    Returns:
        dict: Configuración del modelo
        
    Raises:
        ValueError: Si no hay API key disponible
    """
    settings = get_settings()
    
    # Determinar si el usuario está enviando sus propias keys
    user_has_key = bool(user_openai_key or user_google_key)
    
    # MODO EXCLUSIVO: Usuario O Servidor, no ambos
    if user_has_key:
        # === MODO USUARIO: Solo usar keys del usuario ===
        if not settings.allow_user_api_keys:
            raise ValueError("El servidor no permite API keys de usuario.")
        
        # Determinar proveedor basado en la key enviada o preferencia
        if preferred_provider and preferred_provider.lower() == "openai" and user_openai_key:
            provider = LLMProvider.OPENAI
            api_key = user_openai_key
            default_model = "gpt-4.1-mini"
        elif preferred_provider and preferred_provider.lower() == "google" and user_google_key:
            provider = LLMProvider.GOOGLE
            api_key = user_google_key
            default_model = "gemma-3-27b-it"
        elif user_openai_key:
            provider = LLMProvider.OPENAI
            api_key = user_openai_key
            default_model = "gpt-4.1-mini"
        elif user_google_key:
            provider = LLMProvider.GOOGLE
            api_key = user_google_key
            default_model = "gemma-3-27b-it"
        else:
            raise ValueError("Debe proporcionar una API key válida.")
        
        # Usar modelo del usuario o default
        model = user_model if user_model else default_model
        
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "has_api_key": True,
            "is_user_key": True,
            "source": "user",
        }
    
    else:
        # === MODO SERVIDOR: Solo usar keys del servidor ===
        server_openai = settings.openai_api_key
        server_google = settings.google_api_key
        
        # Verificar que las keys del servidor no sean placeholders
        if server_openai and server_openai.startswith("sk-your"):
            server_openai = None
        if server_google and server_google in ["your-google-api-key", ""]:
            server_google = None
        
        # Determinar proveedor según configuración del servidor
        if settings.llm_provider == LLMProvider.OPENAI and server_openai:
            provider = LLMProvider.OPENAI
            api_key = server_openai
            model = settings.openai_model
        elif settings.llm_provider == LLMProvider.GOOGLE and server_google:
            provider = LLMProvider.GOOGLE
            api_key = server_google
            model = settings.google_model
        # Fallback a cualquier key disponible
        elif server_openai:
            provider = LLMProvider.OPENAI
            api_key = server_openai
            model = settings.openai_model
        elif server_google:
            provider = LLMProvider.GOOGLE
            api_key = server_google
            model = settings.google_model
        else:
            # No hay keys configuradas en el servidor
            if settings.require_api_key:
                raise ValueError(
                    "No hay API key configurada en el servidor. "
                    "Proporcione su propia API key en la configuración."
                )
            return {
                "provider": None,
                "model": None,
                "api_key": None,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "has_api_key": False,
                "source": "none",
            }
        
        return {
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "has_api_key": True,
            "is_user_key": False,
            "source": "server",
        }


def get_embedding_config() -> dict:
    """
    Devuelve la configuración del modelo de embeddings activo.
    
    Returns:
        dict: Configuración con keys:
            - provider: Proveedor activo
            - model: Nombre del modelo
            - api_key: API key del proveedor
    """
    settings = get_settings()
    
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return {
            "provider": EmbeddingProvider.OPENAI,
            "model": settings.openai_embedding_model,
            "api_key": settings.openai_api_key,
        }
    elif settings.embedding_provider == EmbeddingProvider.GOOGLE:
        return {
            "provider": EmbeddingProvider.GOOGLE,
            "model": settings.google_embedding_model,
            "api_key": settings.google_api_key,
        }
    else:
        raise ValueError(f"Proveedor de embeddings no soportado: {settings.embedding_provider}")


def get_rag_config() -> dict:
    """
    Devuelve la configuración de RAG.
    
    Returns:
        dict: Configuración con keys:
            - chunk_size: Tamaño de chunk
            - chunk_overlap: Solapamiento
            - max_context_tokens: Máximo contexto
            - similarity_threshold: Umbral de similitud
            - enable_graph_rag: Flag de GraphRAG
            - enable_cache: Flag de caché
    """
    settings = get_settings()
    
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "max_context_tokens": settings.max_context_tokens,
        "similarity_threshold": settings.similarity_threshold,
        "enable_graph_rag": settings.enable_graph_rag,
        "enable_cache": settings.enable_cache,
        "enable_feedback": settings.enable_feedback,
    }


def get_db_config() -> dict:
    """
    Devuelve la configuración de la base de datos.
    
    Returns:
        dict: Configuración de SurrealDB
    """
    settings = get_settings()
    
    return {
        "url": settings.surreal_url,
        "namespace": settings.surreal_namespace,
        "database": settings.surreal_database,
        "user": settings.surreal_user,
        "password": settings.surreal_pass,
    }
