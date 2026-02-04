"""
embeddings.py
Proveedor de embeddings.
Soporta OpenAI y Google como proveedores configurables.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.settings import (
    EmbeddingProvider,
    get_embedding_config,
    get_settings,
)


# ==========================================
# Interfaz Base para Embeddings
# ==========================================

class BaseEmbedding(ABC):
    """Interfaz abstracta para proveedores de embeddings."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Genera embedding para un texto."""
        pass
    
    @abstractmethod
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para múltiples textos."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensión del vector de embedding."""
        pass


# ==========================================
# Implementación OpenAI
# ==========================================

class OpenAIEmbedding(BaseEmbedding):
    """Implementación de embeddings usando OpenAI."""
    
    # Dimensiones conocidas de modelos OpenAI
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization del cliente OpenAI."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client
    
    @property
    def dimension(self) -> int:
        """Dimensión del embedding."""
        return self.DIMENSIONS.get(self.model, 1536)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Genera embedding para un texto.
        
        Args:
            text: Texto a embeder
            
        Returns:
            Vector de embedding
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        return response.data[0].embedding
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de vectores de embedding
        """
        if not texts:
            return []
        
        # Filtrar textos vacíos y guardar índices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        # Hacer request en batches (OpenAI soporta hasta 2048 inputs)
        batch_size = 100
        all_embeddings = {}
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            
            response = await self.client.embeddings.create(
                model=self.model,
                input=batch_texts,
            )
            
            for j, embedding_data in enumerate(response.data):
                all_embeddings[batch_indices[j]] = embedding_data.embedding
        
        # Reconstruir lista completa
        result = []
        for i in range(len(texts)):
            if i in all_embeddings:
                result.append(all_embeddings[i])
            else:
                result.append([0.0] * self.dimension)
        
        return result


# ==========================================
# Implementación Google
# ==========================================

class GoogleEmbedding(BaseEmbedding):
    """Implementación de embeddings usando Google."""
    
    # Dimensión compatible con OpenAI text-embedding-3-small
    EMBEDDING_DIMENSION = 3072
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = None
    
    @property
    def client(self):
        """Lazy initialization del cliente Google."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    @property
    def dimension(self) -> int:
        """Dimensión del embedding."""
        return self.EMBEDDING_DIMENSION
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Genera embedding para un texto usando Google.
        
        Args:
            text: Texto a embeder
            
        Returns:
            Vector de embedding
        """
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        from google.genai import types
        
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIMENSION),
            )
        )
        
        # La nueva API devuelve embeddings como lista de ContentEmbedding
        if result.embeddings and len(result.embeddings) > 0:
            return list(result.embeddings[0].values)
        return [0.0] * self.dimension
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de vectores de embedding
        """
        if not texts:
            return []
        
        from google.genai import types
        
        results = []
        
        for text in texts:
            if not text or not text.strip():
                results.append([0.0] * self.dimension)
                continue
            
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                lambda t=text: self.client.models.embed_content(
                    model=self.model,
                    contents=t,
                    config=types.EmbedContentConfig(output_dimensionality=self.EMBEDDING_DIMENSION),
                )
            )
            
            if result.embeddings and len(result.embeddings) > 0:
                results.append(list(result.embeddings[0].values))
            else:
                results.append([0.0] * self.dimension)
        
        return results


# ==========================================
# Factory y Funciones de Conveniencia
# ==========================================

_embedding_instance: Optional[BaseEmbedding] = None


def select_embedding_model(
    provider: Optional[EmbeddingProvider] = None,
    **kwargs
) -> BaseEmbedding:
    """
    Selecciona y configura el modelo de embeddings.
    
    Args:
        provider: Proveedor de embeddings
        **kwargs: Argumentos adicionales
        
    Returns:
        Instancia de modelo de embeddings
    """
    config = get_embedding_config()
    
    if provider is None:
        provider = config["provider"]
    
    model = kwargs.get("model", config["model"])
    api_key = kwargs.get("api_key", config["api_key"])
    
    if not api_key:
        raise ValueError(f"API key no configurada para proveedor: {provider}")
    
    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbedding(model=model, api_key=api_key)
    elif provider == EmbeddingProvider.GOOGLE:
        return GoogleEmbedding(model=model, api_key=api_key)
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


def get_embedding_model() -> BaseEmbedding:
    """
    Obtiene la instancia singleton del modelo de embeddings.
    
    Returns:
        Instancia de modelo de embeddings
    """
    global _embedding_instance
    
    if _embedding_instance is None:
        _embedding_instance = select_embedding_model()
    
    return _embedding_instance


def get_embedding_model_with_key(
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> BaseEmbedding:
    """
    Obtiene modelo de embeddings con API keys del cliente.
    
    Si se proporciona una API key del cliente, se usa esa.
    Si no, se usa la configuración del servidor.
    
    Args:
        user_openai_key: API key de OpenAI del cliente
        user_google_key: API key de Google del cliente
        
    Returns:
        Instancia de modelo de embeddings
    """
    config = get_embedding_config()
    provider = config["provider"]
    model = config["model"]
    
    # Determinar qué API key usar
    if provider == EmbeddingProvider.OPENAI:
        api_key = user_openai_key or config["api_key"]
        if not api_key or api_key.startswith("sk-your"):
            # Si no hay key de OpenAI válida, intentar con Google
            if user_google_key:
                return GoogleEmbedding(
                    model=get_settings().google_embedding_model,
                    api_key=user_google_key,
                )
        if not api_key:
            raise ValueError("No hay API key de OpenAI configurada")
        return OpenAIEmbedding(model=model, api_key=api_key)
    
    elif provider == EmbeddingProvider.GOOGLE:
        api_key = user_google_key or config["api_key"]
        if not api_key or api_key == "your-google-api-key":
            # Si no hay key de Google válida, intentar con OpenAI
            if user_openai_key:
                return OpenAIEmbedding(
                    model=get_settings().openai_embedding_model,
                    api_key=user_openai_key,
                )
        if not api_key:
            raise ValueError("No hay API key de Google configurada")
        return GoogleEmbedding(model=model, api_key=api_key)
    
    else:
        raise ValueError(f"Proveedor no soportado: {provider}")


async def embed_text(
    text: str,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> List[float]:
    """
    Genera embedding para un texto.
    
    Args:
        text: Texto a embeder
        user_openai_key: API key de OpenAI del cliente (opcional)
        user_google_key: API key de Google del cliente (opcional)
        
    Returns:
        Vector de embedding
    """
    if user_openai_key or user_google_key:
        model = get_embedding_model_with_key(user_openai_key, user_google_key)
    else:
        model = get_embedding_model()
    return await model.embed_text(text)


async def batch_embed(
    texts: List[str],
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Genera embeddings para múltiples textos.
    
    Args:
        texts: Lista de textos
        user_openai_key: API key de OpenAI del cliente (opcional)
        user_google_key: API key de Google del cliente (opcional)
        
    Returns:
        Lista de vectores de embedding
    """
    if user_openai_key or user_google_key:
        model = get_embedding_model_with_key(user_openai_key, user_google_key)
    else:
        model = get_embedding_model()
    print(f"[EMBEDDING] Using model: {type(model).__name__}, dimension: {model.dimension}")
    return await model.batch_embed(texts)


def get_embedding_dimension() -> int:
    """
    Obtiene la dimensión de los embeddings.
    
    Returns:
        Dimensión del vector
    """
    model = get_embedding_model()
    return model.dimension


def reset_embedding_model() -> None:
    """Resetea la instancia del modelo (útil para testing)."""
    global _embedding_instance
    _embedding_instance = None
