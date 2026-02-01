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
    
    # Dimensión de text-embedding-004
    EMBEDDING_DIMENSION = 768
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._configured = False
    
    def _configure(self):
        """Configura la API de Google."""
        if not self._configured:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._configured = True
    
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
        
        self._configure()
        import google.generativeai as genai
        
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
        )
        
        return result['embedding']
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos.
        Google no tiene batch nativo, se hace secuencial.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de vectores de embedding
        """
        if not texts:
            return []
        
        self._configure()
        import google.generativeai as genai
        
        results = []
        
        for text in texts:
            if not text or not text.strip():
                results.append([0.0] * self.dimension)
                continue
            
            loop = asyncio.get_event_loop()
            
            result = await loop.run_in_executor(
                None,
                lambda t=text: genai.embed_content(
                    model=self.model,
                    content=t,
                    task_type="retrieval_document",
                )
            )
            
            results.append(result['embedding'])
        
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


async def embed_text(text: str) -> List[float]:
    """
    Genera embedding para un texto.
    
    Args:
        text: Texto a embeder
        
    Returns:
        Vector de embedding
    """
    model = get_embedding_model()
    return await model.embed_text(text)


async def batch_embed(texts: List[str]) -> List[List[float]]:
    """
    Genera embeddings para múltiples textos.
    
    Args:
        texts: Lista de textos
        
    Returns:
        Lista de vectores de embedding
    """
    model = get_embedding_model()
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
