"""
cache.py
Cache de consultas frecuentes y embeddings de preguntas.
Sistema de caché multicapa para optimizar rendimiento.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar

from backend.settings import get_rag_config


# ==========================================
# Configuración y Tipos
# ==========================================

T = TypeVar("T")


class CacheStrategy(str, Enum):
    """Estrategia de evicción de caché."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry(Generic[T]):
    """Entrada individual en el caché."""
    
    key: str = ""
    value: T = None  # type: ignore
    
    # Metadata
    created_at: float = 0.0  # Unix timestamp
    last_accessed: float = 0.0
    access_count: int = 0
    ttl_seconds: int = 3600  # 1 hora por defecto
    
    # Metadatos adicionales
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verifica si la entrada ha expirado."""
        if self.ttl_seconds <= 0:
            return False
        return time.time() > self.created_at + self.ttl_seconds
    
    def touch(self) -> None:
        """Actualiza tiempo de acceso y contador."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Estadísticas de caché."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Tasa de aciertos."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


# ==========================================
# Cache Manager
# ==========================================

class CacheManager(Generic[T]):
    """
    Gestor de caché genérico con soporte para múltiples estrategias.
    """
    
    def __init__(
        self,
        name: str,
        max_size: int = 1000,
        default_ttl: int = 3600,
        strategy: CacheStrategy = CacheStrategy.LRU,
    ):
        self.name = name
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[T]:
        """
        Obtiene valor del caché.
        
        Args:
            key: Clave de búsqueda
            
        Returns:
            Valor si existe y no ha expirado, None en caso contrario
        """
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # Verificar expiración
            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return None
            
            # Actualizar acceso
            entry.touch()
            
            # Mover al final para LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            self._stats.hits += 1
            return entry.value
    
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Almacena valor en caché.
        
        Args:
            key: Clave
            value: Valor a almacenar
            ttl: Time-to-live en segundos
            tags: Tags para invalidación por grupo
            metadata: Metadata adicional
        """
        async with self._lock:
            # Evictar si es necesario
            while len(self._cache) >= self.max_size:
                await self._evict_one()
            
            now = time.time()
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl_seconds=ttl if ttl is not None else self.default_ttl,
                tags=tags or [],
                metadata=metadata or {},
            )
            
            self._cache[key] = entry
            self._stats.size = len(self._cache)
    
    async def _evict_one(self) -> None:
        """Evicta una entrada según la estrategia."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Eliminar el más antiguo (primero)
            self._cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # Eliminar el menos usado
            min_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].access_count
            )
            del self._cache[min_key]
        else:  # TTL - eliminar el más antiguo
            self._cache.popitem(last=False)
        
        self._stats.evictions += 1
        self._stats.size = len(self._cache)
    
    async def delete(self, key: str) -> bool:
        """
        Elimina una entrada.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si existía
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalida todas las entradas con un tag.
        
        Args:
            tag: Tag para invalidar
            
        Returns:
            Número de entradas invalidadas
        """
        async with self._lock:
            keys_to_delete = [
                key for key, entry in self._cache.items()
                if tag in entry.tags
            ]
            
            for key in keys_to_delete:
                del self._cache[key]
            
            self._stats.size = len(self._cache)
            self._stats.evictions += len(keys_to_delete)
            
            return len(keys_to_delete)
    
    async def clear(self) -> None:
        """Limpia todo el caché."""
        async with self._lock:
            self._cache.clear()
            self._stats.size = 0
    
    def get_stats(self) -> CacheStats:
        """Obtiene estadísticas."""
        self._stats.size = len(self._cache)
        return self._stats


# ==========================================
# Caches Multi-Database
# ==========================================

# Caches por base de datos: {database_name: CacheManager}
_answer_caches: Dict[str, CacheManager[Dict[str, Any]]] = {}
_embedding_caches: Dict[str, CacheManager[List[float]]] = {}
_chunk_caches: Dict[str, CacheManager[Dict[str, Any]]] = {}

# Lock para creación de caches
_cache_lock = asyncio.Lock()

# Base de datos por defecto (se actualiza desde settings)
_default_database: str = "paideia"


def set_default_database(database: str) -> None:
    """Establece la base de datos por defecto para cache."""
    global _default_database
    _default_database = database


def get_default_database() -> str:
    """Obtiene la base de datos por defecto."""
    return _default_database


def _get_answer_cache(database: Optional[str] = None) -> CacheManager[Dict[str, Any]]:
    """
    Obtiene cache de respuestas para una base de datos específica.
    
    Args:
        database: Nombre de la base de datos (usa default si no se especifica)
        
    Returns:
        CacheManager para esa base de datos
    """
    db_name = database or _default_database
    
    if db_name not in _answer_caches:
        config = get_rag_config()
        _answer_caches[db_name] = CacheManager(
            name=f"answers:{db_name}",
            max_size=config.cache_max_size if hasattr(config, 'cache_max_size') else 500,
            default_ttl=config.cache_ttl if hasattr(config, 'cache_ttl') else 3600,
            strategy=CacheStrategy.LRU,
        )
    return _answer_caches[db_name]


def _get_embedding_cache(database: Optional[str] = None) -> CacheManager[List[float]]:
    """
    Obtiene cache de embeddings para una base de datos específica.
    """
    db_name = database or _default_database
    
    if db_name not in _embedding_caches:
        _embedding_caches[db_name] = CacheManager(
            name=f"embeddings:{db_name}",
            max_size=2000,
            default_ttl=86400,  # 24 horas
            strategy=CacheStrategy.LRU,
        )
    return _embedding_caches[db_name]


def _get_chunk_cache(database: Optional[str] = None) -> CacheManager[Dict[str, Any]]:
    """
    Obtiene cache de chunks para una base de datos específica.
    """
    db_name = database or _default_database
    
    if db_name not in _chunk_caches:
        _chunk_caches[db_name] = CacheManager(
            name=f"chunks:{db_name}",
            max_size=1000,
            default_ttl=7200,  # 2 horas
            strategy=CacheStrategy.LRU,
        )
    return _chunk_caches[db_name]


# ==========================================
# API Principal
# ==========================================

def _generate_cache_key(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    prefix: str = "",
    database: Optional[str] = None,
) -> str:
    """
    Genera clave de caché consistente.
    
    Args:
        query: Consulta o texto principal
        context: Contexto adicional
        prefix: Prefijo para la clave
        database: Base de datos (se incluye en la clave)
        
    Returns:
        Clave de caché hasheada
    """
    # Normalizar query
    normalized = query.strip().lower()
    
    # Incluir contexto relevante
    cache_data = {
        "query": normalized,
        "database": database or _default_database,
    }
    
    if context:
        # Solo incluir campos relevantes para la clave
        relevant_fields = ["student_level", "concepts", "session_id"]
        for field in relevant_fields:
            if field in context:
                cache_data[field] = context[field]
    
    # Generar hash
    cache_string = json.dumps(cache_data, sort_keys=True)
    hash_value = hashlib.sha256(cache_string.encode()).hexdigest()[:16]
    
    if prefix:
        return f"{prefix}:{hash_value}"
    return hash_value


async def get_cached_answer(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Busca respuesta en caché para una consulta.
    
    Args:
        query: Pregunta del usuario
        context: Contexto adicional (nivel, sesión, etc.)
        database: Base de datos específica
        
    Returns:
        Respuesta cacheada o None
    """
    cache = _get_answer_cache(database)
    key = _generate_cache_key(query, context, prefix="answer", database=database)
    
    result = await cache.get(key)
    
    if result:
        # Marcar como resultado de caché
        result["from_cache"] = True
        result["cache_hit_time"] = datetime.now(timezone.utc).isoformat()
    
    return result


async def store_cache(
    query: str,
    answer: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
    tags: Optional[List[str]] = None,
    database: Optional[str] = None,
) -> str:
    """
    Almacena respuesta en caché.
    
    Args:
        query: Pregunta original
        answer: Respuesta a cachear
        context: Contexto de la consulta
        ttl: Tiempo de vida en segundos
        tags: Tags para invalidación
        database: Base de datos específica
        
    Returns:
        Clave de caché
    """
    cache = _get_answer_cache(database)
    key = _generate_cache_key(query, context, prefix="answer", database=database)
    
    # Preparar datos para almacenar
    cache_data = {
        "query": query,
        "answer": answer.get("answer", ""),
        "chunks_used": answer.get("chunks_used", []),
        "concepts": answer.get("concepts", []),
        "confidence": answer.get("confidence", 0.0),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Extraer tags automáticos de conceptos
    auto_tags = [f"concept:{c}" for c in answer.get("concepts", [])]
    all_tags = (tags or []) + auto_tags
    
    await cache.set(
        key=key,
        value=cache_data,
        ttl=ttl,
        tags=all_tags,
        metadata={"query_length": len(query)},
    )
    
    return key


async def invalidate_cache(
    keys: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    concepts: Optional[List[str]] = None,
    database: Optional[str] = None,
) -> int:
    """
    Invalida entradas de caché.
    
    Args:
        keys: Claves específicas a invalidar
        tags: Tags para invalidación grupal
        concepts: Conceptos cuyas respuestas invalidar
        database: Base de datos específica
        
    Returns:
        Número de entradas invalidadas
    """
    cache = _get_answer_cache(database)
    invalidated = 0
    
    # Invalidar por claves
    if keys:
        for key in keys:
            if await cache.delete(key):
                invalidated += 1
    
    # Invalidar por tags
    if tags:
        for tag in tags:
            invalidated += await cache.invalidate_by_tag(tag)
    
    # Invalidar por conceptos
    if concepts:
        for concept in concepts:
            invalidated += await cache.invalidate_by_tag(f"concept:{concept}")
    
    return invalidated


# ==========================================
# Cache de Embeddings
# ==========================================

async def get_cached_embedding(
    text: str,
    database: Optional[str] = None,
) -> Optional[List[float]]:
    """
    Obtiene embedding cacheado.
    
    Args:
        text: Texto del embedding
        database: Base de datos específica
        
    Returns:
        Embedding o None
    """
    cache = _get_embedding_cache(database)
    key = _generate_cache_key(text, prefix="emb", database=database)
    return await cache.get(key)


async def store_embedding(
    text: str,
    embedding: List[float],
    ttl: Optional[int] = None,
    database: Optional[str] = None,
) -> str:
    """
    Almacena embedding en caché.
    
    Args:
        text: Texto original
        embedding: Vector embedding
        ttl: TTL opcional
        database: Base de datos específica
        
    Returns:
        Clave de caché
    """
    cache = _get_embedding_cache(database)
    key = _generate_cache_key(text, prefix="emb", database=database)
    
    await cache.set(key=key, value=embedding, ttl=ttl or 86400)
    
    return key


# ==========================================
# Cache de Chunks
# ==========================================

async def get_cached_chunk(
    chunk_id: str,
    database: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Obtiene chunk cacheado.
    
    Args:
        chunk_id: ID del chunk
        database: Base de datos específica
        
    Returns:
        Datos del chunk o None
    """
    cache = _get_chunk_cache(database)
    return await cache.get(f"chunk:{chunk_id}")


async def store_chunk(
    chunk_id: str,
    chunk_data: Dict[str, Any],
    ttl: Optional[int] = None,
    database: Optional[str] = None,
) -> None:
    """
    Almacena chunk en caché.
    
    Args:
        chunk_id: ID del chunk
        chunk_data: Datos del chunk
        ttl: TTL opcional
        database: Base de datos específica
    """
    cache = _get_chunk_cache(database)
    
    tags = [f"doc:{chunk_data.get('document_id', '')}"]
    concepts = chunk_data.get("concepts", [])
    tags.extend([f"concept:{c}" for c in concepts])
    
    await cache.set(
        key=f"chunk:{chunk_id}",
        value=chunk_data,
        ttl=ttl or 7200,
        tags=tags,
    )


async def invalidate_chunks_by_document(
    document_id: str,
    database: Optional[str] = None,
) -> int:
    """
    Invalida chunks de un documento.
    
    Args:
        document_id: ID del documento
        database: Base de datos específica
        
    Returns:
        Chunks invalidados
    """
    cache = _get_chunk_cache(database)
    return await cache.invalidate_by_tag(f"doc:{document_id}")


# ==========================================
# Utilidades
# ==========================================

async def get_cache_stats(database: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtiene estadísticas de todos los caches.
    
    Args:
        database: Base de datos específica (None para estadísticas globales)
        
    Returns:
        Diccionario con estadísticas
    """
    if database:
        return {
            "database": database,
            "answer_cache": _get_answer_cache(database).get_stats().__dict__,
            "embedding_cache": _get_embedding_cache(database).get_stats().__dict__,
            "chunk_cache": _get_chunk_cache(database).get_stats().__dict__,
        }
    
    # Estadísticas de todas las bases de datos
    all_stats = {}
    for db_name in set(list(_answer_caches.keys()) + list(_embedding_caches.keys()) + list(_chunk_caches.keys())):
        all_stats[db_name] = {
            "answer_cache": _get_answer_cache(db_name).get_stats().__dict__,
            "embedding_cache": _get_embedding_cache(db_name).get_stats().__dict__,
            "chunk_cache": _get_chunk_cache(db_name).get_stats().__dict__,
        }
    return all_stats


async def clear_all_caches(database: Optional[str] = None) -> None:
    """
    Limpia todos los caches.
    
    Args:
        database: Base de datos específica (None para limpiar todos)
    """
    if database:
        await _get_answer_cache(database).clear()
        await _get_embedding_cache(database).clear()
        await _get_chunk_cache(database).clear()
    else:
        for cache in _answer_caches.values():
            await cache.clear()
        for cache in _embedding_caches.values():
            await cache.clear()
        for cache in _chunk_caches.values():
            await cache.clear()


def reset_caches(database: Optional[str] = None) -> None:
    """
    Reinicia instancias de caché (para testing).
    
    Args:
        database: Base de datos específica (None para todos)
    """
    global _answer_caches, _embedding_caches, _chunk_caches
    
    if database:
        _answer_caches.pop(database, None)
        _embedding_caches.pop(database, None)
        _chunk_caches.pop(database, None)
    else:
        _answer_caches = {}
        _embedding_caches = {}
        _chunk_caches = {}


# ==========================================
# Decorador de Caché
# ==========================================

def cached(
    ttl: int = 3600,
    key_prefix: str = "",
    use_database: bool = True,
):
    """
    Decorador para cachear resultados de funciones async.
    
    Args:
        ttl: Tiempo de vida
        key_prefix: Prefijo para la clave
        use_database: Si incluir database en la clave
    """
    def decorator(func):
        async def wrapper(*args, database: Optional[str] = None, **kwargs):
            # Generar clave basada en argumentos
            cache_key = _generate_cache_key(
                f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}",
                prefix=key_prefix or func.__name__,
                database=database if use_database else None,
            )
            
            cache = _get_answer_cache(database)
            
            # Intentar obtener de caché
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Almacenar en caché
            if result is not None:
                await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator
