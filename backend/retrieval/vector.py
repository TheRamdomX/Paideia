"""
vector.py
Vector search por similitud.
Implementa búsqueda semántica usando embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.db.surreal import execute, get_db
from backend.models.embeddings import embed_text, get_embedding_dimension
from backend.settings import get_rag_config


# ==========================================
# Estructuras de Datos
# ==========================================

@dataclass
class VectorResult:
    """Resultado de búsqueda vectorial."""
    
    id: str = ""
    content: str = ""
    score: float = 0.0  # Similaridad coseno (0-1)
    distance: float = 0.0  # Distancia euclidiana
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "distance": self.distance,
            "metadata": self.metadata,
        }


@dataclass
class VectorSearchResponse:
    """Respuesta completa de búsqueda vectorial."""
    
    query: str = ""
    query_embedding: List[float] = field(default_factory=list)
    results: List[VectorResult] = field(default_factory=list)
    total_found: int = 0
    search_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_found": self.total_found,
            "search_time_ms": self.search_time_ms,
        }


# ==========================================
# Funciones de Similaridad
# ==========================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula similaridad coseno entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Similaridad entre 0 y 1
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula distancia euclidiana entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Distancia (menor = más similar)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return float('inf')
    
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def dot_product(vec1: List[float], vec2: List[float]) -> float:
    """
    Calcula producto punto entre dos vectores.
    
    Args:
        vec1: Primer vector
        vec2: Segundo vector
        
    Returns:
        Producto punto
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    return sum(a * b for a, b in zip(vec1, vec2))


# ==========================================
# Funciones de Búsqueda
# ==========================================

async def search_vectors(
    query: str,
    table: str = "chunk",
    embedding_field: str = "embedding",
    limit: int = 10,
    min_score: float = 0.0,
    filters: Optional[Dict[str, Any]] = None,
) -> VectorSearchResponse:
    """
    Búsqueda vectorial por similaridad coseno.
    
    Args:
        query: Texto de búsqueda
        table: Tabla donde buscar
        embedding_field: Campo con embedding
        limit: Número máximo de resultados
        min_score: Score mínimo (similaridad)
        filters: Filtros adicionales
        
    Returns:
        VectorSearchResponse con resultados ordenados por similaridad
    """
    import time
    start_time = time.time()
    
    if not query or not query.strip():
        return VectorSearchResponse(query=query)
    
    try:
        # Generar embedding de la query
        query_embedding = await embed_text(query)
        
        if not query_embedding:
            return VectorSearchResponse(query=query)
        
        # Buscar vectores similares
        results = await search_by_embedding(
            embedding=query_embedding,
            table=table,
            embedding_field=embedding_field,
            limit=limit,
            min_score=min_score,
            filters=filters,
        )
        
        elapsed = (time.time() - start_time) * 1000
        
        return VectorSearchResponse(
            query=query,
            query_embedding=query_embedding,
            results=results,
            total_found=len(results),
            search_time_ms=elapsed,
        )
        
    except Exception as e:
        return VectorSearchResponse(query=query)


async def search_by_embedding(
    embedding: List[float],
    table: str = "chunk",
    embedding_field: str = "embedding",
    limit: int = 10,
    min_score: float = 0.0,
    filters: Optional[Dict[str, Any]] = None,
) -> List[VectorResult]:
    """
    Búsqueda por embedding directo.
    
    Args:
        embedding: Vector de búsqueda
        table: Tabla donde buscar
        embedding_field: Campo con embedding
        limit: Número máximo
        min_score: Score mínimo
        filters: Filtros adicionales
        
    Returns:
        Lista de VectorResult
    """
    try:
        db = await get_db()
        
        # SurrealDB soporta búsqueda vectorial nativa
        # Usamos vector::similarity::cosine
        surql = f"""
            SELECT 
                id,
                content,
                metadata,
                {embedding_field},
                vector::similarity::cosine({embedding_field}, $embedding) AS score
            FROM {table}
            WHERE {embedding_field} IS NOT NONE
            ORDER BY score DESC
            LIMIT $limit
        """
        
        params = {
            "embedding": embedding,
            "limit": limit * 2,  # Obtener más para filtrar
        }
        
        result = await db.execute(surql, params)
        
        print(f"[VECTOR_SEARCH] Query result type: {type(result)}, len: {len(result) if result else 0}")
        if result:
            print(f"[VECTOR_SEARCH] result[0] type: {type(result[0])}")
        
        results = []
        
        if result and len(result) > 0:
            # El resultado puede ser una lista de dicts directamente o un dict con "result"
            if isinstance(result[0], dict) and "result" in result[0]:
                rows = result[0].get("result", [])
            else:
                rows = result  # Ya es la lista de resultados
            
            print(f"[VECTOR_SEARCH] Rows type: {type(rows)}, count: {len(rows) if hasattr(rows, '__len__') else 'N/A'}")
            if rows and len(rows) > 0:
                print(f"[VECTOR_SEARCH] First row sample: {list(rows[0].keys()) if isinstance(rows[0], dict) else rows[0]}")
            
            for row in rows:
                if isinstance(row, dict):
                    score = float(row.get("score", 0))
                    
                    if score >= min_score:
                        results.append(VectorResult(
                            id=str(row.get("id", "")),
                            content=row.get("content", ""),
                            score=score,
                            distance=1 - score,  # Convertir similaridad a distancia
                            metadata=row.get("metadata", {}),
                        ))
        
        print(f"[VECTOR_SEARCH] Found {len(results)} results with min_score >= {min_score}")
        return results[:limit]
        
    except Exception as e:
        print(f"[VECTOR_SEARCH] Error: {e}")
        # Fallback a cálculo manual
        return await _fallback_vector_search(
            embedding, table, embedding_field, limit, min_score
        )


async def _fallback_vector_search(
    embedding: List[float],
    table: str,
    embedding_field: str,
    limit: int,
    min_score: float,
) -> List[VectorResult]:
    """
    Búsqueda vectorial con cálculo manual de similaridad.
    
    Usado cuando SurrealDB no soporta vector::similarity.
    """
    try:
        db = await get_db()
        
        # Obtener todos los documentos con embeddings
        surql = f"""
            SELECT id, content, metadata, {embedding_field}
            FROM {table}
            WHERE {embedding_field} IS NOT NONE
            LIMIT 1000
        """
        
        result = await db.query(surql)
        
        results = []
        
        if result and len(result) > 0:
            rows = result[0].get("result", []) if isinstance(result[0], dict) else result
            
            for row in rows:
                if isinstance(row, dict):
                    doc_embedding = row.get(embedding_field, [])
                    
                    if doc_embedding:
                        score = cosine_similarity(embedding, doc_embedding)
                        
                        if score >= min_score:
                            results.append(VectorResult(
                                id=str(row.get("id", "")),
                                content=row.get("content", ""),
                                score=score,
                                distance=1 - score,
                                metadata=row.get("metadata", {}),
                            ))
        
        # Ordenar por score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]
        
    except Exception:
        return []


# ==========================================
# Filtrado por Threshold
# ==========================================

def filter_by_threshold(
    results: List[VectorResult],
    min_score: float = 0.0,
    max_distance: float = float('inf'),
) -> List[VectorResult]:
    """
    Filtra resultados por umbral de score o distancia.
    
    Args:
        results: Resultados a filtrar
        min_score: Score mínimo (similaridad)
        max_distance: Distancia máxima
        
    Returns:
        Resultados filtrados
    """
    filtered = []
    
    for result in results:
        if result.score >= min_score and result.distance <= max_distance:
            filtered.append(result)
    
    return filtered


def adaptive_threshold(
    results: List[VectorResult],
    target_count: int = 5,
    min_threshold: float = 0.3,
) -> Tuple[List[VectorResult], float]:
    """
    Calcula threshold adaptativo para obtener N resultados.
    
    Args:
        results: Resultados ordenados por score
        target_count: Número objetivo de resultados
        min_threshold: Threshold mínimo permitido
        
    Returns:
        Tupla (resultados_filtrados, threshold_usado)
    """
    if not results:
        return [], min_threshold
    
    if len(results) <= target_count:
        return results, min_threshold
    
    # Obtener score del resultado en posición target
    threshold = max(results[target_count - 1].score, min_threshold)
    
    # Filtrar con el threshold
    filtered = [r for r in results if r.score >= threshold]
    
    return filtered, threshold


# ==========================================
# Funciones de Índice
# ==========================================

async def create_vector_index(
    table: str = "chunk",
    field: str = "embedding",
    dimension: Optional[int] = None,
) -> bool:
    """
    Crea índice vectorial en SurrealDB.
    
    Args:
        table: Tabla para indexar
        field: Campo de embedding
        dimension: Dimensión del vector
        
    Returns:
        True si se creó exitosamente
    """
    try:
        db = await get_db()
        
        # Obtener dimensión si no se especifica
        if dimension is None:
            dimension = get_embedding_dimension()
        
        # Definir índice HNSW para búsqueda aproximada
        surql = f"""
            DEFINE INDEX idx_{table}_{field}_vec
            ON TABLE {table}
            COLUMNS {field}
            MTREE DIMENSION {dimension}
        """
        
        await db.query(surql)
        return True
        
    except Exception:
        return False


async def get_similar_chunks(
    chunk_id: str,
    limit: int = 5,
    min_score: float = 0.5,
) -> List[VectorResult]:
    """
    Encuentra chunks similares a uno dado.
    
    Args:
        chunk_id: ID del chunk de referencia
        limit: Número máximo de resultados
        min_score: Score mínimo
        
    Returns:
        Lista de chunks similares
    """
    try:
        db = await get_db()
        
        # Obtener embedding del chunk
        surql = "SELECT embedding FROM chunk WHERE id = $id"
        result = await db.query(surql, {"id": chunk_id})
        
        if not result or not result[0].get("result"):
            return []
        
        embedding = result[0]["result"][0].get("embedding")
        
        if not embedding:
            return []
        
        # Buscar similares
        results = await search_by_embedding(
            embedding=embedding,
            limit=limit + 1,  # +1 para excluir el mismo chunk
            min_score=min_score,
        )
        
        # Excluir el chunk original
        return [r for r in results if r.id != chunk_id][:limit]
        
    except Exception:
        return []


# ==========================================
# Utilidades
# ==========================================

def normalize_vector(vec: List[float]) -> List[float]:
    """
    Normaliza un vector a longitud unitaria.
    
    Args:
        vec: Vector a normalizar
        
    Returns:
        Vector normalizado
    """
    if not vec:
        return vec
    
    norm = math.sqrt(sum(x * x for x in vec))
    
    if norm == 0:
        return vec
    
    return [x / norm for x in vec]


def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """
    Calcula el embedding promedio de una lista.
    
    Args:
        embeddings: Lista de embeddings
        
    Returns:
        Embedding promedio
    """
    if not embeddings:
        return []
    
    if len(embeddings) == 1:
        return embeddings[0]
    
    # Verificar dimensiones
    dim = len(embeddings[0])
    if not all(len(e) == dim for e in embeddings):
        return embeddings[0]
    
    # Calcular promedio
    avg = [0.0] * dim
    
    for emb in embeddings:
        for i, val in enumerate(emb):
            avg[i] += val
    
    n = len(embeddings)
    avg = [v / n for v in avg]
    
    # Normalizar
    return normalize_vector(avg)
