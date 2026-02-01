"""
bm25.py
Consultas full-text en SurrealDB.
Implementa búsqueda BM25 para retrieval léxico.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.db.surreal import execute, get_db
from backend.settings import get_rag_config
from backend.utils.text import extract_keywords, clean_markdown


# ==========================================
# Estructuras de Datos
# ==========================================

@dataclass
class BM25Result:
    """Resultado de búsqueda BM25."""
    
    id: str = ""
    content: str = ""
    score: float = 0.0
    highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


@dataclass
class BM25SearchResponse:
    """Respuesta completa de búsqueda BM25."""
    
    query: str = ""
    results: List[BM25Result] = field(default_factory=list)
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
# Configuración BM25
# ==========================================

# Parámetros BM25 estándar
BM25_K1 = 1.2  # Saturación de frecuencia de término
BM25_B = 0.75  # Normalización por longitud de documento


def get_bm25_config() -> Dict[str, Any]:
    """Obtiene configuración BM25."""
    config = get_rag_config()
    return {
        "k1": BM25_K1,
        "b": BM25_B,
        "min_score": config.get("bm25_min_score", 0.1),
        "max_results": config.get("bm25_max_results", 20),
    }


# ==========================================
# Funciones de Búsqueda
# ==========================================

async def search_text(
    query: str,
    table: str = "chunk",
    fields: Optional[List[str]] = None,
    limit: int = 10,
    min_score: float = 0.0,
    filters: Optional[Dict[str, Any]] = None,
) -> BM25SearchResponse:
    """
    Búsqueda BM25 full-text en SurrealDB.
    
    Args:
        query: Texto de búsqueda
        table: Tabla donde buscar
        fields: Campos a buscar (default: content)
        limit: Número máximo de resultados
        min_score: Score mínimo para incluir
        filters: Filtros adicionales
        
    Returns:
        BM25SearchResponse con resultados ordenados por relevancia
    """
    import time
    start_time = time.time()
    
    if not query or not query.strip():
        return BM25SearchResponse(query=query)
    
    # Campos por defecto
    search_fields = fields or ["content"]
    
    # Preparar query - escapar caracteres especiales
    clean_query = _prepare_search_query(query)
    
    if not clean_query:
        return BM25SearchResponse(query=query)
    
    try:
        db = await get_db()
        
        # Construir query SurrealDB con búsqueda full-text
        # SurrealDB usa @@ para full-text search
        field_conditions = " OR ".join([
            f"search::score({f}) > 0" for f in search_fields
        ])
        
        # Query con scoring
        surql = f"""
            SELECT 
                id,
                content,
                metadata,
                search::score(content) AS score
            FROM {table}
            WHERE content @@ $query
            ORDER BY score DESC
            LIMIT $limit
        """
        
        params = {"query": clean_query, "limit": limit}
        
        result = await db.query(surql, params)
        
        # Procesar resultados
        results = []
        
        if result and len(result) > 0:
            rows = result[0].get("result", []) if isinstance(result[0], dict) else result
            
            for row in rows:
                if isinstance(row, dict):
                    score = float(row.get("score", 0))
                    
                    if score >= min_score:
                        # Generar highlights
                        content = row.get("content", "")
                        highlights = highlight_matches(content, query)
                        
                        bm25_result = BM25Result(
                            id=str(row.get("id", "")),
                            content=content,
                            score=score,
                            highlights=highlights,
                            metadata=row.get("metadata", {}),
                        )
                        results.append(bm25_result)
        
        elapsed = (time.time() - start_time) * 1000
        
        return BM25SearchResponse(
            query=query,
            results=results,
            total_found=len(results),
            search_time_ms=elapsed,
        )
        
    except Exception as e:
        # Fallback a búsqueda simple si full-text no está disponible
        return await _fallback_search(query, table, limit, min_score)


async def _fallback_search(
    query: str,
    table: str,
    limit: int,
    min_score: float,
) -> BM25SearchResponse:
    """
    Búsqueda de fallback usando CONTAINS.
    
    Args:
        query: Texto de búsqueda
        table: Tabla donde buscar
        limit: Número máximo
        min_score: Score mínimo
        
    Returns:
        BM25SearchResponse
    """
    import time
    start_time = time.time()
    
    try:
        db = await get_db()
        
        # Extraer palabras clave
        keywords = extract_keywords(query)
        
        if not keywords:
            keywords = query.lower().split()[:5]
        
        # Buscar documentos que contengan las palabras
        conditions = " OR ".join([
            f"string::lowercase(content) CONTAINS '{kw.lower()}'"
            for kw in keywords[:5]
        ])
        
        surql = f"""
            SELECT id, content, metadata
            FROM {table}
            WHERE {conditions}
            LIMIT $limit
        """
        
        result = await db.query(surql, {"limit": limit})
        
        results = []
        
        if result and len(result) > 0:
            rows = result[0].get("result", []) if isinstance(result[0], dict) else result
            
            for row in rows:
                if isinstance(row, dict):
                    content = row.get("content", "")
                    # Calcular score simple basado en matches
                    score = _calculate_simple_score(content, keywords)
                    
                    if score >= min_score:
                        highlights = highlight_matches(content, query)
                        
                        results.append(BM25Result(
                            id=str(row.get("id", "")),
                            content=content,
                            score=score,
                            highlights=highlights,
                            metadata=row.get("metadata", {}),
                        ))
        
        # Ordenar por score
        results.sort(key=lambda x: x.score, reverse=True)
        
        elapsed = (time.time() - start_time) * 1000
        
        return BM25SearchResponse(
            query=query,
            results=results[:limit],
            total_found=len(results),
            search_time_ms=elapsed,
        )
        
    except Exception:
        return BM25SearchResponse(query=query)


def _prepare_search_query(query: str) -> str:
    """
    Prepara query para búsqueda full-text.
    
    Args:
        query: Query original
        
    Returns:
        Query preparada
    """
    # Limpiar y normalizar
    query = clean_markdown(query)
    query = query.strip()
    
    # Remover caracteres especiales de búsqueda
    query = re.sub(r'[+\-&|!(){}[\]^"~*?:\\]', ' ', query)
    
    # Colapsar espacios múltiples
    query = re.sub(r'\s+', ' ', query)
    
    return query.strip()


def _calculate_simple_score(content: str, keywords: List[str]) -> float:
    """
    Calcula score simple basado en frecuencia de keywords.
    
    Args:
        content: Contenido del documento
        keywords: Palabras clave a buscar
        
    Returns:
        Score entre 0 y 1
    """
    if not content or not keywords:
        return 0.0
    
    content_lower = content.lower()
    total_matches = 0
    
    for keyword in keywords:
        # Contar ocurrencias
        matches = content_lower.count(keyword.lower())
        total_matches += min(matches, 3)  # Cap por keyword
    
    # Normalizar por número de keywords
    max_score = len(keywords) * 3
    score = total_matches / max_score if max_score > 0 else 0
    
    return min(score, 1.0)


# ==========================================
# Funciones de Highlight
# ==========================================

def highlight_matches(
    content: str,
    query: str,
    context_chars: int = 100,
    max_highlights: int = 3,
) -> List[str]:
    """
    Genera snippets con highlights de los matches.
    
    Args:
        content: Contenido completo
        query: Query de búsqueda
        context_chars: Caracteres de contexto alrededor del match
        max_highlights: Número máximo de highlights
        
    Returns:
        Lista de snippets con contexto
    """
    if not content or not query:
        return []
    
    highlights = []
    content_lower = content.lower()
    
    # Obtener términos de búsqueda
    terms = query.lower().split()
    
    for term in terms:
        if len(term) < 3:
            continue
            
        # Buscar ocurrencias
        start = 0
        while len(highlights) < max_highlights:
            pos = content_lower.find(term, start)
            if pos == -1:
                break
            
            # Extraer contexto
            snippet_start = max(0, pos - context_chars)
            snippet_end = min(len(content), pos + len(term) + context_chars)
            
            snippet = content[snippet_start:snippet_end]
            
            # Añadir elipsis si es necesario
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(content):
                snippet = snippet + "..."
            
            if snippet not in highlights:
                highlights.append(snippet)
            
            start = pos + len(term)
    
    return highlights[:max_highlights]


def create_snippet(
    content: str,
    max_length: int = 200,
) -> str:
    """
    Crea un snippet del contenido.
    
    Args:
        content: Contenido completo
        max_length: Longitud máxima
        
    Returns:
        Snippet truncado
    """
    if not content:
        return ""
    
    if len(content) <= max_length:
        return content
    
    # Truncar en límite de palabra
    truncated = content[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.7:
        truncated = truncated[:last_space]
    
    return truncated + "..."


# ==========================================
# Funciones de Indexación
# ==========================================

async def create_fulltext_index(
    table: str = "chunk",
    field: str = "content",
) -> bool:
    """
    Crea índice full-text en SurrealDB.
    
    Args:
        table: Tabla para indexar
        field: Campo a indexar
        
    Returns:
        True si se creó exitosamente
    """
    try:
        db = await get_db()
        
        # Definir índice full-text
        surql = f"""
            DEFINE INDEX idx_{table}_{field}_search 
            ON TABLE {table} 
            COLUMNS {field} 
            SEARCH ANALYZER ascii BM25 
            HIGHLIGHTS
        """
        
        await db.query(surql)
        return True
        
    except Exception:
        return False


async def search_multi_field(
    query: str,
    table: str = "chunk",
    fields: List[str] = None,
    weights: Dict[str, float] = None,
    limit: int = 10,
) -> BM25SearchResponse:
    """
    Búsqueda en múltiples campos con pesos.
    
    Args:
        query: Texto de búsqueda
        table: Tabla donde buscar
        fields: Campos a buscar
        weights: Pesos por campo
        limit: Número máximo de resultados
        
    Returns:
        BM25SearchResponse
    """
    fields = fields or ["content", "metadata.title"]
    weights = weights or {"content": 1.0, "metadata.title": 1.5}
    
    # Buscar en cada campo y combinar
    all_results: Dict[str, BM25Result] = {}
    
    for field in fields:
        response = await search_text(
            query=query,
            table=table,
            fields=[field],
            limit=limit * 2,
        )
        
        weight = weights.get(field, 1.0)
        
        for result in response.results:
            if result.id in all_results:
                # Combinar scores
                all_results[result.id].score += result.score * weight
            else:
                result.score *= weight
                all_results[result.id] = result
    
    # Ordenar y limitar
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x.score,
        reverse=True
    )[:limit]
    
    return BM25SearchResponse(
        query=query,
        results=sorted_results,
        total_found=len(sorted_results),
    )
