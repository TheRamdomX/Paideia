"""
retrieval_graph.py
Orquestación del sistema de retrieval híbrido.
Combina búsqueda vectorial, BM25 y traversal de grafo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from backend.graph.traversal import (
    GraphNode,
    TraversalResult,
    expand_concepts,
    find_concepts_by_query,
)
from backend.models.embeddings import batch_embed
from backend.retrieval.bm25 import BM25SearchResponse, search_text
from backend.retrieval.hybrid_ranker import (
    HybridRankingConfig,
    RankedResult,
    RetrievalSource,
    combine_scores,
    normalize_scores,
    select_top_k,
    deduplicate_results,
    get_default_config,
)
from backend.retrieval.vector import VectorSearchResponse, search_by_embedding
from backend.settings import get_rag_config


# ==========================================
# Estructuras de Datos
# ==========================================

class RetrievalMode(str, Enum):
    """Modos de retrieval."""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class RetrievalQuery:
    """Query de retrieval."""
    
    text: str = ""
    embedding: Optional[List[float]] = None
    concepts: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "has_embedding": self.embedding is not None,
            "concepts": self.concepts,
            "filters": self.filters,
            "mode": self.mode.value,
            "top_k": self.top_k,
        }


@dataclass
class RetrievalResponse:
    """Respuesta de retrieval."""
    
    results: List[RankedResult] = field(default_factory=list)
    vector_results: int = 0
    bm25_results: int = 0
    graph_results: int = 0
    query: Optional[RetrievalQuery] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "vector_results": self.vector_results,
            "bm25_results": self.bm25_results,
            "graph_results": self.graph_results,
            "query": self.query.to_dict() if self.query else None,
            "metadata": self.metadata,
        }


# ==========================================
# Configuración
# ==========================================

def get_retrieval_config() -> HybridRankingConfig:
    """Obtiene configuración de retrieval."""
    config = get_rag_config()
    return HybridRankingConfig(
        vector_weight=config.get("vector_weight", 0.5),
        bm25_weight=config.get("bm25_weight", 0.3),
        graph_weight=config.get("graph_weight", 0.2),
        combination_method=config.get("combination_method", "weighted"),
        normalization_method=config.get("normalization_method", "minmax"),
        rrf_k=config.get("rrf_k", 60),
    )


# ==========================================
# Retrieval por Componente
# ==========================================

async def vector_retrieval(
    query_embedding: List[float],
    table: str = "chunk",
    field: str = "embedding",
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
) -> List[RankedResult]:
    """
    Búsqueda vectorial semántica.
    
    Args:
        query_embedding: Embedding de la query
        table: Tabla a buscar
        field: Campo de embedding
        top_k: Número de resultados
        filters: Filtros adicionales
        
    Returns:
        Lista de resultados rankeados
    """
    response = await search_by_embedding(
        embedding=query_embedding,
        table=table,
        embedding_field=field,
        limit=top_k,
        filters=filters,
    )
    
    results = []
    
    for vr in response:
        result = RankedResult(
            id=vr.id,
            content=vr.content,
            final_score=vr.score,
            vector_score=vr.score,
            sources=[RetrievalSource.VECTOR],
            metadata=vr.metadata,
        )
        results.append(result)
    
    return results


async def bm25_retrieval(
    query_text: str,
    table: str = "chunk",
    fields: List[str] = None,
    top_k: int = 20,
) -> List[RankedResult]:
    """
    Búsqueda BM25 léxica.
    
    Args:
        query_text: Texto de búsqueda
        table: Tabla a buscar
        fields: Campos a buscar
        top_k: Número de resultados
        
    Returns:
        Lista de resultados rankeados
    """
    if fields is None:
        fields = ["content"]
    
    response = await search_text(
        query=query_text,
        table=table,
        fields=fields,
        limit=top_k,
    )
    
    results = []
    
    for br in response.results:
        result = RankedResult(
            id=br.id,
            content=br.content,
            final_score=br.score,
            bm25_score=br.score,
            sources=[RetrievalSource.BM25],
            metadata=br.metadata,
        )
        results.append(result)
    
    return results


async def graph_retrieval(
    concepts: List[str],
    max_depth: int = 2,
    max_nodes: int = 30,
) -> List[RankedResult]:
    """
    Retrieval basado en traversal del grafo de conocimiento.
    
    Args:
        concepts: IDs de conceptos iniciales
        max_depth: Profundidad máxima
        max_nodes: Nodos máximos a explorar
        
    Returns:
        Lista de resultados rankeados
    """
    if not concepts:
        return []
    
    traversal_result = await expand_concepts(
        concept_ids=concepts,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )
    
    results = []
    
    # Convertir chunks encontrados a resultados
    for chunk in traversal_result.chunks_found:
        result = RankedResult(
            id=chunk.get("id", ""),
            content=chunk.get("content", ""),
            final_score=chunk.get("relevance", 0.5),
            graph_score=chunk.get("relevance", 0.5),
            sources=[RetrievalSource.GRAPH],
            metadata=chunk.get("metadata", {}),
        )
        results.append(result)
    
    # También incluir contenido de conceptos visitados
    for node in traversal_result.nodes_visited:
        if node.content:
            result = RankedResult(
                id=node.id,
                content=node.content,
                final_score=node.score,
                graph_score=node.score,
                sources=[RetrievalSource.GRAPH],
                metadata={"type": "concept", "name": node.name, **node.metadata},
            )
            results.append(result)
    
    return results


# ==========================================
# Merge de Resultados
# ==========================================

def merge_results(
    vector_results: List[RankedResult],
    bm25_results: List[RankedResult],
    graph_results: List[RankedResult],
    config: Optional[HybridRankingConfig] = None,
) -> List[RankedResult]:
    """
    Unifica resultados de múltiples fuentes usando hybrid ranking.
    
    Args:
        vector_results: Resultados de búsqueda vectorial
        bm25_results: Resultados de búsqueda BM25
        graph_results: Resultados de traversal de grafo
        config: Configuración de ranking
        
    Returns:
        Lista unificada y rankeada
    """
    if config is None:
        config = get_default_config()
    
    # Preparar estructura para combine_scores
    # Dict con listas de resultados por fuente
    results_by_source = {
        "vector": [
            {"id": r.id, "score": r.final_score, "content": r.content, "metadata": r.metadata}
            for r in vector_results
        ],
        "bm25": [
            {"id": r.id, "score": r.final_score, "content": r.content, "metadata": r.metadata}
            for r in bm25_results
        ],
        "graph": [
            {"id": r.id, "score": r.final_score, "content": r.content, "metadata": r.metadata}
            for r in graph_results
        ],
    }
    
    # Combinar scores usando la función del hybrid_ranker
    combined = combine_scores(results_by_source, config)
    
    # Deduplicar por contenido similar
    unique = deduplicate_results(combined)
    
    return unique


# ==========================================
# Pipeline Principal
# ==========================================

async def retrieve(
    query: Union[str, RetrievalQuery],
    mode: RetrievalMode = RetrievalMode.HYBRID,
    top_k: int = 10,
    concepts: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    config: Optional[HybridRankingConfig] = None,
    config_override: Optional[Dict[str, Any]] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
) -> RetrievalResponse:
    """
    Pipeline principal de retrieval.
    
    Args:
        query: Texto de consulta o RetrievalQuery
        mode: Modo de retrieval
        top_k: Número de resultados finales
        concepts: Conceptos iniciales para graph retrieval
        filters: Filtros adicionales
        config: Configuración de ranking
        config_override: Override de configuración (para compatibilidad)
        openai_api_key: API key de OpenAI del cliente
        google_api_key: API key de Google del cliente
        
    Returns:
        RetrievalResponse con resultados combinados
    """
    if config is None:
        config = get_retrieval_config()
    
    # Soportar query como str o RetrievalQuery
    if isinstance(query, RetrievalQuery):
        retrieval_query = query
        query_text = query.text
        if query.mode:
            mode = query.mode
        if query.top_k:
            top_k = query.top_k
        if query.concepts:
            concepts = query.concepts
        if query.filters:
            filters = query.filters
    else:
        query_text = query
        retrieval_query = RetrievalQuery(
            text=query_text,
            concepts=concepts or [],
            filters=filters or {},
            mode=mode,
            top_k=top_k,
        )
    
    response = RetrievalResponse(query=retrieval_query)
    
    vector_results: List[RankedResult] = []
    bm25_results: List[RankedResult] = []
    graph_results: List[RankedResult] = []
    
    try:
        # Ejecutar retrievals según modo
        if mode in [RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID, RetrievalMode.ADAPTIVE]:
            # Obtener embedding de la query
            print(f"[RETRIEVAL] Generating query embedding with keys: openai={bool(openai_api_key)}, google={bool(google_api_key)}")
            try:
                embeddings = await batch_embed(
                    [query_text],
                    user_openai_key=openai_api_key,
                    user_google_key=google_api_key,
                )
                print(f"[RETRIEVAL] Embedding result: {len(embeddings[0]) if embeddings and embeddings[0] else 0} dimensions")
            except Exception as emb_error:
                import traceback
                print(f"[RETRIEVAL] Embedding ERROR: {emb_error}")
                print(f"[RETRIEVAL] Traceback: {traceback.format_exc()}")
                embeddings = [[]]
            
            if embeddings and embeddings[0]:
                print(f"[RETRIEVAL] Embedding has {len(embeddings[0])} values, first: {embeddings[0][0]:.4f}")
                retrieval_query.embedding = embeddings[0]
                try:
                    vector_results = await vector_retrieval(
                        query_embedding=embeddings[0],
                        top_k=top_k * 2,  # Obtener más para merge
                        filters=filters,
                    )
                    print(f"[RETRIEVAL] Vector results: {len(vector_results)}")
                except Exception as vr_error:
                    print(f"[RETRIEVAL] Vector search ERROR: {vr_error}")
                    vector_results = []
                response.vector_results = len(vector_results)
            else:
                print(f"[RETRIEVAL] No valid embedding generated")
        
        if mode in [RetrievalMode.BM25_ONLY, RetrievalMode.HYBRID, RetrievalMode.ADAPTIVE]:
            bm25_results = await bm25_retrieval(
                query_text=query_text,
                top_k=top_k * 2,
            )
            response.bm25_results = len(bm25_results)
        
        if mode in [RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID, RetrievalMode.ADAPTIVE]:
            # Si no hay conceptos, buscarlos
            if not concepts:
                found_concepts = await find_concepts_by_query(query_text, limit=3)
                concepts = [c.id for c in found_concepts]
            
            if concepts:
                graph_results = await graph_retrieval(
                    concepts=concepts,
                    max_depth=2,
                    max_nodes=20,
                )
                response.graph_results = len(graph_results)
        
        # Modo adaptativo: ajustar pesos según resultados
        if mode == RetrievalMode.ADAPTIVE:
            config = _adapt_weights(
                config,
                len(vector_results),
                len(bm25_results),
                len(graph_results),
            )
        
        # Combinar resultados
        if mode == RetrievalMode.VECTOR_ONLY:
            combined = vector_results
        elif mode == RetrievalMode.BM25_ONLY:
            combined = bm25_results
        elif mode == RetrievalMode.GRAPH_ONLY:
            combined = graph_results
        else:
            combined = merge_results(
                vector_results,
                bm25_results,
                graph_results,
                config,
            )
        
        # Seleccionar top-k final
        response.results = select_top_k(combined, top_k)
        
        response.metadata = {
            "mode": mode.value,
            "config": {
                "vector_weight": config.vector_weight,
                "bm25_weight": config.bm25_weight,
                "graph_weight": config.graph_weight,
            },
        }
        
    except Exception as e:
        response.metadata["error"] = str(e)
    
    return response


def _adapt_weights(
    config: HybridRankingConfig,
    vector_count: int,
    bm25_count: int,
    graph_count: int,
) -> HybridRankingConfig:
    """
    Adapta pesos según cantidad de resultados por fuente.
    Da más peso a fuentes con más resultados.
    """
    total = vector_count + bm25_count + graph_count
    
    if total == 0:
        return config
    
    # Calcular pesos proporcionales
    vector_weight = vector_count / total if vector_count > 0 else 0.0
    bm25_weight = bm25_count / total if bm25_count > 0 else 0.0
    graph_weight = graph_count / total if graph_count > 0 else 0.0
    
    # Combinar con pesos base (50% base, 50% adaptativo)
    return HybridRankingConfig(
        vector_weight=(config.vector_weight + vector_weight) / 2,
        bm25_weight=(config.bm25_weight + bm25_weight) / 2,
        graph_weight=(config.graph_weight + graph_weight) / 2,
        combination_method=config.combination_method,
        normalization_method=config.normalization_method,
        rrf_k=config.rrf_k,
    )


# ==========================================
# Funciones de Conveniencia
# ==========================================

async def quick_search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Búsqueda rápida con configuración por defecto.
    
    Args:
        query: Texto de consulta
        top_k: Número de resultados
        
    Returns:
        Lista de resultados como dicts
    """
    response = await retrieve(
        query=query,
        mode=RetrievalMode.HYBRID,
        top_k=top_k,
    )
    
    return [r.to_dict() for r in response.results]


async def semantic_search(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Búsqueda puramente semántica (solo vectores).
    
    Args:
        query: Texto de consulta
        top_k: Número de resultados
        
    Returns:
        Lista de resultados
    """
    response = await retrieve(
        query=query,
        mode=RetrievalMode.VECTOR_ONLY,
        top_k=top_k,
    )
    
    return [r.to_dict() for r in response.results]


async def keyword_search(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Búsqueda por palabras clave (solo BM25).
    
    Args:
        query: Texto de consulta
        top_k: Número de resultados
        
    Returns:
        Lista de resultados
    """
    response = await retrieve(
        query=query,
        mode=RetrievalMode.BM25_ONLY,
        top_k=top_k,
    )
    
    return [r.to_dict() for r in response.results]


async def concept_search(
    concept_ids: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Búsqueda basada en conceptos (solo grafo).
    
    Args:
        concept_ids: IDs de conceptos
        top_k: Número de resultados
        
    Returns:
        Lista de resultados
    """
    response = await retrieve(
        query="",  # No necesitamos query para graph
        mode=RetrievalMode.GRAPH_ONLY,
        top_k=top_k,
        concepts=concept_ids,
    )
    
    return [r.to_dict() for r in response.results]


# ==========================================
# Context Building
# ==========================================

async def build_context(
    query: str,
    max_tokens: int = 4000,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> str:
    """
    Construye contexto para el LLM a partir de resultados de retrieval.
    
    Args:
        query: Query del usuario
        max_tokens: Tokens máximos aproximados
        mode: Modo de retrieval
        
    Returns:
        String con contexto formateado
    """
    # Estimar chunks necesarios (aprox 500 tokens por chunk)
    estimated_chunks = max(max_tokens // 500, 3)
    
    response = await retrieve(
        query=query,
        mode=mode,
        top_k=estimated_chunks,
    )
    
    if not response.results:
        return ""
    
    context_parts = []
    current_tokens = 0
    
    for i, result in enumerate(response.results, 1):
        # Estimar tokens (4 chars ~= 1 token)
        result_tokens = len(result.content) // 4
        
        if current_tokens + result_tokens > max_tokens:
            break
        
        # Construir info de fuentes desde scores individuales
        source_parts = []
        if result.vector_score > 0:
            source_parts.append(f"vector: {result.vector_score:.2f}")
        if result.bm25_score > 0:
            source_parts.append(f"bm25: {result.bm25_score:.2f}")
        if result.graph_score > 0:
            source_parts.append(f"graph: {result.graph_score:.2f}")
        source_info = ", ".join(source_parts) if source_parts else "combined"
        
        context_parts.append(
            f"[{i}] (score: {result.final_score:.2f}, {source_info})\n"
            f"{result.content}"
        )
        
        current_tokens += result_tokens
    
    return "\n\n---\n\n".join(context_parts)


async def get_retrieval_stats() -> Dict[str, Any]:
    """
    Obtiene estadísticas del sistema de retrieval.
    
    Returns:
        Dict con estadísticas
    """
    config = get_retrieval_config()
    
    return {
        "config": {
            "vector_weight": config.vector_weight,
            "bm25_weight": config.bm25_weight,
            "graph_weight": config.graph_weight,
            "combination_method": config.combination_method,
            "normalize_method": config.normalize_method,
        },
        "modes_available": [m.value for m in RetrievalMode],
        "sources": [s.value for s in RetrievalSource],
    }
