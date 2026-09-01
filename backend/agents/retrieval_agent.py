"""
retrieval_agent.py
Agente de Retrieval PURO - 100% MODE DRIVEN

Este agente es DETERMINISTA y PASIVO:
- Solo recibe StrategyDecision
- Solo devuelve context (RetrievalResult)
- NO evalúa calidad
- NO decide pedagogía  
- NO detecta mismatch
- NO toma decisiones

El Orchestrator construye la estrategia según el modo.
Reflection evalúa los resultados.
Este agente solo ejecuta.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.mode_router import LearningMode

logger = logging.getLogger(__name__)


# ==========================================
# Enums y Tipos
# ==========================================

class RetrievalStrategy(str, Enum):
    """Estrategias de retrieval disponibles."""
    VECTOR_ONLY = "vector_only"        # Solo búsqueda vectorial
    BM25_ONLY = "bm25_only"            # Solo BM25 keyword search
    HYBRID = "hybrid"                   # Vector + BM25
    GRAPH_ENHANCED = "graph_enhanced"   # Hybrid + Graph traversal
    METADATA_ONLY = "metadata_only"     # Solo metadata (para EXERCISE_LIST)


# ==========================================
# Data Classes
# ==========================================

@dataclass
class StrategyDecision:
    """
    Decisión de estrategia de retrieval.
    
    Esta estructura es CONSTRUIDA por el Orchestrator basándose en el modo.
    El retrieval_agent la recibe y la ejecuta sin modificarla.
    """
    strategy: RetrievalStrategy
    mode: LearningMode
    top_k: int = 10
    min_score: float = 0.3
    
    # Pesos para hybrid search
    vector_weight: float = 0.35
    bm25_weight: float = 0.25
    graph_weight: float = 0.4
    
    # Filtros
    chunk_type_filter: Optional[str] = None  # "exercise", "concept", etc.
    concept_filter: Optional[List[str]] = None
    source_filter: Optional[List[str]] = None
    
    # Flags
    expand_graph: bool = True
    metadata_only: bool = False
    include_prerequisites: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "graph_weight": self.graph_weight,
            "chunk_type_filter": self.chunk_type_filter,
            "expand_graph": self.expand_graph,
            "metadata_only": self.metadata_only,
        }


@dataclass
class RetrievalResultItem:
    """Un item individual de retrieval."""
    id: str
    content: str
    score: float
    chunk_type: str = "chunk"
    source: str = ""
    concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Para EXERCISE_LIST
    exercise_id: Optional[str] = None
    exercise_title: Optional[str] = None
    difficulty: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": self.score,
            "chunk_type": self.chunk_type,
            "source": self.source,
            "concepts": self.concepts[:5],
            "exercise_id": self.exercise_id,
            "exercise_title": self.exercise_title,
            "difficulty": self.difficulty,
        }


@dataclass
class RetrievalResult:
    """
    Resultado del retrieval.
    
    Contiene SOLO datos recuperados.
    NO contiene evaluaciones ni decisiones.
    """
    results: List[RetrievalResultItem] = field(default_factory=list)
    total_found: int = 0
    strategy_used: RetrievalStrategy = RetrievalStrategy.HYBRID
    mode: LearningMode = LearningMode.CONCEPT
    
    # Metadata del retrieval
    query_embedding_time_ms: int = 0
    search_time_ms: int = 0
    graph_time_ms: int = 0
    total_time_ms: int = 0
    
    # Para debugging
    raw_vector_results: int = 0
    raw_bm25_results: int = 0
    raw_graph_results: int = 0
    
    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0
    
    @property
    def top_score(self) -> float:
        if not self.results:
            return 0.0
        return max(r.score for r in self.results)
    
    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_found": self.total_found,
            "result_count": len(self.results),
            "strategy_used": self.strategy_used.value,
            "mode": self.mode.value,
            "top_score": self.top_score,
            "avg_score": self.avg_score,
            "total_time_ms": self.total_time_ms,
        }


# ==========================================
# Funciones de Construcción de Estrategia
# ==========================================

def build_strategy_for_mode(
    mode: LearningMode,
    top_k: int = 10,
    min_score: float = 0.3,
    extra_filters: Optional[Dict[str, Any]] = None,
) -> StrategyDecision:
    """
    Construye una estrategia de retrieval según el modo pedagógico.
    
    Esta función es llamada por el Orchestrator, NO por este agente.
    Está aquí solo por conveniencia de imports.
    
    Args:
        mode: Modo pedagógico activo
        top_k: Número máximo de resultados
        min_score: Score mínimo para incluir
        extra_filters: Filtros adicionales del Orchestrator
        
    Returns:
        StrategyDecision configurada para el modo
    """
    extra_filters = extra_filters or {}
    
    if mode == LearningMode.CONCEPT:
        return StrategyDecision(
            strategy=RetrievalStrategy.GRAPH_ENHANCED,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
            vector_weight=0.35,
            bm25_weight=0.25,
            graph_weight=0.4,
            chunk_type_filter=extra_filters.get("chunk_type"),
            expand_graph=True,
            metadata_only=False,
            include_prerequisites=True,
        )
    
    elif mode == LearningMode.PRACTICE:
        return StrategyDecision(
            strategy=RetrievalStrategy.HYBRID,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
            vector_weight=0.4,
            bm25_weight=0.3,
            graph_weight=0.3,
            chunk_type_filter="worked_example",  # Priorizar ejemplos resueltos
            expand_graph=True,
            metadata_only=False,
            include_prerequisites=False,
        )
    
    elif mode == LearningMode.EXERCISE_LIST:
        # EXERCISE_LIST: Solo metadata de ejercicios, SIN contenido
        return StrategyDecision(
            strategy=RetrievalStrategy.METADATA_ONLY,
            mode=mode,
            top_k=top_k * 2,  # Más resultados para listar
            min_score=0.1,  # Más permisivo
            vector_weight=0.5,
            bm25_weight=0.5,
            graph_weight=0.0,  # Sin graph para listado
            chunk_type_filter="exercise",  # SOLO ejercicios
            expand_graph=False,
            metadata_only=True,  # Solo metadata
            include_prerequisites=False,
        )
    
    else:
        # Default: CONCEPT-like
        return StrategyDecision(
            strategy=RetrievalStrategy.HYBRID,
            mode=mode,
            top_k=top_k,
            min_score=min_score,
        )


# ==========================================
# Función Principal de Retrieval
# ==========================================

async def retrieve(
    query: str,
    strategy: StrategyDecision,
    student_id: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    mode: Optional[LearningMode] = None,
    database: str = "paideia",
) -> RetrievalResult:
    """
    Ejecuta retrieval según la estrategia dada.
    
    Este es un agente PURO:
    - Recibe estrategia completa
    - Ejecuta retrieval
    - Devuelve resultados
    - NO toma decisiones
    
    Args:
        query: Query del usuario
        strategy: Estrategia de retrieval (construida por Orchestrator)
        student_id: ID del estudiante (para filtros opcionales)
        openai_api_key: API key de OpenAI para embeddings
        google_api_key: API key de Google para embeddings
        mode: Modo (opcional, ya incluido en strategy)
        database: Base de datos a usar
        
    Returns:
        RetrievalResult con los resultados recuperados
    """
    import time
    start_time = time.time()
    
    actual_mode = mode or strategy.mode
    
    logger.info(
        f"Retrieval ejecutando: strategy={strategy.strategy.value}, "
        f"mode={actual_mode.value}, top_k={strategy.top_k}"
    )
    
    results: List[RetrievalResultItem] = []
    raw_vector = 0
    raw_bm25 = 0
    raw_graph = 0
    embed_time = 0
    search_time = 0
    graph_time = 0

    try:
        # Importar componentes de retrieval
        # NOTA: las funciones de búsqueda resuelven la conexión con get_db();
        # el parámetro `database` se mantiene en la firma pública pero la
        # selección de base se hace en la capa de dependencias (db/pool.py).
        from backend.graph.traversal import expand_concepts
        from backend.models.embeddings import embed_text
        from backend.retrieval.bm25 import search_text
        from backend.retrieval.vector import search_by_embedding

        # 1. Obtener embedding de la query
        embed_start = time.time()
        query_embedding = await embed_text(
            query,
            user_openai_key=openai_api_key,
            user_google_key=google_api_key,
        )
        embed_time = int((time.time() - embed_start) * 1000)
        
        # 2. Ejecutar según estrategia
        search_start = time.time()
        
        if strategy.strategy == RetrievalStrategy.METADATA_ONLY:
            # EXERCISE_LIST: Solo buscar metadata de ejercicios
            results = await _retrieve_exercise_metadata(
                query=query,
                query_embedding=query_embedding,
                strategy=strategy,
                database=database,
            )
            raw_vector = len(results)
            
        elif strategy.strategy == RetrievalStrategy.VECTOR_ONLY:
            # Solo vector search
            vector_hits = _filter_chunk_type(
                _vector_to_hits(await search_by_embedding(
                    embedding=query_embedding,
                    limit=strategy.top_k * 2,
                    min_score=strategy.min_score,
                )),
                strategy.chunk_type_filter,
            )
            raw_vector = len(vector_hits)
            results = [_convert_to_item(h) for h in vector_hits[:strategy.top_k]]

        elif strategy.strategy == RetrievalStrategy.BM25_ONLY:
            # Solo BM25
            bm25_hits = _filter_chunk_type(
                _bm25_to_hits(await search_text(
                    query=query,
                    limit=strategy.top_k * 2,
                )),
                strategy.chunk_type_filter,
            )
            raw_bm25 = len(bm25_hits)
            results = [_convert_to_item(h) for h in bm25_hits[:strategy.top_k]]

        elif strategy.strategy in [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH_ENHANCED]:
            # Hybrid: Vector + BM25
            vector_hits = _filter_chunk_type(
                _vector_to_hits(await search_by_embedding(
                    embedding=query_embedding,
                    limit=strategy.top_k * 2,  # Más para merge
                    min_score=strategy.min_score * 0.8,
                )),
                strategy.chunk_type_filter,
            )
            raw_vector = len(vector_hits)

            bm25_hits = _filter_chunk_type(
                _bm25_to_hits(await search_text(
                    query=query,
                    limit=strategy.top_k * 2,
                )),
                strategy.chunk_type_filter,
            )
            raw_bm25 = len(bm25_hits)

            graph_hits: List[Dict[str, Any]] = []

            # Si es GRAPH_ENHANCED, expandir por conceptos
            if strategy.strategy == RetrievalStrategy.GRAPH_ENHANCED and strategy.expand_graph:
                graph_start = time.time()

                concept_ids = await _seed_concept_ids(
                    query=query,
                    hits=vector_hits + bm25_hits,
                )

                if concept_ids:
                    traversal = await expand_concepts(
                        concept_ids=concept_ids,
                        max_depth=2,
                        max_nodes=max(strategy.top_k, 10),
                    )
                    graph_hits = _graph_to_hits(traversal)
                    raw_graph = len(graph_hits)

                graph_time = int((time.time() - graph_start) * 1000)

            # Merge y ranking con los pesos del modo
            results = _merge_and_rank(
                vector_hits=vector_hits,
                bm25_hits=bm25_hits,
                graph_hits=graph_hits,
                strategy=strategy,
            )

        search_time = int((time.time() - search_start) * 1000)
        
    except ImportError as e:
        logger.warning(f"Import error in retrieval: {e}. Using mock results.")
        results = await _mock_retrieve(query, strategy)
        embed_time = 0
        search_time = 0
        graph_time = 0
        
    except Exception as e:
        logger.error(f"Error in retrieval: {e}")
        results = []
        embed_time = 0
        search_time = 0
        graph_time = 0
    
    total_time = int((time.time() - start_time) * 1000)
    
    return RetrievalResult(
        results=results,
        total_found=len(results),
        strategy_used=strategy.strategy,
        mode=actual_mode,
        query_embedding_time_ms=embed_time,
        search_time_ms=search_time,
        graph_time_ms=graph_time,
        total_time_ms=total_time,
        raw_vector_results=raw_vector,
        raw_bm25_results=raw_bm25,
        raw_graph_results=raw_graph,
    )


# ==========================================
# Normalización de resultados por fuente
# ==========================================

def _hit_fields(metadata: Optional[Dict[str, Any]]) -> Tuple[str, str, List[str]]:
    """Extrae chunk_type, source y conceptos desde la metadata de un nodo."""
    metadata = metadata or {}
    chunk_type = metadata.get("chunk_type") or metadata.get("type") or "chunk"
    source = metadata.get("source") or metadata.get("source_id") or ""
    concepts = metadata.get("concepts") or []
    return str(chunk_type), str(source), list(concepts)


def _make_hit(
    node_id: str,
    content: str,
    score: float,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construye el dict común que consumen el ranker y _convert_to_item."""
    chunk_type, source, concepts = _hit_fields(metadata)
    return {
        "id": str(node_id),
        "content": content or "",
        "score": float(score or 0.0),
        "chunk_type": chunk_type,
        "source": source,
        "concepts": concepts,
        "metadata": metadata or {},
    }


def _vector_to_hits(results: List[Any]) -> List[Dict[str, Any]]:
    """Normaliza los VectorResult de la búsqueda vectorial."""
    return [
        _make_hit(r.id, r.content, r.score, r.metadata)
        for r in results
    ]


def _bm25_to_hits(response: Any) -> List[Dict[str, Any]]:
    """Normaliza la BM25SearchResponse de la búsqueda literal."""
    hits = []
    for r in getattr(response, "results", []):
        hit = _make_hit(r.id, r.content, r.score, r.metadata)
        hit["highlights"] = list(getattr(r, "highlights", []))
        hits.append(hit)
    return hits


def _graph_to_hits(traversal: Any) -> List[Dict[str, Any]]:
    """
    Normaliza el TraversalResult del recorrido del grafo.

    Incluye tanto los chunks de evidencia como la descripción de los
    conceptos visitados: ambos son contexto válido para el LLM.
    """
    hits = []

    for chunk in traversal.chunks_found:
        hits.append(_make_hit(
            chunk.get("id", ""),
            chunk.get("content", ""),
            chunk.get("relevance", 0.5),
            chunk.get("metadata", {}),
        ))

    for node in traversal.nodes_visited:
        if not node.content:
            continue
        hits.append(_make_hit(
            node.id,
            node.content,
            node.score,
            {"type": "concept", "name": node.name, **node.metadata},
        ))

    return hits


def _filter_chunk_type(
    hits: List[Dict[str, Any]],
    chunk_type: Optional[str],
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """
    Filtra por tipo de chunk.

    Las funciones de búsqueda no filtran por metadata en la query, así que
    el filtro del modo se aplica acá. En modo no estricto (p. ej. PRACTICE
    priorizando 'worked_example') un filtro que deja el resultado vacío se
    descarta: es una preferencia, no un requisito.
    """
    if not chunk_type:
        return hits

    filtered = [h for h in hits if h["chunk_type"] == chunk_type]

    if filtered or strict:
        return filtered

    logger.debug(
        f"Filtro chunk_type='{chunk_type}' sin coincidencias; "
        f"se usan los {len(hits)} resultados sin filtrar"
    )
    return hits


async def _seed_concept_ids(
    query: str,
    hits: List[Dict[str, Any]],
    limit: int = 5,
) -> List[str]:
    """
    Determina los conceptos semilla para el recorrido del grafo.

    Combina los conceptos ya presentes en los resultados (cuando vienen como
    record id de SurrealDB) con una búsqueda de conceptos por texto.
    """
    from backend.graph.traversal import find_concepts_by_query

    seeds: List[str] = []

    for hit in hits[:5]:
        for concept in hit.get("concepts", []):
            concept_id = str(concept)
            if ":" in concept_id and concept_id not in seeds:
                seeds.append(concept_id)

    for node in await find_concepts_by_query(query, limit=limit):
        if node.id and node.id not in seeds:
            seeds.append(node.id)

    return seeds[:limit]


def _merge_and_rank(
    vector_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    graph_hits: List[Dict[str, Any]],
    strategy: StrategyDecision,
) -> List[RetrievalResultItem]:
    """
    Fusiona las tres fuentes con los pesos que dictó el modo pedagógico.

    El min_score ya se aplicó por fuente; acá el corte es por top_k, porque
    el score combinado vive en otra escala que el score crudo de cada fuente.
    """
    from backend.retrieval.hybrid_ranker import (
        HybridRankingConfig,
        combine_scores,
        deduplicate_results,
        select_top_k,
    )

    config = HybridRankingConfig(
        vector_weight=strategy.vector_weight,
        bm25_weight=strategy.bm25_weight,
        graph_weight=strategy.graph_weight,
        max_results=strategy.top_k * 3,  # margen para deduplicar después
    )

    combined = combine_scores(
        {
            "vector": vector_hits,
            "bm25": bm25_hits,
            "graph": graph_hits,
        },
        config,
    )

    unique = deduplicate_results(combined)
    top = select_top_k(unique, k=strategy.top_k)

    return [
        _convert_to_item(_make_hit(r.id, r.content, r.final_score, r.metadata))
        for r in top
    ]


# ==========================================
# Funciones Auxiliares
# ==========================================

async def _retrieve_exercise_metadata(
    query: str,
    query_embedding: List[float],
    strategy: StrategyDecision,
    database: str,
) -> List[RetrievalResultItem]:
    """
    Recupera SOLO metadata de ejercicios para EXERCISE_LIST.
    
    NO incluye contenido explicativo.
    Solo: título, dificultad, conceptos, ID.
    """
    try:
        from backend.retrieval.vector import search_by_embedding

        # Buscar ejercicios
        hits = _filter_chunk_type(
            _vector_to_hits(await search_by_embedding(
                embedding=query_embedding,
                limit=strategy.top_k * 2,
                min_score=strategy.min_score,
            )),
            strategy.chunk_type_filter or "exercise",
            strict=True,  # EXERCISE_LIST solo puede listar ejercicios
        )

        items = []
        for hit in hits[:strategy.top_k]:
            metadata = hit["metadata"]
            # Solo metadata, contenido mínimo
            item = RetrievalResultItem(
                id=hit["id"],
                content="",  # Sin contenido para EXERCISE_LIST
                score=hit["score"],
                chunk_type="exercise",
                source=hit["source"],
                concepts=hit["concepts"],
                metadata=metadata,
                exercise_id=metadata.get("exercise_id") or hit["id"],
                exercise_title=metadata.get("title") or _first_line(hit["content"]),
                difficulty=metadata.get("difficulty", "medium"),
            )
            items.append(item)

        return items
        
    except Exception as e:
        logger.error(f"Error retrieving exercise metadata: {e}")
        return []


def _first_line(content: str, max_chars: int = 80) -> str:
    """Primera línea del chunk, usada como título cuando no hay metadata."""
    line = (content or "").strip().split("\n")[0].strip()
    if not line:
        return "Sin título"
    return line[:max_chars]


def _convert_to_item(raw_result: Any) -> RetrievalResultItem:
    """Convierte un resultado raw a RetrievalResultItem."""
    if isinstance(raw_result, dict):
        return RetrievalResultItem(
            id=raw_result.get("id", ""),
            content=raw_result.get("content", raw_result.get("text", "")),
            score=raw_result.get("score", raw_result.get("final_score", 0.0)),
            chunk_type=raw_result.get("chunk_type", "chunk"),
            source=raw_result.get("source", ""),
            concepts=raw_result.get("concepts", []),
            metadata=raw_result.get("metadata", {}),
            exercise_id=raw_result.get("exercise_id"),
            exercise_title=raw_result.get("title"),
            difficulty=raw_result.get("difficulty"),
        )
    elif hasattr(raw_result, "id"):
        # Objeto con atributos
        return RetrievalResultItem(
            id=getattr(raw_result, "id", ""),
            content=getattr(raw_result, "content", getattr(raw_result, "text", "")),
            score=getattr(raw_result, "score", getattr(raw_result, "final_score", 0.0)),
            chunk_type=getattr(raw_result, "chunk_type", "chunk"),
            source=getattr(raw_result, "source", ""),
            concepts=getattr(raw_result, "concepts", []),
            metadata=getattr(raw_result, "metadata", {}),
        )
    else:
        return RetrievalResultItem(
            id=str(hash(str(raw_result))),
            content=str(raw_result),
            score=0.5,
        )


async def _mock_retrieve(
    query: str,
    strategy: StrategyDecision,
) -> List[RetrievalResultItem]:
    """
    Mock retrieval para testing cuando los componentes no están disponibles.
    """
    logger.warning("Using mock retrieval results")
    
    if strategy.mode == LearningMode.EXERCISE_LIST:
        return [
            RetrievalResultItem(
                id=f"exercise_{i}",
                content="",
                score=0.9 - i * 0.1,
                chunk_type="exercise",
                exercise_id=f"EX-{i+1}",
                exercise_title=f"Ejercicio {i+1} sobre {query[:30]}",
                difficulty=["easy", "medium", "hard"][i % 3],
                concepts=["concepto_1", "concepto_2"],
            )
            for i in range(5)
        ]
    else:
        return [
            RetrievalResultItem(
                id=f"chunk_{i}",
                content=f"Contenido relacionado con: {query}. Este es el chunk {i+1}.",
                score=0.85 - i * 0.1,
                chunk_type="concept" if strategy.mode == LearningMode.CONCEPT else "worked_example",
                source="documento_ejemplo.pdf",
                concepts=["concepto_1", "concepto_2"],
            )
            for i in range(5)
        ]


# ==========================================
# Exports
# ==========================================

__all__ = [
    "RetrievalStrategy",
    "StrategyDecision",
    "RetrievalResultItem",
    "RetrievalResult",
    "retrieve",
    "build_strategy_for_mode",
]
