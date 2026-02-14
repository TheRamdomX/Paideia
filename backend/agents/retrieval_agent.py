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
    
    try:
        # Importar componentes de retrieval
        from backend.retrieval.vector import search_similar
        from backend.retrieval.bm25 import search_bm25
        from backend.retrieval.hybrid_ranker import merge_and_rank
        from backend.graph.traversal import expand_by_concepts
        from backend.ingestion.vectorizer import get_embedding
        
        # 1. Obtener embedding de la query
        embed_start = time.time()
        query_embedding = await get_embedding(
            query,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
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
            vector_results = await search_similar(
                embedding=query_embedding,
                top_k=strategy.top_k,
                min_score=strategy.min_score,
                chunk_type=strategy.chunk_type_filter,
                database=database,
            )
            raw_vector = len(vector_results)
            results = [_convert_to_item(r) for r in vector_results]
            
        elif strategy.strategy == RetrievalStrategy.BM25_ONLY:
            # Solo BM25
            bm25_results = await search_bm25(
                query=query,
                top_k=strategy.top_k,
                chunk_type=strategy.chunk_type_filter,
                database=database,
            )
            raw_bm25 = len(bm25_results)
            results = [_convert_to_item(r) for r in bm25_results]
            
        elif strategy.strategy in [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH_ENHANCED]:
            # Hybrid: Vector + BM25
            vector_results = await search_similar(
                embedding=query_embedding,
                top_k=strategy.top_k * 2,  # Más para merge
                min_score=strategy.min_score * 0.8,
                chunk_type=strategy.chunk_type_filter,
                database=database,
            )
            raw_vector = len(vector_results)
            
            bm25_results = await search_bm25(
                query=query,
                top_k=strategy.top_k * 2,
                chunk_type=strategy.chunk_type_filter,
                database=database,
            )
            raw_bm25 = len(bm25_results)
            
            # Merge y ranking
            merged = await merge_and_rank(
                vector_results=vector_results,
                bm25_results=bm25_results,
                vector_weight=strategy.vector_weight,
                bm25_weight=strategy.bm25_weight,
                top_k=strategy.top_k,
            )
            
            results = [_convert_to_item(r) for r in merged]
            
            # Si es GRAPH_ENHANCED, expandir por conceptos
            if strategy.strategy == RetrievalStrategy.GRAPH_ENHANCED and strategy.expand_graph:
                graph_start = time.time()
                
                # Extraer conceptos de los resultados
                concepts = set()
                for r in results[:5]:  # Top 5
                    concepts.update(r.concepts)
                
                if concepts:
                    graph_results = await expand_by_concepts(
                        concepts=list(concepts),
                        exclude_ids=[r.id for r in results],
                        top_k=strategy.top_k // 2,
                        database=database,
                    )
                    raw_graph = len(graph_results)
                    
                    # Agregar con peso de graph
                    for gr in graph_results:
                        item = _convert_to_item(gr)
                        item.score *= strategy.graph_weight
                        results.append(item)
                    
                    # Re-ordenar
                    results.sort(key=lambda x: x.score, reverse=True)
                    results = results[:strategy.top_k]
                
                graph_time = int((time.time() - graph_start) * 1000)
            else:
                graph_time = 0
        
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
        graph_time_ms=graph_time if 'graph_time' in dir() else 0,
        total_time_ms=total_time,
        raw_vector_results=raw_vector,
        raw_bm25_results=raw_bm25,
        raw_graph_results=raw_graph,
    )


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
        from backend.retrieval.vector import search_similar
        
        # Buscar ejercicios
        results = await search_similar(
            embedding=query_embedding,
            top_k=strategy.top_k,
            min_score=strategy.min_score,
            chunk_type="exercise",
            database=database,
        )
        
        items = []
        for r in results:
            # Solo metadata, contenido mínimo
            item = RetrievalResultItem(
                id=r.get("id", ""),
                content="",  # Sin contenido para EXERCISE_LIST
                score=r.get("score", 0.0),
                chunk_type="exercise",
                source=r.get("source", ""),
                concepts=r.get("concepts", []),
                metadata=r.get("metadata", {}),
                exercise_id=r.get("exercise_id") or r.get("id"),
                exercise_title=r.get("title") or r.get("metadata", {}).get("title", "Sin título"),
                difficulty=r.get("difficulty") or r.get("metadata", {}).get("difficulty", "medium"),
            )
            items.append(item)
        
        return items
        
    except Exception as e:
        logger.error(f"Error retrieving exercise metadata: {e}")
        return []


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
