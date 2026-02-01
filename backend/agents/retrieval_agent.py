"""
retrieval_agent.py
Agente de Retrieval - Decide estrategia y ejecuta recuperación de contexto.
Combina múltiples fuentes: grafo, BM25, embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.graphs.retrieval_graph import (
    RetrievalMode,
    RetrievalQuery,
    RetrievalResponse,
    retrieve as run_retrieval_graph,
)
from backend.memory.student_profile import (
    ProficiencyLevel,
    StudentProfile,
    load_profile,
    get_level_config,
)
from backend.retrieval.hybrid_ranker import (
    RankedResult,
    normalize_scores,
    select_top_k,
)
from backend.settings import get_rag_config


# ==========================================
# Estrategias de Retrieval
# ==========================================

class RetrievalStrategy(str, Enum):
    """Estrategias disponibles de retrieval."""
    VECTOR = "vector"           # Solo búsqueda semántica
    BM25 = "bm25"              # Solo keyword matching
    GRAPH = "graph"            # Solo traversal de grafo
    HYBRID = "hybrid"          # Combinación ponderada
    ADAPTIVE = "adaptive"      # Decide según contexto


@dataclass
class StrategyDecision:
    """Resultado de la decisión de estrategia."""
    
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    mode: RetrievalMode = RetrievalMode.HYBRID
    
    # Pesos para cada fuente
    vector_weight: float = 0.4
    bm25_weight: float = 0.3
    graph_weight: float = 0.3
    
    # Parámetros
    top_k: int = 10
    min_score: float = 0.3
    expand_concepts: bool = True
    
    # Razón de la decisión
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "graph_weight": self.graph_weight,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "expand_concepts": self.expand_concepts,
            "reason": self.reason,
        }


@dataclass
class RetrievalResult:
    """Resultado del agente de retrieval."""
    
    results: List[RankedResult] = field(default_factory=list)
    strategy_used: Optional[StrategyDecision] = None
    query_text: str = ""
    total_found: int = 0
    sources_used: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "strategy_used": self.strategy_used.to_dict() if self.strategy_used else None,
            "query_text": self.query_text,
            "total_found": self.total_found,
            "sources_used": self.sources_used,
            "metadata": self.metadata,
        }


# ==========================================
# Decisión de Estrategia
# ==========================================

async def decide_strategy(
    query: str,
    student_id: Optional[str] = None,
    query_type: Optional[str] = None,
    previous_results: Optional[List[RankedResult]] = None,
) -> StrategyDecision:
    """
    Elige la estrategia óptima de retrieval según el contexto.
    
    Args:
        query: Texto de la consulta
        student_id: ID del estudiante (para personalización)
        query_type: Tipo de consulta detectado
        previous_results: Resultados previos (para refinamiento)
        
    Returns:
        StrategyDecision con la estrategia elegida
    """
    decision = StrategyDecision()
    reasons: List[str] = []
    
    # 1. Analizar características de la query
    query_lower = query.lower()
    query_len = len(query.split())
    
    # Detectar tipo de consulta
    is_definition = any(
        kw in query_lower 
        for kw in ["qué es", "what is", "define", "definición", "definition"]
    )
    is_comparison = any(
        kw in query_lower
        for kw in ["diferencia", "difference", "versus", "vs", "comparar", "compare"]
    )
    is_howto = any(
        kw in query_lower
        for kw in ["cómo", "how to", "pasos", "steps", "proceso", "process"]
    )
    is_why = any(
        kw in query_lower
        for kw in ["por qué", "why", "razón", "reason", "causa", "cause"]
    )
    
    # 2. Ajustar pesos según tipo de query
    if is_definition:
        # Definiciones: priorizar BM25 para matches exactos
        decision.bm25_weight = 0.5
        decision.vector_weight = 0.3
        decision.graph_weight = 0.2
        decision.expand_concepts = False
        reasons.append("definition_query_bm25_priority")
        
    elif is_comparison:
        # Comparaciones: priorizar grafo para relaciones
        decision.graph_weight = 0.5
        decision.vector_weight = 0.3
        decision.bm25_weight = 0.2
        decision.expand_concepts = True
        reasons.append("comparison_query_graph_priority")
        
    elif is_howto:
        # Procedimientos: balance vector y grafo
        decision.vector_weight = 0.4
        decision.graph_weight = 0.4
        decision.bm25_weight = 0.2
        reasons.append("howto_query_balanced")
        
    elif is_why:
        # Explicaciones causales: priorizar semántica
        decision.vector_weight = 0.5
        decision.graph_weight = 0.35
        decision.bm25_weight = 0.15
        reasons.append("why_query_semantic_priority")
    
    # 3. Ajustar según longitud de query
    if query_len <= 3:
        # Queries cortas: más BM25
        decision.bm25_weight = min(0.5, decision.bm25_weight + 0.1)
        decision.vector_weight = max(0.2, decision.vector_weight - 0.1)
        reasons.append("short_query_bm25_boost")
        
    elif query_len >= 15:
        # Queries largas: más vector
        decision.vector_weight = min(0.6, decision.vector_weight + 0.15)
        decision.bm25_weight = max(0.15, decision.bm25_weight - 0.1)
        reasons.append("long_query_vector_boost")
    
    # 4. Personalización según perfil del estudiante
    if student_id:
        profile = await load_profile(student_id)
        
        if profile:
            level_config = get_level_config(profile.level)
            
            # Estudiantes principiantes: más contexto de grafo
            if profile.level in (ProficiencyLevel.BEGINNER, ProficiencyLevel.ELEMENTARY):
                decision.graph_weight = min(0.5, decision.graph_weight + 0.1)
                decision.expand_concepts = True
                decision.top_k = 15  # Más resultados para contexto
                reasons.append(f"beginner_level_graph_boost")
            
            # Estudiantes avanzados: más precisión
            elif profile.level in (ProficiencyLevel.ADVANCED, ProficiencyLevel.EXPERT):
                decision.vector_weight = min(0.55, decision.vector_weight + 0.1)
                decision.min_score = 0.4  # Umbral más alto
                decision.top_k = 8  # Menos pero más relevantes
                reasons.append(f"advanced_level_precision_boost")
    
    # 5. Ajustar si hay resultados previos pobres
    if previous_results is not None:
        avg_score = (
            sum(r.final_score for r in previous_results) / len(previous_results)
            if previous_results else 0
        )
        
        if avg_score < 0.4:
            # Resultados pobres: expandir búsqueda
            decision.expand_concepts = True
            decision.top_k = min(20, decision.top_k + 5)
            decision.min_score = max(0.2, decision.min_score - 0.1)
            reasons.append("poor_previous_results_expand")
    
    # 6. Determinar modo final
    if decision.vector_weight >= 0.5:
        if decision.bm25_weight < 0.2 and decision.graph_weight < 0.2:
            decision.mode = RetrievalMode.VECTOR_ONLY
        else:
            decision.mode = RetrievalMode.HYBRID
    elif decision.bm25_weight >= 0.5:
        if decision.vector_weight < 0.2 and decision.graph_weight < 0.2:
            decision.mode = RetrievalMode.BM25_ONLY
        else:
            decision.mode = RetrievalMode.HYBRID
    elif decision.graph_weight >= 0.5:
        decision.mode = RetrievalMode.HYBRID  # Grafo siempre combinado
    else:
        decision.mode = RetrievalMode.HYBRID
    
    decision.reason = "; ".join(reasons) if reasons else "default_hybrid"
    
    return decision


# ==========================================
# Ejecución de Retrieval
# ==========================================

async def retrieve(
    query: str,
    student_id: Optional[str] = None,
    strategy: Optional[StrategyDecision] = None,
    concepts: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
    """
    Ejecuta retrieval con la estrategia seleccionada.
    
    Args:
        query: Texto de la consulta
        student_id: ID del estudiante
        strategy: Estrategia a usar (si no se proporciona, se decide)
        concepts: Conceptos conocidos relacionados
        filters: Filtros adicionales
        
    Returns:
        RetrievalResult con los documentos recuperados
    """
    # Decidir estrategia si no se proporciona
    if strategy is None:
        strategy = await decide_strategy(query, student_id)
    
    # Construir query de retrieval
    retrieval_query = RetrievalQuery(
        text=query,
        concepts=concepts or [],
        filters=filters or {},
        mode=strategy.mode,
        top_k=strategy.top_k,
    )
    
    # Ejecutar retrieval graph con pesos personalizados
    config_override = {
        "vector_weight": strategy.vector_weight,
        "bm25_weight": strategy.bm25_weight,
        "graph_weight": strategy.graph_weight,
        "expand_concepts": strategy.expand_concepts,
        "min_score": strategy.min_score,
    }
    
    try:
        response = await run_retrieval_graph(
            query=retrieval_query,
            config_override=config_override,
        )
        
        # Construir resultado
        result = RetrievalResult(
            results=response.results,
            strategy_used=strategy,
            query_text=query,
            total_found=len(response.results),
            sources_used={
                "vector": response.vector_results,
                "bm25": response.bm25_results,
                "graph": response.graph_results,
            },
            metadata={
                "mode": strategy.mode.value,
                "config": config_override,
            },
        )
        
        return result
        
    except Exception as e:
        # Fallback a búsqueda simple
        return RetrievalResult(
            results=[],
            strategy_used=strategy,
            query_text=query,
            total_found=0,
            metadata={"error": str(e)},
        )


# ==========================================
# Scoring de Resultados
# ==========================================

async def score_results(
    results: List[RankedResult],
    query: str,
    student_id: Optional[str] = None,
    boost_factors: Optional[Dict[str, float]] = None,
) -> List[RankedResult]:
    """
    Aplica scoring adicional y personalizado a los resultados.
    
    Args:
        results: Resultados a re-rankear
        query: Query original
        student_id: ID del estudiante
        boost_factors: Factores de boost personalizados
        
    Returns:
        Resultados con scores actualizados
    """
    if not results:
        return []
    
    boost_factors = boost_factors or {}
    
    # Cargar perfil para personalización
    profile = None
    if student_id:
        profile = await load_profile(student_id)
    
    scored_results: List[RankedResult] = []
    
    for result in results:
        new_score = result.final_score
        
        # 1. Boost por conceptos del estudiante
        if profile and profile.concept_mastery:
            # Boost chunks de conceptos que el estudiante domina
            for concept_id in result.concepts:
                if concept_id in profile.concept_mastery:
                    mastery = profile.concept_mastery[concept_id]
                    # Boost moderado para conceptos familiares
                    if mastery.mastery_score > 0.7:
                        new_score *= 1.05
                    # Boost mayor para debilidades (priorizar refuerzo)
                    elif concept_id in profile.weaknesses:
                        new_score *= 1.15
        
        # 2. Boost por fuente
        if result.sources:
            # Use first source if multiple
            source_boost = boost_factors.get(result.sources[0].value, 1.0)
        else:
            source_boost = 1.0
        new_score *= source_boost
        
        # 3. Penalización por contenido muy largo
        content_len = len(result.content)
        if content_len > 2000:
            new_score *= 0.95  # Ligera penalización
        
        # 4. Boost por highlights (indica match exacto)
        if result.highlights:
            highlight_boost = min(1.1, 1 + len(result.highlights) * 0.02)
            new_score *= highlight_boost
        
        # Actualizar score
        result.final_score = min(1.0, new_score)
        scored_results.append(result)
    
    # Re-ordenar por score
    scored_results.sort(key=lambda r: r.final_score, reverse=True)
    
    return scored_results


# ==========================================
# Utilidades
# ==========================================

async def retrieve_for_concepts(
    concepts: List[str],
    student_id: Optional[str] = None,
    top_k: int = 5,
) -> RetrievalResult:
    """
    Recupera contenido relevante para conceptos específicos.
    
    Args:
        concepts: Lista de IDs de conceptos
        student_id: ID del estudiante
        top_k: Número de resultados por concepto
        
    Returns:
        RetrievalResult combinado
    """
    # Crear query basada en conceptos
    query = f"Explain: {', '.join(concepts)}"
    
    strategy = StrategyDecision(
        strategy=RetrievalStrategy.GRAPH,
        mode=RetrievalMode.HYBRID,
        graph_weight=0.6,
        vector_weight=0.3,
        bm25_weight=0.1,
        top_k=top_k * len(concepts),
        expand_concepts=True,
        reason="concept_based_retrieval",
    )
    
    return await retrieve(
        query=query,
        student_id=student_id,
        strategy=strategy,
        concepts=concepts,
    )


def get_retrieval_summary(result: RetrievalResult) -> Dict[str, Any]:
    """
    Obtiene resumen del retrieval para debugging.
    
    Args:
        result: Resultado del retrieval
        
    Returns:
        Diccionario con resumen
    """
    if not result.results:
        return {
            "total": 0,
            "strategy": result.strategy_used.strategy.value if result.strategy_used else "unknown",
        }
    
    scores = [r.final_score for r in result.results]
    
    return {
        "total": result.total_found,
        "strategy": result.strategy_used.strategy.value if result.strategy_used else "unknown",
        "mode": result.strategy_used.mode.value if result.strategy_used else "unknown",
        "sources": result.sources_used,
        "score_stats": {
            "min": min(scores),
            "max": max(scores),
            "avg": sum(scores) / len(scores),
        },
        "top_concepts": _extract_top_concepts(result.results),
    }


def _extract_top_concepts(results: List[RankedResult], limit: int = 5) -> List[str]:
    """Extrae los conceptos más frecuentes."""
    concept_counts: Dict[str, int] = {}
    
    for result in results:
        for concept in result.concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    sorted_concepts = sorted(
        concept_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [c for c, _ in sorted_concepts[:limit]]
