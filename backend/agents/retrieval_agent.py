"""
retrieval_agent.py
Agente de Retrieval - Decide estrategia y ejecuta recuperación de contexto.
Combina múltiples fuentes: grafo, BM25, embeddings.
Soporta modos pedagógicos: CONCEPT, PRACTICE, EXERCISE_LIST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.mode_router import LearningMode
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
    learning_mode: LearningMode = LearningMode.CONCEPT
    
    # Pesos para cada fuente
    vector_weight: float = 0.4
    bm25_weight: float = 0.3
    graph_weight: float = 0.3
    
    # Parámetros
    top_k: int = 10
    min_score: float = 0.3
    expand_concepts: bool = True
    
    # Filtros por tipo de chunk
    chunk_type_filter: Optional[str] = None
    
    # Para EXERCISE_LIST: solo metadatos
    metadata_only: bool = False
    
    # Razón de la decisión
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "mode": self.mode.value,
            "learning_mode": self.learning_mode.value,
            "vector_weight": self.vector_weight,
            "bm25_weight": self.bm25_weight,
            "graph_weight": self.graph_weight,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "expand_concepts": self.expand_concepts,
            "chunk_type_filter": self.chunk_type_filter,
            "metadata_only": self.metadata_only,
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
    learning_mode: LearningMode = LearningMode.CONCEPT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "strategy_used": self.strategy_used.to_dict() if self.strategy_used else None,
            "query_text": self.query_text,
            "total_found": self.total_found,
            "sources_used": self.sources_used,
            "learning_mode": self.learning_mode.value,
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
    mode: LearningMode = LearningMode.CONCEPT,
) -> StrategyDecision:
    """
    Elige la estrategia óptima de retrieval según el contexto y modo pedagógico.
    
    Args:
        query: Texto de la consulta
        student_id: ID del estudiante (para personalización)
        query_type: Tipo de consulta detectado
        previous_results: Resultados previos (para refinamiento)
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        StrategyDecision con la estrategia elegida
    """
    decision = StrategyDecision()
    decision.learning_mode = mode
    reasons: List[str] = []
    
    # ==========================================
    # PASO 1: Ajustar según MODO PEDAGÓGICO
    # ==========================================
    
    if mode == LearningMode.EXERCISE_LIST:
        # EXERCISE_LIST: Solo traversal de grafo, solo metadatos
        decision.strategy = RetrievalStrategy.GRAPH
        decision.mode = RetrievalMode.GRAPH_ONLY
        decision.vector_weight = 0.0
        decision.bm25_weight = 0.0
        decision.graph_weight = 1.0
        decision.expand_concepts = True
        decision.chunk_type_filter = "exercise"
        decision.metadata_only = True
        decision.top_k = 20  # Más resultados para lista
        decision.min_score = 0.1  # Umbral bajo para listar más
        reasons.append("exercise_list_mode_graph_only")
        
    elif mode == LearningMode.PRACTICE:
        # PRACTICE: Priorizar ejemplos resueltos
        decision.strategy = RetrievalStrategy.HYBRID
        decision.mode = RetrievalMode.HYBRID
        decision.vector_weight = 0.45
        decision.bm25_weight = 0.15
        decision.graph_weight = 0.4
        decision.expand_concepts = True
        decision.chunk_type_filter = "worked_example"
        decision.top_k = 10
        reasons.append("practice_mode_worked_examples")
        
    else:  # LearningMode.CONCEPT
        # CONCEPT: Balance con prioridad a grafo para relaciones
        decision.strategy = RetrievalStrategy.HYBRID
        decision.mode = RetrievalMode.HYBRID
        decision.vector_weight = 0.35
        decision.bm25_weight = 0.25
        decision.graph_weight = 0.4
        decision.expand_concepts = True
        decision.top_k = 10
        reasons.append("concept_mode_balanced")
    
    # ==========================================
    # PASO 2: Ajustes adicionales por tipo de query (solo si no es EXERCISE_LIST)
    # ==========================================
    
    if mode != LearningMode.EXERCISE_LIST:
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
        
        # Ajustar pesos según tipo de query
        if is_definition and mode == LearningMode.CONCEPT:
            # Definiciones: priorizar BM25 para matches exactos
            decision.bm25_weight = min(0.5, decision.bm25_weight + 0.15)
            decision.vector_weight = max(0.2, decision.vector_weight - 0.1)
            decision.expand_concepts = False
            reasons.append("definition_query_bm25_priority")
            
        elif is_comparison:
            # Comparaciones: priorizar grafo para relaciones
            decision.graph_weight = min(0.55, decision.graph_weight + 0.15)
            decision.expand_concepts = True
            reasons.append("comparison_query_graph_priority")
            
        elif is_howto and mode == LearningMode.PRACTICE:
            # Procedimientos en modo práctica: boost adicional
            decision.vector_weight = min(0.5, decision.vector_weight + 0.1)
            reasons.append("howto_query_practice_boost")
            
        elif is_why:
            # Explicaciones causales: priorizar semántica
            decision.vector_weight = min(0.5, decision.vector_weight + 0.1)
            reasons.append("why_query_semantic_priority")
        
        # Ajustar según longitud de query
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
    
    # ==========================================
    # PASO 3: Personalización según perfil del estudiante
    # ==========================================
    
    if student_id:
        profile = await load_profile(student_id)
        
        if profile:
            level_config = get_level_config(profile.level)
            
            # Estudiantes principiantes: más contexto de grafo
            if profile.level in (ProficiencyLevel.BEGINNER, ProficiencyLevel.ELEMENTARY):
                decision.graph_weight = min(0.55, decision.graph_weight + 0.1)
                decision.expand_concepts = True
                decision.top_k = min(20, decision.top_k + 5)  # Más resultados para contexto
                reasons.append(f"beginner_level_graph_boost")
            
            # Estudiantes avanzados: más precisión
            elif profile.level in (ProficiencyLevel.ADVANCED, ProficiencyLevel.EXPERT):
                decision.vector_weight = min(0.55, decision.vector_weight + 0.1)
                decision.min_score = max(decision.min_score, 0.4)  # Umbral más alto
                decision.top_k = max(5, decision.top_k - 2)  # Menos pero más relevantes
                reasons.append(f"advanced_level_precision_boost")
    
    # ==========================================
    # PASO 4: Ajustar si hay resultados previos pobres
    # ==========================================
    
    if previous_results is not None:
        avg_score = (
            sum(r.final_score for r in previous_results) / len(previous_results)
            if previous_results else 0
        )
        
        if avg_score < 0.4:
            # Resultados pobres: expandir búsqueda
            decision.expand_concepts = True
            decision.top_k = min(25, decision.top_k + 5)
            decision.min_score = max(0.15, decision.min_score - 0.1)
            reasons.append("poor_previous_results_expand")
    
    # ==========================================
    # PASO 5: Normalizar pesos y determinar modo final
    # ==========================================
    
    # Normalizar pesos para que sumen ~1.0
    total_weight = decision.vector_weight + decision.bm25_weight + decision.graph_weight
    if total_weight > 0 and mode != LearningMode.EXERCISE_LIST:
        decision.vector_weight /= total_weight
        decision.bm25_weight /= total_weight
        decision.graph_weight /= total_weight
    
    # Determinar modo de retrieval si no fue forzado
    if mode != LearningMode.EXERCISE_LIST:
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
    openai_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> RetrievalResult:
    """
    Ejecuta retrieval con la estrategia seleccionada y modo pedagógico.
    
    Args:
        query: Texto de la consulta
        student_id: ID del estudiante
        strategy: Estrategia a usar (si no se proporciona, se decide)
        concepts: Conceptos conocidos relacionados
        filters: Filtros adicionales
        openai_api_key: API key de OpenAI del cliente
        google_api_key: API key de Google del cliente
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        RetrievalResult con los documentos recuperados
    """
    # Decidir estrategia si no se proporciona
    if strategy is None:
        strategy = await decide_strategy(query, student_id, mode=mode)
    
    # Aplicar filtros de tipo de chunk según estrategia
    final_filters = filters.copy() if filters else {}
    if strategy.chunk_type_filter:
        final_filters["chunk_type"] = strategy.chunk_type_filter
    
    # Construir query de retrieval
    retrieval_query = RetrievalQuery(
        text=query,
        concepts=concepts or [],
        filters=final_filters,
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
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
        )
        
        # Post-procesar resultados según modo
        results = response.results
        
        # Para EXERCISE_LIST: filtrar a solo metadatos
        if strategy.metadata_only and mode == LearningMode.EXERCISE_LIST:
            results = _filter_to_metadata_only(results)
        
        # Construir resultado
        result = RetrievalResult(
            results=results,
            strategy_used=strategy,
            query_text=query,
            total_found=len(results),
            sources_used={
                "vector": response.vector_results,
                "bm25": response.bm25_results,
                "graph": response.graph_results,
            },
            learning_mode=mode,
            metadata={
                "mode": strategy.mode.value,
                "learning_mode": mode.value,
                "config": config_override,
                "metadata_only": strategy.metadata_only,
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
            learning_mode=mode,
            metadata={"error": str(e)},
        )


def _filter_to_metadata_only(results: List[RankedResult]) -> List[RankedResult]:
    """
    Filtra resultados para solo incluir metadatos (para EXERCISE_LIST).
    Retorna solo: título, dificultad, concepto, ID de referencia.
    NO incluye contenido explicativo.
    """
    filtered: List[RankedResult] = []
    
    for r in results:
        # Crear versión con solo metadatos
        metadata = r.metadata.copy() if r.metadata else {}
        
        # Extraer campos relevantes para ejercicios
        exercise_info = {
            "id": r.id,
            "title": metadata.get("title", metadata.get("name", f"Ejercicio {r.id}")),
            "difficulty": metadata.get("difficulty", metadata.get("nivel", "no especificado")),
            "concept": ", ".join(r.concepts) if r.concepts else metadata.get("concept", ""),
            "source_id": r.source_id or metadata.get("source_id", ""),
            "type": metadata.get("type", metadata.get("chunk_type", "exercise")),
        }
        
        # Crear resultado con contenido mínimo (solo identificación)
        filtered_result = RankedResult(
            id=r.id,
            content=f"[{exercise_info['title']}] - Dificultad: {exercise_info['difficulty']}",
            final_score=r.final_score,
            vector_score=r.vector_score,
            bm25_score=r.bm25_score,
            graph_score=r.graph_score,
            sources=r.sources,
            metadata=exercise_info,
            concepts=r.concepts,
            source_id=r.source_id,
            highlights=[],  # Sin highlights para lista
        )
        filtered.append(filtered_result)
    
    return filtered


# ==========================================
# Scoring de Resultados
# ==========================================

async def score_results(
    results: List[RankedResult],
    query: str,
    student_id: Optional[str] = None,
    boost_factors: Optional[Dict[str, float]] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> List[RankedResult]:
    """
    Aplica scoring adicional y personalizado a los resultados.
    
    Args:
        results: Resultados a re-rankear
        query: Query original
        student_id: ID del estudiante
        boost_factors: Factores de boost personalizados
        mode: Modo de aprendizaje pedagógico
        
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
        
        # 1. Boost según modo pedagógico
        chunk_type = result.metadata.get("chunk_type", result.metadata.get("type", ""))
        
        if mode == LearningMode.EXERCISE_LIST:
            # Boost ejercicios, penalizar otros
            if chunk_type in ["exercise", "problem", "ejercicio"]:
                new_score *= 1.3
            else:
                new_score *= 0.5
                
        elif mode == LearningMode.PRACTICE:
            # Boost ejemplos resueltos
            if chunk_type in ["worked_example", "solution", "ejemplo_resuelto"]:
                new_score *= 1.25
            elif chunk_type in ["exercise", "problem"]:
                new_score *= 1.1  # También útil para práctica
                
        else:  # CONCEPT
            # Boost definiciones y conceptos
            if chunk_type in ["definition", "concept", "theory", "definición"]:
                new_score *= 1.2
        
        # 2. Boost por conceptos del estudiante
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
        
        # 3. Boost por fuente
        if result.sources:
            # Use first source if multiple
            source_boost = boost_factors.get(result.sources[0].value, 1.0)
        else:
            source_boost = 1.0
        new_score *= source_boost
        
        # 4. Penalización por contenido muy largo (menos relevante para EXERCISE_LIST)
        if mode != LearningMode.EXERCISE_LIST:
            content_len = len(result.content)
            if content_len > 2000:
                new_score *= 0.95  # Ligera penalización
        
        # 5. Boost por highlights (indica match exacto)
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
    mode: LearningMode = LearningMode.CONCEPT,
) -> RetrievalResult:
    """
    Recupera contenido relevante para conceptos específicos.
    
    Args:
        concepts: Lista de IDs de conceptos
        student_id: ID del estudiante
        top_k: Número de resultados por concepto
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        RetrievalResult combinado
    """
    # Crear query basada en conceptos
    query = f"Explain: {', '.join(concepts)}"
    
    # Ajustar estrategia según modo
    if mode == LearningMode.EXERCISE_LIST:
        strategy = StrategyDecision(
            strategy=RetrievalStrategy.GRAPH,
            mode=RetrievalMode.GRAPH_ONLY,
            learning_mode=mode,
            graph_weight=1.0,
            vector_weight=0.0,
            bm25_weight=0.0,
            top_k=top_k * len(concepts),
            expand_concepts=True,
            chunk_type_filter="exercise",
            metadata_only=True,
            reason="concept_based_exercise_list",
        )
    elif mode == LearningMode.PRACTICE:
        strategy = StrategyDecision(
            strategy=RetrievalStrategy.HYBRID,
            mode=RetrievalMode.HYBRID,
            learning_mode=mode,
            graph_weight=0.5,
            vector_weight=0.35,
            bm25_weight=0.15,
            top_k=top_k * len(concepts),
            expand_concepts=True,
            chunk_type_filter="worked_example",
            reason="concept_based_practice",
        )
    else:
        strategy = StrategyDecision(
            strategy=RetrievalStrategy.GRAPH,
            mode=RetrievalMode.HYBRID,
            learning_mode=mode,
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
        mode=mode,
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
