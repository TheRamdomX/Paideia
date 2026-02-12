"""
reflection_agent.py
Agente de Reflexión - Evalúa calidad de respuestas y decide acciones correctivas.
Puede solicitar reintento, más contexto, o aplicar fallback.
Soporta modos pedagógicos: CONCEPT, PRACTICE, EXERCISE_LIST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.mode_router import LearningMode
from backend.agents.reasoning_agent import GeneratedAnswer, PromptConfig, _get_default_context_limit
from backend.models.model_limits import get_safe_context_limit
from backend.retrieval.hybrid_ranker import RankedResult
from backend.utils.text import token_count
from backend.settings import get_settings


# ==========================================
# Configuración y Estructuras
# ==========================================

class EvaluationDimension(str, Enum):
    """Dimensiones de evaluación."""
    RELEVANCE = "relevance"       # ¿Responde la pregunta?
    COVERAGE = "coverage"         # ¿Cubre todos los aspectos?
    COHERENCE = "coherence"       # ¿Es coherente y bien estructurada?
    ACCURACY = "accuracy"         # ¿Es precisa según el contexto?
    COMPLETENESS = "completeness" # ¿Está completa?
    MODE_ALIGNMENT = "mode_alignment"  # ¿Está alineada con el modo pedagógico?


class ReflectionDecision(str, Enum):
    """Decisiones del agente de reflexión."""
    ACCEPT = "accept"                    # Aceptar respuesta
    RETRY = "retry"                      # Reintentar generación
    REQUEST_MORE_CONTEXT = "request_more_context"  # Necesita más contexto
    SIMPLIFY = "simplify"                # Simplificar respuesta
    EXPAND = "expand"                    # Expandir respuesta
    FALLBACK = "fallback"                # Usar respuesta de fallback
    MODE_MISMATCH = "mode_mismatch"      # Desalineación con modo pedagógico


@dataclass
class DimensionScore:
    """Puntuación en una dimensión."""
    dimension: EvaluationDimension
    score: float  # 0.0 a 1.0
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass
class EvaluationResult:
    """Resultado de evaluación de respuesta."""
    
    overall_score: float = 0.0
    dimension_scores: List[DimensionScore] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    decision: ReflectionDecision = ReflectionDecision.ACCEPT
    confidence: float = 0.0
    learning_mode: LearningMode = LearningMode.CONCEPT
    mode_alignment_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "issues": self.issues,
            "suggestions": self.suggestions,
            "decision": self.decision.value,
            "confidence": self.confidence,
            "learning_mode": self.learning_mode.value,
            "mode_alignment_score": self.mode_alignment_score,
        }


@dataclass
class RetryConfig:
    """Configuración para reintentos."""
    max_retries: int = 2
    min_score_threshold: float = 0.5
    require_improvement: bool = True
    different_strategy: bool = True
    mode_alignment_threshold: float = 0.5  # Umbral para MODE_MISMATCH


# ==========================================
# Evaluación de Respuestas
# ==========================================

async def evaluate_answer(
    query: str,
    answer: GeneratedAnswer,
    context_chunks: List[RankedResult],
    previous_evaluation: Optional[EvaluationResult] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> EvaluationResult:
    """
    Evalúa la calidad de una respuesta generada.
    
    Args:
        query: Pregunta original
        answer: Respuesta generada
        context_chunks: Contexto usado
        previous_evaluation: Evaluación anterior (si es reintento)
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        Resultado de evaluación
    """
    dimension_scores: List[DimensionScore] = []
    issues: List[str] = []
    suggestions: List[str] = []
    
    # Evaluar cada dimensión
    
    # 1. Relevancia - ¿Responde la pregunta?
    relevance = await _evaluate_relevance(query, answer.answer)
    dimension_scores.append(relevance)
    if relevance.score < 0.5:
        issues.append("La respuesta no parece abordar la pregunta directamente")
        suggestions.append("Regenerar enfocándose más en la pregunta específica")
    
    # 2. Cobertura - ¿Usa el contexto disponible?
    coverage = await _evaluate_coverage(answer.answer, context_chunks)
    dimension_scores.append(coverage)
    if coverage.score < 0.5:
        issues.append("La respuesta no aprovecha suficientemente el contexto")
        suggestions.append("Incluir más información del contexto recuperado")
    
    # 3. Coherencia - ¿Está bien estructurada?
    coherence = await _evaluate_coherence(answer.answer)
    dimension_scores.append(coherence)
    if coherence.score < 0.5:
        issues.append("La respuesta podría estar mejor estructurada")
        suggestions.append("Mejorar organización y flujo de la respuesta")
    
    # 4. Completitud - ¿Es suficientemente completa?
    completeness = await _evaluate_completeness(query, answer.answer, mode)
    dimension_scores.append(completeness)
    if completeness.score < 0.5:
        issues.append("La respuesta parece incompleta")
        suggestions.append("Expandir para cubrir más aspectos")
    
    # 5. NUEVA DIMENSIÓN: Alineación con modo pedagógico
    mode_alignment = await _evaluate_mode_alignment(answer.answer, mode, context_chunks)
    dimension_scores.append(mode_alignment)
    if mode_alignment.score < 0.5:
        issues.append(f"La respuesta no está alineada con el modo {mode.value}")
        suggestions.append(f"Ajustar respuesta para modo {mode.value}")
    
    # Calcular score general (promedio ponderado)
    # Ajustar pesos según modo
    if mode == LearningMode.EXERCISE_LIST:
        weights = {
            EvaluationDimension.RELEVANCE: 0.20,
            EvaluationDimension.COVERAGE: 0.15,
            EvaluationDimension.COHERENCE: 0.15,
            EvaluationDimension.COMPLETENESS: 0.10,
            EvaluationDimension.MODE_ALIGNMENT: 0.40,  # Muy importante para EXERCISE_LIST
        }
    elif mode == LearningMode.PRACTICE:
        weights = {
            EvaluationDimension.RELEVANCE: 0.25,
            EvaluationDimension.COVERAGE: 0.20,
            EvaluationDimension.COHERENCE: 0.20,
            EvaluationDimension.COMPLETENESS: 0.15,
            EvaluationDimension.MODE_ALIGNMENT: 0.20,
        }
    else:  # CONCEPT
        weights = {
            EvaluationDimension.RELEVANCE: 0.30,
            EvaluationDimension.COVERAGE: 0.25,
            EvaluationDimension.COHERENCE: 0.15,
            EvaluationDimension.COMPLETENESS: 0.15,
            EvaluationDimension.MODE_ALIGNMENT: 0.15,
        }
    
    overall_score = sum(
        ds.score * weights.get(ds.dimension, 0.15)
        for ds in dimension_scores
    )
    
    # Determinar decisión (incluyendo modo)
    decision = _determine_decision(
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        issues=issues,
        previous_evaluation=previous_evaluation,
        mode=mode,
    )
    
    return EvaluationResult(
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        issues=issues,
        suggestions=suggestions,
        decision=decision,
        confidence=answer.confidence,
        learning_mode=mode,
        mode_alignment_score=mode_alignment.score,
    )


async def _evaluate_relevance(
    query: str,
    answer: str,
) -> DimensionScore:
    """Evalúa relevancia de la respuesta a la pregunta."""
    # Heurísticas simples (sin LLM adicional)
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    # Palabras clave de la pregunta presentes en respuesta
    overlap = len(query_words & answer_words)
    query_coverage = overlap / len(query_words) if query_words else 0
    
    # Longitud razonable
    length_score = 1.0 if 50 < len(answer) < 5000 else 0.5
    
    # Detectar respuestas genéricas/evasivas
    evasive_phrases = [
        "no tengo información",
        "no puedo responder",
        "no está claro",
        "no sé",
    ]
    evasive_penalty = 0.3 if any(p in answer.lower() for p in evasive_phrases) else 0
    
    score = (query_coverage * 0.6 + length_score * 0.4) - evasive_penalty
    score = max(0.0, min(1.0, score))
    
    return DimensionScore(
        dimension=EvaluationDimension.RELEVANCE,
        score=score,
        reason=f"Cobertura de términos: {query_coverage:.2f}",
    )


async def _evaluate_coverage(
    answer: str,
    context_chunks: List[RankedResult],
) -> DimensionScore:
    """Evalúa si la respuesta usa el contexto disponible."""
    if not context_chunks:
        return DimensionScore(
            dimension=EvaluationDimension.COVERAGE,
            score=0.5,
            reason="Sin contexto para evaluar",
        )
    
    # Verificar cuántos chunks tienen contenido reflejado en la respuesta
    answer_lower = answer.lower()
    chunks_used = 0
    
    for chunk in context_chunks[:5]:  # Evaluar top 5
        # Verificar que el chunk tenga contenido válido
        if not chunk.content:
            continue
            
        # Buscar frases del chunk en la respuesta
        chunk_words = set(chunk.content.lower().split()[:20])  # Primeras 20 palabras
        answer_words = set(answer_lower.split())
        
        overlap = len(chunk_words & answer_words)
        if overlap >= 3:  # Al menos 3 palabras en común
            chunks_used += 1
    
    coverage_ratio = chunks_used / min(5, len(context_chunks)) if context_chunks else 0.5
    
    # Bonus si menciona conceptos
    concept_bonus = 0
    all_concepts = []
    for chunk in context_chunks:
        if chunk.concepts:
            all_concepts.extend(chunk.concepts)
    
    if all_concepts:
        concepts_mentioned = sum(
            1 for c in set(all_concepts) 
            if c.lower() in answer_lower
        )
        concept_bonus = min(0.2, concepts_mentioned * 0.05)
    
    score = min(1.0, coverage_ratio + concept_bonus)
    
    return DimensionScore(
        dimension=EvaluationDimension.COVERAGE,
        score=score,
        reason=f"Chunks usados: {chunks_used}/{min(5, len(context_chunks))}",
    )


async def _evaluate_coherence(answer: str) -> DimensionScore:
    """Evalúa coherencia estructural de la respuesta."""
    score = 1.0
    reasons: List[str] = []
    
    # Longitud razonable
    if len(answer) < 30:
        score -= 0.3
        reasons.append("Muy corta")
    elif len(answer) > 10000:
        score -= 0.2
        reasons.append("Muy larga")
    
    # Tiene estructura (párrafos, puntuación)
    paragraphs = answer.count("\n\n")
    sentences = answer.count(".") + answer.count("?") + answer.count("!")
    
    if sentences < 2:
        score -= 0.2
        reasons.append("Pocas oraciones")
    
    # No tiene texto repetido
    words = answer.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            score -= 0.3
            reasons.append("Posible repetición")
    
    # No termina abruptamente
    if answer and not answer.strip()[-1] in ".!?:)":
        score -= 0.1
        reasons.append("Terminación abrupta")
    
    score = max(0.0, score)
    
    return DimensionScore(
        dimension=EvaluationDimension.COHERENCE,
        score=score,
        reason="; ".join(reasons) if reasons else "Estructura adecuada",
    )


async def _evaluate_completeness(
    query: str,
    answer: str,
    mode: LearningMode = LearningMode.CONCEPT,
) -> DimensionScore:
    """Evalúa completitud de la respuesta según el modo."""
    query_lower = query.lower()
    score = 1.0
    reasons: List[str] = []
    
    # Ajustar expectativas según modo pedagógico
    if mode == LearningMode.EXERCISE_LIST:
        # Para lista de ejercicios, no necesita ser larga
        min_length = 30
        expected_structure = ["ejercicio", "dificultad", "concepto"]
    elif mode == LearningMode.PRACTICE:
        # Para práctica, necesita pasos
        min_length = 150
        expected_structure = ["paso", "1.", "resultado", "solución"]
    else:  # CONCEPT
        # Detectar tipo de pregunta para CONCEPT
        if any(w in query_lower for w in ["comparar", "diferencia", "vs", "versus"]):
            min_length = 200
            expected_structure = ["por un lado", "por otro", "mientras que", "en cambio"]
        elif any(w in query_lower for w in ["explicar", "explica", "cómo", "por qué"]):
            min_length = 150
            expected_structure = ["porque", "debido", "por lo tanto", "esto significa"]
        elif any(w in query_lower for w in ["qué es", "qué son", "definir", "define"]):
            min_length = 80
            expected_structure = ["es", "se refiere", "significa", "se define"]
        else:
            min_length = 100
            expected_structure = []
    
    # Verificar longitud
    if len(answer) < min_length:
        length_penalty = (min_length - len(answer)) / min_length * 0.3
        score -= length_penalty
        reasons.append(f"Longitud: {len(answer)}/{min_length}")
    
    # Verificar estructura esperada
    if expected_structure:
        structures_found = sum(
            1 for s in expected_structure 
            if s in answer.lower()
        )
        structure_ratio = structures_found / len(expected_structure)
        if structure_ratio < 0.3:
            score -= 0.2
            reasons.append("Falta estructura esperada")
    
    score = max(0.0, score)
    
    return DimensionScore(
        dimension=EvaluationDimension.COMPLETENESS,
        score=score,
        reason="; ".join(reasons) if reasons else "Completitud adecuada",
    )


async def _evaluate_mode_alignment(
    answer: str,
    mode: LearningMode,
    context_chunks: List[RankedResult],
) -> DimensionScore:
    """
    Evalúa si la respuesta está alineada con el modo pedagógico.
    
    CONCEPT: Debe tener definiciones y relaciones
    PRACTICE: Debe tener pasos procedimentales y resultado
    EXERCISE_LIST: NO debe tener explicaciones, solo lista
    """
    score = 1.0
    reasons: List[str] = []
    answer_lower = answer.lower()
    
    if mode == LearningMode.EXERCISE_LIST:
        # CRÍTICO: No debe haber explicaciones
        explanation_markers = [
            "porque", "debido a", "por lo tanto", "esto significa",
            "el primer paso", "para resolver", "la solución es",
            "se calcula", "aplicamos", "utilizamos"
        ]
        
        explanations_found = sum(1 for m in explanation_markers if m in answer_lower)
        
        if explanations_found > 2:
            score -= 0.5
            reasons.append(f"Contiene {explanations_found} explicaciones (prohibido)")
        
        # Debe tener estructura de lista
        list_markers = ["ejercicio", "dificultad", "📝", "ref:", "concepto:"]
        list_found = sum(1 for m in list_markers if m in answer_lower)
        
        if list_found < 2:
            score -= 0.3
            reasons.append("No tiene formato de lista de ejercicios")
        
        # Verificar que no sea muy larga (indica explicación)
        if len(answer) > 1500:
            score -= 0.2
            reasons.append("Respuesta muy larga para lista")
            
    elif mode == LearningMode.PRACTICE:
        # Debe tener pasos
        step_markers = ["paso", "1.", "2.", "primero", "luego", "después", "finalmente"]
        steps_found = sum(1 for m in step_markers if m in answer_lower)
        
        if steps_found < 2:
            score -= 0.4
            reasons.append("Faltan pasos procedimentales")
        
        # Debe tener resultado
        result_markers = ["resultado", "respuesta", "solución", "=", "obtenemos"]
        result_found = any(m in answer_lower for m in result_markers)
        
        if not result_found:
            score -= 0.2
            reasons.append("Falta resultado final")
            
    else:  # CONCEPT
        # Debe tener definiciones o explicaciones conceptuales
        concept_markers = ["es", "se define", "significa", "consiste en", "se refiere"]
        concepts_found = sum(1 for m in concept_markers if m in answer_lower)
        
        if concepts_found < 1:
            score -= 0.2
            reasons.append("Falta definición o explicación conceptual")
        
        # Bonus si menciona relaciones
        relation_markers = ["relaciona", "conecta", "depende", "implica", "causa"]
        if any(m in answer_lower for m in relation_markers):
            score = min(1.0, score + 0.1)
            reasons.append("Incluye relaciones")
    
    score = max(0.0, min(1.0, score))
    
    return DimensionScore(
        dimension=EvaluationDimension.MODE_ALIGNMENT,
        score=score,
        reason="; ".join(reasons) if reasons else f"Alineado con modo {mode.value}",
    )


def _determine_decision(
    overall_score: float,
    dimension_scores: List[DimensionScore],
    issues: List[str],
    previous_evaluation: Optional[EvaluationResult],
    mode: LearningMode = LearningMode.CONCEPT,
) -> ReflectionDecision:
    """Determina la decisión basada en la evaluación y modo."""
    
    # Verificar MODE_ALIGNMENT primero
    mode_alignment_score = next(
        (ds.score for ds in dimension_scores 
         if ds.dimension == EvaluationDimension.MODE_ALIGNMENT),
        1.0
    )
    
    # Si MODE_ALIGNMENT es muy bajo, es MODE_MISMATCH
    if mode_alignment_score < 0.4:
        return ReflectionDecision.MODE_MISMATCH
    
    # Si ya hubo evaluación previa y no mejoró, usar fallback
    if previous_evaluation:
        if overall_score <= previous_evaluation.overall_score:
            return ReflectionDecision.FALLBACK
    
    # Score muy bajo
    if overall_score < 0.3:
        return ReflectionDecision.RETRY
    
    # Score bajo pero aceptable
    if overall_score < 0.5:
        # Verificar dimensiones específicas
        low_dims = [
            ds for ds in dimension_scores 
            if ds.score < 0.4
        ]
        
        # Priorizar MODE_MISMATCH si el problema es de alineación
        if any(ds.dimension == EvaluationDimension.MODE_ALIGNMENT for ds in low_dims):
            return ReflectionDecision.MODE_MISMATCH
        
        if any(ds.dimension == EvaluationDimension.COVERAGE for ds in low_dims):
            return ReflectionDecision.REQUEST_MORE_CONTEXT
        
        return ReflectionDecision.RETRY
    
    # Score medio
    if overall_score < 0.7:
        # Podría expandirse (pero no para EXERCISE_LIST)
        if mode != LearningMode.EXERCISE_LIST:
            completeness_score = next(
                (ds.score for ds in dimension_scores 
                 if ds.dimension == EvaluationDimension.COMPLETENESS),
                1.0
            )
            if completeness_score < 0.5:
                return ReflectionDecision.EXPAND
        
        return ReflectionDecision.ACCEPT
    
    # Score alto
    return ReflectionDecision.ACCEPT


# ==========================================
# Decisiones de Reintento
# ==========================================

async def decide_retry(
    evaluation: EvaluationResult,
    retry_count: int = 0,
    config: Optional[RetryConfig] = None,
) -> Tuple[bool, str]:
    """
    Decide si reintentar la generación.
    
    Args:
        evaluation: Resultado de evaluación
        retry_count: Número de reintentos ya realizados
        config: Configuración de reintentos
        
    Returns:
        Tupla (should_retry, reason)
    """
    config = config or RetryConfig()
    
    # Límite de reintentos
    if retry_count >= config.max_retries:
        return False, f"Máximo de reintentos alcanzado ({config.max_retries})"
    
    # Decidir según la decisión de evaluación
    if evaluation.decision == ReflectionDecision.ACCEPT:
        return False, "Respuesta aceptada"
    
    if evaluation.decision == ReflectionDecision.FALLBACK:
        return False, "Usando fallback"
    
    # MODE_MISMATCH requiere re-routing, no solo retry
    if evaluation.decision == ReflectionDecision.MODE_MISMATCH:
        if evaluation.mode_alignment_score < config.mode_alignment_threshold:
            return True, f"Desalineación de modo ({evaluation.learning_mode.value}): requiere re-routing"
        return True, "Mode mismatch detectado"
    
    if evaluation.decision in [
        ReflectionDecision.RETRY, 
        ReflectionDecision.EXPAND,
        ReflectionDecision.SIMPLIFY,
    ]:
        if evaluation.overall_score < config.min_score_threshold:
            return True, f"Score bajo ({evaluation.overall_score:.2f})"
        
        # Verificar si hay problemas específicos que se pueden mejorar
        if evaluation.issues:
            return True, f"Issues: {', '.join(evaluation.issues[:2])}"
    
    if evaluation.decision == ReflectionDecision.REQUEST_MORE_CONTEXT:
        return True, "Necesita más contexto"
    
    return False, "Sin necesidad de reintento"


def get_retry_adjustments(
    evaluation: EvaluationResult,
    current_config: PromptConfig,
    mode: LearningMode = LearningMode.CONCEPT,
) -> Tuple[PromptConfig, Dict[str, Any]]:
    """
    Obtiene ajustes para el reintento.
    
    Args:
        evaluation: Resultado de evaluación
        current_config: Configuración actual
        mode: Modo de aprendizaje actual
        
    Returns:
        Tupla (configuración ajustada, ajustes de retrieval)
    """
    new_config = PromptConfig(
        max_context_tokens=current_config.max_context_tokens,
        max_conversation_tokens=current_config.max_conversation_tokens,
        include_examples=current_config.include_examples,
        include_sources=current_config.include_sources,
        language=current_config.language,
        model=current_config.model,
    )
    
    retrieval_adjustments: Dict[str, Any] = {
        "change_strategy": False,
        "force_mode": None,
        "expand_graph": False,
    }
    
    # Obtener límite máximo del modelo para los ajustes
    base_context_limit = _get_default_context_limit(current_config.model)
    
    # Ajustar según decisión
    if evaluation.decision == ReflectionDecision.MODE_MISMATCH:
        # MODE_MISMATCH: forzar cambio de estrategia
        retrieval_adjustments["change_strategy"] = True
        retrieval_adjustments["force_mode"] = mode
        
        if mode == LearningMode.EXERCISE_LIST:
            # Para EXERCISE_LIST, forzar solo grafo y expandir búsqueda
            retrieval_adjustments["expand_graph"] = True
            
    elif evaluation.decision == ReflectionDecision.EXPAND:
        # Incrementar contexto en 25%, respetando el límite del modelo
        new_limit = int(current_config.max_context_tokens * 1.25)
        new_config.max_context_tokens = min(base_context_limit, new_limit)
        
    elif evaluation.decision == ReflectionDecision.SIMPLIFY:
        new_config.include_examples = True
        
    elif evaluation.decision == ReflectionDecision.REQUEST_MORE_CONTEXT:
        # Incrementar contexto en 50%, respetando el límite del modelo
        new_limit = int(current_config.max_context_tokens * 1.5)
        new_config.max_context_tokens = min(base_context_limit, new_limit)
        retrieval_adjustments["expand_graph"] = True
    
    # Si cobertura es baja, pedir más ejemplos
    coverage_score = next(
        (ds.score for ds in evaluation.dimension_scores 
         if ds.dimension == EvaluationDimension.COVERAGE),
        1.0
    )
    if coverage_score < 0.5:
        new_config.include_examples = True
    
    return new_config, retrieval_adjustments


def get_mode_mismatch_corrections(
    evaluation: EvaluationResult,
    mode: LearningMode,
) -> Dict[str, Any]:
    """
    Obtiene correcciones específicas para MODE_MISMATCH.
    
    Args:
        evaluation: Resultado de evaluación
        mode: Modo de aprendizaje objetivo
        
    Returns:
        Diccionario con correcciones a aplicar
    """
    corrections: Dict[str, Any] = {
        "reroute": True,
        "retrieval_changes": {},
        "prompt_changes": {},
    }
    
    if mode == LearningMode.EXERCISE_LIST:
        # Forzar retrieval solo de grafo para ejercicios
        corrections["retrieval_changes"] = {
            "mode": "graph_only",
            "vector_weight": 0.0,
            "bm25_weight": 0.0,
            "graph_weight": 1.0,
            "chunk_type_filter": "exercise",
            "metadata_only": True,
        }
        corrections["prompt_changes"] = {
            "instruction": "SOLO listar ejercicios. NO explicar.",
        }
        
    elif mode == LearningMode.PRACTICE:
        # Ajustar para ejemplos resueltos
        corrections["retrieval_changes"] = {
            "chunk_type_filter": "worked_example",
            "expand_concepts": True,
        }
        corrections["prompt_changes"] = {
            "instruction": "Explicar paso a paso. Incluir resultado final.",
        }
        
    else:  # CONCEPT
        # Priorizar definiciones y relaciones
        corrections["retrieval_changes"] = {
            "graph_weight": 0.5,
            "expand_concepts": True,
        }
        corrections["prompt_changes"] = {
            "instruction": "Explicar concepto con definiciones y relaciones.",
        }
    
    return corrections


# ==========================================
# Solicitud de Más Contexto
# ==========================================

async def request_more_context(
    query: str,
    current_chunks: List[RankedResult],
    evaluation: EvaluationResult,
) -> Dict[str, Any]:
    """
    Genera solicitud de más contexto.
    
    Args:
        query: Pregunta original
        current_chunks: Chunks actuales
        evaluation: Evaluación que indica necesidad de más contexto
        
    Returns:
        Especificación de contexto adicional necesario
    """
    request: Dict[str, Any] = {
        "action": "expand_search",
        "reason": "Cobertura insuficiente",
        "suggestions": [],
    }
    
    # Identificar qué falta
    covered_concepts: set = set()
    for chunk in current_chunks:
        covered_concepts.update(chunk.concepts)
    
    # Extraer conceptos de la pregunta que no están cubiertos
    query_words = set(query.lower().split())
    important_words = query_words - {"qué", "cómo", "por", "qué", "es", "son", "el", "la", "los", "las"}
    
    uncovered = important_words - {c.lower() for c in covered_concepts}
    
    if uncovered:
        request["suggestions"].append({
            "type": "search_terms",
            "terms": list(uncovered)[:5],
        })
    
    # Sugerir expandir búsqueda
    request["suggestions"].append({
        "type": "expand_strategy",
        "strategy": "graph",  # Usar grafo para encontrar conceptos relacionados
    })
    
    # Indicar si necesita fuentes adicionales
    current_sources = set(c.source_id for c in current_chunks if c.source_id)
    if len(current_sources) < 2:
        request["suggestions"].append({
            "type": "more_sources",
            "reason": "Solo hay una fuente representada",
        })
    
    return request


async def generate_fallback_response(
    query: str,
    context_chunks: List[RankedResult],
    evaluation: EvaluationResult,
) -> str:
    """
    Genera respuesta de fallback cuando los reintentos fallan.
    
    Args:
        query: Pregunta original
        context_chunks: Contexto disponible
        evaluation: Última evaluación
        
    Returns:
        Respuesta de fallback
    """
    # Crear respuesta honesta sobre limitaciones
    response_parts: List[str] = []
    
    if not context_chunks:
        response_parts.append(
            "**No encontré información relevante en los documentos del sistema.**\n\n"
            "Para poder ayudarte con esta pregunta, necesitaría que subas documentos "
            "relacionados con el tema. Puedes hacerlo usando el botón de adjuntar archivos."
        )
    else:
        response_parts.append(
            "**Información parcial encontrada en los documentos:**\n"
        )
        
        # Incluir información de los chunks disponibles
        for i, chunk in enumerate(context_chunks[:3], 1):
            # Verificar que el chunk tenga contenido
            if not chunk.content:
                continue
            # Extracto del chunk
            content = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
            response_parts.append(f"[{i}] {content}\n")
        
        response_parts.append(
            "\n**Nota:** Los documentos disponibles no contienen información suficiente "
            "para responder completamente tu pregunta. Te sugiero:\n"
            "- Subir más documentos relacionados con el tema\n"
            "- Reformular la pregunta con términos más específicos"
        )
    
    return "\n".join(response_parts)


# ==========================================
# Análisis de Patrones
# ==========================================

async def analyze_failure_patterns(
    evaluations: List[EvaluationResult],
) -> Dict[str, Any]:
    """
    Analiza patrones de fallo para mejorar el sistema.
    
    Args:
        evaluations: Lista de evaluaciones recientes
        
    Returns:
        Análisis de patrones
    """
    if not evaluations:
        return {"patterns": [], "recommendations": []}
    
    # Contar problemas por dimensión
    dimension_issues: Dict[str, int] = {}
    total_issues: List[str] = []
    
    for ev in evaluations:
        for ds in ev.dimension_scores:
            if ds.score < 0.5:
                dim_name = ds.dimension.value
                dimension_issues[dim_name] = dimension_issues.get(dim_name, 0) + 1
        
        total_issues.extend(ev.issues)
    
    # Identificar patrones
    patterns: List[str] = []
    recommendations: List[str] = []
    
    # Problema más común
    if dimension_issues:
        worst_dim = max(dimension_issues.items(), key=lambda x: x[1])
        patterns.append(f"Dimensión más problemática: {worst_dim[0]} ({worst_dim[1]} ocurrencias)")
        
        if worst_dim[0] == EvaluationDimension.COVERAGE.value:
            recommendations.append("Mejorar estrategia de retrieval")
        elif worst_dim[0] == EvaluationDimension.RELEVANCE.value:
            recommendations.append("Ajustar prompts del sistema")
        elif worst_dim[0] == EvaluationDimension.COHERENCE.value:
            recommendations.append("Considerar modelo LLM diferente")
    
    # Score promedio
    avg_score = sum(ev.overall_score for ev in evaluations) / len(evaluations)
    patterns.append(f"Score promedio: {avg_score:.2f}")
    
    if avg_score < 0.5:
        recommendations.append("Revisar pipeline completo de generación")
    
    return {
        "patterns": patterns,
        "recommendations": recommendations,
        "dimension_issues": dimension_issues,
        "average_score": avg_score,
    }
