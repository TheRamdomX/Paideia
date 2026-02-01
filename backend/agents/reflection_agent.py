"""
reflection_agent.py
Agente de Reflexión - Evalúa calidad de respuestas y decide acciones correctivas.
Puede solicitar reintento, más contexto, o aplicar fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.reasoning_agent import GeneratedAnswer, PromptConfig
from backend.retrieval.hybrid_ranker import RankedResult
from backend.utils.text import token_count


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


class ReflectionDecision(str, Enum):
    """Decisiones del agente de reflexión."""
    ACCEPT = "accept"                    # Aceptar respuesta
    RETRY = "retry"                      # Reintentar generación
    REQUEST_MORE_CONTEXT = "request_more_context"  # Necesita más contexto
    SIMPLIFY = "simplify"                # Simplificar respuesta
    EXPAND = "expand"                    # Expandir respuesta
    FALLBACK = "fallback"                # Usar respuesta de fallback


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "dimension_scores": [d.to_dict() for d in self.dimension_scores],
            "issues": self.issues,
            "suggestions": self.suggestions,
            "decision": self.decision.value,
            "confidence": self.confidence,
        }


@dataclass
class RetryConfig:
    """Configuración para reintentos."""
    max_retries: int = 2
    min_score_threshold: float = 0.5
    require_improvement: bool = True
    different_strategy: bool = True


# ==========================================
# Evaluación de Respuestas
# ==========================================

async def evaluate_answer(
    query: str,
    answer: GeneratedAnswer,
    context_chunks: List[RankedResult],
    previous_evaluation: Optional[EvaluationResult] = None,
) -> EvaluationResult:
    """
    Evalúa la calidad de una respuesta generada.
    
    Args:
        query: Pregunta original
        answer: Respuesta generada
        context_chunks: Contexto usado
        previous_evaluation: Evaluación anterior (si es reintento)
        
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
    completeness = await _evaluate_completeness(query, answer.answer)
    dimension_scores.append(completeness)
    if completeness.score < 0.5:
        issues.append("La respuesta parece incompleta")
        suggestions.append("Expandir para cubrir más aspectos")
    
    # Calcular score general (promedio ponderado)
    weights = {
        EvaluationDimension.RELEVANCE: 0.35,
        EvaluationDimension.COVERAGE: 0.25,
        EvaluationDimension.COHERENCE: 0.20,
        EvaluationDimension.COMPLETENESS: 0.20,
    }
    
    overall_score = sum(
        ds.score * weights.get(ds.dimension, 0.25)
        for ds in dimension_scores
    )
    
    # Determinar decisión
    decision = _determine_decision(
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        issues=issues,
        previous_evaluation=previous_evaluation,
    )
    
    return EvaluationResult(
        overall_score=overall_score,
        dimension_scores=dimension_scores,
        issues=issues,
        suggestions=suggestions,
        decision=decision,
        confidence=answer.confidence,
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
        # Buscar frases del chunk en la respuesta
        chunk_words = set(chunk.content.lower().split()[:20])  # Primeras 20 palabras
        answer_words = set(answer_lower.split())
        
        overlap = len(chunk_words & answer_words)
        if overlap >= 3:  # Al menos 3 palabras en común
            chunks_used += 1
    
    coverage_ratio = chunks_used / min(5, len(context_chunks))
    
    # Bonus si menciona conceptos
    concept_bonus = 0
    all_concepts = []
    for chunk in context_chunks:
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
) -> DimensionScore:
    """Evalúa completitud de la respuesta."""
    # Detectar tipo de pregunta
    query_lower = query.lower()
    
    # Preguntas de comparación necesitan más contenido
    if any(w in query_lower for w in ["comparar", "diferencia", "vs", "versus"]):
        min_length = 200
        expected_structure = ["por un lado", "por otro", "mientras que", "en cambio"]
    # Preguntas de explicación
    elif any(w in query_lower for w in ["explicar", "explica", "cómo", "por qué"]):
        min_length = 150
        expected_structure = ["porque", "debido", "por lo tanto", "esto significa"]
    # Preguntas de definición
    elif any(w in query_lower for w in ["qué es", "qué son", "definir", "define"]):
        min_length = 80
        expected_structure = ["es", "se refiere", "significa", "se define"]
    else:
        min_length = 100
        expected_structure = []
    
    score = 1.0
    reasons: List[str] = []
    
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


def _determine_decision(
    overall_score: float,
    dimension_scores: List[DimensionScore],
    issues: List[str],
    previous_evaluation: Optional[EvaluationResult],
) -> ReflectionDecision:
    """Determina la decisión basada en la evaluación."""
    
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
        
        if any(ds.dimension == EvaluationDimension.COVERAGE for ds in low_dims):
            return ReflectionDecision.REQUEST_MORE_CONTEXT
        
        return ReflectionDecision.RETRY
    
    # Score medio
    if overall_score < 0.7:
        # Podría expandirse
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
) -> PromptConfig:
    """
    Obtiene ajustes para el reintento.
    
    Args:
        evaluation: Resultado de evaluación
        current_config: Configuración actual
        
    Returns:
        Configuración ajustada
    """
    from backend.agents.reasoning_agent import ResponseStyle
    
    new_config = PromptConfig(
        max_context_tokens=current_config.max_context_tokens,
        max_conversation_tokens=current_config.max_conversation_tokens,
        include_examples=current_config.include_examples,
        include_sources=current_config.include_sources,
        language=current_config.language,
        style=current_config.style,
    )
    
    # Ajustar según decisión
    if evaluation.decision == ReflectionDecision.EXPAND:
        new_config.max_context_tokens = min(5000, current_config.max_context_tokens + 1000)
        new_config.style = ResponseStyle.DETAILED
        
    elif evaluation.decision == ReflectionDecision.SIMPLIFY:
        new_config.style = ResponseStyle.CONCISE
        new_config.include_examples = True
        
    elif evaluation.decision == ReflectionDecision.REQUEST_MORE_CONTEXT:
        new_config.max_context_tokens = min(6000, current_config.max_context_tokens + 2000)
    
    # Si cobertura es baja, pedir más ejemplos
    coverage_score = next(
        (ds.score for ds in evaluation.dimension_scores 
         if ds.dimension == EvaluationDimension.COVERAGE),
        1.0
    )
    if coverage_score < 0.5:
        new_config.include_examples = True
    
    return new_config


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
    
    response_parts.append(
        "Basándome en la información disponible, puedo ofrecerte lo siguiente:"
    )
    
    # Incluir información de los chunks disponibles
    if context_chunks:
        response_parts.append("\n**Información relevante encontrada:**\n")
        for i, chunk in enumerate(context_chunks[:3], 1):
            # Extracto del chunk
            content = chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content
            response_parts.append(f"{i}. {content}\n")
    
    # Indicar limitaciones
    if evaluation.issues:
        response_parts.append("\n**Nota:** ")
        response_parts.append(
            "La información disponible podría no cubrir completamente tu pregunta. "
            "Te sugiero reformular la pregunta o consultar fuentes adicionales."
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
