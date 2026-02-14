"""
reflection_agent.py
Agente de Reflexión SUPERVISOR - Evalúa y decide, NO ejecuta.

Este agente es el SUPERVISOR del pipeline:
- Evalúa alineación con el modo pedagógico
- Evalúa calidad de la respuesta
- DECIDE la siguiente acción
- NUNCA ejecuta retrieval
- NUNCA modifica prompts directamente

Devuelve: ReflectionDecision + AdjustmentHints
El Orchestrator aplica las decisiones.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.mode_router import LearningMode
from backend.agents.retrieval_agent import RetrievalResultItem
from backend.agents.reasoning_agent import GeneratedAnswer

logger = logging.getLogger(__name__)


# ==========================================
# Enums de Decisión
# ==========================================

class ReflectionDecision(str, Enum):
    """
    Decisiones que puede tomar el Reflection.
    
    El Orchestrator actúa según esta decisión.
    Reflection NO ejecuta.
    """
    ACCEPT = "accept"                      # Respuesta aceptable, terminar
    RETRY = "retry"                        # Regenerar con mismos datos
    MODE_MISMATCH = "mode_mismatch"        # Cambiar modo y reintentar
    REQUEST_MORE_CONTEXT = "request_more_context"  # Expandir retrieval
    FALLBACK = "fallback"                  # Usar mensaje de fallback


class EvaluationDimension(str, Enum):
    """Dimensiones de evaluación."""
    MODE_ALIGNMENT = "mode_alignment"      # ¿Cumple el modo?
    CONTENT_QUALITY = "content_quality"    # ¿Contenido adecuado?
    CONTEXT_COVERAGE = "context_coverage"  # ¿Usa el contexto?
    FORMAT_COMPLIANCE = "format_compliance"  # ¿Formato correcto?
    PROHIBITION_CHECK = "prohibition_check"  # ¿Viola prohibiciones?


# ==========================================
# Data Classes
# ==========================================

@dataclass
class EvaluationScore:
    """Score de una dimensión de evaluación."""
    dimension: EvaluationDimension
    score: float  # 0.0 - 1.0
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjustmentHints:
    """
    Sugerencias de ajuste para el Orchestrator.
    
    Reflection SUGIERE, Orchestrator APLICA.
    """
    expand_context: bool = False
    change_strategy: bool = False
    suggested_mode: Optional[LearningMode] = None
    increase_top_k: bool = False
    decrease_temperature: bool = False
    add_prohibitions: List[str] = field(default_factory=list)
    retry_focus: str = ""  # Qué enfocar en el retry
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expand_context": self.expand_context,
            "change_strategy": self.change_strategy,
            "suggested_mode": self.suggested_mode.value if self.suggested_mode else None,
            "increase_top_k": self.increase_top_k,
            "decrease_temperature": self.decrease_temperature,
            "add_prohibitions": self.add_prohibitions,
            "retry_focus": self.retry_focus,
            "reason": self.reason,
        }


@dataclass
class RetryConfig:
    """Configuración para reintentos."""
    max_retries: int = 3
    current_retry: int = 0
    previous_decisions: List[ReflectionDecision] = field(default_factory=list)
    
    @property
    def can_retry(self) -> bool:
        return self.current_retry < self.max_retries
    
    @property
    def should_fallback(self) -> bool:
        return self.current_retry >= self.max_retries


@dataclass
class EvaluationResult:
    """
    Resultado completo de la evaluación.
    
    Contiene:
    - Decisión (qué hacer)
    - Hints (cómo ajustar)
    - Scores (por qué)
    """
    decision: ReflectionDecision
    hints: AdjustmentHints
    scores: List[EvaluationScore] = field(default_factory=list)
    overall_score: float = 0.0
    mode_aligned: bool = True
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "overall_score": self.overall_score,
            "mode_aligned": self.mode_aligned,
            "reason": self.reason,
            "hints": self.hints.to_dict(),
            "scores": [
                {
                    "dimension": s.dimension.value,
                    "score": s.score,
                    "passed": s.passed,
                    "reason": s.reason,
                }
                for s in self.scores
            ],
        }


# ==========================================
# Función Principal de Evaluación
# ==========================================

async def evaluate_answer(
    query: str,
    answer: GeneratedAnswer,
    context_chunks: List[RetrievalResultItem],
    mode: Optional[LearningMode] = None,
    retry_config: Optional[RetryConfig] = None,
    previous_evaluation: Optional[EvaluationResult] = None,
) -> EvaluationResult:
    """
    Evalúa una respuesta y decide la siguiente acción.
    
    Este agente es SUPERVISOR:
    - Evalúa alineación con modo
    - Evalúa calidad
    - DECIDE acción
    - NO ejecuta nada
    
    Args:
        query: Pregunta original
        answer: Respuesta generada
        context_chunks: Chunks usados
        mode: Modo pedagógico activo
        retry_config: Configuración de reintentos
        previous_evaluation: Evaluación previa (si es retry)
        
    Returns:
        EvaluationResult con decisión y hints
    """
    mode = mode or answer.mode_used
    retry_config = retry_config or RetryConfig()
    
    logger.info(f"Evaluating answer: mode={mode.value}, retry={retry_config.current_retry}")
    
    scores: List[EvaluationScore] = []
    hints = AdjustmentHints()
    
    # 1. Evaluar alineación con modo (MÁS IMPORTANTE)
    mode_score = await _evaluate_mode_alignment(answer.answer, mode)
    scores.append(mode_score)
    
    # 2. Evaluar calidad del contenido
    content_score = await _evaluate_content_quality(
        answer.answer, 
        context_chunks, 
        query
    )
    scores.append(content_score)
    
    # 3. Evaluar uso del contexto
    context_score = await _evaluate_context_coverage(
        answer.answer, 
        context_chunks
    )
    scores.append(context_score)
    
    # 4. Evaluar formato
    format_score = await _evaluate_format_compliance(answer.answer, mode)
    scores.append(format_score)
    
    # 5. Verificar prohibiciones (especialmente para EXERCISE_LIST)
    prohibition_score = await _evaluate_prohibitions(answer.answer, mode)
    scores.append(prohibition_score)
    
    # Calcular score general
    weights = {
        EvaluationDimension.MODE_ALIGNMENT: 0.35,
        EvaluationDimension.PROHIBITION_CHECK: 0.25,
        EvaluationDimension.CONTENT_QUALITY: 0.20,
        EvaluationDimension.CONTEXT_COVERAGE: 0.10,
        EvaluationDimension.FORMAT_COMPLIANCE: 0.10,
    }
    
    overall_score = sum(
        s.score * weights.get(s.dimension, 0.1)
        for s in scores
    )
    
    # Determinar decisión
    decision, hints, reason = _determine_decision(
        scores=scores,
        overall_score=overall_score,
        mode=mode,
        retry_config=retry_config,
        previous_evaluation=previous_evaluation,
    )
    
    # Verificar si modo está alineado
    mode_aligned = mode_score.passed and prohibition_score.passed
    
    return EvaluationResult(
        decision=decision,
        hints=hints,
        scores=scores,
        overall_score=overall_score,
        mode_aligned=mode_aligned,
        reason=reason,
        metadata={
            "query_length": len(query.split()),
            "answer_length": len(answer.answer.split()),
            "context_count": len(context_chunks),
            "retry_count": retry_config.current_retry,
        },
    )


# ==========================================
# Funciones de Evaluación por Dimensión
# ==========================================

async def _evaluate_mode_alignment(answer: str, mode: LearningMode) -> EvaluationScore:
    """
    Evalúa si la respuesta se alinea con el modo pedagógico.
    
    Esta es la evaluación MÁS CRÍTICA.
    """
    answer_lower = answer.lower()
    
    if mode == LearningMode.CONCEPT:
        # Debe tener explicaciones, definiciones
        has_definition = any(marker in answer_lower for marker in [
            "es ", "son ", "se define", "significa", "consiste en",
            "is ", "are ", "means", "refers to"
        ])
        has_explanation = len(answer.split()) > 50  # Mínimo contenido
        
        score = 0.0
        if has_definition:
            score += 0.5
        if has_explanation:
            score += 0.5
            
        return EvaluationScore(
            dimension=EvaluationDimension.MODE_ALIGNMENT,
            score=score,
            passed=score >= 0.5,
            reason="CONCEPT requires definitions and explanations",
            details={"has_definition": has_definition, "has_explanation": has_explanation},
        )
    
    elif mode == LearningMode.PRACTICE:
        # Debe tener pasos numerados
        has_steps = bool(re.search(r'(paso|step|\d+[\.\)])', answer_lower))
        has_result = any(marker in answer_lower for marker in [
            "resultado", "respuesta", "solución", "result", "answer", "solution"
        ])
        
        score = 0.0
        if has_steps:
            score += 0.6
        if has_result:
            score += 0.4
            
        return EvaluationScore(
            dimension=EvaluationDimension.MODE_ALIGNMENT,
            score=score,
            passed=score >= 0.5,
            reason="PRACTICE requires steps and result",
            details={"has_steps": has_steps, "has_result": has_result},
        )
    
    elif mode == LearningMode.EXERCISE_LIST:
        # SOLO debe listar, NO explicar
        is_list = bool(re.search(r'(\d+[\.\)]\s|\-\s|•)', answer))
        has_exercises = any(marker in answer_lower for marker in [
            "ejercicio", "problema", "exercise", "problem"
        ])
        
        # Penalizar si explica demasiado
        word_count = len(answer.split())
        is_concise = word_count < 300  # Debe ser breve
        
        score = 0.0
        if is_list:
            score += 0.4
        if has_exercises:
            score += 0.3
        if is_concise:
            score += 0.3
        else:
            score -= 0.3  # Penalización por ser muy largo
            
        score = max(0.0, min(1.0, score))
        
        return EvaluationScore(
            dimension=EvaluationDimension.MODE_ALIGNMENT,
            score=score,
            passed=score >= 0.5 and is_concise,
            reason="EXERCISE_LIST requires concise listing only",
            details={"is_list": is_list, "has_exercises": has_exercises, "is_concise": is_concise},
        )
    
    # Default
    return EvaluationScore(
        dimension=EvaluationDimension.MODE_ALIGNMENT,
        score=0.5,
        passed=True,
        reason="Default mode alignment",
    )


async def _evaluate_content_quality(
    answer: str,
    context_chunks: List[RetrievalResultItem],
    query: str,
) -> EvaluationScore:
    """Evalúa calidad general del contenido."""
    
    # Longitud razonable
    word_count = len(answer.split())
    length_score = min(1.0, word_count / 50)  # Al menos 50 palabras
    
    # Relevancia básica (¿menciona términos del query?)
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    overlap = len(query_terms & answer_terms) / max(len(query_terms), 1)
    
    # Score combinado
    score = (length_score * 0.4) + (overlap * 0.6)
    
    return EvaluationScore(
        dimension=EvaluationDimension.CONTENT_QUALITY,
        score=score,
        passed=score >= 0.4,
        reason=f"Content quality based on length ({word_count} words) and relevance",
        details={"word_count": word_count, "query_overlap": overlap},
    )


async def _evaluate_context_coverage(
    answer: str,
    context_chunks: List[RetrievalResultItem],
) -> EvaluationScore:
    """Evalúa si la respuesta usa el contexto proporcionado."""
    
    if not context_chunks:
        return EvaluationScore(
            dimension=EvaluationDimension.CONTEXT_COVERAGE,
            score=0.5,
            passed=True,
            reason="No context to evaluate",
        )
    
    answer_lower = answer.lower()
    
    # Contar conceptos del contexto mencionados
    context_concepts = set()
    for chunk in context_chunks:
        context_concepts.update(c.lower() for c in chunk.concepts)
    
    mentioned_concepts = sum(1 for c in context_concepts if c in answer_lower)
    concept_coverage = mentioned_concepts / max(len(context_concepts), 1)
    
    # Verificar si usa contenido de los chunks
    content_used = 0
    for chunk in context_chunks[:5]:
        # Verificar overlap de palabras significativas
        chunk_words = set(chunk.content.lower().split())
        answer_words = set(answer_lower.split())
        if len(chunk_words & answer_words) > 3:
            content_used += 1
    
    content_coverage = content_used / min(len(context_chunks), 5)
    
    score = (concept_coverage * 0.5) + (content_coverage * 0.5)
    
    return EvaluationScore(
        dimension=EvaluationDimension.CONTEXT_COVERAGE,
        score=score,
        passed=score >= 0.3,
        reason=f"Context coverage: {mentioned_concepts} concepts, {content_used} chunks used",
        details={"concepts_mentioned": mentioned_concepts, "chunks_used": content_used},
    )


async def _evaluate_format_compliance(answer: str, mode: LearningMode) -> EvaluationScore:
    """Evalúa si el formato es apropiado para el modo."""
    
    has_structure = any([
        bool(re.search(r'\d+[\.\)]', answer)),  # Numeración
        bool(re.search(r'[\-\*•]', answer)),     # Bullets
        '\n\n' in answer,                         # Párrafos
    ])
    
    # Para EXERCISE_LIST, verificar formato de lista
    if mode == LearningMode.EXERCISE_LIST:
        is_proper_list = bool(re.search(r'(\d+[\.\)]\s.+\n?){2,}', answer))
        score = 0.8 if is_proper_list else 0.4
    elif mode == LearningMode.PRACTICE:
        has_numbered_steps = bool(re.search(r'(\d+[\.\)]\s.+\n?){2,}', answer))
        score = 0.9 if has_numbered_steps else 0.5
    else:
        score = 0.7 if has_structure else 0.5
    
    return EvaluationScore(
        dimension=EvaluationDimension.FORMAT_COMPLIANCE,
        score=score,
        passed=score >= 0.4,
        reason=f"Format compliance for {mode.value}",
        details={"has_structure": has_structure},
    )


async def _evaluate_prohibitions(answer: str, mode: LearningMode) -> EvaluationScore:
    """
    Evalúa si la respuesta viola prohibiciones del modo.
    
    CRÍTICO para EXERCISE_LIST: NO debe explicar.
    """
    answer_lower = answer.lower()
    violations = []
    
    if mode == LearningMode.EXERCISE_LIST:
        # Prohibiciones para EXERCISE_LIST
        explanation_markers = [
            "para resolver", "se resuelve", "la solución es",
            "primero", "segundo", "paso", "entonces",
            "porque", "debido a", "ya que",
            "to solve", "the solution", "first", "then", "because",
            "procedimiento", "método", "fórmula",
        ]
        
        for marker in explanation_markers:
            if marker in answer_lower:
                violations.append(f"Found explanation marker: '{marker}'")
        
        # Verificar longitud (no debe ser muy largo)
        if len(answer.split()) > 200:
            violations.append("Answer too long for exercise list")
    
    # Score basado en violaciones
    if violations:
        score = max(0.0, 1.0 - (len(violations) * 0.2))
        passed = len(violations) < 3
    else:
        score = 1.0
        passed = True
    
    return EvaluationScore(
        dimension=EvaluationDimension.PROHIBITION_CHECK,
        score=score,
        passed=passed,
        reason=f"Prohibition check: {len(violations)} violations" if violations else "No violations",
        details={"violations": violations[:5]},
    )


# ==========================================
# Determinación de Decisión
# ==========================================

def _determine_decision(
    scores: List[EvaluationScore],
    overall_score: float,
    mode: LearningMode,
    retry_config: RetryConfig,
    previous_evaluation: Optional[EvaluationResult],
) -> Tuple[ReflectionDecision, AdjustmentHints, str]:
    """
    Determina la decisión final basada en scores.
    
    LÓGICA DE DECISIÓN:
    1. Si viola prohibiciones (EXERCISE_LIST) → MODE_MISMATCH o RETRY
    2. Si modo no alineado → MODE_MISMATCH
    3. Si contexto insuficiente → REQUEST_MORE_CONTEXT
    4. Si score bajo pero retriable → RETRY
    5. Si max retries → FALLBACK
    6. Si score aceptable → ACCEPT
    """
    hints = AdjustmentHints()
    
    # Extraer scores por dimensión
    score_by_dim = {s.dimension: s for s in scores}
    mode_score = score_by_dim.get(EvaluationDimension.MODE_ALIGNMENT)
    prohibition_score = score_by_dim.get(EvaluationDimension.PROHIBITION_CHECK)
    context_score = score_by_dim.get(EvaluationDimension.CONTEXT_COVERAGE)
    
    # 1. Verificar prohibiciones (CRÍTICO para EXERCISE_LIST)
    if prohibition_score and not prohibition_score.passed:
        if mode == LearningMode.EXERCISE_LIST:
            # Para EXERCISE_LIST, las violaciones son graves
            hints.add_prohibitions = [
                "NO explicar procedimientos",
                "NO dar soluciones",
                "SOLO listar ejercicios",
            ]
            hints.decrease_temperature = True
            hints.reason = "EXERCISE_LIST violó prohibición de no explicar"
            
            if retry_config.can_retry:
                return (ReflectionDecision.RETRY, hints, "Violación de prohibiciones en EXERCISE_LIST")
            else:
                return (ReflectionDecision.FALLBACK, hints, "Max retries alcanzado con violaciones")
    
    # 2. Verificar alineación de modo
    if mode_score and not mode_score.passed:
        # El contenido no coincide con el modo solicitado
        hints.reason = f"Respuesta no alineada con modo {mode.value}"
        
        # Sugerir modo correcto si podemos inferirlo
        if "paso" in str(mode_score.details).lower() or "step" in str(mode_score.details).lower():
            hints.suggested_mode = LearningMode.PRACTICE
        elif "list" in str(mode_score.details).lower() or "ejercicio" in str(mode_score.details).lower():
            hints.suggested_mode = LearningMode.EXERCISE_LIST
        
        if retry_config.can_retry:
            return (ReflectionDecision.MODE_MISMATCH, hints, f"Modo esperado: {mode.value}")
        else:
            return (ReflectionDecision.FALLBACK, hints, "Max retries con mode mismatch")
    
    # 3. Verificar contexto
    if context_score and context_score.score < 0.2:
        hints.expand_context = True
        hints.increase_top_k = True
        hints.reason = "Contexto insuficiente utilizado"
        
        if retry_config.can_retry:
            return (ReflectionDecision.REQUEST_MORE_CONTEXT, hints, "Necesita más contexto")
    
    # 4. Verificar score general
    if overall_score >= 0.6:
        return (ReflectionDecision.ACCEPT, hints, f"Score aceptable: {overall_score:.2f}")
    
    # 5. Score bajo pero retriable
    if overall_score < 0.6 and retry_config.can_retry:
        hints.decrease_temperature = True
        hints.retry_focus = "Mejorar alineación con el modo y calidad"
        hints.reason = f"Score bajo: {overall_score:.2f}"
        return (ReflectionDecision.RETRY, hints, f"Score bajo: {overall_score:.2f}")
    
    # 6. Max retries alcanzado
    if retry_config.should_fallback:
        return (ReflectionDecision.FALLBACK, hints, "Máximo de reintentos alcanzado")
    
    # 7. Default: Aceptar si no hay problemas graves
    return (ReflectionDecision.ACCEPT, hints, "Aceptado por defecto")


# ==========================================
# Funciones Auxiliares
# ==========================================

def should_retry(
    evaluation: EvaluationResult,
    retry_config: RetryConfig,
) -> bool:
    """
    Determina si se debe reintentar.
    
    Útil para el Orchestrator.
    """
    if not retry_config.can_retry:
        return False
    
    return evaluation.decision in [
        ReflectionDecision.RETRY,
        ReflectionDecision.MODE_MISMATCH,
        ReflectionDecision.REQUEST_MORE_CONTEXT,
    ]


def get_fallback_message(mode: LearningMode, language: str = "es") -> str:
    """Obtiene mensaje de fallback según el modo."""
    messages = {
        "es": {
            LearningMode.CONCEPT: "No pude encontrar suficiente información sobre este concepto. ¿Podrías reformular tu pregunta?",
            LearningMode.PRACTICE: "No encontré ejemplos resueltos para este ejercicio. ¿Podrías especificar qué tipo de problema quieres resolver?",
            LearningMode.EXERCISE_LIST: "No encontré ejercicios sobre este tema en los documentos disponibles.",
        },
        "en": {
            LearningMode.CONCEPT: "I couldn't find enough information about this concept. Could you rephrase your question?",
            LearningMode.PRACTICE: "I couldn't find worked examples for this exercise. Could you specify what kind of problem you want to solve?",
            LearningMode.EXERCISE_LIST: "I couldn't find exercises on this topic in the available documents.",
        },
    }
    
    lang_messages = messages.get(language, messages["es"])
    return lang_messages.get(mode, lang_messages[LearningMode.CONCEPT])


# ==========================================
# Exports
# ==========================================

__all__ = [
    "ReflectionDecision",
    "EvaluationDimension",
    "EvaluationScore",
    "AdjustmentHints",
    "RetryConfig",
    "EvaluationResult",
    "evaluate_answer",
    "should_retry",
    "get_fallback_message",
]
