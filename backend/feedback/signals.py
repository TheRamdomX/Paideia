"""
signals.py
Normaliza feedback (explícito e implícito).
Procesa señales de usuario para mejorar el sistema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.settings import get_rag_config


# ==========================================
# Tipos de Señales
# ==========================================

class SignalType(str, Enum):
    """Tipos de señales de feedback."""
    # Explícitas
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"
    REPORT = "report"
    
    # Implícitas
    DWELL_TIME = "dwell_time"
    SCROLL_DEPTH = "scroll_depth"
    CLICK = "click"
    COPY = "copy"
    FOLLOWUP = "followup"
    REPHRASE = "rephrase"
    ABANDON = "abandon"


class SignalCategory(str, Enum):
    """Categorías de señales."""
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    BEHAVIORAL = "behavioral"


class SentimentPolarity(str, Enum):
    """Polaridad del sentimiento."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class FeedbackSignal:
    """Señal de feedback individual."""
    
    signal_id: str = ""
    signal_type: SignalType = SignalType.RATING
    category: SignalCategory = SignalCategory.EXPLICIT
    
    # Valor de la señal
    value: float = 0.0  # Normalizado 0-1
    raw_value: Any = None  # Valor original
    
    # Contexto
    query_id: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    concept_ids: List[str] = field(default_factory=list)
    session_id: str = ""
    student_id: str = ""
    
    # Metadata
    timestamp: Optional[datetime] = None
    polarity: SentimentPolarity = SentimentPolarity.NEUTRAL
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "signal_type": self.signal_type.value,
            "category": self.category.value,
            "value": self.value,
            "raw_value": self.raw_value,
            "query_id": self.query_id,
            "chunk_ids": self.chunk_ids,
            "concept_ids": self.concept_ids,
            "session_id": self.session_id,
            "student_id": self.student_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedFeedback:
    """Feedback normalizado y agregado."""
    
    query_id: str = ""
    
    # Scores normalizados
    overall_score: float = 0.5  # 0-1
    relevance_score: float = 0.5
    helpfulness_score: float = 0.5
    accuracy_score: float = 0.5
    
    # Señales componentes
    signals: List[FeedbackSignal] = field(default_factory=list)
    
    # Metadata
    signal_count: int = 0
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    polarity: SentimentPolarity = SentimentPolarity.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "overall_score": self.overall_score,
            "relevance_score": self.relevance_score,
            "helpfulness_score": self.helpfulness_score,
            "accuracy_score": self.accuracy_score,
            "signal_count": self.signal_count,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "confidence": self.confidence,
            "polarity": self.polarity.value,
        }


# ==========================================
# Parseo de Feedback
# ==========================================

def parse_feedback(raw_feedback: Dict[str, Any]) -> FeedbackSignal:
    """
    Convierte feedback crudo a señal normalizada.
    
    Args:
        raw_feedback: Datos crudos del feedback
        
    Returns:
        FeedbackSignal normalizada
    """
    signal_type_str = raw_feedback.get("type", "rating")
    
    try:
        signal_type = SignalType(signal_type_str)
    except ValueError:
        signal_type = SignalType.RATING
    
    # Determinar categoría
    category = _determine_category(signal_type)
    
    # Obtener y normalizar valor
    raw_value = raw_feedback.get("value")
    normalized_value = _normalize_value(signal_type, raw_value)
    
    # Determinar polaridad
    polarity = _determine_polarity(signal_type, normalized_value)
    
    # Calcular confianza basada en tipo de señal
    confidence = _calculate_signal_confidence(signal_type, raw_feedback)
    
    signal = FeedbackSignal(
        signal_id=raw_feedback.get("id", f"sig_{datetime.now().timestamp()}"),
        signal_type=signal_type,
        category=category,
        value=normalized_value,
        raw_value=raw_value,
        query_id=raw_feedback.get("query_id", ""),
        chunk_ids=raw_feedback.get("chunk_ids", []),
        concept_ids=raw_feedback.get("concept_ids", []),
        session_id=raw_feedback.get("session_id", ""),
        student_id=raw_feedback.get("student_id", ""),
        timestamp=datetime.now(timezone.utc),
        polarity=polarity,
        confidence=confidence,
        metadata=raw_feedback.get("metadata", {}),
    )
    
    return signal


def _determine_category(signal_type: SignalType) -> SignalCategory:
    """Determina categoría de una señal."""
    explicit_types = {
        SignalType.THUMBS_UP, SignalType.THUMBS_DOWN,
        SignalType.RATING, SignalType.COMMENT, SignalType.REPORT
    }
    
    implicit_types = {
        SignalType.DWELL_TIME, SignalType.SCROLL_DEPTH,
        SignalType.CLICK, SignalType.COPY
    }
    
    if signal_type in explicit_types:
        return SignalCategory.EXPLICIT
    elif signal_type in implicit_types:
        return SignalCategory.IMPLICIT
    else:
        return SignalCategory.BEHAVIORAL


def _normalize_value(signal_type: SignalType, raw_value: Any) -> float:
    """
    Normaliza valor según tipo de señal.
    
    Returns:
        Valor normalizado entre 0 y 1
    """
    # Tipos que no dependen del raw_value
    if signal_type == SignalType.THUMBS_UP:
        return 1.0
    
    elif signal_type == SignalType.THUMBS_DOWN:
        return 0.0
    
    elif signal_type == SignalType.REPORT:
        # Reportar es siempre negativo
        return 0.0
    
    elif signal_type == SignalType.FOLLOWUP:
        # Pregunta de seguimiento puede ser positiva o negativa
        # Depende del tipo de followup
        return 0.6  # Neutral-positivo
    
    elif signal_type == SignalType.REPHRASE:
        # Reformular indica que la respuesta no fue satisfactoria
        return 0.3
    
    elif signal_type == SignalType.ABANDON:
        # Abandonar indica insatisfacción
        return 0.1
    
    # Para otros tipos, verificar si hay valor
    if raw_value is None:
        return 0.5
    
    if signal_type == SignalType.RATING:
        # Asumir escala 1-5
        try:
            rating = float(raw_value)
            # Normalizar a 0-1
            return max(0.0, min(1.0, (rating - 1) / 4))
        except (ValueError, TypeError):
            return 0.5
    
    elif signal_type == SignalType.DWELL_TIME:
        # Normalizar tiempo de lectura (en segundos)
        # <5s = bajo interés, >60s = alto interés
        try:
            seconds = float(raw_value)
            if seconds < 5:
                return 0.2
            elif seconds < 15:
                return 0.4
            elif seconds < 30:
                return 0.6
            elif seconds < 60:
                return 0.8
            else:
                return 1.0
        except (ValueError, TypeError):
            return 0.5
    
    elif signal_type == SignalType.SCROLL_DEPTH:
        # Normalizar porcentaje de scroll
        try:
            depth = float(raw_value)
            return max(0.0, min(1.0, depth / 100))
        except (ValueError, TypeError):
            return 0.5
    
    elif signal_type == SignalType.CLICK:
        # Click es binario, pero puede tener peso
        return 0.7 if raw_value else 0.3
    
    elif signal_type == SignalType.COPY:
        # Copiar texto indica utilidad
        return 0.8 if raw_value else 0.2
    
    elif signal_type == SignalType.COMMENT:
        # Comentarios requieren análisis de sentimiento
        # Por ahora, neutral
        return 0.5
    
    return 0.5


def _determine_polarity(signal_type: SignalType, value: float) -> SentimentPolarity:
    """Determina polaridad del sentimiento."""
    if value >= 0.7:
        return SentimentPolarity.POSITIVE
    elif value <= 0.3:
        return SentimentPolarity.NEGATIVE
    else:
        return SentimentPolarity.NEUTRAL


def _calculate_signal_confidence(
    signal_type: SignalType,
    raw_feedback: Dict[str, Any],
) -> float:
    """Calcula confianza de la señal."""
    # Señales explícitas tienen mayor confianza
    base_confidence = {
        SignalType.THUMBS_UP: 0.95,
        SignalType.THUMBS_DOWN: 0.95,
        SignalType.RATING: 0.9,
        SignalType.COMMENT: 0.8,
        SignalType.REPORT: 0.99,
        SignalType.DWELL_TIME: 0.6,
        SignalType.SCROLL_DEPTH: 0.5,
        SignalType.CLICK: 0.5,
        SignalType.COPY: 0.7,
        SignalType.FOLLOWUP: 0.6,
        SignalType.REPHRASE: 0.7,
        SignalType.ABANDON: 0.8,
    }
    
    return base_confidence.get(signal_type, 0.5)


# ==========================================
# Ponderación de Feedback
# ==========================================

# Pesos por tipo de señal
_DEFAULT_WEIGHTS = {
    SignalType.THUMBS_UP: 1.0,
    SignalType.THUMBS_DOWN: 1.0,
    SignalType.RATING: 0.9,
    SignalType.COMMENT: 0.7,
    SignalType.REPORT: 1.2,
    SignalType.DWELL_TIME: 0.4,
    SignalType.SCROLL_DEPTH: 0.3,
    SignalType.CLICK: 0.3,
    SignalType.COPY: 0.5,
    SignalType.FOLLOWUP: 0.4,
    SignalType.REPHRASE: 0.6,
    SignalType.ABANDON: 0.7,
}


def weight_feedback(
    signals: List[FeedbackSignal],
    weights: Optional[Dict[SignalType, float]] = None,
) -> NormalizedFeedback:
    """
    Combina múltiples señales en score unificado ponderado.
    
    Args:
        signals: Lista de señales a combinar
        weights: Pesos personalizados por tipo
        
    Returns:
        NormalizedFeedback con scores agregados
    """
    if not signals:
        return NormalizedFeedback()
    
    effective_weights = weights or _DEFAULT_WEIGHTS
    
    # Separar señales por categoría
    explicit_signals = [s for s in signals if s.category == SignalCategory.EXPLICIT]
    implicit_signals = [s for s in signals if s.category != SignalCategory.EXPLICIT]
    
    # Calcular scores ponderados
    overall_score = _weighted_average(signals, effective_weights)
    
    # Relevance: basado principalmente en señales de engagement
    relevance_signals = [
        s for s in signals 
        if s.signal_type in {
            SignalType.DWELL_TIME, SignalType.SCROLL_DEPTH,
            SignalType.CLICK, SignalType.COPY
        }
    ]
    relevance_score = (
        _weighted_average(relevance_signals, effective_weights)
        if relevance_signals else overall_score
    )
    
    # Helpfulness: basado en feedback explícito
    helpfulness_score = (
        _weighted_average(explicit_signals, effective_weights)
        if explicit_signals else overall_score
    )
    
    # Accuracy: basado en ausencia de reportes y rephrases
    accuracy_signals = [
        s for s in signals
        if s.signal_type in {SignalType.REPORT, SignalType.REPHRASE}
    ]
    if accuracy_signals:
        # Invertir: más reportes = menor accuracy
        accuracy_score = 1.0 - _weighted_average(accuracy_signals, effective_weights)
    else:
        accuracy_score = overall_score
    
    # Determinar polaridad general
    polarity = _determine_overall_polarity(signals)
    
    # Calcular confianza agregada
    confidence = _aggregate_confidence(signals)
    
    # Obtener query_id del primer signal
    query_id = signals[0].query_id if signals else ""
    
    return NormalizedFeedback(
        query_id=query_id,
        overall_score=overall_score,
        relevance_score=relevance_score,
        helpfulness_score=helpfulness_score,
        accuracy_score=accuracy_score,
        signals=signals,
        signal_count=len(signals),
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        polarity=polarity,
    )


def _weighted_average(
    signals: List[FeedbackSignal],
    weights: Dict[SignalType, float],
) -> float:
    """Calcula promedio ponderado de señales."""
    if not signals:
        return 0.5
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for signal in signals:
        weight = weights.get(signal.signal_type, 0.5) * signal.confidence
        weighted_sum += signal.value * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.5
    
    return weighted_sum / total_weight


def _determine_overall_polarity(signals: List[FeedbackSignal]) -> SentimentPolarity:
    """Determina polaridad general de múltiples señales."""
    if not signals:
        return SentimentPolarity.NEUTRAL
    
    positive_count = sum(1 for s in signals if s.polarity == SentimentPolarity.POSITIVE)
    negative_count = sum(1 for s in signals if s.polarity == SentimentPolarity.NEGATIVE)
    
    if positive_count > negative_count * 2:
        return SentimentPolarity.POSITIVE
    elif negative_count > positive_count * 2:
        return SentimentPolarity.NEGATIVE
    else:
        return SentimentPolarity.NEUTRAL


def _aggregate_confidence(signals: List[FeedbackSignal]) -> float:
    """Calcula confianza agregada."""
    if not signals:
        return 0.0
    
    # Más señales = mayor confianza, hasta cierto punto
    signal_boost = min(1.0, len(signals) / 5)
    
    # Promedio de confianzas individuales
    avg_confidence = sum(s.confidence for s in signals) / len(signals)
    
    return avg_confidence * (0.7 + 0.3 * signal_boost)


# ==========================================
# Utilidades
# ==========================================

def parse_multiple_feedbacks(
    raw_feedbacks: List[Dict[str, Any]],
) -> List[FeedbackSignal]:
    """
    Parsea múltiples feedbacks.
    
    Args:
        raw_feedbacks: Lista de feedbacks crudos
        
    Returns:
        Lista de señales parseadas
    """
    return [parse_feedback(fb) for fb in raw_feedbacks]


def filter_signals_by_confidence(
    signals: List[FeedbackSignal],
    min_confidence: float = 0.5,
) -> List[FeedbackSignal]:
    """
    Filtra señales por confianza mínima.
    
    Args:
        signals: Señales a filtrar
        min_confidence: Confianza mínima
        
    Returns:
        Señales filtradas
    """
    return [s for s in signals if s.confidence >= min_confidence]


def get_signals_for_chunk(
    signals: List[FeedbackSignal],
    chunk_id: str,
) -> List[FeedbackSignal]:
    """
    Obtiene señales relacionadas con un chunk.
    
    Args:
        signals: Lista de señales
        chunk_id: ID del chunk
        
    Returns:
        Señales del chunk
    """
    return [s for s in signals if chunk_id in s.chunk_ids]


def get_signals_for_concept(
    signals: List[FeedbackSignal],
    concept_id: str,
) -> List[FeedbackSignal]:
    """
    Obtiene señales relacionadas con un concepto.
    
    Args:
        signals: Lista de señales
        concept_id: ID del concepto
        
    Returns:
        Señales del concepto
    """
    return [s for s in signals if concept_id in s.concept_ids]


def detect_feedback_pattern(
    signals: List[FeedbackSignal],
) -> Dict[str, Any]:
    """
    Detecta patrones en el feedback.
    
    Args:
        signals: Señales a analizar
        
    Returns:
        Patrones detectados
    """
    if not signals:
        return {"pattern": "none", "confidence": 0.0}
    
    recent_signals = sorted(signals, key=lambda s: s.timestamp or datetime.min)[-10:]
    
    # Detectar tendencias
    values = [s.value for s in recent_signals]
    
    if len(values) >= 3:
        # Tendencia descendente
        if values[-1] < values[-2] < values[-3]:
            return {
                "pattern": "declining",
                "confidence": 0.7,
                "suggestion": "review_content",
            }
        
        # Tendencia ascendente
        if values[-1] > values[-2] > values[-3]:
            return {
                "pattern": "improving",
                "confidence": 0.7,
                "suggestion": "continue_approach",
            }
    
    # Detectar alta variabilidad
    if len(values) >= 5:
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        
        if variance > 0.2:
            return {
                "pattern": "inconsistent",
                "confidence": 0.6,
                "suggestion": "investigate_chunks",
            }
    
    return {
        "pattern": "stable",
        "confidence": 0.5,
        "average_score": sum(values) / len(values) if values else 0.5,
    }
