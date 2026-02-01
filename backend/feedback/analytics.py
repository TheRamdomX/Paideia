"""
analytics.py
Detecta chunks pobres, conceptos confusos.
Sistema de análisis de calidad del contenido.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.db.surreal import execute, get_db
from backend.feedback.signals import (
    FeedbackSignal,
    NormalizedFeedback,
    SentimentPolarity,
    SignalType,
    weight_feedback,
)
from backend.settings import get_rag_config


# ==========================================
# Estructuras de Datos
# ==========================================

class ContentQuality(str, Enum):
    """Niveles de calidad del contenido."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ChunkMetrics:
    """Métricas de un chunk."""
    
    chunk_id: str = ""
    
    # Uso
    times_retrieved: int = 0
    times_used_in_answer: int = 0
    
    # Feedback
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    average_score: float = 0.5
    
    # Calidad
    quality: ContentQuality = ContentQuality.GOOD
    needs_review: bool = False
    
    # Timestamps
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "times_retrieved": self.times_retrieved,
            "times_used_in_answer": self.times_used_in_answer,
            "total_feedback": self.total_feedback,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "average_score": self.average_score,
            "quality": self.quality.value,
            "needs_review": self.needs_review,
        }


@dataclass
class ConceptMetrics:
    """Métricas de un concepto."""
    
    concept_id: str = ""
    concept_name: str = ""
    
    # Interacciones
    total_queries: int = 0
    unique_students: int = 0
    
    # Feedback
    average_satisfaction: float = 0.5
    confusion_score: float = 0.0  # 0 = claro, 1 = muy confuso
    
    # Patrones
    common_followups: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    
    # Estado
    is_confusing: bool = False
    needs_improvement: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "total_queries": self.total_queries,
            "unique_students": self.unique_students,
            "average_satisfaction": self.average_satisfaction,
            "confusion_score": self.confusion_score,
            "is_confusing": self.is_confusing,
            "needs_improvement": self.needs_improvement,
        }


@dataclass 
class SystemMetrics:
    """Métricas globales del sistema."""
    
    # Totales
    total_queries: int = 0
    total_feedback: int = 0
    total_chunks: int = 0
    total_concepts: int = 0
    
    # Promedios
    average_satisfaction: float = 0.5
    average_relevance: float = 0.5
    
    # Problemas detectados
    poor_chunks: int = 0
    confusing_concepts: int = 0
    
    # Periodo
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "total_feedback": self.total_feedback,
            "total_chunks": self.total_chunks,
            "total_concepts": self.total_concepts,
            "average_satisfaction": self.average_satisfaction,
            "average_relevance": self.average_relevance,
            "poor_chunks": self.poor_chunks,
            "confusing_concepts": self.confusing_concepts,
        }


# ==========================================
# Storage In-Memory (para métricas en tiempo real)
# ==========================================

_chunk_metrics: Dict[str, ChunkMetrics] = {}
_concept_metrics: Dict[str, ConceptMetrics] = {}
_feedback_history: List[NormalizedFeedback] = []


def _get_chunk_metrics(chunk_id: str) -> ChunkMetrics:
    """Obtiene o crea métricas de chunk."""
    if chunk_id not in _chunk_metrics:
        _chunk_metrics[chunk_id] = ChunkMetrics(
            chunk_id=chunk_id,
            created_at=datetime.now(timezone.utc),
        )
    return _chunk_metrics[chunk_id]


def _get_concept_metrics(concept_id: str) -> ConceptMetrics:
    """Obtiene o crea métricas de concepto."""
    if concept_id not in _concept_metrics:
        _concept_metrics[concept_id] = ConceptMetrics(concept_id=concept_id)
    return _concept_metrics[concept_id]


def clear_analytics_state() -> None:
    """Limpia estado (para testing)."""
    _chunk_metrics.clear()
    _concept_metrics.clear()
    _feedback_history.clear()


# ==========================================
# Detección de Chunks Pobres
# ==========================================

async def detect_poor_chunks(
    threshold: float = 0.4,
    min_feedback: int = 3,
) -> List[ChunkMetrics]:
    """
    Identifica chunks con feedback consistentemente bajo.
    
    Args:
        threshold: Umbral de calidad (bajo = pobre)
        min_feedback: Mínimo de feedback para considerar
        
    Returns:
        Lista de chunks con problemas de calidad
    """
    poor_chunks: List[ChunkMetrics] = []
    
    for chunk_id, metrics in _chunk_metrics.items():
        # Necesita suficiente feedback para ser significativo
        if metrics.total_feedback < min_feedback:
            continue
        
        # Calcular ratio de feedback positivo/negativo
        if metrics.total_feedback > 0:
            positive_ratio = metrics.positive_feedback / metrics.total_feedback
        else:
            positive_ratio = 0.5
        
        # Verificar múltiples criterios
        is_poor = False
        
        # Criterio 1: Score promedio bajo
        if metrics.average_score < threshold:
            is_poor = True
        
        # Criterio 2: Mayoría de feedback negativo
        if positive_ratio < 0.3:
            is_poor = True
        
        # Criterio 3: Recuperado mucho pero poco usado en respuestas
        if metrics.times_retrieved > 10 and metrics.times_used_in_answer < metrics.times_retrieved * 0.2:
            is_poor = True
        
        if is_poor:
            metrics.quality = _calculate_quality(metrics.average_score)
            metrics.needs_review = True
            poor_chunks.append(metrics)
    
    # También buscar en base de datos si existe
    try:
        db_poor_chunks = await _fetch_poor_chunks_from_db(threshold, min_feedback)
        
        # Merge con in-memory
        seen_ids = {m.chunk_id for m in poor_chunks}
        for chunk in db_poor_chunks:
            if chunk.chunk_id not in seen_ids:
                poor_chunks.append(chunk)
                
    except Exception:
        pass
    
    # Ordenar por severidad
    poor_chunks.sort(key=lambda m: m.average_score)
    
    return poor_chunks


async def _fetch_poor_chunks_from_db(
    threshold: float,
    min_feedback: int,
) -> List[ChunkMetrics]:
    """Busca chunks pobres en la base de datos."""
    try:
        db = await get_db()
        
        surql = """
            SELECT 
                chunk_id,
                total_feedback,
                positive_feedback,
                negative_feedback,
                average_score
            FROM chunk_metrics
            WHERE average_score < $threshold
            AND total_feedback >= $min_feedback
            ORDER BY average_score ASC
            LIMIT 50
        """
        
        result = await db.query(surql, {
            "threshold": threshold,
            "min_feedback": min_feedback,
        })
        
        chunks: List[ChunkMetrics] = []
        
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                metrics = ChunkMetrics(
                    chunk_id=row.get("chunk_id", ""),
                    total_feedback=row.get("total_feedback", 0),
                    positive_feedback=row.get("positive_feedback", 0),
                    negative_feedback=row.get("negative_feedback", 0),
                    average_score=row.get("average_score", 0.5),
                    quality=_calculate_quality(row.get("average_score", 0.5)),
                    needs_review=True,
                )
                chunks.append(metrics)
        
        return chunks
        
    except Exception:
        return []


def _calculate_quality(score: float) -> ContentQuality:
    """Calcula nivel de calidad según score."""
    if score >= 0.85:
        return ContentQuality.EXCELLENT
    elif score >= 0.7:
        return ContentQuality.GOOD
    elif score >= 0.5:
        return ContentQuality.FAIR
    elif score >= 0.3:
        return ContentQuality.POOR
    else:
        return ContentQuality.CRITICAL


# ==========================================
# Tracking de Confusión
# ==========================================

async def track_confusion(
    concept_id: str,
    signals: List[FeedbackSignal],
    student_id: Optional[str] = None,
) -> ConceptMetrics:
    """
    Rastrea señales de confusión en conceptos.
    
    Args:
        concept_id: ID del concepto
        signals: Señales de feedback relacionadas
        student_id: ID del estudiante (opcional)
        
    Returns:
        Métricas actualizadas del concepto
    """
    metrics = _get_concept_metrics(concept_id)
    now = datetime.now(timezone.utc)
    
    # Actualizar contadores
    metrics.total_queries += 1
    
    if student_id:
        # Rastrear estudiantes únicos (simplificado)
        metrics.unique_students += 1
    
    # Analizar señales de confusión
    confusion_indicators = 0
    total_signals = len(signals) if signals else 1
    
    for signal in signals:
        # Señales que indican confusión
        if signal.signal_type == SignalType.REPHRASE:
            confusion_indicators += 2
        elif signal.signal_type == SignalType.FOLLOWUP:
            confusion_indicators += 1
        elif signal.signal_type == SignalType.THUMBS_DOWN:
            confusion_indicators += 2
        elif signal.signal_type == SignalType.REPORT:
            confusion_indicators += 3
        elif signal.value < 0.4:
            confusion_indicators += 1
    
    # Calcular score de confusión (0-1)
    new_confusion = min(1.0, confusion_indicators / (total_signals * 2))
    
    # Media móvil exponencial
    alpha = 0.2
    metrics.confusion_score = alpha * new_confusion + (1 - alpha) * metrics.confusion_score
    
    # Actualizar satisfacción promedio
    if signals:
        weighted = weight_feedback(signals)
        metrics.average_satisfaction = (
            alpha * weighted.overall_score + 
            (1 - alpha) * metrics.average_satisfaction
        )
    
    # Determinar si es confuso
    metrics.is_confusing = metrics.confusion_score > 0.5
    metrics.needs_improvement = metrics.confusion_score > 0.3 or metrics.average_satisfaction < 0.5
    
    # Persistir cambios
    await _save_concept_metrics(metrics)
    
    return metrics


async def _save_concept_metrics(metrics: ConceptMetrics) -> None:
    """Guarda métricas de concepto en la base de datos."""
    try:
        db = await get_db()
        
        surql = """
            UPSERT concept_metrics SET
                concept_id = $concept_id,
                concept_name = $concept_name,
                total_queries = $total_queries,
                unique_students = $unique_students,
                average_satisfaction = $average_satisfaction,
                confusion_score = $confusion_score,
                is_confusing = $is_confusing,
                needs_improvement = $needs_improvement,
                updated_at = $updated_at
            WHERE concept_id = $concept_id
        """
        
        await db.query(surql, {
            "concept_id": metrics.concept_id,
            "concept_name": metrics.concept_name,
            "total_queries": metrics.total_queries,
            "unique_students": metrics.unique_students,
            "average_satisfaction": metrics.average_satisfaction,
            "confusion_score": metrics.confusion_score,
            "is_confusing": metrics.is_confusing,
            "needs_improvement": metrics.needs_improvement,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        
    except Exception:
        pass


async def get_confusing_concepts(
    threshold: float = 0.5,
    limit: int = 20,
) -> List[ConceptMetrics]:
    """
    Obtiene conceptos con alta confusión.
    
    Args:
        threshold: Umbral de confusión
        limit: Máximo de resultados
        
    Returns:
        Lista de conceptos confusos
    """
    confusing: List[ConceptMetrics] = []
    
    # Desde in-memory
    for metrics in _concept_metrics.values():
        if metrics.confusion_score >= threshold or metrics.is_confusing:
            confusing.append(metrics)
    
    # Desde base de datos
    try:
        db = await get_db()
        
        surql = """
            SELECT *
            FROM concept_metrics
            WHERE confusion_score >= $threshold OR is_confusing = true
            ORDER BY confusion_score DESC
            LIMIT $limit
        """
        
        result = await db.query(surql, {
            "threshold": threshold,
            "limit": limit,
        })
        
        if result and result[0].get("result"):
            seen_ids = {m.concept_id for m in confusing}
            
            for row in result[0]["result"]:
                if row.get("concept_id") not in seen_ids:
                    metrics = ConceptMetrics(
                        concept_id=row.get("concept_id", ""),
                        concept_name=row.get("concept_name", ""),
                        total_queries=row.get("total_queries", 0),
                        unique_students=row.get("unique_students", 0),
                        average_satisfaction=row.get("average_satisfaction", 0.5),
                        confusion_score=row.get("confusion_score", 0.0),
                        is_confusing=row.get("is_confusing", False),
                        needs_improvement=row.get("needs_improvement", False),
                    )
                    confusing.append(metrics)
                    
    except Exception:
        pass
    
    # Ordenar y limitar
    confusing.sort(key=lambda m: m.confusion_score, reverse=True)
    
    return confusing[:limit]


# ==========================================
# Agregación de Métricas
# ==========================================

async def aggregate_metrics(
    period_days: int = 7,
) -> SystemMetrics:
    """
    Agrega métricas del sistema para un período.
    
    Args:
        period_days: Días a considerar
        
    Returns:
        Métricas agregadas del sistema
    """
    now = datetime.now(timezone.utc)
    period_start = now - timedelta(days=period_days)
    
    metrics = SystemMetrics(
        period_start=period_start,
        period_end=now,
    )
    
    # Agregar desde in-memory
    metrics.total_chunks = len(_chunk_metrics)
    metrics.total_concepts = len(_concept_metrics)
    
    # Calcular promedios de chunks
    if _chunk_metrics:
        total_chunk_score = sum(m.average_score for m in _chunk_metrics.values())
        metrics.average_relevance = total_chunk_score / len(_chunk_metrics)
        metrics.poor_chunks = sum(1 for m in _chunk_metrics.values() if m.needs_review)
    
    # Calcular promedios de conceptos
    if _concept_metrics:
        total_satisfaction = sum(m.average_satisfaction for m in _concept_metrics.values())
        metrics.average_satisfaction = total_satisfaction / len(_concept_metrics)
        metrics.confusing_concepts = sum(1 for m in _concept_metrics.values() if m.is_confusing)
        metrics.total_queries = sum(m.total_queries for m in _concept_metrics.values())
    
    # Contar feedback
    metrics.total_feedback = len(_feedback_history)
    
    # Complementar con datos de la base de datos
    try:
        db_metrics = await _fetch_db_metrics(period_start)
        
        # Combinar métricas
        metrics.total_queries = max(metrics.total_queries, db_metrics.get("total_queries", 0))
        metrics.total_feedback = max(metrics.total_feedback, db_metrics.get("total_feedback", 0))
        
    except Exception:
        pass
    
    return metrics


async def _fetch_db_metrics(period_start: datetime) -> Dict[str, Any]:
    """Obtiene métricas de la base de datos."""
    try:
        db = await get_db()
        
        surql = """
            SELECT 
                count() as total_queries
            FROM query_log
            WHERE timestamp >= $period_start
            GROUP ALL
        """
        
        result = await db.query(surql, {
            "period_start": period_start.isoformat(),
        })
        
        if result and result[0].get("result"):
            return result[0]["result"][0]
            
    except Exception:
        pass
    
    return {}


# ==========================================
# Registro de Feedback
# ==========================================

async def record_feedback(
    feedback: NormalizedFeedback,
    chunk_ids: Optional[List[str]] = None,
    concept_ids: Optional[List[str]] = None,
) -> None:
    """
    Registra feedback y actualiza métricas.
    
    Args:
        feedback: Feedback normalizado
        chunk_ids: IDs de chunks relacionados
        concept_ids: IDs de conceptos relacionados
    """
    now = datetime.now(timezone.utc)
    
    # Guardar en historial
    _feedback_history.append(feedback)
    
    # Mantener historial limitado
    if len(_feedback_history) > 1000:
        _feedback_history.pop(0)
    
    # Actualizar métricas de chunks
    if chunk_ids:
        for chunk_id in chunk_ids:
            metrics = _get_chunk_metrics(chunk_id)
            
            metrics.times_used_in_answer += 1
            metrics.total_feedback += 1
            metrics.last_used = now
            metrics.last_updated = now
            
            if feedback.overall_score >= 0.6:
                metrics.positive_feedback += 1
            else:
                metrics.negative_feedback += 1
            
            # Actualizar promedio
            alpha = 0.3
            metrics.average_score = (
                alpha * feedback.overall_score +
                (1 - alpha) * metrics.average_score
            )
            
            metrics.quality = _calculate_quality(metrics.average_score)
            metrics.needs_review = metrics.average_score < 0.4
    
    # Actualizar métricas de conceptos
    if concept_ids and feedback.signals:
        for concept_id in concept_ids:
            await track_confusion(
                concept_id=concept_id,
                signals=feedback.signals,
            )
    
    # Persistir feedback en la base de datos
    await _save_feedback_to_db(feedback, chunk_ids, concept_ids)


async def _save_feedback_to_db(
    feedback: NormalizedFeedback,
    chunk_ids: Optional[List[str]],
    concept_ids: Optional[List[str]],
) -> None:
    """Guarda feedback en la base de datos."""
    try:
        db = await get_db()
        
        surql = """
            CREATE feedback SET
                query_id = $query_id,
                overall_score = $overall_score,
                relevance_score = $relevance_score,
                helpfulness_score = $helpfulness_score,
                accuracy_score = $accuracy_score,
                signal_count = $signal_count,
                polarity = $polarity,
                chunk_ids = $chunk_ids,
                concept_ids = $concept_ids,
                timestamp = $timestamp
        """
        
        await db.query(surql, {
            "query_id": feedback.query_id,
            "overall_score": feedback.overall_score,
            "relevance_score": feedback.relevance_score,
            "helpfulness_score": feedback.helpfulness_score,
            "accuracy_score": feedback.accuracy_score,
            "signal_count": feedback.signal_count,
            "polarity": feedback.polarity.value,
            "chunk_ids": chunk_ids or [],
            "concept_ids": concept_ids or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
    except Exception:
        pass


# ==========================================
# Registro de Uso
# ==========================================

async def record_chunk_retrieval(
    chunk_ids: List[str],
    used_in_answer: bool = False,
) -> None:
    """
    Registra que chunks fueron recuperados.
    
    Args:
        chunk_ids: IDs de chunks recuperados
        used_in_answer: Si fueron usados en la respuesta
    """
    now = datetime.now(timezone.utc)
    
    for chunk_id in chunk_ids:
        metrics = _get_chunk_metrics(chunk_id)
        
        metrics.times_retrieved += 1
        metrics.last_used = now
        
        if used_in_answer:
            metrics.times_used_in_answer += 1


# ==========================================
# Utilidades
# ==========================================

def get_chunk_quality_distribution() -> Dict[ContentQuality, int]:
    """
    Obtiene distribución de calidad de chunks.
    
    Returns:
        Dict con conteo por nivel de calidad
    """
    distribution: Dict[ContentQuality, int] = defaultdict(int)
    
    for metrics in _chunk_metrics.values():
        distribution[metrics.quality] += 1
    
    return dict(distribution)


async def get_trending_concepts(
    limit: int = 10,
    period_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Obtiene conceptos más consultados recientemente.
    
    Args:
        limit: Máximo de resultados
        period_days: Período a considerar
        
    Returns:
        Lista de conceptos trending
    """
    # Ordenar por queries totales
    sorted_concepts = sorted(
        _concept_metrics.values(),
        key=lambda m: m.total_queries,
        reverse=True
    )
    
    return [
        {
            "concept_id": m.concept_id,
            "concept_name": m.concept_name,
            "total_queries": m.total_queries,
            "satisfaction": m.average_satisfaction,
        }
        for m in sorted_concepts[:limit]
    ]
