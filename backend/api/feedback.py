"""
feedback.py
Endpoints para sistema de feedback.
Recibe feedback explícito e implícito de los estudiantes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query as QueryParam,
    status,
)
from pydantic import BaseModel, Field

from backend.deps import get_db, get_student_id
from backend.feedback.signals import (
    FeedbackSignal,
    SignalType,
    create_signal,
    signal_to_dict,
)
from backend.feedback.analytics import (
    ContentEngagement,
    compute_engagement,
    aggregate_engagement,
)
from backend.feedback.graph_updates import (
    apply_feedback_to_graph,
    FeedbackGraphUpdate,
)


# ==========================================
# Router
# ==========================================

router = APIRouter(prefix="/feedback", tags=["Feedback"])


# ==========================================
# Models
# ==========================================

class FeedbackType(str, Enum):
    """Tipos de feedback explícito."""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    TOO_COMPLEX = "too_complex"
    TOO_SIMPLE = "too_simple"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"


class ExplicitFeedbackRequest(BaseModel):
    """Request de feedback explícito."""
    query_id: str
    feedback_type: FeedbackType
    comment: Optional[str] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)


class ImplicitFeedbackRequest(BaseModel):
    """Request de feedback implícito."""
    query_id: str
    signal_type: SignalType
    value: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Respuesta de registro de feedback."""
    feedback_id: str
    status: str
    message: str


class EngagementMetrics(BaseModel):
    """Métricas de engagement."""
    content_id: str
    total_views: int = 0
    helpful_count: int = 0
    not_helpful_count: int = 0
    avg_dwell_time_seconds: float = 0.0
    avg_scroll_depth: float = 0.0
    click_through_rate: float = 0.0
    engagement_score: float = 0.0


class StudentFeedbackSummary(BaseModel):
    """Resumen de feedback de un estudiante."""
    student_id: str
    total_interactions: int = 0
    helpful_ratio: float = 0.0
    avg_session_duration_minutes: float = 0.0
    preferred_complexity: str = "intermediate"
    topics_engaged: List[str] = Field(default_factory=list)
    recent_feedback: List[Dict[str, Any]] = Field(default_factory=list)


class ContentFeedbackSummary(BaseModel):
    """Resumen de feedback para contenido."""
    content_id: str
    total_feedback: int = 0
    positive_ratio: float = 0.0
    negative_ratio: float = 0.0
    complexity_feedback: Dict[str, int] = Field(default_factory=dict)
    common_issues: List[str] = Field(default_factory=list)
    suggested_improvements: List[str] = Field(default_factory=list)


# ==========================================
# In-memory storage (para simplificar)
# ==========================================

_explicit_feedback: List[Dict[str, Any]] = []
_implicit_feedback: List[Dict[str, Any]] = []


# ==========================================
# Endpoints
# ==========================================

@router.post(
    "/explicit",
    response_model=FeedbackResponse,
    summary="Feedback explícito",
    description="Registra feedback explícito del estudiante",
)
async def submit_explicit_feedback(
    request: ExplicitFeedbackRequest,
    x_student_id: Optional[str] = Header(default=None),
    db: Any = Depends(get_db),
) -> FeedbackResponse:
    """
    Registra feedback explícito (útil/no útil, rating, etc.).
    """
    import uuid
    
    feedback_id = str(uuid.uuid4())
    
    # Mapear tipo de feedback a SignalType
    signal_type_map = {
        FeedbackType.HELPFUL: SignalType.THUMBS_UP,
        FeedbackType.NOT_HELPFUL: SignalType.THUMBS_DOWN,
        FeedbackType.TOO_COMPLEX: SignalType.THUMBS_DOWN,
        FeedbackType.TOO_SIMPLE: SignalType.THUMBS_DOWN,
        FeedbackType.INCORRECT: SignalType.THUMBS_DOWN,
        FeedbackType.INCOMPLETE: SignalType.THUMBS_DOWN,
    }
    
    signal_type = signal_type_map.get(request.feedback_type, SignalType.THUMBS_DOWN)
    
    # Crear señal
    signal = create_signal(
        signal_type=signal_type,
        query_id=request.query_id,
        student_id=x_student_id or "",
        raw_value=request.rating,
        metadata={
            "feedback_type": request.feedback_type.value,
            "comment": request.comment,
        },
    )
    
    # Almacenar
    feedback_record = {
        "feedback_id": feedback_id,
        "query_id": request.query_id,
        "student_id": x_student_id,
        "feedback_type": request.feedback_type.value,
        "comment": request.comment,
        "rating": request.rating,
        "created_at": datetime.utcnow().isoformat(),
        "signal": signal_to_dict(signal),
    }
    _explicit_feedback.append(feedback_record)
    
    # Aplicar al grafo (si es posible)
    try:
        update = FeedbackGraphUpdate(
            query_id=request.query_id,
        )
        await apply_feedback_to_graph(update, db)
    except Exception:
        pass  # No fallar por error de actualización de grafo
    
    return FeedbackResponse(
        feedback_id=feedback_id,
        status="recorded",
        message=f"Feedback '{request.feedback_type.value}' registrado",
    )


@router.post(
    "/implicit",
    response_model=FeedbackResponse,
    summary="Feedback implícito",
    description="Registra señales implícitas de comportamiento",
)
async def submit_implicit_feedback(
    request: ImplicitFeedbackRequest,
    x_student_id: Optional[str] = Header(default=None),
    db: Any = Depends(get_db),
) -> FeedbackResponse:
    """
    Registra feedback implícito (tiempo de lectura, scroll, clicks, etc.).
    """
    import uuid
    
    feedback_id = str(uuid.uuid4())
    
    # Crear señal
    signal = create_signal(
        signal_type=request.signal_type,
        query_id=request.query_id,
        student_id=x_student_id or "",
        raw_value=request.value,
        metadata=request.metadata,
    )
    
    # Almacenar
    feedback_record = {
        "feedback_id": feedback_id,
        "query_id": request.query_id,
        "student_id": x_student_id,
        "signal_type": request.signal_type.value,
        "value": request.value,
        "metadata": request.metadata,
        "created_at": datetime.utcnow().isoformat(),
        "signal": signal_to_dict(signal),
    }
    _implicit_feedback.append(feedback_record)
    
    return FeedbackResponse(
        feedback_id=feedback_id,
        status="recorded",
        message=f"Señal '{request.signal_type.value}' registrada",
    )


@router.post(
    "/batch",
    response_model=FeedbackResponse,
    summary="Feedback en lote",
    description="Registra múltiples señales de feedback",
)
async def submit_batch_feedback(
    signals: List[ImplicitFeedbackRequest],
    x_student_id: Optional[str] = Header(default=None),
    db: Any = Depends(get_db),
) -> FeedbackResponse:
    """
    Registra múltiples señales de feedback en lote.
    
    Útil para enviar todas las métricas de una sesión.
    """
    import uuid
    
    batch_id = str(uuid.uuid4())
    recorded = 0
    
    for signal_req in signals:
        try:
            signal = create_signal(
                signal_type=signal_req.signal_type,
                query_id=signal_req.query_id,
                student_id=x_student_id or "",
                raw_value=signal_req.value,
                metadata=signal_req.metadata,
            )
            
            feedback_record = {
                "feedback_id": str(uuid.uuid4()),
                "batch_id": batch_id,
                "query_id": signal_req.query_id,
                "student_id": x_student_id,
                "signal_type": signal_req.signal_type.value,
                "value": signal_req.value,
                "created_at": datetime.utcnow().isoformat(),
                "signal": signal_to_dict(signal),
            }
            _implicit_feedback.append(feedback_record)
            recorded += 1
        except Exception:
            continue
    
    return FeedbackResponse(
        feedback_id=batch_id,
        status="recorded",
        message=f"Registradas {recorded}/{len(signals)} señales",
    )


@router.get(
    "/summary/student/{student_id}",
    response_model=StudentFeedbackSummary,
    summary="Resumen de feedback del estudiante",
    description="Obtiene resumen de feedback de un estudiante",
)
async def get_student_feedback_summary(
    student_id: str,
    days: int = QueryParam(default=30, ge=1, le=365),
    db: Any = Depends(get_db),
) -> StudentFeedbackSummary:
    """
    Obtiene resumen de feedback de un estudiante.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # Filtrar feedback del estudiante
    student_explicit = [
        f for f in _explicit_feedback
        if f.get("student_id") == student_id
        and datetime.fromisoformat(f["created_at"]) > cutoff
    ]
    
    student_implicit = [
        f for f in _implicit_feedback
        if f.get("student_id") == student_id
        and datetime.fromisoformat(f["created_at"]) > cutoff
    ]
    
    total = len(student_explicit) + len(student_implicit)
    
    # Calcular ratio de helpful
    helpful = sum(
        1 for f in student_explicit
        if f.get("feedback_type") == "helpful"
    )
    not_helpful = sum(
        1 for f in student_explicit
        if f.get("feedback_type") == "not_helpful"
    )
    helpful_ratio = helpful / (helpful + not_helpful) if (helpful + not_helpful) > 0 else 0.5
    
    # Determinar complejidad preferida
    too_complex = sum(
        1 for f in student_explicit
        if f.get("feedback_type") == "too_complex"
    )
    too_simple = sum(
        1 for f in student_explicit
        if f.get("feedback_type") == "too_simple"
    )
    
    if too_complex > too_simple:
        preferred_complexity = "simpler"
    elif too_simple > too_complex:
        preferred_complexity = "advanced"
    else:
        preferred_complexity = "intermediate"
    
    # Recopilar feedback reciente
    recent = sorted(
        student_explicit,
        key=lambda x: x["created_at"],
        reverse=True,
    )[:5]
    
    return StudentFeedbackSummary(
        student_id=student_id,
        total_interactions=total,
        helpful_ratio=helpful_ratio,
        preferred_complexity=preferred_complexity,
        recent_feedback=[
            {
                "type": f["feedback_type"],
                "rating": f.get("rating"),
                "date": f["created_at"],
            }
            for f in recent
        ],
    )


@router.get(
    "/summary/content/{content_id}",
    response_model=ContentFeedbackSummary,
    summary="Resumen de feedback del contenido",
    description="Obtiene resumen de feedback para un contenido",
)
async def get_content_feedback_summary(
    content_id: str,
    db: Any = Depends(get_db),
) -> ContentFeedbackSummary:
    """
    Obtiene resumen de feedback para un contenido específico.
    """
    # Filtrar feedback del contenido
    content_explicit = [
        f for f in _explicit_feedback
        if f.get("query_id") == content_id
    ]
    
    content_implicit = [
        f for f in _implicit_feedback
        if f.get("query_id") == content_id
    ]
    
    total = len(content_explicit)
    
    # Calcular ratios
    positive = sum(
        1 for f in content_explicit
        if f.get("feedback_type") == "helpful"
    )
    negative = sum(
        1 for f in content_explicit
        if f.get("feedback_type") in ["not_helpful", "incorrect", "incomplete"]
    )
    
    positive_ratio = positive / total if total > 0 else 0.0
    negative_ratio = negative / total if total > 0 else 0.0
    
    # Conteo de complejidad
    complexity = {
        "too_complex": sum(1 for f in content_explicit if f.get("feedback_type") == "too_complex"),
        "too_simple": sum(1 for f in content_explicit if f.get("feedback_type") == "too_simple"),
        "appropriate": sum(1 for f in content_explicit if f.get("feedback_type") == "helpful"),
    }
    
    # Identificar issues comunes
    common_issues = []
    if complexity["too_complex"] > total * 0.3:
        common_issues.append("Contenido demasiado complejo para muchos usuarios")
    if complexity["too_simple"] > total * 0.3:
        common_issues.append("Contenido demasiado simple para muchos usuarios")
    if negative_ratio > 0.5:
        common_issues.append("Alto ratio de feedback negativo")
    
    # Sugerencias
    suggested_improvements = []
    if complexity["too_complex"] > complexity["too_simple"]:
        suggested_improvements.append("Simplificar explicaciones")
        suggested_improvements.append("Agregar más ejemplos")
    if complexity["too_simple"] > complexity["too_complex"]:
        suggested_improvements.append("Agregar contenido más avanzado")
    
    return ContentFeedbackSummary(
        content_id=content_id,
        total_feedback=total,
        positive_ratio=positive_ratio,
        negative_ratio=negative_ratio,
        complexity_feedback=complexity,
        common_issues=common_issues,
        suggested_improvements=suggested_improvements,
    )


@router.get(
    "/engagement/{content_id}",
    response_model=EngagementMetrics,
    summary="Métricas de engagement",
    description="Obtiene métricas de engagement de contenido",
)
async def get_engagement_metrics(
    content_id: str,
    db: Any = Depends(get_db),
) -> EngagementMetrics:
    """
    Obtiene métricas de engagement para un contenido.
    """
    # Filtrar feedback del contenido
    content_explicit = [
        f for f in _explicit_feedback
        if f.get("query_id") == content_id
    ]
    
    content_implicit = [
        f for f in _implicit_feedback
        if f.get("query_id") == content_id
    ]
    
    # Calcular métricas
    helpful = sum(1 for f in content_explicit if f.get("feedback_type") == "helpful")
    not_helpful = sum(1 for f in content_explicit if f.get("feedback_type") != "helpful")
    
    # Dwell time (de señales implícitas)
    dwell_times = [
        f.get("value", 0)
        for f in content_implicit
        if f.get("signal_type") == "dwell_time"
    ]
    avg_dwell = sum(dwell_times) / len(dwell_times) if dwell_times else 0
    
    # Scroll depth
    scroll_depths = [
        f.get("value", 0)
        for f in content_implicit
        if f.get("signal_type") == "scroll_depth"
    ]
    avg_scroll = sum(scroll_depths) / len(scroll_depths) if scroll_depths else 0
    
    # Clicks
    clicks = sum(
        1 for f in content_implicit
        if f.get("signal_type") == "click"
    )
    views = len(set(f.get("student_id") for f in content_implicit if f.get("student_id")))
    ctr = clicks / views if views > 0 else 0
    
    # Score de engagement
    engagement_score = (
        (helpful / (helpful + not_helpful) if (helpful + not_helpful) > 0 else 0.5) * 0.3 +
        min(1.0, avg_dwell / 60) * 0.3 +  # Normalizar a 60 segundos
        avg_scroll * 0.2 +
        ctr * 0.2
    )
    
    return EngagementMetrics(
        content_id=content_id,
        total_views=views,
        helpful_count=helpful,
        not_helpful_count=not_helpful,
        avg_dwell_time_seconds=avg_dwell,
        avg_scroll_depth=avg_scroll,
        click_through_rate=ctr,
        engagement_score=engagement_score,
    )


@router.get(
    "/recent",
    response_model=List[Dict[str, Any]],
    summary="Feedback reciente",
    description="Lista feedback reciente",
)
async def list_recent_feedback(
    limit: int = QueryParam(default=50, le=200),
    feedback_type: Optional[FeedbackType] = QueryParam(default=None),
    db: Any = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Lista feedback reciente.
    """
    all_feedback = _explicit_feedback.copy()
    
    if feedback_type:
        all_feedback = [
            f for f in all_feedback
            if f.get("feedback_type") == feedback_type.value
        ]
    
    # Ordenar por fecha
    all_feedback.sort(
        key=lambda x: x.get("created_at", ""),
        reverse=True,
    )
    
    return all_feedback[:limit]


@router.delete(
    "/clear",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Limpiar feedback",
    description="Limpia todo el feedback (solo para desarrollo)",
)
async def clear_feedback() -> None:
    """
    Limpia todo el feedback almacenado.
    
    Solo para uso en desarrollo/testing.
    """
    global _explicit_feedback, _implicit_feedback
    _explicit_feedback = []
    _implicit_feedback = []
