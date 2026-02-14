"""
query.py
Endpoint principal para consultas del estudiante.
Usa el Orchestrator para coordinar el pipeline completo.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query as QueryParam,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agents.mode_router import (
    LearningMode,
    detect_mode,
    detect_mode_with_details,
    get_mode_description,
)
from backend.agents.orchestrator import (
    PipelineOrchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    orchestrate,
)
from backend.agents.retrieval_agent import (
    RetrievalResult,
    RetrievalStrategy,
    StrategyDecision,
    retrieve,
    build_strategy_for_mode,
)
from backend.agents.reasoning_agent import (
    GeneratedAnswer,
    PromptConfig,
    generate_answer,
)
from backend.agents.reflection_agent import (
    EvaluationResult,
    ReflectionDecision,
    evaluate_answer,
    get_fallback_message,
)
from backend.deps import (
    get_agents,
    get_db,
    get_database_name,
    get_session_id,
    get_student_id,
    AgentContainer,
)
from backend.memory.cache import (
    get_cached_answer,
    store_cache,
    _generate_cache_key as get_cache_key,
)
from backend.memory.session_memory import (
    store_turn as add_turn,
    TurnRole,
    _create_session,
    _get_session,
)


# ==========================================
# Session Helper
# ==========================================

async def get_or_create_session(session_id: str, student_id: Optional[str] = None):
    """Gets or creates a session."""
    session = _get_session(session_id)
    if session is None:
        session = _create_session(session_id, student_id or "")
    return session


# ==========================================
# Router
# ==========================================

router = APIRouter(prefix="/query", tags=["Query"])


# ==========================================
# Models
# ==========================================

class QueryRequest(BaseModel):
    """Request de consulta."""
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[str] = None
    student_id: Optional[str] = None
    stream: bool = False
    use_cache: bool = True
    max_context: Optional[int] = None
    # Modo pedagógico explícito (opcional, auto-detectado si no se especifica)
    learning_mode: Optional[str] = Field(
        default=None,
        description="Modo pedagógico: 'concept', 'practice', 'exercise_list'. Si se especifica, se usa directamente sin auto-detección."
    )


class QueryResponse(BaseModel):
    """Respuesta de consulta."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    session_id: str
    query_id: str
    strategy_used: str = ""
    learning_mode: str = "concept"  # Nuevo: modo pedagógico usado
    cached: bool = False
    processing_time_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryDebugResponse(QueryResponse):
    """Respuesta con información de debug."""
    retrieval_results: List[Dict[str, Any]] = Field(default_factory=list)
    strategy_decision: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    prompt_used: str = ""
    retries: int = 0


class SuggestionsResponse(BaseModel):
    """Sugerencias de seguimiento."""
    suggestions: List[str]
    based_on_concepts: List[str]


# ==========================================
# Cache (simple in-memory) - database-aware
# ==========================================

_response_cache: Dict[str, QueryResponse] = {}


def _get_cache_key(question: str, student_id: Optional[str], database: Optional[str] = None) -> str:
    """
    Genera clave de caché incluyendo la base de datos.
    
    Args:
        question: Pregunta del estudiante
        student_id: ID del estudiante (opcional)
        database: Nombre de la base de datos (opcional)
    
    Returns:
        Clave única para el cache
    """
    base = question.lower().strip()
    if database:
        base = f"{database}:{base}"
    if student_id:
        base += f":{student_id}"
    return base


# ==========================================
# Endpoints
# ==========================================

@router.post(
    "",
    response_model=QueryResponse,
    summary="Consulta principal",
    description="Endpoint principal para preguntas del estudiante",
)
async def query_student(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
    x_student_id: Optional[str] = Header(default=None),
    x_openai_key: Optional[str] = Header(default=None),
    x_google_key: Optional[str] = Header(default=None),
    x_preferred_provider: Optional[str] = Header(default=None),
    x_model: Optional[str] = Header(default=None),
    db: Any = Depends(get_db),
    database: str = Depends(get_database_name),
) -> QueryResponse:
    """
    Procesa una pregunta del estudiante usando el Orchestrator.
    
    El Orchestrator coordina:
    1. Mode Router → detecta intención pedagógica
    2. Retrieval Agent → obtiene contexto
    3. Reasoning Agent → genera respuesta
    4. Reflection Agent → evalúa y decide
    5. Loop controlado hasta ACCEPT o FALLBACK
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    # Resolver IDs
    session_id = request.session_id or x_session_id
    student_id = request.student_id or x_student_id
    
    # Crear/obtener sesión
    if not session_id:
        session_id = str(uuid.uuid4())
    
    await get_or_create_session(session_id, student_id)
    
    # Determinar modo forzado si se especificó
    forced_mode = None
    if request.learning_mode:
        try:
            forced_mode = LearningMode(request.learning_mode)
        except ValueError:
            pass  # Se auto-detectará
    
    # Verificar caché
    if request.use_cache:
        # Pre-detectar modo para cache key
        learning_mode = forced_mode or await detect_mode(request.question)
        cache_key = _get_cache_key(request.question, student_id, database) + f":{learning_mode.value}"
        if cache_key in _response_cache:
            cached = _response_cache[cache_key]
            cached.cached = True
            cached.query_id = query_id
            cached.processing_time_ms = int((time.time() - start_time) * 1000)
            return cached
    
    # Configurar orchestrator
    orch_config = OrchestratorConfig(
        max_retries=3,
        min_acceptable_score=0.5,
        enable_reflection_loop=True,
        fallback_on_max_retries=True,
    )
    
    # Ejecutar pipeline via orchestrator
    try:
        result = await orchestrate(
            query=request.question,
            session_id=session_id,
            student_id=student_id,
            user_openai_key=x_openai_key,
            user_google_key=x_google_key,
            preferred_provider=x_preferred_provider,
            user_model=x_model,
            forced_mode=forced_mode,
            config=orch_config,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Modelo no encontrado. Verifica el nombre del modelo.",
            )
        elif "api key" in error_msg or "invalid" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key inválida o sin permisos.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error procesando consulta: {str(e)[:200]}",
            )
    
    # Guardar en sesión
    await add_turn(
        session_id=session_id,
        role=TurnRole.USER,
        content=request.question,
    )
    await add_turn(
        session_id=session_id,
        role=TurnRole.ASSISTANT,
        content=result.answer,
        metadata={"concepts": result.concepts, "learning_mode": result.mode.value},
    )
    
    # Construir respuesta
    processing_time = int((time.time() - start_time) * 1000)
    
    response = QueryResponse(
        answer=result.answer,
        sources=result.sources,
        concepts=result.concepts,
        confidence=result.confidence,
        session_id=session_id,
        query_id=query_id,
        strategy_used="orchestrated",
        learning_mode=result.mode.value,
        cached=False,
        processing_time_ms=processing_time,
        metadata={
            "iterations": result.iterations,
            "final_decision": result.final_decision.value,
            "mode_changes": result.mode_changes,
            "retrieval_expansions": result.retrieval_expansions,
            "mode_description": get_mode_description(result.mode),
        },
    )
    
    # Guardar en caché
    if request.use_cache:
        cache_key = _get_cache_key(request.question, student_id, database) + f":{result.mode.value}"
        _response_cache[cache_key] = response
    
    return response


@router.post(
    "/stream",
    summary="Consulta con streaming",
    description="Consulta con respuesta en streaming (simplificado)",
)
async def query_stream(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
    x_student_id: Optional[str] = Header(default=None),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_google_key: Optional[str] = Header(default=None, alias="X-Google-Key"),
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_model: Optional[str] = Header(default=None, alias="X-Model"),
    db: Any = Depends(get_db),
    database: str = Depends(get_database_name),
) -> StreamingResponse:
    """
    Procesa consulta con respuesta en streaming.
    Nota: Streaming usa pipeline simplificado sin loop de reflexión completo.
    Para reflexión completa, usar endpoint principal sin streaming.
    """
    session_id = request.session_id or x_session_id or str(uuid.uuid4())
    student_id = request.student_id or x_student_id
    
    await get_or_create_session(session_id, student_id)
    
    # Detectar modo pedagógico
    if request.learning_mode:
        try:
            learning_mode = LearningMode(request.learning_mode)
        except ValueError:
            learning_mode = await detect_mode(request.question)
    else:
        learning_mode = await detect_mode(request.question)
    
    # Construir estrategia usando el builder
    strategy = build_strategy_for_mode(learning_mode)
    
    # Recuperar contexto
    retrieval_result = await retrieve(
        query=request.question,
        strategy=strategy,
        student_id=student_id,
        openai_api_key=x_openai_key,
        google_api_key=x_google_key,
        mode=learning_mode,
    )
    
    # Configurar prompt
    prompt_config = PromptConfig(language="es")
    
    async def generate_stream():
        """Generador de streaming simplificado."""
        # Generar respuesta no-streaming y enviarla en chunks
        answer = await generate_answer(
            query=request.question,
            context_chunks=retrieval_result.results,
            mode=learning_mode,
            session_id=session_id,
            config=prompt_config,
            user_openai_key=x_openai_key,
            user_google_key=x_google_key,
            preferred_provider=x_preferred_provider,
            user_model=x_model,
        )
        
        # Enviar respuesta en chunks
        full_response = answer.answer
        chunk_size = 50
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i+chunk_size]
            yield f"data: {chunk}\n\n"
        
        # Guardar en sesión al final
        await add_turn(session_id, TurnRole.USER, request.question)
        await add_turn(
            session_id, TurnRole.ASSISTANT, full_response,
            metadata={"learning_mode": learning_mode.value}
        )
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
            "X-Learning-Mode": learning_mode.value,
        },
    )


@router.post(
    "/debug",
    response_model=QueryDebugResponse,
    summary="Consulta con debug",
    description="Consulta con información de debug detallada",
)
async def query_debug(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
    x_student_id: Optional[str] = Header(default=None),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_google_key: Optional[str] = Header(default=None, alias="X-Google-Key"),
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_model: Optional[str] = Header(default=None, alias="X-Model"),
    db: Any = Depends(get_db),
) -> QueryDebugResponse:
    """
    Consulta con información completa de debug.
    Usa el orchestrator con trazabilidad completa.
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    session_id = request.session_id or x_session_id or str(uuid.uuid4())
    student_id = request.student_id or x_student_id
    
    await get_or_create_session(session_id, student_id)
    
    # Detectar modo
    if request.learning_mode:
        try:
            forced_mode = LearningMode(request.learning_mode)
        except ValueError:
            forced_mode = None
    else:
        forced_mode = None
    
    # Ejecutar via orchestrator
    orch_config = OrchestratorConfig(
        max_retries=3,
        enable_reflection_loop=True,
        log_decisions=True,
    )
    
    orchestrator = PipelineOrchestrator(orch_config)
    result = await orchestrator.process(
        query=request.question,
        session_id=session_id,
        student_id=student_id,
        user_openai_key=x_openai_key,
        user_google_key=x_google_key,
        preferred_provider=x_preferred_provider,
        user_model=x_model,
        forced_mode=forced_mode,
    )
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return QueryDebugResponse(
        answer=result.answer,
        sources=result.sources,
        concepts=result.concepts,
        confidence=result.confidence,
        session_id=session_id,
        query_id=query_id,
        strategy_used="orchestrated",
        learning_mode=result.mode.value,
        cached=False,
        processing_time_ms=processing_time,
        metadata={
            "iterations": result.iterations,
            "mode_changes": result.mode_changes,
        },
        retrieval_results=[],  # Simplificado
        strategy_decision={"mode": result.mode.value},
        evaluation={"final_decision": result.final_decision.value},
        prompt_used="[via orchestrator]",
        retries=result.iterations - 1,
    )


@router.get(
    "/suggestions",
    response_model=SuggestionsResponse,
    summary="Sugerencias de seguimiento",
    description="Obtiene preguntas sugeridas basadas en la sesión",
)
async def get_suggestions(
    session_id: str = QueryParam(...),
    db: Any = Depends(get_db),
) -> SuggestionsResponse:
    """
    Obtiene sugerencias de preguntas de seguimiento.
    """
    from backend.agents.reasoning_agent import get_suggested_followups
    from backend.memory.session_memory import get_active_concepts
    
    concepts = await get_active_concepts(session_id)
    
    suggestions = await get_suggested_followups(
        query="",  # No hay query específico
        answer="",
        concepts=concepts,
    )
    
    return SuggestionsResponse(
        suggestions=suggestions,
        based_on_concepts=concepts[:5],
    )


@router.get(
    "/cached",
    response_model=Optional[QueryResponse],
    summary="Obtener respuesta cacheada",
    description="Busca respuesta en caché sin regenerar",
)
async def get_cached(
    question: str = QueryParam(...),
    student_id: Optional[str] = QueryParam(default=None),
    database: str = Depends(get_database_name),
) -> Optional[QueryResponse]:
    """
    Busca respuesta en caché.
    """
    cache_key = _get_cache_key(question, student_id, database)
    
    if cache_key in _response_cache:
        response = _response_cache[cache_key]
        response.cached = True
        return response
    
    return None


@router.delete(
    "/cache",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Limpiar caché",
    description="Limpia el caché de respuestas",
)
async def clear_cache() -> None:
    """
    Limpia el caché de respuestas.
    """
    global _response_cache
    _response_cache = {}
