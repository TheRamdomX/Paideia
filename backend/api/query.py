"""
query.py
Endpoint principal para consultas del estudiante.
Orquesta retrieval + reasoning + reflection.
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

from backend.agents.reasoning_agent import (
    GeneratedAnswer,
    PromptConfig,
    ResponseStyle,
    build_prompt,
    generate_answer,
    generate_answer_stream,
)
from backend.agents.reflection_agent import (
    EvaluationResult,
    ReflectionDecision,
    RetryConfig,
    decide_retry,
    evaluate_answer,
    generate_fallback_response,
    get_retry_adjustments,
)
from backend.agents.retrieval_agent import (
    RetrievalResult,
    RetrievalStrategy,
    decide_strategy,
    retrieve,
    score_results,
)
from backend.deps import (
    get_agents,
    get_db,
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
    style: Optional[ResponseStyle] = None
    max_context: Optional[int] = None


class QueryResponse(BaseModel):
    """Respuesta de consulta."""
    answer: str
    sources: List[str] = Field(default_factory=list)
    concepts: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    session_id: str
    query_id: str
    strategy_used: str = ""
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
# Cache (simple in-memory)
# ==========================================

_response_cache: Dict[str, QueryResponse] = {}


def _get_cache_key(question: str, student_id: Optional[str]) -> str:
    """Genera clave de caché."""
    base = question.lower().strip()
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
    x_session_id: Optional[str] = Header(default=None, alias="X-Session-ID"),
    x_student_id: Optional[str] = Header(default=None, alias="X-Student-ID"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_google_key: Optional[str] = Header(default=None, alias="X-Google-Key"),
    x_preferred_provider: Optional[str] = Header(default=None, alias="X-Preferred-Provider"),
    x_model: Optional[str] = Header(default=None, alias="X-Model"),
    db: Any = Depends(get_db),
) -> QueryResponse:
    """
    Procesa una pregunta del estudiante.
    
    Pipeline:
    1. Verificar caché
    2. Decidir estrategia de retrieval
    3. Recuperar contexto
    4. Generar respuesta
    5. Evaluar y posiblemente reintentar
    6. Guardar en sesión y caché
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
    
    # 1. Verificar caché
    if request.use_cache:
        cache_key = _get_cache_key(request.question, student_id)
        if cache_key in _response_cache:
            cached = _response_cache[cache_key]
            cached.cached = True
            cached.query_id = query_id
            cached.processing_time_ms = int((time.time() - start_time) * 1000)
            return cached
    
    # 2. Decidir estrategia
    strategy_decision = await decide_strategy(
        query=request.question,
        student_id=student_id,
    )
    
    # 3. Recuperar contexto
    retrieval_result = await retrieve(
        query=request.question,
        strategy=strategy_decision,
        student_id=student_id,
    )
    
    # Re-score si hay student_id
    if student_id:
        scored_results = await score_results(
            results=retrieval_result.results,
            query=request.question,
            student_id=student_id,
        )
    else:
        scored_results = retrieval_result.results
    
    # 4. Configurar prompt
    prompt_config = PromptConfig(
        language="es",
        style=request.style or ResponseStyle.DETAILED,
    )
    if request.max_context:
        prompt_config.max_context_tokens = request.max_context
    
    # 5. Generar respuesta
    try:
        answer = await generate_answer(
            query=request.question,
            context_chunks=scored_results,
            session_id=session_id,
            student_id=student_id,
            config=prompt_config,
            user_openai_key=x_openai_key,
            user_google_key=x_google_key,
            preferred_provider=x_preferred_provider,
            user_model=x_model,
        )
    except ValueError as e:
        # Errores de configuración (API key inválida, modelo no encontrado)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Modelo no encontrado. Verifica el nombre del modelo.",
            )
        elif "api key" in error_msg or "invalid" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key inválida o sin permisos.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error al generar respuesta: {str(e)[:200]}",
            )
    
    # 6. Evaluar respuesta
    evaluation = await evaluate_answer(
        query=request.question,
        answer=answer,
        context_chunks=scored_results,
    )
    
    # 7. Reintentar si es necesario
    retries = 0
    max_retries = 2
    
    while evaluation.decision in [
        ReflectionDecision.RETRY,
        ReflectionDecision.REQUEST_MORE_CONTEXT,
    ] and retries < max_retries:
        retries += 1
        
        # Ajustar configuración
        prompt_config = get_retry_adjustments(evaluation, prompt_config)
        
        # Regenerar
        answer = await generate_answer(
            query=request.question,
            context_chunks=scored_results,
            session_id=session_id,
            student_id=student_id,
            config=prompt_config,
            user_openai_key=x_openai_key,
            user_google_key=x_google_key,
            preferred_provider=x_preferred_provider,
            user_model=x_model,
        )
        
        # Re-evaluar
        evaluation = await evaluate_answer(
            query=request.question,
            answer=answer,
            context_chunks=scored_results,
            previous_evaluation=evaluation,
        )
    
    # 8. Fallback si aún falla
    if evaluation.decision == ReflectionDecision.FALLBACK:
        fallback_text = await generate_fallback_response(
            query=request.question,
            context_chunks=scored_results,
            evaluation=evaluation,
        )
        answer.answer = fallback_text
        answer.confidence = 0.5
    
    # 9. Guardar en sesión
    await add_turn(
        session_id=session_id,
        role=TurnRole.USER,
        content=request.question,
    )
    await add_turn(
        session_id=session_id,
        role=TurnRole.ASSISTANT,
        content=answer.answer,
        metadata={"concepts": answer.concepts_covered},
    )
    
    # 10. Construir respuesta
    processing_time = int((time.time() - start_time) * 1000)
    
    response = QueryResponse(
        answer=answer.answer,
        sources=answer.sources,
        concepts=answer.concepts_covered,
        confidence=answer.confidence,
        session_id=session_id,
        query_id=query_id,
        strategy_used=strategy_decision.strategy.value,
        cached=False,
        processing_time_ms=processing_time,
        metadata={
            "style": answer.style_used.value,
            "retries": retries,
            "evaluation_score": evaluation.overall_score,
        },
    )
    
    # 11. Guardar en caché
    if request.use_cache:
        cache_key = _get_cache_key(request.question, student_id)
        _response_cache[cache_key] = response
    
    return response


@router.post(
    "/stream",
    summary="Consulta con streaming",
    description="Consulta con respuesta en streaming",
)
async def query_stream(
    request: QueryRequest,
    x_session_id: Optional[str] = Header(default=None),
    x_student_id: Optional[str] = Header(default=None),
    db: Any = Depends(get_db),
) -> StreamingResponse:
    """
    Procesa consulta con respuesta en streaming.
    """
    session_id = request.session_id or x_session_id or str(uuid.uuid4())
    student_id = request.student_id or x_student_id
    
    await get_or_create_session(session_id, student_id)
    
    # Decidir estrategia
    strategy_decision = await decide_strategy(
        query=request.question,
        student_id=student_id,
    )
    
    # Recuperar contexto
    retrieval_result = await retrieve(
        query=request.question,
        strategy=strategy_decision,
        session_id=session_id,
    )
    
    # Configurar prompt
    prompt_config = PromptConfig(
        language="es",
        style=request.style or ResponseStyle.DETAILED,
    )
    
    async def generate_stream():
        """Generador de streaming."""
        full_response = ""
        
        async for chunk in generate_answer_stream(
            query=request.question,
            context_chunks=retrieval_result.results,
            session_id=session_id,
            student_id=student_id,
            config=prompt_config,
        ):
            full_response += chunk
            yield f"data: {chunk}\n\n"
        
        # Guardar en sesión al final
        await add_turn(session_id, TurnRole.USER, request.question)
        await add_turn(session_id, TurnRole.ASSISTANT, full_response)
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-Id": session_id,
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
    db: Any = Depends(get_db),
) -> QueryDebugResponse:
    """
    Consulta con información completa de debug.
    
    Incluye:
    - Resultados de retrieval
    - Decisión de estrategia
    - Evaluación de respuesta
    - Prompt utilizado
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    session_id = request.session_id or x_session_id or str(uuid.uuid4())
    student_id = request.student_id or x_student_id
    
    await get_or_create_session(session_id, student_id)
    
    # Estrategia
    strategy_decision = await decide_strategy(
        query=request.question,
        student_id=student_id,
    )
    
    # Retrieval
    retrieval_result = await retrieve(
        query=request.question,
        strategy=strategy_decision,
        student_id=student_id,
    )
    
    # Re-score
    if student_id:
        scored_results = await score_results(
            results=retrieval_result.results,
            query=request.question,
            student_id=student_id,
        )
    else:
        scored_results = retrieval_result.results
    
    # Configuración
    prompt_config = PromptConfig(
        language="es",
        style=request.style or ResponseStyle.DETAILED,
    )
    
    # Prompt
    prompt = await build_prompt(
        query=request.question,
        context_chunks=scored_results,
        session_id=session_id,
        config=prompt_config,
    )
    
    # Generar respuesta
    answer = await generate_answer(
        query=request.question,
        context_chunks=scored_results,
        session_id=session_id,
        student_id=student_id,
        config=prompt_config,
    )
    
    # Evaluar
    evaluation = await evaluate_answer(
        query=request.question,
        answer=answer,
        context_chunks=scored_results,
    )
    
    # Reintentos
    retries = 0
    max_retries = 2
    
    while evaluation.decision in [
        ReflectionDecision.RETRY,
        ReflectionDecision.REQUEST_MORE_CONTEXT,
    ] and retries < max_retries:
        retries += 1
        prompt_config = get_retry_adjustments(evaluation, prompt_config)
        answer = await generate_answer(
            query=request.question,
            context_chunks=scored_results,
            session_id=session_id,
            student_id=student_id,
            config=prompt_config,
        )
        evaluation = await evaluate_answer(
            query=request.question,
            answer=answer,
            context_chunks=scored_results,
            previous_evaluation=evaluation,
        )
    
    processing_time = int((time.time() - start_time) * 1000)
    
    return QueryDebugResponse(
        answer=answer.answer,
        sources=answer.sources,
        concepts=answer.concepts_covered,
        confidence=answer.confidence,
        session_id=session_id,
        query_id=query_id,
        strategy_used=strategy_decision.strategy.value,
        cached=False,
        processing_time_ms=processing_time,
        metadata=answer.metadata,
        retrieval_results=[
            {
                "id": r.id,
                "score": r.final_score,
                "concepts": r.concepts[:5],
            }
            for r in scored_results[:10]
        ],
        strategy_decision=strategy_decision.to_dict(),
        evaluation=evaluation.to_dict(),
        prompt_used=prompt[:2000],  # Truncar
        retries=retries,
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
) -> Optional[QueryResponse]:
    """
    Busca respuesta en caché.
    """
    cache_key = _get_cache_key(question, student_id)
    
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
