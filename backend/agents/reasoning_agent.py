"""
reasoning_agent.py
Agente de Razonamiento - Genera respuestas finales con contexto curado.
Adapta el nivel pedagógico según el perfil del estudiante.
Soporta modos pedagógicos: CONCEPT, PRACTICE, EXERCISE_LIST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.agents.mode_router import LearningMode
from backend.memory.session_memory import (
    TurnRole,
    get_context_for_llm,
    get_active_concepts,
)
from backend.memory.student_profile import (
    ProficiencyLevel,
    StudentProfile,
    load_profile,
    get_level_config,
)
from backend.models.llm import (
    generate, 
    generate_stream, 
    get_llm,
    generate_with_user_keys,
)
from backend.retrieval.hybrid_ranker import RankedResult
from backend.utils.text import truncate_context, token_count, clean_whitespace


# ==========================================
# Configuración y Estructuras
# ==========================================


@dataclass
class PromptConfig:
    """Configuración para construcción de prompts."""
    
    max_context_tokens: int = 3000
    max_conversation_tokens: int = 4000
    include_examples: bool = True
    include_sources: bool = True
    language: str = "es"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "max_conversation_tokens": self.max_conversation_tokens,
            "include_examples": self.include_examples,
            "include_sources": self.include_sources,
            "language": self.language,
        }


@dataclass
class GeneratedAnswer:
    """Respuesta generada por el agente."""
    
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    concepts_covered: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tokens_used: int = 0
    learning_mode: LearningMode = LearningMode.CONCEPT
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "concepts_covered": self.concepts_covered,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "learning_mode": self.learning_mode.value,
            "metadata": self.metadata,
        }


# ==========================================
# Prompts por Modo Pedagógico
# ==========================================

LEARNING_MODE_PROMPTS = {
    "es": {
        LearningMode.CONCEPT: """Eres un tutor educativo especializado en EXPLICAR CONCEPTOS Y TEORÍA.

MODO: CONCEPTO - Tu objetivo es explicar definiciones, teoría y relaciones entre conceptos.

REGLAS ESTRICTAS:
1. SOLO usa información del contexto proporcionado
2. NO uses conocimiento externo ni inventes información
3. Explica DEFINICIONES de forma clara y precisa
4. Describe RELACIONES entre conceptos cuando existan en el contexto
5. Menciona PRERREQUISITOS si están disponibles
6. Si falta información, indica: "No tengo información suficiente sobre este concepto en los documentos."
7. Cita las fuentes [1], [2], etc.

ESTRUCTURA DE RESPUESTA:
- Definición principal
- Características clave
- Relaciones con otros conceptos (si aplica)
- Ejemplos del contexto (si existen)""",

        LearningMode.PRACTICE: """Eres un tutor educativo especializado en RESOLVER EJERCICIOS PASO A PASO.

MODO: PRÁCTICA - Tu objetivo es explicar la resolución de ejercicios de forma detallada.

REGLAS ESTRICTAS:
1. SOLO usa información y métodos del contexto proporcionado
2. NO inventes procedimientos ni uses conocimiento externo
3. Explica CADA PASO de la solución claramente
4. Justifica POR QUÉ se realiza cada paso
5. Muestra el RESULTADO FINAL claramente
6. Si el contexto no tiene suficiente información para resolver, indícalo
7. Cita las fuentes [1], [2], etc.

ESTRUCTURA DE RESPUESTA:
- Identificación del problema
- Datos e incógnitas
- Pasos de resolución (numerados)
- Resultado final
- Verificación (si aplica)""",

        LearningMode.EXERCISE_LIST: """Eres un asistente que LISTA EJERCICIOS disponibles.

MODO: LISTA DE EJERCICIOS - Tu objetivo es SOLO enumerar ejercicios, SIN explicarlos.

REGLAS CRÍTICAS:
1. SOLO lista ejercicios que existan en el contexto
2. NO EXPLIQUES ni resuelvas los ejercicios
3. NO INVENTES ejercicios que no estén en los documentos
4. Para cada ejercicio incluye SOLO:
   - Nombre/Título
   - Dificultad (si está disponible)
   - Concepto relacionado
   - ID de referencia

FORMATO OBLIGATORIO:
📝 **[Título del Ejercicio]**
   - Dificultad: [nivel]
   - Concepto: [concepto relacionado]
   - Ref: [id]

Si NO hay ejercicios en el contexto, responde:
"No encontré ejercicios en los documentos cargados. Te sugiero subir material con ejercicios prácticos."

PROHIBIDO:
❌ Explicar cómo resolver
❌ Dar pistas o pasos
❌ Inventar ejercicios
❌ Resumir contenido""",
    },
    "en": {
        LearningMode.CONCEPT: """You are an educational tutor specialized in EXPLAINING CONCEPTS AND THEORY.

MODE: CONCEPT - Your goal is to explain definitions, theory, and relationships between concepts.

STRICT RULES:
1. ONLY use information from the provided context
2. DO NOT use external knowledge or invent information
3. Explain DEFINITIONS clearly and precisely
4. Describe RELATIONSHIPS between concepts when they exist in context
5. Mention PREREQUISITES if available
6. If information is missing, indicate: "I don't have enough information about this concept in the documents."
7. Cite sources [1], [2], etc.

RESPONSE STRUCTURE:
- Main definition
- Key characteristics
- Relationships with other concepts (if applicable)
- Examples from context (if they exist)""",

        LearningMode.PRACTICE: """You are an educational tutor specialized in SOLVING EXERCISES STEP BY STEP.

MODE: PRACTICE - Your goal is to explain exercise solutions in detail.

STRICT RULES:
1. ONLY use information and methods from the provided context
2. DO NOT invent procedures or use external knowledge
3. Explain EACH STEP of the solution clearly
4. Justify WHY each step is performed
5. Show the FINAL RESULT clearly
6. If context lacks sufficient information to solve, indicate it
7. Cite sources [1], [2], etc.

RESPONSE STRUCTURE:
- Problem identification
- Data and unknowns
- Resolution steps (numbered)
- Final result
- Verification (if applicable)""",

        LearningMode.EXERCISE_LIST: """You are an assistant that LISTS available exercises.

MODE: EXERCISE LIST - Your goal is to ONLY enumerate exercises, WITHOUT explaining them.

CRITICAL RULES:
1. ONLY list exercises that exist in the context
2. DO NOT EXPLAIN or solve exercises
3. DO NOT INVENT exercises that are not in the documents
4. For each exercise include ONLY:
   - Name/Title
   - Difficulty (if available)
   - Related concept
   - Reference ID

MANDATORY FORMAT:
📝 **[Exercise Title]**
   - Difficulty: [level]
   - Concept: [related concept]
   - Ref: [id]

If there are NO exercises in the context, respond:
"I didn't find exercises in the loaded documents. I suggest uploading material with practical exercises."

FORBIDDEN:
❌ Explaining how to solve
❌ Giving hints or steps
❌ Inventing exercises
❌ Summarizing content""",
    }
}


# ==========================================
# Construcción de Prompts
# ==========================================

async def build_prompt(
    query: str,
    context_chunks: List[RankedResult],
    session_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> str:
    """
    Construye prompt completo con contexto recuperado.
    
    Args:
        query: Pregunta del estudiante
        context_chunks: Chunks de contexto recuperados
        session_id: ID de sesión para historial
        config: Configuración del prompt
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        Prompt construido
    """
    config = config or PromptConfig()
    
    parts: List[str] = []
    
    # 1. Contexto recuperado (formateo diferente según modo)
    if context_chunks:
        if mode == LearningMode.EXERCISE_LIST:
            context_text = _format_exercise_list_context(context_chunks)
            parts.append(f"### Ejercicios Disponibles:\n{context_text}")
        else:
            context_text = _format_context(context_chunks, config.max_context_tokens)
            parts.append(f"### Contexto Relevante:\n{context_text}")
    
    # 2. Historial de conversación (menos relevante para EXERCISE_LIST)
    if session_id and mode != LearningMode.EXERCISE_LIST:
        conversation = await get_context_for_llm(
            session_id, 
            max_tokens=config.max_conversation_tokens
        )
        
        if conversation:
            conv_text = _format_conversation(conversation)
            parts.append(f"### Conversación Previa:\n{conv_text}")
    
    # 3. Pregunta actual
    parts.append(f"### Pregunta del Estudiante:\n{query}")
    
    # 4. Instrucciones adicionales según modo
    if mode == LearningMode.EXERCISE_LIST:
        parts.append("\n(Lista SOLO los ejercicios encontrados. NO expliques ni resuelvas.)")
    elif mode == LearningMode.PRACTICE:
        parts.append("\n(Explica la resolución paso a paso. Cita las fuentes.)")
    elif config.include_sources:
        parts.append("\n(Menciona las fuentes relevantes en tu respuesta)")
    
    return "\n\n".join(parts)


def _format_context(
    chunks: List[RankedResult],
    max_tokens: int,
) -> str:
    """Formatea chunks de contexto."""
    formatted_parts: List[str] = []
    current_tokens = 0
    
    for i, chunk in enumerate(chunks, 1):
        chunk_text = f"[{i}] {chunk.content}"
        chunk_tokens = token_count(chunk_text)
        
        if current_tokens + chunk_tokens > max_tokens:
            # Truncar este chunk si es necesario
            remaining = max_tokens - current_tokens - 50
            if remaining > 100:
                chunk_text = truncate_context(chunk_text, remaining)
                formatted_parts.append(chunk_text)
            break
        
        formatted_parts.append(chunk_text)
        current_tokens += chunk_tokens
    
    return "\n\n".join(formatted_parts)


def _format_exercise_list_context(chunks: List[RankedResult]) -> str:
    """
    Formatea chunks para modo EXERCISE_LIST.
    Solo incluye metadatos de ejercicios, sin contenido explicativo.
    """
    formatted_parts: List[str] = []
    
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata or {}
        
        # Extraer información del ejercicio
        title = metadata.get("title", metadata.get("name", f"Ejercicio {i}"))
        difficulty = metadata.get("difficulty", metadata.get("nivel", "no especificado"))
        concepts = ", ".join(chunk.concepts) if chunk.concepts else metadata.get("concept", "general")
        ref_id = chunk.id or metadata.get("source_id", f"ref_{i}")
        exercise_type = metadata.get("type", metadata.get("chunk_type", "ejercicio"))
        
        # Formato estructurado para el LLM
        exercise_info = f"""[{i}] EJERCICIO:
- Título: {title}
- Dificultad: {difficulty}
- Concepto: {concepts}
- Tipo: {exercise_type}
- Referencia: {ref_id}"""
        
        formatted_parts.append(exercise_info)
    
    if not formatted_parts:
        return "No se encontraron ejercicios en el contexto."
    
    return "\n\n".join(formatted_parts)


def _format_conversation(messages: List[Dict[str, str]]) -> str:
    """Formatea historial de conversación."""
    formatted: List[str] = []
    
    for msg in messages[-6:]:  # Últimos 6 mensajes
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        prefix = "Estudiante" if role == "user" else "Asistente"
        # Truncar mensajes largos
        if len(content) > 500:
            content = content[:500] + "..."
        
        formatted.append(f"{prefix}: {content}")
    
    return "\n".join(formatted)


# ==========================================
# Generación de Respuestas
# ==========================================

async def generate_answer(
    query: str,
    context_chunks: List[RankedResult],
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> GeneratedAnswer:
    """
    Genera respuesta final usando el LLM.
    
    Args:
        query: Pregunta del estudiante
        context_chunks: Contexto recuperado
        session_id: ID de sesión
        student_id: ID del estudiante (para adaptación)
        config: Configuración del prompt
        user_openai_key: API key de OpenAI del usuario (opcional)
        user_google_key: API key de Google del usuario (opcional)
        preferred_provider: Proveedor preferido ('openai' o 'google')
        user_model: Modelo específico a usar (ej: 'gpt-4', 'gemini-pro')
        mode: Modo de aprendizaje pedagógico
        
    Returns:
        GeneratedAnswer con la respuesta
    """
    config = config or PromptConfig()
    
    # Adaptar según perfil del estudiante (no para EXERCISE_LIST)
    if student_id and mode != LearningMode.EXERCISE_LIST:
        config = await adapt_to_student(student_id, config)
    
    # Construir prompt con modo
    prompt = await build_prompt(
        query=query,
        context_chunks=context_chunks,
        session_id=session_id,
        config=config,
        mode=mode,
    )
    
    # Obtener system prompt por modo pedagógico
    lang = config.language if config.language in LEARNING_MODE_PROMPTS else "es"
    
    # Usar prompt de modo pedagógico (siempre disponible para cada modo)
    system_prompt = LEARNING_MODE_PROMPTS.get(lang, LEARNING_MODE_PROMPTS["es"]).get(
        mode,
        LEARNING_MODE_PROMPTS["es"][LearningMode.CONCEPT]  # Fallback a CONCEPT
    )
    
    # Generar respuesta (siempre usar generate_with_user_keys para lógica unificada)
    try:
        # Ajustar temperatura según modo
        temperature = 0.3 if mode == LearningMode.EXERCISE_LIST else 0.7
        max_tokens = 1024 if mode == LearningMode.EXERCISE_LIST else 2048
        
        answer_text = await generate_with_user_keys(
            prompt=prompt,
            system_prompt=system_prompt,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
            preferred_provider=preferred_provider,
            user_model=user_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Extraer fuentes legibles y conceptos
        sources = _format_sources(context_chunks[:5])
        concepts = _extract_concepts_from_chunks(context_chunks)
        
        # Calcular confianza basada en contexto y modo
        confidence = _calculate_confidence(context_chunks, answer_text, mode)
        
        return GeneratedAnswer(
            answer=clean_whitespace(answer_text),
            sources=sources,
            concepts_covered=concepts,
            confidence=confidence,
            tokens_used=token_count(prompt) + token_count(answer_text),
            learning_mode=mode,
            metadata={
                "context_chunks": len(context_chunks),
                "language": config.language,
                "learning_mode": mode.value,
            },
        )
        
    except Exception as e:
        return GeneratedAnswer(
            answer=f"Lo siento, hubo un error generando la respuesta: {str(e)}",
            confidence=0.0,
            learning_mode=mode,
            metadata={"error": str(e)},
        )


async def generate_answer_stream(
    query: str,
    context_chunks: List[RankedResult],
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
    mode: LearningMode = LearningMode.CONCEPT,
) -> AsyncGenerator[str, None]:
    """
    Genera respuesta en streaming.
    
    Args:
        query: Pregunta del estudiante
        context_chunks: Contexto recuperado
        session_id: ID de sesión
        student_id: ID del estudiante
        config: Configuración
        mode: Modo de aprendizaje pedagógico
        
    Yields:
        Chunks de texto de la respuesta
    """
    config = config or PromptConfig()
    
    if student_id and mode != LearningMode.EXERCISE_LIST:
        config = await adapt_to_student(student_id, config)
    
    prompt = await build_prompt(
        query=query,
        context_chunks=context_chunks,
        session_id=session_id,
        config=config,
        mode=mode,
    )
    
    # Obtener system prompt por modo pedagógico
    lang = config.language if config.language in LEARNING_MODE_PROMPTS else "es"
    
    system_prompt = LEARNING_MODE_PROMPTS.get(lang, LEARNING_MODE_PROMPTS["es"]).get(
        mode,
        LEARNING_MODE_PROMPTS["es"][LearningMode.CONCEPT]
    )
    
    async for chunk in generate_stream(prompt, system_prompt):
        yield chunk


# ==========================================
# Adaptación al Estudiante
# ==========================================

async def adapt_to_student(
    student_id: str,
    config: PromptConfig,
) -> PromptConfig:
    """
    Ajusta configuración según nivel y preferencias del estudiante.
    
    Args:
        student_id: ID del estudiante
        config: Configuración base
        
    Returns:
        Configuración adaptada
    """
    profile = await load_profile(student_id)
    
    if not profile:
        return config
    
    # Obtener config según nivel
    level_config = get_level_config(profile.level)
    
    # Ajustar configuración según nivel de competencia
    if profile.level == ProficiencyLevel.BEGINNER:
        config.include_examples = True
        config.max_context_tokens = 2000  # Menos contexto, más explicación
        
    elif profile.level == ProficiencyLevel.ELEMENTARY:
        config.include_examples = True
        
    elif profile.level == ProficiencyLevel.INTERMEDIATE:
        config.include_examples = True
        
    elif profile.level == ProficiencyLevel.ADVANCED:
        config.include_examples = False
        config.max_context_tokens = 4000  # Más contexto
        
    elif profile.level == ProficiencyLevel.EXPERT:
        config.include_examples = False
        config.max_context_tokens = 4000
    
    return config


def adapt_response_complexity(
    response: str,
    profile: StudentProfile,
) -> str:
    """
    Ajusta complejidad de respuesta post-generación.
    
    Args:
        response: Respuesta generada
        profile: Perfil del estudiante
        
    Returns:
        Respuesta ajustada
    """
    # Por ahora retorna sin modificar
    # En el futuro podría simplificar vocabulario, etc.
    return response


# ==========================================
# Utilidades
# ==========================================

def _format_sources(chunks: List[RankedResult]) -> List[str]:
    """
    Formatea fuentes legibles para mostrar al usuario.
    
    En lugar de IDs de chunks, muestra:
    - Nombre del documento/fuente
    - Posición (página, sección, o índice de chunk)
    
    Args:
        chunks: Lista de chunks usados como contexto
        
    Returns:
        Lista de strings legibles con las fuentes
    """
    sources: List[str] = []
    seen_sources: set = set()
    
    for chunk in chunks:
        metadata = chunk.metadata or {}
        
        # Obtener título del documento
        source_title = (
            metadata.get("source_title") or
            metadata.get("title") or
            metadata.get("file_name") or
            metadata.get("name") or
            chunk.source_id or
            "Documento"
        )
        
        # Obtener posición en el documento
        page = metadata.get("page") or metadata.get("page_number")
        section = metadata.get("section") or metadata.get("heading")
        chunk_index = metadata.get("chunk_index", metadata.get("index"))
        line_start = metadata.get("line_start") or metadata.get("start_line")
        
        # Construir indicador de posición
        position_parts: List[str] = []
        if page:
            position_parts.append(f"pág. {page}")
        if section:
            position_parts.append(f"§ {section}")
        if line_start:
            position_parts.append(f"línea {line_start}")
        elif chunk_index is not None:
            position_parts.append(f"fragmento {chunk_index + 1}")
        
        # Formatear fuente
        if position_parts:
            source_str = f"{source_title} ({', '.join(position_parts)})"
        else:
            source_str = source_title
        
        # Evitar duplicados
        source_key = f"{source_title}:{page or chunk_index or 0}"
        if source_key not in seen_sources:
            sources.append(source_str)
            seen_sources.add(source_key)
    
    return sources


def _extract_concepts_from_chunks(chunks: List[RankedResult]) -> List[str]:
    """Extrae conceptos únicos de los chunks."""
    concepts: List[str] = []
    seen: set = set()
    
    for chunk in chunks:
        for concept in chunk.concepts:
            if concept not in seen:
                concepts.append(concept)
                seen.add(concept)
    
    return concepts[:10]  # Limitar a 10


def _calculate_confidence(
    chunks: List[RankedResult],
    answer: str,
    mode: LearningMode = LearningMode.CONCEPT,
) -> float:
    """
    Calcula confianza de la respuesta.
    
    Factores:
    - Scores de los chunks
    - Cantidad de chunks usados
    - Longitud de la respuesta
    - Alineación con el modo pedagógico
    """
    if not chunks:
        return 0.3
    
    # Factor 1: Score promedio de chunks
    avg_chunk_score = sum(c.final_score for c in chunks) / len(chunks)
    
    # Factor 2: Cantidad de chunks (más = mejor hasta cierto punto)
    chunk_factor = min(1.0, len(chunks) / 5)
    
    # Factor 3: Respuesta no vacía (ajustado por modo)
    if mode == LearningMode.EXERCISE_LIST:
        # Para lista de ejercicios, respuestas más cortas son aceptables
        answer_factor = 0.9 if len(answer) > 50 else 0.5
    else:
        answer_factor = 0.9 if len(answer) > 100 else 0.5
    
    # Factor 4: Alineación con modo
    mode_factor = 1.0
    if mode == LearningMode.EXERCISE_LIST:
        # Verificar que no haya explicaciones largas
        if "paso" in answer.lower() or "solución" in answer.lower():
            mode_factor = 0.7  # Penalizar si explica en modo lista
    elif mode == LearningMode.PRACTICE:
        # Verificar que haya pasos
        if "1." in answer or "paso" in answer.lower():
            mode_factor = 1.1  # Bonus si tiene estructura de pasos
    
    # Combinar factores
    confidence = (
        avg_chunk_score * 0.4 + 
        chunk_factor * 0.25 + 
        answer_factor * 0.2 +
        (mode_factor - 1.0) * 0.15 + 0.15
    )
    
    return min(1.0, max(0.0, confidence))


async def get_suggested_followups(
    query: str,
    answer: str,
    concepts: List[str],
    student_id: Optional[str] = None,
) -> List[str]:
    """
    Genera sugerencias de preguntas de seguimiento.
    
    Args:
        query: Pregunta original
        answer: Respuesta dada
        concepts: Conceptos cubiertos
        student_id: ID del estudiante
        
    Returns:
        Lista de preguntas sugeridas
    """
    # Por ahora retorna sugerencias estáticas basadas en conceptos
    suggestions: List[str] = []
    
    if concepts:
        suggestions.append(f"¿Puedes explicar más sobre {concepts[0]}?")
        
        if len(concepts) > 1:
            suggestions.append(
                f"¿Cuál es la relación entre {concepts[0]} y {concepts[1]}?"
            )
    
    suggestions.append("¿Puedes darme un ejemplo práctico?")
    
    return suggestions[:3]
