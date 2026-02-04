"""
reasoning_agent.py
Agente de Razonamiento - Genera respuestas finales con contexto curado.
Adapta el nivel pedagógico según el perfil del estudiante.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

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

class ResponseStyle(str, Enum):
    """Estilos de respuesta."""
    CONCISE = "concise"       # Breve y directo
    DETAILED = "detailed"     # Explicación completa
    STEP_BY_STEP = "step_by_step"  # Paso a paso
    SOCRATIC = "socratic"     # Guía con preguntas
    EXAMPLE_BASED = "example_based"  # Basado en ejemplos


@dataclass
class PromptConfig:
    """Configuración para construcción de prompts."""
    
    max_context_tokens: int = 3000
    max_conversation_tokens: int = 1000
    include_examples: bool = True
    include_sources: bool = True
    language: str = "es"
    style: ResponseStyle = ResponseStyle.DETAILED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_context_tokens": self.max_context_tokens,
            "max_conversation_tokens": self.max_conversation_tokens,
            "include_examples": self.include_examples,
            "include_sources": self.include_sources,
            "language": self.language,
            "style": self.style.value,
        }


@dataclass
class GeneratedAnswer:
    """Respuesta generada por el agente."""
    
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    concepts_covered: List[str] = field(default_factory=list)
    confidence: float = 0.0
    tokens_used: int = 0
    style_used: ResponseStyle = ResponseStyle.DETAILED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "concepts_covered": self.concepts_covered,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "style_used": self.style_used.value,
            "metadata": self.metadata,
        }


# ==========================================
# System Prompts
# ==========================================

SYSTEM_PROMPTS = {
    "es": {
        ResponseStyle.CONCISE: """Eres un asistente educativo que SOLO responde usando la información del contexto proporcionado.

REGLAS ESTRICTAS:
- ÚNICAMENTE usa información del contexto dado
- NO uses conocimiento externo ni inventes información
- Si el contexto NO contiene la información necesaria, responde: "No tengo información suficiente en los documentos para responder esta pregunta."
- Sé breve y directo
- Cita las fuentes [1], [2], etc. cuando uses información del contexto""",

        ResponseStyle.DETAILED: """Eres un asistente educativo que SOLO responde usando la información del contexto proporcionado.

REGLAS ESTRICTAS:
- ÚNICAMENTE usa información del contexto dado (documentos, grafos de conocimiento)
- NO uses conocimiento externo, pre-entrenado ni inventes información
- Si el contexto NO contiene la información necesaria, responde: "No tengo información suficiente en los documentos para responder esta pregunta. Te sugiero subir documentos relacionados con este tema."
- Explica los conceptos de forma clara y estructurada
- Cita las fuentes [1], [2], etc. cuando uses información
- Si hay relaciones entre conceptos en el contexto, explícalas
- NO supongas ni completes información que no esté explícita""",

        ResponseStyle.STEP_BY_STEP: """Eres un tutor educativo que SOLO responde usando la información del contexto proporcionado.

REGLAS ESTRICTAS:
- ÚNICAMENTE usa información del contexto dado
- NO uses conocimiento externo ni inventes información
- Si el contexto NO contiene la información necesaria, indícalo claramente
- Descompón el problema en pasos claros basándote SOLO en el contexto
- Numera cada paso
- Cita las fuentes [1], [2], etc.
- Si falta información para algún paso, dilo explícitamente""",

        ResponseStyle.SOCRATIC: """Eres un tutor que usa el método socrático, pero SOLO con información del contexto proporcionado.

REGLAS ESTRICTAS:
- ÚNICAMENTE usa información del contexto dado
- NO uses conocimiento externo ni inventes información
- Guía al estudiante con preguntas basadas SOLO en el contexto disponible
- Si el contexto no tiene información suficiente, indícalo
- Las preguntas deben poder responderse con el contexto dado""",

        ResponseStyle.EXAMPLE_BASED: """Eres un educador que enseña mediante ejemplos, usando SOLO el contexto proporcionado.

REGLAS ESTRICTAS:
- ÚNICAMENTE usa información y ejemplos del contexto dado
- NO uses conocimiento externo ni inventes ejemplos
- Si el contexto NO contiene ejemplos o información suficiente, indícalo
- Cita las fuentes [1], [2], etc.
- NO crees ejemplos ficticios - solo usa los del contexto""",
    },
    "en": {
        ResponseStyle.CONCISE: """You are an educational assistant that ONLY responds using information from the provided context.

STRICT RULES:
- ONLY use information from the given context
- DO NOT use external knowledge or invent information
- If the context does NOT contain the necessary information, respond: "I don't have enough information in the documents to answer this question."
- Be brief and direct
- Cite sources [1], [2], etc. when using context information""",

        ResponseStyle.DETAILED: """You are an educational assistant that ONLY responds using information from the provided context.

STRICT RULES:
- ONLY use information from the given context (documents, knowledge graphs)
- DO NOT use external, pre-trained knowledge or invent information
- If the context does NOT contain the necessary information, respond: "I don't have enough information in the documents to answer this question. I suggest uploading related documents."
- Explain concepts clearly and in a structured manner
- Cite sources [1], [2], etc. when using information
- If there are relationships between concepts in the context, explain them
- DO NOT assume or complete information that is not explicit""",

        ResponseStyle.STEP_BY_STEP: """You are an educational tutor that ONLY responds using information from the provided context.

STRICT RULES:
- ONLY use information from the given context
- DO NOT use external knowledge or invent information
- If the context does NOT contain the necessary information, clearly indicate it
- Break down the problem into clear steps based ONLY on the context
- Number each step
- Cite sources [1], [2], etc.
- If information is missing for any step, say so explicitly""",

        ResponseStyle.SOCRATIC: """You are a tutor using the Socratic method, but ONLY with information from the provided context.

STRICT RULES:
- ONLY use information from the given context
- DO NOT use external knowledge or invent information
- Guide the student with questions based ONLY on available context
- If the context lacks sufficient information, indicate it
- Questions must be answerable with the given context""",

        ResponseStyle.EXAMPLE_BASED: """You are an educator who teaches through examples, using ONLY the provided context.

STRICT RULES:
- ONLY use information and examples from the given context
- DO NOT use external knowledge or invent examples
- If the context does NOT contain examples or sufficient information, indicate it
- Cite sources [1], [2], etc.
- DO NOT create fictional examples - only use those from the context""",
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
) -> str:
    """
    Construye prompt completo con contexto recuperado.
    
    Args:
        query: Pregunta del estudiante
        context_chunks: Chunks de contexto recuperados
        session_id: ID de sesión para historial
        config: Configuración del prompt
        
    Returns:
        Prompt construido
    """
    config = config or PromptConfig()
    
    parts: List[str] = []
    
    # 1. Contexto recuperado
    if context_chunks:
        context_text = _format_context(context_chunks, config.max_context_tokens)
        parts.append(f"### Contexto Relevante:\n{context_text}")
    
    # 2. Historial de conversación
    if session_id:
        conversation = await get_context_for_llm(
            session_id, 
            max_tokens=config.max_conversation_tokens
        )
        
        if conversation:
            conv_text = _format_conversation(conversation)
            parts.append(f"### Conversación Previa:\n{conv_text}")
    
    # 3. Pregunta actual
    parts.append(f"### Pregunta del Estudiante:\n{query}")
    
    # 4. Instrucciones adicionales
    if config.include_sources:
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
        
    Returns:
        GeneratedAnswer con la respuesta
    """
    config = config or PromptConfig()
    
    # Adaptar según perfil del estudiante
    if student_id:
        config = await adapt_to_student(student_id, config)
    
    # Construir prompt
    prompt = await build_prompt(
        query=query,
        context_chunks=context_chunks,
        session_id=session_id,
        config=config,
    )
    
    # Obtener system prompt según estilo
    lang = config.language if config.language in SYSTEM_PROMPTS else "es"
    system_prompt = SYSTEM_PROMPTS[lang].get(
        config.style, 
        SYSTEM_PROMPTS[lang][ResponseStyle.DETAILED]
    )
    
    # Generar respuesta (siempre usar generate_with_user_keys para lógica unificada)
    try:
        answer_text = await generate_with_user_keys(
            prompt=prompt,
            system_prompt=system_prompt,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
            preferred_provider=preferred_provider,
            user_model=user_model,
            temperature=0.7,
            max_tokens=2048,
        )
        
        # Extraer fuentes y conceptos
        sources = [chunk.id for chunk in context_chunks[:5]]
        concepts = _extract_concepts_from_chunks(context_chunks)
        
        # Calcular confianza basada en contexto
        confidence = _calculate_confidence(context_chunks, answer_text)
        
        return GeneratedAnswer(
            answer=clean_whitespace(answer_text),
            sources=sources,
            concepts_covered=concepts,
            confidence=confidence,
            tokens_used=token_count(prompt) + token_count(answer_text),
            style_used=config.style,
            metadata={
                "context_chunks": len(context_chunks),
                "language": config.language,
            },
        )
        
    except Exception as e:
        return GeneratedAnswer(
            answer=f"Lo siento, hubo un error generando la respuesta: {str(e)}",
            confidence=0.0,
            metadata={"error": str(e)},
        )


async def generate_answer_stream(
    query: str,
    context_chunks: List[RankedResult],
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
) -> AsyncGenerator[str, None]:
    """
    Genera respuesta en streaming.
    
    Args:
        query: Pregunta del estudiante
        context_chunks: Contexto recuperado
        session_id: ID de sesión
        student_id: ID del estudiante
        config: Configuración
        
    Yields:
        Chunks de texto de la respuesta
    """
    config = config or PromptConfig()
    
    if student_id:
        config = await adapt_to_student(student_id, config)
    
    prompt = await build_prompt(
        query=query,
        context_chunks=context_chunks,
        session_id=session_id,
        config=config,
    )
    
    lang = config.language if config.language in SYSTEM_PROMPTS else "es"
    system_prompt = SYSTEM_PROMPTS[lang].get(
        config.style,
        SYSTEM_PROMPTS[lang][ResponseStyle.DETAILED]
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
    
    # Ajustar estilo según nivel
    if profile.level == ProficiencyLevel.BEGINNER:
        config.style = ResponseStyle.STEP_BY_STEP
        config.include_examples = True
        config.max_context_tokens = 2000  # Menos contexto, más explicación
        
    elif profile.level == ProficiencyLevel.ELEMENTARY:
        config.style = ResponseStyle.EXAMPLE_BASED
        config.include_examples = True
        
    elif profile.level == ProficiencyLevel.INTERMEDIATE:
        config.style = ResponseStyle.DETAILED
        config.include_examples = True
        
    elif profile.level == ProficiencyLevel.ADVANCED:
        config.style = ResponseStyle.DETAILED
        config.include_examples = False
        config.max_context_tokens = 4000  # Más contexto
        
    elif profile.level == ProficiencyLevel.EXPERT:
        config.style = ResponseStyle.CONCISE
        config.include_examples = False
        config.max_context_tokens = 4000
    
    # Ajustar según estilo de aprendizaje
    if profile.learning_style.value == "visual":
        # Podríamos incluir instrucciones para diagramas
        pass
    elif profile.learning_style.value == "kinesthetic":
        config.style = ResponseStyle.STEP_BY_STEP
    
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
) -> float:
    """
    Calcula confianza de la respuesta.
    
    Factores:
    - Scores de los chunks
    - Cantidad de chunks usados
    - Longitud de la respuesta
    """
    if not chunks:
        return 0.3
    
    # Factor 1: Score promedio de chunks
    avg_chunk_score = sum(c.final_score for c in chunks) / len(chunks)
    
    # Factor 2: Cantidad de chunks (más = mejor hasta cierto punto)
    chunk_factor = min(1.0, len(chunks) / 5)
    
    # Factor 3: Respuesta no vacía
    answer_factor = 0.9 if len(answer) > 100 else 0.5
    
    # Combinar factores
    confidence = (avg_chunk_score * 0.5 + chunk_factor * 0.3 + answer_factor * 0.2)
    
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
