"""
reasoning_agent.py
Agente de Razonamiento PASIVO - Solo genera texto.

Este agente es completamente PASIVO:
- Solo recibe contexto curado
- Solo recibe instrucción de modo explícita
- NO decide formato
- NO detecta errores
- NO evalúa resultados
- NO adapta pedagogía

Solo genera texto según lo que se le pide.
Reflection evalúa el resultado.
Orchestrator controla el flujo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from backend.agents.mode_router import LearningMode
from backend.agents.retrieval_agent import RetrievalResultItem

logger = logging.getLogger(__name__)


# ==========================================
# Configuración de Prompt
# ==========================================

@dataclass
class PromptConfig:
    """
    Configuración para generación de prompts.
    
    Esta configuración es CONSTRUIDA por el Orchestrator.
    Reasoning la usa sin modificarla.
    """
    language: str = "es"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Instrucciones explícitas (inyectadas por Orchestrator)
    mode_instruction: str = ""
    format_instruction: str = ""
    
    # Contexto de conversación
    include_conversation: bool = True
    conversation_turns: int = 3
    
    # Restricciones
    prohibitions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "include_conversation": self.include_conversation,
        }


@dataclass
class GeneratedAnswer:
    """
    Respuesta generada por el agente.
    
    NO contiene evaluaciones.
    Solo contiene el texto generado y metadata básica.
    """
    answer: str
    sources: List[str] = field(default_factory=list)
    concepts_covered: List[str] = field(default_factory=list)
    mode_used: LearningMode = LearningMode.CONCEPT
    
    # Metadata de generación
    tokens_used: int = 0
    model_used: str = ""
    generation_time_ms: int = 0
    
    # Para trazabilidad (NO para decisiones)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer[:500] + "..." if len(self.answer) > 500 else self.answer,
            "sources": self.sources,
            "concepts_covered": self.concepts_covered,
            "mode_used": self.mode_used.value,
            "tokens_used": self.tokens_used,
            "model_used": self.model_used,
        }


# ==========================================
# System Prompts por Modo
# ==========================================

LEARNING_MODE_PROMPTS = {
    LearningMode.CONCEPT: """Eres un tutor educativo experto.
Tu tarea es EXPLICAR CONCEPTOS de forma clara y pedagógica.

DEBES incluir:
- Definiciones claras
- Relaciones entre conceptos
- Ejemplos ilustrativos
- Contexto relevante

Usa lenguaje accesible y estructurado.
Responde en {language}.""",

    LearningMode.PRACTICE: """Eres un tutor educativo experto en resolución de problemas.
Tu tarea es EXPLICAR PASO A PASO cómo resolver ejercicios.

DEBES incluir:
- Identificación del problema
- Pasos numerados de resolución
- Justificación de cada paso
- Resultado final claro

Sé detallado y metódico.
Responde en {language}.""",

    LearningMode.EXERCISE_LIST: """Eres un asistente educativo.
Tu tarea es LISTAR ejercicios disponibles.

SOLO debes:
- Enumerar los ejercicios encontrados
- Incluir título y dificultad
- Agrupar por concepto si es posible

PROHIBIDO:
- Explicar cómo resolver los ejercicios
- Dar pistas o soluciones
- Agregar contenido explicativo

Solo lista. Nada más.
Responde en {language}.""",
}

# Formatos de respuesta por modo
MODE_FORMAT_INSTRUCTIONS = {
    LearningMode.CONCEPT: """
Formato de respuesta:
1. Introducción breve
2. Definición principal
3. Conceptos relacionados
4. Ejemplo o aplicación
5. Resumen (opcional)""",

    LearningMode.PRACTICE: """
Formato de respuesta:
1. Planteamiento del problema
2. Datos e incógnitas
3. Pasos de resolución (numerados)
4. Resultado final
5. Verificación (opcional)""",

    LearningMode.EXERCISE_LIST: """
Formato de respuesta:
Lista numerada de ejercicios:
- Número
- Título
- Dificultad (fácil/medio/difícil)
- Conceptos relacionados

NO incluir explicaciones ni soluciones.""",
}


# ==========================================
# Función Principal de Generación
# ==========================================

async def generate_answer(
    query: str,
    context_chunks: List[RetrievalResultItem],
    mode: LearningMode,
    session_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
) -> GeneratedAnswer:
    """
    Genera una respuesta usando el LLM.
    
    Este agente es PASIVO:
    - Recibe contexto curado (del Retrieval)
    - Recibe modo explícito (del Orchestrator)
    - Genera texto
    - NO toma decisiones pedagógicas
    
    Args:
        query: Pregunta del usuario
        context_chunks: Chunks de contexto (del Retrieval)
        mode: Modo pedagógico (del Orchestrator)
        session_id: ID de sesión para historial
        config: Configuración de prompt
        user_openai_key: API key de OpenAI
        user_google_key: API key de Google
        preferred_provider: Proveedor preferido
        user_model: Modelo específico
        
    Returns:
        GeneratedAnswer con el texto generado
    """
    import time
    start_time = time.time()
    
    config = config or PromptConfig()
    
    logger.info(f"Generating answer: mode={mode.value}, chunks={len(context_chunks)}")
    
    # Construir prompt completo
    system_prompt = _build_system_prompt(mode, config)
    user_prompt = _build_user_prompt(query, context_chunks, mode, config)
    
    # Agregar prohibiciones si las hay
    if config.prohibitions:
        prohibition_text = "\n\nPROHIBIDO:\n" + "\n".join(f"- {p}" for p in config.prohibitions)
        system_prompt += prohibition_text
    
    try:
        # Importar LLM
        from backend.models.llm import get_llm_with_user_keys
        
        # Obtener cliente LLM
        llm = get_llm_with_user_keys(
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
            preferred_provider=preferred_provider,
            user_model=user_model,
        )
        
        # Generar respuesta
        answer_text = await llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        
        tokens_used = 0  # No disponible directamente
        model_used = llm.model
        
    except ImportError:
        logger.warning("LLM module not available. Using mock response.")
        answer_text = _mock_generate(query, context_chunks, mode)
        tokens_used = 0
        model_used = "mock"
        
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        answer_text = _generate_error_response(mode, config.language)
        tokens_used = 0
        model_used = "error"
    
    generation_time = int((time.time() - start_time) * 1000)
    
    # Extraer fuentes y conceptos de los chunks
    sources = list(set(c.source for c in context_chunks if c.source))
    concepts = []
    for c in context_chunks:
        concepts.extend(c.concepts)
    concepts = list(set(concepts))[:10]
    
    return GeneratedAnswer(
        answer=answer_text,
        sources=sources,
        concepts_covered=concepts,
        mode_used=mode,
        tokens_used=tokens_used,
        model_used=model_used,
        generation_time_ms=generation_time,
        confidence=0.0,  # No evaluamos aquí, Reflection lo hace
        metadata={
            "chunk_count": len(context_chunks),
            "config": config.to_dict(),
        },
    )


async def generate_answer_stream(
    query: str,
    context_chunks: List[RetrievalResultItem],
    mode: LearningMode,
    session_id: Optional[str] = None,
    config: Optional[PromptConfig] = None,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Genera respuesta con streaming.
    
    Versión streaming de generate_answer.
    Misma lógica PASIVA.
    """
    config = config or PromptConfig()
    
    system_prompt = _build_system_prompt(mode, config)
    user_prompt = _build_user_prompt(query, context_chunks, mode, config)
    
    if config.prohibitions:
        prohibition_text = "\n\nPROHIBIDO:\n" + "\n".join(f"- {p}" for p in config.prohibitions)
        system_prompt += prohibition_text
    
    try:
        from backend.models.llm import get_llm_with_user_keys
        
        llm = get_llm_with_user_keys(
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
            preferred_provider=preferred_provider,
            user_model=user_model,
        )
        
        async for chunk in llm.generate_stream(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ):
            yield chunk
            
    except ImportError:
        logger.warning("LLM streaming not available. Using mock.")
        mock_response = _mock_generate(query, context_chunks, mode)
        for word in mock_response.split():
            yield word + " "
            
    except Exception as e:
        logger.error(f"Error in stream generation: {e}")
        yield _generate_error_response(mode, config.language)


# ==========================================
# Construcción de Prompts
# ==========================================

def _build_system_prompt(mode: LearningMode, config: PromptConfig) -> str:
    """Construye el system prompt según el modo."""
    base_prompt = LEARNING_MODE_PROMPTS.get(mode, LEARNING_MODE_PROMPTS[LearningMode.CONCEPT])
    base_prompt = base_prompt.format(language=config.language)
    
    # Agregar instrucciones de modo si las hay
    if config.mode_instruction:
        base_prompt += f"\n\nInstrucción adicional:\n{config.mode_instruction}"
    
    # Agregar formato
    format_instruction = MODE_FORMAT_INSTRUCTIONS.get(mode, "")
    if format_instruction:
        base_prompt += f"\n{format_instruction}"
    
    # Agregar instrucciones de formato custom si las hay
    if config.format_instruction:
        base_prompt += f"\n\n{config.format_instruction}"
    
    return base_prompt


def _build_user_prompt(
    query: str,
    context_chunks: List[RetrievalResultItem],
    mode: LearningMode,
    config: PromptConfig,
) -> str:
    """Construye el user prompt con contexto."""
    
    if mode == LearningMode.EXERCISE_LIST:
        # Para EXERCISE_LIST: Solo listar ejercicios
        return _build_exercise_list_prompt(query, context_chunks)
    
    # Para otros modos: Incluir contexto completo
    context_text = _format_context(context_chunks, mode)
    
    prompt = f"""Pregunta del estudiante:
{query}

Contexto disponible:
{context_text}

Responde según el modo indicado."""
    
    return prompt


def _build_exercise_list_prompt(
    query: str,
    context_chunks: List[RetrievalResultItem],
) -> str:
    """Construye prompt específico para EXERCISE_LIST."""
    
    exercises = []
    for i, chunk in enumerate(context_chunks, 1):
        title = chunk.exercise_title or f"Ejercicio {i}"
        difficulty = chunk.difficulty or "medio"
        concepts = ", ".join(chunk.concepts[:3]) if chunk.concepts else "general"
        
        exercises.append(f"- {title} (Dificultad: {difficulty}) - Conceptos: {concepts}")
    
    if not exercises:
        exercises = ["No se encontraron ejercicios para esta consulta."]
    
    return f"""Consulta del estudiante: {query}

Ejercicios encontrados:
{chr(10).join(exercises)}

Lista estos ejercicios de forma clara y organizada.
NO expliques cómo resolverlos. Solo lista."""


def _format_context(
    chunks: List[RetrievalResultItem],
    mode: LearningMode,
) -> str:
    """Formatea los chunks de contexto para el prompt."""
    if not chunks:
        return "No hay contexto disponible."
    
    formatted = []
    for i, chunk in enumerate(chunks[:8], 1):  # Max 8 chunks
        source = f" (Fuente: {chunk.source})" if chunk.source else ""
        concepts = f" [Conceptos: {', '.join(chunk.concepts[:3])}]" if chunk.concepts else ""
        
        # Truncar contenido muy largo
        content = chunk.content
        if len(content) > 800:
            content = content[:800] + "..."
        
        formatted.append(f"[{i}]{source}{concepts}\n{content}")
    
    return "\n\n".join(formatted)


# ==========================================
# Funciones Auxiliares
# ==========================================

def _mock_generate(
    query: str,
    context_chunks: List[RetrievalResultItem],
    mode: LearningMode,
) -> str:
    """Genera respuesta mock para testing."""
    
    if mode == LearningMode.EXERCISE_LIST:
        exercises = [c.exercise_title or f"Ejercicio {i+1}" for i, c in enumerate(context_chunks[:5])]
        return "Ejercicios disponibles:\n" + "\n".join(f"{i+1}. {e}" for i, e in enumerate(exercises))
    
    elif mode == LearningMode.PRACTICE:
        return f"""Para resolver este problema sobre "{query[:50]}":

1. Identificar los datos del problema
2. Aplicar la fórmula correspondiente
3. Sustituir valores
4. Calcular el resultado

Resultado: [Respuesta simulada]"""
    
    else:  # CONCEPT
        return f"""Explicación sobre "{query[:50]}":

Este concepto se refiere a... [contenido simulado].

Los aspectos principales son:
- Aspecto 1
- Aspecto 2
- Aspecto 3

En resumen, [resumen simulado]."""


def _generate_error_response(mode: LearningMode, language: str) -> str:
    """Genera respuesta de error genérica."""
    if language == "es":
        return "Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo."
    return "Sorry, there was an error generating the response. Please try again."


def get_fallback_message(mode: LearningMode, language: str = "es") -> str:
    """Obtiene mensaje de fallback según el modo."""
    fallbacks = {
        "es": {
            LearningMode.CONCEPT: "No pude encontrar información suficiente sobre este concepto.",
            LearningMode.PRACTICE: "No pude encontrar ejemplos resueltos para este ejercicio.",
            LearningMode.EXERCISE_LIST: "No encontré ejercicios disponibles sobre este tema.",
        },
        "en": {
            LearningMode.CONCEPT: "I couldn't find enough information about this concept.",
            LearningMode.PRACTICE: "I couldn't find worked examples for this exercise.",
            LearningMode.EXERCISE_LIST: "I couldn't find available exercises on this topic.",
        },
    }
    
    lang_fallbacks = fallbacks.get(language, fallbacks["es"])
    return lang_fallbacks.get(mode, lang_fallbacks[LearningMode.CONCEPT])


# ==========================================
# Exports
# ==========================================

__all__ = [
    "PromptConfig",
    "GeneratedAnswer",
    "generate_answer",
    "generate_answer_stream",
    "get_fallback_message",
    "LEARNING_MODE_PROMPTS",
    "MODE_FORMAT_INSTRUCTIONS",
]
