"""
mode_specs.py
Especificaciones pedagógicas explícitas para cada modo de aprendizaje.

Este archivo define los REQUISITOS y RESTRICCIONES de cada modo.
Reflection usa estas especificaciones para evaluar.
Reasoning NO las usa directamente (solo recibe prompts).

MODOS:
- CONCEPT: Explicar definiciones, teoría, relaciones entre conceptos
- PRACTICE: Resolver ejercicios paso a paso con justificación
- EXERCISE_LIST: Listar ejercicios disponibles SIN explicación
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.agents.mode_router import LearningMode


# ==========================================
# Especificación de Modo Pedagógico
# ==========================================

@dataclass(frozen=True)
class ContentRequirement:
    """Requisito de contenido para un modo."""
    name: str
    description: str
    required: bool = True
    weight: float = 1.0  # Peso en evaluación
    detection_markers: Tuple[str, ...] = field(default_factory=tuple)
    
    def is_present(self, text: str) -> bool:
        """Verifica si el requisito está presente en el texto."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.detection_markers)


@dataclass(frozen=True)
class ContentProhibition:
    """Prohibición de contenido para un modo."""
    name: str
    description: str
    detection_markers: Tuple[str, ...] = field(default_factory=tuple)
    penalty: float = 0.3  # Penalización si se detecta
    
    def is_violated(self, text: str) -> bool:
        """Verifica si la prohibición fue violada."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.detection_markers)


@dataclass(frozen=True)
class RetrievalSpec:
    """Especificación de retrieval para un modo."""
    chunk_type_filter: Optional[str] = None
    metadata_only: bool = False
    expand_concepts: bool = True
    top_k: int = 10
    min_score: float = 0.3
    vector_weight: float = 0.35
    bm25_weight: float = 0.25
    graph_weight: float = 0.4


@dataclass(frozen=True)
class PromptSpec:
    """Especificación de prompt para un modo."""
    system_prompt_key: str
    max_temperature: float = 0.7
    max_tokens: int = 2048
    include_conversation: bool = True
    format_instructions: str = ""


@dataclass
class ModeSpec:
    """
    Especificación completa de un modo pedagógico.
    
    Define:
    - Qué debe contener la respuesta (requirements)
    - Qué está prohibido (prohibitions)
    - Cómo hacer retrieval (retrieval_spec)
    - Cómo generar el prompt (prompt_spec)
    """
    mode: LearningMode
    name: str
    description: str
    
    # Requisitos de contenido
    requirements: List[ContentRequirement] = field(default_factory=list)
    
    # Prohibiciones de contenido  
    prohibitions: List[ContentProhibition] = field(default_factory=list)
    
    # Especificaciones técnicas
    retrieval_spec: RetrievalSpec = field(default_factory=RetrievalSpec)
    prompt_spec: PromptSpec = field(default_factory=lambda: PromptSpec(system_prompt_key="default"))
    
    # Metadatos
    evaluation_weights: Dict[str, float] = field(default_factory=dict)
    
    def validate_response(self, response: str) -> Tuple[float, List[str], List[str]]:
        """
        Valida una respuesta contra las especificaciones del modo.
        
        Args:
            response: Texto de la respuesta a validar
            
        Returns:
            Tupla (score, issues, suggestions)
        """
        score = 1.0
        issues: List[str] = []
        suggestions: List[str] = []
        
        # Verificar requisitos
        total_weight = sum(r.weight for r in self.requirements if r.required)
        for req in self.requirements:
            if req.required and not req.is_present(response):
                penalty = (req.weight / total_weight) * 0.4 if total_weight > 0 else 0.2
                score -= penalty
                issues.append(f"Falta: {req.name}")
                suggestions.append(f"Incluir: {req.description}")
        
        # Verificar prohibiciones
        for prohibition in self.prohibitions:
            if prohibition.is_violated(response):
                score -= prohibition.penalty
                issues.append(f"Prohibido: {prohibition.name}")
                suggestions.append(f"Eliminar: {prohibition.description}")
        
        return max(0.0, min(1.0, score)), issues, suggestions


# ==========================================
# Especificación: CONCEPT
# ==========================================

CONCEPT_SPEC = ModeSpec(
    mode=LearningMode.CONCEPT,
    name="Modo Concepto",
    description="Explicar definiciones, teoría y relaciones entre conceptos",
    
    requirements=[
        ContentRequirement(
            name="definición",
            description="Definición clara del concepto",
            required=True,
            weight=1.5,
            detection_markers=(
                "es", "se define", "significa", "consiste en", 
                "se refiere", "representa", "es un", "es una",
            ),
        ),
        ContentRequirement(
            name="explicación",
            description="Explicación del concepto",
            required=True,
            weight=1.0,
            detection_markers=(
                "porque", "debido", "por lo tanto", "esto significa",
                "es decir", "en otras palabras", "implica",
            ),
        ),
        ContentRequirement(
            name="relaciones",
            description="Relaciones con otros conceptos (si aplica)",
            required=False,
            weight=0.5,
            detection_markers=(
                "relaciona", "conecta", "depende", "implica",
                "causa", "similar", "diferente", "contrario",
            ),
        ),
    ],
    
    prohibitions=[
        # En CONCEPT no hay prohibiciones fuertes
    ],
    
    retrieval_spec=RetrievalSpec(
        chunk_type_filter=None,  # Sin filtro, cualquier tipo
        metadata_only=False,
        expand_concepts=True,
        top_k=10,
        min_score=0.3,
        vector_weight=0.35,
        bm25_weight=0.25,
        graph_weight=0.4,
    ),
    
    prompt_spec=PromptSpec(
        system_prompt_key="concept",
        max_temperature=0.7,
        max_tokens=2048,
        include_conversation=True,
        format_instructions="Estructura: Definición → Explicación → Relaciones → Ejemplos",
    ),
    
    evaluation_weights={
        "relevance": 0.30,
        "coverage": 0.25,
        "coherence": 0.15,
        "completeness": 0.15,
        "mode_alignment": 0.15,
    },
)


# ==========================================
# Especificación: PRACTICE
# ==========================================

PRACTICE_SPEC = ModeSpec(
    mode=LearningMode.PRACTICE,
    name="Modo Práctica",
    description="Resolver ejercicios paso a paso con justificación",
    
    requirements=[
        ContentRequirement(
            name="identificación",
            description="Identificación del problema y datos",
            required=True,
            weight=1.0,
            detection_markers=(
                "problema", "ejercicio", "datos", "dado", "encontrar",
                "calcular", "determinar", "hallar",
            ),
        ),
        ContentRequirement(
            name="pasos",
            description="Pasos de resolución numerados",
            required=True,
            weight=1.5,
            detection_markers=(
                "paso", "1.", "2.", "primero", "luego", "después",
                "siguiente", "a continuación", "finalmente",
            ),
        ),
        ContentRequirement(
            name="justificación",
            description="Justificación de cada paso",
            required=True,
            weight=1.0,
            detection_markers=(
                "porque", "ya que", "dado que", "aplicamos",
                "utilizamos", "según", "de acuerdo",
            ),
        ),
        ContentRequirement(
            name="resultado",
            description="Resultado final claro",
            required=True,
            weight=1.5,
            detection_markers=(
                "resultado", "respuesta", "solución", "obtenemos",
                "=", "igual a", "es igual", "por lo tanto",
            ),
        ),
    ],
    
    prohibitions=[
        ContentProhibition(
            name="respuesta_sin_proceso",
            description="Dar respuesta directa sin mostrar proceso",
            detection_markers=(),  # Se evalúa por ausencia de pasos
            penalty=0.4,
        ),
    ],
    
    retrieval_spec=RetrievalSpec(
        chunk_type_filter="worked_example",
        metadata_only=False,
        expand_concepts=True,
        top_k=10,
        min_score=0.3,
        vector_weight=0.45,
        bm25_weight=0.15,
        graph_weight=0.4,
    ),
    
    prompt_spec=PromptSpec(
        system_prompt_key="practice",
        max_temperature=0.5,
        max_tokens=3000,
        include_conversation=True,
        format_instructions="Estructura: Problema → Datos → Pasos numerados → Resultado → Verificación",
    ),
    
    evaluation_weights={
        "relevance": 0.25,
        "coverage": 0.20,
        "coherence": 0.20,
        "completeness": 0.15,
        "mode_alignment": 0.20,
    },
)


# ==========================================
# Especificación: EXERCISE_LIST
# ==========================================

EXERCISE_LIST_SPEC = ModeSpec(
    mode=LearningMode.EXERCISE_LIST,
    name="Modo Lista de Ejercicios",
    description="Listar ejercicios disponibles SIN explicación",
    
    requirements=[
        ContentRequirement(
            name="lista",
            description="Lista de ejercicios con formato",
            required=True,
            weight=2.0,
            detection_markers=(
                "ejercicio", "problema", "📝", "•", "-",
                "1.", "2.", "3.",
            ),
        ),
        ContentRequirement(
            name="metadatos",
            description="Metadatos de cada ejercicio (dificultad, concepto)",
            required=True,
            weight=1.0,
            detection_markers=(
                "dificultad", "concepto", "ref", "tipo",
                "nivel", "tema",
            ),
        ),
    ],
    
    prohibitions=[
        ContentProhibition(
            name="explicación_procedimiento",
            description="Explicaciones de cómo resolver",
            detection_markers=(
                "para resolver", "el primer paso", "se calcula",
                "aplicamos", "utilizamos la fórmula", "sustituyendo",
                "resolviendo", "operando", "simplificando",
            ),
            penalty=0.5,
        ),
        ContentProhibition(
            name="explicación_teórica",
            description="Explicaciones teóricas o conceptuales extensas",
            detection_markers=(
                "esto significa", "por lo tanto", "debido a",
                "la razón es", "porque", "ya que", "dado que",
            ),
            penalty=0.3,
        ),
        ContentProhibition(
            name="pasos_resolución",
            description="Pasos de resolución de ejercicios",
            detection_markers=(
                "paso 1", "paso 2", "primero", "luego", "después",
                "a continuación", "finalmente obtenemos",
            ),
            penalty=0.5,
        ),
        ContentProhibition(
            name="pistas_solución",
            description="Pistas o hints sobre cómo resolver",
            detection_markers=(
                "pista", "hint", "sugerencia", "recuerda que",
                "ten en cuenta", "considera",
            ),
            penalty=0.3,
        ),
    ],
    
    retrieval_spec=RetrievalSpec(
        chunk_type_filter="exercise",
        metadata_only=True,  # CRÍTICO: Solo metadatos
        expand_concepts=True,
        top_k=20,  # Más resultados para lista
        min_score=0.1,  # Umbral bajo para listar más
        vector_weight=0.0,
        bm25_weight=0.0,
        graph_weight=1.0,  # Solo grafo
    ),
    
    prompt_spec=PromptSpec(
        system_prompt_key="exercise_list",
        max_temperature=0.3,  # Baja creatividad
        max_tokens=1024,
        include_conversation=False,  # Sin historial
        format_instructions="SOLO listar: Título, Dificultad, Concepto, Referencia. SIN explicar.",
    ),
    
    evaluation_weights={
        "relevance": 0.20,
        "coverage": 0.15,
        "coherence": 0.15,
        "completeness": 0.10,
        "mode_alignment": 0.40,  # MUY importante
    },
)


# ==========================================
# Registry de Especificaciones
# ==========================================

_MODE_SPECS: Dict[LearningMode, ModeSpec] = {
    LearningMode.CONCEPT: CONCEPT_SPEC,
    LearningMode.PRACTICE: PRACTICE_SPEC,
    LearningMode.EXERCISE_LIST: EXERCISE_LIST_SPEC,
}


def get_mode_spec(mode: LearningMode) -> ModeSpec:
    """
    Obtiene la especificación para un modo.
    
    Args:
        mode: Modo de aprendizaje
        
    Returns:
        ModeSpec para el modo
    """
    return _MODE_SPECS.get(mode, CONCEPT_SPEC)


def validate_response_for_mode(
    response: str,
    mode: LearningMode,
) -> Tuple[float, List[str], List[str]]:
    """
    Valida una respuesta contra las especificaciones del modo.
    
    Args:
        response: Texto de la respuesta
        mode: Modo de aprendizaje
        
    Returns:
        Tupla (score, issues, suggestions)
    """
    spec = get_mode_spec(mode)
    return spec.validate_response(response)


def get_retrieval_config_for_mode(mode: LearningMode) -> Dict[str, Any]:
    """
    Obtiene configuración de retrieval para un modo.
    
    Args:
        mode: Modo de aprendizaje
        
    Returns:
        Diccionario con configuración de retrieval
    """
    spec = get_mode_spec(mode)
    rs = spec.retrieval_spec
    
    return {
        "chunk_type_filter": rs.chunk_type_filter,
        "metadata_only": rs.metadata_only,
        "expand_concepts": rs.expand_concepts,
        "top_k": rs.top_k,
        "min_score": rs.min_score,
        "vector_weight": rs.vector_weight,
        "bm25_weight": rs.bm25_weight,
        "graph_weight": rs.graph_weight,
    }


def get_prompt_template_for_mode(mode: LearningMode) -> Dict[str, Any]:
    """
    Obtiene configuración de prompt para un modo.
    
    Args:
        mode: Modo de aprendizaje
        
    Returns:
        Diccionario con configuración de prompt
    """
    spec = get_mode_spec(mode)
    ps = spec.prompt_spec
    
    return {
        "system_prompt_key": ps.system_prompt_key,
        "max_temperature": ps.max_temperature,
        "max_tokens": ps.max_tokens,
        "include_conversation": ps.include_conversation,
        "format_instructions": ps.format_instructions,
    }


def get_evaluation_weights_for_mode(mode: LearningMode) -> Dict[str, float]:
    """
    Obtiene pesos de evaluación para un modo.
    
    Args:
        mode: Modo de aprendizaje
        
    Returns:
        Diccionario con pesos por dimensión
    """
    spec = get_mode_spec(mode)
    return spec.evaluation_weights.copy()


# ==========================================
# Helpers para Orchestrator
# ==========================================

def should_include_conversation(mode: LearningMode) -> bool:
    """Indica si el modo debe incluir historial de conversación."""
    spec = get_mode_spec(mode)
    return spec.prompt_spec.include_conversation


def get_max_tokens_for_mode(mode: LearningMode) -> int:
    """Obtiene el máximo de tokens de salida para un modo."""
    spec = get_mode_spec(mode)
    return spec.prompt_spec.max_tokens


def get_temperature_for_mode(mode: LearningMode) -> float:
    """Obtiene la temperatura recomendada para un modo."""
    spec = get_mode_spec(mode)
    return spec.prompt_spec.max_temperature


def is_read_only_mode(mode: LearningMode) -> bool:
    """
    Indica si el modo es "read-only" (solo lista, sin generación creativa).
    
    EXERCISE_LIST es read-only: solo formatea datos existentes.
    """
    return mode == LearningMode.EXERCISE_LIST


def requires_metadata_only(mode: LearningMode) -> bool:
    """Indica si el modo solo necesita metadatos de retrieval."""
    spec = get_mode_spec(mode)
    return spec.retrieval_spec.metadata_only
