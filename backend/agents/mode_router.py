"""
mode_router.py
Agente de Enrutamiento de Modos - Detecta la intención pedagógica del estudiante.
Clasifica queries en modos de aprendizaje: CONCEPT, PRACTICE, EXERCISE_LIST.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import re


# ==========================================
# Modos de Aprendizaje
# ==========================================

class LearningMode(str, Enum):
    """Modos pedagógicos de aprendizaje."""
    CONCEPT = "concept"              # Explicar teoría / definiciones / relaciones
    PRACTICE = "practice"            # Explicar ejercicios resueltos paso a paso
    EXERCISE_LIST = "exercise_list"  # Listar ejercicios por tipo o concepto (SIN explicación)


@dataclass
class ModeDetectionResult:
    """Resultado de la detección de modo."""
    
    mode: LearningMode = LearningMode.CONCEPT
    confidence: float = 0.0
    matched_patterns: List[str] = field(default_factory=list)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "reason": self.reason,
            "metadata": self.metadata,
        }


# ==========================================
# Patrones de Detección
# ==========================================

# Patrones para EXERCISE_LIST (mayor prioridad)
EXERCISE_LIST_PATTERNS = [
    # Español
    r"\b(dame|muestra|lista|enumera|presenta)\s+(los\s+)?(ejercicios|problemas|prácticas)\b",
    r"\b(ejercicios|problemas)\s+(de|sobre|para)\b",
    r"\b(qué|cuáles)\s+(ejercicios|problemas)\s+(hay|tiene|existen)\b",
    r"\blista\s+de\s+(ejercicios|problemas|prácticas)\b",
    r"\bejercicio[s]?\s+(disponible|relacionado)[s]?\b",
    r"\bmuéstrame\s+(ejercicios|problemas)\b",
    r"\bquiero\s+(ejercicios|problemas|practicar)\b",
    r"\bnecesito\s+(ejercicios|problemas|práctica)\b",
    r"\btiene[s]?\s+(ejercicios|problemas)\b",
    r"\bpráctica\s+(de|sobre|para)\b",
    r"\bejemplos\s+de\s+ejercicios\b",
    # Inglés
    r"\b(give|show|list|enumerate)\s+(me\s+)?(exercises|problems|practice)\b",
    r"\b(exercises|problems)\s+(on|about|for)\b",
    r"\b(what|which)\s+(exercises|problems)\s+(are|do)\b",
    r"\blist\s+of\s+(exercises|problems)\b",
    r"\bi\s+(want|need)\s+(exercises|problems|practice)\b",
]

# Keywords simples para EXERCISE_LIST
EXERCISE_LIST_KEYWORDS = [
    "ejercicios", "problemas", "práctica", "dame ejercicios",
    "lista de ejercicios", "muestra ejercicios", "ejercicios disponibles",
    "exercises", "problems", "practice problems", "show exercises",
]

# Patrones para PRACTICE (paso a paso)
PRACTICE_PATTERNS = [
    # Español
    r"\b(cómo|como)\s+(se\s+)?(resuelve|resolver|soluciona|solucionar|hace|hacer)\b",
    r"\bresuelve\s+(este|el|un|una|la)\b",
    r"\b(explica|explicar)\s+(cómo|como)\s+(se\s+)?(resuelve|resolver)\b",
    r"\bpaso\s+a\s+paso\b",
    r"\bpaso\s+por\s+paso\b",
    r"\bejemplo\s+(resuelto|de\s+solución)\b",
    r"\bsolución\s+(de|del|detallada)\b",
    r"\bresolver\s+(este|el|un|una)\s+(problema|ejercicio)\b",
    r"\b(muestra|explica)\s+(el|un)\s+(procedimiento|proceso)\b",
    r"\bcálculo\s+de\b",
    r"\bdesarrollo\s+de\b",
    r"\b(demuestra|demostrar|demostración)\b",
    # Inglés
    r"\bhow\s+(do\s+)?(i|you|we)\s+(solve|calculate|find|compute)\b",
    r"\bsolve\s+(this|the|a|an)\b",
    r"\bstep[\s-]by[\s-]step\b",
    r"\bworked\s+(example|solution)\b",
    r"\bshow\s+(me\s+)?(how|the\s+solution)\b",
    r"\bwalk\s+(me\s+)?through\b",
    r"\bexplain\s+how\s+to\b",
]

# Keywords simples para PRACTICE
PRACTICE_KEYWORDS = [
    "cómo resolver", "como resolver", "resuelve", "paso a paso",
    "ejemplo resuelto", "solución detallada", "procedimiento",
    "how to solve", "step by step", "worked example", "show me how",
    "resolver el ejercicio", "resolver el problema",
]

# Patrones para CONCEPT (explicación teórica) - menor prioridad
CONCEPT_PATTERNS = [
    # Español
    r"\b(qué|que)\s+(es|son|significa)\b",
    r"\b(define|definir|definición)\b",
    r"\b(explica|explicar)\s+(qué|que|el|la|los|las)\b",
    r"\bconcepto\s+de\b",
    r"\b(teoría|teórico|teórica)\b",
    r"\brelación\s+entre\b",
    r"\bdiferencia\s+entre\b",
    r"\bcaracterísticas\s+de\b",
    r"\bpropiedades\s+de\b",
    r"\btipos\s+de\b",
    r"\bclasificación\s+de\b",
    r"\bfundamentos\b",
    r"\bprincipios\b",
    # Inglés
    r"\bwhat\s+(is|are|does)\b",
    r"\bdefine\b",
    r"\b(explain|describe)\s+(what|the)\b",
    r"\bconcept\s+of\b",
    r"\b(theory|theoretical)\b",
    r"\brelationship\s+between\b",
    r"\bdifference\s+between\b",
    r"\bcharacteristics\s+of\b",
    r"\bproperties\s+of\b",
]


# ==========================================
# Detección de Modo
# ==========================================

async def detect_mode(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> LearningMode:
    """
    Detecta el modo de aprendizaje de una query.
    
    Args:
        query: Texto de la consulta del estudiante
        context: Contexto adicional (sesión, historial, etc.)
        
    Returns:
        LearningMode detectado
    """
    result = await detect_mode_with_details(query, context)
    return result.mode


async def detect_mode_with_details(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> ModeDetectionResult:
    """
    Detecta el modo de aprendizaje con detalles completos.
    
    Args:
        query: Texto de la consulta del estudiante
        context: Contexto adicional
        
    Returns:
        ModeDetectionResult con modo y metadatos
    """
    query_lower = query.lower().strip()
    context = context or {}
    
    # Scores para cada modo
    scores: Dict[LearningMode, float] = {
        LearningMode.CONCEPT: 0.0,
        LearningMode.PRACTICE: 0.0,
        LearningMode.EXERCISE_LIST: 0.0,
    }
    
    matched_patterns: Dict[LearningMode, List[str]] = {
        LearningMode.CONCEPT: [],
        LearningMode.PRACTICE: [],
        LearningMode.EXERCISE_LIST: [],
    }
    
    # 1. Verificar EXERCISE_LIST primero (mayor prioridad)
    exercise_score, exercise_matches = _check_patterns(
        query_lower, 
        EXERCISE_LIST_PATTERNS, 
        EXERCISE_LIST_KEYWORDS
    )
    scores[LearningMode.EXERCISE_LIST] = exercise_score
    matched_patterns[LearningMode.EXERCISE_LIST] = exercise_matches
    
    # 2. Verificar PRACTICE
    practice_score, practice_matches = _check_patterns(
        query_lower,
        PRACTICE_PATTERNS,
        PRACTICE_KEYWORDS
    )
    scores[LearningMode.PRACTICE] = practice_score
    matched_patterns[LearningMode.PRACTICE] = practice_matches
    
    # 3. Verificar CONCEPT
    concept_score, concept_matches = _check_patterns(
        query_lower,
        CONCEPT_PATTERNS,
        []  # Sin keywords adicionales, es el default
    )
    scores[LearningMode.CONCEPT] = concept_score
    matched_patterns[LearningMode.CONCEPT] = concept_matches
    
    # 4. Aplicar ajustes contextuales
    scores = _apply_context_adjustments(scores, query_lower, context)
    
    # 5. Determinar modo ganador
    # EXERCISE_LIST tiene prioridad si score >= 0.5
    if scores[LearningMode.EXERCISE_LIST] >= 0.5:
        winning_mode = LearningMode.EXERCISE_LIST
    # PRACTICE tiene segunda prioridad
    elif scores[LearningMode.PRACTICE] >= 0.5:
        winning_mode = LearningMode.PRACTICE
    # CONCEPT es default
    elif scores[LearningMode.CONCEPT] >= 0.3:
        winning_mode = LearningMode.CONCEPT
    # Si ninguno tiene score alto, usar heurísticas adicionales
    else:
        winning_mode = _fallback_detection(query_lower)
    
    # Calcular confianza
    winning_score = scores[winning_mode]
    other_scores = [s for m, s in scores.items() if m != winning_mode]
    max_other = max(other_scores) if other_scores else 0
    confidence = min(1.0, winning_score - max_other + 0.5) if winning_score > 0 else 0.5
    
    # Construir razón
    reason = _build_reason(winning_mode, matched_patterns[winning_mode], winning_score)
    
    return ModeDetectionResult(
        mode=winning_mode,
        confidence=confidence,
        matched_patterns=matched_patterns[winning_mode],
        reason=reason,
        metadata={
            "all_scores": {m.value: s for m, s in scores.items()},
            "query_length": len(query.split()),
            "has_question_mark": "?" in query,
        },
    )


def _check_patterns(
    query: str,
    patterns: List[str],
    keywords: List[str],
) -> Tuple[float, List[str]]:
    """
    Verifica patrones regex y keywords en la query.
    
    Returns:
        Tupla (score, matched_patterns)
    """
    score = 0.0
    matches: List[str] = []
    
    # Verificar patrones regex
    for pattern in patterns:
        try:
            if re.search(pattern, query, re.IGNORECASE):
                score += 0.4
                matches.append(f"pattern:{pattern[:30]}...")
        except re.error:
            continue
    
    # Verificar keywords simples
    for keyword in keywords:
        if keyword.lower() in query:
            score += 0.3
            matches.append(f"keyword:{keyword}")
    
    # Normalizar score
    score = min(1.0, score)
    
    return score, matches


def _apply_context_adjustments(
    scores: Dict[LearningMode, float],
    query: str,
    context: Dict[str, Any],
) -> Dict[LearningMode, float]:
    """Aplica ajustes basados en contexto."""
    
    # Si la query es muy corta y contiene "ejercicio", boost EXERCISE_LIST
    words = query.split()
    if len(words) <= 5 and any(w in query for w in ["ejercicio", "problema", "práctica"]):
        scores[LearningMode.EXERCISE_LIST] += 0.2
    
    # Si contiene números o fórmulas, probablemente es PRACTICE
    if re.search(r'\d+|\+|\-|\*|\/|=', query):
        scores[LearningMode.PRACTICE] += 0.15
    
    # Si pregunta "qué es" o "define", boost CONCEPT
    if re.search(r'(qué|que)\s+(es|son|significa)|define|definición', query):
        scores[LearningMode.CONCEPT] += 0.3
    
    # Ajuste por contexto de sesión (si existe modo previo)
    if context.get("previous_mode"):
        prev_mode = LearningMode(context["previous_mode"])
        # Ligero boost al modo anterior por continuidad
        scores[prev_mode] += 0.1
    
    # Normalizar
    for mode in scores:
        scores[mode] = min(1.0, scores[mode])
    
    return scores


def _fallback_detection(query: str) -> LearningMode:
    """Detección fallback cuando los scores son bajos."""
    
    # Heurísticas simples
    query_lower = query.lower()
    
    # Si menciona ejercicios de cualquier forma
    if any(w in query_lower for w in ["ejercicio", "problema", "práctica", "exercise", "problem"]):
        # Verificar si quiere lista o resolución
        if any(w in query_lower for w in ["cómo", "como", "resolver", "resuelve", "solve", "how"]):
            return LearningMode.PRACTICE
        return LearningMode.EXERCISE_LIST
    
    # Si pregunta "cómo"
    if any(w in query_lower for w in ["cómo", "como", "how"]):
        return LearningMode.PRACTICE
    
    # Default: CONCEPT
    return LearningMode.CONCEPT


def _build_reason(
    mode: LearningMode,
    matches: List[str],
    score: float,
) -> str:
    """Construye explicación de la decisión."""
    
    if not matches:
        return f"Default mode ({mode.value}) - no strong patterns detected"
    
    match_summary = ", ".join(matches[:3])
    if len(matches) > 3:
        match_summary += f" (+{len(matches) - 3} more)"
    
    return f"Detected {mode.value} (score={score:.2f}): {match_summary}"


# ==========================================
# Utilidades
# ==========================================

def get_mode_description(mode: LearningMode, language: str = "es") -> str:
    """Obtiene descripción del modo para mostrar al usuario."""
    
    descriptions = {
        "es": {
            LearningMode.CONCEPT: "Modo Concepto: Explicación de teoría y definiciones",
            LearningMode.PRACTICE: "Modo Práctica: Resolución paso a paso",
            LearningMode.EXERCISE_LIST: "Modo Ejercicios: Lista de ejercicios disponibles",
        },
        "en": {
            LearningMode.CONCEPT: "Concept Mode: Theory and definitions explanation",
            LearningMode.PRACTICE: "Practice Mode: Step-by-step solution",
            LearningMode.EXERCISE_LIST: "Exercise Mode: List of available exercises",
        },
    }
    
    lang_descs = descriptions.get(language, descriptions["es"])
    return lang_descs.get(mode, str(mode.value))


def is_mode_switch_request(query: str) -> Optional[LearningMode]:
    """
    Detecta si el usuario quiere cambiar de modo explícitamente.
    
    Returns:
        LearningMode si hay solicitud explícita, None si no
    """
    query_lower = query.lower()
    
    # Solicitudes explícitas de cambio
    if any(p in query_lower for p in [
        "cambiar a modo", "switch to mode", "modo concepto",
        "modo práctica", "modo ejercicios", "quiero ver ejercicios",
        "explícame la teoría", "resuelve paso a paso"
    ]):
        if any(w in query_lower for w in ["concepto", "teoría", "concept", "theory"]):
            return LearningMode.CONCEPT
        elif any(w in query_lower for w in ["práctica", "paso", "practice", "step"]):
            return LearningMode.PRACTICE
        elif any(w in query_lower for w in ["ejercicio", "lista", "exercise", "list"]):
            return LearningMode.EXERCISE_LIST
    
    return None


async def validate_mode_for_content(
    mode: LearningMode,
    available_content: Dict[str, int],
) -> Tuple[bool, str]:
    """
    Valida si el modo es viable dado el contenido disponible.
    
    Args:
        mode: Modo detectado
        available_content: Conteo de tipos de contenido disponible
        
    Returns:
        Tupla (is_valid, message)
    """
    if mode == LearningMode.EXERCISE_LIST:
        exercise_count = available_content.get("exercise", 0)
        if exercise_count == 0:
            return False, "No hay ejercicios disponibles en los documentos cargados"
        return True, f"{exercise_count} ejercicios disponibles"
    
    elif mode == LearningMode.PRACTICE:
        worked_examples = available_content.get("worked_example", 0)
        exercises = available_content.get("exercise", 0)
        if worked_examples == 0 and exercises == 0:
            return False, "No hay ejemplos resueltos o ejercicios disponibles"
        return True, f"{worked_examples} ejemplos resueltos, {exercises} ejercicios"
    
    else:  # CONCEPT
        concepts = available_content.get("concept", 0)
        definitions = available_content.get("definition", 0)
        chunks = available_content.get("chunk", 0)
        if chunks == 0:
            return False, "No hay contenido conceptual disponible"
        return True, f"{concepts} conceptos, {definitions} definiciones"
