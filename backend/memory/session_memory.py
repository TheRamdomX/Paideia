"""
session_memory.py
Contexto de corto plazo (ventana conversacional).
Gestiona la memoria dentro de una sesión de aprendizaje.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from backend.settings import get_rag_config


# ==========================================
# Estructuras de Datos
# ==========================================

class TurnRole(str, Enum):
    """Roles en un turno conversacional."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnType(str, Enum):
    """Tipos de turno."""
    QUESTION = "question"
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    FOLLOWUP = "followup"
    FEEDBACK = "feedback"


@dataclass
class ConversationTurn:
    """Turno individual en la conversación."""
    
    turn_id: str = ""
    role: TurnRole = TurnRole.USER
    content: str = ""
    turn_type: TurnType = TurnType.QUESTION
    
    # Contexto asociado
    concepts: List[str] = field(default_factory=list)
    chunks_used: List[str] = field(default_factory=list)
    
    # Embeddings (opcional, para búsqueda semántica en contexto)
    embedding: Optional[List[float]] = None
    
    # Metadatos
    timestamp: Optional[datetime] = None
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role.value,
            "content": self.content,
            "turn_type": self.turn_type.value,
            "concepts": self.concepts,
            "chunks_used": self.chunks_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        turn = cls(
            turn_id=data.get("turn_id", ""),
            role=TurnRole(data.get("role", "user")),
            content=data.get("content", ""),
            turn_type=TurnType(data.get("turn_type", "question")),
            concepts=data.get("concepts", []),
            chunks_used=data.get("chunks_used", []),
            token_count=data.get("token_count", 0),
            metadata=data.get("metadata", {}),
        )
        
        if data.get("timestamp"):
            turn.timestamp = datetime.fromisoformat(data["timestamp"])
        
        return turn


@dataclass
class SessionContext:
    """Contexto de una sesión."""
    
    session_id: str = ""
    student_id: str = ""
    
    # Configuración de ventana
    max_turns: int = 20
    max_tokens: int = 4000
    
    # Historial
    turns: Deque[ConversationTurn] = field(default_factory=deque)
    
    # Conceptos activos en la sesión
    active_concepts: List[str] = field(default_factory=list)
    
    # Estado
    total_tokens: int = 0
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc)


# ==========================================
# Storage In-Memory
# ==========================================

# Almacenamiento de sesiones activas
_active_sessions: Dict[str, SessionContext] = {}


def _get_session(session_id: str) -> Optional[SessionContext]:
    """Obtiene sesión activa."""
    return _active_sessions.get(session_id)


def _create_session(
    session_id: str,
    student_id: str = "",
    max_turns: int = 20,
    max_tokens: int = 4000,
) -> SessionContext:
    """Crea nueva sesión."""
    session = SessionContext(
        session_id=session_id,
        student_id=student_id,
        max_turns=max_turns,
        max_tokens=max_tokens,
        turns=deque(maxlen=max_turns),
    )
    _active_sessions[session_id] = session
    return session


def clear_session(session_id: str) -> None:
    """Elimina una sesión."""
    if session_id in _active_sessions:
        del _active_sessions[session_id]


def clear_all_sessions() -> None:
    """Limpia todas las sesiones (para testing)."""
    _active_sessions.clear()


# ==========================================
# Almacenamiento de Turnos
# ==========================================

async def store_turn(
    session_id: str,
    role: TurnRole,
    content: str,
    turn_type: TurnType = TurnType.QUESTION,
    concepts: Optional[List[str]] = None,
    chunks_used: Optional[List[str]] = None,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ConversationTurn:
    """
    Guarda turno conversacional en la sesión.
    
    Args:
        session_id: ID de la sesión
        role: Rol (user/assistant/system)
        content: Contenido del turno
        turn_type: Tipo de turno
        concepts: Conceptos relacionados
        chunks_used: IDs de chunks usados
        embedding: Embedding del contenido (opcional)
        metadata: Metadata adicional
        
    Returns:
        ConversationTurn almacenado
    """
    # Obtener o crear sesión
    session = _get_session(session_id)
    if not session:
        session = _create_session(session_id)
    
    now = datetime.now(timezone.utc)
    
    # Estimar tokens (aproximación: ~4 chars = 1 token)
    token_count = len(content) // 4
    
    # Crear turno
    turn = ConversationTurn(
        turn_id=f"{session_id}_{len(session.turns)}",
        role=role,
        content=content,
        turn_type=turn_type,
        concepts=concepts or [],
        chunks_used=chunks_used or [],
        embedding=embedding,
        timestamp=now,
        token_count=token_count,
        metadata=metadata or {},
    )
    
    # Verificar si necesitamos podar antes de añadir
    if session.total_tokens + token_count > session.max_tokens:
        await prune_context(session_id, target_tokens=session.max_tokens - token_count)
    
    # Añadir turno
    session.turns.append(turn)
    session.total_tokens += token_count
    session.last_activity = now
    
    # Actualizar conceptos activos
    if concepts:
        for concept in concepts:
            if concept not in session.active_concepts:
                session.active_concepts.append(concept)
        
        # Mantener solo los últimos N conceptos activos
        session.active_concepts = session.active_concepts[-10:]
    
    return turn


# ==========================================
# Recuperación de Contexto
# ==========================================

async def get_recent_context(
    session_id: str,
    max_turns: Optional[int] = None,
    max_tokens: Optional[int] = None,
    include_system: bool = False,
) -> List[ConversationTurn]:
    """
    Devuelve los N últimos turnos relevantes (ventana de contexto).
    
    Args:
        session_id: ID de la sesión
        max_turns: Máximo de turnos a retornar
        max_tokens: Máximo de tokens a incluir
        include_system: Incluir mensajes del sistema
        
    Returns:
        Lista de turnos recientes
    """
    session = _get_session(session_id)
    
    if not session or not session.turns:
        return []
    
    # Valores por defecto
    if max_turns is None:
        max_turns = session.max_turns
    if max_tokens is None:
        max_tokens = session.max_tokens
    
    result: List[ConversationTurn] = []
    total_tokens = 0
    
    # Iterar desde el más reciente
    for turn in reversed(list(session.turns)):
        # Filtrar mensajes del sistema si no se solicitan
        if not include_system and turn.role == TurnRole.SYSTEM:
            continue
        
        # Verificar límite de tokens
        if total_tokens + turn.token_count > max_tokens:
            break
        
        # Verificar límite de turnos
        if len(result) >= max_turns:
            break
        
        result.append(turn)
        total_tokens += turn.token_count
    
    # Revertir para orden cronológico
    return list(reversed(result))


async def get_context_for_llm(
    session_id: str,
    max_tokens: int = 2000,
) -> List[Dict[str, str]]:
    """
    Obtiene contexto formateado para LLM.
    
    Args:
        session_id: ID de la sesión
        max_tokens: Máximo de tokens
        
    Returns:
        Lista de mensajes en formato LLM (role, content)
    """
    turns = await get_recent_context(session_id, max_tokens=max_tokens)
    
    messages = []
    for turn in turns:
        messages.append({
            "role": turn.role.value,
            "content": turn.content,
        })
    
    return messages


async def get_related_turns(
    session_id: str,
    concepts: List[str],
    limit: int = 5,
) -> List[ConversationTurn]:
    """
    Obtiene turnos relacionados con conceptos específicos.
    
    Args:
        session_id: ID de la sesión
        concepts: Conceptos a buscar
        limit: Máximo de resultados
        
    Returns:
        Turnos que mencionan los conceptos
    """
    session = _get_session(session_id)
    
    if not session:
        return []
    
    concepts_set = set(concepts)
    related: List[Tuple[ConversationTurn, int]] = []
    
    for turn in session.turns:
        overlap = len(set(turn.concepts) & concepts_set)
        if overlap > 0:
            related.append((turn, overlap))
    
    # Ordenar por relevancia
    related.sort(key=lambda x: x[1], reverse=True)
    
    return [turn for turn, _ in related[:limit]]


async def get_active_concepts(session_id: str) -> List[str]:
    """
    Obtiene conceptos activos en la sesión actual.
    
    Args:
        session_id: ID de la sesión
        
    Returns:
        Lista de IDs de conceptos activos
    """
    session = _get_session(session_id)
    
    if not session:
        return []
    
    return session.active_concepts.copy()


# ==========================================
# Poda de Contexto
# ==========================================

async def prune_context(
    session_id: str,
    target_tokens: Optional[int] = None,
    keep_last: int = 2,
) -> int:
    """
    Elimina turnos antiguos para respetar límite de tokens.
    
    Args:
        session_id: ID de la sesión
        target_tokens: Tokens objetivo después de poda
        keep_last: Mínimo de turnos recientes a mantener
        
    Returns:
        Número de turnos eliminados
    """
    session = _get_session(session_id)
    
    if not session:
        return 0
    
    if target_tokens is None:
        target_tokens = session.max_tokens
    
    removed = 0
    
    # Mantener al menos los últimos N turnos
    while (
        len(session.turns) > keep_last and 
        session.total_tokens > target_tokens
    ):
        # Eliminar el turno más antiguo
        old_turn = session.turns.popleft()
        session.total_tokens -= old_turn.token_count
        removed += 1
    
    return removed


async def summarize_and_prune(
    session_id: str,
    summarize_count: int = 10,
) -> Optional[ConversationTurn]:
    """
    Resume turnos antiguos y los reemplaza por un resumen.
    
    Args:
        session_id: ID de la sesión
        summarize_count: Número de turnos a resumir
        
    Returns:
        Turno de resumen si se creó, None si no
    """
    session = _get_session(session_id)
    
    if not session or len(session.turns) <= summarize_count:
        return None
    
    # Obtener turnos a resumir
    turns_to_summarize = list(session.turns)[:summarize_count]
    
    # Crear resumen simple (en producción, usar LLM)
    concepts_mentioned: List[str] = []
    chunks_used: List[str] = []
    summary_parts: List[str] = []
    
    for turn in turns_to_summarize:
        concepts_mentioned.extend(turn.concepts)
        chunks_used.extend(turn.chunks_used)
        
        if turn.role == TurnRole.USER:
            summary_parts.append(f"- User asked about: {turn.content[:100]}...")
        elif turn.role == TurnRole.ASSISTANT:
            summary_parts.append(f"- Assistant explained: {turn.content[:100]}...")
    
    summary_content = (
        "[Conversation Summary]\n" + 
        "\n".join(summary_parts[:5])  # Limitar a 5 items
    )
    
    # Eliminar turnos resumidos
    removed_tokens = 0
    for _ in range(summarize_count):
        if session.turns:
            old = session.turns.popleft()
            removed_tokens += old.token_count
    
    # Crear turno de resumen
    summary_turn = ConversationTurn(
        turn_id=f"{session_id}_summary_{datetime.now().timestamp()}",
        role=TurnRole.SYSTEM,
        content=summary_content,
        turn_type=TurnType.ANSWER,
        concepts=list(set(concepts_mentioned)),
        chunks_used=list(set(chunks_used)),
        timestamp=datetime.now(timezone.utc),
        token_count=len(summary_content) // 4,
        metadata={"is_summary": True, "summarized_turns": summarize_count},
    )
    
    # Insertar al inicio
    session.turns.appendleft(summary_turn)
    session.total_tokens = session.total_tokens - removed_tokens + summary_turn.token_count
    
    return summary_turn


# ==========================================
# Utilidades
# ==========================================

def get_session_stats(session_id: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas de la sesión.
    
    Args:
        session_id: ID de la sesión
        
    Returns:
        Diccionario con estadísticas
    """
    session = _get_session(session_id)
    
    if not session:
        return {
            "exists": False,
            "total_turns": 0,
            "total_tokens": 0,
        }
    
    user_turns = sum(1 for t in session.turns if t.role == TurnRole.USER)
    assistant_turns = sum(1 for t in session.turns if t.role == TurnRole.ASSISTANT)
    
    return {
        "exists": True,
        "session_id": session_id,
        "student_id": session.student_id,
        "total_turns": len(session.turns),
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "total_tokens": session.total_tokens,
        "max_tokens": session.max_tokens,
        "active_concepts": len(session.active_concepts),
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "last_activity": session.last_activity.isoformat() if session.last_activity else None,
    }


async def export_session(session_id: str) -> Dict[str, Any]:
    """
    Exporta sesión completa para persistencia.
    
    Args:
        session_id: ID de la sesión
        
    Returns:
        Sesión serializada
    """
    session = _get_session(session_id)
    
    if not session:
        return {}
    
    return {
        "session_id": session.session_id,
        "student_id": session.student_id,
        "max_turns": session.max_turns,
        "max_tokens": session.max_tokens,
        "turns": [t.to_dict() for t in session.turns],
        "active_concepts": session.active_concepts,
        "total_tokens": session.total_tokens,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "last_activity": session.last_activity.isoformat() if session.last_activity else None,
    }


async def import_session(data: Dict[str, Any]) -> SessionContext:
    """
    Importa sesión desde datos serializados.
    
    Args:
        data: Datos de sesión
        
    Returns:
        Sesión restaurada
    """
    session = _create_session(
        session_id=data.get("session_id", ""),
        student_id=data.get("student_id", ""),
        max_turns=data.get("max_turns", 20),
        max_tokens=data.get("max_tokens", 4000),
    )
    
    # Restaurar turnos
    for turn_data in data.get("turns", []):
        turn = ConversationTurn.from_dict(turn_data)
        session.turns.append(turn)
        session.total_tokens += turn.token_count
    
    session.active_concepts = data.get("active_concepts", [])
    
    if data.get("created_at"):
        session.created_at = datetime.fromisoformat(data["created_at"])
    if data.get("last_activity"):
        session.last_activity = datetime.fromisoformat(data["last_activity"])
    
    return session
