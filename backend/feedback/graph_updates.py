"""
graph_updates.py
Refuerza o debilita relaciones del grafo basado en feedback.
Sistema de aprendizaje continuo del grafo de conocimiento.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.db.surreal import execute, get_db
from backend.feedback.signals import (
    FeedbackSignal,
    NormalizedFeedback,
    SentimentPolarity,
)
from backend.settings import get_rag_config


# ==========================================
# Tipos y Estructuras
# ==========================================

class UpdateAction(str, Enum):
    """Acciones de actualización de grafo."""
    REINFORCE = "reinforce"
    WEAKEN = "weaken"
    CREATE = "create"
    DELETE = "delete"


class EdgeType(str, Enum):
    """Tipos de aristas en el grafo."""
    PREREQUISITE = "prerequisite"
    RELATED = "related"
    PART_OF = "part_of"
    DERIVED_FROM = "derived_from"
    EXPLAINS = "explains"


@dataclass
class EdgeUpdate:
    """Actualización pendiente de una arista."""
    
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.RELATED
    action: UpdateAction = UpdateAction.REINFORCE
    
    # Delta de peso
    weight_delta: float = 0.0
    new_weight: Optional[float] = None
    
    # Razón
    reason: str = ""
    feedback_id: str = ""
    
    # Estado
    applied: bool = False
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "action": self.action.value,
            "weight_delta": self.weight_delta,
            "new_weight": self.new_weight,
            "reason": self.reason,
            "applied": self.applied,
        }


@dataclass
class RevectorizationTask:
    """Tarea de re-vectorización pendiente."""
    
    task_id: str = ""
    chunk_ids: List[str] = field(default_factory=list)
    reason: str = ""
    priority: int = 1  # 1 = alta, 5 = baja
    
    # Estado
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "chunk_ids": self.chunk_ids,
            "reason": self.reason,
            "priority": self.priority,
            "status": self.status,
        }


# ==========================================
# Storage de Tareas Pendientes
# ==========================================

_pending_edge_updates: List[EdgeUpdate] = []
_pending_revectorizations: List[RevectorizationTask] = []


def clear_pending_tasks() -> None:
    """Limpia tareas pendientes (para testing)."""
    _pending_edge_updates.clear()
    _pending_revectorizations.clear()


# ==========================================
# Reforzamiento de Aristas
# ==========================================

async def reinforce_edges(
    chunk_ids: List[str],
    concept_ids: List[str],
    feedback: NormalizedFeedback,
) -> List[EdgeUpdate]:
    """
    Incrementa peso de aristas asociadas a contenido bien evaluado.
    
    Args:
        chunk_ids: IDs de chunks con buen feedback
        concept_ids: IDs de conceptos relacionados
        feedback: Feedback recibido
        
    Returns:
        Lista de actualizaciones aplicadas
    """
    updates: List[EdgeUpdate] = []
    
    # Solo reforzar si el feedback es positivo
    if feedback.overall_score < 0.6:
        return updates
    
    # Calcular delta basado en el score
    base_delta = 0.05
    score_multiplier = feedback.overall_score
    confidence_multiplier = feedback.confidence
    
    weight_delta = base_delta * score_multiplier * confidence_multiplier
    
    try:
        db = await get_db()
        
        # Reforzar aristas entre conceptos mencionados
        if len(concept_ids) >= 2:
            for i, source_id in enumerate(concept_ids):
                for target_id in concept_ids[i+1:]:
                    update = await _reinforce_edge(
                        db=db,
                        source_id=source_id,
                        target_id=target_id,
                        weight_delta=weight_delta,
                        reason=f"positive_feedback_{feedback.query_id}",
                    )
                    if update:
                        updates.append(update)
        
        # Reforzar aristas chunk -> concepto
        for chunk_id in chunk_ids:
            for concept_id in concept_ids:
                update = await _reinforce_chunk_concept_edge(
                    db=db,
                    chunk_id=chunk_id,
                    concept_id=concept_id,
                    weight_delta=weight_delta,
                    reason=f"positive_feedback_{feedback.query_id}",
                )
                if update:
                    updates.append(update)
                    
    except Exception:
        pass
    
    return updates


async def _reinforce_edge(
    db,
    source_id: str,
    target_id: str,
    weight_delta: float,
    reason: str,
) -> Optional[EdgeUpdate]:
    """Refuerza una arista específica entre conceptos."""
    try:
        # Buscar arista existente
        surql_select = """
            SELECT weight, type
            FROM edge
            WHERE (out = $source AND in = $target)
               OR (out = $target AND in = $source)
            LIMIT 1
        """
        
        result = await db.query(surql_select, {
            "source": source_id,
            "target": target_id,
        })
        
        current_weight = 0.5
        edge_type = EdgeType.RELATED
        
        if result and result[0].get("result"):
            row = result[0]["result"][0]
            current_weight = float(row.get("weight", 0.5))
            edge_type = EdgeType(row.get("type", "related"))
        
        # Calcular nuevo peso (máximo 1.0)
        new_weight = min(1.0, current_weight + weight_delta)
        
        # Actualizar en DB
        surql_update = """
            UPDATE edge SET
                weight = $weight,
                last_reinforced = $timestamp
            WHERE (out = $source AND in = $target)
               OR (out = $target AND in = $source)
        """
        
        await db.query(surql_update, {
            "source": source_id,
            "target": target_id,
            "weight": new_weight,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        update = EdgeUpdate(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            action=UpdateAction.REINFORCE,
            weight_delta=weight_delta,
            new_weight=new_weight,
            reason=reason,
            applied=True,
            timestamp=datetime.now(timezone.utc),
        )
        
        return update
        
    except Exception:
        return None


async def _reinforce_chunk_concept_edge(
    db,
    chunk_id: str,
    concept_id: str,
    weight_delta: float,
    reason: str,
) -> Optional[EdgeUpdate]:
    """Refuerza arista chunk -> concepto."""
    try:
        surql = """
            UPDATE chunk_concept SET
                relevance = math::min(1.0, relevance + $delta),
                last_reinforced = $timestamp
            WHERE chunk_id = $chunk_id AND concept_id = $concept_id
        """
        
        await db.query(surql, {
            "chunk_id": chunk_id,
            "concept_id": concept_id,
            "delta": weight_delta,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return EdgeUpdate(
            source_id=chunk_id,
            target_id=concept_id,
            edge_type=EdgeType.EXPLAINS,
            action=UpdateAction.REINFORCE,
            weight_delta=weight_delta,
            reason=reason,
            applied=True,
            timestamp=datetime.now(timezone.utc),
        )
        
    except Exception:
        return None


# ==========================================
# Debilitamiento de Aristas
# ==========================================

async def weaken_edges(
    chunk_ids: List[str],
    concept_ids: List[str],
    feedback: NormalizedFeedback,
) -> List[EdgeUpdate]:
    """
    Reduce peso de aristas asociadas a contenido mal evaluado.
    
    Args:
        chunk_ids: IDs de chunks con mal feedback
        concept_ids: IDs de conceptos relacionados
        feedback: Feedback recibido
        
    Returns:
        Lista de actualizaciones aplicadas
    """
    updates: List[EdgeUpdate] = []
    
    # Solo debilitar si el feedback es negativo
    if feedback.overall_score > 0.4:
        return updates
    
    # Calcular delta negativo basado en el score
    base_delta = -0.08  # Más agresivo que el refuerzo
    score_multiplier = 1.0 - feedback.overall_score  # Más bajo score = mayor penalización
    confidence_multiplier = feedback.confidence
    
    weight_delta = base_delta * score_multiplier * confidence_multiplier
    
    try:
        db = await get_db()
        
        # Debilitar aristas chunk -> concepto
        for chunk_id in chunk_ids:
            for concept_id in concept_ids:
                update = await _weaken_edge(
                    db=db,
                    source_id=chunk_id,
                    target_id=concept_id,
                    weight_delta=weight_delta,
                    reason=f"negative_feedback_{feedback.query_id}",
                )
                if update:
                    updates.append(update)
        
        # Si hay señales muy negativas, considerar marcar para revisión
        if feedback.overall_score < 0.2:
            await _mark_for_review(db, chunk_ids, concept_ids, feedback)
            
    except Exception:
        pass
    
    return updates


async def _weaken_edge(
    db,
    source_id: str,
    target_id: str,
    weight_delta: float,
    reason: str,
    min_weight: float = 0.1,
) -> Optional[EdgeUpdate]:
    """Debilita una arista específica."""
    try:
        # Actualizar peso con límite inferior
        surql = """
            UPDATE edge SET
                weight = math::max($min_weight, weight + $delta),
                last_weakened = $timestamp,
                negative_signals = negative_signals + 1
            WHERE (out = $source AND in = $target)
               OR (out = $target AND in = $source)
        """
        
        await db.query(surql, {
            "source": source_id,
            "target": target_id,
            "delta": weight_delta,
            "min_weight": min_weight,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return EdgeUpdate(
            source_id=source_id,
            target_id=target_id,
            action=UpdateAction.WEAKEN,
            weight_delta=weight_delta,
            reason=reason,
            applied=True,
            timestamp=datetime.now(timezone.utc),
        )
        
    except Exception:
        return None


async def _mark_for_review(
    db,
    chunk_ids: List[str],
    concept_ids: List[str],
    feedback: NormalizedFeedback,
) -> None:
    """Marca contenido para revisión manual."""
    try:
        now = datetime.now(timezone.utc)
        
        # Marcar chunks
        for chunk_id in chunk_ids:
            surql = """
                UPDATE chunk SET
                    needs_review = true,
                    review_reason = $reason,
                    marked_for_review_at = $timestamp
                WHERE id = $chunk_id
            """
            
            await db.query(surql, {
                "chunk_id": chunk_id,
                "reason": f"low_feedback_score_{feedback.overall_score:.2f}",
                "timestamp": now.isoformat(),
            })
            
    except Exception:
        pass


# ==========================================
# Programación de Re-vectorización
# ==========================================

async def schedule_revectorization(
    chunk_ids: List[str],
    reason: str = "",
    priority: int = 3,
) -> RevectorizationTask:
    """
    Programa re-indexación de chunks con problemas.
    
    Args:
        chunk_ids: IDs de chunks a re-vectorizar
        reason: Razón de la re-vectorización
        priority: Prioridad (1=alta, 5=baja)
        
    Returns:
        Tarea creada
    """
    now = datetime.now(timezone.utc)
    
    task = RevectorizationTask(
        task_id=f"revec_{now.timestamp()}",
        chunk_ids=chunk_ids,
        reason=reason or "feedback_triggered",
        priority=priority,
        scheduled_at=now,
        status="pending",
    )
    
    # Agregar a pendientes
    _pending_revectorizations.append(task)
    
    # Persistir en base de datos
    try:
        db = await get_db()
        
        surql = """
            CREATE revectorization_task SET
                task_id = $task_id,
                chunk_ids = $chunk_ids,
                reason = $reason,
                priority = $priority,
                scheduled_at = $scheduled_at,
                status = $status
        """
        
        await db.query(surql, {
            "task_id": task.task_id,
            "chunk_ids": chunk_ids,
            "reason": task.reason,
            "priority": priority,
            "scheduled_at": now.isoformat(),
            "status": "pending",
        })
        
    except Exception:
        pass
    
    return task


async def process_revectorization_queue(
    max_tasks: int = 5,
) -> List[RevectorizationTask]:
    """
    Procesa tareas de re-vectorización pendientes.
    
    Args:
        max_tasks: Máximo de tareas a procesar
        
    Returns:
        Tareas procesadas
    """
    processed: List[RevectorizationTask] = []
    
    # Ordenar por prioridad
    pending = sorted(
        [t for t in _pending_revectorizations if t.status == "pending"],
        key=lambda t: t.priority
    )
    
    for task in pending[:max_tasks]:
        try:
            task.started_at = datetime.now(timezone.utc)
            task.status = "processing"
            
            # Llamar al vectorizador (importación diferida para evitar ciclos)
            success = await _execute_revectorization(task.chunk_ids)
            
            if success:
                task.completed_at = datetime.now(timezone.utc)
                task.status = "completed"
            else:
                task.status = "failed"
            
            processed.append(task)
            
            # Actualizar en DB
            await _update_task_status(task)
            
        except Exception:
            task.status = "failed"
    
    # Limpiar completadas
    _pending_revectorizations[:] = [
        t for t in _pending_revectorizations 
        if t.status not in ("completed", "failed")
    ]
    
    return processed


async def _execute_revectorization(chunk_ids: List[str]) -> bool:
    """Ejecuta la re-vectorización de chunks."""
    try:
        # Importación diferida
        from backend.ingestion.vectorizer import submit_vectorization
        
        db = await get_db()
        
        for chunk_id in chunk_ids:
            # Obtener contenido del chunk
            surql = """
                SELECT content, metadata
                FROM chunk
                WHERE id = $chunk_id
            """
            
            result = await db.query(surql, {"chunk_id": chunk_id})
            
            if result and result[0].get("result"):
                chunk_data = result[0]["result"][0]
                content = chunk_data.get("content", "")
                
                if content:
                    # Re-vectorizar
                    await submit_vectorization(chunk_id, content)
        
        return True
        
    except ImportError:
        # vectorizer no disponible
        return False
    except Exception:
        return False


async def _update_task_status(task: RevectorizationTask) -> None:
    """Actualiza estado de tarea en DB."""
    try:
        db = await get_db()
        
        surql = """
            UPDATE revectorization_task SET
                status = $status,
                started_at = $started_at,
                completed_at = $completed_at
            WHERE task_id = $task_id
        """
        
        await db.query(surql, {
            "task_id": task.task_id,
            "status": task.status,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        })
        
    except Exception:
        pass


# ==========================================
# Actualización Basada en Feedback
# ==========================================

async def process_feedback(
    feedback: NormalizedFeedback,
    chunk_ids: List[str],
    concept_ids: List[str],
) -> Dict[str, Any]:
    """
    Procesa feedback y aplica actualizaciones al grafo.
    
    Args:
        feedback: Feedback normalizado
        chunk_ids: IDs de chunks afectados
        concept_ids: IDs de conceptos afectados
        
    Returns:
        Resumen de actualizaciones aplicadas
    """
    result = {
        "reinforced": [],
        "weakened": [],
        "revectorization_scheduled": False,
    }
    
    # Determinar acción basada en score
    if feedback.overall_score >= 0.6:
        # Feedback positivo -> reforzar
        updates = await reinforce_edges(chunk_ids, concept_ids, feedback)
        result["reinforced"] = [u.to_dict() for u in updates]
        
    elif feedback.overall_score <= 0.4:
        # Feedback negativo -> debilitar
        updates = await weaken_edges(chunk_ids, concept_ids, feedback)
        result["weakened"] = [u.to_dict() for u in updates]
        
        # Programar re-vectorización si es muy malo
        if feedback.overall_score <= 0.25 and chunk_ids:
            await schedule_revectorization(
                chunk_ids=chunk_ids,
                reason=f"very_low_score_{feedback.overall_score:.2f}",
                priority=2,
            )
            result["revectorization_scheduled"] = True
    
    return result


# ==========================================
# Análisis de Grafo
# ==========================================

async def get_weak_edges(
    threshold: float = 0.3,
    min_signals: int = 5,
) -> List[Dict[str, Any]]:
    """
    Obtiene aristas débiles que podrían necesitar revisión.
    
    Args:
        threshold: Umbral de peso
        min_signals: Mínimo de señales negativas
        
    Returns:
        Lista de aristas débiles
    """
    try:
        db = await get_db()
        
        surql = """
            SELECT 
                out as source,
                in as target,
                type,
                weight,
                negative_signals
            FROM edge
            WHERE weight < $threshold
            AND negative_signals >= $min_signals
            ORDER BY weight ASC
            LIMIT 50
        """
        
        result = await db.query(surql, {
            "threshold": threshold,
            "min_signals": min_signals,
        })
        
        if result and result[0].get("result"):
            return result[0]["result"]
            
    except Exception:
        pass
    
    return []


async def get_strong_edges(
    threshold: float = 0.8,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Obtiene aristas más fuertes del grafo.
    
    Args:
        threshold: Umbral de peso
        limit: Máximo de resultados
        
    Returns:
        Lista de aristas fuertes
    """
    try:
        db = await get_db()
        
        surql = """
            SELECT 
                out as source,
                in as target,
                type,
                weight,
                last_reinforced
            FROM edge
            WHERE weight >= $threshold
            ORDER BY weight DESC
            LIMIT $limit
        """
        
        result = await db.query(surql, {
            "threshold": threshold,
            "limit": limit,
        })
        
        if result and result[0].get("result"):
            return result[0]["result"]
            
    except Exception:
        pass
    
    return []


# ==========================================
# Utilidades
# ==========================================

def get_pending_tasks_summary() -> Dict[str, Any]:
    """
    Obtiene resumen de tareas pendientes.
    
    Returns:
        Diccionario con estadísticas
    """
    edge_updates = len(_pending_edge_updates)
    revectorizations = len([
        t for t in _pending_revectorizations 
        if t.status == "pending"
    ])
    
    return {
        "pending_edge_updates": edge_updates,
        "pending_revectorizations": revectorizations,
        "total_pending": edge_updates + revectorizations,
    }


async def apply_pending_updates() -> int:
    """
    Aplica todas las actualizaciones pendientes.
    
    Returns:
        Número de actualizaciones aplicadas
    """
    applied = 0
    
    # Aplicar edge updates
    for update in _pending_edge_updates:
        if not update.applied:
            try:
                db = await get_db()
                
                if update.action == UpdateAction.REINFORCE:
                    await _reinforce_edge(
                        db, update.source_id, update.target_id,
                        update.weight_delta, update.reason
                    )
                elif update.action == UpdateAction.WEAKEN:
                    await _weaken_edge(
                        db, update.source_id, update.target_id,
                        update.weight_delta, update.reason
                    )
                
                update.applied = True
                applied += 1
                
            except Exception:
                pass
    
    # Limpiar aplicadas
    _pending_edge_updates[:] = [u for u in _pending_edge_updates if not u.applied]
    
    return applied
