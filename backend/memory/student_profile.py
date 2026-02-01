"""
student_profile.py
Perfil del estudiante: nivel, objetivos, debilidades detectadas.
Implementa adaptación personalizada del contenido educativo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from backend.db.surreal import execute, get_db
from backend.settings import get_rag_config


# ==========================================
# Enums y Estructuras de Datos
# ==========================================

class ProficiencyLevel(str, Enum):
    """Niveles de competencia del estudiante."""
    BEGINNER = "beginner"
    ELEMENTARY = "elementary"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class LearningStyle(str, Enum):
    """Estilos de aprendizaje."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    READING = "reading"
    KINESTHETIC = "kinesthetic"


@dataclass
class ConceptMastery:
    """Nivel de dominio de un concepto."""
    
    concept_id: str = ""
    concept_name: str = ""
    mastery_score: float = 0.0  # 0-1
    interactions: int = 0
    last_reviewed: Optional[datetime] = None
    is_weakness: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "mastery_score": self.mastery_score,
            "interactions": self.interactions,
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "is_weakness": self.is_weakness,
        }


@dataclass
class StudentProfile:
    """Perfil completo de un estudiante."""
    
    student_id: str = ""
    name: str = ""
    email: str = ""
    
    # Nivel general
    level: ProficiencyLevel = ProficiencyLevel.BEGINNER
    learning_style: LearningStyle = LearningStyle.READING
    
    # Dominio por concepto
    concept_mastery: Dict[str, ConceptMastery] = field(default_factory=dict)
    
    # Objetivos y preferencias
    goals: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    
    # Debilidades detectadas
    weaknesses: List[str] = field(default_factory=list)
    
    # Estadísticas
    total_sessions: int = 0
    total_questions: int = 0
    correct_answers: int = 0
    average_response_quality: float = 0.0
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "name": self.name,
            "email": self.email,
            "level": self.level.value,
            "learning_style": self.learning_style.value,
            "concept_mastery": {
                k: v.to_dict() for k, v in self.concept_mastery.items()
            },
            "goals": self.goals,
            "interests": self.interests,
            "weaknesses": self.weaknesses,
            "total_sessions": self.total_sessions,
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "average_response_quality": self.average_response_quality,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "metadata": self.metadata,
        }
    
    @property
    def accuracy_rate(self) -> float:
        """Tasa de precisión del estudiante."""
        if self.total_questions == 0:
            return 0.0
        return self.correct_answers / self.total_questions


# ==========================================
# Carga de Perfil
# ==========================================

async def load_profile(student_id: str) -> Optional[StudentProfile]:
    """
    Carga perfil del estudiante desde la base de datos.
    
    Args:
        student_id: ID del estudiante
        
    Returns:
        StudentProfile o None si no existe
    """
    try:
        db = await get_db()
        
        surql = """
            SELECT * FROM student WHERE id = $student_id
        """
        
        result = await db.query(surql, {"student_id": f"student:{student_id}"})
        
        if not result or not result[0].get("result"):
            return None
        
        data = result[0]["result"][0]
        
        # Construir perfil desde datos
        profile = StudentProfile(
            student_id=student_id,
            name=data.get("name", ""),
            email=data.get("email", ""),
            level=ProficiencyLevel(data.get("level", "beginner")),
            learning_style=LearningStyle(data.get("learning_style", "reading")),
            goals=data.get("goals", []),
            interests=data.get("interests", []),
            weaknesses=data.get("weaknesses", []),
            total_sessions=data.get("total_sessions", 0),
            total_questions=data.get("total_questions", 0),
            correct_answers=data.get("correct_answers", 0),
            average_response_quality=data.get("average_response_quality", 0.0),
            metadata=data.get("metadata", {}),
        )
        
        # Parsear timestamps
        if data.get("created_at"):
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            profile.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("last_active"):
            profile.last_active = datetime.fromisoformat(data["last_active"])
        
        # Cargar dominio de conceptos
        profile.concept_mastery = await _load_concept_mastery(db, student_id)
        
        return profile
        
    except Exception as e:
        return None


async def _load_concept_mastery(
    db,
    student_id: str,
) -> Dict[str, ConceptMastery]:
    """Carga el dominio de conceptos del estudiante."""
    try:
        surql = """
            SELECT 
                concept_id,
                concept_name,
                mastery_score,
                interactions,
                last_reviewed,
                is_weakness
            FROM student_concept
            WHERE student_id = $student_id
        """
        
        result = await db.query(surql, {"student_id": f"student:{student_id}"})
        
        mastery = {}
        
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                cm = ConceptMastery(
                    concept_id=row.get("concept_id", ""),
                    concept_name=row.get("concept_name", ""),
                    mastery_score=float(row.get("mastery_score", 0.0)),
                    interactions=row.get("interactions", 0),
                    is_weakness=row.get("is_weakness", False),
                )
                
                if row.get("last_reviewed"):
                    cm.last_reviewed = datetime.fromisoformat(row["last_reviewed"])
                
                mastery[cm.concept_id] = cm
        
        return mastery
        
    except Exception:
        return {}


async def create_profile(
    student_id: str,
    name: str = "",
    email: str = "",
    level: ProficiencyLevel = ProficiencyLevel.BEGINNER,
) -> StudentProfile:
    """
    Crea un nuevo perfil de estudiante.
    
    Args:
        student_id: ID del estudiante
        name: Nombre
        email: Email
        level: Nivel inicial
        
    Returns:
        Perfil creado
    """
    now = datetime.now(timezone.utc)
    
    profile = StudentProfile(
        student_id=student_id,
        name=name,
        email=email,
        level=level,
        created_at=now,
        updated_at=now,
        last_active=now,
    )
    
    try:
        db = await get_db()
        
        surql = """
            CREATE student SET
                id = $id,
                name = $name,
                email = $email,
                level = $level,
                learning_style = $learning_style,
                goals = $goals,
                interests = $interests,
                weaknesses = $weaknesses,
                total_sessions = $total_sessions,
                total_questions = $total_questions,
                correct_answers = $correct_answers,
                average_response_quality = $average_response_quality,
                created_at = $created_at,
                updated_at = $updated_at,
                last_active = $last_active,
                metadata = $metadata
        """
        
        await db.query(surql, {
            "id": f"student:{student_id}",
            "name": name,
            "email": email,
            "level": level.value,
            "learning_style": profile.learning_style.value,
            "goals": [],
            "interests": [],
            "weaknesses": [],
            "total_sessions": 0,
            "total_questions": 0,
            "correct_answers": 0,
            "average_response_quality": 0.0,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "last_active": now.isoformat(),
            "metadata": {},
        })
        
    except Exception:
        pass
    
    return profile


# ==========================================
# Actualización de Perfil
# ==========================================

async def update_profile(
    profile: StudentProfile,
    feedback_score: Optional[float] = None,
    concept_interactions: Optional[Dict[str, float]] = None,
) -> StudentProfile:
    """
    Ajusta perfil del estudiante según feedback e interacciones.
    
    Args:
        profile: Perfil actual
        feedback_score: Score de feedback reciente (0-1)
        concept_interactions: Dict de concepto_id -> score de interacción
        
    Returns:
        Perfil actualizado
    """
    now = datetime.now(timezone.utc)
    profile.updated_at = now
    profile.last_active = now
    
    # Actualizar estadísticas si hay feedback
    if feedback_score is not None:
        profile.total_questions += 1
        
        if feedback_score >= 0.7:
            profile.correct_answers += 1
        
        # Actualizar promedio con media móvil exponencial
        alpha = 0.1
        profile.average_response_quality = (
            alpha * feedback_score + 
            (1 - alpha) * profile.average_response_quality
        )
    
    # Actualizar dominio de conceptos
    if concept_interactions:
        for concept_id, score in concept_interactions.items():
            if concept_id in profile.concept_mastery:
                cm = profile.concept_mastery[concept_id]
                cm.interactions += 1
                cm.last_reviewed = now
                
                # Actualizar mastery con media móvil
                alpha = 0.2
                cm.mastery_score = alpha * score + (1 - alpha) * cm.mastery_score
                
                # Detectar si es debilidad
                cm.is_weakness = cm.mastery_score < 0.4 and cm.interactions >= 3
            else:
                # Nuevo concepto
                profile.concept_mastery[concept_id] = ConceptMastery(
                    concept_id=concept_id,
                    mastery_score=score,
                    interactions=1,
                    last_reviewed=now,
                    is_weakness=score < 0.4,
                )
    
    # Ajustar nivel basado en rendimiento
    profile.level = _calculate_level(profile)
    
    # Persistir cambios
    await _save_profile(profile)
    
    return profile


def _calculate_level(profile: StudentProfile) -> ProficiencyLevel:
    """Calcula nivel basado en estadísticas."""
    # Calcular score general
    accuracy = profile.accuracy_rate
    avg_mastery = _average_mastery(profile)
    
    # Combinar métricas
    overall_score = (accuracy * 0.4 + avg_mastery * 0.6)
    
    if overall_score >= 0.9:
        return ProficiencyLevel.EXPERT
    elif overall_score >= 0.75:
        return ProficiencyLevel.ADVANCED
    elif overall_score >= 0.5:
        return ProficiencyLevel.INTERMEDIATE
    elif overall_score >= 0.25:
        return ProficiencyLevel.ELEMENTARY
    else:
        return ProficiencyLevel.BEGINNER


def _average_mastery(profile: StudentProfile) -> float:
    """Calcula promedio de mastery."""
    if not profile.concept_mastery:
        return 0.0
    
    total = sum(cm.mastery_score for cm in profile.concept_mastery.values())
    return total / len(profile.concept_mastery)


async def _save_profile(profile: StudentProfile) -> None:
    """Persiste perfil en la base de datos."""
    try:
        db = await get_db()
        
        surql = """
            UPDATE student SET
                name = $name,
                email = $email,
                level = $level,
                learning_style = $learning_style,
                goals = $goals,
                interests = $interests,
                weaknesses = $weaknesses,
                total_sessions = $total_sessions,
                total_questions = $total_questions,
                correct_answers = $correct_answers,
                average_response_quality = $average_response_quality,
                updated_at = $updated_at,
                last_active = $last_active,
                metadata = $metadata
            WHERE id = $id
        """
        
        await db.query(surql, {
            "id": f"student:{profile.student_id}",
            "name": profile.name,
            "email": profile.email,
            "level": profile.level.value,
            "learning_style": profile.learning_style.value,
            "goals": profile.goals,
            "interests": profile.interests,
            "weaknesses": profile.weaknesses,
            "total_sessions": profile.total_sessions,
            "total_questions": profile.total_questions,
            "correct_answers": profile.correct_answers,
            "average_response_quality": profile.average_response_quality,
            "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
            "last_active": profile.last_active.isoformat() if profile.last_active else None,
            "metadata": profile.metadata,
        })
        
        # Guardar concept_mastery
        for concept_id, cm in profile.concept_mastery.items():
            await _save_concept_mastery(db, profile.student_id, cm)
            
    except Exception:
        pass


async def _save_concept_mastery(
    db,
    student_id: str,
    cm: ConceptMastery,
) -> None:
    """Guarda mastery de un concepto."""
    surql = """
        UPSERT student_concept SET
            student_id = $student_id,
            concept_id = $concept_id,
            concept_name = $concept_name,
            mastery_score = $mastery_score,
            interactions = $interactions,
            last_reviewed = $last_reviewed,
            is_weakness = $is_weakness
        WHERE student_id = $student_id AND concept_id = $concept_id
    """
    
    await db.query(surql, {
        "student_id": f"student:{student_id}",
        "concept_id": cm.concept_id,
        "concept_name": cm.concept_name,
        "mastery_score": cm.mastery_score,
        "interactions": cm.interactions,
        "last_reviewed": cm.last_reviewed.isoformat() if cm.last_reviewed else None,
        "is_weakness": cm.is_weakness,
    })


# ==========================================
# Detección de Debilidades
# ==========================================

async def infer_weaknesses(
    profile: StudentProfile,
    recent_interactions: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Detecta vacíos de conocimiento basados en el historial.
    
    Args:
        profile: Perfil del estudiante
        recent_interactions: Interacciones recientes (opcional)
        
    Returns:
        Lista de IDs de conceptos identificados como debilidades
    """
    weaknesses: Set[str] = set()
    
    # 1. Conceptos con bajo mastery
    for concept_id, cm in profile.concept_mastery.items():
        if cm.is_weakness or (cm.mastery_score < 0.4 and cm.interactions >= 2):
            weaknesses.add(concept_id)
    
    # 2. Conceptos con decaimiento temporal
    now = datetime.now(timezone.utc)
    for concept_id, cm in profile.concept_mastery.items():
        if cm.last_reviewed:
            days_since = (now - cm.last_reviewed).days
            
            # Aplicar curva de olvido de Ebbinghaus simplificada
            if days_since > 7 and cm.mastery_score < 0.7:
                weaknesses.add(concept_id)
            elif days_since > 30 and cm.mastery_score < 0.85:
                weaknesses.add(concept_id)
    
    # 3. Analizar interacciones recientes
    if recent_interactions:
        confusion_concepts = _detect_confusion_from_interactions(recent_interactions)
        weaknesses.update(confusion_concepts)
    
    # 4. Detectar prerequisitos no dominados
    prerequisite_gaps = await _find_prerequisite_gaps(profile, weaknesses)
    weaknesses.update(prerequisite_gaps)
    
    # Actualizar perfil
    profile.weaknesses = list(weaknesses)
    
    return list(weaknesses)


def _detect_confusion_from_interactions(
    interactions: List[Dict[str, Any]],
) -> Set[str]:
    """Detecta conceptos confusos de interacciones recientes."""
    confusion_counts: Dict[str, int] = {}
    
    for interaction in interactions:
        # Buscar señales de confusión
        feedback = interaction.get("feedback", {})
        
        if feedback.get("confused") or feedback.get("score", 1.0) < 0.5:
            concepts = interaction.get("concepts", [])
            
            for concept in concepts:
                confusion_counts[concept] = confusion_counts.get(concept, 0) + 1
    
    # Conceptos con múltiples señales de confusión
    return {
        concept for concept, count in confusion_counts.items()
        if count >= 2
    }


async def _find_prerequisite_gaps(
    profile: StudentProfile,
    known_weaknesses: Set[str],
) -> Set[str]:
    """Encuentra prerequisitos no dominados."""
    gaps: Set[str] = set()
    
    try:
        db = await get_db()
        
        # Buscar prerequisitos de conceptos dominados pero con problemas
        for concept_id, cm in profile.concept_mastery.items():
            if cm.mastery_score >= 0.5 and concept_id not in known_weaknesses:
                continue
            
            # Buscar prerequisitos
            surql = """
                SELECT in.id as prereq_id
                FROM edge
                WHERE out = $concept_id AND type = 'prerequisite'
            """
            
            result = await db.query(surql, {"concept_id": concept_id})
            
            if result and result[0].get("result"):
                for row in result[0]["result"]:
                    prereq_id = row.get("prereq_id", "")
                    
                    # Verificar si el prerequisito está dominado
                    if prereq_id and prereq_id not in profile.concept_mastery:
                        gaps.add(prereq_id)
                    elif prereq_id in profile.concept_mastery:
                        prereq_cm = profile.concept_mastery[prereq_id]
                        if prereq_cm.mastery_score < 0.6:
                            gaps.add(prereq_id)
                            
    except Exception:
        pass
    
    return gaps


# ==========================================
# Utilidades
# ==========================================

async def get_recommended_concepts(
    profile: StudentProfile,
    limit: int = 5,
) -> List[str]:
    """
    Obtiene conceptos recomendados para el estudiante.
    
    Args:
        profile: Perfil del estudiante
        limit: Número máximo de recomendaciones
        
    Returns:
        Lista de IDs de conceptos recomendados
    """
    # Priorizar debilidades
    recommendations = list(profile.weaknesses)[:limit]
    
    if len(recommendations) >= limit:
        return recommendations
    
    # Añadir conceptos con bajo mastery pero no debilidades
    for concept_id, cm in sorted(
        profile.concept_mastery.items(),
        key=lambda x: x[1].mastery_score
    ):
        if concept_id not in recommendations:
            recommendations.append(concept_id)
            
            if len(recommendations) >= limit:
                break
    
    return recommendations


def get_level_config(level: ProficiencyLevel) -> Dict[str, Any]:
    """
    Obtiene configuración basada en nivel.
    
    Returns:
        Dict con parámetros adaptados al nivel
    """
    configs = {
        ProficiencyLevel.BEGINNER: {
            "max_complexity": 0.3,
            "explanation_depth": "detailed",
            "use_analogies": True,
            "vocabulary_level": "simple",
            "chunk_size": "small",
        },
        ProficiencyLevel.ELEMENTARY: {
            "max_complexity": 0.5,
            "explanation_depth": "moderate",
            "use_analogies": True,
            "vocabulary_level": "basic",
            "chunk_size": "medium",
        },
        ProficiencyLevel.INTERMEDIATE: {
            "max_complexity": 0.7,
            "explanation_depth": "standard",
            "use_analogies": False,
            "vocabulary_level": "intermediate",
            "chunk_size": "medium",
        },
        ProficiencyLevel.ADVANCED: {
            "max_complexity": 0.9,
            "explanation_depth": "concise",
            "use_analogies": False,
            "vocabulary_level": "advanced",
            "chunk_size": "large",
        },
        ProficiencyLevel.EXPERT: {
            "max_complexity": 1.0,
            "explanation_depth": "brief",
            "use_analogies": False,
            "vocabulary_level": "technical",
            "chunk_size": "large",
        },
    }
    
    return configs.get(level, configs[ProficiencyLevel.INTERMEDIATE])
