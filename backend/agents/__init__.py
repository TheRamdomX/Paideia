"""
Agents package.
Agentes de mode routing, retrieval, reasoning y reflection.
"""

from backend.agents.mode_router import (
    LearningMode,
    ModeDetectionResult,
    detect_mode,
    detect_mode_with_details,
    get_mode_description,
    is_mode_switch_request,
)

__all__ = [
    "LearningMode",
    "ModeDetectionResult",
    "detect_mode",
    "detect_mode_with_details",
    "get_mode_description",
    "is_mode_switch_request",
]
