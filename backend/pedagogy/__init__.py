"""
pedagogy - Especificaciones pedagógicas para el sistema RAG educativo.
"""

from backend.pedagogy.mode_specs import (
    ModeSpec,
    CONCEPT_SPEC,
    PRACTICE_SPEC,
    EXERCISE_LIST_SPEC,
    get_mode_spec,
    validate_response_for_mode,
    get_retrieval_config_for_mode,
    get_prompt_template_for_mode,
)

__all__ = [
    "ModeSpec",
    "CONCEPT_SPEC",
    "PRACTICE_SPEC",
    "EXERCISE_LIST_SPEC",
    "get_mode_spec",
    "validate_response_for_mode",
    "get_retrieval_config_for_mode",
    "get_prompt_template_for_mode",
]
