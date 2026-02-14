"""
Agents package.
Agentes de orquestación, mode routing, retrieval, reasoning y reflection.

Arquitectura:
- Orchestrator: Controlador central del pipeline
- Mode Router: Detecta intención pedagógica
- Retrieval Agent: Obtiene contexto (pasivo)
- Reasoning Agent: Genera respuestas (pasivo)
- Reflection Agent: Evalúa y decide (supervisor)
"""

from backend.agents.mode_router import (
    LearningMode,
    ModeDetectionResult,
    detect_mode,
    detect_mode_with_details,
    get_mode_description,
    is_mode_switch_request,
)

from backend.agents.orchestrator import (
    PipelineOrchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    AdjustmentHints,
    orchestrate,
    quick_query,
)

from backend.agents.retrieval_agent import (
    RetrievalStrategy,
    StrategyDecision,
    RetrievalResult,
    retrieve,
    build_strategy_for_mode,
)

from backend.agents.reasoning_agent import (
    PromptConfig,
    GeneratedAnswer,
    generate_answer,
)

from backend.agents.reflection_agent import (
    EvaluationDimension,
    ReflectionDecision,
    EvaluationResult,
    evaluate_answer,
    should_retry,
)

__all__ = [
    # Mode Router
    "LearningMode",
    "ModeDetectionResult",
    "detect_mode",
    "detect_mode_with_details",
    "get_mode_description",
    "is_mode_switch_request",
    # Orchestrator
    "PipelineOrchestrator",
    "OrchestratorConfig",
    "OrchestratorResult",
    "AdjustmentHints",
    "orchestrate",
    "quick_query",
    # Retrieval
    "RetrievalStrategy",
    "StrategyDecision",
    "RetrievalResult",
    "retrieve",
    "build_strategy_for_mode",
    # Reasoning
    "PromptConfig",
    "GeneratedAnswer",
    "generate_answer",
    # Reflection
    "EvaluationDimension",
    "ReflectionDecision",
    "EvaluationResult",
    "evaluate_answer",
    "should_retry",
]
