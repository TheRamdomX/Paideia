"""
orchestrator.py
Orquestador Central del Pipeline RAG Educativo.

Este es el ÚNICO controlador del flujo de procesamiento.
Implementa el loop: Router → Retrieval → Reasoning → Reflection

Responsabilidades:
- Coordinar agentes sin acoplamiento directo
- Implementar loop de reflexión controlado
- Aplicar ajustes de Reflection sin que Reflection ejecute
- Mantener el MODE como autoridad central

El orchestrator es STATELESS por diseño.
Estado de sesión se maneja en memory/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from backend.agents.mode_router import (
    LearningMode,
    ModeDetectionResult,
    detect_mode_with_details,
    is_mode_switch_request,
)
from backend.agents.retrieval_agent import (
    RetrievalResult,
    RetrievalStrategy,
    StrategyDecision,
)
from backend.agents.reasoning_agent import (
    GeneratedAnswer,
    PromptConfig,
)
from backend.agents.reflection_agent import (
    EvaluationResult,
    ReflectionDecision,
    RetryConfig,
)
from backend.pedagogy.mode_specs import (
    get_mode_spec,
    get_retrieval_config_for_mode,
    get_prompt_template_for_mode,
    is_read_only_mode,
    requires_metadata_only,
    should_include_conversation,
    get_temperature_for_mode,
    get_max_tokens_for_mode,
)
from backend.retrieval.hybrid_ranker import RankedResult


logger = logging.getLogger(__name__)


# ==========================================
# Configuración del Orchestrator
# ==========================================

@dataclass
class OrchestratorConfig:
    """Configuración del orquestador."""
    max_retries: int = 3
    min_acceptable_score: float = 0.5
    mode_alignment_threshold: float = 0.4
    enable_reflection_loop: bool = True
    fallback_on_max_retries: bool = True
    log_decisions: bool = True


@dataclass
class AdjustmentHints:
    """
    Sugerencias de ajuste del Reflection para el próximo ciclo.
    
    El Orchestrator APLICA estos ajustes.
    Reflection solo los SUGIERE.
    """
    expand_context: bool = False
    change_strategy: bool = False
    force_mode: Optional[LearningMode] = None
    additional_filters: Dict[str, Any] = field(default_factory=dict)
    increase_top_k: bool = False
    decrease_temperature: bool = False
    reroute: bool = False
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "expand_context": self.expand_context,
            "change_strategy": self.change_strategy,
            "force_mode": self.force_mode.value if self.force_mode else None,
            "additional_filters": self.additional_filters,
            "increase_top_k": self.increase_top_k,
            "decrease_temperature": self.decrease_temperature,
            "reroute": self.reroute,
            "reason": self.reason,
        }


@dataclass
class OrchestratorResult:
    """Resultado final del pipeline orquestado."""
    
    answer: str = ""
    mode: LearningMode = LearningMode.CONCEPT
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    
    # Metadatos del proceso
    iterations: int = 1
    final_decision: ReflectionDecision = ReflectionDecision.ACCEPT
    mode_changes: int = 0
    retrieval_expansions: int = 0
    
    # Debug info
    trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "mode": self.mode.value,
            "confidence": self.confidence,
            "sources": self.sources,
            "concepts": self.concepts,
            "iterations": self.iterations,
            "final_decision": self.final_decision.value,
            "mode_changes": self.mode_changes,
            "retrieval_expansions": self.retrieval_expansions,
        }


# ==========================================
# Orchestrator Principal
# ==========================================

class PipelineOrchestrator:
    """
    Orquestador central del pipeline RAG educativo.
    
    Flujo:
    1. Query → Mode Router (detectar intención)
    2. Mode → Retrieval Agent (obtener contexto)
    3. Context → Reasoning Agent (generar respuesta)
    4. Answer → Reflection Agent (evaluar)
    5. Reflection Decision → Loop o Return
    
    El MODE gobierna TODO el pipeline.
    Reflection controla los loops.
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self._trace: List[Dict[str, Any]] = []
    
    async def process(
        self,
        query: str,
        session_id: Optional[str] = None,
        student_id: Optional[str] = None,
        user_openai_key: Optional[str] = None,
        user_google_key: Optional[str] = None,
        preferred_provider: Optional[str] = None,
        user_model: Optional[str] = None,
        forced_mode: Optional[LearningMode] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """
        Procesa una query a través del pipeline completo.
        
        Args:
            query: Pregunta del estudiante
            session_id: ID de sesión para historial
            student_id: ID del estudiante para personalización
            user_openai_key: API key de OpenAI del usuario
            user_google_key: API key de Google del usuario
            preferred_provider: Proveedor preferido
            user_model: Modelo específico a usar
            forced_mode: Modo forzado (saltea router)
            context: Contexto adicional
            
        Returns:
            OrchestratorResult con la respuesta final
        """
        self._trace = []
        context = context or {}
        
        # ==========================================
        # PASO 1: Detectar Modo
        # ==========================================
        
        if forced_mode:
            mode = forced_mode
            mode_detection = ModeDetectionResult(
                mode=forced_mode,
                confidence=1.0,
                reason="forced_mode",
            )
        else:
            # Verificar si es solicitud explícita de cambio de modo
            explicit_mode = is_mode_switch_request(query)
            if explicit_mode:
                mode = explicit_mode
                mode_detection = ModeDetectionResult(
                    mode=explicit_mode,
                    confidence=1.0,
                    reason="explicit_mode_switch",
                )
            else:
                mode_detection = await detect_mode_with_details(query, context)
                mode = mode_detection.mode
        
        self._log_step("mode_detection", {
            "mode": mode.value,
            "confidence": mode_detection.confidence,
            "reason": mode_detection.reason,
        })
        
        # ==========================================
        # PASO 2: Obtener especificación pedagógica
        # ==========================================
        
        mode_spec = get_mode_spec(mode)
        retrieval_config = get_retrieval_config_for_mode(mode)
        prompt_config = get_prompt_template_for_mode(mode)
        
        self._log_step("mode_spec_loaded", {
            "mode": mode.value,
            "retrieval_config": retrieval_config,
            "is_read_only": is_read_only_mode(mode),
        })
        
        # ==========================================
        # PASO 3: Ejecutar Loop Principal
        # ==========================================
        
        result = await self._execute_loop(
            query=query,
            mode=mode,
            session_id=session_id,
            student_id=student_id,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
            preferred_provider=preferred_provider,
            user_model=user_model,
            retrieval_config=retrieval_config,
            prompt_config=prompt_config,
        )
        
        result.trace = self._trace
        return result
    
    async def _execute_loop(
        self,
        query: str,
        mode: LearningMode,
        session_id: Optional[str],
        student_id: Optional[str],
        user_openai_key: Optional[str],
        user_google_key: Optional[str],
        preferred_provider: Optional[str],
        user_model: Optional[str],
        retrieval_config: Dict[str, Any],
        prompt_config: Dict[str, Any],
    ) -> OrchestratorResult:
        """
        Ejecuta el loop principal de reflexión.
        
        Loop:
        1. Retrieve
        2. Reason
        3. Reflect
        4. Si ACCEPT → return
           Si MODE_MISMATCH → reroute + retry
           Si REQUEST_MORE_CONTEXT → expand + retry
           Si RETRY → retry
           Si FALLBACK → return fallback
        """
        # Importaciones tardías para evitar ciclos
        from backend.agents.retrieval_agent import retrieve as do_retrieve
        from backend.agents.reasoning_agent import generate_answer as do_reason
        from backend.agents.reflection_agent import evaluate_answer as do_reflect
        
        current_mode = mode
        iteration = 0
        mode_changes = 0
        retrieval_expansions = 0
        previous_evaluation: Optional[EvaluationResult] = None
        current_retrieval_config = retrieval_config.copy()
        
        # Construir estrategia inicial desde config
        strategy = self._build_strategy_from_config(current_mode, current_retrieval_config)
        
        best_answer: Optional[GeneratedAnswer] = None
        best_score = 0.0
        
        while iteration < self.config.max_retries:
            iteration += 1
            
            self._log_step(f"iteration_{iteration}_start", {
                "mode": current_mode.value,
                "retrieval_config": current_retrieval_config,
            })
            
            # ==========================================
            # RETRIEVE: Obtener contexto
            # ==========================================
            
            retrieval_result = await do_retrieve(
                query=query,
                student_id=student_id,
                strategy=strategy,
                openai_api_key=user_openai_key,
                google_api_key=user_google_key,
                mode=current_mode,
            )
            
            self._log_step(f"iteration_{iteration}_retrieve", {
                "total_found": retrieval_result.total_found,
                "top_score": retrieval_result.top_score,
            })
            
            # Verificar si hay contexto suficiente
            if not retrieval_result.results:
                self._log_step(f"iteration_{iteration}_no_context", {})
                return self._create_no_context_result(query, current_mode, iteration)
            
            # ==========================================
            # REASON: Generar respuesta
            # ==========================================
            
            # Construir PromptConfig para reasoning
            reasoning_config = PromptConfig(
                language="es",
                include_conversation=should_include_conversation(current_mode),
                temperature=get_temperature_for_mode(current_mode),
                max_tokens=get_max_tokens_for_mode(current_mode),
            )
            
            answer = await do_reason(
                query=query,
                context_chunks=retrieval_result.results,
                mode=current_mode,
                session_id=session_id if should_include_conversation(current_mode) else None,
                config=reasoning_config,
                user_openai_key=user_openai_key,
                user_google_key=user_google_key,
                preferred_provider=preferred_provider,
                user_model=user_model,
            )
            
            self._log_step(f"iteration_{iteration}_reason", {
                "answer_length": len(answer.answer),
                "confidence": answer.confidence,
            })
            
            # Guardar mejor respuesta
            if answer.confidence > best_score:
                best_answer = answer
                best_score = answer.confidence
            
            # ==========================================
            # Si reflexión está deshabilitada, retornar
            # ==========================================
            
            if not self.config.enable_reflection_loop:
                return self._create_result(
                    answer=answer,
                    mode=current_mode,
                    iteration=iteration,
                    decision=ReflectionDecision.ACCEPT,
                    mode_changes=mode_changes,
                    retrieval_expansions=retrieval_expansions,
                )
            
            # ==========================================
            # REFLECT: Evaluar respuesta
            # ==========================================
            
            evaluation = await do_reflect(
                query=query,
                answer=answer,
                context_chunks=retrieval_result.results,
                previous_evaluation=previous_evaluation,
                mode=current_mode,
            )
            
            self._log_step(f"iteration_{iteration}_reflect", {
                "overall_score": evaluation.overall_score,
                "decision": evaluation.decision.value,
                "mode_aligned": evaluation.mode_aligned,
            })
            
            # ==========================================
            # Procesar decisión de Reflection
            # ==========================================
            
            decision = evaluation.decision
            
            # ACCEPT: Respuesta aceptable
            if decision == ReflectionDecision.ACCEPT:
                return self._create_result(
                    answer=answer,
                    mode=current_mode,
                    iteration=iteration,
                    decision=decision,
                    mode_changes=mode_changes,
                    retrieval_expansions=retrieval_expansions,
                )
            
            # FALLBACK: No se puede mejorar
            if decision == ReflectionDecision.FALLBACK:
                # Usar mejor respuesta obtenida
                final_answer = best_answer if best_answer else answer
                return self._create_result(
                    answer=final_answer,
                    mode=current_mode,
                    iteration=iteration,
                    decision=decision,
                    mode_changes=mode_changes,
                    retrieval_expansions=retrieval_expansions,
                )
            
            # ==========================================
            # Calcular ajustes para siguiente iteración
            # ==========================================
            
            hints = self._compute_adjustment_hints(
                evaluation=evaluation,
                current_mode=current_mode,
                iteration=iteration,
            )
            
            self._log_step(f"iteration_{iteration}_hints", hints.to_dict())
            
            # Aplicar ajustes
            
            # MODE_MISMATCH: Cambiar modo y reintentar
            if decision == ReflectionDecision.MODE_MISMATCH and hints.reroute:
                if hints.force_mode and hints.force_mode != current_mode:
                    current_mode = hints.force_mode
                    current_retrieval_config = get_retrieval_config_for_mode(current_mode)
                    strategy = self._build_strategy_from_config(current_mode, current_retrieval_config)
                    mode_changes += 1
                    self._log_step(f"iteration_{iteration}_mode_change", {
                        "new_mode": current_mode.value,
                    })
            
            # REQUEST_MORE_CONTEXT: Expandir retrieval
            elif decision == ReflectionDecision.REQUEST_MORE_CONTEXT or hints.expand_context:
                current_retrieval_config = self._expand_retrieval_config(current_retrieval_config)
                strategy = self._build_strategy_from_config(current_mode, current_retrieval_config)
                retrieval_expansions += 1
                self._log_step(f"iteration_{iteration}_expand_context", {
                    "new_top_k": current_retrieval_config.get("top_k"),
                })
            
            # RETRY/EXPAND/SIMPLIFY: Ajustar configuración
            else:
                if hints.change_strategy:
                    strategy = self._adjust_strategy(strategy, hints)
                if hints.decrease_temperature:
                    # Se aplicará en la siguiente llamada a reasoning
                    pass
            
            # Guardar evaluación para siguiente iteración
            previous_evaluation = evaluation
        
        # ==========================================
        # Máximo de reintentos alcanzado
        # ==========================================
        
        self._log_step("max_retries_reached", {
            "iterations": iteration,
        })
        
        if self.config.fallback_on_max_retries and best_answer:
            return self._create_result(
                answer=best_answer,
                mode=current_mode,
                iteration=iteration,
                decision=ReflectionDecision.FALLBACK,
                mode_changes=mode_changes,
                retrieval_expansions=retrieval_expansions,
            )
        
        # Fallback final
        return self._create_fallback_result(query, current_mode, iteration)
    
    # ==========================================
    # Métodos de construcción de estrategia
    # ==========================================
    
    def _build_strategy_from_config(
        self,
        mode: LearningMode,
        config: Dict[str, Any],
    ) -> StrategyDecision:
        """Construye StrategyDecision desde config de modo."""
        # Determinar estrategia de retrieval
        if config.get("metadata_only", False):
            strategy = RetrievalStrategy.METADATA_ONLY
        elif config.get("graph_weight", 0) >= 0.4:
            strategy = RetrievalStrategy.GRAPH_ENHANCED
        else:
            strategy = RetrievalStrategy.HYBRID
        
        return StrategyDecision(
            strategy=strategy,
            mode=mode,
            vector_weight=config.get("vector_weight", 0.35),
            bm25_weight=config.get("bm25_weight", 0.25),
            graph_weight=config.get("graph_weight", 0.4),
            top_k=config.get("top_k", 10),
            min_score=config.get("min_score", 0.3),
            expand_graph=config.get("expand_concepts", True),
            chunk_type_filter=config.get("chunk_type_filter"),
            metadata_only=config.get("metadata_only", False),
        )
    
    def _expand_retrieval_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expande la configuración de retrieval para obtener más contexto."""
        expanded = config.copy()
        expanded["top_k"] = min(30, expanded.get("top_k", 10) + 5)
        expanded["min_score"] = max(0.1, expanded.get("min_score", 0.3) - 0.1)
        expanded["expand_concepts"] = True
        return expanded
    
    def _adjust_strategy(
        self,
        strategy: StrategyDecision,
        hints: AdjustmentHints,
    ) -> StrategyDecision:
        """Ajusta la estrategia según hints."""
        # Determinar el modo a usar
        new_mode = hints.force_mode if hints.force_mode else strategy.mode
        
        # Crear nueva estrategia con ajustes
        return StrategyDecision(
            strategy=strategy.strategy,
            mode=new_mode,
            vector_weight=strategy.vector_weight,
            bm25_weight=strategy.bm25_weight,
            graph_weight=strategy.graph_weight,
            top_k=strategy.top_k + 5 if hints.increase_top_k else strategy.top_k,
            min_score=strategy.min_score,
            expand_graph=True if hints.expand_context else strategy.expand_graph,
            chunk_type_filter=strategy.chunk_type_filter,
            metadata_only=strategy.metadata_only,
        )
    
    # ==========================================
    # Cálculo de adjustment hints
    # ==========================================
    
    def _compute_adjustment_hints(
        self,
        evaluation: EvaluationResult,
        current_mode: LearningMode,
        iteration: int,
    ) -> AdjustmentHints:
        """
        Calcula hints de ajuste basado en la evaluación.
        
        El Orchestrator CALCULA estos hints.
        Reflection solo EVALÚA y DECIDE.
        """
        hints = AdjustmentHints()
        
        decision = evaluation.decision
        
        if decision == ReflectionDecision.MODE_MISMATCH:
            hints.reroute = True
            # Determinar modo correcto basado en evaluación
            # Si estamos en EXERCISE_LIST y hubo explicación, mantener en EXERCISE_LIST pero reforzar
            # Si estamos en otro modo, verificar qué modo esperaba el estudiante
            if current_mode == LearningMode.EXERCISE_LIST:
                # El modo es correcto, pero el LLM explicó. Reforzar.
                hints.force_mode = LearningMode.EXERCISE_LIST
                hints.decrease_temperature = True
            else:
                # Inferir modo correcto de las dimensiones
                hints.force_mode = self._infer_correct_mode(evaluation, current_mode)
            hints.reason = f"mode_mismatch_reroute_to_{hints.force_mode.value if hints.force_mode else 'unknown'}"
        
        elif decision == ReflectionDecision.REQUEST_MORE_CONTEXT:
            hints.expand_context = True
            hints.increase_top_k = True
            hints.reason = "request_more_context"
        
        elif decision == ReflectionDecision.RETRY:
            hints.change_strategy = True
            hints.decrease_temperature = True
            hints.reason = "retry_with_adjustments"
        
        elif decision == ReflectionDecision.EXPAND:
            hints.expand_context = True
            hints.reason = "expand_response"
        
        elif decision == ReflectionDecision.SIMPLIFY:
            hints.decrease_temperature = True
            hints.reason = "simplify_response"
        
        return hints
    
    def _infer_correct_mode(
        self,
        evaluation: EvaluationResult,
        current_mode: LearningMode,
    ) -> LearningMode:
        """Infiere el modo correcto basado en la evaluación."""
        # Si Reflection sugirió un modo, usarlo
        if evaluation.hints.suggested_mode:
            return evaluation.hints.suggested_mode
        # Por defecto, mantener el modo actual
        return current_mode
    
    # ==========================================
    # Creación de resultados
    # ==========================================
    
    def _create_result(
        self,
        answer: GeneratedAnswer,
        mode: LearningMode,
        iteration: int,
        decision: ReflectionDecision,
        mode_changes: int,
        retrieval_expansions: int,
    ) -> OrchestratorResult:
        """Crea resultado exitoso."""
        return OrchestratorResult(
            answer=answer.answer,
            mode=mode,
            confidence=answer.confidence,
            sources=answer.sources,
            concepts=answer.concepts_covered,
            iterations=iteration,
            final_decision=decision,
            mode_changes=mode_changes,
            retrieval_expansions=retrieval_expansions,
        )
    
    def _create_no_context_result(
        self,
        query: str,
        mode: LearningMode,
        iteration: int,
    ) -> OrchestratorResult:
        """Crea resultado cuando no hay contexto."""
        no_context_messages = {
            LearningMode.CONCEPT: "No encontré información sobre este concepto en los documentos. Por favor, sube material relacionado.",
            LearningMode.PRACTICE: "No encontré ejercicios resueltos sobre este tema. Por favor, sube material con ejemplos.",
            LearningMode.EXERCISE_LIST: "No encontré ejercicios en los documentos cargados. Te sugiero subir material con ejercicios prácticos.",
        }
        
        return OrchestratorResult(
            answer=no_context_messages.get(mode, "No encontré información relevante."),
            mode=mode,
            confidence=0.0,
            iterations=iteration,
            final_decision=ReflectionDecision.FALLBACK,
        )
    
    def _create_fallback_result(
        self,
        query: str,
        mode: LearningMode,
        iteration: int,
    ) -> OrchestratorResult:
        """Crea resultado fallback."""
        return OrchestratorResult(
            answer="Lo siento, no pude generar una respuesta satisfactoria. Por favor, intenta reformular tu pregunta.",
            mode=mode,
            confidence=0.0,
            iterations=iteration,
            final_decision=ReflectionDecision.FALLBACK,
        )
    
    # ==========================================
    # Logging
    # ==========================================
    
    def _log_step(self, step_name: str, data: Dict[str, Any]) -> None:
        """Registra un paso del proceso."""
        entry = {
            "step": step_name,
            **data,
        }
        self._trace.append(entry)
        
        if self.config.log_decisions:
            logger.debug(f"Orchestrator [{step_name}]: {data}")


# ==========================================
# Funciones de conveniencia
# ==========================================

async def orchestrate(
    query: str,
    session_id: Optional[str] = None,
    student_id: Optional[str] = None,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
    preferred_provider: Optional[str] = None,
    user_model: Optional[str] = None,
    forced_mode: Optional[LearningMode] = None,
    config: Optional[OrchestratorConfig] = None,
) -> OrchestratorResult:
    """
    Función de conveniencia para procesar una query.
    
    Crea un orchestrator y procesa la query en un solo paso.
    
    Args:
        query: Pregunta del estudiante
        session_id: ID de sesión
        student_id: ID del estudiante
        user_openai_key: API key de OpenAI
        user_google_key: API key de Google
        preferred_provider: Proveedor preferido
        user_model: Modelo específico
        forced_mode: Modo forzado
        config: Configuración del orchestrator
        
    Returns:
        OrchestratorResult con la respuesta
    """
    orchestrator = PipelineOrchestrator(config)
    return await orchestrator.process(
        query=query,
        session_id=session_id,
        student_id=student_id,
        user_openai_key=user_openai_key,
        user_google_key=user_google_key,
        preferred_provider=preferred_provider,
        user_model=user_model,
        forced_mode=forced_mode,
    )


async def quick_query(
    query: str,
    mode: Optional[LearningMode] = None,
) -> str:
    """
    Query rápida sin configuración avanzada.
    
    Args:
        query: Pregunta
        mode: Modo opcional
        
    Returns:
        Texto de la respuesta
    """
    result = await orchestrate(query, forced_mode=mode)
    return result.answer
