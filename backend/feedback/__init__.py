"""
Feedback package.
Señales, analytics y actualizaciones del grafo.
"""

from backend.feedback.signals import (
    SignalType,
    SignalCategory,
    SentimentPolarity,
    FeedbackSignal,
    NormalizedFeedback,
    parse_feedback,
    weight_feedback,
    parse_multiple_feedbacks,
    filter_signals_by_confidence,
    get_signals_for_chunk,
    get_signals_for_concept,
    detect_feedback_pattern,
)

from backend.feedback.analytics import (
    ContentQuality,
    ChunkMetrics,
    ConceptMetrics,
    SystemMetrics,
    detect_poor_chunks,
    track_confusion,
    get_confusing_concepts,
    aggregate_metrics,
    record_feedback,
    record_chunk_retrieval,
    get_chunk_quality_distribution,
    get_trending_concepts,
    clear_analytics_state,
)

from backend.feedback.graph_updates import (
    UpdateAction,
    EdgeType,
    EdgeUpdate,
    RevectorizationTask,
    reinforce_edges,
    weaken_edges,
    schedule_revectorization,
    process_revectorization_queue,
    process_feedback,
    get_weak_edges,
    get_strong_edges,
    get_pending_tasks_summary,
    apply_pending_updates,
    clear_pending_tasks,
)

__all__ = [
    # signals
    "SignalType",
    "SignalCategory",
    "SentimentPolarity",
    "FeedbackSignal",
    "NormalizedFeedback",
    "parse_feedback",
    "weight_feedback",
    "parse_multiple_feedbacks",
    "filter_signals_by_confidence",
    "get_signals_for_chunk",
    "get_signals_for_concept",
    "detect_feedback_pattern",
    # analytics
    "ContentQuality",
    "ChunkMetrics",
    "ConceptMetrics",
    "SystemMetrics",
    "detect_poor_chunks",
    "track_confusion",
    "get_confusing_concepts",
    "aggregate_metrics",
    "record_feedback",
    "record_chunk_retrieval",
    "get_chunk_quality_distribution",
    "get_trending_concepts",
    "clear_analytics_state",
    # graph_updates
    "UpdateAction",
    "EdgeType",
    "EdgeUpdate",
    "RevectorizationTask",
    "reinforce_edges",
    "weaken_edges",
    "schedule_revectorization",
    "process_revectorization_queue",
    "process_feedback",
    "get_weak_edges",
    "get_strong_edges",
    "get_pending_tasks_summary",
    "apply_pending_updates",
    "clear_pending_tasks",
]
