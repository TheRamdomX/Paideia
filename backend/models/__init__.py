"""
Models package.
Abstracciones sobre LLM, embeddings y STT.
"""

from backend.models.llm import (
    BaseLLM,
    OpenAILLM,
    GoogleLLM,
    generate,
    generate_stream,
    get_llm,
    select_model,
    handle_fallback,
)

from backend.models.embeddings import (
    BaseEmbedding,
    OpenAIEmbedding,
    GoogleEmbedding,
    embed_text,
    batch_embed,
    get_embedding_model,
    get_embedding_dimension,
    select_embedding_model,
)

from backend.models.model_limits import (
    ModelConfig,
    OPENAI_MODELS,
    GOOGLE_MODELS,
    ALL_MODELS,
    get_model_config as get_model_limits,
    get_context_window,
    get_max_output_tokens,
    get_safe_context_limit,
    list_models_by_provider,
    get_models_summary,
    validate_token_count,
    get_default_model,
)

__all__ = [
    # LLM
    "BaseLLM",
    "OpenAILLM",
    "GoogleLLM",
    "generate",
    "generate_stream",
    "get_llm",
    "select_model",
    "handle_fallback",
    # Embeddings
    "BaseEmbedding",
    "OpenAIEmbedding",
    "GoogleEmbedding",
    "embed_text",
    "batch_embed",
    "get_embedding_model",
    "get_embedding_dimension",
    "select_embedding_model",
    # Model Limits
    "ModelConfig",
    "OPENAI_MODELS",
    "GOOGLE_MODELS",
    "ALL_MODELS",
    "get_model_limits",
    "get_context_window",
    "get_max_output_tokens",
    "get_safe_context_limit",
    "list_models_by_provider",
    "get_models_summary",
    "validate_token_count",
    "get_default_model",
]
