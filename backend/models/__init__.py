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
]
