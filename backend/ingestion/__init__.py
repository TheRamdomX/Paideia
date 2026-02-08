"""
Ingestion package.
Procesamiento de contenido, chunking y vectorización.
"""

from backend.ingestion.content_processor import (
    ContentType,
    ProcessedContent,
    process_content,
    extract_title_from_content,
    validate_content,
)
from backend.ingestion.chunking import (
    Chunk,
    ChunkingResult,
    hierarchical_chunk,
    create_overlapping_chunks,
    validate_chunks,
)
from backend.ingestion.vectorizer import (
    VectorizedChunk,
    VectorizationQueue,
    embed_chunks_batch,
    submit_vectorization,
)

__all__ = [
    "ContentType",
    "ProcessedContent",
    "process_content",
    "extract_title_from_content",
    "validate_content",
    "Chunk",
    "ChunkingResult",
    "hierarchical_chunk",
    "create_overlapping_chunks",
    "validate_chunks",
    "VectorizedChunk",
    "VectorizationQueue",
    "embed_chunks_batch",
    "submit_vectorization",
]
