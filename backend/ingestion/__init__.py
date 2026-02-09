"""
Ingestion package.
Procesamiento de contenido, chunking, vectorización y OCR.
"""

from backend.ingestion.content_processor import (
    ContentType,
    ProcessedContent,
    detect_content_type,
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
    embed_chunk,
    embed_chunks_batch,
    submit_vectorization,
)
from backend.ingestion.ocr import (
    OCRResult,
    is_pdf,
    is_image,
    is_text_file,
    pdf_has_text,
    extract_native_pdf_text,
    run_ocr,
    read_text_file,
    check_tesseract_installed,
)

__all__ = [
    # Content Processor
    "ContentType",
    "ProcessedContent",
    "detect_content_type",
    "process_content",
    "extract_title_from_content",
    "validate_content",
    # Chunking
    "Chunk",
    "ChunkingResult",
    "hierarchical_chunk",
    "create_overlapping_chunks",
    "validate_chunks",
    # Vectorizer
    "VectorizedChunk",
    "VectorizationQueue",
    "embed_chunk",
    "embed_chunks_batch",
    "submit_vectorization",
    # OCR
    "OCRResult",
    "is_pdf",
    "is_image",
    "is_text_file",
    "pdf_has_text",
    "extract_native_pdf_text",
    "run_ocr",
    "read_text_file",
    "check_tesseract_installed",
]

