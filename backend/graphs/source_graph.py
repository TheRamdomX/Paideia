"""
source_graph.py
Pipeline de ingestión: extracción, persistencia, vectorización, transformaciones.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db.surreal import execute
from backend.graph.builders import (
    create_chunk_node,
    create_source_node,
    link_chunk_to_source,
    link_parent_child_chunks,
    update_chunk_embedding,
)
from backend.ingestion.chunking import (
    Chunk,
    ChunkingResult,
    hierarchical_chunk,
    validate_chunks,
)
from backend.ingestion.content_processor import ProcessedContent, process_content, validate_content
from backend.ingestion.extractor import extract_content
from backend.ingestion.vectorizer import VectorizedChunk, embed_chunks_batch


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    VECTORIZING = "vectorizing"
    TRANSFORMING = "transforming"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionState:
    id: str = field(default_factory=lambda: str(uuid4()))
    status: IngestionStatus = IngestionStatus.PENDING
    source_id: Optional[str] = None
    processed_content: Optional[ProcessedContent] = None
    chunking_result: Optional[ChunkingResult] = None
    chunk_ids: List[str] = field(default_factory=list)
    vectorization_job_id: Optional[str] = None
    vectorized_chunks: List[VectorizedChunk] = field(default_factory=list)
    transformation_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "source_id": self.source_id,
            "chunk_count": len(self.chunk_ids),
            "vectorization_job_id": self.vectorization_job_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metadata": self.metadata,
        }


def _compute_document_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe_chunks(chunks: List[Chunk]) -> List[Chunk]:
    seen: set[str] = set()
    deduped: List[Chunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        deduped.append(chunk)
    return deduped


async def content_process(
    state: IngestionState,
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> IngestionState:
    state.status = IngestionStatus.PROCESSING

    try:
        source_result = await extract_content(
            content=content,
            file_path=file_path,
            url=url,
            metadata=metadata,
        )

        processed = await process_content(
            source_result=source_result,
            title=title,
            metadata=metadata,
        )

        processed.metadata["document_fingerprint"] = _compute_document_fingerprint(
            processed.normalized_content
        )

        if "page" not in processed.metadata and "page_number" in processed.metadata:
            processed.metadata["page"] = processed.metadata["page_number"]

        is_valid, error = validate_content(processed)
        if not is_valid:
            state.error = error
            state.status = IngestionStatus.FAILED
            return state

        state.processed_content = processed
        state.metadata.update(
            {
                "content_type": processed.content_type.value,
                "word_count": processed.word_count,
                "title": processed.title,
                "document_fingerprint": processed.metadata.get("document_fingerprint"),
            }
        )

    except Exception as e:
        state.error = f"Error procesando contenido: {e}"
        state.status = IngestionStatus.FAILED

    return state


async def save_source(state: IngestionState) -> IngestionState:
    if not state.processed_content:
        state.error = "No hay contenido procesado"
        state.status = IngestionStatus.FAILED
        return state

    try:
        content = state.processed_content

        result = await create_source_node(
            title=content.title,
            content_type=content.content_type.value,
            url=content.url,
            file_path=content.file_path,
            author=content.author,
            metadata=content.metadata,
        )

        raw_id = result.get("id", str(uuid4()))
        state.source_id = raw_id.split(":")[-1] if isinstance(raw_id, str) and ":" in raw_id else str(raw_id)

    except Exception as e:
        state.error = f"Error guardando fuente: {e}"
        state.status = IngestionStatus.FAILED

    return state


async def chunk_content(state: IngestionState) -> IngestionState:
    if not state.processed_content:
        state.error = "No hay contenido procesado"
        state.status = IngestionStatus.FAILED
        return state

    state.status = IngestionStatus.CHUNKING

    try:
        processed = state.processed_content
        chunk_metadata = {
            **processed.metadata,
            "source_id": state.source_id,
            "content_type": processed.content_type.value,
            "language": processed.language,
        }

        chunking_result = hierarchical_chunk(
            text=processed.normalized_content,
            metadata=chunk_metadata,
        )

        is_valid, errors = validate_chunks(chunking_result.chunks)
        if not is_valid:
            state.metadata["chunking_warnings"] = errors

        state.chunking_result = chunking_result
        state.chunk_ids = [c.id for c in chunking_result.chunks]
        state.metadata["chunk_stats"] = {
            "total": len(chunking_result.chunks),
            "parents": len(chunking_result.parent_chunks),
            "children": len(chunking_result.child_chunks),
            "total_tokens": chunking_result.total_tokens,
        }

    except Exception as e:
        state.error = f"Error en chunking: {e}"
        state.status = IngestionStatus.FAILED

    return state


async def save_chunks(state: IngestionState) -> IngestionState:
    if not state.chunking_result:
        state.error = "No hay chunks para guardar"
        state.status = IngestionStatus.FAILED
        return state

    try:
        for chunk in state.chunking_result.chunks:
            await create_chunk_node(
                content=chunk.content,
                source_id=state.source_id or "",
                chunk_index=chunk.index,
                parent_chunk_id=chunk.parent_id,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                chunk_id=chunk.id,
            )

            if state.source_id:
                await link_chunk_to_source(chunk.id, state.source_id)

            if chunk.parent_id:
                await link_parent_child_chunks(chunk.parent_id, chunk.id)

        if state.source_id:
            clean_id = state.source_id.replace("source:", "").strip("⟨⟩`")
            escaped_id = f"`{clean_id}`" if "-" in clean_id else clean_id
            chunk_count = len(state.chunking_result.chunks)
            await execute(f"UPDATE source:{escaped_id} SET total_chunks = {chunk_count}")

    except Exception as e:
        state.error = f"Error guardando chunks: {e}"
        state.status = IngestionStatus.FAILED

    return state


async def vectorize_chunks(state: IngestionState) -> IngestionState:
    if not state.chunking_result:
        state.error = "No hay chunks para vectorizar"
        state.status = IngestionStatus.FAILED
        return state

    state.status = IngestionStatus.VECTORIZING
    user_openai_key = state.metadata.get("_user_openai_key")
    user_google_key = state.metadata.get("_user_google_key")

    try:
        child_chunks = [c for c in state.chunking_result.chunks if c.level == 1]
        deduped_children = _dedupe_chunks(child_chunks)

        vectorized = await embed_chunks_batch(
            deduped_children,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
        )

        state.vectorized_chunks = vectorized

        for vc in vectorized:
            await update_chunk_embedding(vc.chunk_id, vc.embedding)

        state.metadata["vectorization"] = {
            "child_candidates": len(child_chunks),
            "deduped_children": len(deduped_children),
            "total_vectorized": len(vectorized),
            "dimension": vectorized[0].dimension if vectorized else 0,
        }

    except Exception as e:
        state.error = f"Error vectorizando: {e}"
        state.metadata["vectorization_error"] = str(e)

    return state


async def trigger_transformations(state: IngestionState, run_transformations: bool = True) -> IngestionState:
    if not run_transformations:
        return state

    state.status = IngestionStatus.TRANSFORMING

    try:
        from backend.graphs.transform_graph import run_transform_graph

        transform_result = await run_transform_graph(
            content=state.processed_content.normalized_content if state.processed_content else "",
            source_id=state.source_id,
            chunks=state.chunking_result.chunks if state.chunking_result else [],
        )
        state.transformation_results = transform_result

    except Exception as e:
        state.error = f"Error en transformaciones: {e}"
        state.metadata["transformation_error"] = str(e)

    return state


async def finalize(state: IngestionState) -> IngestionState:
    if state.status != IngestionStatus.FAILED:
        state.status = IngestionStatus.COMPLETED
        state.completed_at = datetime.utcnow()
    return state


async def run_source_graph(
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    run_transformations: bool = True,
    skip_vectorization: bool = False,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> Dict[str, Any]:
    state = IngestionState(metadata=metadata or {})
    state.metadata["_user_openai_key"] = user_openai_key
    state.metadata["_user_google_key"] = user_google_key

    try:
        state = await content_process(
            state=state,
            content=content,
            file_path=file_path,
            url=url,
            title=title,
            metadata=metadata,
        )
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()

        state = await save_source(state)
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()

        state = await chunk_content(state)
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()

        state = await save_chunks(state)
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()

        if not skip_vectorization:
            state = await vectorize_chunks(state)

        state = await trigger_transformations(state, run_transformations)
        state = await finalize(state)

    except Exception as e:
        state.error = f"Error en pipeline: {e}"
        state.status = IngestionStatus.FAILED

    return state.to_dict()


async def get_ingestion_status(ingestion_id: str) -> Optional[Dict[str, Any]]:
    return None
