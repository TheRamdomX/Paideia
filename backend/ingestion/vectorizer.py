"""
vectorizer.py
Envía jobs de embedding asíncronos con persistencia básica.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.ingestion.chunking import Chunk
from backend.models.embeddings import batch_embed


class VectorizationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class VectorizationJob:
    id: str = field(default_factory=lambda: str(uuid4()))
    chunk_ids: List[str] = field(default_factory=list)
    status: VectorizationStatus = VectorizationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "chunk_ids": self.chunk_ids,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class VectorizedChunk:
    chunk_id: str = ""
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    dimension: int = 0
    vectorized_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "dimension": self.dimension,
            "vectorized_at": self.vectorized_at.isoformat(),
        }


class VectorizationQueue:
    """Cola persistente simple en archivo JSON."""

    _instance: Optional["VectorizationQueue"] = None

    def __new__(cls) -> "VectorizationQueue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jobs = {}
            cls._instance._results = {}
            cls._instance._path = Path("backend/data/vectorization_queue.json")
            cls._instance._path.parent.mkdir(parents=True, exist_ok=True)
            cls._instance._load()
        return cls._instance

    def _serialize(self) -> Dict[str, Any]:
        return {
            "jobs": {job_id: job.to_dict() for job_id, job in self._jobs.items()},
            "results": {
                chunk_id: {
                    **result.to_dict(),
                    "content": result.content,
                }
                for chunk_id, result in self._results.items()
            },
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for job_id, payload in data.get("jobs", {}).items():
                self._jobs[job_id] = VectorizationJob(
                    id=payload["id"],
                    chunk_ids=payload.get("chunk_ids", []),
                    status=VectorizationStatus(payload.get("status", "pending")),
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    completed_at=(
                        datetime.fromisoformat(payload["completed_at"])
                        if payload.get("completed_at")
                        else None
                    ),
                    error=payload.get("error"),
                    retry_count=payload.get("retry_count", 0),
                    max_retries=payload.get("max_retries", 3),
                    metadata=payload.get("metadata", {}),
                )
            for chunk_id, payload in data.get("results", {}).items():
                self._results[chunk_id] = VectorizedChunk(
                    chunk_id=payload["chunk_id"],
                    content=payload.get("content", ""),
                    embedding=payload.get("embedding", []),
                    dimension=payload.get("dimension", 0),
                    vectorized_at=datetime.fromisoformat(payload["vectorized_at"]),
                )
        except Exception:
            self._jobs = {}
            self._results = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._serialize(), ensure_ascii=False), encoding="utf-8")

    def add_job(self, job: VectorizationJob) -> str:
        self._jobs[job.id] = job
        self._save()
        return job.id

    def get_job(self, job_id: str) -> Optional[VectorizationJob]:
        return self._jobs.get(job_id)

    def get_result(self, chunk_id: str) -> Optional[VectorizedChunk]:
        return self._results.get(chunk_id)

    def store_result(self, result: VectorizedChunk) -> None:
        self._results[result.chunk_id] = result
        self._save()

    def clear(self) -> None:
        self._jobs.clear()
        self._results.clear()
        self._save()


_queue = VectorizationQueue()
_embedding_semaphore = asyncio.Semaphore(5)


def _dedupe_by_content_hash(chunks: List[Chunk]) -> List[Chunk]:
    seen: set[str] = set()
    deduped: List[Chunk] = []
    for chunk in chunks:
        if chunk.content_hash in seen:
            continue
        seen.add(chunk.content_hash)
        deduped.append(chunk)
    return deduped


async def embed_chunks_batch(
    chunks: List[Chunk],
    batch_size: int = 50,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> List[VectorizedChunk]:
    if not chunks:
        return []

    # Vectorizar SOLO chunks hijos
    child_chunks = [c for c in chunks if c.level == 1]
    deduped_chunks = _dedupe_by_content_hash(child_chunks)

    results: List[VectorizedChunk] = []

    async with _embedding_semaphore:
        for i in range(0, len(deduped_chunks), batch_size):
            batch = deduped_chunks[i : i + batch_size]
            contents = [c.content for c in batch]
            embeddings = await batch_embed(
                contents,
                user_openai_key=user_openai_key,
                user_google_key=user_google_key,
            )
            for chunk, embedding in zip(batch, embeddings):
                results.append(
                    VectorizedChunk(
                        chunk_id=chunk.id,
                        content=chunk.content,
                        embedding=embedding,
                        dimension=len(embedding),
                    )
                )

    return results


async def submit_vectorization(chunks: List[Chunk], metadata: Optional[Dict[str, Any]] = None) -> str:
    job = VectorizationJob(chunk_ids=[c.id for c in chunks], metadata=metadata or {})
    _queue.add_job(job)
    asyncio.create_task(process_vectorization_job(job, chunks))
    return job.id


async def process_vectorization_job(job: VectorizationJob, chunks: List[Chunk]) -> None:
    job.status = VectorizationStatus.PROCESSING
    _queue.add_job(job)

    try:
        results = await embed_chunks_batch(chunks)
        for result in results:
            _queue.store_result(result)
        job.status = VectorizationStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        _queue.add_job(job)
    except Exception as e:
        job.error = str(e)
        job.retry_count += 1
        job.status = VectorizationStatus.RETRY if job.retry_count < job.max_retries else VectorizationStatus.FAILED
        _queue.add_job(job)


def get_vectorization_status(job_id: str) -> Optional[Dict[str, Any]]:
    job = _queue.get_job(job_id)
    if not job:
        return None

    completed_count = sum(1 for chunk_id in job.chunk_ids if _queue.get_result(chunk_id) is not None)
    return {
        **job.to_dict(),
        "progress": {
            "total": len(job.chunk_ids),
            "completed": completed_count,
            "percentage": (completed_count / len(job.chunk_ids) * 100) if job.chunk_ids else 0,
        },
    }


def get_chunk_embedding(chunk_id: str) -> Optional[List[float]]:
    result = _queue.get_result(chunk_id)
    return result.embedding if result else None


async def vectorize_and_store(chunks: List[Chunk], store_callback: Optional[callable] = None) -> List[VectorizedChunk]:
    results = await embed_chunks_batch(chunks)
    if store_callback:
        for result in results:
            await store_callback(result)
    return results
