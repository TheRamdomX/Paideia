"""
vectorizer.py
Envía jobs de embedding asíncronos.
Gestiona la vectorización de chunks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.models.embeddings import batch_embed, embed_text, get_embedding_dimension
from backend.ingestion.chunking import Chunk


class VectorizationStatus(str, Enum):
    """Estados de vectorización."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class VectorizationJob:
    """Representa un job de vectorización."""
    
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
    """Chunk con su embedding."""
    
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


# ==========================================
# Cola de Vectorización (In-Memory)
# ==========================================

class VectorizationQueue:
    """Cola para gestionar jobs de vectorización."""
    
    _instance: Optional["VectorizationQueue"] = None
    _jobs: Dict[str, VectorizationJob]
    _results: Dict[str, VectorizedChunk]
    _processing: bool
    
    def __new__(cls) -> "VectorizationQueue":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._jobs = {}
            cls._instance._results = {}
            cls._instance._processing = False
        return cls._instance
    
    def add_job(self, job: VectorizationJob) -> str:
        """Agrega un job a la cola."""
        self._jobs[job.id] = job
        return job.id
    
    def get_job(self, job_id: str) -> Optional[VectorizationJob]:
        """Obtiene un job por ID."""
        return self._jobs.get(job_id)
    
    def get_result(self, chunk_id: str) -> Optional[VectorizedChunk]:
        """Obtiene el resultado de un chunk."""
        return self._results.get(chunk_id)
    
    def store_result(self, result: VectorizedChunk) -> None:
        """Almacena un resultado."""
        self._results[result.chunk_id] = result
    
    def get_pending_jobs(self) -> List[VectorizationJob]:
        """Obtiene jobs pendientes."""
        return [
            job for job in self._jobs.values()
            if job.status in [VectorizationStatus.PENDING, VectorizationStatus.RETRY]
        ]
    
    def clear(self) -> None:
        """Limpia la cola."""
        self._jobs.clear()
        self._results.clear()


_queue = VectorizationQueue()


# ==========================================
# Funciones de Vectorización
# ==========================================

async def embed_chunk(chunk: Chunk) -> VectorizedChunk:
    """
    Genera embedding para un chunk individual.
    
    Args:
        chunk: Chunk a vectorizar
        
    Returns:
        VectorizedChunk con el embedding
    """
    if not chunk.content:
        return VectorizedChunk(
            chunk_id=chunk.id,
            content="",
            embedding=[0.0] * get_embedding_dimension(),
            dimension=get_embedding_dimension(),
        )
    
    embedding = await embed_text(chunk.content)
    
    return VectorizedChunk(
        chunk_id=chunk.id,
        content=chunk.content,
        embedding=embedding,
        dimension=len(embedding),
    )


async def embed_chunks_batch(
    chunks: List[Chunk],
    batch_size: int = 50,
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> List[VectorizedChunk]:
    """
    Genera embeddings para múltiples chunks en batch.
    
    Args:
        chunks: Lista de chunks
        batch_size: Tamaño del batch
        user_openai_key: API key de OpenAI del cliente
        user_google_key: API key de Google del cliente
        
    Returns:
        Lista de VectorizedChunks
    """
    if not chunks:
        return []
    
    results = []
    
    # Procesar en batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Extraer contenidos
        contents = [c.content for c in batch]
        
        # Obtener embeddings en batch con las API keys del cliente
        embeddings = await batch_embed(
            contents,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
        )
        
        # Crear resultados
        for chunk, embedding in zip(batch, embeddings):
            results.append(VectorizedChunk(
                chunk_id=chunk.id,
                content=chunk.content,
                embedding=embedding,
                dimension=len(embedding),
            ))
    
    return results


async def submit_vectorization(
    chunks: List[Chunk],
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Envía chunks para vectorización asíncrona.
    
    Args:
        chunks: Chunks a vectorizar
        metadata: Metadatos del job
        
    Returns:
        ID del job creado
    """
    job = VectorizationJob(
        chunk_ids=[c.id for c in chunks],
        metadata=metadata or {},
    )
    
    _queue.add_job(job)
    
    # Procesar inmediatamente (en producción sería un worker separado)
    asyncio.create_task(process_vectorization_job(job, chunks))
    
    return job.id


async def process_vectorization_job(
    job: VectorizationJob,
    chunks: List[Chunk]
) -> None:
    """
    Procesa un job de vectorización.
    
    Args:
        job: Job a procesar
        chunks: Chunks asociados al job
    """
    job.status = VectorizationStatus.PROCESSING
    
    try:
        # Vectorizar en batch
        results = await embed_chunks_batch(chunks)
        
        # Almacenar resultados
        for result in results:
            _queue.store_result(result)
        
        job.status = VectorizationStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        
    except Exception as e:
        job.error = str(e)
        job.retry_count += 1
        
        if job.retry_count < job.max_retries:
            job.status = VectorizationStatus.RETRY
        else:
            job.status = VectorizationStatus.FAILED


async def retry_failed_chunks(job_id: str) -> bool:
    """
    Reintenta vectorizar chunks fallidos.
    
    Args:
        job_id: ID del job a reintentar
        
    Returns:
        True si se reintentó exitosamente
    """
    job = _queue.get_job(job_id)
    
    if not job:
        return False
    
    if job.status != VectorizationStatus.FAILED:
        return False
    
    if job.retry_count >= job.max_retries:
        return False
    
    # Resetear estado
    job.status = VectorizationStatus.RETRY
    job.error = None
    
    return True


def get_vectorization_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene el estado de un job de vectorización.
    
    Args:
        job_id: ID del job
        
    Returns:
        Estado del job o None si no existe
    """
    job = _queue.get_job(job_id)
    
    if not job:
        return None
    
    # Contar resultados
    completed_count = sum(
        1 for chunk_id in job.chunk_ids
        if _queue.get_result(chunk_id) is not None
    )
    
    return {
        **job.to_dict(),
        "progress": {
            "total": len(job.chunk_ids),
            "completed": completed_count,
            "percentage": (completed_count / len(job.chunk_ids) * 100) if job.chunk_ids else 0,
        }
    }


def get_chunk_embedding(chunk_id: str) -> Optional[List[float]]:
    """
    Obtiene el embedding de un chunk.
    
    Args:
        chunk_id: ID del chunk
        
    Returns:
        Vector de embedding o None
    """
    result = _queue.get_result(chunk_id)
    return result.embedding if result else None


async def vectorize_and_store(
    chunks: List[Chunk],
    store_callback: Optional[callable] = None
) -> List[VectorizedChunk]:
    """
    Vectoriza chunks y opcionalmente los almacena.
    
    Args:
        chunks: Chunks a vectorizar
        store_callback: Función para almacenar resultados
        
    Returns:
        Lista de chunks vectorizados
    """
    results = await embed_chunks_batch(chunks)
    
    if store_callback:
        for result in results:
            await store_callback(result)
    
    return results
