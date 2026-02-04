"""
source_graph.py
Pipeline de ingestión: extracción, persistencia, vectorización, transformaciones.
Orquesta el flujo completo desde contenido raw hasta grafo de conocimiento.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db.surreal import execute, get_db
from backend.graph.builders import (
    create_chunk_node,
    create_source_node,
    link_chunk_to_source,
    link_parent_child_chunks,
    update_chunk_embedding,
)
from backend.ingestion.content_processor import (
    ContentType,
    ProcessedContent,
    process_content,
    validate_content,
)
from backend.ingestion.chunking import (
    Chunk,
    ChunkingResult,
    hierarchical_chunk,
    validate_chunks,
)
from backend.ingestion.vectorizer import (
    VectorizedChunk,
    embed_chunks_batch,
    submit_vectorization,
)


class IngestionStatus(str, Enum):
    """Estados del proceso de ingestión."""
    PENDING = "pending"
    PROCESSING = "processing"
    CHUNKING = "chunking"
    VECTORIZING = "vectorizing"
    TRANSFORMING = "transforming"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionState:
    """Estado del pipeline de ingestión."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    status: IngestionStatus = IngestionStatus.PENDING
    source_id: Optional[str] = None
    
    # Contenido
    processed_content: Optional[ProcessedContent] = None
    
    # Chunks
    chunking_result: Optional[ChunkingResult] = None
    chunk_ids: List[str] = field(default_factory=list)
    
    # Vectorización
    vectorization_job_id: Optional[str] = None
    vectorized_chunks: List[VectorizedChunk] = field(default_factory=list)
    
    # Transformaciones
    transformation_results: Dict[str, Any] = field(default_factory=dict)
    
    # Metadatos
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


# ==========================================
# Nodos del Grafo de Ingestión
# ==========================================

async def content_process(
    state: IngestionState,
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> IngestionState:
    """
    Nodo: Procesa y normaliza el contenido.
    
    Args:
        state: Estado actual del pipeline
        content: Contenido como texto
        file_path: Path al archivo
        url: URL del contenido
        title: Título opcional
        metadata: Metadatos adicionales
        
    Returns:
        Estado actualizado con contenido procesado
    """
    state.status = IngestionStatus.PROCESSING
    
    try:
        processed = await process_content(
            content=content,
            file_path=file_path,
            url=url,
            title=title,
            metadata=metadata
        )
        
        # Validar contenido
        is_valid, error = validate_content(processed)
        if not is_valid:
            state.error = error
            state.status = IngestionStatus.FAILED
            return state
        
        state.processed_content = processed
        state.metadata.update({
            "content_type": processed.content_type.value,
            "word_count": processed.word_count,
            "title": processed.title,
        })
        
    except Exception as e:
        state.error = f"Error procesando contenido: {e}"
        state.status = IngestionStatus.FAILED
    
    return state


async def save_source(state: IngestionState) -> IngestionState:
    """
    Nodo: Persiste la fuente en la base de datos.
    
    Args:
        state: Estado con contenido procesado
        
    Returns:
        Estado actualizado con source_id
    """
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
        
        # El ID puede venir como "source:uuid" o solo "uuid"
        raw_id = result.get("id", str(uuid4()))
        # Extraer solo el UUID sin el prefijo de tabla
        if isinstance(raw_id, str) and ":" in raw_id:
            state.source_id = raw_id.split(":")[-1]
        else:
            state.source_id = str(raw_id)
        
    except Exception as e:
        state.error = f"Error guardando fuente: {e}"
        state.status = IngestionStatus.FAILED
    
    return state


async def chunk_content(state: IngestionState) -> IngestionState:
    """
    Nodo: Divide el contenido en chunks jerárquicos.
    
    Args:
        state: Estado con contenido procesado
        
    Returns:
        Estado actualizado con chunks
    """
    if not state.processed_content:
        state.error = "No hay contenido procesado"
        state.status = IngestionStatus.FAILED
        return state
    
    state.status = IngestionStatus.CHUNKING
    
    try:
        # Aplicar chunking jerárquico
        chunking_result = hierarchical_chunk(
            text=state.processed_content.normalized_content,
            metadata={
                "source_id": state.source_id,
                "content_type": state.processed_content.content_type.value,
            }
        )
        
        # Validar chunks
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
    """
    Nodo: Persiste los chunks en la base de datos.
    
    Args:
        state: Estado con chunks
        
    Returns:
        Estado actualizado
    """
    if not state.chunking_result:
        state.error = "No hay chunks para guardar"
        state.status = IngestionStatus.FAILED
        return state
    
    try:
        for chunk in state.chunking_result.chunks:
            # Crear nodo de chunk con el mismo ID que el chunk local
            await create_chunk_node(
                content=chunk.content,
                source_id=state.source_id or "",
                chunk_index=chunk.index,
                parent_chunk_id=chunk.parent_id,
                token_count=chunk.token_count,
                metadata=chunk.metadata,
                chunk_id=chunk.id,  # Usar el ID del chunk local
            )
            
            # Enlazar con source
            if state.source_id:
                await link_chunk_to_source(chunk.id, state.source_id)
            
            # Enlazar padre-hijo
            if chunk.parent_id:
                await link_parent_child_chunks(chunk.parent_id, chunk.id)
        
        # Actualizar total de chunks en source
        if state.source_id:
            # Limpiar el source_id de cualquier prefijo o caracteres especiales
            clean_id = state.source_id.replace("source:", "").strip("⟨⟩`")
            # Escapar ID con guiones
            escaped_id = f"`{clean_id}`" if "-" in clean_id else clean_id
            chunk_count = len(state.chunking_result.chunks)
            
            update_query = f"UPDATE source:{escaped_id} SET total_chunks = {chunk_count}"
            await execute(update_query)
        
    except Exception as e:
        state.error = f"Error guardando chunks: {e}"
        state.status = IngestionStatus.FAILED
    
    return state


async def vectorize_chunks(state: IngestionState) -> IngestionState:
    """
    Nodo: Vectoriza los chunks.
    
    Args:
        state: Estado con chunks
        
    Returns:
        Estado actualizado con embeddings
    """
    if not state.chunking_result:
        state.error = "No hay chunks para vectorizar"
        state.status = IngestionStatus.FAILED
        return state
    
    state.status = IngestionStatus.VECTORIZING
    
    # Obtener API keys del estado si están disponibles
    user_openai_key = state.metadata.get("_user_openai_key")
    user_google_key = state.metadata.get("_user_google_key")
    
    print(f"[VECTORIZE] Starting vectorization of {len(state.chunking_result.chunks)} chunks")
    print(f"[VECTORIZE] Using client keys: openai={bool(user_openai_key)}, google={bool(user_google_key)}")
    
    try:
        # Vectorizar en batch con las API keys del cliente
        vectorized = await embed_chunks_batch(
            state.chunking_result.chunks,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
        )
        print(f"[VECTORIZE] Got {len(vectorized)} vectorized chunks")
        
        state.vectorized_chunks = vectorized
        
        # Actualizar chunks con embeddings
        for vc in vectorized:
            print(f"[VECTORIZE] Updating chunk {vc.chunk_id} with embedding dim={len(vc.embedding)}")
            await update_chunk_embedding(vc.chunk_id, vc.embedding)
        
        state.metadata["vectorization"] = {
            "total_vectorized": len(vectorized),
            "dimension": vectorized[0].dimension if vectorized else 0,
        }
        print(f"[VECTORIZE] Completed successfully")
        
    except Exception as e:
        import traceback
        print(f"[VECTORIZE] Error: {e}")
        traceback.print_exc()
        state.error = f"Error vectorizando: {e}"
        # No fallar completamente, continuar sin vectores
        state.metadata["vectorization_error"] = str(e)
    
    return state


async def trigger_transformations(
    state: IngestionState,
    run_transformations: bool = True
) -> IngestionState:
    """
    Nodo: Decide y ejecuta transformaciones.
    
    Args:
        state: Estado actual
        run_transformations: Si ejecutar transformaciones
        
    Returns:
        Estado con transformaciones aplicadas
    """
    if not run_transformations:
        return state
    
    state.status = IngestionStatus.TRANSFORMING
    
    try:
        # Importar aquí para evitar circular import
        from backend.graphs.transform_graph import run_transform_graph
        
        # Ejecutar transformaciones
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
    """
    Nodo final: Marca el proceso como completado.
    
    Args:
        state: Estado actual
        
    Returns:
        Estado finalizado
    """
    if state.status != IngestionStatus.FAILED:
        state.status = IngestionStatus.COMPLETED
        state.completed_at = datetime.utcnow()
    
    return state


# ==========================================
# Orquestador Principal
# ==========================================

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
    """
    Ejecuta el pipeline completo de ingestión.
    
    Este es el punto de entrada principal para ingestar contenido.
    Orquesta todos los pasos: procesamiento, chunking, vectorización
    y transformaciones.
    
    Args:
        content: Contenido como texto
        file_path: Path al archivo
        url: URL del contenido
        title: Título opcional
        metadata: Metadatos adicionales
        run_transformations: Si ejecutar transformaciones de grafos
        skip_vectorization: Si saltar vectorización
        user_openai_key: API key de OpenAI del cliente
        user_google_key: API key de Google del cliente
        
    Returns:
        Resultado de la ingestión con IDs y métricas
    """
    # Crear estado inicial
    state = IngestionState(metadata=metadata or {})
    # Guardar las API keys en el estado para usarlas en vectorización
    state.metadata["_user_openai_key"] = user_openai_key
    state.metadata["_user_google_key"] = user_google_key
    
    try:
        # Paso 1: Procesar contenido
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
        
        # Paso 2: Guardar fuente
        state = await save_source(state)
        
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()
        
        # Paso 3: Dividir en chunks
        state = await chunk_content(state)
        
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()
        
        # Paso 4: Guardar chunks
        state = await save_chunks(state)
        
        if state.status == IngestionStatus.FAILED:
            return state.to_dict()
        
        # Paso 5: Vectorizar (opcional)
        if not skip_vectorization:
            state = await vectorize_chunks(state)
        
        # Paso 6: Transformaciones (opcional)
        state = await trigger_transformations(state, run_transformations)
        
        # Paso 7: Finalizar
        state = await finalize(state)
        
    except Exception as e:
        state.error = f"Error en pipeline: {e}"
        state.status = IngestionStatus.FAILED
    
    return state.to_dict()


# ==========================================
# Funciones de Utilidad
# ==========================================

async def get_ingestion_status(ingestion_id: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene el estado de una ingestión.
    
    Args:
        ingestion_id: ID de la ingestión
        
    Returns:
        Estado o None si no existe
    """
    # En producción, esto consultaría la DB
    # Por ahora retornamos None (no implementado storage persistente)
    return None


async def reprocess_source(source_id: str) -> Dict[str, Any]:
    """
    Reprocesa una fuente existente.
    
    Args:
        source_id: ID de la fuente
        
    Returns:
        Resultado del reprocesamiento
    """
    # Obtener fuente
    result = await execute(
        "SELECT * FROM source WHERE id = type::thing('source', $id)",
        {"id": source_id}
    )
    
    if not result:
        return {"error": f"Fuente no encontrada: {source_id}"}
    
    source = result[0]
    
    # Re-ejecutar pipeline
    return await run_source_graph(
        url=source.get("url"),
        file_path=source.get("file_path"),
        title=source.get("title"),
        metadata=source.get("metadata", {}),
    )
