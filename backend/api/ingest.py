"""
ingest.py
Endpoints para ingesta de documentos.
Soporta archivos, URLs y contenido de audio/video.
Dispara el source_graph para procesamiento.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field, HttpUrl

from backend.deps import get_db, get_agents, AgentContainer
from backend.graphs.source_graph import run_source_graph

logger = logging.getLogger(__name__)


# ==========================================
# Router
# ==========================================

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


# ==========================================
# Models
# ==========================================

class IngestStatus(str, Enum):
    """Estados de ingesta."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SourceType(str, Enum):
    """Tipos de fuentes."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    DOCX = "docx"
    WEB = "web"
    YOUTUBE = "youtube"
    AUDIO = "audio"
    VIDEO = "video"


class IngestRequest(BaseModel):
    """Request para ingesta de URL."""
    url: HttpUrl
    source_type: Optional[SourceType] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class IngestTextRequest(BaseModel):
    """Request para ingesta de texto directo."""
    content: str = Field(..., min_length=10)
    title: str
    source_type: SourceType = SourceType.TEXT
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    """Respuesta de ingesta."""
    ingest_id: str
    status: IngestStatus
    message: str
    source_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestStatusResponse(BaseModel):
    """Respuesta de estado de ingesta."""
    ingest_id: str
    status: IngestStatus
    progress: float = 0.0
    message: str = ""
    source_id: Optional[str] = None
    chunks_created: int = 0
    concepts_extracted: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchIngestRequest(BaseModel):
    """Request para ingesta en lote."""
    urls: List[HttpUrl]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class BatchIngestResponse(BaseModel):
    """Respuesta de ingesta en lote."""
    batch_id: str
    total: int
    ingests: List[IngestResponse]


# ==========================================
# Estado de Ingesta (en memoria para simplificar)
# ==========================================

_ingest_status: Dict[str, IngestStatusResponse] = {}


def _get_ingest_status(ingest_id: str) -> Optional[IngestStatusResponse]:
    """Obtiene estado de ingesta."""
    return _ingest_status.get(ingest_id)


def _set_ingest_status(ingest_id: str, status_resp: IngestStatusResponse) -> None:
    """Actualiza estado de ingesta."""
    _ingest_status[ingest_id] = status_resp


# ==========================================
# Background Tasks
# ==========================================

async def _process_ingest_url(
    ingest_id: str,
    url: str,
    title: Optional[str],
    metadata: Dict[str, Any],
) -> None:
    """
    Procesa ingesta de URL en background.
    """
    status_resp = _get_ingest_status(ingest_id)
    if not status_resp:
        return
    
    try:
        # Actualizar a procesando
        status_resp.status = IngestStatus.PROCESSING
        status_resp.started_at = datetime.utcnow()
        status_resp.message = "Procesando URL..."
        _set_ingest_status(ingest_id, status_resp)
        
        # Ejecutar source_graph
        result = await run_source_graph(
            url=url,
            title=title,
            metadata=metadata,
        )
        
        # Actualizar estado final
        if result.get("status") == "failed":
            status_resp.status = IngestStatus.FAILED
            status_resp.error = result.get("error", "Error desconocido")
            status_resp.message = f"Error: {status_resp.error}"
        else:
            status_resp.status = IngestStatus.COMPLETED
            status_resp.source_id = result.get("source_id")
            status_resp.chunks_created = result.get("chunk_count", 0)
            status_resp.progress = 1.0
            status_resp.message = f"Completado: {status_resp.chunks_created} chunks"
        
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)
        
    except Exception as e:
        status_resp.status = IngestStatus.FAILED
        status_resp.error = str(e)
        status_resp.message = f"Error: {str(e)}"
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)


async def _process_ingest_content(
    ingest_id: str,
    content: str,
    title: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Procesa ingesta de contenido en background.
    """
    status_resp = _get_ingest_status(ingest_id)
    if not status_resp:
        return
    
    try:
        status_resp.status = IngestStatus.PROCESSING
        status_resp.started_at = datetime.utcnow()
        status_resp.message = "Procesando contenido..."
        _set_ingest_status(ingest_id, status_resp)
        
        result = await run_source_graph(
            content=content,
            title=title,
            metadata=metadata,
        )
        
        if result.get("status") == "failed":
            status_resp.status = IngestStatus.FAILED
            status_resp.error = result.get("error", "Error desconocido")
            status_resp.message = f"Error: {status_resp.error}"
        else:
            status_resp.status = IngestStatus.COMPLETED
            status_resp.source_id = result.get("source_id")
            status_resp.chunks_created = result.get("chunk_count", 0)
            status_resp.progress = 1.0
            status_resp.message = f"Completado: {status_resp.chunks_created} chunks"
        
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)
        
    except Exception as e:
        status_resp.status = IngestStatus.FAILED
        status_resp.error = str(e)
        status_resp.message = f"Error: {str(e)}"
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)


async def _process_ingest_file(
    ingest_id: str,
    file_path: str,
    title: str,
    metadata: Dict[str, Any],
    user_openai_key: Optional[str] = None,
    user_google_key: Optional[str] = None,
) -> None:
    """
    Procesa ingesta de archivo en background.
    """
    logger.info(f"[Ingest {ingest_id}] Iniciando procesamiento de archivo: {file_path}")
    
    status_resp = _get_ingest_status(ingest_id)
    if not status_resp:
        logger.error(f"[Ingest {ingest_id}] Estado no encontrado")
        return
    
    try:
        status_resp.status = IngestStatus.PROCESSING
        status_resp.started_at = datetime.utcnow()
        status_resp.message = "Procesando archivo..."
        _set_ingest_status(ingest_id, status_resp)
        
        logger.info(f"[Ingest {ingest_id}] Ejecutando source_graph...")
        result = await run_source_graph(
            file_path=file_path,
            title=title,
            metadata=metadata,
            user_openai_key=user_openai_key,
            user_google_key=user_google_key,
        )
        logger.info(f"[Ingest {ingest_id}] Resultado: {result}")
        
        if result.get("status") == "failed":
            status_resp.status = IngestStatus.FAILED
            status_resp.error = result.get("error", "Error desconocido")
            status_resp.message = f"Error: {status_resp.error}"
            logger.error(f"[Ingest {ingest_id}] Falló: {status_resp.error}")
        else:
            status_resp.status = IngestStatus.COMPLETED
            status_resp.source_id = result.get("source_id")
            status_resp.chunks_created = result.get("chunk_count", 0)
            status_resp.progress = 1.0
            status_resp.message = f"Completado: {status_resp.chunks_created} chunks"
            logger.info(f"[Ingest {ingest_id}] Completado: {status_resp.chunks_created} chunks")
        
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)
        
    except Exception as e:
        logger.exception(f"[Ingest {ingest_id}] Excepción: {e}")
        status_resp.status = IngestStatus.FAILED
        status_resp.error = str(e)
        status_resp.message = f"Error: {str(e)}"
        status_resp.completed_at = datetime.utcnow()
        _set_ingest_status(ingest_id, status_resp)


# ==========================================
# Endpoints
# ==========================================

@router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingestar archivo",
    description="Sube y procesa un archivo (PDF, TXT, MD, DOCX)",
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: List[str] = Form(default=[]),
    metadata: str = Form(default="{}"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
    x_google_key: Optional[str] = Header(default=None, alias="X-Google-Key"),
    db: Any = Depends(get_db),
) -> IngestResponse:
    """
    Ingesta un archivo subido.
    
    Formatos soportados:
    - PDF
    - TXT
    - Markdown (.md)
    - DOCX
    """
    import json
    import tempfile
    import os
    
    # Validar extensión
    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc"}
    filename = file.filename or "document"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado. Permitidos: {allowed_extensions}",
        )
    
    # Leer contenido y guardar temporalmente
    content = await file.read()
    
    # Crear archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # Parsear metadata
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}
    
    meta["tags"] = tags
    meta["original_filename"] = filename
    
    # Crear ID de ingesta
    ingest_id = str(uuid.uuid4())
    
    # Inicializar estado
    ingest_status = IngestStatusResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message="Ingesta programada",
    )
    _set_ingest_status(ingest_id, ingest_status)
    
    # Programar procesamiento en background
    background_tasks.add_task(
        _process_ingest_file, 
        ingest_id, 
        tmp_path,
        filename,
        meta,
        x_openai_key,
        x_google_key,
    )
    
    return IngestResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message=f"Archivo '{filename}' programado para procesamiento",
    )


@router.post(
    "/url",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingestar URL",
    description="Procesa contenido desde una URL (web, YouTube, etc.)",
)
async def ingest_url(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    db: Any = Depends(get_db),
) -> IngestResponse:
    """
    Ingesta contenido desde una URL.
    
    Soporta:
    - Páginas web (HTML)
    - YouTube (transcripción)
    - PDFs remotos
    """
    url_str = str(request.url)
    
    # Crear ID de ingesta
    ingest_id = str(uuid.uuid4())
    
    # Preparar metadata
    meta = request.metadata.copy()
    meta["tags"] = request.tags
    meta["source_type"] = request.source_type.value if request.source_type else None
    
    # Inicializar estado
    ingest_status = IngestStatusResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message="Ingesta desde URL programada",
    )
    _set_ingest_status(ingest_id, ingest_status)
    
    # Programar procesamiento
    background_tasks.add_task(
        _process_ingest_url,
        ingest_id,
        url_str,
        None,  # title
        meta,
    )
    
    return IngestResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message=f"URL programada para procesamiento",
    )


@router.post(
    "/text",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingestar texto",
    description="Procesa texto directamente",
)
async def ingest_text(
    request: IngestTextRequest,
    background_tasks: BackgroundTasks,
    db: Any = Depends(get_db),
) -> IngestResponse:
    """
    Ingesta texto directo (notas, apuntes, etc.).
    """
    ingest_id = str(uuid.uuid4())
    
    meta = request.metadata.copy()
    meta["tags"] = request.tags
    meta["source_type"] = request.source_type.value
    
    ingest_status = IngestStatusResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message="Texto programado para procesamiento",
    )
    _set_ingest_status(ingest_id, ingest_status)
    
    background_tasks.add_task(
        _process_ingest_content,
        ingest_id,
        request.content,
        request.title,
        meta,
    )
    
    return IngestResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message=f"Texto '{request.title}' programado para procesamiento",
    )


@router.post(
    "/media",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingestar audio/video",
    description="Procesa archivos de audio o video (transcripción)",
)
async def ingest_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form(default="es"),
    tags: List[str] = Form(default=[]),
    db: Any = Depends(get_db),
) -> IngestResponse:
    """
    Ingesta archivo de audio o video.
    
    Realiza transcripción automática usando Whisper.
    
    Formatos soportados:
    - Audio: MP3, WAV, M4A, OGG
    - Video: MP4, MOV, AVI, MKV
    """
    import tempfile
    
    allowed_audio = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    allowed_video = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    allowed_extensions = allowed_audio | allowed_video
    
    filename = file.filename or "media"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado. Permitidos: {allowed_extensions}",
        )
    
    # Determinar tipo
    if ext in allowed_audio:
        source_type = "audio"
    else:
        source_type = "video"
    
    content = await file.read()
    
    # Guardar temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    ingest_id = str(uuid.uuid4())
    
    meta = {
        "language": language,
        "tags": tags,
        "source_type": source_type,
        "original_filename": filename,
    }
    
    ingest_status = IngestStatusResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message="Media programado para transcripción y procesamiento",
    )
    _set_ingest_status(ingest_id, ingest_status)
    
    background_tasks.add_task(
        _process_ingest_file,
        ingest_id,
        tmp_path,
        filename,
        meta,
    )
    
    return IngestResponse(
        ingest_id=ingest_id,
        status=IngestStatus.PENDING,
        message=f"Media '{filename}' programado para procesamiento",
    )


@router.post(
    "/batch",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingesta en lote",
    description="Procesa múltiples URLs en paralelo",
)
async def ingest_batch(
    request: BatchIngestRequest,
    background_tasks: BackgroundTasks,
    db: Any = Depends(get_db),
) -> BatchIngestResponse:
    """
    Ingesta múltiples URLs en lote.
    """
    batch_id = str(uuid.uuid4())
    ingests: List[IngestResponse] = []
    
    meta = request.metadata.copy()
    meta["tags"] = request.tags
    meta["batch_id"] = batch_id
    
    for url in request.urls:
        url_str = str(url)
        ingest_id = str(uuid.uuid4())
        
        ingest_status = IngestStatusResponse(
            ingest_id=ingest_id,
            status=IngestStatus.PENDING,
            message="Programado",
        )
        _set_ingest_status(ingest_id, ingest_status)
        
        background_tasks.add_task(
            _process_ingest_url,
            ingest_id,
            url_str,
            None,
            meta.copy(),
        )
        
        ingests.append(IngestResponse(
            ingest_id=ingest_id,
            status=IngestStatus.PENDING,
            message=f"URL programada",
        ))
    
    return BatchIngestResponse(
        batch_id=batch_id,
        total=len(request.urls),
        ingests=ingests,
    )


@router.get(
    "/status/{ingest_id}",
    response_model=IngestStatusResponse,
    summary="Estado de ingesta",
    description="Consulta el estado de una ingesta",
)
async def get_ingest_status_endpoint(
    ingest_id: str,
) -> IngestStatusResponse:
    """
    Obtiene el estado de una ingesta.
    """
    status_resp = _get_ingest_status(ingest_id)
    
    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingesta {ingest_id} no encontrada",
        )
    
    return status_resp


@router.get(
    "/list",
    response_model=List[IngestStatusResponse],
    summary="Listar ingestas",
    description="Lista todas las ingestas recientes",
)
async def list_ingests(
    status_filter: Optional[IngestStatus] = Query(default=None),
    limit: int = Query(default=50, le=100),
) -> List[IngestStatusResponse]:
    """
    Lista ingestas recientes.
    """
    all_ingests = list(_ingest_status.values())
    
    if status_filter:
        all_ingests = [i for i in all_ingests if i.status == status_filter]
    
    # Ordenar por fecha (más recientes primero)
    all_ingests.sort(
        key=lambda x: x.started_at or datetime.min,
        reverse=True,
    )
    
    return all_ingests[:limit]


@router.delete(
    "/{ingest_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancelar ingesta",
    description="Cancela una ingesta pendiente",
)
async def cancel_ingest(
    ingest_id: str,
) -> None:
    """
    Cancela una ingesta pendiente.
    """
    status_resp = _get_ingest_status(ingest_id)
    
    if not status_resp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingesta {ingest_id} no encontrada",
        )
    
    if status_resp.status not in [IngestStatus.PENDING, IngestStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se pueden cancelar ingestas pendientes o en proceso",
        )
    
    status_resp.status = IngestStatus.FAILED
    status_resp.message = "Cancelado por el usuario"
    status_resp.completed_at = datetime.utcnow()
    _set_ingest_status(ingest_id, status_resp)


# ==========================================
# Models para Sources
# ==========================================

class SourceInfo(BaseModel):
    """Información de una fuente/documento."""
    source_id: str
    title: str
    source_type: str
    status: str = "completed"
    chunk_count: int = 0
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SourcesListResponse(BaseModel):
    """Respuesta con lista de fuentes."""
    sources: List[SourceInfo]
    total: int


@router.get(
    "/sources",
    response_model=SourcesListResponse,
    summary="Listar documentos",
    description="Lista todos los documentos/fuentes ingestados",
)
async def list_sources(
    limit: int = Query(default=100, le=500),
    db: Any = Depends(get_db),
) -> SourcesListResponse:
    """
    Lista todas las fuentes ingestadas en el sistema.
    Combina información de la base de datos con ingestas pendientes.
    """
    sources: List[SourceInfo] = []
    
    # Obtener fuentes de la base de datos
    try:
        if db:
            result = await db.execute(
                f"SELECT * FROM source ORDER BY created_at DESC LIMIT {limit}"
            )
            
            if result:
                records = result if isinstance(result, list) else [result]
                for record in records:
                    if isinstance(record, dict) and "id" in record:
                        # Manejar total_chunks que puede ser int, lista vacía, o None
                        total_chunks = record.get("total_chunks", record.get("chunk_count", 0))
                        if isinstance(total_chunks, list):
                            total_chunks = len(total_chunks) if total_chunks else 0
                        elif not isinstance(total_chunks, int):
                            total_chunks = 0
                        
                        sources.append(SourceInfo(
                            source_id=str(record.get("id", "")),
                            title=record.get("title", record.get("name", "Sin título")),
                            source_type=record.get("source_type", record.get("type", "text")),
                            status="completed",
                            chunk_count=total_chunks,
                            created_at=record.get("created_at"),
                            metadata=record.get("metadata", {})
                        ))
    except Exception as e:
        # Si falla la BD, continuar con ingestas en memoria
        import traceback
        print(f"Error querying sources from DB: {e}")
        traceback.print_exc()
    
    # Agregar ingestas completadas de la memoria (que podrían no estar en BD aún)
    for ingest_id, ingest_status in _ingest_status.items():
        if ingest_status.source_id and ingest_status.status == IngestStatus.COMPLETED:
            # Verificar que no esté ya en la lista
            existing = any(s.source_id == ingest_status.source_id for s in sources)
            if not existing:
                sources.append(SourceInfo(
                    source_id=ingest_status.source_id,
                    title=ingest_status.message.replace("Completado: ", "").replace(" chunks", " chunks"),
                    source_type="file",
                    status="completed",
                    chunk_count=ingest_status.chunks_created,
                    created_at=ingest_status.completed_at,
                    metadata={}
                ))
    
    # Agregar ingestas en proceso
    for ingest_id, ingest_status in _ingest_status.items():
        if ingest_status.status in [IngestStatus.PENDING, IngestStatus.PROCESSING]:
            sources.append(SourceInfo(
                source_id=ingest_id,
                title=ingest_status.message,
                source_type="file",
                status=ingest_status.status.value,
                chunk_count=0,
                created_at=ingest_status.started_at,
                metadata={}
            ))
    
    # Ordenar por fecha
    sources.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
    
    return SourcesListResponse(
        sources=sources[:limit],
        total=len(sources)
    )
