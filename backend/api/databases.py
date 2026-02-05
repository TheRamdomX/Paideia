"""
databases.py
Endpoints para gestión de múltiples bases de datos (espacios de conocimiento).
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from backend.db.surreal import get_db, SurrealDBClient


# ==========================================
# Router
# ==========================================

router = APIRouter(prefix="/databases", tags=["databases"])


# ==========================================
# Modelos
# ==========================================

class DatabaseInfo(BaseModel):
    """Información de una base de datos."""
    name: str
    is_current: bool = False


class DatabaseListResponse(BaseModel):
    """Lista de bases de datos disponibles."""
    databases: List[DatabaseInfo]
    current: str
    namespace: str


class CreateDatabaseRequest(BaseModel):
    """Request para crear una base de datos."""
    name: str = Field(..., min_length=1, max_length=50, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    description: Optional[str] = None


class CreateDatabaseResponse(BaseModel):
    """Response de creación de base de datos."""
    name: str
    created: bool
    message: str


class SwitchDatabaseRequest(BaseModel):
    """Request para cambiar de base de datos."""
    database: str


class SwitchDatabaseResponse(BaseModel):
    """Response de cambio de base de datos."""
    previous: str
    current: str
    message: str


# ==========================================
# Endpoints
# ==========================================

@router.get(
    "",
    response_model=DatabaseListResponse,
    summary="Listar bases de datos",
    description="Lista todas las bases de datos disponibles en el namespace",
)
async def list_databases(
    db: SurrealDBClient = Depends(get_db),
) -> DatabaseListResponse:
    """Lista las bases de datos disponibles."""
    databases = await db.list_databases()
    current = db.current_database
    
    db_list = [
        DatabaseInfo(name=name, is_current=(name == current))
        for name in databases
    ]
    
    return DatabaseListResponse(
        databases=db_list,
        current=current,
        namespace=db.current_namespace,
    )


@router.post(
    "",
    response_model=CreateDatabaseResponse,
    summary="Crear base de datos",
    description="Crea una nueva base de datos (espacio de conocimiento)",
)
async def create_database(
    request: CreateDatabaseRequest,
    db: SurrealDBClient = Depends(get_db),
) -> CreateDatabaseResponse:
    """Crea una nueva base de datos."""
    try:
        # Verificar si ya existe
        existing = await db.list_databases()
        if request.name in existing:
            return CreateDatabaseResponse(
                name=request.name,
                created=False,
                message=f"La base de datos '{request.name}' ya existe",
            )
        
        # Crear
        await db.create_database(request.name)
        
        # Cambiar a la nueva DB para inicializarla
        await db.use_database(request.name)
        
        # Crear metadata de la base de datos
        await db.execute("""
            CREATE database_info:meta SET
                name = $name,
                description = $description,
                created_at = time::now()
        """, {
            "name": request.name,
            "description": request.description or "",
        })
        
        return CreateDatabaseResponse(
            name=request.name,
            created=True,
            message=f"Base de datos '{request.name}' creada exitosamente",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/switch",
    response_model=SwitchDatabaseResponse,
    summary="Cambiar base de datos",
    description="Cambia a otra base de datos activa",
)
async def switch_database(
    request: SwitchDatabaseRequest,
    db: SurrealDBClient = Depends(get_db),
) -> SwitchDatabaseResponse:
    """Cambia a otra base de datos."""
    previous = db.current_database
    
    # Verificar que existe
    existing = await db.list_databases()
    if request.database not in existing:
        raise HTTPException(
            status_code=404,
            detail=f"Base de datos '{request.database}' no encontrada"
        )
    
    try:
        await db.use_database(request.database)
        
        return SwitchDatabaseResponse(
            previous=previous,
            current=request.database,
            message=f"Cambiado de '{previous}' a '{request.database}'",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/current",
    summary="Base de datos actual",
    description="Devuelve la base de datos actualmente en uso",
)
async def get_current_database(
    db: SurrealDBClient = Depends(get_db),
) -> dict:
    """Devuelve información de la base de datos actual."""
    return {
        "database": db.current_database,
        "namespace": db.current_namespace,
    }


@router.delete(
    "/{database_name}",
    summary="Eliminar base de datos",
    description="Elimina una base de datos (¡irreversible!)",
)
async def delete_database(
    database_name: str,
    db: SurrealDBClient = Depends(get_db),
) -> dict:
    """Elimina una base de datos."""
    # No permitir eliminar la base de datos actual
    if database_name == db.current_database:
        raise HTTPException(
            status_code=400,
            detail="No se puede eliminar la base de datos actualmente en uso"
        )
    
    # No permitir eliminar la base de datos por defecto
    if database_name == "education":
        raise HTTPException(
            status_code=400,
            detail="No se puede eliminar la base de datos por defecto"
        )
    
    try:
        await db.execute(f"REMOVE DATABASE {database_name};")
        
        return {
            "deleted": database_name,
            "message": f"Base de datos '{database_name}' eliminada",
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
