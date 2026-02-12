"""
deps.py
Inyección de dependencias para FastAPI.
Proporciona DB, modelos y agentes como dependencias.

Multi-tenant: Cada request usa su propia conexión a la DB especificada
sin afectar a otros requests concurrentes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional, Type

from fastapi import Header, Request
from backend.db.surreal import get_db as get_surreal_db, connect, close, execute, switch_database
from backend.db.pool import DatabasePool, DatabaseConnection
from backend.models.embeddings import (
    BaseEmbedding,
    get_embedding_model,
)
from backend.models.llm import BaseLLM, get_llm, LLMProvider
from backend.settings import (
    Settings, 
    get_settings as _get_settings,
    get_model_config,
    get_embedding_config,
    get_db_config,
)


# ==========================================
# Configuración Global
# ==========================================

_db_connection: Optional[Any] = None  # Legacy, para compatibilidad
_llm_instance: Optional[BaseLLM] = None
_embedding_model: Optional[BaseEmbedding] = None
_db_pool: Optional[DatabasePool] = None


@dataclass
class RequestContext:
    """Contexto de request con información de DB."""
    database: str
    connection: DatabaseConnection
    
    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Ejecuta query en la DB del contexto."""
        return await self.connection.execute(query, params)
    
    async def create(self, table: str, data: Dict) -> Any:
        """Crea registro en la DB del contexto."""
        return await self.connection.create(table, data)
    
    async def select(self, table: str, record_id: Optional[str] = None) -> Any:
        """Selecciona de la DB del contexto."""
        return await self.connection.select(table, record_id)
    
    async def update(self, table: str, record_id: str, data: Dict) -> Any:
        """Actualiza registro en la DB del contexto."""
        return await self.connection.update(table, record_id, data)
    
    async def delete(self, table: str, record_id: str) -> Any:
        """Elimina registro de la DB del contexto."""
        return await self.connection.delete(table, record_id)


def get_settings() -> Settings:
    """
    Obtiene la configuración de la aplicación (singleton).
    
    Returns:
        Instancia de Settings
    """
    return _get_settings()


# ==========================================
# Dependencias de Base de Datos
# ==========================================

def get_db_pool() -> DatabasePool:
    """
    Obtiene el pool de conexiones (singleton).
    
    Returns:
        Instancia del pool de conexiones
    """
    global _db_pool
    if _db_pool is None:
        _db_pool = DatabasePool()
    return _db_pool


async def get_db(
    x_database: Optional[str] = Header(None, alias="X-Database")
) -> AsyncGenerator[Any, None]:
    """
    Dependencia para obtener conexión a la base de datos.
    
    LEGACY: Esta función cambia la DB global. Para multi-tenant, usa get_db_context.
    
    Args:
        x_database: Nombre de la base de datos a usar (opcional)
    
    Yields:
        Conexión raw a SurrealDB (para compatibilidad)
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = await get_surreal_db()
    
    # Cambiar base de datos si se especifica en el header
    if x_database:
        await switch_database(x_database)
    
    try:
        yield _db_connection
    finally:
        # La conexión se mantiene abierta (pool)
        pass


async def get_db_context(
    x_database: Optional[str] = Header(None, alias="X-Database")
) -> AsyncGenerator[RequestContext, None]:
    """
    Dependencia para obtener conexión multi-tenant a la base de datos.
    
    MULTI-TENANT: Cada request obtiene su propia conexión a la DB
    especificada en el header X-Database, sin afectar otras requests.
    
    Args:
        x_database: Nombre de la base de datos a usar (opcional, usa default si no se especifica)
    
    Yields:
        RequestContext con conexión a la DB especificada y el nombre de la DB
        
    Example:
        ```python
        @app.get("/items")
        async def get_items(ctx: RequestContext = Depends(get_db_context)):
            result = await ctx.execute("SELECT * FROM items")
            # ctx.database contiene el nombre de la DB para cache
            return result
        ```
    """
    settings = _get_settings()
    database = x_database or settings.surreal_database
    
    pool = get_db_pool()
    connection = await pool.get_connection(database)
    
    ctx = RequestContext(database=database, connection=connection)
    
    try:
        yield ctx
    finally:
        # La conexión se mantiene en el pool para reutilización
        pass


def get_database_name(
    x_database: Optional[str] = Header(None, alias="X-Database")
) -> str:
    """
    Dependencia simple para obtener el nombre de la base de datos actual.
    
    Útil cuando solo necesitas el nombre de la DB para cache o logging,
    sin necesitar la conexión completa.
    
    Args:
        x_database: Nombre de la base de datos del header
    
    Returns:
        Nombre de la base de datos a usar
    """
    settings = _get_settings()
    return x_database or settings.surreal_database


async def init_db() -> None:
    """Inicializa el pool de conexiones a la base de datos."""
    global _db_pool
    _db_pool = DatabasePool()
    # Opcionalmente, pre-conectar a la DB por defecto
    settings = _get_settings()
    await _db_pool.get_connection(settings.surreal_database)


async def close_db() -> None:
    """Cierra todas las conexiones del pool."""
    global _db_connection, _db_pool
    
    # Cerrar pool nuevo
    if _db_pool is not None:
        await _db_pool.close_all()
        _db_pool = None
    
    # Cerrar conexión legacy si existe
    if _db_connection is not None:
        await close()
        _db_connection = None


# ==========================================
# Dependencias de Modelos
# ==========================================

async def get_llm_model() -> BaseLLM:
    """
    Dependencia para obtener el modelo LLM.
    
    Returns:
        Instancia del modelo LLM
        
    Example:
        ```python
        @app.post("/generate")
        async def generate(
            prompt: str,
            llm: BaseLLM = Depends(get_llm_model)
        ):
            response = await llm.generate(prompt)
            return {"response": response}
        ```
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = get_llm()
    
    return _llm_instance


async def get_embedding() -> BaseEmbedding:
    """
    Dependencia para obtener el modelo de embeddings.
    
    Returns:
        Instancia del modelo de embeddings
        
    Example:
        ```python
        @app.post("/embed")
        async def embed(
            text: str,
            model: BaseEmbedding = Depends(get_embedding)
        ):
            vector = await model.embed(text)
            return {"vector": vector}
        ```
    """
    global _embedding_model
    
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    
    return _embedding_model


# ==========================================
# Dependencias de Agentes
# ==========================================

class AgentContainer:
    """Contenedor de agentes para inyección de dependencias."""
    
    def __init__(self):
        self._retrieval_agent = None
        self._reasoning_agent = None
        self._reflection_agent = None
    
    @property
    def retrieval(self):
        """Agente de retrieval."""
        if self._retrieval_agent is None:
            from backend.agents import retrieval_agent
            self._retrieval_agent = retrieval_agent
        return self._retrieval_agent
    
    @property
    def reasoning(self):
        """Agente de razonamiento."""
        if self._reasoning_agent is None:
            from backend.agents import reasoning_agent
            self._reasoning_agent = reasoning_agent
        return self._reasoning_agent
    
    @property
    def reflection(self):
        """Agente de reflexión."""
        if self._reflection_agent is None:
            from backend.agents import reflection_agent
            self._reflection_agent = reflection_agent
        return self._reflection_agent


_agent_container: Optional[AgentContainer] = None


def get_agents() -> AgentContainer:
    """
    Dependencia para obtener el contenedor de agentes.
    
    Returns:
        AgentContainer con acceso a todos los agentes
        
    Example:
        ```python
        @app.post("/query")
        async def query(
            question: str,
            agents: AgentContainer = Depends(get_agents)
        ):
            strategy = await agents.retrieval.decide_strategy(question)
            ...
        ```
    """
    global _agent_container
    
    if _agent_container is None:
        _agent_container = AgentContainer()
    
    return _agent_container


# ==========================================
# Dependencias de Sesión
# ==========================================

async def get_session_id(
    session_id: Optional[str] = None,
) -> Optional[str]:
    """
    Dependencia para obtener/validar session_id.
    
    Args:
        session_id: ID de sesión del header o query param
        
    Returns:
        ID de sesión validado o None
    """
    if session_id:
        # Validar formato (UUID-like)
        if len(session_id) >= 32:
            return session_id
    return None


async def get_student_id(
    student_id: Optional[str] = None,
    authorization: Optional[str] = None,
) -> Optional[str]:
    """
    Dependencia para obtener student_id.
    
    Args:
        student_id: ID explícito
        authorization: Token de autorización (para extraer student_id)
        
    Returns:
        ID del estudiante o None
    """
    if student_id:
        return student_id
    
    # En producción, extraer del token JWT
    if authorization and authorization.startswith("Bearer "):
        # TODO: Decodificar JWT y extraer student_id
        pass
    
    return None


# ==========================================
# Lifecycle Management
# ==========================================

@asynccontextmanager
async def lifespan_context():
    """
    Context manager para el ciclo de vida de la aplicación.
    
    Inicializa y limpia recursos.
    
    Example:
        ```python
        app = FastAPI(lifespan=lifespan)
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with lifespan_context():
                yield
        ```
    """
    # Startup
    await startup()
    
    try:
        yield
    finally:
        # Shutdown
        await shutdown()


async def startup() -> None:
    """
    Inicialización de la aplicación.
    
    - Conecta a la base de datos
    - Inicializa modelos
    - Carga configuración
    """
    settings = get_settings()
    
    # Inicializar base de datos
    await init_db()
    
    # Pre-cargar modelos (opcional, para warm-up)
    # Solo si explícitamente habilitado
    # await get_llm_model()
    # await get_embedding()


async def shutdown() -> None:
    """
    Limpieza al cerrar la aplicación.
    
    - Cierra conexiones de base de datos
    - Libera recursos de modelos
    """
    global _llm_instance, _embedding_model, _agent_container
    
    # Cerrar conexión a DB
    await close_db()
    
    # Limpiar referencias
    _llm_instance = None
    _embedding_model = None
    _agent_container = None


# ==========================================
# Health Check Dependencies
# ==========================================

async def check_db_health() -> Dict[str, Any]:
    """
    Verifica salud de la conexión a DB.
    
    Returns:
        Estado de la conexión
    """
    try:
        pool = get_db_pool()
        settings = _get_settings()
        connection = await pool.get_connection(settings.surreal_database)
        # Ejecutar query simple
        result = await connection.execute("INFO FOR DB")
        return {
            "status": "healthy",
            "connected": True,
            "active_connections": len(pool._connections),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }


async def check_models_health() -> Dict[str, Any]:
    """
    Verifica salud de los modelos.
    
    Returns:
        Estado de los modelos
    """
    status: Dict[str, Any] = {
        "llm": {"status": "unknown"},
        "embedding": {"status": "unknown"},
    }
    
    try:
        llm = await get_llm_model()
        status["llm"] = {
            "status": "healthy",
            "provider": llm.provider.value if hasattr(llm, 'provider') else "unknown",
        }
    except Exception as e:
        status["llm"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    try:
        embed = await get_embedding()
        status["embedding"] = {
            "status": "healthy",
            "model_type": embed.model_type.value if hasattr(embed, 'model_type') else "unknown",
        }
    except Exception as e:
        status["embedding"] = {
            "status": "unhealthy",
            "error": str(e),
        }
    
    return status


# ==========================================
# Testing Utilities
# ==========================================

def reset_dependencies() -> None:
    """
    Resetea todas las dependencias (para testing).
    
    Limpia el estado global para tests.
    """
    global _db_connection, _llm_instance, _embedding_model, _agent_container, _db_pool
    
    _db_connection = None
    _db_pool = None
    _llm_instance = None
    _embedding_model = None
    _agent_container = None
    
    # Reset pool singleton - al poner _instance en None, la próxima 
    # instanciación creará un pool nuevo
    DatabasePool._instance = None
    
    # Limpiar cache de get_settings en settings.py
    from backend.settings import get_settings as settings_get
    settings_get.cache_clear()


async def override_db(mock_db: Any) -> None:
    """
    Override de conexión a DB (para testing).
    
    Args:
        mock_db: Mock de conexión
    """
    global _db_connection
    _db_connection = mock_db


async def override_llm(mock_llm: BaseLLM) -> None:
    """
    Override de modelo LLM (para testing).
    
    Args:
        mock_llm: Mock de LLM
    """
    global _llm_instance
    _llm_instance = mock_llm


async def override_embedding(mock_embed: BaseEmbedding) -> None:
    """
    Override de modelo de embeddings (para testing).
    
    Args:
        mock_embed: Mock de embedding model
    """
    global _embedding_model
    _embedding_model = mock_embed

