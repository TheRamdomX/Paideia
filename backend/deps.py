"""
deps.py
Inyección de dependencias para FastAPI.
Proporciona DB, modelos y agentes como dependencias.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, AsyncGenerator, Dict, Optional, Type

from backend.db.surreal import get_db as get_surreal_db, connect, close, execute
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

_db_connection: Optional[Any] = None
_llm_instance: Optional[BaseLLM] = None
_embedding_model: Optional[BaseEmbedding] = None


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

async def get_db() -> AsyncGenerator[Any, None]:
    """
    Dependencia para obtener conexión a la base de datos.
    
    Yields:
        Conexión a SurrealDB
        
    Example:
        ```python
        @app.get("/items")
        async def get_items(db = Depends(get_db)):
            result = await db.query("SELECT * FROM items")
            return result
        ```
    """
    global _db_connection
    
    if _db_connection is None:
        _db_connection = await get_surreal_db()
    
    try:
        yield _db_connection
    finally:
        # La conexión se mantiene abierta (pool)
        pass


async def init_db() -> None:
    """Inicializa la base de datos con el schema."""
    # La conexión se inicializa automáticamente
    await get_surreal_db()


async def close_db() -> None:
    """Cierra la conexión a la base de datos."""
    global _db_connection
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
        async for db in get_db():
            # Ejecutar query simple
            result = await db.query("INFO FOR DB")
            return {
                "status": "healthy",
                "connected": True,
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
    global _db_connection, _llm_instance, _embedding_model, _agent_container
    
    _db_connection = None
    _llm_instance = None
    _embedding_model = None
    _agent_container = None
    
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

