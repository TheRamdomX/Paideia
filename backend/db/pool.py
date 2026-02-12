"""
pool.py
Pool de conexiones multi-database para SurrealDB.
Permite múltiples usuarios en diferentes bases de datos simultáneamente.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict

from surrealdb import AsyncSurreal

from backend.settings import get_db_config


@dataclass
class DatabaseConnection:
    """Conexión a una base de datos específica."""
    client: AsyncSurreal
    database: str
    namespace: str
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    in_use: bool = False
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Ejecuta una query en esta conexión."""
        if params:
            return await self.client.query(query, params)
        return await self.client.query(query)
    
    async def create(self, table: str, data: Dict[str, Any]) -> Any:
        """Crea un registro."""
        return await self.client.create(table, data)
    
    async def select(self, resource: str) -> Any:
        """Selecciona registros."""
        return await self.client.select(resource)
    
    async def update(self, resource: str, data: Dict[str, Any]) -> Any:
        """Actualiza registros."""
        return await self.client.update(resource, data)
    
    async def delete(self, resource: str) -> Any:
        """Elimina registros."""
        return await self.client.delete(resource)


class DatabasePool:
    """
    Pool de conexiones multi-database.
    Mantiene una conexión por base de datos en lugar de cambiar la DB global.
    """
    
    _instance: Optional["DatabasePool"] = None
    
    def __new__(cls) -> "DatabasePool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._connections: Dict[str, DatabaseConnection] = {}
        self._config = get_db_config()
        self._lock = asyncio.Lock()
        self._default_database = self._config["database"]
        self._namespace = self._config["namespace"]
        self._initialized = True
    
    @property
    def default_database(self) -> str:
        """Base de datos por defecto."""
        return self._default_database
    
    async def _create_connection(self, database: str) -> DatabaseConnection:
        """
        Crea una nueva conexión a una base de datos.
        
        Args:
            database: Nombre de la base de datos
            
        Returns:
            Conexión establecida
        """
        client = AsyncSurreal(self._config["url"])
        await client.connect()
        
        await client.signin({
            "username": self._config["user"],
            "password": self._config["password"],
        })
        
        await client.use(self._namespace, database)
        
        return DatabaseConnection(
            client=client,
            database=database,
            namespace=self._namespace,
        )
    
    async def get_connection(self, database: Optional[str] = None) -> DatabaseConnection:
        """
        Obtiene una conexión para la base de datos especificada.
        Crea una nueva si no existe.
        
        Args:
            database: Nombre de la base de datos (usa default si no se especifica)
            
        Returns:
            Conexión a la base de datos
        """
        db_name = database or self._default_database
        
        async with self._lock:
            if db_name not in self._connections:
                self._connections[db_name] = await self._create_connection(db_name)
            
            conn = self._connections[db_name]
            conn.last_used = datetime.now(timezone.utc)
            return conn
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una query en la base de datos especificada.
        
        Args:
            query: Query SurrealQL
            params: Parámetros opcionales
            database: Base de datos a usar
            
        Returns:
            Resultados de la query
        """
        conn = await self.get_connection(database)
        
        try:
            if params:
                result = await conn.client.query(query, params)
            else:
                result = await conn.client.query(query)
            
            # Procesar resultados
            if isinstance(result, list):
                all_results = []
                for r in result:
                    if isinstance(r, dict) and "result" in r:
                        res = r["result"]
                        if isinstance(res, list):
                            all_results.extend(res)
                        elif res is not None:
                            all_results.append(res)
                    elif isinstance(r, list):
                        all_results.extend(r)
                    elif r is not None:
                        all_results.append(r)
                return all_results
            
            return [result] if result else []
            
        except Exception as e:
            raise RuntimeError(f"Error ejecutando query en {database}: {e}") from e
    
    async def create(
        self,
        table: str,
        data: Dict[str, Any],
        record_id: Optional[str] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crea un registro en la base de datos especificada.
        """
        conn = await self.get_connection(database)
        
        try:
            if record_id:
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                thing = f"{table}:{escaped_id}"
                result = await conn.client.create(thing, data)
            else:
                result = await conn.client.create(table, data)
            
            return result if isinstance(result, dict) else result[0] if result else {}
            
        except Exception as e:
            raise RuntimeError(f"Error creando registro: {e}") from e
    
    async def select(
        self,
        table: str,
        record_id: Optional[str] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Selecciona registros de la base de datos especificada.
        """
        conn = await self.get_connection(database)
        
        try:
            if record_id:
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                result = await conn.client.select(f"{table}:{escaped_id}")
            else:
                result = await conn.client.select(table)
            
            if isinstance(result, list):
                return result
            return [result] if result else []
            
        except Exception as e:
            raise RuntimeError(f"Error seleccionando registros: {e}") from e
    
    async def update(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any],
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Actualiza un registro en la base de datos especificada.
        """
        conn = await self.get_connection(database)
        
        try:
            escaped_id = f"`{record_id}`" if "-" in record_id else record_id
            result = await conn.client.merge(f"{table}:{escaped_id}", data)
            return result if isinstance(result, dict) else result[0] if result else {}
            
        except Exception as e:
            raise RuntimeError(f"Error actualizando registro: {e}") from e
    
    async def delete(
        self,
        table: str,
        record_id: Optional[str] = None,
        database: Optional[str] = None,
    ) -> bool:
        """
        Elimina registros de la base de datos especificada.
        """
        conn = await self.get_connection(database)
        
        try:
            if record_id:
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                await conn.client.delete(f"{table}:{escaped_id}")
            else:
                await conn.client.delete(table)
            return True
            
        except Exception as e:
            raise RuntimeError(f"Error eliminando registro: {e}") from e
    
    async def list_databases(self) -> List[str]:
        """
        Lista las bases de datos disponibles.
        """
        conn = await self.get_connection()
        
        try:
            result = await conn.client.query("INFO FOR NS;")
            
            if result:
                if isinstance(result, dict):
                    dbs = result.get("databases") or {}
                    if isinstance(dbs, dict):
                        return list(dbs.keys())
                elif isinstance(result, list):
                    for r in result:
                        if isinstance(r, dict):
                            data = r.get("result", r)
                            if isinstance(data, dict):
                                dbs = data.get("databases") or data.get("db") or {}
                                if isinstance(dbs, dict):
                                    return list(dbs.keys())
            
            return [self._default_database]
        except Exception as e:
            print(f"Error listing databases: {e}")
            return [self._default_database]
    
    async def create_database(self, database: str) -> bool:
        """
        Crea una nueva base de datos.
        """
        conn = await self.get_connection()
        
        try:
            await conn.client.query(f"DEFINE DATABASE {database};")
            return True
        except Exception as e:
            raise RuntimeError(f"Error creando database {database}: {e}") from e
    
    async def close_all(self) -> None:
        """
        Cierra todas las conexiones del pool.
        """
        async with self._lock:
            for conn in self._connections.values():
                try:
                    await conn.client.close()
                except Exception:
                    pass
            self._connections.clear()
    
    async def close_connection(self, database: str) -> None:
        """
        Cierra la conexión a una base de datos específica.
        """
        async with self._lock:
            if database in self._connections:
                try:
                    await self._connections[database].client.close()
                except Exception:
                    pass
                del self._connections[database]
    
    def get_active_databases(self) -> List[str]:
        """
        Retorna lista de bases de datos con conexiones activas.
        """
        return list(self._connections.keys())


# ==========================================
# Instancia Global y Funciones de Conveniencia
# ==========================================

_pool: Optional[DatabasePool] = None


def get_pool() -> DatabasePool:
    """Obtiene el pool de conexiones (singleton)."""
    global _pool
    if _pool is None:
        _pool = DatabasePool()
    return _pool


async def get_db_for_database(database: Optional[str] = None) -> DatabaseConnection:
    """
    Obtiene conexión para una base de datos específica.
    
    Args:
        database: Nombre de la base de datos
        
    Returns:
        Conexión a la base de datos
    """
    pool = get_pool()
    return await pool.get_connection(database)


async def execute_in_db(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    database: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Ejecuta query en una base de datos específica.
    """
    pool = get_pool()
    return await pool.execute(query, params, database)


async def close_pool() -> None:
    """Cierra todas las conexiones del pool."""
    global _pool
    if _pool:
        await _pool.close_all()
        _pool = None
