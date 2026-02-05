"""
surreal.py
Cliente SurrealDB y helpers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional, List, Dict

from surrealdb import AsyncSurreal

from backend.settings import get_db_config


class SurrealDBClient:
    """
    Cliente singleton para SurrealDB.
    Maneja conexión, queries y transacciones.
    Soporta cambio dinámico de database.
    """
    
    _instance: Optional["SurrealDBClient"] = None
    _client: Optional[AsyncSurreal] = None
    _connected: bool = False
    _current_namespace: str = ""
    _current_database: str = ""
    
    def __new__(cls) -> "SurrealDBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def is_connected(self) -> bool:
        """Verifica si hay conexión activa."""
        return self._connected and self._client is not None
    
    @property
    def current_database(self) -> str:
        """Devuelve la base de datos actual."""
        return self._current_database
    
    @property
    def current_namespace(self) -> str:
        """Devuelve el namespace actual."""
        return self._current_namespace
    
    async def connect(self) -> None:
        """
        Establece conexión con SurrealDB.
        
        Raises:
            ConnectionError: Si no se puede conectar
        """
        if self._connected:
            return
        
        config = get_db_config()
        
        try:
            self._client = AsyncSurreal(config["url"])
            await self._client.connect()
            
            # Autenticación root
            await self._client.signin({
                "username": config["user"],
                "password": config["password"],
            })
            
            # Seleccionar namespace y database
            await self._client.use(config["namespace"], config["database"])
            self._current_namespace = config["namespace"]
            self._current_database = config["database"]
            
            self._connected = True
            
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Error conectando a SurrealDB: {e}") from e
    
    async def use_database(self, database: str, namespace: Optional[str] = None) -> None:
        """
        Cambia a otra base de datos.
        
        Args:
            database: Nombre de la base de datos
            namespace: Namespace (opcional, usa el actual si no se especifica)
        """
        if not self.is_connected:
            await self.connect()
        
        ns = namespace or self._current_namespace
        
        try:
            await self._client.use(ns, database)
            self._current_namespace = ns
            self._current_database = database
        except Exception as e:
            raise RuntimeError(f"Error cambiando a database {database}: {e}") from e
    
    async def list_databases(self) -> List[str]:
        """
        Lista las bases de datos disponibles en el namespace actual.
        
        Returns:
            Lista de nombres de bases de datos
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            result = await self._client.query("INFO FOR NS;")
            
            # Extraer databases del resultado
            # El resultado puede venir en diferentes formatos según la versión
            if result:
                # Si es dict directo
                if isinstance(result, dict):
                    dbs = result.get("databases") or {}
                    if isinstance(dbs, dict):
                        return list(dbs.keys())
                # Si es lista
                elif isinstance(result, list):
                    for r in result:
                        if isinstance(r, dict):
                            # Formato: {"result": {"databases": {...}}} o {"databases": {...}}
                            data = r.get("result", r)
                            if isinstance(data, dict):
                                dbs = data.get("databases") or data.get("db") or {}
                                if isinstance(dbs, dict):
                                    return list(dbs.keys())
            
            return [self._current_database] if self._current_database else []
        except Exception as e:
            print(f"Error listing databases: {e}")
            return [self._current_database] if self._current_database else []
    
    async def create_database(self, database: str) -> bool:
        """
        Crea una nueva base de datos.
        
        Args:
            database: Nombre de la base de datos a crear
            
        Returns:
            True si se creó exitosamente
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # Crear database usando USE (la crea si no existe)
            await self._client.query(f"DEFINE DATABASE {database};")
            return True
        except Exception as e:
            raise RuntimeError(f"Error creando database {database}: {e}") from e
    
    async def disconnect(self) -> None:
        """Cierra la conexión con SurrealDB."""
        if self._client and self._connected:
            await self._client.close()
            self._connected = False
            self._client = None
    
    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una query en SurrealDB.
        
        Args:
            query: Query SurrealQL a ejecutar
            params: Parámetros para la query
            
        Returns:
            Lista de resultados
            
        Raises:
            ConnectionError: Si no hay conexión
            RuntimeError: Si la query falla
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if params:
                result = await self._client.query(query, params)
            else:
                result = await self._client.query(query)
            
            # SurrealDB devuelve lista de resultados por cada statement
            if isinstance(result, list):
                # Aplanar resultados si es necesario
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
            raise RuntimeError(f"Error ejecutando query: {e}") from e
    
    async def create(
        self,
        table: str,
        data: Dict[str, Any],
        record_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Crea un registro en una tabla.
        
        Args:
            table: Nombre de la tabla
            data: Datos del registro
            record_id: ID opcional del registro
            
        Returns:
            Registro creado
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if record_id:
                # Escapar IDs con caracteres especiales usando backticks
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                thing = f"{table}:{escaped_id}"
                result = await self._client.create(thing, data)
            else:
                result = await self._client.create(table, data)
            
            return result if isinstance(result, dict) else result[0] if result else {}
            
        except Exception as e:
            raise RuntimeError(f"Error creando registro: {e}") from e
    
    async def select(
        self,
        table: str,
        record_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Selecciona registros de una tabla.
        
        Args:
            table: Nombre de la tabla
            record_id: ID opcional para seleccionar uno específico
            
        Returns:
            Lista de registros
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if record_id:
                # Escapar IDs con caracteres especiales usando backticks
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                result = await self._client.select(f"{table}:{escaped_id}")
            else:
                result = await self._client.select(table)
            
            if isinstance(result, list):
                return result
            return [result] if result else []
            
        except Exception as e:
            raise RuntimeError(f"Error seleccionando registros: {e}") from e
    
    async def update(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Actualiza un registro (merge parcial).
        
        Args:
            table: Nombre de la tabla
            record_id: ID del registro
            data: Datos a actualizar
            
        Returns:
            Registro actualizado
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # Escapar IDs con caracteres especiales usando backticks
            escaped_id = f"`{record_id}`" if "-" in record_id else record_id
            # Usar merge en lugar de update para preservar campos existentes
            result = await self._client.merge(f"{table}:{escaped_id}", data)
            return result if isinstance(result, dict) else result[0] if result else {}
            
        except Exception as e:
            raise RuntimeError(f"Error actualizando registro: {e}") from e
    
    async def delete(
        self,
        table: str,
        record_id: Optional[str] = None
    ) -> bool:
        """
        Elimina registros.
        
        Args:
            table: Nombre de la tabla
            record_id: ID opcional del registro
            
        Returns:
            True si se eliminó correctamente
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            if record_id:
                # Escapar IDs con caracteres especiales usando backticks
                escaped_id = f"`{record_id}`" if "-" in record_id else record_id
                await self._client.delete(f"{table}:{escaped_id}")
            else:
                await self._client.delete(table)
            return True
            
        except Exception as e:
            raise RuntimeError(f"Error eliminando registro: {e}") from e
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager para transacciones.
        
        Usage:
            async with db.transaction():
                await db.execute("CREATE ...")
                await db.execute("UPDATE ...")
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            await self._client.query("BEGIN TRANSACTION;")
            yield self
            await self._client.query("COMMIT TRANSACTION;")
        except Exception as e:
            await self._client.query("CANCEL TRANSACTION;")
            raise RuntimeError(f"Transacción cancelada: {e}") from e


# ==========================================
# Funciones de conveniencia
# ==========================================

_db_client: Optional[SurrealDBClient] = None


async def get_db() -> SurrealDBClient:
    """
    Obtiene la instancia del cliente de base de datos.
    
    Returns:
        SurrealDBClient: Cliente conectado
    """
    global _db_client
    
    if _db_client is None:
        _db_client = SurrealDBClient()
    
    if not _db_client.is_connected:
        await _db_client.connect()
    
    return _db_client


async def connect() -> SurrealDBClient:
    """
    Conecta a la base de datos y devuelve el cliente.
    Alias de get_db() para compatibilidad.
    
    Returns:
        SurrealDBClient: Cliente conectado
    """
    return await get_db()


async def execute(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Ejecuta una query directamente.
    
    Args:
        query: Query SurrealQL
        params: Parámetros opcionales
        
    Returns:
        Resultados de la query
    """
    db = await get_db()
    return await db.execute(query, params)


async def close() -> None:
    """Cierra la conexión a la base de datos."""
    global _db_client
    
    if _db_client and _db_client.is_connected:
        await _db_client.disconnect()
        _db_client = None


async def switch_database(database: str) -> None:
    """
    Cambia a una base de datos diferente.
    
    Args:
        database: Nombre de la base de datos
    """
    db = await get_db()
    await db.use_database(database)


async def list_databases() -> List[str]:
    """
    Lista las bases de datos disponibles.
    
    Returns:
        Lista de nombres de bases de datos
    """
    db = await get_db()
    return await db.list_databases()


async def create_database(database: str) -> bool:
    """
    Crea una nueva base de datos.
    
    Args:
        database: Nombre de la base de datos a crear
        
    Returns:
        True si se creó exitosamente
    """
    db = await get_db()
    return await db.create_database(database)


def get_current_database() -> str:
    """
    Obtiene el nombre de la base de datos actual.
    
    Returns:
        Nombre de la base de datos actual
    """
    if _db_client and _db_client.is_connected:
        return _db_client.current_database
    return ""
