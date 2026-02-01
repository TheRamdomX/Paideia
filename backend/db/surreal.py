"""
surreal.py
Cliente SurrealDB y helpers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Optional, List, Dict

from surrealdb import Surreal

from backend.settings import get_db_config


class SurrealDBClient:
    """
    Cliente singleton para SurrealDB.
    Maneja conexión, queries y transacciones.
    """
    
    _instance: Optional["SurrealDBClient"] = None
    _client: Optional[Surreal] = None
    _connected: bool = False
    
    def __new__(cls) -> "SurrealDBClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def is_connected(self) -> bool:
        """Verifica si hay conexión activa."""
        return self._connected and self._client is not None
    
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
            self._client = Surreal(config["url"])
            await self._client.connect()
            
            # Autenticación
            await self._client.signin({
                "user": config["user"],
                "pass": config["password"],
            })
            
            # Seleccionar namespace y database
            await self._client.use(config["namespace"], config["database"])
            
            self._connected = True
            
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Error conectando a SurrealDB: {e}") from e
    
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
                result = await self._client.create(f"{table}:{record_id}", data)
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
                result = await self._client.select(f"{table}:{record_id}")
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
        Actualiza un registro.
        
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
            result = await self._client.update(f"{table}:{record_id}", data)
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
                await self._client.delete(f"{table}:{record_id}")
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
