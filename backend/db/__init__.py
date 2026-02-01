"""
Database package.
Cliente SurrealDB y migraciones.
"""

from backend.db.surreal import (
    SurrealDBClient,
    get_db,
    connect,
    execute,
    close,
)

__all__ = [
    "SurrealDBClient",
    "get_db",
    "connect",
    "execute",
    "close",
]
