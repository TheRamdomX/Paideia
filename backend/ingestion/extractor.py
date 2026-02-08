"""Adaptador de extracción usando content-core."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional


async def extract_content(
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Delega extracción a content-core sin parsing manual local."""
    try:
        from content_core import extract_content as core_extract_content  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "content-core no está disponible. Instálalo para extraer contenido."
        ) from exc

    kwargs = {
        "content": content,
        "file_path": file_path,
        "url": url,
        "metadata": metadata or {},
    }

    if inspect.iscoroutinefunction(core_extract_content):
        return await core_extract_content(**kwargs)

    result = core_extract_content(**kwargs)
    if inspect.isawaitable(result):
        return await result
    return result
