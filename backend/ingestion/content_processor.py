"""
content_processor.py
Post-procesa resultados de extracción provenientes de content-core.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from backend.utils.text import clean_markdown, normalize_text


class ContentType(str, Enum):
    """Tipos de contenido soportados."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    DOCX = "docx"
    URL = "url"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class ProcessedContent:
    """Resultado del post-procesamiento de contenido."""

    raw_content: str = ""
    normalized_content: str = ""
    content_type: ContentType = ContentType.UNKNOWN
    title: str = ""
    author: Optional[str] = None
    url: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_count: int = 0
    language: str = "es"

    def __post_init__(self):
        if self.normalized_content:
            self.word_count = len(self.normalized_content.split())


def extract_title_from_content(content: str) -> str:
    """Extrae un título desde markdown/texto ya extraído."""
    if not content:
        return ""

    header_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if header_match:
        return header_match.group(1).strip()

    lines = content.strip().split("\n")
    for line in lines:
        line = line.strip().lstrip("#").strip()
        if line and len(line) > 5:
            return line[:100] + "..." if len(line) > 100 else line

    return "Sin título"


def _pick_content_type(metadata: Dict[str, Any]) -> ContentType:
    raw_type = str(
        metadata.get("content_type")
        or metadata.get("mime_type")
        or metadata.get("source_type")
        or "unknown"
    ).lower()

    for ctype in ContentType:
        if ctype.value == raw_type:
            return ctype
    return ContentType.UNKNOWN


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _extract_text(source_result: Any) -> str:
    for candidate in ("text", "content", "markdown", "normalized_text"):
        value = getattr(source_result, candidate, None)
        if isinstance(value, str) and value.strip():
            return value

    payload = _to_dict(source_result)
    for candidate in ("text", "content", "markdown", "normalized_text"):
        value = payload.get(candidate)
        if isinstance(value, str) and value.strip():
            return value

    return ""


async def process_content(
    source_result: Any,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> ProcessedContent:
    """
    Post-procesa exclusivamente el resultado de content-core.

    No realiza parsing manual, OCR, descarga ni detección de tipo.
    """
    if source_result is None:
        raise ValueError("source_result es requerido")

    source_metadata = _to_dict(getattr(source_result, "metadata", None))
    payload = _to_dict(source_result)
    if not source_metadata and isinstance(payload.get("metadata"), dict):
        source_metadata = dict(payload["metadata"])

    raw_text = _extract_text(source_result)
    cleaned = clean_markdown(raw_text)
    normalized = normalize_text(cleaned).strip()

    merged_metadata: Dict[str, Any] = {
        **source_metadata,
        **(metadata or {}),
    }

    detected_title = title or payload.get("title") or extract_title_from_content(normalized)

    lang = str(
        merged_metadata.get("language")
        or merged_metadata.get("lang")
        or merged_metadata.get("ocr_language")
        or "es"
    )

    return ProcessedContent(
        raw_content=raw_text,
        normalized_content=normalized,
        content_type=_pick_content_type(merged_metadata),
        title=detected_title,
        author=payload.get("author") or merged_metadata.get("author"),
        url=payload.get("url") or merged_metadata.get("source"),
        file_path=payload.get("file_path") or merged_metadata.get("file_path"),
        metadata=merged_metadata,
        language=lang,
    )


def validate_content(content: ProcessedContent) -> tuple[bool, Optional[str]]:
    """Valida contenido procesado."""
    if not content.normalized_content:
        return False, "Contenido normalizado vacío"

    if content.word_count < 10:
        return False, f"Contenido muy corto: {content.word_count} palabras"

    if not content.title:
        return False, "Título no detectado"

    return True, None
