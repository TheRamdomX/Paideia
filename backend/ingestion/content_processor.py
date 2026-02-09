"""
content_processor.py
Wrapper sobre content-core.
Normaliza a markdown.
"""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from pypdf import PdfReader

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
    """Resultado del procesamiento de contenido."""
    
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


# ==========================================
# Detección de Tipo de Contenido
# ==========================================

def detect_content_type(
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None
) -> ContentType:
    """
    Identifica el tipo de contenido basándose en el input.
    
    Args:
        content: Contenido como string
        file_path: Path al archivo
        url: URL del contenido
        
    Returns:
        ContentType detectado
    """
    # Detección por URL
    if url:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            # Verificar extensión en URL
            path_lower = parsed.path.lower()
            if path_lower.endswith(".pdf"):
                return ContentType.PDF
            elif path_lower.endswith((".mp3", ".wav", ".m4a", ".ogg")):
                return ContentType.AUDIO
            elif path_lower.endswith((".mp4", ".avi", ".mov", ".webm")):
                return ContentType.VIDEO
            return ContentType.URL
    
    # Detección por archivo
    if file_path:
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if suffix == ".pdf" or (mime_type and "pdf" in mime_type):
            return ContentType.PDF
        elif suffix == ".docx":
            return ContentType.DOCX
        elif suffix in (".md", ".markdown"):
            return ContentType.MARKDOWN
        elif suffix in (".html", ".htm"):
            return ContentType.HTML
        elif suffix in (".txt", ".text"):
            return ContentType.TEXT
        elif suffix in (".mp3", ".wav", ".m4a", ".ogg", ".flac"):
            return ContentType.AUDIO
        elif suffix in (".mp4", ".avi", ".mov", ".webm", ".mkv"):
            return ContentType.VIDEO
    
    # Detección por contenido
    if content:
        content_lower = content.lower().strip()
        
        # HTML
        if content_lower.startswith(("<!doctype", "<html", "<head", "<body")):
            return ContentType.HTML
        
        # Markdown (heurísticas)
        md_patterns = [
            r'^#{1,6}\s',      # Headers
            r'\*\*[^*]+\*\*',  # Bold
            r'\[.+\]\(.+\)',   # Links
            r'^[-*+]\s',       # Lists
            r'```',            # Code blocks
        ]
        
        md_matches = sum(1 for p in md_patterns if re.search(p, content, re.MULTILINE))
        if md_matches >= 2:
            return ContentType.MARKDOWN
        
        return ContentType.TEXT
    
    return ContentType.UNKNOWN


# ==========================================
# Procesamiento de Contenido
# ==========================================

def cleanup_text(text: str) -> str:
    """
    Limpieza básica de texto.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    if not text:
        return ""
    
    # Normalizar texto
    text = normalize_text(text)
    
    # Limpiar markdown si es necesario
    text = clean_markdown(text)
    
    return text.strip()


def extract_title_from_content(content: str) -> str:
    """
    Extrae el título del contenido.
    
    Args:
        content: Contenido a analizar
        
    Returns:
        Título extraído o string vacío
    """
    if not content:
        return ""
    
    # Buscar header markdown
    header_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if header_match:
        return header_match.group(1).strip()
    
    # Buscar título HTML
    title_match = re.search(r'<title>([^<]+)</title>', content, re.IGNORECASE)
    if title_match:
        return title_match.group(1).strip()
    
    # Usar primera línea no vacía
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            # Truncar si es muy largo
            return line[:100] + "..." if len(line) > 100 else line
    
    return "Sin título"


def process_text_content(content: str) -> str:
    """
    Procesa contenido de texto plano.
    
    Args:
        content: Texto a procesar
        
    Returns:
        Contenido normalizado
    """
    return normalize_text(content)


def process_markdown_content(content: str) -> str:
    """
    Procesa contenido markdown.
    
    Args:
        content: Markdown a procesar
        
    Returns:
        Contenido normalizado (mantiene estructura)
    """
    # Normalizar pero mantener estructura
    content = normalize_text(content)
    
    # Normalizar headers
    content = re.sub(r'^(#{1,6})\s*', r'\1 ', content, flags=re.MULTILINE)
    
    # Normalizar listas
    content = re.sub(r'^(\s*)[-*+](\s+)', r'\1- ', content, flags=re.MULTILINE)
    
    return content


def process_html_content(content: str) -> str:
    """
    Procesa contenido HTML y lo convierte a texto limpio.
    
    Args:
        content: HTML a procesar
        
    Returns:
        Contenido como texto
    """
    # Eliminar scripts y estilos
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Convertir algunos elementos a markdown-like
    content = re.sub(r'<h1[^>]*>([^<]+)</h1>', r'# \1\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<h2[^>]*>([^<]+)</h2>', r'## \1\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<h3[^>]*>([^<]+)</h3>', r'### \1\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<p[^>]*>([^<]+)</p>', r'\1\n\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<br\s*/?>', '\n', content, flags=re.IGNORECASE)
    content = re.sub(r'<li[^>]*>([^<]+)</li>', r'- \1\n', content, flags=re.IGNORECASE)
    
    # Eliminar todas las etiquetas restantes
    content = re.sub(r'<[^>]+>', '', content)
    
    # Decodificar entidades HTML comunes
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
    }
    for entity, char in html_entities.items():
        content = content.replace(entity, char)
    
    return normalize_text(content)


async def process_url_content(url: str) -> str:
    """
    Descarga y procesa contenido de una URL.
    
    Args:
        url: URL a procesar
        
    Returns:
        Contenido procesado
    """
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get("content-type", "")
            
            if "html" in content_type:
                return process_html_content(response.text)
            elif "json" in content_type:
                import json
                data = response.json()
                return json.dumps(data, indent=2, ensure_ascii=False)
            else:
                return response.text
                
    except Exception as e:
        raise RuntimeError(f"Error descargando URL {url}: {e}") from e


async def process_pdf_content(file_path: str) -> str:
    """
    Procesa contenido de un archivo PDF.
    
    Args:
        file_path: Path al archivo PDF
        
    Returns:
        Texto extraído del PDF
        
    Note:
        Requiere pypdf instalado. Si no está disponible,
        retorna un mensaje indicando que se necesita la librería.
    """
    try:        
        reader = PdfReader(file_path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        return normalize_text("\n\n".join(text_parts))
        
    except ImportError:
        return f"[PDF: {file_path}] - Instalar pypdf para procesar PDFs"
    except Exception as e:
        raise RuntimeError(f"Error procesando PDF {file_path}: {e}") from e


async def process_content(
    content: Optional[str] = None,
    file_path: Optional[str] = None,
    url: Optional[str] = None,
    title: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ProcessedContent:
    """
    Procesa contenido de cualquier fuente.
    
    Esta es la función principal que orquesta el procesamiento
    de diferentes tipos de contenido.
    
    Args:
        content: Contenido como string (texto, markdown, html)
        file_path: Path a un archivo
        url: URL para descargar contenido
        title: Título opcional (se detecta automáticamente si no se provee)
        metadata: Metadatos adicionales
        
    Returns:
        ProcessedContent con el contenido normalizado
        
    Raises:
        ValueError: Si no se proporciona ninguna fuente de contenido
    """
    if not any([content, file_path, url]):
        raise ValueError("Debe proporcionar content, file_path o url")
    
    # Detectar tipo de contenido
    content_type = detect_content_type(content, file_path, url)
    
    raw_content = ""
    normalized_content = ""
    detected_title = title or ""
    
    # Procesar según el tipo
    if url and content_type == ContentType.URL:
        raw_content = await process_url_content(url)
        normalized_content = raw_content
        if not detected_title:
            detected_title = extract_title_from_content(raw_content)
    
    elif file_path:
        path = Path(file_path)
        
        if content_type == ContentType.PDF:
            raw_content = await process_pdf_content(file_path)
            normalized_content = raw_content
        else:
            # Leer archivo de texto
            raw_content = path.read_text(encoding="utf-8")
            
            if content_type == ContentType.HTML:
                normalized_content = process_html_content(raw_content)
            elif content_type == ContentType.MARKDOWN:
                normalized_content = process_markdown_content(raw_content)
            else:
                normalized_content = process_text_content(raw_content)
        
        if not detected_title:
            detected_title = path.stem.replace("_", " ").replace("-", " ").title()
    
    elif content:
        raw_content = content
        
        if content_type == ContentType.HTML:
            normalized_content = process_html_content(content)
        elif content_type == ContentType.MARKDOWN:
            normalized_content = process_markdown_content(content)
        else:
            normalized_content = process_text_content(content)
        
        if not detected_title:
            detected_title = extract_title_from_content(content)
    
    return ProcessedContent(
        raw_content=raw_content,
        normalized_content=normalized_content,
        content_type=content_type,
        title=detected_title,
        url=url,
        file_path=file_path,
        metadata=metadata or {},
    )


# ==========================================
# Utilidades de Validación
# ==========================================

def validate_content(content: ProcessedContent) -> tuple[bool, Optional[str]]:
    """
    Valida que el contenido procesado sea válido.
    
    Args:
        content: Contenido a validar
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    if not content.normalized_content:
        return False, "Contenido normalizado vacío"
    
    if content.word_count < 10:
        return False, f"Contenido muy corto: {content.word_count} palabras"
    
    if not content.title:
        return False, "Título no detectado"
    
    return True, None
