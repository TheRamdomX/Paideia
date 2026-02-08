"""
chunking.py
Implementa hierarchical chunking (parent/child).
Divide documentos en chunks con relaciones jerárquicas.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from backend.settings import get_rag_config
from backend.utils.text import token_count


@dataclass
class Chunk:
    """Representa un chunk de texto."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    index: int = 0
    token_count: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    level: int = 0  # 0 = parent, 1+ = child
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Información de posición
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def __post_init__(self):
        if self.content and self.token_count == 0:
            self.token_count = token_count(self.content)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el chunk a diccionario."""
        return {
            "id": self.id,
            "content": self.content,
            "index": self.index,
            "token_count": self.token_count,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "level": self.level,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }
    
    @property
    def content_hash(self) -> str:
        """Hash del contenido para deduplicación."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class ChunkingResult:
    """Resultado del proceso de chunking."""
    
    chunks: List[Chunk] = field(default_factory=list)
    parent_chunks: List[Chunk] = field(default_factory=list)
    child_chunks: List[Chunk] = field(default_factory=list)
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.chunks and self.total_tokens == 0:
            self.total_tokens = sum(c.token_count for c in self.chunks)


# ==========================================
# Funciones de División de Texto
# ==========================================

def split_by_separators(
    text: str,
    separators: List[str]
) -> List[str]:
    """
    Divide texto por una lista de separadores en orden de prioridad.
    
    Args:
        text: Texto a dividir
        separators: Lista de separadores ordenados por prioridad
        
    Returns:
        Lista de partes
    """
    if not text:
        return []
    
    parts = [text]
    
    for separator in separators:
        new_parts = []
        for part in parts:
            if separator in part:
                split_parts = part.split(separator)
                # Mantener el separador al final de cada parte excepto la última
                for i, sp in enumerate(split_parts[:-1]):
                    new_parts.append(sp + separator)
                if split_parts[-1]:  # Agregar última parte si no está vacía
                    new_parts.append(split_parts[-1])
            else:
                new_parts.append(part)
        parts = new_parts
    
    return [p for p in parts if p.strip()]


def split_into_sentences(text: str) -> List[str]:
    """
    Divide texto en oraciones.
    
    Args:
        text: Texto a dividir
        
    Returns:
        Lista de oraciones
    """
    if not text:
        return []
    
    # Patrón para detectar fin de oración
    sentence_pattern = r'(?<=[.!?])\s+'
    
    sentences = re.split(sentence_pattern, text)
    
    return [s.strip() for s in sentences if s.strip()]


def merge_small_chunks(
    chunks: List[str],
    min_tokens: int = 50
) -> List[str]:
    """
    Combina chunks pequeños con el siguiente.
    
    Args:
        chunks: Lista de chunks
        min_tokens: Mínimo de tokens por chunk
        
    Returns:
        Lista de chunks combinados
    """
    if not chunks:
        return []
    
    merged = []
    current = ""
    
    for chunk in chunks:
        if current:
            combined = current + " " + chunk
            if token_count(current) < min_tokens:
                current = combined
            else:
                merged.append(current)
                current = chunk
        else:
            current = chunk
    
    if current:
        merged.append(current)
    
    return merged


# ==========================================
# Chunking Jerárquico
# ==========================================

def hierarchical_chunk(
    text: str,
    parent_chunk_size: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ChunkingResult:
    """
    Implementa chunking jerárquico (parent/child).
    
    Crea chunks padres grandes y chunks hijos más pequeños,
    estableciendo relaciones entre ellos.
    
    Args:
        text: Texto a dividir
        parent_chunk_size: Tamaño de chunks padre en tokens (default: 2x chunk_size)
        child_chunk_size: Tamaño de chunks hijo en tokens (default: chunk_size)
        chunk_overlap: Solapamiento entre chunks hijo
        metadata: Metadatos a agregar a todos los chunks
        
    Returns:
        ChunkingResult con chunks organizados jerárquicamente
    """
    config = get_rag_config()
    
    # Configuración de tamaños
    child_size = child_chunk_size or config["chunk_size"]
    _ = parent_chunk_size or (child_size * 2)  # Compatibilidad de firma
    overlap = chunk_overlap or config["chunk_overlap"]
    
    if not text or not text.strip():
        return ChunkingResult(metadata={"error": "Texto vacío"})
    
    # Separadores jerárquicos
    section_separators = ["\n\n\n", "\n\n", "\n"]
    
    # Paso 1: Dividir en secciones grandes (padres potenciales)
    sections = split_by_separators(text, section_separators[:1])  # Solo doble salto
    
    if not sections:
        sections = [text]
    
    all_chunks: List[Chunk] = []
    parent_chunks: List[Chunk] = []
    child_chunks: List[Chunk] = []
    
    global_index = 0

    base_metadata = dict(metadata or {})
    if "page" not in base_metadata and "page_number" in base_metadata:
        base_metadata["page"] = base_metadata["page_number"]
    
    for section in sections:
        section_tokens = token_count(section)
        
        # Si la sección es pequeña, es un chunk hijo directo
        if section_tokens <= child_size:
            chunk_metadata = dict(base_metadata)
            chunk_metadata["chunk_level"] = 1
            chunk_metadata["approximate_offsets"] = True
            chunk = Chunk(
                content=section.strip(),
                index=global_index,
                level=1,
                metadata=chunk_metadata,
            )
            all_chunks.append(chunk)
            child_chunks.append(chunk)
            global_index += 1
            continue
        
        # Crear chunk padre
        parent_content = section
        parent_metadata = dict(base_metadata)
        parent_metadata["chunk_level"] = 0
        parent_metadata["approximate_offsets"] = True
        parent_chunk = Chunk(
            content=parent_content.strip(),
            index=global_index,
            level=0,
            metadata=parent_metadata,
        )
        parent_chunks.append(parent_chunk)
        all_chunks.append(parent_chunk)
        global_index += 1
        
        # Crear chunks hijos de esta sección
        child_texts = create_overlapping_chunks(
            section,
            chunk_size=child_size,
            overlap=overlap
        )
        
        for child_text in child_texts:
            child_metadata = dict(base_metadata)
            child_metadata["chunk_level"] = 1
            child_metadata["approximate_offsets"] = True
            child_chunk = Chunk(
                content=child_text.strip(),
                index=global_index,
                level=1,
                parent_id=parent_chunk.id,
                metadata=child_metadata,
            )
            
            parent_chunk.children_ids.append(child_chunk.id)
            all_chunks.append(child_chunk)
            child_chunks.append(child_chunk)
            global_index += 1
    
    return ChunkingResult(
        chunks=all_chunks,
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        metadata=metadata or {},
    )


def create_overlapping_chunks(
    text: str,
    chunk_size: int,
    overlap: int
) -> List[str]:
    """
    Crea chunks con solapamiento.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño objetivo en tokens
        overlap: Tokens de solapamiento
        
    Returns:
        Lista de chunks con overlap
    """
    if not text:
        return []
    
    # Dividir primero en oraciones para cortes limpios
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = token_count(sentence)
        
        # Si agregar esta oración excede el límite
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Guardar chunk actual
            chunks.append(" ".join(current_chunk))
            
            # Calcular overlap: mantener últimas oraciones
            overlap_tokens = 0
            overlap_sentences = []
            
            for s in reversed(current_chunk):
                s_tokens = token_count(s)
                if overlap_tokens + s_tokens <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tokens
                else:
                    break
            
            current_chunk = overlap_sentences
            current_tokens = overlap_tokens
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Agregar último chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def simple_chunk(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> ChunkingResult:
    """
    Chunking simple sin jerarquía.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño en tokens
        chunk_overlap: Solapamiento en tokens
        metadata: Metadatos
        
    Returns:
        ChunkingResult con chunks planos
    """
    config = get_rag_config()
    
    size = chunk_size or config["chunk_size"]
    overlap = chunk_overlap or config["chunk_overlap"]
    
    chunk_texts = create_overlapping_chunks(text, size, overlap)
    
    chunks = []
    base_metadata = dict(metadata or {})
    if "page" not in base_metadata and "page_number" in base_metadata:
        base_metadata["page"] = base_metadata["page_number"]
    
    for i, chunk_text in enumerate(chunk_texts):
        chunk_metadata = dict(base_metadata)
        chunk_metadata["chunk_level"] = 0
        chunk_metadata["approximate_offsets"] = True
        chunk = Chunk(
            content=chunk_text,
            index=i,
            level=0,
            metadata=chunk_metadata,
        )
        chunks.append(chunk)
    
    return ChunkingResult(
        chunks=chunks,
        parent_chunks=[],
        child_chunks=chunks,
        metadata=metadata or {},
    )


# ==========================================
# Validación y Linking
# ==========================================

def validate_chunks(chunks: List[Chunk]) -> Tuple[bool, List[str]]:
    """
    Valida que los chunks cumplan con los criterios de calidad.
    
    Args:
        chunks: Lista de chunks a validar
        
    Returns:
        Tupla (todos_válidos, lista_de_errores)
    """
    errors = []
    
    for chunk in chunks:
        # Verificar contenido
        if not chunk.content or not chunk.content.strip():
            errors.append(f"Chunk {chunk.id}: contenido vacío")
            continue
        
        # Verificar longitud mínima
        if chunk.token_count < 10:
            errors.append(f"Chunk {chunk.id}: muy corto ({chunk.token_count} tokens)")
        
        # Verificar longitud máxima
        config = get_rag_config()
        max_tokens = config["chunk_size"] * 3  # 3x como máximo
        
        if chunk.token_count > max_tokens:
            errors.append(f"Chunk {chunk.id}: muy largo ({chunk.token_count} tokens)")
        
        # Verificar consistencia padre-hijo
        if chunk.parent_id and chunk.level == 0:
            errors.append(f"Chunk {chunk.id}: tiene parent_id pero level=0")
    
    return len(errors) == 0, errors


def link_chunks(
    chunks: List[Chunk]
) -> Dict[str, List[str]]:
    """
    Genera el mapa de enlaces entre chunks.
    
    Args:
        chunks: Lista de chunks
        
    Returns:
        Diccionario de relaciones {parent_id: [child_ids]}
    """
    links = {}
    
    for chunk in chunks:
        if chunk.children_ids:
            links[chunk.id] = chunk.children_ids
    
    return links


def get_chunk_hierarchy(
    chunks: List[Chunk]
) -> Dict[str, Any]:
    """
    Genera estructura jerárquica de chunks.
    
    Args:
        chunks: Lista de chunks
        
    Returns:
        Estructura jerárquica
    """
    # Indexar por ID
    chunk_map = {c.id: c for c in chunks}
    
    # Encontrar raíces (sin padre)
    roots = [c for c in chunks if c.parent_id is None and c.level == 0]
    
    def build_tree(chunk: Chunk) -> Dict[str, Any]:
        return {
            "id": chunk.id,
            "index": chunk.index,
            "token_count": chunk.token_count,
            "children": [
                build_tree(chunk_map[cid])
                for cid in chunk.children_ids
                if cid in chunk_map
            ]
        }
    
    return {
        "roots": [build_tree(r) for r in roots],
        "total_chunks": len(chunks),
        "parent_count": len([c for c in chunks if c.level == 0]),
        "child_count": len([c for c in chunks if c.level > 0]),
    }
