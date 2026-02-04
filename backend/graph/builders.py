"""
builders.py
Crea y actualiza nodos desde transformaciones.
Funciones para construir el grafo de conocimiento.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.db.surreal import execute, get_db
from backend.graph.schema import (
    BaseNode,
    ChunkNode,
    ConceptNode,
    DefinitionNode,
    Edge,
    EdgeType,
    NodeType,
    SourceNode,
    TopicNode,
    validate_edge,
    validate_node,
)


# ==========================================
# Creación de Nodos
# ==========================================

async def create_node(
    node: BaseNode,
    table: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crea un nodo genérico en la base de datos.
    
    Args:
        node: Nodo a crear
        table: Nombre de la tabla (si no se especifica, usa el tipo)
        
    Returns:
        Nodo creado
        
    Raises:
        ValueError: Si el nodo no es válido
    """
    # Validar nodo
    is_valid, error = validate_node(node)
    if not is_valid:
        raise ValueError(f"Nodo inválido: {error}")
    
    # Determinar tabla
    if table is None:
        table = node.type.value
    
    # Generar ID si no existe
    if node.id is None:
        node.id = str(uuid4())
    
    # Actualizar timestamps
    node.updated_at = datetime.utcnow()
    
    # Crear en DB
    db = await get_db()
    node_dict = node.to_dict()
    
    # Remover el id del dict (lo usamos en la tabla:id)
    node_id = node_dict.pop("id")
    
    result = await db.create(table, node_dict, record_id=node_id)
    
    return result


async def create_concept_node(
    name: str,
    description: str = "",
    aliases: Optional[List[str]] = None,
    difficulty_level: int = 1,
    importance_score: float = 0.5,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea un nodo de concepto.
    
    Args:
        name: Nombre del concepto
        description: Descripción del concepto
        aliases: Nombres alternativos
        difficulty_level: Nivel de dificultad (1-5)
        importance_score: Importancia del concepto (0-1)
        embedding: Vector de embedding
        metadata: Metadatos adicionales
        
    Returns:
        Concepto creado
    """
    concept = ConceptNode(
        name=name,
        description=description,
        aliases=aliases or [],
        difficulty_level=difficulty_level,
        importance_score=importance_score,
        embedding=embedding,
        metadata=metadata or {},
    )
    
    return await create_node(concept)


async def create_topic_node(
    name: str,
    description: str = "",
    learning_objectives: Optional[List[str]] = None,
    order_index: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea un nodo de tema/unidad.
    
    Args:
        name: Nombre del tema
        description: Descripción
        learning_objectives: Objetivos de aprendizaje
        order_index: Orden en el curso
        metadata: Metadatos adicionales
        
    Returns:
        Tema creado
    """
    topic = TopicNode(
        name=name,
        description=description,
        learning_objectives=learning_objectives or [],
        order_index=order_index,
        metadata=metadata or {},
    )
    
    return await create_node(topic)


async def create_chunk_node(
    content: str,
    source_id: str,
    chunk_index: int = 0,
    parent_chunk_id: Optional[str] = None,
    token_count: int = 0,
    embedding: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crea un nodo de chunk (fragmento de texto).
    
    Args:
        content: Contenido del chunk
        source_id: ID de la fuente
        chunk_index: Índice del chunk en la fuente
        parent_chunk_id: ID del chunk padre (para jerarquía)
        token_count: Número de tokens
        embedding: Vector de embedding
        metadata: Metadatos adicionales
        chunk_id: ID específico para el chunk (opcional)
        
    Returns:
        Chunk creado
    """
    chunk = ChunkNode(
        id=chunk_id,
        content=content,
        source_id=source_id,
        chunk_index=chunk_index,
        parent_chunk_id=parent_chunk_id,
        token_count=token_count,
        embedding=embedding,
        metadata=metadata or {},
    )
    
    return await create_node(chunk)


async def create_source_node(
    title: str,
    content_type: str,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    author: Optional[str] = None,
    total_chunks: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea un nodo de fuente.
    
    Args:
        title: Título del documento
        content_type: Tipo de contenido (pdf, url, audio, etc.)
        url: URL si es web
        file_path: Path si es archivo local
        author: Autor del documento
        total_chunks: Total de chunks generados
        metadata: Metadatos adicionales
        
    Returns:
        Fuente creada
    """
    source = SourceNode(
        title=title,
        content_type=content_type,
        url=url,
        file_path=file_path,
        author=author,
        total_chunks=total_chunks,
        metadata=metadata or {},
    )
    
    return await create_node(source)


async def create_definition_node(
    concept_id: str,
    text: str,
    source_chunk_id: Optional[str] = None,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea un nodo de definición.
    
    Args:
        concept_id: ID del concepto que define
        text: Texto de la definición
        source_chunk_id: ID del chunk fuente
        confidence: Confianza en la definición
        metadata: Metadatos adicionales
        
    Returns:
        Definición creada
    """
    definition = DefinitionNode(
        concept_id=concept_id,
        text=text,
        source_chunk_id=source_chunk_id,
        confidence=confidence,
        metadata=metadata or {},
    )
    
    return await create_node(definition)


# ==========================================
# Creación de Relaciones
# ==========================================

async def create_edge(
    from_table: str,
    from_id: str,
    to_table: str,
    to_id: str,
    edge_type: EdgeType,
    weight: float = 1.0,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea una relación entre dos nodos.
    
    Args:
        from_table: Tabla del nodo origen
        from_id: ID del nodo origen
        to_table: Tabla del nodo destino
        to_id: ID del nodo destino
        edge_type: Tipo de relación
        weight: Peso de la relación
        confidence: Confianza en la relación
        metadata: Metadatos adicionales
        
    Returns:
        Relación creada
    """
    # Escapar IDs con caracteres especiales usando backticks
    escaped_from_id = f"`{from_id}`" if "-" in from_id else from_id
    escaped_to_id = f"`{to_id}`" if "-" in to_id else to_id
    
    query = f"""
    RELATE {from_table}:{escaped_from_id}->{edge_type.value}->{to_table}:{escaped_to_id}
    SET 
        weight = $weight,
        confidence = $confidence,
        created_at = time::now(),
        metadata = $metadata
    ;
    """
    
    result = await execute(query, {
        "weight": weight,
        "confidence": confidence,
        "metadata": metadata or {},
    })
    
    return result[0] if result else {}


async def link_concepts(
    from_concept_id: str,
    to_concept_id: str,
    edge_type: EdgeType = EdgeType.RELATES_TO,
    weight: float = 1.0,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Crea una relación entre dos conceptos.
    
    Args:
        from_concept_id: ID del concepto origen
        to_concept_id: ID del concepto destino
        edge_type: Tipo de relación
        weight: Peso de la relación
        confidence: Confianza
        metadata: Metadatos
        
    Returns:
        Relación creada
    """
    return await create_edge(
        from_table="concept",
        from_id=from_concept_id,
        to_table="concept",
        to_id=to_concept_id,
        edge_type=edge_type,
        weight=weight,
        confidence=confidence,
        metadata=metadata,
    )


async def attach_evidence(
    concept_id: str,
    chunk_id: str,
    weight: float = 1.0,
    confidence: float = 0.8,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Enlaza un chunk como evidencia de un concepto.
    
    Args:
        concept_id: ID del concepto
        chunk_id: ID del chunk de evidencia
        weight: Peso de la evidencia
        confidence: Confianza
        metadata: Metadatos
        
    Returns:
        Relación creada
    """
    return await create_edge(
        from_table="concept",
        from_id=concept_id,
        to_table="chunk",
        to_id=chunk_id,
        edge_type=EdgeType.EVIDENCED_BY,
        weight=weight,
        confidence=confidence,
        metadata=metadata,
    )


async def link_chunk_to_source(
    chunk_id: str,
    source_id: str,
) -> Dict[str, Any]:
    """
    Enlaza un chunk a su fuente.
    
    Args:
        chunk_id: ID del chunk
        source_id: ID de la fuente
        
    Returns:
        Relación creada
    """
    return await create_edge(
        from_table="chunk",
        from_id=chunk_id,
        to_table="source",
        to_id=source_id,
        edge_type=EdgeType.BELONGS_TO,
        weight=1.0,
        confidence=1.0,
    )


async def link_parent_child_chunks(
    parent_chunk_id: str,
    child_chunk_id: str,
) -> Dict[str, Any]:
    """
    Enlaza un chunk padre con su hijo.
    
    Args:
        parent_chunk_id: ID del chunk padre
        child_chunk_id: ID del chunk hijo
        
    Returns:
        Relación creada
    """
    return await create_edge(
        from_table="chunk",
        from_id=parent_chunk_id,
        to_table="chunk",
        to_id=child_chunk_id,
        edge_type=EdgeType.PARENT_OF,
        weight=1.0,
        confidence=1.0,
    )


async def add_concept_to_topic(
    concept_id: str,
    topic_id: str,
    order: int = 0,
) -> Dict[str, Any]:
    """
    Agrega un concepto a un tema.
    
    Args:
        concept_id: ID del concepto
        topic_id: ID del tema
        order: Orden del concepto en el tema
        
    Returns:
        Relación creada
    """
    return await create_edge(
        from_table="concept",
        from_id=concept_id,
        to_table="topic",
        to_id=topic_id,
        edge_type=EdgeType.BELONGS_TO,
        weight=1.0,
        confidence=1.0,
        metadata={"order": order},
    )


# ==========================================
# Actualización de Nodos y Edges
# ==========================================

async def update_node(
    table: str,
    node_id: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Actualiza un nodo existente.
    
    Args:
        table: Nombre de la tabla
        node_id: ID del nodo
        updates: Campos a actualizar
        
    Returns:
        Nodo actualizado
    """
    updates["updated_at"] = datetime.utcnow().isoformat()
    
    db = await get_db()
    return await db.update(table, node_id, updates)


async def update_concept_embedding(
    concept_id: str,
    embedding: List[float],
) -> Dict[str, Any]:
    """
    Actualiza el embedding de un concepto.
    
    Args:
        concept_id: ID del concepto
        embedding: Nuevo vector de embedding
        
    Returns:
        Concepto actualizado
    """
    return await update_node("concept", concept_id, {"embedding": embedding})


async def update_chunk_embedding(
    chunk_id: str,
    embedding: List[float],
) -> Dict[str, Any]:
    """
    Actualiza el embedding de un chunk.
    
    Args:
        chunk_id: ID del chunk
        embedding: Nuevo vector de embedding
        
    Returns:
        Chunk actualizado
    """
    return await update_node("chunk", chunk_id, {"embedding": embedding})


async def update_edge_weight(
    from_table: str,
    from_id: str,
    edge_type: EdgeType,
    to_table: str,
    to_id: str,
    delta_weight: float,
) -> Dict[str, Any]:
    """
    Actualiza el peso de una relación.
    
    Args:
        from_table: Tabla origen
        from_id: ID origen
        edge_type: Tipo de relación
        to_table: Tabla destino
        to_id: ID destino
        delta_weight: Cambio en el peso (puede ser negativo)
        
    Returns:
        Relación actualizada
    """
    query = f"""
    UPDATE {edge_type.value} 
    SET weight = weight + $delta
    WHERE in = type::thing($from_table, $from_id) 
      AND out = type::thing($to_table, $to_id)
    ;
    """
    
    result = await execute(query, {
        "delta": delta_weight,
        "from_table": from_table,
        "from_id": from_id,
        "to_table": to_table,
        "to_id": to_id,
    })
    
    return result[0] if result else {}


# ==========================================
# Eliminación
# ==========================================

async def delete_node(table: str, node_id: str) -> bool:
    """
    Elimina un nodo y sus relaciones.
    
    Args:
        table: Nombre de la tabla
        node_id: ID del nodo
        
    Returns:
        True si se eliminó correctamente
    """
    db = await get_db()
    
    # Eliminar relaciones entrantes y salientes
    for edge_type in EdgeType:
        await execute(f"""
        DELETE {edge_type.value} 
        WHERE in = type::thing($table, $id) 
           OR out = type::thing($table, $id)
        ;
        """, {"table": table, "id": node_id})
    
    # Eliminar el nodo
    return await db.delete(table, node_id)


async def delete_edge(
    from_table: str,
    from_id: str,
    edge_type: EdgeType,
    to_table: str,
    to_id: str,
) -> bool:
    """
    Elimina una relación específica.
    
    Args:
        from_table: Tabla origen
        from_id: ID origen
        edge_type: Tipo de relación
        to_table: Tabla destino
        to_id: ID destino
        
    Returns:
        True si se eliminó
    """
    query = f"""
    DELETE {edge_type.value} 
    WHERE in = type::thing($from_table, $from_id) 
      AND out = type::thing($to_table, $to_id)
    ;
    """
    
    await execute(query, {
        "from_table": from_table,
        "from_id": from_id,
        "to_table": to_table,
        "to_id": to_id,
    })
    
    return True


# ==========================================
# Utilidades de Batch
# ==========================================

async def batch_create_concepts(
    concepts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Crea múltiples conceptos en batch.
    
    Args:
        concepts: Lista de diccionarios con datos de conceptos
        
    Returns:
        Lista de conceptos creados
    """
    results = []
    
    for concept_data in concepts:
        result = await create_concept_node(**concept_data)
        results.append(result)
    
    return results


async def batch_create_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Crea múltiples chunks en batch.
    
    Args:
        chunks: Lista de diccionarios con datos de chunks
        
    Returns:
        Lista de chunks creados
    """
    results = []
    
    for chunk_data in chunks:
        result = await create_chunk_node(**chunk_data)
        results.append(result)
    
    return results


async def batch_link_concepts(
    links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Crea múltiples relaciones entre conceptos en batch.
    
    Args:
        links: Lista de diccionarios con:
            - from_id: ID origen
            - to_id: ID destino
            - edge_type: Tipo de relación (opcional)
            - weight: Peso (opcional)
            
    Returns:
        Lista de relaciones creadas
    """
    results = []
    
    for link in links:
        result = await link_concepts(
            from_concept_id=link["from_id"],
            to_concept_id=link["to_id"],
            edge_type=link.get("edge_type", EdgeType.RELATES_TO),
            weight=link.get("weight", 1.0),
            confidence=link.get("confidence", 0.8),
        )
        results.append(result)
    
    return results
