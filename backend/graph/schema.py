"""
schema.py
Define tipos de nodos y relaciones para el grafo de conocimiento educativo.
(Concept, Topic, Course, Chunk, Source, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ==========================================
# Tipos de Nodos
# ==========================================

class NodeType(str, Enum):
    """Tipos de nodos soportados en el grafo."""
    
    # Contenido educativo
    CONCEPT = "concept"          # Concepto educativo
    TOPIC = "topic"              # Tema/Unidad
    COURSE = "course"            # Curso completo
    
    # Contenido fuente
    SOURCE = "source"            # Documento fuente
    CHUNK = "chunk"              # Fragmento de texto
    
    # Metadatos
    DEFINITION = "definition"    # Definición de concepto
    SUMMARY = "summary"          # Resumen
    EXAMPLE = "example"          # Ejemplo
    
    # Usuario
    STUDENT = "student"          # Perfil de estudiante
    SESSION = "session"          # Sesión de estudio


class EdgeType(str, Enum):
    """Tipos de relaciones soportadas en el grafo."""
    
    # Relaciones jerárquicas
    BELONGS_TO = "belongs_to"        # Concepto pertenece a tema
    CONTAINS = "contains"            # Tema contiene conceptos
    PARENT_OF = "parent_of"          # Chunk padre de child
    
    # Relaciones semánticas
    RELATES_TO = "relates_to"        # Relación genérica
    PREREQUISITE = "prerequisite"    # A es prerequisito de B
    SIMILAR_TO = "similar_to"        # Similitud semántica
    CONTRASTS = "contrasts"          # Contraste/oposición
    EXEMPLIFIES = "exemplifies"      # Ejemplo de concepto
    
    # Relaciones de evidencia
    EVIDENCED_BY = "evidenced_by"    # Concepto evidenciado por chunk
    DEFINES = "defines"              # Definición de concepto
    SUMMARIZES = "summarizes"        # Resumen de contenido
    
    # Relaciones de usuario
    STUDIED = "studied"              # Estudiante estudió concepto
    CONFUSED_BY = "confused_by"      # Estudiante confundido por
    MASTERED = "mastered"            # Estudiante domina concepto


# ==========================================
# Schemas de Nodos
# ==========================================

@dataclass
class BaseNode:
    """Schema base para todos los nodos."""
    
    id: Optional[str] = None
    type: NodeType = NodeType.CONCEPT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el nodo a diccionario para DB."""
        return {
            "id": self.id,
            "type": self.type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ConceptNode(BaseNode):
    """Schema para nodos de concepto."""
    
    type: NodeType = NodeType.CONCEPT
    name: str = ""
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    difficulty_level: int = 1  # 1-5
    importance_score: float = 0.5  # 0-1
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "name": self.name,
            "description": self.description,
            "aliases": self.aliases,
            "difficulty_level": self.difficulty_level,
            "importance_score": self.importance_score,
            "embedding": self.embedding,
        })
        return base


@dataclass
class TopicNode(BaseNode):
    """Schema para nodos de tema."""
    
    type: NodeType = NodeType.TOPIC
    name: str = ""
    description: str = ""
    learning_objectives: List[str] = field(default_factory=list)
    order_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "name": self.name,
            "description": self.description,
            "learning_objectives": self.learning_objectives,
            "order_index": self.order_index,
        })
        return base


@dataclass
class ChunkNode(BaseNode):
    """Schema para nodos de chunk (fragmento de texto)."""
    
    type: NodeType = NodeType.CHUNK
    content: str = ""
    source_id: str = ""
    chunk_index: int = 0
    parent_chunk_id: Optional[str] = None
    token_count: int = 0
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "content": self.content,
            "source_id": self.source_id,
            "chunk_index": self.chunk_index,
            "parent_chunk_id": self.parent_chunk_id,
            "token_count": self.token_count,
            "embedding": self.embedding,
        })
        return base


@dataclass
class SourceNode(BaseNode):
    """Schema para nodos de fuente."""
    
    type: NodeType = NodeType.SOURCE
    title: str = ""
    content_type: str = ""  # pdf, url, audio, etc.
    url: Optional[str] = None
    file_path: Optional[str] = None
    author: Optional[str] = None
    total_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "title": self.title,
            "content_type": self.content_type,
            "url": self.url,
            "file_path": self.file_path,
            "author": self.author,
            "total_chunks": self.total_chunks,
        })
        return base


@dataclass
class DefinitionNode(BaseNode):
    """Schema para nodos de definición."""
    
    type: NodeType = NodeType.DEFINITION
    concept_id: str = ""
    text: str = ""
    source_chunk_id: Optional[str] = None
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({
            "concept_id": self.concept_id,
            "text": self.text,
            "source_chunk_id": self.source_chunk_id,
            "confidence": self.confidence,
        })
        return base


# ==========================================
# Schema de Edges
# ==========================================

@dataclass
class Edge:
    """Schema para relaciones entre nodos."""
    
    id: Optional[str] = None
    type: EdgeType = EdgeType.RELATES_TO
    from_id: str = ""
    to_id: str = ""
    weight: float = 1.0  # Peso/fuerza de la relación
    confidence: float = 0.8  # Confianza en la relación
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "weight": self.weight,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


# ==========================================
# Definiciones del Schema
# ==========================================

# Nodos válidos por tipo
NODE_SCHEMAS: Dict[NodeType, type] = {
    NodeType.CONCEPT: ConceptNode,
    NodeType.TOPIC: TopicNode,
    NodeType.CHUNK: ChunkNode,
    NodeType.SOURCE: SourceNode,
    NodeType.DEFINITION: DefinitionNode,
}


# Relaciones válidas entre tipos de nodos
VALID_EDGES: Dict[EdgeType, List[tuple]] = {
    # (from_type, to_type)
    EdgeType.BELONGS_TO: [
        (NodeType.CONCEPT, NodeType.TOPIC),
        (NodeType.TOPIC, NodeType.COURSE),
        (NodeType.CHUNK, NodeType.SOURCE),
    ],
    EdgeType.CONTAINS: [
        (NodeType.TOPIC, NodeType.CONCEPT),
        (NodeType.COURSE, NodeType.TOPIC),
        (NodeType.SOURCE, NodeType.CHUNK),
    ],
    EdgeType.PARENT_OF: [
        (NodeType.CHUNK, NodeType.CHUNK),
    ],
    EdgeType.RELATES_TO: [
        (NodeType.CONCEPT, NodeType.CONCEPT),
        (NodeType.TOPIC, NodeType.TOPIC),
    ],
    EdgeType.PREREQUISITE: [
        (NodeType.CONCEPT, NodeType.CONCEPT),
        (NodeType.TOPIC, NodeType.TOPIC),
    ],
    EdgeType.SIMILAR_TO: [
        (NodeType.CONCEPT, NodeType.CONCEPT),
        (NodeType.CHUNK, NodeType.CHUNK),
    ],
    EdgeType.EVIDENCED_BY: [
        (NodeType.CONCEPT, NodeType.CHUNK),
    ],
    EdgeType.DEFINES: [
        (NodeType.DEFINITION, NodeType.CONCEPT),
    ],
    EdgeType.EXEMPLIFIES: [
        (NodeType.EXAMPLE, NodeType.CONCEPT),
    ],
}


def define_nodes() -> Dict[NodeType, Dict[str, Any]]:
    """
    Define los tipos de nodos y sus campos requeridos.
    
    Returns:
        Diccionario con definiciones de nodos
    """
    return {
        NodeType.CONCEPT: {
            "required": ["name"],
            "optional": ["description", "aliases", "difficulty_level", "importance_score", "embedding"],
            "searchable": ["name", "description", "aliases"],
        },
        NodeType.TOPIC: {
            "required": ["name"],
            "optional": ["description", "learning_objectives", "order_index"],
            "searchable": ["name", "description"],
        },
        NodeType.CHUNK: {
            "required": ["content", "source_id"],
            "optional": ["chunk_index", "parent_chunk_id", "token_count", "embedding"],
            "searchable": ["content"],
        },
        NodeType.SOURCE: {
            "required": ["title", "content_type"],
            "optional": ["url", "file_path", "author", "total_chunks"],
            "searchable": ["title", "author"],
        },
        NodeType.DEFINITION: {
            "required": ["concept_id", "text"],
            "optional": ["source_chunk_id", "confidence"],
            "searchable": ["text"],
        },
    }


def define_edges() -> Dict[EdgeType, Dict[str, Any]]:
    """
    Define los tipos de relaciones y sus restricciones.
    
    Returns:
        Diccionario con definiciones de edges
    """
    return {
        edge_type: {
            "valid_pairs": pairs,
            "bidirectional": edge_type in [EdgeType.RELATES_TO, EdgeType.SIMILAR_TO],
            "weighted": True,
        }
        for edge_type, pairs in VALID_EDGES.items()
    }


def validate_node(node: BaseNode) -> tuple[bool, Optional[str]]:
    """
    Valida que un nodo cumpla con el schema.
    
    Args:
        node: Nodo a validar
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    definitions = define_nodes()
    
    if node.type not in definitions:
        return False, f"Tipo de nodo no soportado: {node.type}"
    
    schema = definitions[node.type]
    node_dict = node.to_dict()
    
    # Verificar campos requeridos
    for field in schema["required"]:
        if field not in node_dict or not node_dict[field]:
            return False, f"Campo requerido faltante: {field}"
    
    return True, None


def validate_edge(
    edge: Edge,
    from_node_type: NodeType,
    to_node_type: NodeType
) -> tuple[bool, Optional[str]]:
    """
    Valida que una relación sea válida entre dos tipos de nodos.
    
    Args:
        edge: Edge a validar
        from_node_type: Tipo del nodo origen
        to_node_type: Tipo del nodo destino
        
    Returns:
        Tupla (es_válido, mensaje_error)
    """
    if edge.type not in VALID_EDGES:
        return False, f"Tipo de edge no soportado: {edge.type}"
    
    valid_pairs = VALID_EDGES[edge.type]
    pair = (from_node_type, to_node_type)
    
    if pair not in valid_pairs:
        return False, f"Relación {edge.type} no válida entre {from_node_type} y {to_node_type}"
    
    # Validar pesos
    if not 0 <= edge.weight <= 10:
        return False, f"Peso fuera de rango: {edge.weight}"
    
    if not 0 <= edge.confidence <= 1:
        return False, f"Confianza fuera de rango: {edge.confidence}"
    
    return True, None


def validate_graph(
    nodes: List[BaseNode],
    edges: List[Edge]
) -> tuple[bool, List[str]]:
    """
    Valida la consistencia de un grafo completo.
    
    Args:
        nodes: Lista de nodos
        edges: Lista de edges
        
    Returns:
        Tupla (es_válido, lista_errores)
    """
    errors = []
    node_ids: Set[str] = set()
    node_types: Dict[str, NodeType] = {}
    
    # Validar nodos
    for node in nodes:
        is_valid, error = validate_node(node)
        if not is_valid:
            errors.append(f"Nodo {node.id}: {error}")
        
        if node.id:
            if node.id in node_ids:
                errors.append(f"ID duplicado: {node.id}")
            node_ids.add(node.id)
            node_types[node.id] = node.type
    
    # Validar edges
    for edge in edges:
        # Verificar que los nodos existan
        if edge.from_id not in node_ids:
            errors.append(f"Edge {edge.id}: nodo origen no existe: {edge.from_id}")
            continue
        
        if edge.to_id not in node_ids:
            errors.append(f"Edge {edge.id}: nodo destino no existe: {edge.to_id}")
            continue
        
        # Validar tipos
        from_type = node_types[edge.from_id]
        to_type = node_types[edge.to_id]
        
        is_valid, error = validate_edge(edge, from_type, to_type)
        if not is_valid:
            errors.append(f"Edge {edge.id}: {error}")
    
    return len(errors) == 0, errors


# ==========================================
# Queries SurrealDB para el Schema
# ==========================================

def get_schema_queries() -> List[str]:
    """
    Genera queries SurrealQL para crear el schema en SurrealDB.
    
    Returns:
        Lista de queries para ejecutar
    """
    queries = []
    
    # Definir tablas de nodos
    queries.extend([
        # Conceptos
        """
        DEFINE TABLE concept SCHEMAFULL;
        DEFINE FIELD name ON concept TYPE string;
        DEFINE FIELD description ON concept TYPE option<string>;
        DEFINE FIELD aliases ON concept TYPE array;
        DEFINE FIELD difficulty_level ON concept TYPE int DEFAULT 1;
        DEFINE FIELD importance_score ON concept TYPE float DEFAULT 0.5;
        DEFINE FIELD embedding ON concept TYPE option<array>;
        DEFINE FIELD created_at ON concept TYPE datetime DEFAULT time::now();
        DEFINE FIELD updated_at ON concept TYPE datetime DEFAULT time::now();
        DEFINE FIELD metadata ON concept TYPE object DEFAULT {};
        DEFINE INDEX concept_name ON concept FIELDS name;
        """,
        
        # Topics
        """
        DEFINE TABLE topic SCHEMAFULL;
        DEFINE FIELD name ON topic TYPE string;
        DEFINE FIELD description ON topic TYPE option<string>;
        DEFINE FIELD learning_objectives ON topic TYPE array DEFAULT [];
        DEFINE FIELD order_index ON topic TYPE int DEFAULT 0;
        DEFINE FIELD created_at ON topic TYPE datetime DEFAULT time::now();
        DEFINE FIELD updated_at ON topic TYPE datetime DEFAULT time::now();
        DEFINE FIELD metadata ON topic TYPE object DEFAULT {};
        DEFINE INDEX topic_name ON topic FIELDS name;
        """,
        
        # Chunks
        """
        DEFINE TABLE chunk SCHEMAFULL;
        DEFINE FIELD content ON chunk TYPE string;
        DEFINE FIELD source_id ON chunk TYPE string;
        DEFINE FIELD chunk_index ON chunk TYPE int DEFAULT 0;
        DEFINE FIELD parent_chunk_id ON chunk TYPE option<string>;
        DEFINE FIELD token_count ON chunk TYPE int DEFAULT 0;
        DEFINE FIELD embedding ON chunk TYPE option<array>;
        DEFINE FIELD created_at ON chunk TYPE datetime DEFAULT time::now();
        DEFINE FIELD metadata ON chunk TYPE object DEFAULT {};
        DEFINE INDEX chunk_source ON chunk FIELDS source_id;
        """,
        
        # Sources
        """
        DEFINE TABLE source SCHEMAFULL;
        DEFINE FIELD title ON source TYPE string;
        DEFINE FIELD content_type ON source TYPE string;
        DEFINE FIELD url ON source TYPE option<string>;
        DEFINE FIELD file_path ON source TYPE option<string>;
        DEFINE FIELD author ON source TYPE option<string>;
        DEFINE FIELD total_chunks ON source TYPE int DEFAULT 0;
        DEFINE FIELD created_at ON source TYPE datetime DEFAULT time::now();
        DEFINE FIELD metadata ON source TYPE object DEFAULT {};
        DEFINE INDEX source_title ON source FIELDS title;
        """,
        
        # Definitions
        """
        DEFINE TABLE definition SCHEMAFULL;
        DEFINE FIELD concept_id ON definition TYPE string;
        DEFINE FIELD text ON definition TYPE string;
        DEFINE FIELD source_chunk_id ON definition TYPE option<string>;
        DEFINE FIELD confidence ON definition TYPE float DEFAULT 0.8;
        DEFINE FIELD created_at ON definition TYPE datetime DEFAULT time::now();
        DEFINE INDEX def_concept ON definition FIELDS concept_id;
        """,
    ])
    
    # Definir tablas de edges (relaciones)
    for edge_type in EdgeType:
        queries.append(f"""
        DEFINE TABLE {edge_type.value} SCHEMAFULL;
        DEFINE FIELD in ON {edge_type.value} TYPE record;
        DEFINE FIELD out ON {edge_type.value} TYPE record;
        DEFINE FIELD weight ON {edge_type.value} TYPE float DEFAULT 1.0;
        DEFINE FIELD confidence ON {edge_type.value} TYPE float DEFAULT 0.8;
        DEFINE FIELD created_at ON {edge_type.value} TYPE datetime DEFAULT time::now();
        DEFINE FIELD metadata ON {edge_type.value} TYPE object DEFAULT {{}};
        """)
    
    return queries
