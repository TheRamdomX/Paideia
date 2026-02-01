"""
traversal.py
Algoritmos de exploración del grafo para retrieval.
Implementa expansión de conceptos, ranking de paths y control de profundidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.db.surreal import execute, get_db
from backend.graph.schema import EdgeType, NodeType
from backend.settings import get_rag_config


# ==========================================
# Estructuras de Datos
# ==========================================

class TraversalStrategy(str, Enum):
    """Estrategias de traversal."""
    BFS = "bfs"  # Breadth-first
    DFS = "dfs"  # Depth-first
    WEIGHTED = "weighted"  # Por peso de aristas
    BIDIRECTIONAL = "bidirectional"  # Desde query y conceptos


@dataclass
class GraphNode:
    """Nodo del grafo para traversal."""
    
    id: str = ""
    type: NodeType = NodeType.CONCEPT
    name: str = ""
    content: str = ""
    score: float = 0.0
    depth: int = 0
    path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "content": self.content,
            "score": self.score,
            "depth": self.depth,
            "path": self.path,
            "metadata": self.metadata,
        }


@dataclass
class GraphPath:
    """Camino en el grafo."""
    
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    total_score: float = 0.0
    length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": self.edges,
            "total_score": self.total_score,
            "length": self.length,
        }


@dataclass
class TraversalResult:
    """Resultado de traversal."""
    
    start_node: str = ""
    nodes_visited: List[GraphNode] = field(default_factory=list)
    paths: List[GraphPath] = field(default_factory=list)
    chunks_found: List[Dict[str, Any]] = field(default_factory=list)
    total_nodes: int = 0
    max_depth_reached: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_node": self.start_node,
            "nodes_visited": [n.to_dict() for n in self.nodes_visited],
            "paths": [p.to_dict() for p in self.paths],
            "chunks_found": self.chunks_found,
            "total_nodes": self.total_nodes,
            "max_depth_reached": self.max_depth_reached,
        }


# ==========================================
# Configuración
# ==========================================

def get_traversal_config() -> Dict[str, Any]:
    """Obtiene configuración de traversal."""
    config = get_rag_config()
    return {
        "max_depth": config.get("graph_max_depth", 3),
        "max_nodes": config.get("graph_max_nodes", 50),
        "min_edge_weight": config.get("graph_min_edge_weight", 0.3),
        "decay_factor": config.get("graph_decay_factor", 0.8),
    }


# ==========================================
# Expansión de Conceptos
# ==========================================

async def expand_concepts(
    concept_ids: List[str],
    max_depth: int = 2,
    max_nodes: int = 30,
    edge_types: Optional[List[EdgeType]] = None,
    min_weight: float = 0.3,
) -> TraversalResult:
    """
    Expande conceptos explorando vecindad en el grafo.
    
    Args:
        concept_ids: IDs de conceptos iniciales
        max_depth: Profundidad máxima de exploración
        max_nodes: Número máximo de nodos a visitar
        edge_types: Tipos de aristas a seguir
        min_weight: Peso mínimo de arista
        
    Returns:
        TraversalResult con nodos alcanzados
    """
    if not concept_ids:
        return TraversalResult()
    
    config = get_traversal_config()
    max_depth = min(max_depth, config["max_depth"])
    max_nodes = min(max_nodes, config["max_nodes"])
    
    # Tipos de aristas por defecto
    if edge_types is None:
        edge_types = [
            EdgeType.RELATES_TO,
            EdgeType.PREREQUISITE,
            EdgeType.SIMILAR_TO,
        ]
    
    visited: Set[str] = set()
    result = TraversalResult(start_node=concept_ids[0] if concept_ids else "")
    
    # Cola de exploración: (node_id, depth, score, path)
    queue: List[Tuple[str, int, float, List[str]]] = [
        (cid, 0, 1.0, [cid]) for cid in concept_ids
    ]
    
    try:
        db = await get_db()
        
        while queue and len(visited) < max_nodes:
            # Ordenar por score (mayor primero)
            queue.sort(key=lambda x: x[2], reverse=True)
            
            node_id, depth, score, path = queue.pop(0)
            
            if node_id in visited:
                continue
            
            visited.add(node_id)
            
            # Obtener información del nodo
            node_info = await _get_node_info(db, node_id)
            
            if node_info:
                graph_node = GraphNode(
                    id=node_id,
                    type=NodeType(node_info.get("type", "concept")),
                    name=node_info.get("name", ""),
                    content=node_info.get("description", ""),
                    score=score,
                    depth=depth,
                    path=path,
                    metadata=node_info.get("metadata", {}),
                )
                result.nodes_visited.append(graph_node)
                result.max_depth_reached = max(result.max_depth_reached, depth)
            
            # Explorar vecinos si no alcanzamos profundidad máxima
            if depth < max_depth:
                neighbors = await _get_neighbors(
                    db, node_id, edge_types, min_weight
                )
                
                for neighbor_id, edge_type, weight in neighbors:
                    if neighbor_id not in visited:
                        # Calcular score decayendo con profundidad
                        new_score = score * weight * config["decay_factor"]
                        new_path = path + [neighbor_id]
                        
                        queue.append((neighbor_id, depth + 1, new_score, new_path))
        
        result.total_nodes = len(visited)
        
        # Obtener chunks asociados a los conceptos encontrados
        result.chunks_found = await _get_associated_chunks(
            db, list(visited)
        )
        
    except Exception as e:
        result.metadata = {"error": str(e)}
    
    return result


async def _get_node_info(db, node_id: str) -> Optional[Dict[str, Any]]:
    """Obtiene información de un nodo."""
    try:
        surql = """
            SELECT * FROM $id
        """
        result = await db.query(surql, {"id": node_id})
        
        if result and result[0].get("result"):
            return result[0]["result"][0]
        
        return None
        
    except Exception:
        return None


async def _get_neighbors(
    db,
    node_id: str,
    edge_types: List[EdgeType],
    min_weight: float,
) -> List[Tuple[str, str, float]]:
    """
    Obtiene vecinos de un nodo.
    
    Returns:
        Lista de tuplas (neighbor_id, edge_type, weight)
    """
    try:
        edge_type_values = [et.value for et in edge_types]
        
        # Buscar aristas salientes e entrantes
        surql = """
            SELECT 
                out as target,
                type,
                weight
            FROM edge
            WHERE in = $node_id 
            AND type IN $edge_types
            AND weight >= $min_weight
            
            UNION
            
            SELECT 
                in as target,
                type,
                weight
            FROM edge
            WHERE out = $node_id 
            AND type IN $edge_types
            AND weight >= $min_weight
        """
        
        result = await db.query(surql, {
            "node_id": node_id,
            "edge_types": edge_type_values,
            "min_weight": min_weight,
        })
        
        neighbors = []
        
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                target = row.get("target", "")
                edge_type = row.get("type", "")
                weight = float(row.get("weight", 0.5))
                
                if target and target != node_id:
                    neighbors.append((target, edge_type, weight))
        
        return neighbors
        
    except Exception:
        return []


async def _get_associated_chunks(
    db,
    concept_ids: List[str],
) -> List[Dict[str, Any]]:
    """Obtiene chunks asociados a conceptos."""
    if not concept_ids:
        return []
    
    try:
        surql = """
            SELECT 
                chunk.id,
                chunk.content,
                chunk.metadata,
                edge.weight as relevance
            FROM edge
            WHERE in IN $concept_ids
            AND type = 'evidence'
            FETCH chunk
        """
        
        result = await db.query(surql, {"concept_ids": concept_ids})
        
        chunks = []
        
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                if row.get("chunk"):
                    chunk = row["chunk"]
                    chunks.append({
                        "id": chunk.get("id", ""),
                        "content": chunk.get("content", ""),
                        "metadata": chunk.get("metadata", {}),
                        "relevance": row.get("relevance", 0.5),
                    })
        
        return chunks
        
    except Exception:
        return []


# ==========================================
# Ranking de Paths
# ==========================================

def rank_paths(
    paths: List[GraphPath],
    scoring_method: str = "weighted",
) -> List[GraphPath]:
    """
    Ordena caminos por relevancia.
    
    Args:
        paths: Lista de caminos
        scoring_method: Método de scoring (weighted, shortest, coverage)
        
    Returns:
        Paths ordenados por score
    """
    if not paths:
        return paths
    
    for path in paths:
        if scoring_method == "weighted":
            # Score basado en pesos de nodos
            path.total_score = sum(n.score for n in path.nodes) / len(path.nodes)
        elif scoring_method == "shortest":
            # Preferir caminos cortos
            path.total_score = 1.0 / (path.length + 1)
        elif scoring_method == "coverage":
            # Preferir caminos que cubren más tipos de nodos
            types = set(n.type for n in path.nodes)
            path.total_score = len(types) / 5  # Normalizado por tipos totales
    
    return sorted(paths, key=lambda p: p.total_score, reverse=True)


async def find_paths_between(
    source_id: str,
    target_id: str,
    max_length: int = 4,
) -> List[GraphPath]:
    """
    Encuentra caminos entre dos nodos.
    
    Args:
        source_id: Nodo origen
        target_id: Nodo destino
        max_length: Longitud máxima del camino
        
    Returns:
        Lista de caminos encontrados
    """
    try:
        db = await get_db()
        
        # Usar traversal de SurrealDB
        surql = f"""
            SELECT * FROM $source->edge->(concept WHERE id != $source)<-edge<-concept
            WHERE id = $target
            LIMIT 10
        """
        
        # Simplificación: usar BFS manual
        paths = []
        visited_paths: Set[str] = set()
        
        queue: List[Tuple[str, List[GraphNode], List[str]]] = [
            (source_id, [], [])
        ]
        
        while queue:
            current_id, current_nodes, current_edges = queue.pop(0)
            
            if len(current_nodes) > max_length:
                continue
            
            path_key = "->".join([n.id for n in current_nodes] + [current_id])
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            # Obtener info del nodo actual
            node_info = await _get_node_info(db, current_id)
            
            if node_info:
                current_node = GraphNode(
                    id=current_id,
                    name=node_info.get("name", ""),
                    content=node_info.get("description", ""),
                    depth=len(current_nodes),
                )
                new_nodes = current_nodes + [current_node]
                
                if current_id == target_id:
                    # Encontramos el destino
                    paths.append(GraphPath(
                        nodes=new_nodes,
                        edges=current_edges,
                        length=len(new_nodes),
                    ))
                else:
                    # Continuar explorando
                    neighbors = await _get_neighbors(
                        db, current_id, list(EdgeType), 0.1
                    )
                    
                    for neighbor_id, edge_type, _ in neighbors:
                        queue.append((
                            neighbor_id,
                            new_nodes,
                            current_edges + [edge_type]
                        ))
        
        return rank_paths(paths)
        
    except Exception:
        return []


# ==========================================
# Control de Profundidad
# ==========================================

def limit_depth(
    result: TraversalResult,
    max_depth: int,
) -> TraversalResult:
    """
    Limita resultados a una profundidad máxima.
    
    Args:
        result: Resultado de traversal
        max_depth: Profundidad máxima permitida
        
    Returns:
        Resultado filtrado
    """
    filtered_nodes = [
        node for node in result.nodes_visited
        if node.depth <= max_depth
    ]
    
    filtered_paths = [
        path for path in result.paths
        if path.length <= max_depth + 1
    ]
    
    return TraversalResult(
        start_node=result.start_node,
        nodes_visited=filtered_nodes,
        paths=filtered_paths,
        chunks_found=result.chunks_found,
        total_nodes=len(filtered_nodes),
        max_depth_reached=min(result.max_depth_reached, max_depth),
    )


def prune_by_score(
    result: TraversalResult,
    min_score: float = 0.1,
) -> TraversalResult:
    """
    Elimina nodos con score bajo.
    
    Args:
        result: Resultado de traversal
        min_score: Score mínimo
        
    Returns:
        Resultado filtrado
    """
    filtered_nodes = [
        node for node in result.nodes_visited
        if node.score >= min_score
    ]
    
    return TraversalResult(
        start_node=result.start_node,
        nodes_visited=filtered_nodes,
        paths=result.paths,
        chunks_found=result.chunks_found,
        total_nodes=len(filtered_nodes),
        max_depth_reached=result.max_depth_reached,
    )


# ==========================================
# Búsqueda de Conceptos
# ==========================================

async def find_concepts_by_query(
    query: str,
    limit: int = 5,
) -> List[GraphNode]:
    """
    Busca conceptos relevantes para una query.
    
    Args:
        query: Texto de búsqueda
        limit: Número máximo de resultados
        
    Returns:
        Lista de conceptos
    """
    try:
        db = await get_db()
        
        # Buscar por nombre y descripción
        surql = """
            SELECT id, name, description, metadata
            FROM concept
            WHERE string::lowercase(name) CONTAINS string::lowercase($query)
            OR string::lowercase(description) CONTAINS string::lowercase($query)
            LIMIT $limit
        """
        
        result = await db.query(surql, {"query": query, "limit": limit})
        
        concepts = []
        
        if result and result[0].get("result"):
            for row in result[0]["result"]:
                concepts.append(GraphNode(
                    id=str(row.get("id", "")),
                    type=NodeType.CONCEPT,
                    name=row.get("name", ""),
                    content=row.get("description", ""),
                    metadata=row.get("metadata", {}),
                ))
        
        return concepts
        
    except Exception:
        return []


async def get_related_concepts(
    concept_id: str,
    relation_types: Optional[List[EdgeType]] = None,
    limit: int = 10,
) -> List[Tuple[GraphNode, EdgeType, float]]:
    """
    Obtiene conceptos relacionados a uno dado.
    
    Args:
        concept_id: ID del concepto
        relation_types: Tipos de relación a buscar
        limit: Número máximo
        
    Returns:
        Lista de tuplas (concepto, tipo_relación, peso)
    """
    if relation_types is None:
        relation_types = list(EdgeType)
    
    try:
        db = await get_db()
        
        neighbors = await _get_neighbors(
            db, concept_id, relation_types, 0.0
        )
        
        related = []
        
        for neighbor_id, edge_type, weight in neighbors[:limit]:
            node_info = await _get_node_info(db, neighbor_id)
            
            if node_info:
                concept = GraphNode(
                    id=neighbor_id,
                    type=NodeType.CONCEPT,
                    name=node_info.get("name", ""),
                    content=node_info.get("description", ""),
                    score=weight,
                )
                
                related.append((
                    concept,
                    EdgeType(edge_type) if edge_type in EdgeType._value2member_map_ else EdgeType.RELATES_TO,
                    weight
                ))
        
        return related
        
    except Exception:
        return []


# ==========================================
# Subgrafo Contextual
# ==========================================

async def extract_subgraph(
    center_ids: List[str],
    radius: int = 2,
    include_chunks: bool = True,
) -> Dict[str, Any]:
    """
    Extrae un subgrafo centrado en nodos específicos.
    
    Args:
        center_ids: IDs de nodos centrales
        radius: Radio de expansión
        include_chunks: Si incluir chunks asociados
        
    Returns:
        Dict con nodos y aristas del subgrafo
    """
    result = await expand_concepts(
        concept_ids=center_ids,
        max_depth=radius,
        max_nodes=100,
    )
    
    # Construir estructura de subgrafo
    nodes = {}
    edges = []
    
    for node in result.nodes_visited:
        nodes[node.id] = node.to_dict()
        
        # Reconstruir aristas desde paths
        if len(node.path) > 1:
            for i in range(len(node.path) - 1):
                edge = {
                    "from": node.path[i],
                    "to": node.path[i + 1],
                    "type": "relates_to",
                }
                if edge not in edges:
                    edges.append(edge)
    
    subgraph = {
        "nodes": list(nodes.values()),
        "edges": edges,
        "center_ids": center_ids,
        "radius": radius,
    }
    
    if include_chunks:
        subgraph["chunks"] = result.chunks_found
    
    return subgraph
