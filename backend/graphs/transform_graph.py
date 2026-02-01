"""
transform_graph.py
Aplica prompts de extracción.
(conceptos, definiciones, resúmenes)
Transforma contenido raw en conocimiento estructurado.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from backend.graph.builders import (
    attach_evidence,
    create_concept_node,
    create_definition_node,
    link_concepts,
)
from backend.graph.schema import EdgeType
from backend.ingestion.chunking import Chunk
from backend.models.llm import generate
from backend.models.embeddings import embed_text



# ==========================================
# Prompts de Extracción
# ==========================================

CONCEPT_EXTRACTION_PROMPT = """Analiza el siguiente texto educativo y extrae los conceptos principales.

TEXTO:
{text}

Extrae los conceptos más importantes del texto. Para cada concepto proporciona:
1. name: Nombre del concepto (corto y preciso)
2. description: Descripción breve (1-2 oraciones)
3. difficulty: Nivel de dificultad (1-5, donde 1 es básico y 5 es avanzado)
4. importance: Importancia en el contexto (0.0 a 1.0)

Responde SOLO con un JSON array válido, sin explicaciones adicionales:
[
  {{"name": "...", "description": "...", "difficulty": N, "importance": 0.X}},
  ...
]

Máximo 10 conceptos. Solo los más relevantes."""

ENTITY_RESOLUTION_PROMPT = """Analiza esta lista de conceptos y encuentra duplicados o conceptos muy similares.

CONCEPTOS:
{concepts}

Identifica grupos de conceptos que se refieren a lo mismo y deberían unificarse.
Responde con un JSON que mapea conceptos duplicados a su forma canónica:
{{
  "duplicates": [
    {{"canonical": "nombre canónico", "aliases": ["alias1", "alias2"]}}
  ],
  "unique": ["concepto único 1", "concepto único 2"]
}}"""

SUMMARY_PROMPT = """Genera un resumen pedagógico del siguiente contenido educativo.

CONTENIDO:
{text}

El resumen debe:
1. Capturar las ideas principales
2. Estar escrito para facilitar el aprendizaje
3. Mantener un tono claro y accesible
4. Tener entre 100-200 palabras

RESUMEN:"""

DEFINITION_EXTRACTION_PROMPT = """Extrae definiciones precisas de los siguientes conceptos basándote en el texto.

CONCEPTOS A DEFINIR:
{concepts}

TEXTO FUENTE:
{text}

Para cada concepto, extrae o genera una definición precisa basada en el texto.
Responde SOLO con un JSON array:
[
  {{"concept": "nombre", "definition": "definición clara y precisa", "confidence": 0.X}}
]

Si un concepto no tiene definición clara en el texto, omítelo."""

RELATIONSHIP_EXTRACTION_PROMPT = """Analiza las relaciones entre estos conceptos basándote en el texto.

CONCEPTOS:
{concepts}

TEXTO:
{text}

Identifica relaciones entre los conceptos. Tipos de relaciones válidas:
- prerequisite: A es prerequisito de B
- relates_to: A se relaciona con B
- similar_to: A es similar a B
- contrasts: A contrasta con B

Responde SOLO con un JSON array:
[
  {{"from": "concepto A", "to": "concepto B", "type": "tipo_relacion", "confidence": 0.X}}
]

Máximo 15 relaciones más relevantes."""


# ==========================================
# Estado de Transformación
# ==========================================

@dataclass
class TransformState:
    """Estado del proceso de transformación."""
    
    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: Optional[str] = None
    content: str = ""
    chunks: List[Chunk] = field(default_factory=list)
    
    # Resultados de extracción
    concepts: List[Dict[str, Any]] = field(default_factory=list)
    definitions: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    
    # IDs creados
    concept_ids: List[str] = field(default_factory=list)
    definition_ids: List[str] = field(default_factory=list)
    
    # Metadatos
    created_at: datetime = field(default_factory=datetime.utcnow)
    errors: List[str] = field(default_factory=list)


# ==========================================
# Nodos de Transformación
# ==========================================

def parse_json_response(response: str) -> Any:
    """
    Parsea respuesta JSON del LLM con tolerancia a errores.
    
    Args:
        response: Respuesta del LLM
        
    Returns:
        Objeto parseado o lista vacía si falla
    """
    # Limpiar respuesta
    response = response.strip()
    
    # Intentar extraer JSON de la respuesta
    json_match = re.search(r'[\[{].*[\]}]', response, re.DOTALL)
    if json_match:
        response = json_match.group()
    
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Intentar arreglar JSON común
        response = response.replace("'", '"')
        response = re.sub(r',\s*}', '}', response)
        response = re.sub(r',\s*]', ']', response)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return []


async def extract_concepts(state: TransformState) -> TransformState:
    """
    Nodo: Extrae conceptos del contenido.
    
    Args:
        state: Estado actual
        
    Returns:
        Estado con conceptos extraídos
    """
    if not state.content:
        return state
    
    try:
        # Truncar contenido si es muy largo
        content_for_llm = state.content[:8000] if len(state.content) > 8000 else state.content
        
        prompt = CONCEPT_EXTRACTION_PROMPT.format(text=content_for_llm)
        
        response = await generate(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de contenido educativo. Extraes conceptos clave de forma precisa.",
            temperature=0.3,
        )
        
        concepts = parse_json_response(response)
        
        if isinstance(concepts, list):
            state.concepts = concepts
        
    except Exception as e:
        state.errors.append(f"Error extrayendo conceptos: {e}")
    
    return state


async def resolve_entities(state: TransformState) -> TransformState:
    """
    Nodo: Deduplica y normaliza conceptos.
    
    Args:
        state: Estado con conceptos
        
    Returns:
        Estado con conceptos normalizados
    """
    if not state.concepts or len(state.concepts) < 2:
        return state
    
    try:
        concept_names = [c.get("name", "") for c in state.concepts]
        
        prompt = ENTITY_RESOLUTION_PROMPT.format(
            concepts=json.dumps(concept_names, ensure_ascii=False)
        )
        
        response = await generate(
            prompt=prompt,
            system_prompt="Eres un experto en desambiguación de entidades.",
            temperature=0.2,
        )
        
        resolution = parse_json_response(response)
        
        if isinstance(resolution, dict) and "duplicates" in resolution:
            # Aplicar resolución
            canonical_map = {}
            for dup in resolution.get("duplicates", []):
                canonical = dup.get("canonical", "")
                for alias in dup.get("aliases", []):
                    canonical_map[alias.lower()] = canonical
            
            # Actualizar conceptos
            seen = set()
            unique_concepts = []
            
            for concept in state.concepts:
                name = concept.get("name", "")
                # Resolver a forma canónica
                canonical = canonical_map.get(name.lower(), name)
                
                if canonical.lower() not in seen:
                    concept["name"] = canonical
                    if name.lower() != canonical.lower():
                        concept.setdefault("aliases", []).append(name)
                    unique_concepts.append(concept)
                    seen.add(canonical.lower())
            
            state.concepts = unique_concepts
        
    except Exception as e:
        state.errors.append(f"Error resolviendo entidades: {e}")
    
    return state


async def generate_summary(state: TransformState) -> TransformState:
    """
    Nodo: Genera resumen pedagógico.
    
    Args:
        state: Estado actual
        
    Returns:
        Estado con resumen
    """
    if not state.content:
        return state
    
    try:
        content_for_llm = state.content[:6000] if len(state.content) > 6000 else state.content
        
        prompt = SUMMARY_PROMPT.format(text=content_for_llm)
        
        response = await generate(
            prompt=prompt,
            system_prompt="Eres un experto en pedagogía que crea resúmenes claros y efectivos.",
            temperature=0.5,
        )
        
        state.summary = response.strip()
        
    except Exception as e:
        state.errors.append(f"Error generando resumen: {e}")
    
    return state


async def extract_definitions(state: TransformState) -> TransformState:
    """
    Nodo: Extrae definiciones de conceptos.
    
    Args:
        state: Estado con conceptos
        
    Returns:
        Estado con definiciones
    """
    if not state.concepts or not state.content:
        return state
    
    try:
        concept_names = [c.get("name", "") for c in state.concepts[:10]]  # Limitar
        content_for_llm = state.content[:6000]
        
        prompt = DEFINITION_EXTRACTION_PROMPT.format(
            concepts=json.dumps(concept_names, ensure_ascii=False),
            text=content_for_llm
        )
        
        response = await generate(
            prompt=prompt,
            system_prompt="Eres un experto en definir conceptos de forma clara y precisa.",
            temperature=0.3,
        )
        
        definitions = parse_json_response(response)
        
        if isinstance(definitions, list):
            state.definitions = definitions
        
    except Exception as e:
        state.errors.append(f"Error extrayendo definiciones: {e}")
    
    return state


async def extract_relationships(state: TransformState) -> TransformState:
    """
    Nodo: Extrae relaciones entre conceptos.
    
    Args:
        state: Estado con conceptos
        
    Returns:
        Estado con relaciones
    """
    if not state.concepts or len(state.concepts) < 2:
        return state
    
    try:
        concept_names = [c.get("name", "") for c in state.concepts]
        content_for_llm = state.content[:4000] if state.content else ""
        
        prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(
            concepts=json.dumps(concept_names, ensure_ascii=False),
            text=content_for_llm
        )
        
        response = await generate(
            prompt=prompt,
            system_prompt="Eres un experto en análisis de relaciones conceptuales.",
            temperature=0.3,
        )
        
        relationships = parse_json_response(response)
        
        if isinstance(relationships, list):
            # Validar tipos de relación
            valid_types = {"prerequisite", "relates_to", "similar_to", "contrasts"}
            state.relationships = [
                r for r in relationships
                if r.get("type") in valid_types
            ]
        
    except Exception as e:
        state.errors.append(f"Error extrayendo relaciones: {e}")
    
    return state


async def save_concepts(state: TransformState) -> TransformState:
    """
    Nodo: Persiste conceptos en la base de datos.
    
    Args:
        state: Estado con conceptos
        
    Returns:
        Estado con IDs de conceptos creados
    """
    if not state.concepts:
        return state
    
    try:
        for concept in state.concepts:
            # Generar embedding para el concepto
            concept_text = f"{concept.get('name', '')}. {concept.get('description', '')}"
            embedding = await embed_text(concept_text)
            
            result = await create_concept_node(
                name=concept.get("name", ""),
                description=concept.get("description", ""),
                aliases=concept.get("aliases", []),
                difficulty_level=concept.get("difficulty", 1),
                importance_score=concept.get("importance", 0.5),
                embedding=embedding,
                metadata={
                    "source_id": state.source_id,
                    "extraction_id": state.id,
                }
            )
            
            concept_id = result.get("id")
            if concept_id:
                state.concept_ids.append(concept_id)
                concept["_id"] = concept_id
        
    except Exception as e:
        state.errors.append(f"Error guardando conceptos: {e}")
    
    return state


async def save_definitions(state: TransformState) -> TransformState:
    """
    Nodo: Persiste definiciones en la base de datos.
    
    Args:
        state: Estado con definiciones
        
    Returns:
        Estado con IDs de definiciones creadas
    """
    if not state.definitions:
        return state
    
    try:
        # Mapear nombres a IDs
        concept_id_map = {
            c.get("name", "").lower(): c.get("_id")
            for c in state.concepts
            if c.get("_id")
        }
        
        for definition in state.definitions:
            concept_name = definition.get("concept", "").lower()
            concept_id = concept_id_map.get(concept_name)
            
            if not concept_id:
                continue
            
            result = await create_definition_node(
                concept_id=concept_id,
                text=definition.get("definition", ""),
                confidence=definition.get("confidence", 0.8),
                metadata={"source_id": state.source_id}
            )
            
            def_id = result.get("id")
            if def_id:
                state.definition_ids.append(def_id)
        
    except Exception as e:
        state.errors.append(f"Error guardando definiciones: {e}")
    
    return state


async def save_relationships(state: TransformState) -> TransformState:
    """
    Nodo: Persiste relaciones entre conceptos.
    
    Args:
        state: Estado con relaciones
        
    Returns:
        Estado actualizado
    """
    if not state.relationships:
        return state
    
    try:
        # Mapear nombres a IDs
        concept_id_map = {
            c.get("name", "").lower(): c.get("_id")
            for c in state.concepts
            if c.get("_id")
        }
        
        # Mapeo de tipos de relación
        edge_type_map = {
            "prerequisite": EdgeType.PREREQUISITE,
            "relates_to": EdgeType.RELATES_TO,
            "similar_to": EdgeType.SIMILAR_TO,
            "contrasts": EdgeType.CONTRASTS,
        }
        
        for rel in state.relationships:
            from_name = rel.get("from", "").lower()
            to_name = rel.get("to", "").lower()
            rel_type = rel.get("type", "relates_to")
            
            from_id = concept_id_map.get(from_name)
            to_id = concept_id_map.get(to_name)
            
            if not from_id or not to_id:
                continue
            
            edge_type = edge_type_map.get(rel_type, EdgeType.RELATES_TO)
            
            await link_concepts(
                from_concept_id=from_id,
                to_concept_id=to_id,
                edge_type=edge_type,
                confidence=rel.get("confidence", 0.7),
            )
        
    except Exception as e:
        state.errors.append(f"Error guardando relaciones: {e}")
    
    return state


async def link_evidence(state: TransformState) -> TransformState:
    """
    Nodo: Enlaza chunks como evidencia de conceptos.
    
    Args:
        state: Estado con conceptos y chunks
        
    Returns:
        Estado actualizado
    """
    if not state.concept_ids or not state.chunks:
        return state
    
    try:
        # Estrategia simple: enlazar cada concepto con chunks relevantes
        for concept in state.concepts:
            concept_id = concept.get("_id")
            if not concept_id:
                continue
            
            concept_name = concept.get("name", "").lower()
            
            # Buscar chunks que mencionan el concepto
            for chunk in state.chunks[:20]:  # Limitar
                if concept_name in chunk.content.lower():
                    await attach_evidence(
                        concept_id=concept_id,
                        chunk_id=chunk.id,
                        weight=0.8,
                        confidence=0.7,
                    )
        
    except Exception as e:
        state.errors.append(f"Error enlazando evidencia: {e}")
    
    return state


# ==========================================
# Orquestador Principal
# ==========================================

async def run_transform_graph(
    content: str,
    source_id: Optional[str] = None,
    chunks: Optional[List[Chunk]] = None,
    extract_concepts_flag: bool = True,
    extract_definitions_flag: bool = True,
    extract_relationships_flag: bool = True,
    generate_summary_flag: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta el pipeline de transformación completo.
    
    Extrae conocimiento estructurado del contenido:
    conceptos, definiciones, relaciones y resúmenes.
    
    Args:
        content: Contenido a transformar
        source_id: ID de la fuente
        chunks: Chunks del contenido
        extract_concepts_flag: Si extraer conceptos
        extract_definitions_flag: Si extraer definiciones
        extract_relationships_flag: Si extraer relaciones
        generate_summary_flag: Si generar resumen
        
    Returns:
        Resultado de las transformaciones
    """
    state = TransformState(
        content=content,
        source_id=source_id,
        chunks=chunks or [],
    )
    
    try:
        # Ejecutar extracciones en paralelo donde sea posible
        tasks = []
        
        if extract_concepts_flag:
            state = await extract_concepts(state)
            state = await resolve_entities(state)
        
        if generate_summary_flag:
            state = await generate_summary(state)
        
        # Estas dependen de conceptos
        if state.concepts:
            if extract_definitions_flag:
                state = await extract_definitions(state)
            
            if extract_relationships_flag:
                state = await extract_relationships(state)
        
        # Persistir en DB
        state = await save_concepts(state)
        state = await save_definitions(state)
        state = await save_relationships(state)
        state = await link_evidence(state)
        
    except Exception as e:
        state.errors.append(f"Error en pipeline: {e}")
    
    return {
        "id": state.id,
        "source_id": state.source_id,
        "concepts_extracted": len(state.concepts),
        "concepts_created": len(state.concept_ids),
        "definitions_created": len(state.definition_ids),
        "relationships_created": len(state.relationships),
        "summary": state.summary[:200] + "..." if len(state.summary) > 200 else state.summary,
        "errors": state.errors,
        "concept_ids": state.concept_ids,
    }


# ==========================================
# Transformaciones Individuales
# ==========================================

async def extract_concepts_only(content: str) -> List[Dict[str, Any]]:
    """
    Extrae solo conceptos sin persistir.
    
    Args:
        content: Contenido a analizar
        
    Returns:
        Lista de conceptos extraídos
    """
    state = TransformState(content=content)
    state = await extract_concepts(state)
    state = await resolve_entities(state)
    return state.concepts


async def generate_summary_only(content: str) -> str:
    """
    Genera solo resumen sin persistir.
    
    Args:
        content: Contenido a resumir
        
    Returns:
        Resumen generado
    """
    state = TransformState(content=content)
    state = await generate_summary(state)
    return state.summary


async def extract_relationships_only(
    content: str,
    concepts: List[str]
) -> List[Dict[str, Any]]:
    """
    Extrae solo relaciones sin persistir.
    
    Args:
        content: Contenido a analizar
        concepts: Lista de nombres de conceptos
        
    Returns:
        Lista de relaciones
    """
    state = TransformState(
        content=content,
        concepts=[{"name": c} for c in concepts]
    )
    state = await extract_relationships(state)
    return state.relationships
