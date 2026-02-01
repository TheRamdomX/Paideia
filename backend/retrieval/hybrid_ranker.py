"""
hybrid_ranker.py
Combina scores de grafo, BM25 y embeddings.
Implementa ranking híbrido para retrieval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.settings import get_rag_config


# ==========================================
# Enums y Configuración
# ==========================================

class RetrievalSource(str, Enum):
    """Fuentes de retrieval."""
    VECTOR = "vector"
    BM25 = "bm25"
    GRAPH = "graph"
    HYBRID = "hybrid"


@dataclass
class RankedResult:
    """Resultado con scores combinados."""
    
    id: str = ""
    content: str = ""
    final_score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    sources: List[RetrievalSource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "final_score": self.final_score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "graph_score": self.graph_score,
            "sources": [s.value for s in self.sources],
            "metadata": self.metadata,
            "rank": self.rank,
        }


@dataclass
class HybridRankingConfig:
    """Configuración para ranking híbrido."""
    
    vector_weight: float = 0.5
    bm25_weight: float = 0.3
    graph_weight: float = 0.2
    
    # Normalización
    normalize_scores: bool = True
    normalization_method: str = "minmax"  # minmax, zscore, rank
    
    # Combinación
    combination_method: str = "weighted"  # weighted, rrf, max
    rrf_k: int = 60  # Parámetro k para RRF
    
    # Filtros
    min_final_score: float = 0.0
    max_results: int = 10
    
    # Boost
    source_boost: Dict[str, float] = field(default_factory=dict)


def get_default_config() -> HybridRankingConfig:
    """Obtiene configuración por defecto."""
    config = get_rag_config()
    
    return HybridRankingConfig(
        vector_weight=config.get("vector_weight", 0.5),
        bm25_weight=config.get("bm25_weight", 0.3),
        graph_weight=config.get("graph_weight", 0.2),
        max_results=config.get("top_k", 10),
    )


# ==========================================
# Normalización de Scores
# ==========================================

def normalize_scores(
    scores: List[float],
    method: str = "minmax",
) -> List[float]:
    """
    Normaliza una lista de scores.
    
    Args:
        scores: Scores originales
        method: Método de normalización (minmax, zscore, rank)
        
    Returns:
        Scores normalizados (0-1)
    """
    if not scores:
        return scores
    
    if method == "minmax":
        return _minmax_normalize(scores)
    elif method == "zscore":
        return _zscore_normalize(scores)
    elif method == "rank":
        return _rank_normalize(scores)
    else:
        return _minmax_normalize(scores)


def _minmax_normalize(scores: List[float]) -> List[float]:
    """Normalización Min-Max."""
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


def _zscore_normalize(scores: List[float]) -> List[float]:
    """Normalización Z-Score."""
    if not scores:
        return scores
    
    n = len(scores)
    mean = sum(scores) / n
    
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = math.sqrt(variance) if variance > 0 else 1
    
    # Normalizar a 0-1 usando sigmoid
    z_scores = [(s - mean) / std for s in scores]
    
    return [1 / (1 + math.exp(-z)) for z in z_scores]


def _rank_normalize(scores: List[float]) -> List[float]:
    """Normalización por ranking."""
    if not scores:
        return scores
    
    n = len(scores)
    
    # Obtener ranking (mayor score = rank más alto)
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    ranks = [0] * n
    
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = n - rank
    
    # Normalizar ranks a 0-1
    return [(r - 1) / (n - 1) if n > 1 else 1.0 for r in ranks]


# ==========================================
# Combinación de Scores
# ==========================================

def combine_scores(
    results_by_source: Dict[str, List[Dict[str, Any]]],
    config: Optional[HybridRankingConfig] = None,
) -> List[RankedResult]:
    """
    Combina resultados de múltiples fuentes.
    
    Args:
        results_by_source: Dict con listas de resultados por fuente
            {
                "vector": [{"id": ..., "score": ..., "content": ...}, ...],
                "bm25": [...],
                "graph": [...]
            }
        config: Configuración de ranking
        
    Returns:
        Lista de RankedResult ordenados por score combinado
    """
    if config is None:
        config = get_default_config()
    
    if config.combination_method == "rrf":
        return _combine_rrf(results_by_source, config)
    elif config.combination_method == "max":
        return _combine_max(results_by_source, config)
    else:
        return _combine_weighted(results_by_source, config)


def _combine_weighted(
    results_by_source: Dict[str, List[Dict[str, Any]]],
    config: HybridRankingConfig,
) -> List[RankedResult]:
    """Combinación por suma ponderada."""
    
    # Mapeo de pesos
    weights = {
        "vector": config.vector_weight,
        "bm25": config.bm25_weight,
        "graph": config.graph_weight,
    }
    
    # Agregar resultados por ID
    combined: Dict[str, RankedResult] = {}
    
    for source, results in results_by_source.items():
        # Normalizar scores de esta fuente
        if config.normalize_scores and results:
            scores = [r.get("score", 0) for r in results]
            normalized = normalize_scores(scores, config.normalization_method)
        else:
            normalized = [r.get("score", 0) for r in results]
        
        for i, result in enumerate(results):
            result_id = result.get("id", "")
            
            if not result_id:
                continue
            
            norm_score = normalized[i] if i < len(normalized) else 0
            weight = weights.get(source, 0.1)
            
            if result_id in combined:
                ranked = combined[result_id]
            else:
                ranked = RankedResult(
                    id=result_id,
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                )
                combined[result_id] = ranked
            
            # Actualizar scores por fuente
            if source == "vector":
                ranked.vector_score = norm_score
            elif source == "bm25":
                ranked.bm25_score = norm_score
            elif source == "graph":
                ranked.graph_score = norm_score
            
            # Agregar fuente
            source_enum = RetrievalSource(source) if source in RetrievalSource._value2member_map_ else RetrievalSource.HYBRID
            if source_enum not in ranked.sources:
                ranked.sources.append(source_enum)
            
            # Actualizar score final
            ranked.final_score += norm_score * weight
    
    # Ordenar por score final
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x.final_score,
        reverse=True
    )
    
    # Filtrar y asignar ranks
    final_results = []
    for i, result in enumerate(sorted_results):
        if result.final_score >= config.min_final_score:
            result.rank = i + 1
            final_results.append(result)
            
            if len(final_results) >= config.max_results:
                break
    
    return final_results


def _combine_rrf(
    results_by_source: Dict[str, List[Dict[str, Any]]],
    config: HybridRankingConfig,
) -> List[RankedResult]:
    """
    Combinación por Reciprocal Rank Fusion (RRF).
    
    RRF score = Σ 1/(k + rank_i)
    """
    k = config.rrf_k
    
    # Agregar por ID
    combined: Dict[str, RankedResult] = {}
    
    for source, results in results_by_source.items():
        for rank, result in enumerate(results, 1):
            result_id = result.get("id", "")
            
            if not result_id:
                continue
            
            rrf_score = 1.0 / (k + rank)
            
            if result_id in combined:
                ranked = combined[result_id]
            else:
                ranked = RankedResult(
                    id=result_id,
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                )
                combined[result_id] = ranked
            
            # Actualizar score
            if source == "vector":
                ranked.vector_score = rrf_score
            elif source == "bm25":
                ranked.bm25_score = rrf_score
            elif source == "graph":
                ranked.graph_score = rrf_score
            
            ranked.final_score += rrf_score
            
            source_enum = RetrievalSource(source) if source in RetrievalSource._value2member_map_ else RetrievalSource.HYBRID
            if source_enum not in ranked.sources:
                ranked.sources.append(source_enum)
    
    # Ordenar y filtrar
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x.final_score,
        reverse=True
    )
    
    final_results = []
    for i, result in enumerate(sorted_results[:config.max_results]):
        if result.final_score >= config.min_final_score:
            result.rank = i + 1
            final_results.append(result)
    
    return final_results


def _combine_max(
    results_by_source: Dict[str, List[Dict[str, Any]]],
    config: HybridRankingConfig,
) -> List[RankedResult]:
    """Combinación por score máximo."""
    
    combined: Dict[str, RankedResult] = {}
    
    for source, results in results_by_source.items():
        # Normalizar
        if config.normalize_scores and results:
            scores = [r.get("score", 0) for r in results]
            normalized = normalize_scores(scores, config.normalization_method)
        else:
            normalized = [r.get("score", 0) for r in results]
        
        for i, result in enumerate(results):
            result_id = result.get("id", "")
            
            if not result_id:
                continue
            
            norm_score = normalized[i] if i < len(normalized) else 0
            
            if result_id in combined:
                ranked = combined[result_id]
                ranked.final_score = max(ranked.final_score, norm_score)
            else:
                ranked = RankedResult(
                    id=result_id,
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}),
                    final_score=norm_score,
                )
                combined[result_id] = ranked
            
            # Actualizar scores por fuente
            if source == "vector":
                ranked.vector_score = max(ranked.vector_score, norm_score)
            elif source == "bm25":
                ranked.bm25_score = max(ranked.bm25_score, norm_score)
            elif source == "graph":
                ranked.graph_score = max(ranked.graph_score, norm_score)
            
            source_enum = RetrievalSource(source) if source in RetrievalSource._value2member_map_ else RetrievalSource.HYBRID
            if source_enum not in ranked.sources:
                ranked.sources.append(source_enum)
    
    # Ordenar y filtrar
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x.final_score,
        reverse=True
    )
    
    final_results = []
    for i, result in enumerate(sorted_results[:config.max_results]):
        if result.final_score >= config.min_final_score:
            result.rank = i + 1
            final_results.append(result)
    
    return final_results


# ==========================================
# Selección Final
# ==========================================

def select_top_k(
    results: List[RankedResult],
    k: int = 10,
    min_score: float = 0.0,
    diversity_penalty: float = 0.0,
) -> List[RankedResult]:
    """
    Selecciona los top-k resultados.
    
    Args:
        results: Resultados rankeados
        k: Número de resultados a seleccionar
        min_score: Score mínimo
        diversity_penalty: Penalización por similitud (0-1)
        
    Returns:
        Top-k resultados
    """
    if not results:
        return []
    
    # Filtrar por score mínimo
    filtered = [r for r in results if r.final_score >= min_score]
    
    if diversity_penalty <= 0:
        return filtered[:k]
    
    # Selección con diversidad (MMR-like)
    selected = []
    remaining = filtered.copy()
    
    while len(selected) < k and remaining:
        if not selected:
            # Primer resultado: el de mayor score
            best = max(remaining, key=lambda x: x.final_score)
        else:
            # Siguiente resultado: balance entre score y diversidad
            best = None
            best_mmr = -float('inf')
            
            for candidate in remaining:
                # Calcular similitud con ya seleccionados
                max_sim = max(
                    _content_similarity(candidate.content, s.content)
                    for s in selected
                )
                
                # MMR score
                mmr = candidate.final_score - diversity_penalty * max_sim
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = candidate
        
        if best:
            selected.append(best)
            remaining.remove(best)
    
    # Actualizar ranks
    for i, result in enumerate(selected):
        result.rank = i + 1
    
    return selected


def _content_similarity(content1: str, content2: str) -> float:
    """
    Similaridad simple entre contenidos (Jaccard).
    
    Args:
        content1: Primer contenido
        content2: Segundo contenido
        
    Returns:
        Similaridad entre 0 y 1
    """
    if not content1 or not content2:
        return 0.0
    
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


# ==========================================
# Boost y Ajustes
# ==========================================

def apply_boosts(
    results: List[RankedResult],
    boosts: Dict[str, float],
) -> List[RankedResult]:
    """
    Aplica boosts a resultados basados en metadatos.
    
    Args:
        results: Resultados a modificar
        boosts: Dict de campo -> factor de boost
            Ejemplo: {"metadata.is_definition": 1.5}
        
    Returns:
        Resultados con scores ajustados
    """
    for result in results:
        boost_factor = 1.0
        
        for field, factor in boosts.items():
            # Navegar por campos anidados
            value = _get_nested_value(result.metadata, field.replace("metadata.", ""))
            
            if value:
                boost_factor *= factor
        
        result.final_score *= boost_factor
    
    # Re-ordenar
    results.sort(key=lambda x: x.final_score, reverse=True)
    
    # Actualizar ranks
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results


def _get_nested_value(data: Dict, path: str) -> Any:
    """Obtiene valor anidado de un dict."""
    keys = path.split(".")
    value = data
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value


def deduplicate_results(
    results: List[RankedResult],
    similarity_threshold: float = 0.9,
) -> List[RankedResult]:
    """
    Elimina resultados duplicados o muy similares.
    
    Args:
        results: Resultados a deduplicar
        similarity_threshold: Umbral de similitud para considerar duplicado
        
    Returns:
        Resultados sin duplicados
    """
    if not results:
        return results
    
    unique = []
    
    for result in results:
        is_duplicate = False
        
        for existing in unique:
            similarity = _content_similarity(result.content, existing.content)
            
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(result)
    
    # Actualizar ranks
    for i, result in enumerate(unique):
        result.rank = i + 1
    
    return unique
