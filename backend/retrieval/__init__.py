"""
Retrieval package.
BM25, vector search y hybrid ranking.
"""

from backend.retrieval.bm25 import (
    BM25Result,
    BM25SearchResponse,
    search_text,
    highlight_matches,
    create_snippet,
    create_fulltext_index,
    search_multi_field,
)

from backend.retrieval.vector import (
    VectorResult,
    VectorSearchResponse,
    cosine_similarity,
    euclidean_distance,
    search_vectors,
    search_by_embedding,
    filter_by_threshold,
    adaptive_threshold,
    get_similar_chunks,
    normalize_vector,
    average_embeddings,
)

from backend.retrieval.hybrid_ranker import (
    RetrievalSource,
    RankedResult,
    HybridRankingConfig,
    normalize_scores,
    combine_scores,
    select_top_k,
    apply_boosts,
    deduplicate_results,
)

__all__ = [
    # BM25
    "BM25Result",
    "BM25SearchResponse",
    "search_text",
    "highlight_matches",
    "create_snippet",
    "create_fulltext_index",
    "search_multi_field",
    # Vector
    "VectorResult",
    "VectorSearchResponse",
    "cosine_similarity",
    "euclidean_distance",
    "search_vectors",
    "search_by_embedding",
    "filter_by_threshold",
    "adaptive_threshold",
    "get_similar_chunks",
    "normalize_vector",
    "average_embeddings",
    # Hybrid Ranker
    "RetrievalSource",
    "RankedResult",
    "HybridRankingConfig",
    "normalize_scores",
    "combine_scores",
    "select_top_k",
    "apply_boosts",
    "deduplicate_results",
]
