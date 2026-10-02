"""rag-eval-lab: local hybrid-retrieval RAG evaluation harness."""

from .ingest import build_corpus, Chunk
from .retrieval import HybridRetriever
from .groundedness import check_answer
from .metrics import precision_recall_at_k, percentile

__all__ = [
    "build_corpus",
    "Chunk",
    "HybridRetriever",
    "check_answer",
    "precision_recall_at_k",
    "percentile",
]
