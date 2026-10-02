"""Retrieval metrics and percentile helpers."""

from __future__ import annotations

import numpy as np


def precision_recall_at_k(
    retrieved_doc_ids: list[str], relevant_doc_ids: list[str], k: int
) -> tuple[float, float]:
    """Doc-level precision@k and recall@k for one query."""
    relevant = set(relevant_doc_ids)
    retrieved = retrieved_doc_ids[:k]
    if not retrieved or not relevant:
        return 0.0, 0.0
    hits = len(set(retrieved) & relevant)
    return hits / len(retrieved), hits / len(relevant)


def percentile(values: list[float], p: float) -> float:
    """p-th percentile (0-100) of values; 0.0 for empty input."""
    if not values:
        return 0.0
    return float(np.percentile(values, p))
