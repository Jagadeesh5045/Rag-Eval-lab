"""Hybrid retrieval: BM25 keyword search + TF-IDF cosine search, fused with RRF."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .ingest import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridRetriever:
    """Rank chunks with BM25 and TF-IDF cosine similarity, fuse via Reciprocal Rank Fusion."""

    def __init__(self, chunks: list[Chunk], rrf_k: int = 60):
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty chunk list")
        self.chunks = chunks
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self._vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9]+", lowercase=True)
        self._tfidf = self._vectorizer.fit_transform([c.text for c in chunks])

    @staticmethod
    def _ranked(scores: np.ndarray, depth: int) -> list[tuple[int, float]]:
        order = np.argsort(scores, kind="stable")[::-1][:depth]
        return [(int(i), float(scores[i])) for i in order]

    def search(self, query: str, top_k: int = 5, depth: int = 50) -> list[dict]:
        """Return top_k chunks as dicts with fused RRF scores and per-ranker diagnostics."""
        bm25_scores = self._bm25.get_scores(tokenize(query))
        query_vec = self._vectorizer.transform([query])
        cosine_scores = np.asarray(linear_kernel(query_vec, self._tfidf)).ravel()

        bm25_ranked = self._ranked(bm25_scores, depth)
        tfidf_ranked = self._ranked(cosine_scores, depth)

        fused: dict[int, float] = {}
        for rank, (idx, _) in enumerate(bm25_ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(tfidf_ranked, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        bm25_rank = {idx: r for r, (idx, _) in enumerate(bm25_ranked, start=1)}
        tfidf_rank = {idx: r for r, (idx, _) in enumerate(tfidf_ranked, start=1)}

        results = []
        for idx, rrf_score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
            chunk = self.chunks[idx]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "rrf_score": round(rrf_score, 6),
                    "bm25_rank": bm25_rank.get(idx),
                    "tfidf_rank": tfidf_rank.get(idx),
                    "bm25_score": round(float(bm25_scores[idx]), 4),
                    "cosine_score": round(float(cosine_scores[idx]), 4),
                }
            )
        return results
