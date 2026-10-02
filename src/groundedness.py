"""Sentence-level groundedness checking — fully local, no API calls.

For each sentence (claim) in a candidate answer we measure how much of its
content vocabulary is covered by the best retrieved chunk. Sentences whose
coverage falls below a threshold are flagged UNSUPPORTED, which is how
hallucinated claims get caught.
"""

from __future__ import annotations

import re

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Compact stopword list so the checker stays dependency-free and offline.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "at", "by", "for", "with", "about", "into", "through", "during",
    "before", "after", "between", "under", "over", "to", "from", "in", "on",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "should", "may",
    "might", "must", "shall", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "our",
    "their", "my", "your", "as", "so", "such", "no", "not", "only", "also",
    "than", "too", "very", "just", "there", "here", "which", "who", "whom",
    "what", "where", "how", "why", "all", "any", "both", "each", "few",
    "more", "most", "other", "some",
}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def content_tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def coverage(sentence: str, chunk_text: str) -> float | None:
    """Fraction of the sentence's content tokens present in the chunk.

    Returns None for sentences too short to judge (fewer than 3 content tokens).
    """
    sent_tokens = set(content_tokens(sentence))
    if len(sent_tokens) < 3:
        return None
    chunk_tokens = set(content_tokens(chunk_text))
    return len(sent_tokens & chunk_tokens) / len(sent_tokens)


def check_answer(
    answer: str, retrieved_chunks: list[dict], threshold: float = 0.45
) -> dict:
    """Verify every sentence of `answer` against the retrieved chunks.

    Returns per-sentence verdicts with the best supporting chunk, plus an
    overall groundedness score in [0, 1] and a coarse verdict label.
    """
    sentences = split_sentences(answer)
    details = []
    for sent in sentences:
        best_chunk, best_cov = None, -1.0
        for chunk in retrieved_chunks:
            cov = coverage(sent, chunk["text"])
            if cov is not None and cov > best_cov:
                best_chunk, best_cov = chunk["chunk_id"], cov
        if best_chunk is None:
            details.append(
                {"sentence": sent, "verdict": "SKIPPED", "coverage": None, "evidence_chunk": None}
            )
        else:
            details.append(
                {
                    "sentence": sent,
                    "verdict": "SUPPORTED" if best_cov >= threshold else "UNSUPPORTED",
                    "coverage": round(best_cov, 3),
                    "evidence_chunk": best_chunk,
                }
            )

    scored = [d for d in details if d["verdict"] != "SKIPPED"]
    supported = sum(1 for d in scored if d["verdict"] == "SUPPORTED")
    score = (supported / len(scored)) if scored else 1.0
    label = "GROUNDED" if score >= 0.8 else ("PARTIAL" if score >= 0.5 else "UNGROUNDED")
    return {
        "groundedness": round(score, 4),
        "verdict": label,
        "n_sentences": len(sentences),
        "n_scored": len(scored),
        "n_supported": supported,
        "threshold": threshold,
        "sentences": details,
    }
