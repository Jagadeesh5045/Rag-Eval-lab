"""Eval runner: scores retrieval + groundedness + latency over the test question set.

Usage:
    python eval.py [--k 5] [--data data] [--evals evals/test_questions.json]
                   [--out results/eval_results.json] [--threshold 0.45]

For every question the runner:
  1. retrieves top-k chunks with the hybrid (BM25 + TF-IDF + RRF) retriever,
  2. scores retrieval precision@k / recall@k against the gold relevant docs,
  3. runs the groundedness checker on a grounded reference answer AND on an
     adversarial answer containing an injected hallucination,
  4. records per-stage latency.

Results (per-query detail + summary) are written to --out as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.groundedness import check_answer
from src.ingest import build_corpus
from src.metrics import percentile, precision_recall_at_k
from src.retrieval import HybridRetriever


def run_eval(
    data_dir: str,
    evals_path: str,
    top_k: int = 5,
    threshold: float = 0.45,
) -> dict:
    corpus = build_corpus(data_dir)
    retriever = HybridRetriever(corpus)
    questions = json.loads(Path(evals_path).read_text(encoding="utf-8"))

    per_query = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_doc_ids = [h["doc_id"] for h in hits]
        precision, recall = precision_recall_at_k(
            retrieved_doc_ids, q["relevant_docs"], top_k
        )

        t1 = time.perf_counter()
        grounded = check_answer(q["reference_answer"], hits, threshold=threshold)
        adversarial = check_answer(q["adversarial_answer"], hits, threshold=threshold)
        check_ms = (time.perf_counter() - t1) * 1000.0

        per_query.append(
            {
                "id": q["id"],
                "question": q["question"],
                "relevant_docs": q["relevant_docs"],
                "retrieved_doc_ids": retrieved_doc_ids,
                "precision_at_k": round(precision, 4),
                "recall_at_k": round(recall, 4),
                "groundedness_reference": grounded["groundedness"],
                "verdict_reference": grounded["verdict"],
                "groundedness_adversarial": adversarial["groundedness"],
                "verdict_adversarial": adversarial["verdict"],
                "groundedness_drop": round(
                    grounded["groundedness"] - adversarial["groundedness"], 4
                ),
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "check_latency_ms": round(check_ms, 2),
                "total_latency_ms": round(retrieval_ms + check_ms, 2),
                "retrieved_chunks": [
                    {
                        "chunk_id": h["chunk_id"],
                        "doc_id": h["doc_id"],
                        "title": h["title"],
                        "rrf_score": h["rrf_score"],
                        "bm25_rank": h["bm25_rank"],
                        "tfidf_rank": h["tfidf_rank"],
                        "text": h["text"],
                    }
                    for h in hits
                ],
                "sentence_verdicts_reference": grounded["sentences"],
                "sentence_verdicts_adversarial": adversarial["sentences"],
            }
        )

    def avg(key: str) -> float:
        return round(sum(r[key] for r in per_query) / len(per_query), 4)

    latencies = [r["total_latency_ms"] for r in per_query]
    summary = {
        "n_questions": len(per_query),
        "top_k": top_k,
        "threshold": threshold,
        "n_chunks": len(corpus),
        "n_docs": len({c.doc_id for c in corpus}),
        "avg_precision_at_k": avg("precision_at_k"),
        "avg_recall_at_k": avg("recall_at_k"),
        "avg_groundedness_reference": avg("groundedness_reference"),
        "avg_groundedness_adversarial": avg("groundedness_adversarial"),
        "avg_groundedness_drop": avg("groundedness_drop"),
        "p50_latency_ms": round(percentile(latencies, 50), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
    }
    return {"summary": summary, "per_query": per_query}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG eval lab benchmark.")
    parser.add_argument("--k", type=int, default=3, help="top-k chunks retrieved per query")
    parser.add_argument("--data", default="data", help="document folder")
    parser.add_argument("--evals", default="evals/test_questions.json", help="question set JSON")
    parser.add_argument("--out", default="results/eval_results.json", help="output JSON path")
    parser.add_argument("--threshold", type=float, default=0.45, help="groundedness coverage threshold")
    args = parser.parse_args()

    results = run_eval(args.data, args.evals, top_k=args.k, threshold=args.threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    s = results["summary"]
    print(f"\nRAG Eval Lab — {s['n_questions']} questions, {s['n_docs']} docs, {s['n_chunks']} chunks")
    print(f"{'qid':<6}{'P@k':>7}{'R@k':>7}{'grnd':>7}{'adv':>7}{'drop':>7}{'ms':>9}")
    for r in results["per_query"]:
        print(
            f"{r['id']:<6}{r['precision_at_k']:>7.2f}{r['recall_at_k']:>7.2f}"
            f"{r['groundedness_reference']:>7.2f}{r['groundedness_adversarial']:>7.2f}"
            f"{r['groundedness_drop']:>7.2f}{r['total_latency_ms']:>9.1f}"
        )
    print("\nSummary:")
    for key in (
        "avg_precision_at_k", "avg_recall_at_k", "avg_groundedness_reference",
        "avg_groundedness_adversarial", "avg_groundedness_drop",
        "p50_latency_ms", "p95_latency_ms",
    ):
        print(f"  {key}: {s[key]}")
    print(f"\nWrote {out_path}\n")


if __name__ == "__main__":
    main()
