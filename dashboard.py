"""Streamlit dashboard for the RAG Eval Lab.

Run with:
    streamlit run dashboard.py

Sections: KPI cards, per-query bar charts, latency histogram, per-query
drill-down (retrieved chunks + sentence-level groundedness verdicts), and a
live demo where you can ask a question and test an answer for groundedness.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_PATH = BASE_DIR / "results" / "eval_results.json"

# Streamlit is imported lazily inside main() so `import dashboard` stays light
# for smoke tests; `streamlit run dashboard.py` executes main() via __main__.


def _load_results() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def _get_retriever(st):
    from src.ingest import build_corpus
    from src.retrieval import HybridRetriever

    @st.cache_resource(show_spinner="Indexing corpus…")
    def _build():
        corpus = build_corpus(str(BASE_DIR / "data"))
        return HybridRetriever(corpus)

    return _build()


def main() -> None:
    import pandas as pd
    import streamlit as st

    from src.groundedness import check_answer

    st.set_page_config(page_title="RAG Eval Lab", page_icon="🧪", layout="wide")
    st.title("🧪 RAG Eval Lab — Retrieval & Groundedness Dashboard")
    st.caption(
        "Hybrid retrieval (BM25 + TF-IDF, RRF fusion) over 10 AI/ML guides, "
        "scored with a local sentence-level groundedness checker. No API calls."
    )

    results = _load_results()
    if results is None:
        st.warning(
            "No eval results found. Run `python eval.py` first to generate "
            "`results/eval_results.json`, then refresh this page."
        )
        return

    summary = results["summary"]
    per_query = results["per_query"]
    df = pd.DataFrame(per_query)

    # ---- KPI cards ----
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg precision@k", f"{summary['avg_precision_at_k']:.2f}")
    k2.metric("Avg recall@k", f"{summary['avg_recall_at_k']:.2f}")
    k3.metric("Avg groundedness", f"{summary['avg_groundedness_reference']:.2f}")
    k4.metric("p95 latency", f"{summary['p95_latency_ms']:.1f} ms")
    st.caption(
        f"{summary['n_questions']} questions · {summary['n_docs']} docs · "
        f"{summary['n_chunks']} chunks · top_k={summary['top_k']} · "
        f"adversarial groundedness {summary['avg_groundedness_adversarial']:.2f} "
        f"(drop {summary['avg_groundedness_drop']:.2f} — the checker catches hallucinations)"
    )

    # ---- Per-query bars ----
    st.subheader("Per-query scores")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Retrieval precision@k / recall@k")
        st.bar_chart(df.set_index("id")[["precision_at_k", "recall_at_k"]])
    with c2:
        st.caption("Groundedness: reference vs adversarial answer")
        st.bar_chart(
            df.set_index("id")[["groundedness_reference", "groundedness_adversarial"]]
        )

    # ---- Latency histogram ----
    st.subheader("Latency distribution")
    bins = pd.cut(df["total_latency_ms"], bins=8)
    hist = df.groupby(bins, observed=True).size()
    hist.index = [f"{iv.left:.0f}–{iv.right:.0f} ms" for iv in hist.index]
    st.bar_chart(hist)
    st.caption(
        f"p50 {summary['p50_latency_ms']:.1f} ms · p95 {summary['p95_latency_ms']:.1f} ms "
        "(retrieval + groundedness check, local only)"
    )

    # ---- Drill-down ----
    st.subheader("Per-query drill-down")
    qid = st.selectbox("Question", df["id"].tolist())
    row = df[df["id"] == qid].iloc[0].to_dict()

    st.markdown(f"**Q:** {row['question']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision@k", f"{row['precision_at_k']:.2f}")
    m2.metric("Recall@k", f"{row['recall_at_k']:.2f}")
    m3.metric("Grounded (ref)", f"{row['groundedness_reference']:.2f}")
    m4.metric("Grounded (adv)", f"{row['groundedness_adversarial']:.2f}")

    st.markdown("**Retrieved chunks**")
    chunk_rows = [
        {
            "chunk_id": c["chunk_id"],
            "doc": c["doc_id"],
            "rrf": c["rrf_score"],
            "bm25_rank": c["bm25_rank"],
            "tfidf_rank": c["tfidf_rank"],
            "relevant": "✅" if c["doc_id"] in row["relevant_docs"] else "",
            "preview": c["text"][:160] + "…",
        }
        for c in row["retrieved_chunks"]
    ]
    st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)

    for label, key in (("Reference answer", "sentence_verdicts_reference"),
                       ("Adversarial answer", "sentence_verdicts_adversarial")):
        st.markdown(f"**Sentence verdicts — {label}**")
        sent_rows = [
            {
                "verdict": ("✅" if s["verdict"] == "SUPPORTED"
                            else "❌" if s["verdict"] == "UNSUPPORTED" else "⏭️"),
                "coverage": s["coverage"],
                "evidence": s["evidence_chunk"],
                "sentence": s["sentence"],
            }
            for s in row[key]
        ]
        st.dataframe(pd.DataFrame(sent_rows), use_container_width=True, hide_index=True)

    # ---- Live demo ----
    st.subheader("🔴 Live demo — try your own question")
    retriever = _get_retriever(st)
    question = st.text_input("Ask something about the corpus", "What is RAG?")
    if question.strip():
        hits = retriever.search(question.strip(), top_k=5)
        st.markdown("**Top retrieved chunks**")
        st.dataframe(
            pd.DataFrame([
                {"chunk_id": h["chunk_id"], "doc": h["doc_id"],
                 "rrf": h["rrf_score"], "preview": h["text"][:140] + "…"}
                for h in hits
            ]),
            use_container_width=True, hide_index=True,
        )
        answer = st.text_area(
            "Paste a candidate answer to check for groundedness",
            "RAG retrieves relevant passages and feeds them to the model. "
            "It was invented on Mars in 2019.",
        )
        if st.button("Check groundedness") and answer.strip():
            verdict = check_answer(answer.strip(), hits)
            vc1, vc2 = st.columns(2)
            vc1.metric("Groundedness", f"{verdict['groundedness']:.2f}")
            vc2.metric("Verdict", verdict["verdict"])
            for s in verdict["sentences"]:
                icon = ("✅" if s["verdict"] == "SUPPORTED"
                        else "❌" if s["verdict"] == "UNSUPPORTED" else "⏭️")
                st.write(f"{icon} `{s['verdict']}` (coverage {s['coverage']}) — {s['sentence']}")


if __name__ == "__main__":
    main()
