# 🧪 RAG Eval Lab

A fully local evaluation harness for Retrieval-Augmented Generation pipelines: hybrid retrieval (BM25 + TF-IDF with RRF fusion), a sentence-level groundedness checker, a scripted benchmark runner, and a Streamlit dashboard. No API keys, no network calls at runtime — everything runs on your machine.

## Problem statement

RAG systems fail in two places: the retriever returns the wrong passages, or the generator makes claims the passages don't support. Most teams eyeball a few examples and ship. This lab makes both failure modes measurable:

1. **Retrieval quality** — precision@k / recall@k of a hybrid retriever against gold relevant documents.
2. **Answer groundedness** — what fraction of each generated answer's sentences are actually supported by the retrieved context, including adversarial answers with injected hallucinations to prove the checker catches them.
3. **Latency** — per-query retrieval + checking time, reported as p50/p95.

## Architecture

```mermaid
flowchart LR
    A[data/*.md<br/>10 AI/ML guides] --> B[ingest.py<br/>word chunking + overlap]
    B --> C[(chunk corpus)]
    C --> D[BM25 index<br/>rank-bm25]
    C --> E[TF-IDF index<br/>scikit-learn]
    Q([query]) --> D
    Q --> E
    D --> F[RRF fusion<br/>1/(60+rank)]
    E --> F
    F --> G[top-k chunks]
    ANS([candidate answer]) --> H[groundedness.py<br/>sentence coverage check]
    G --> H
    H --> I[verdicts + scores]
    G --> J[eval.py<br/>P@k · R@k · latency]
    J --> K[(results/eval_results.json)]
    K --> L[dashboard.py<br/>Streamlit]
```

## How it works

**Ingestion** (`src/ingest.py`) — loads the Markdown guides from `data/`, splits them into ~200-word chunks with 40-word overlap, and assigns stable IDs (`doc_id::cNN`).

**Hybrid retrieval** (`src/retrieval.py`) — each query is ranked twice: keyword search with `BM25Okapi` and cosine similarity over a TF-IDF matrix. The two rankings are fused with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`), which needs no score normalisation and is robust when the rankers disagree. Per-ranker ranks are kept for drill-down.

**Groundedness checker** (`src/groundedness.py`) — splits a candidate answer into sentences and, for each one, measures content-token coverage against every retrieved chunk (stopwords removed, no external models). A sentence is `SUPPORTED` if some chunk covers ≥ 45% of its content tokens, else `UNSUPPORTED`. The answer score is the supported fraction; answers score `GROUNDED` (≥ 0.8), `PARTIAL` (≥ 0.5) or `UNGROUNDED`.

**Eval runner** (`eval.py`) — runs 12 test questions (`evals/test_questions.json`). Each question ships with gold relevant docs, a grounded reference answer, and an adversarial answer with an injected hallucination. The runner records retrieval precision/recall, groundedness for both answers (the drop proves the checker discriminates), and per-stage latency.

## Project structure

```
rag-eval-lab/
├── data/                  # 10 sample AI/ML guides (Markdown)
├── evals/
│   └── test_questions.json  # 12 questions: gold docs + reference/adversarial answers
├── src/
│   ├── ingest.py          # loading + chunking
│   ├── retrieval.py       # BM25 + TF-IDF + RRF hybrid retriever
│   ├── groundedness.py    # sentence-level claim verification
│   └── metrics.py         # precision/recall@k, percentiles
├── eval.py                # benchmark runner → results/eval_results.json
├── dashboard.py           # Streamlit dashboard + live demo
├── results/               # eval output (generated)
└── requirements.txt
```

## How to run

```bash
# 1. Create a virtualenv and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the benchmark (12 questions, ~seconds, fully offline)
python eval.py
# options: --k 5 --threshold 0.45 --out results/eval_results.json

# 3. Launch the dashboard
streamlit run dashboard.py
```

The dashboard shows KPI cards (avg precision, recall, groundedness, p95 latency), per-query bar charts, a latency histogram, a per-query drill-down (retrieved chunks with RRF/BM25/TF-IDF ranks plus sentence-level verdicts), and a live demo where you can ask your own question and test any answer for groundedness.

## Sample output

Real output from `python eval.py` on this machine:

```
RAG Eval Lab — 12 questions, 10 docs, 11 chunks
qid       P@k    R@k   grnd    adv   drop       ms
q01      0.33   1.00   1.00   0.50   0.50      7.0
q02      0.33   1.00   1.00   0.50   0.50      9.2
q03      0.67   1.00   1.00   0.50   0.50      6.6
q04      0.33   1.00   1.00   0.50   0.50      9.8
q05      0.67   1.00   1.00   0.50   0.50      5.1
q06      0.33   1.00   1.00   0.50   0.50      5.6
q07      0.33   1.00   1.00   0.50   0.50      5.9
q08      0.33   1.00   1.00   0.50   0.50      6.0
q09      0.33   1.00   1.00   0.50   0.50      4.4
q10      0.33   1.00   1.00   0.50   0.50      7.7
q11      0.33   1.00   1.00   0.50   0.50      2.5
q12      0.33   0.50   0.67   0.50   0.17     26.1

Summary:
  avg_precision_at_k: 0.3889
  avg_recall_at_k: 0.9583
  avg_groundedness_reference: 0.9722
  avg_groundedness_adversarial: 0.5
  avg_groundedness_drop: 0.4722
  p50_latency_ms: 6.33
  p95_latency_ms: 17.11
```

Reading the numbers: recall is near-perfect (the hybrid retriever reliably surfaces the right guide; precision@3 is bounded by questions with a single relevant doc). Every adversarial answer — each carrying one injected hallucination — scores exactly 0.50 groundedness versus ~1.0 for the reference, so the checker cleanly separates them. `q12` is the instructive edge case: the retriever missed one relevant doc (recall 0.50), and the checker *correctly* flagged a claim drawn from that missing doc as unsupported — groundedness is bounded by retrieval recall.

## Limitations & next steps

- The groundedness heuristic is lexical (token coverage), so paraphrased-but-true claims can score low — a real deployment would add an NLI model.
- TF-IDF stands in for dense embeddings to keep the lab dependency-free and offline; swapping in a sentence-transformer is a clean extension point (`retrieval.py`).
- Next: faithfulness vs. completeness split, chunking-strategy ablations, and CI gating on the summary metrics.
