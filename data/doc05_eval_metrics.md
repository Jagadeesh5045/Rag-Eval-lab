# Evaluation Metrics for Language Models

Evaluating language models requires different metrics for different tasks. For classification-style outputs, precision measures the fraction of positive predictions that are correct, recall measures the fraction of actual positives that were found, and the F1 score is their harmonic mean. These same definitions carry over to information retrieval: precision at k is the fraction of the top-k retrieved documents that are relevant, and recall at k is the fraction of all relevant documents captured in the top-k.

For generated text, lexical overlap metrics such as BLEU and ROUGE compare n-gram overlap with reference answers, but they correlate poorly with factual correctness. Modern RAG evaluation therefore emphasises groundedness: the proportion of claims in a generated answer that are entailed or supported by the retrieved context. A grounded answer may still be incomplete, so completeness and relevance are tracked as separate dimensions.

Latency and cost are operational metrics that matter in production. Retrieval latency, time to first token, and total generation time are usually reported as percentiles such as p50 and p95 rather than means, because tail latency determines user experience. A good evaluation harness records per-query scores so failures can be drilled into individually.
