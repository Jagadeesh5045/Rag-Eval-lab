# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation combines a retriever with a generator to ground language model outputs in external knowledge. At query time the system first retrieves the most relevant passages from a document corpus, then feeds those passages to the language model alongside the question so the answer can cite and rely on retrieved evidence. This reduces hallucinations and lets the model answer questions about documents it never saw during training.

A typical RAG pipeline has three stages. First, documents are split into chunks of a few hundred tokens, often with some overlap so context is not lost at boundaries. Second, each chunk is indexed for retrieval, commonly with dense embeddings, keyword indexes like BM25, or a hybrid of both. Third, the top-ranked chunks are inserted into the prompt and the model generates an answer conditioned on them.

Chunking strategy strongly affects quality: chunks that are too small lose context, while chunks that are too large dilute relevance and waste context window. Common practice is 200 to 500 tokens per chunk with 10 to 20 percent overlap. Evaluation of RAG systems focuses on retrieval precision and recall plus answer groundedness, which measures whether every claim in the answer is supported by the retrieved passages.
