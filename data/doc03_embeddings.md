# Embeddings and Vector Search

Embeddings are dense numeric vectors that represent the meaning of text as a point in a high-dimensional space. Texts with similar meaning end up close together, so semantic search becomes a nearest-neighbour problem: embed the query, then find the stored vectors with the smallest distance. Cosine similarity is the most common distance measure for this task, computed as the dot product of two vectors divided by the product of their lengths.

Modern embedding models are usually Transformer encoders trained with contrastive objectives on query-passage pairs. The dimensionality typically ranges from 384 to 3072. Smaller models are faster and cheaper to serve, while larger models capture finer semantic distinctions. Domain-specific fine-tuning on in-domain pairs can substantially improve retrieval quality for specialised corpora such as legal or medical documents.

Exact nearest-neighbour search scans every vector and becomes too slow at scale, so production systems use approximate nearest neighbour (ANN) indexes. Algorithms like HNSW build navigable graph structures that return very good neighbours in logarithmic time. The trade-off is between recall of the true nearest neighbours, query latency, and memory consumption of the index.
