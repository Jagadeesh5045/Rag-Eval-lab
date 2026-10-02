# Vector Databases

Vector databases store embeddings alongside metadata and serve low-latency similarity search at scale. Under the hood they combine an approximate nearest neighbour index with a storage engine: HNSW graphs give excellent recall-latency trade-offs for in-memory workloads, while IVF (inverted file) partitions quantise the space into clusters for very large datasets that do not fit in RAM.

Popular options include ChromaDB for lightweight local prototyping, and managed services such as Pinecone and Weaviate for production. Qdrant and Milvus are strong open-source choices that can be self-hosted. Most support metadata filtering, so a query can restrict candidates by fields like date or category before or during the vector search, which is essential for multi-tenant RAG applications.

Key operational concerns are indexing time, recall at a given latency budget, and the cost of keeping the index in memory. Hybrid search, which fuses dense vector scores with keyword scores like BM25, is increasingly built in because pure vector search can miss exact term matches such as product codes or names.
