# Transformer Architecture

The Transformer is a neural network architecture introduced for sequence modelling that relies entirely on self-attention instead of recurrence. Each input token is mapped to query, key and value vectors, and attention weights are computed as the scaled dot-product of queries and keys, followed by a softmax. This lets every position attend to every other position in parallel, which makes training far more efficient than recurrent models on modern hardware.

A standard Transformer stacks multiple encoder and decoder layers. Each layer contains a multi-head attention block, where several attention heads learn different relationship patterns, followed by a position-wise feed-forward network. Because attention itself is order-agnostic, positional encodings are added to the input embeddings so the model knows token order; these can be fixed sinusoidal patterns or learned vectors.

Large language models such as GPT and BERT are built on Transformer blocks. Decoder-only variants generate text autoregressively, while encoder-only variants produce rich contextual embeddings for classification and retrieval tasks. The main cost of attention is quadratic in sequence length, which is why long-context handling remains an active research area with sparse and linear attention variants.
