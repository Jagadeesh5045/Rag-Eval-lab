# NLP Preprocessing

Raw text must be cleaned before it can be used for modelling. Tokenization splits text into units such as words or subwords; subword methods like byte-pair encoding balance vocabulary size against the ability to handle rare words. Lowercasing and stripping punctuation are common early steps, though casing can carry signal for tasks like named entity recognition.

Morphological normalization reduces inflected forms to a base representation. Stemming chops suffixes with crude heuristics and is fast but can produce non-words, while lemmatization uses a dictionary and part-of-speech information to return valid base forms, at higher computational cost. Stopword removal drops frequent low-information words such as "the" and "is", which helps keyword methods like BM25 but is usually skipped for neural models that learn their own weighting.

The right pipeline depends on the downstream method: sparse retrieval benefits from aggressive normalization, while Transformer models prefer minimal preprocessing because their tokenizers and attention layers handle raw text well. Whatever the choice, the same preprocessing must be applied identically to documents and queries.
