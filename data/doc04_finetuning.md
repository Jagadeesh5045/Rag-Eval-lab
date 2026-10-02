# Fine-Tuning Large Language Models

Fine-tuning adapts a pre-trained language model to a specific task or domain by continuing training on labelled examples. Full fine-tuning updates every parameter, which is effective but expensive in memory and compute for billion-parameter models. It also risks catastrophic forgetting, where the model loses general capabilities while over-specialising on the new data.

Parameter-efficient methods freeze most of the model and train only a small set of added weights. LoRA (Low-Rank Adaptation) is the most popular: it injects trainable low-rank matrices into the attention layers, cutting trainable parameters to well under one percent while matching full fine-tuning quality on many tasks. The low-rank adapters can be merged back into the base weights at inference time, so there is no latency penalty.

Instruction tuning is a common fine-tuning recipe in which the model is trained on prompt-response pairs demonstrating desired behaviour, often followed by preference optimisation on human feedback. Good practice includes a held-out validation set, a small learning rate, and early stopping to avoid overfitting the fine-tuning data.
