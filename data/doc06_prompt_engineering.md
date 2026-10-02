# Prompt Engineering

Prompt engineering is the practice of designing inputs that steer a language model toward better outputs without changing its weights. A system prompt sets the model's role, tone and constraints for the whole conversation, while the user prompt carries the specific task. Clear, explicit instructions with the desired output format usually outperform vague requests.

Few-shot prompting includes a handful of input-output examples in the prompt so the model can infer the pattern, which is effective for classification and structured extraction. Chain-of-thought prompting asks the model to reason step by step before answering, which improves accuracy on arithmetic and multi-hop reasoning problems. Both techniques cost extra tokens, so they trade accuracy against latency and price.

Reliability techniques include asking the model to quote its sources, constraining outputs to JSON schemas, and lowering temperature for deterministic tasks. Prompt injection is a security concern: untrusted text retrieved from documents or tools can hijack the model's instructions, so retrieved content should be clearly delimited and never blindly trusted.
