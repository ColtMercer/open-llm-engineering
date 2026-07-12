# Myth detector

Short claims often hide category mistakes. Use these replacements.

| Myth | Better model |
|---|---|
| “The model stores documents in its database.” | Training compresses patterns into distributed parameters; models can still memorize and reproduce some strings. Retrieval is a separate system unless explicitly added. |
| “Open weights means open source.” | Audit weights, architecture, training code, data information/artifacts, logs, checkpoints, and licenses separately. |
| “More parameters means more compute per token.” | Dense models often follow that direction, but sparse MoE separates total from activated parameters. Serving still needs weight memory and communication. |
| “Each MoE expert has a named subject.” | Experts are learned FFN paths selected from token representations. Specialization can be distributed, overlapping, unstable, or hard to label. |
| “Prompting the right phrase calls the math expert.” | Prompt text changes token representations and can indirectly change hidden routes; normal APIs do not expose or guarantee expert selection. |
| “Temperature adds creativity or knowledge.” | Temperature reshapes a fixed next-token distribution before sampling. It changes selection diversity, not weights or evidence. |
| “The context window is memory.” | Context is request-time input/cache. Persistent application memory requires storage, retrieval, summarization, or parameter updates. |
| “RAG prevents hallucinations.” | Retrieval can supply evidence; relevance, prompt use, generation, and citation verification can still fail. |
| “A benchmark score is the model.” | It is a measurement of a specific model/runtime/prompt/decoder/harness configuration on a dataset. |
| “Chain-of-thought text reveals the model's true internal reasoning.” | Generated rationale is an output sequence. It can help performance or inspection in some settings but is not a guaranteed faithful transcript of internal computation. |
| “Lossless tokenizer means fair tokenizer.” | Byte fallback can preserve every string while using far more tokens for underrepresented languages or domains. |
| “Loss went down, so the model got better.” | Training loss measures one objective on sampled data. Validate held-out loss, target tasks, safety, robustness, and serving behavior. |
| “Quantization only makes the file smaller.” | It also changes kernels, memory bandwidth, cache behavior, throughput, and sometimes quality. |
| “A seed makes training reproducible.” | Data order, kernels, reductions, device topology, versions, and failure/resume behavior also matter. |
| “Agents are models that think repeatedly.” | An agent is a control loop around model calls with state, actions, observations, policies, and termination. |

## A five-question claim filter

1. **Which layer?** Data, tokenizer, architecture, checkpoint, runtime, prompt, tool, or product?
2. **Which version?** Model revision, code commit, dataset snapshot, harness, and date?
3. **Which measurement?** Units, denominator, sample, uncertainty, and controls?
4. **Which evidence?** Primary source, derived calculation, inference, or heuristic?
5. **Which boundary?** What would make the statement stop being true?

