# Concept ladder

This is a map of when technical vocabulary enters the canonical course. Every chapter still defines a term inline; this page helps readers see how later ideas depend on earlier ones.

## Stage 0 — ordinary ideas

First taught in [Before the jargon](../01-foundations/00-before-the-jargon.md):

| Term | Plain meaning |
|---|---|
| Example or data | Something the program can learn from |
| Pattern | A relationship that appears across examples |
| Model | A program whose behavior was shaped by examples |
| Input / output | What goes into the program / what comes out |
| Prediction | A possible output selected before the answer is known |
| Parameter | An adjustable internal number |
| Training | Adjusting parameters using examples |
| Inference | Using the trained model without changing those parameters |
| Token | A numbered piece of text |

## Stage 1 — language-model basics

First taught across [What an LLM is](../01-foundations/01-what-is-an-llm.md), [Learning from examples](../01-foundations/02-learning-from-data.md), and [Text becomes tokens](../02-tokenization/01-text-becomes-tokens.md):

| Term | What it adds |
|---|---|
| Context | The earlier text available for the next prediction |
| Vocabulary | The complete set of token IDs a tokenizer can produce |
| Probability | A number describing how likely an option is under the model |
| Loss | One number measuring how poor a training prediction was |
| Optimizer | The rule that turns learning signals into parameter changes |

## Stage 2 — the mathematical toolkit

First taught in [Math with shapes](../01-foundations/03-math-with-shapes.md) and [PyTorch mental model](../01-foundations/04-pytorch-mental-model.md):

| Term | What it adds |
|---|---|
| Vector | An ordered list of numbers |
| Matrix | A rectangular grid of numbers |
| Tensor | A general multi-dimensional collection of numbers |
| Shape | The size of each tensor dimension |
| Gradient | A local signal showing how a small change affects loss |

## Stage 3 — the ordinary Transformer

First taught in the [Transformer section](../04-transformer/01-transformer-mental-model.md):

- embedding;
- position information;
- query, key, and value;
- causal self-attention;
- attention head;
- feed-forward layer;
- normalization;
- residual connection;
- language-model head and logits.

## Stage 4 — training at scale

First taught in [Pretraining](../06-training/01-pretraining-objective.md):

- batch and training step;
- learning-rate schedule;
- checkpoint;
- data, tensor, pipeline, context, and sequence parallelism;
- validation and benchmark evaluation.

## Stage 5 — optional architecture extensions

Only after the dense Transformer is established, the [Mixture of experts section](../05-moe/01-why-sparse-models.md) introduces:

- dense versus sparse computation;
- expert feed-forward networks;
- router and top-k selection;
- expert capacity and load balancing;
- expert parallelism.

## Stage 6 — behavior after pretraining

The [Post-training section](../07-post-training/01-sft-and-instruction-tuning.md) introduces supervised instruction tuning, preference data, reward models, preference optimization, reinforcement learning, verifiers, and reasoning-oriented training.

## Stage 7 — running and operating the model

The [Inference section](../08-inference/01-decoding.md) introduces logits, decoding, temperature, top-k/top-p sampling, key-value cache, batching, quantization, and speculative decoding in their production context.

## Stage 8 — systems around the model

The [Prompting](../09-prompting/01-prompting-as-interface.md) and [Agents](../10-agents/01-agent-loop.md) sections introduce structured output, tool calls, retrieval-augmented generation, memory, planning, authorization, and agent evaluation.
