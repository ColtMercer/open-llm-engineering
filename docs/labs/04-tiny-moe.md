# Lab 4: inspect a tiny MoE router

```bash
python labs/04_tiny_moe.py --capacity-factor 1.0
```

The report shows, for each token vector:

- top-2 selected expert indices;
- normalized combine weights;
- requested and accepted assignments per expert;
- capacity and overflow;
- an auxiliary load-balancing term.

The printed words are labels for the reader. The untrained router receives only random vectors, so calling one row “math” does not create a math expert.

## Experiments

1. Use a very small capacity factor and count dropped assignments.
2. Set all router weights to zero and inspect deterministic top-k ties.
3. Train the layer on two synthetic transformations and graph expert load.
4. Compare top-1 and top-2 activated parameter count and output combination.
5. Remove the auxiliary loss in a synthetic training task and watch for collapse.

Read the [router math](../05-moe/02-router-math.md) before interpreting specialization.

