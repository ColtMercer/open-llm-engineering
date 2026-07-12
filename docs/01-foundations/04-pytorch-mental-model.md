# The PyTorch mental model

**Level:** Builder · **Time:** 35 minutes · **Prerequisite:** basic Python

PyTorch gives you multidimensional arrays, differentiable operations, parameter containers, and execution backends. A model is ordinary Python organizing tensor operations.

## Five objects to recognize

```python
import torch
from torch import nn

x = torch.randn(2, 4, 8)           # Tensor: data and shape
projection = nn.Linear(8, 16)      # Module: operation plus Parameters
y = projection(x)                  # Computation tracked by autograd
loss = y.square().mean()           # Scalar objective
loss.backward()                    # Populate parameter gradients
```

1. `Tensor` holds values, dtype, shape, device, and optional gradient history.
2. `Parameter` is a tensor registered as trainable state on a module.
3. `Module` contains parameters, buffers, child modules, and a `forward` computation.
4. Autograd records differentiable operations and applies reverse-mode differentiation.
5. An optimizer reads `.grad` and updates parameter values.

The [PyTorch documentation](https://pytorch.org/docs/stable/index.html) is the source of truth for a particular release.

## Parameters versus buffers versus activations

| Kind | Saved in `state_dict`? | Updated by optimizer? | Example |
|---|:---:|:---:|---|
| Parameter | Yes | Usually | projection weight |
| Persistent buffer | Yes | No | running statistic |
| Non-persistent buffer | No | No | regenerable mask |
| Activation | No | No | output of an attention layer |

Model checkpoints may store more than `state_dict`: optimizer tensors, schedule, scaler, and data progress are needed for faithful training resumption.

## Training and evaluation modes

`model.train()` and `model.eval()` change the behavior of modules such as dropout and batch normalization. They do **not** enable or disable gradients. Use `torch.no_grad()` or `torch.inference_mode()` when gradients are unnecessary.

```python
model.eval()
with torch.inference_mode():
    logits = model(token_ids)
```

## Dtypes and devices

The same computation can run with different numeric formats and backends. Float32 offers wide support; float16 and bfloat16 reduce memory and can accelerate supported hardware; lower-bit formats generally require specialized kernels and scaling strategies. Changing dtype can change stability and exact output.

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
token_ids = token_ids.to(device)
```

Never hard-code CUDA in an educational example that should run on a laptop.

## A module with an explicit shape contract

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, d_model]
        gate, value = self.gate_up(x).chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * value
        return self.down(hidden)  # [batch, time, d_model]
```

This is a gated feed-forward sublayer. A sparse MoE layer replaces one such shared transformation with several expert transformations plus a router; it does not replace attention.

## Reproducibility is more than a seed

Setting a seed controls some random streams, but exact reproducibility may also depend on backend algorithms, device count, data-worker order, distributed reductions, library versions, and nondeterministic kernels.

```python
torch.manual_seed(7)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(7)
```

Record environment and configuration alongside results. Do not promise bitwise reproduction when the runtime cannot provide it.

## Read production code without drowning

1. Find the configuration object and write down dimensions.
2. Find the top-level model `forward` signature.
3. Trace one block, ignoring optimizations at first.
4. Mark reshapes, transposes, normalization axes, and residual additions.
5. Separate mathematical equivalence from kernel fusion.
6. Return to cache, quantization, parallelism, and compilation only after the eager path is clear.

The compact [nanoGPT model implementation](https://github.com/karpathy/nanoGPT/blob/master/model.py) is a useful first production-adjacent trail. The companion code in this repository is even smaller and covered by shape tests.

## Exercises

1. Why does `model.eval()` not reduce memory by itself?
2. What breaks if a residual branch returns `[B,T,4C]` instead of `[B,T,C]`?
3. Why might a fused attention kernel look unlike the equation but compute the same function?

