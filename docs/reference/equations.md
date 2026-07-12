# Equation sheet

Every equation is a shape contract. Symbols: batch \(B\), sequence \(T\), residual width \(C\), heads \(H\), head width \(D=C/H\), vocabulary \(V\), experts \(E\), selected experts \(K\).

## Embedding

\[
X=W_E[\text{token IDs}],\qquad [B,T]\rightarrow[B,T,C]
\]

## Layer normalization and RMS normalization

\[
\operatorname{LayerNorm}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

\[
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{C}\sum_i x_i^2+\epsilon}}
\]

## Scaled dot-product attention

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt D}+M\right),\qquad O=AV
\]

Per head: `[B,T,D] @ [B,D,T] -> [B,T,T]`; then `[B,T,T] @ [B,T,D] -> [B,T,D]`. Causal mask \(M_{ij}=-\infty\) when key position \(j\) is in the future of query \(i\).

## SwiGLU feed-forward

\[
\operatorname{FFN}(x)=W_2\big(\operatorname{SiLU}(xW_g)\odot(xW_u)\big)
\]

The transformation expands from \(C\) to hidden width \(F\), gates elementwise, then projects back to \(C\).

## Pre-normalized residual block

\[
h'=h+\operatorname{Attention}(\operatorname{Norm}(h))
\]

\[
h_{next}=h'+\operatorname{FFN}(\operatorname{Norm}(h'))
\]

## Next-token cross-entropy

\[
\mathcal{L}=-\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(y_n\mid x_n)
\]

Perplexity under the same tokenization and loss convention:

\[
\operatorname{PPL}=e^{\mathcal{L}}
\]

Do not compare perplexity directly across different tokenizers without careful normalization.

## MoE routing

Router logits and probabilities for token vector \(x\):

\[
r=xW_r,\qquad p=\operatorname{softmax}(r),\qquad r,p\in\mathbb{R}^{E}
\]

Let \(S=\operatorname{TopK}(p,K)\). Renormalized combine weight for selected expert \(e\):

\[
g_e=\frac{p_e}{\sum_{j\in S}p_j},\quad e\in S
\]

Sparse FFN output:

\[
y=\sum_{e\in S}g_e\operatorname{Expert}_e(x)
\]

A simple expected capacity per expert for \(N\) token vectors is:

\[
\text{capacity}=\left\lceil\text{capacity factor}\cdot\frac{NK}{E}\right\rceil
\]

Actual implementations differ in grouping, capacity, padding, token dropping, and routing objective.

## AdamW sketch

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

With bias-corrected moments \(\hat m_t,\hat v_t\), a simplified decoupled weight-decay update is:

\[
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

## Sampling temperature

\[
p_i=\operatorname{softmax}(z_i/\tau)
\]

As \(\tau\) decreases above zero, the distribution sharpens. \(\tau=0\) is implemented as a special greedy case, not literal division.

## KV-cache size approximation

For \(L\) layers, batch \(B\), cached length \(T\), key/value heads \(H_{kv}\), head width \(D\), and \(s\) bytes per element:

\[
\text{KV bytes}\approx2LBTH_{kv}Ds
\]

Allocator metadata, block rounding, prefixes, and parallel placement add implementation-specific overhead.

