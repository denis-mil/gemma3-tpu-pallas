"""Pure-JAX fp32 reference implementations.

These are the ground truth every Pallas kernel is asserted against. They are
written for obviousness, not speed -- if a reference and a kernel disagree, the
reference is assumed right.

Kernel boundary: RoPE, QK-norm and the input RMSNorm are applied *outside* these
functions. The kernels take post-RoPE Q/K/V, matching the interface that
`splash_attention` and flash-attention kernels expose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .shapes import GEMMA3_1B, Gemma3Config


def gelu_tanh(x: jax.Array) -> jax.Array:
    """Tanh-approximate GELU -- Gemma's `gelu_pytorch_tanh` activation."""
    return jax.nn.gelu(x, approximate=True)


def gated_mlp(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
) -> jax.Array:
    """Gemma's GeGLU feed-forward block, computed in fp32.

    x       [T, embed_dim]
    w_gate  [embed_dim, hidden_dim]
    w_up    [embed_dim, hidden_dim]
    w_down  [hidden_dim, embed_dim]
    """
    x32 = x.astype(jnp.float32)
    gate = gelu_tanh(x32 @ w_gate.astype(jnp.float32))
    up = x32 @ w_up.astype(jnp.float32)
    return (gate * up) @ w_down.astype(jnp.float32)


def attention_mask(
    q_len: int,
    kv_len: int,
    *,
    window: int | None = None,
    offset: int = 0,
) -> jax.Array:
    """Boolean mask, True where attention is allowed.

    Causal: query i may attend to key j only when j <= i.
    Windowed: additionally j > i - window, i.e. a band of `window` keys.

    `offset` is the position of the first query within the key sequence, so the
    same helper serves prefill (offset 0) and decode.
    """
    q_pos = jnp.arange(q_len)[:, None] + offset
    k_pos = jnp.arange(kv_len)[None, :]
    allowed = k_pos <= q_pos
    if window is not None:
        allowed = allowed & (k_pos > q_pos - window)
    return allowed


def attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    window: int | None = None,
    cfg: Gemma3Config = GEMMA3_1B,
) -> jax.Array:
    """Multi-query attention in fp32.

    q  [T, num_heads, head_dim]
    k  [S, num_kv_heads, head_dim]
    v  [S, num_kv_heads, head_dim]

    Returns [T, num_heads, head_dim].

    With num_kv_heads == 1 the single KV head is broadcast across all query
    heads, which is what makes the decode shape so thin against a 128x128 MXU.
    """
    q32, k32, v32 = (a.astype(jnp.float32) for a in (q, k, v))

    q_len, num_heads, _ = q32.shape
    kv_len, num_kv_heads, _ = k32.shape
    if num_heads % num_kv_heads:
        raise ValueError(f"{num_heads} query heads is not a multiple of {num_kv_heads} KV heads")
    repeats = num_heads // num_kv_heads

    k32 = jnp.repeat(k32, repeats, axis=1)
    v32 = jnp.repeat(v32, repeats, axis=1)

    # [heads, q_len, kv_len]
    logits = jnp.einsum("qhd,khd->hqk", q32, k32) * cfg.attn_scale
    allowed = attention_mask(q_len, kv_len, window=window)
    logits = jnp.where(allowed[None, :, :], logits, jnp.finfo(jnp.float32).min)

    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("hqk,khd->qhd", probs, v32)
