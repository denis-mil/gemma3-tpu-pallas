"""Gemma 3 1B architecture constants and TPU v5e hardware constants.

Every benchmark and test in this repo draws its shapes from here. Nothing
hardcodes a dimension inline -- if a number appears in a plot, it traces back
to this module.

Model values are taken from the published `gemma-3-1b-it` config.json.
Hardware values are TPU v5e per-chip spec.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Gemma3Config:
    """Architecture of Gemma 3 1B."""

    num_layers: int = 26
    embed_dim: int = 1152
    hidden_dim: int = 6912  # MLP intermediate
    num_heads: int = 4  # query heads
    num_kv_heads: int = 1  # MQA: all query heads share one KV head
    head_dim: int = 256
    sliding_window: int = 512
    sliding_window_pattern: int = 6  # every 6th layer is global
    max_position: int = 32768
    rope_theta_global: float = 1_000_000.0
    rope_theta_local: float = 10_000.0
    vocab_size: int = 262_144
    query_pre_attn_scalar: int = 256

    def is_local(self, layer_idx: int) -> bool:
        """True when `layer_idx` uses sliding-window (local) attention.

        Global layers are those where (idx + 1) is divisible by the pattern,
        i.e. layers 5, 11, 17 and 23 for the 26-layer 1B model.
        """
        return bool((layer_idx + 1) % self.sliding_window_pattern)

    @property
    def num_local_layers(self) -> int:
        return sum(self.is_local(i) for i in range(self.num_layers))

    @property
    def num_global_layers(self) -> int:
        return self.num_layers - self.num_local_layers

    @property
    def attn_scale(self) -> float:
        """Softmax scale applied to Q before QK^T."""
        return self.query_pre_attn_scalar**-0.5

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """KV cache bytes for one token in one layer (K and V together)."""
        return 2 * self.num_kv_heads * self.head_dim * dtype_bytes

    def kv_cache_bytes(
        self, seq_len: int, batch: int = 1, *, windowed: bool = True, dtype_bytes: int = 2
    ) -> int:
        """Total KV cache bytes across all layers.

        With `windowed=True`, local layers only retain `sliding_window` tokens --
        which is the whole reason a long-context batch fits in 16 GB at all.
        """
        per_token = self.kv_bytes_per_token(dtype_bytes)
        local_len = min(seq_len, self.sliding_window) if windowed else seq_len
        local = self.num_local_layers * local_len * per_token
        global_ = self.num_global_layers * seq_len * per_token
        return batch * (local + global_)


@dataclass(frozen=True)
class TpuV5e:
    """Per-chip TPU v5e hardware limits, used for roofline analysis."""

    peak_bf16_flops: float = 197e12
    peak_hbm_bandwidth: float = 819e9
    hbm_bytes: int = 16 * 1024**3
    mxu_dim: int = 128  # v6e/Trillium is 256 -- block sizes do not port

    def peak_flops(self, passes: int = 1) -> float:
        """Effective compute roof when each multiply costs `passes` bf16 passes.

        There is no published fp32 peak for v5e, and `tpu_info` does not carry
        one. What TPU does instead is emulate an fp32 matmul with several bf16
        passes -- `jax.lax.DotAlgorithmPreset` names `BF16_BF16_F32_X3` and
        `_X6` -- so the fp32 roof is the bf16 roof divided by the pass count.

        `passes=1` is `precision=DEFAULT` (bf16 multiplies, fp32 accumulate),
        3 is HIGH, 6 is HIGHEST. Which one a given kernel actually gets is a
        measurement, not a constant; that is why this is a parameter and not a
        `peak_fp32_flops` field.
        """
        if passes < 1:
            raise ValueError(f"passes {passes} must be at least 1")
        return self.peak_bf16_flops / passes

    def ridge_point(self, passes: int = 1) -> float:
        """Arithmetic intensity (FLOP/byte) where the two roofs cross.

        Below it a kernel is bandwidth-bound, above it compute-bound. Dividing
        the compute roof by `passes` *lowers* the ridge, so asking for more
        precision makes a kernel compute-bound at smaller sizes.
        """
        return self.peak_flops(passes) / self.peak_hbm_bandwidth


GEMMA3_1B = Gemma3Config()
V5E = TpuV5e()


def attention_flops(
    seq_len: int,
    *,
    cfg: Gemma3Config = GEMMA3_1B,
    windowed: bool = False,
    causal: bool = True,
) -> int:
    """FLOPs for one attention layer's prefill over `seq_len` tokens.

    QK^T and PV each cost 2*N*S*D per query head. For the windowed case each
    query attends to at most `sliding_window` keys, so S collapses from N to W --
    this is the asymptotic win the artifact is built on.
    """
    n = seq_len
    span = min(seq_len, cfg.sliding_window) if windowed else seq_len
    pairs = n * span
    if causal and not windowed:
        pairs //= 2
    return 4 * pairs * cfg.head_dim * cfg.num_heads


def mlp_flops(tokens: int, *, cfg: Gemma3Config = GEMMA3_1B) -> int:
    """FLOPs for one gated-MLP layer over `tokens` tokens.

    Three matmuls -- gate, up, down -- each `tokens x embed_dim x hidden_dim`
    at 2 FLOPs per multiply-accumulate. The GELU and the elementwise gate are
    `O(tokens * hidden_dim)` and are dropped, as the roofline convention does.
    """
    return 6 * tokens * cfg.embed_dim * cfg.hidden_dim


def mlp_bytes(
    tokens: int,
    *,
    block_t: int,
    cfg: Gemma3Config = GEMMA3_1B,
    dtype_bytes: int = 4,
    elide_weights: bool = False,
) -> int:
    """HBM traffic for `fused_gated_mlp` at this geometry.

    The weights term is the load-bearing one. The grid is
    `(tokens // block_t, hidden_dim // block_h)` with the hidden axis
    **innermost**, and the `w_*` `index_map`s vary in the inner index -- so one
    `t` step walks every hidden block, and when the inner index resets for the
    next `t` step, block 0 is no longer the previously-fetched slice. Copy
    elision only skips a transfer between *consecutive* identical slices, so it
    cannot apply across that reset: the weights are re-read once per `t` step,
    `tokens // block_t` times in total, not once.

    `elide_weights=True` gives the counterfactual where they are read once. The
    two counts differ by a factor of `tokens // block_t`, which is what an xprof
    DMA count adjudicates between -- neither is measured yet.

    Activations are `x` in and the output out, `tokens * embed_dim` each, read
    and written exactly once regardless of blocking.
    """
    if tokens % block_t:
        raise ValueError(f"tokens {tokens} is not divisible by block_t {block_t}")
    passes = 1 if elide_weights else tokens // block_t
    weights = 3 * cfg.embed_dim * cfg.hidden_dim * dtype_bytes
    activations = 2 * tokens * cfg.embed_dim * dtype_bytes
    return passes * weights + activations


def arithmetic_intensity(flops: int, bytes_moved: int) -> float:
    """FLOPs per byte of HBM traffic -- the x-axis of a roofline plot."""
    if bytes_moved <= 0:
        raise ValueError(f"bytes_moved {bytes_moved} must be positive")
    return flops / bytes_moved


def roofline_bound(
    flops: int,
    bytes_moved: int,
    *,
    peak_flops: float,
    bandwidth: float = V5E.peak_hbm_bandwidth,
) -> tuple[float, str]:
    """Return (best achievable seconds, which roof binds).

    Compares the compute roof against the bandwidth roof so a benchmark can say
    *why* it is slow, not merely that it is.

    `peak_flops` is required and has no default. v5e publishes one peak, for
    bf16; an fp32-typed kernel runs against that roof divided by however many
    bf16 passes the requested precision costs, so there is no single number a
    call site could safely inherit. Use `TpuV5e.peak_flops(passes)`. The
    bandwidth roof does have one published value, so it keeps its default.
    """
    compute_s = flops / peak_flops
    memory_s = bytes_moved / bandwidth
    if compute_s >= memory_s:
        return compute_s, "compute"
    return memory_s, "memory"
