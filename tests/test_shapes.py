"""The architecture and memory arithmetic the whole project rests on.

These numbers are quoted in the README and drive every design decision, so they
are tested rather than trusted.
"""

import pytest

from gemma3_pallas.shapes import GEMMA3_1B, V5E, attention_flops, roofline_bound

GiB = 1024**3


def test_local_global_layer_split():
    """Pattern 6 over 26 layers puts global attention at 5, 11, 17, 23."""
    globals_ = [i for i in range(GEMMA3_1B.num_layers) if not GEMMA3_1B.is_local(i)]
    assert globals_ == [5, 11, 17, 23]
    assert GEMMA3_1B.num_local_layers == 22
    assert GEMMA3_1B.num_global_layers == 4


def test_kv_bytes_per_token():
    """MQA with one KV head makes the per-token cache tiny: 1 KB per layer."""
    assert GEMMA3_1B.kv_bytes_per_token() == 1024


def test_windowed_kv_cache_is_roughly_6x_smaller_at_32k():
    """Exact byte counts -- decimal vs binary 'MB' is too easy to get wrong."""
    dense = GEMMA3_1B.kv_cache_bytes(32768, windowed=False)
    windowed = GEMMA3_1B.kv_cache_bytes(32768, windowed=True)

    assert dense == 26 * 32768 * 1024  # 872,415,232 B == 872.4 MB == 832 MiB
    # 22 local layers capped at the 512 window, 4 global layers holding all 32K.
    assert windowed == 22 * 512 * 1024 + 4 * 32768 * 1024  # 145,752,064 B == 145.8 MB
    assert dense / windowed == pytest.approx(5.99, rel=0.01)


def test_windowing_is_what_makes_batch_32_fit_in_hbm():
    """The headline claim: at 32K x batch 32, dense KV alone exceeds 16 GiB."""
    dense = GEMMA3_1B.kv_cache_bytes(32768, batch=32, windowed=False)
    windowed = GEMMA3_1B.kv_cache_bytes(32768, batch=32, windowed=True)
    weights = 2 * GiB  # bf16

    assert dense > V5E.hbm_bytes  # 27.9 GB, does not fit
    assert windowed + weights < V5E.hbm_bytes  # 4.66 GB + 2 GiB, fits
    assert windowed == pytest.approx(4.66e9, rel=0.01)


def test_windowed_attention_does_far_less_work_at_long_context():
    """At a 32K prompt a local layer needs ~1.6% of dense attention's pairs."""
    dense = attention_flops(32768, windowed=False, causal=False)
    windowed = attention_flops(32768, windowed=True)
    assert windowed / dense == pytest.approx(512 / 32768, rel=1e-6)


def test_short_sequences_get_no_windowing_benefit():
    """Below the window there is nothing to skip -- the kernel must not 'win' here."""
    for seq in (128, 512):
        assert attention_flops(seq, windowed=True) == attention_flops(
            seq, windowed=False, causal=False
        )


def test_windowed_attention_scales_with_the_pass_count():
    """The same factor `mlp_flops` applies, for the same reason (ADR-0004)."""
    assert attention_flops(1024, passes=6) == 6 * attention_flops(1024, passes=1)


def test_the_windowing_ratio_does_not_depend_on_the_pass_count():
    """Why `attention_flops` may default `passes=1` where `mlp_flops` may not.

    Every consumer of this function is a *ratio* claim -- windowed against dense
    -- and a common factor cancels out of a ratio. So the pass count cannot
    change any conclusion drawn from it, and the material it feeds is bf16
    throughout, where 1 is the physical truth rather than a convenience.
    `mlp_flops` feeds an achieved-throughput headline, where a wrong pass count
    is a wrong number, so that one stays required.
    """
    for passes in (1, 3, 6):
        dense = attention_flops(32768, passes=passes, windowed=False, causal=False)
        windowed = attention_flops(32768, passes=passes, windowed=True)
        assert windowed / dense == pytest.approx(512 / 32768, rel=1e-6)


def test_roofline_picks_the_binding_constraint():
    """The compute roof is inherited here: v5e publishes exactly one (ADR-0004).

    This is the same test that used to pass `peak_flops` explicitly on the
    grounds that no default was safe. What was un-inheritable was never the
    roof -- it was the bf16 pass count, and that argument moved with it to
    `mlp_flops`.
    """
    compute_bound, which = roofline_bound(flops=int(1e15), bytes_moved=1)
    assert which == "compute"
    memory_bound, which = roofline_bound(flops=1, bytes_moved=int(1e12))
    assert which == "memory"
    assert compute_bound > 0 and memory_bound > 0
