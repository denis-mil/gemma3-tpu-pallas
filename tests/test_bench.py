"""The benchmark harness and the roofline arithmetic it reports against.

Two kinds of test live here. The first kind pins the *predictions* registered
before the Colab run -- the intensity table, the ridge points, and which side of
its ridge each token count falls on. Those are claims about hardware that this
machine cannot check; putting them in a test means a later change to the byte
model has to confront them rather than quietly move them.

The second kind exercises the harness itself under interpret mode, so that no
typo in `bench.py` is discovered on a TPU runtime that costs something.

The byte model under test is the corrected one: with the hidden axis innermost,
the weights are re-read once per `t` step, because copy elision only skips a
transfer between *consecutive* identical slices and the reset of the inner index
breaks that. `elide_weights=True` is the counterfactual, and the xprof DMA count
is what adjudicates -- neither is measured yet.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gemma3_pallas import bench
from gemma3_pallas.mlp import fused_gated_mlp
from gemma3_pallas.reference import gated_mlp
from gemma3_pallas.shapes import (
    GEMMA3_1B,
    V5E,
    Gemma3Config,
    arithmetic_intensity,
    mlp_bytes,
    mlp_flops,
    roofline_bound,
)

# The five token counts the notebook sweeps, and the intensity each one has when
# the whole sequence is one `t` step (`block_t == tokens`, so the weights are
# read once). Registered before the run; ±0.5 FLOP/byte.
REGISTERED_INTENSITY = {128: 63.2, 256: 124.9, 512: 244.0, 1024: 466.0, 2048: 855.1}

# Small enough to run in interpret mode in well under a second, and not Gemma's
# shape, so nothing in the harness can be shape-hardcoded.
SMALL = Gemma3Config(embed_dim=128, hidden_dim=256)


def _intensity(tokens, *, block_t=None, cfg=GEMMA3_1B, elide_weights=False):
    block_t = tokens if block_t is None else block_t
    return arithmetic_intensity(
        mlp_flops(tokens, cfg=cfg),
        mlp_bytes(tokens, block_t=block_t, cfg=cfg, elide_weights=elide_weights),
    )


# --------------------------------------------------------------------------
# The arithmetic: FLOPs, bytes, and the registered predictions
# --------------------------------------------------------------------------


def test_mlp_flops_matches_a_hand_count():
    """Three matmuls of T x E x H, two FLOPs each: 6*T*E*H, counted by hand."""
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    assert mlp_flops(8, cfg=cfg) == 6 * 8 * 128 * 256


def test_mlp_bytes_counts_weights_activations_and_output():
    """Every operand is in the count -- the workspace's own rule about intensity."""
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    weights = 3 * 128 * 256 * 4
    activations = 2 * 8 * 128 * 4  # x in, output out
    assert mlp_bytes(8, block_t=8, cfg=cfg) == weights + activations


def test_mlp_bytes_doubles_the_weights_when_block_t_halves_tokens():
    """The correction: weights are re-read once per `t` step, not once.

    The `w_*` index_maps vary in the innermost index, so one `t` step walks all
    of them; when that index resets, block 0 is no longer the previously fetched
    slice and copy elision cannot apply. Halving `block_t` therefore doubles the
    weight traffic while leaving the activation traffic alone.
    """
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    weights = 3 * 128 * 256 * 4
    activations = 2 * 16 * 128 * 4

    one_pass = mlp_bytes(16, block_t=16, cfg=cfg)
    two_passes = mlp_bytes(16, block_t=8, cfg=cfg)

    assert one_pass == weights + activations
    assert two_passes == 2 * weights + activations
    assert two_passes - one_pass == weights


def test_elide_weights_recovers_the_single_pass_count():
    """The counterfactual the DMA count adjudicates against."""
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    assert mlp_bytes(16, block_t=8, cfg=cfg, elide_weights=True) == mlp_bytes(
        16, block_t=16, cfg=cfg
    )


def test_registered_intensity_table():
    """The prediction, in a test rather than only in prose."""
    for tokens, expected in REGISTERED_INTENSITY.items():
        assert _intensity(tokens) == pytest.approx(expected, abs=0.5)


@pytest.mark.parametrize(
    "passes, hbm_bound, compute_bound",
    [
        (1, (128, 256), (1024, 2048)),  # 512 sits on the ridge -- see the test below
        (3, (128,), (256, 512, 1024, 2048)),
        (6, (), (128, 256, 512, 1024, 2048)),
    ],
)
def test_crossovers_match_the_registered_table(passes, hbm_bound, compute_bound):
    """Which side of its ridge each token count falls on, per precision.

    Asserted against `ridge_point(passes)` rather than against literals, so a
    corrected hardware constant moves the test with it instead of leaving a
    stale number to be argued with.

    The interesting rows are the ones the obvious reading gets wrong. `HIGH` (3
    passes) is *not* compute-bound everywhere: at T=128 the kernel sits 21%
    below its ridge, so it has a crossover of its own between 128 and 256 -- and
    that margin makes it a better test of the byte model than `DEFAULT`'s.
    """
    ridge = V5E.ridge_point(passes)
    for tokens in hbm_bound:
        assert _intensity(tokens) < ridge
    for tokens in compute_bound:
        assert _intensity(tokens) > ridge


def test_default_precision_ridge_is_unresolvable_at_512():
    """T=512 at DEFAULT is registered in advance as a point that cannot be scored.

    It lands within 2% of the ridge, which is inside the noise of any timing
    the notebook collects, so calling it either way after the fact would be
    reading the dice. Pinning it here means a later change that moves the ridge
    has to confront the claim rather than silently make it scoreable.
    """
    assert abs(_intensity(512) / V5E.ridge_point(1) - 1) < 0.02


def test_roofline_bound_requires_an_explicit_peak():
    """No call site may inherit a compute roof: v5e publishes only a bf16 one."""
    with pytest.raises(TypeError):
        roofline_bound(flops=int(1e12), bytes_moved=int(1e9))


def test_peak_flops_divides_by_passes():
    """fp32 on TPU is emulated with 3 or 6 bf16 passes, so the roof divides."""
    assert V5E.peak_flops(1) == V5E.peak_bf16_flops
    assert V5E.peak_flops(3) == pytest.approx(V5E.peak_bf16_flops / 3)
    assert V5E.peak_flops(6) == pytest.approx(V5E.peak_bf16_flops / 6)
    with pytest.raises(ValueError):
        V5E.peak_flops(0)


def test_ridge_point_tracks_the_peak():
    """More precision lowers the ridge, so a kernel goes compute-bound sooner."""
    for passes in (1, 3, 6):
        assert V5E.ridge_point(passes) == pytest.approx(
            V5E.peak_flops(passes) / V5E.peak_hbm_bandwidth
        )
    assert V5E.ridge_point(1) > V5E.ridge_point(3) > V5E.ridge_point(6)


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------


def test_require_tpu_raises_here():
    """Pairs with test_pallas_smoke's 'if this fails we grew a TPU'."""
    with pytest.raises(RuntimeError, match="cpu"):
        bench.require_tpu()


def test_time_call_discards_warmup_and_returns_repeats():
    calls = []

    def fn(x):
        calls.append(x)
        return jnp.asarray(x) * 2

    times = bench.time_call(fn, 1.0, warmup=3, repeats=5)

    assert len(calls) == 8  # 3 warmup + 5 timed
    assert len(times) == 5
    assert all(t > 0 for t in times)


def test_bench_kernel_under_interpret_matches_the_reference():
    """The harness times the same computation the correctness tests assert on."""
    result = bench.bench_kernel(
        16, 8, 128, precision=None, interpret=True, cfg=SMALL, warmup=1, repeats=3
    )
    assert result.status == "ok"
    assert result.label == "kernel"
    assert len(result.times_s) == 3
    assert result.median() > 0

    x, w_gate, w_up, w_down = bench.operands(16, cfg=SMALL)
    got = fused_gated_mlp(x, w_gate, w_up, w_down, block_t=8, block_h=128, interpret=True)
    want = gated_mlp(x, w_gate, w_up, w_down)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_bench_kernel_records_a_bad_config_instead_of_raising():
    """A config that will not compile is a data point, not an error.

    This is how the VMEM sweep measures the budget: it runs configs until they
    stop compiling, and the boundary is the measurement. So `bench_kernel`
    catches, records the message verbatim, and lets the sweep continue.
    """
    result = bench.bench_kernel(
        16, 8, 192, precision=None, interpret=True, cfg=SMALL, warmup=1, repeats=1
    )
    assert result.status == "failed"
    assert result.times_s == ()
    assert result.median() is None
    assert "hidden_dim" in result.error


def test_bench_kernel_lets_keyboard_interrupt_through():
    """Catching broadly must not make a wedged sweep unstoppable."""

    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bench, "time_call", boom)
        with pytest.raises(KeyboardInterrupt):
            bench.bench_kernel(
                16, 8, 128, precision=None, interpret=True, cfg=SMALL, repeats=1
            )


def test_sweep_saves_after_every_config(tmp_path):
    """`on_result` fires per config, so a disconnect keeps what was finished."""
    path = tmp_path / "sweep.json"
    calls, seen = [], []

    def on_result(result):
        calls.append(result)
        if len(calls) == 3:  # the runtime dies before this one is persisted
            raise RuntimeError("runtime disconnected")
        seen.append(result)
        bench.save(seen, path)

    specs = [
        dict(tokens=16, block_t=8, block_h=128, precision=None, cfg=SMALL, warmup=1, repeats=1)
        for _ in range(4)
    ]
    with pytest.raises(RuntimeError, match="disconnected"):
        bench.sweep(specs, interpret=True, on_result=on_result)

    # The callback fired once per config, and the two that completed before the
    # third one died are on disk -- which is the whole point of the hook.
    assert len(calls) == 3
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2


def test_sweep_dispatches_xla_specs_to_the_baseline():
    results = bench.sweep(
        [
            dict(label="xla", tokens=16, precision=None, cfg=SMALL, warmup=1, repeats=1),
            dict(
                tokens=16,
                block_t=8,
                block_h=128,
                precision=None,
                cfg=SMALL,
                warmup=1,
                repeats=1,
            ),
        ],
        interpret=True,
    )
    assert [r.label for r in results] == ["xla", "kernel"]
    assert results[0].block_t is None and results[0].status == "ok"


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "results.json"
    written = bench.sweep(
        [
            dict(
                tokens=16,
                block_t=8,
                block_h=128,
                precision=None,
                cfg=SMALL,
                warmup=1,
                repeats=2,
            )
        ],
        interpret=True,
    )
    bench.save(written, path)
    assert bench.load(path) == written


@pytest.mark.parametrize("precision", [None, "highest", jax.lax.Precision.HIGH])
def test_precision_threads_through_the_kernel(precision):
    """All three precisions still match the reference; the plumbing is the test.

    On CPU these are near no-ops -- the divergence they cause is a TPU fact, and
    the notebook is what measures it. What is under test here is that
    `precision` reaches all three `jnp.dot` calls.

    It is *not* evidence that a precision lowers on hardware. `interpret=True`
    never reaches Mosaic, and `HIGH` passing here is exactly what made the v5e
    notebook's first run fail with `NotImplementedError: Unsupported dot
    precision: HIGH` -- jax 0.11.0's Mosaic `dot_general` rule has branches for
    `DEFAULT` and `HIGHEST` only. Whether a precision is lowerable is a
    hardware-path question and only a TPU can answer it.
    """
    x, w_gate, w_up, w_down = bench.operands(16, cfg=SMALL)
    got = fused_gated_mlp(
        x, w_gate, w_up, w_down, block_t=8, block_h=128, interpret=True, precision=precision
    )
    want = gated_mlp(x, w_gate, w_up, w_down)
    assert np.allclose(np.asarray(got), np.asarray(want), rtol=1e-4, atol=1e-4)


def test_summarise_names_the_roof_and_flags_the_ridge():
    """The summary must not score a point that sits inside the noise of its ridge."""
    at_ridge = bench.BenchResult(
        label="kernel",
        tokens=512,
        block_t=512,
        block_h=768,
        precision="DEFAULT",
        vmem_limit_bytes=None,
        status="ok",
        times_s=(1e-3,),
        error=None,
    )
    compute_bound = dataclasses.replace(
        at_ridge, tokens=2048, block_t=2048, times_s=(4e-3,)
    )

    text = bench.summarise([at_ridge], passes=1)
    assert "at ridge" in text
    assert "HBM-bound" not in text and "compute-bound" not in text

    text = bench.summarise([compute_bound], passes=1)
    assert "compute-bound" in text
