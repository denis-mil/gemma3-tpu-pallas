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

FLOP counts here are *hardware* counts, with the bf16 emulation pass count
folded in (ADR-0004). The chip has one roof, 197 TFLOP/s, and one ridge; asking
for more precision moves a kernel's intensity to the right rather than lowering
the roof underneath it. `passes * I` against `P` is the same inequality as `I`
against `P / passes`, which is why every crossover and every verdict below is
the number it was under the old convention.
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
#
# These are the *one-pass* intensities, and they are left at their registered
# values on purpose. `docs/measurements/0001` quotes this table as a prediction
# made before any hardware time; rescaling a registered number after the fact is
# what this workspace forbids. The pass count inverting means the name has to say
# which convention the numbers are in -- not that the numbers move.
REGISTERED_INTENSITY_1_PASS = {128: 63.2, 256: 124.9, 512: 244.0, 1024: 466.0, 2048: 855.1}

# Small enough to run in interpret mode in well under a second, and not Gemma's
# shape, so nothing in the harness can be shape-hardcoded.
SMALL = Gemma3Config(embed_dim=128, hidden_dim=256)


def _intensity(tokens, *, passes, block_t=None, cfg=GEMMA3_1B, elide_weights=False):
    block_t = tokens if block_t is None else block_t
    return arithmetic_intensity(
        mlp_flops(tokens, passes=passes, cfg=cfg),
        mlp_bytes(tokens, block_t=block_t, cfg=cfg, elide_weights=elide_weights),
    )


# --------------------------------------------------------------------------
# The arithmetic: FLOPs, bytes, and the registered predictions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("passes", [1, 3, 6])
def test_mlp_flops_requires_an_explicit_pass_count(passes):
    """Three matmuls of T x E x H, two FLOPs each, once per bf16 pass.

    No call site may inherit a pass count. "How many FLOPs is this matmul" has
    no dtype-free answer on a chip with no fp32 MXU: v5e emulates an
    fp32-precision multiply with 3 or 6 bf16 passes, so a default of 1 would
    understate a `HIGHEST` kernel's work by 6x. The compute roof, by contrast,
    *does* have a single published value, which is why `roofline_bound` gets a
    default and this does not.
    """
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    assert mlp_flops(8, passes=passes, cfg=cfg) == passes * 6 * 8 * 128 * 256

    with pytest.raises(TypeError):
        mlp_flops(8, cfg=cfg)
    with pytest.raises(ValueError):
        mlp_flops(8, passes=0, cfg=cfg)


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

    one_read = mlp_bytes(16, block_t=16, cfg=cfg)
    two_reads = mlp_bytes(16, block_t=8, cfg=cfg)

    assert one_read == weights + activations
    assert two_reads == 2 * weights + activations
    assert two_reads - one_read == weights


def test_elide_weights_recovers_the_single_read_count():
    """The counterfactual the DMA count adjudicates against."""
    cfg = Gemma3Config(embed_dim=128, hidden_dim=256)
    assert mlp_bytes(16, block_t=8, cfg=cfg, elide_weights=True) == mlp_bytes(
        16, block_t=16, cfg=cfg
    )


def test_registered_intensity_table():
    """The prediction, in a test rather than only in prose."""
    for tokens, expected in REGISTERED_INTENSITY_1_PASS.items():
        assert _intensity(tokens, passes=1) == pytest.approx(expected, abs=0.5)


def test_the_pass_count_lands_in_the_intensity():
    """The whole geometric content of ADR-0004, in one assertion.

    The pass count now multiplies the numerator of `flops / bytes`, so asking
    for `HIGHEST` slides a kernel six times to the right on the roofline. It
    moves not one byte -- which is exactly why intensity is a fact about a
    kernel *as executed*, precision included, and not about its algebra alone.
    """
    for tokens, one_pass in REGISTERED_INTENSITY_1_PASS.items():
        assert _intensity(tokens, passes=3) == pytest.approx(3 * one_pass, abs=1.5)
        assert _intensity(tokens, passes=6) == pytest.approx(6 * one_pass, abs=3.0)


@pytest.mark.parametrize(
    "passes, hbm_bound, compute_bound",
    [
        (1, (128, 256), (1024, 2048)),  # 512 sits on the ridge -- see the test below
        (3, (128,), (256, 512, 1024, 2048)),
        (6, (), (128, 256, 512, 1024, 2048)),
    ],
)
def test_crossovers_match_the_registered_table(passes, hbm_bound, compute_bound):
    """Which side of the ridge each token count falls on, per precision.

    Asserted against `V5E.ridge_point` rather than against literals, so a
    corrected hardware constant moves the test with it instead of leaving a
    stale number to be argued with.

    The data here is unchanged from when the pass count divided the roof, and
    that is the point: it is the same crossover at the same token count for the
    same precision, because `passes * I > P / beta` is the same inequality as
    `I > (P / passes) / beta`. Only which side of it scales has moved. What used
    to be three roofs with fixed points is now one roof with points that slide.

    The interesting rows are the ones the obvious reading gets wrong. `HIGH` (3
    passes) is *not* compute-bound everywhere: at T=128 the kernel sits 21%
    below the ridge, so it has a crossover of its own between 128 and 256 -- and
    that margin makes it a better test of the byte model than `DEFAULT`'s.
    """
    ridge = V5E.ridge_point
    for tokens in hbm_bound:
        assert _intensity(tokens, passes=passes) < ridge
    for tokens in compute_bound:
        assert _intensity(tokens, passes=passes) > ridge


def test_default_precision_ridge_is_unresolvable_at_512():
    """T=512 at DEFAULT is registered in advance as a point that cannot be scored.

    It lands within 2% of the ridge, which is inside the noise of any timing
    the notebook collects, so calling it either way after the fact would be
    reading the dice. Pinning it here means a later change that moves the ridge
    has to confront the claim rather than silently make it scoreable.
    """
    assert abs(_intensity(512, passes=1) / V5E.ridge_point - 1) < 0.02


def test_the_roof_is_a_single_published_constant():
    """v5e has one compute peak, and after ADR-0004 nothing divides it.

    `peak_flops` is the roofline vocabulary word -- `P` in `min(P, I * beta)` --
    and it is now an alias of the published bf16 field rather than a function of
    a pass count. The pass count is a property of the work, so it lives in the
    FLOP count; there was never a chip with a 32.8 TFLOP/s roof.
    """
    assert V5E.peak_flops == V5E.peak_bf16_flops == 197e12
    assert V5E.ridge_point == pytest.approx(V5E.peak_flops / V5E.peak_hbm_bandwidth)
    assert V5E.ridge_point == pytest.approx(240.5, abs=0.1)


def test_roofline_bound_defaults_to_the_published_peak():
    """Both roofs now have safe defaults, because both are published numbers."""
    assert roofline_bound(int(1e12), int(1e9)) == roofline_bound(
        int(1e12), int(1e9), peak_flops=V5E.peak_bf16_flops
    )


@pytest.mark.parametrize("tokens", sorted(REGISTERED_INTENSITY_1_PASS))
@pytest.mark.parametrize("passes", [1, 3, 6])
def test_moving_the_pass_count_changes_no_prediction(tokens, passes):
    """The anchor: inverting the convention is a change of variables, not of physics.

    Left side is the current convention -- the pass count multiplies the FLOPs
    and the roof is the one published peak. Right side is the convention this
    workspace used to hold, written out inline: algebraic FLOPs against a roof
    divided by the pass count. They agree to the bit, in seconds *and* in which
    roof binds, at every token count and every precision.

    That is why deleting `peak_flops(passes)` was safe, and it is why every
    verdict registered in `docs/measurements/0001` still reads true. The old
    divisor survives here, as a test fixture rather than as API.
    """
    moved = mlp_bytes(tokens, block_t=tokens)

    now = roofline_bound(mlp_flops(tokens, passes=passes), moved)
    before = roofline_bound(
        mlp_flops(tokens, passes=1), moved, peak_flops=197e12 / passes
    )

    assert now[0] == pytest.approx(before[0])
    assert now[1] == before[1]


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

    text = bench.summarise([at_ridge])
    assert "at ridge" in text
    assert "HBM-bound" not in text and "compute-bound" not in text

    text = bench.summarise([compute_bound])
    assert "compute-bound" in text

    # The same geometry at HIGHEST is six passes of bf16 work over the same
    # bytes, so I = 1464 -- nowhere near the ridge, and no longer unscoreable.
    # If the per-row pass count did not reach the verdict this would still say
    # "at ridge", so this is what pins the derivation end to end.
    text = bench.summarise([dataclasses.replace(at_ridge, precision="HIGHEST")])
    assert "at ridge" not in text
    assert "compute-bound" in text


def test_summarise_derives_the_pass_count_from_the_row():
    """One table, one roof, two precisions -- the count comes off each row.

    `summarise` used to take a table-level `passes`, which was a second copy of
    a fact every `BenchResult` already carries, and `summarise(highest_rows,
    passes=1)` printed a silently wrong table. Deriving it per row makes that
    unrepresentable, and it lets one table hold both precisions -- which is the
    story the single-roof plot now tells.
    """
    default = bench.BenchResult(
        label="kernel",
        tokens=2048,
        block_t=2048,
        block_h=768,
        precision="DEFAULT",
        vmem_limit_bytes=None,
        status="ok",
        times_s=(4e-3,),
        error=None,
    )
    highest = dataclasses.replace(default, precision="HIGHEST")

    text = bench.summarise([default, highest])
    rows = [line.split() for line in text.splitlines() if line.startswith("kernel")]
    assert len(rows) == 2
    default_row, highest_row = rows

    # prec, p, ms, GFLOP/s, I -- the last two scale by exactly the pass count.
    assert (default_row[4], default_row[5]) == ("DEFAULT", "1")
    assert (highest_row[4], highest_row[5]) == ("HIGHEST", "6")
    assert default_row[6] == highest_row[6]  # same milliseconds
    assert float(highest_row[7]) == pytest.approx(6 * float(default_row[7]), rel=1e-3)
    assert float(highest_row[8]) == pytest.approx(6 * float(default_row[8]), rel=1e-3)

    # One roof, named once. 32.8 was the six-pass roof of the old convention.
    assert "197.0" in text
    assert "32.8" not in text


def test_summarise_compares_kernel_to_xla_within_a_precision():
    """A kernel row is only comparable to the XLA row at the same precision.

    One table can now hold both, so the ratio block is keyed on (precision,
    tokens). Keyed on tokens alone -- which is what it was when each precision
    got its own table -- HIGHEST would overwrite DEFAULT and one set of ratios
    would be printed as though it were both.
    """
    kernel = bench.BenchResult(
        label="kernel",
        tokens=2048,
        block_t=2048,
        block_h=768,
        precision="DEFAULT",
        vmem_limit_bytes=None,
        status="ok",
        times_s=(4e-3,),
        error=None,
    )
    xla = dataclasses.replace(kernel, label="xla", block_t=None, block_h=None)
    rows = [
        kernel,
        xla,
        dataclasses.replace(kernel, precision="HIGHEST", times_s=(8e-3,)),
        dataclasses.replace(xla, precision="HIGHEST", times_s=(4e-3,)),
    ]

    ratios = [
        line.split() for line in bench.summarise(rows).splitlines() if "x" == line[-1:]
    ]
    assert ratios == [
        ["DEFAULT", "T=2048", "1.00x"],
        ["HIGHEST", "T=2048", "0.50x"],
    ]


def test_summarise_rejects_a_timed_row_whose_precision_has_no_pass_count():
    """A `DotAlgorithmPreset` label is deliberately unmapped, so it must raise.

    A row that actually timed and whose pass count is unknown cannot be scored
    against any roof -- inventing 1 for it is the failure this whole change is
    about. Rows that *failed* print `-` everywhere and never ask.
    """
    timed = bench.BenchResult(
        label="kernel",
        tokens=512,
        block_t=512,
        block_h=768,
        precision="BF16_BF16_F32_X3",
        vmem_limit_bytes=None,
        status="ok",
        times_s=(1e-3,),
        error=None,
    )
    with pytest.raises(ValueError, match="BF16_BF16_F32_X3"):
        bench.summarise([timed])

    failed = dataclasses.replace(timed, status="failed", times_s=(), error="nope")
    assert "FAILED" in bench.summarise([failed])
