"""Benchmark harness for the Pallas kernels, exercised on CPU before it runs on a TPU.

The measurement logic lives here rather than in the notebook so that it can be
tested under interpret mode on a machine with no accelerator. ADR-0003's rule --
scarce hardware absorbs no typos -- applies to the instrument as much as to the
kernel: everything below is green locally before a Colab runtime is opened, and
the notebook is a thin driver that calls into it.

Three properties are deliberate and load-bearing:

* **`bench_kernel` does not raise on a bad configuration.** A Mosaic VMEM
  failure is the *measurement*: it is how a sweep finds the budget a
  `pallas_call` actually gets. Such a config is recorded with `status="failed"`
  and the compiler's message, verbatim, and the sweep continues.
* **`sweep` calls `on_result` after every single config**, so a driver can
  persist incrementally and a runtime that dies mid-sweep still leaves
  everything up to that point on disk.
* **`interpret` has no default in this module**, and `require_tpu` is called by
  the driver rather than buried in a benchmark -- so the same code paths run
  under emulation locally and on hardware remotely.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from .mlp import fused_gated_mlp
from .reference import gated_mlp
from .shapes import (
    GEMMA3_1B,
    V5E,
    Gemma3Config,
    TpuV5e,
    arithmetic_intensity,
    mlp_bytes,
    mlp_flops,
)

# A measured intensity this close to the ridge cannot be called for either roof
# by any timing this harness collects, so `summarise` refuses to call it and
# says "at ridge" instead. `DEFAULT` precision at T=512 is the known case: it
# lands ~1.4% above its ridge, which is inside the noise.
RIDGE_TOLERANCE = 0.02


@dataclass(frozen=True)
class BenchResult:
    """One timed configuration, or one that would not compile.

    `times_s` holds the full distribution rather than a summary: a bimodal set
    of samples is a fact about the run, and a mean would hide it.
    """

    label: str  # "kernel" | "xla"
    tokens: int
    block_t: int | None
    block_h: int | None
    precision: str
    vmem_limit_bytes: int | None
    status: str  # "ok" | "failed"
    times_s: tuple[float, ...]
    error: str | None  # the compiler message, verbatim, when status == "failed"

    def median(self) -> float | None:
        """Median seconds per call, or None if nothing was timed."""
        if not self.times_s:
            return None
        ordered = sorted(self.times_s)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["times_s"] = list(self.times_s)
        d["median_s"] = self.median()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> BenchResult:
        d = dict(d)
        d.pop("median_s", None)
        d["times_s"] = tuple(d.get("times_s") or ())
        return cls(**d)


def require_tpu() -> None:
    """Raise unless this process is actually talking to a TPU.

    Called by the driver, first thing, so that a runtime which silently fell
    back to CPU cannot produce a number that looks publishable.
    """
    platform = jax.devices()[0].platform
    if platform != "tpu":
        raise RuntimeError(
            f"expected a TPU backend, got {platform!r} "
            f"({[str(d) for d in jax.devices()]}). No timing here is publishable."
        )


def time_call(fn: Callable, *args, warmup: int = 3, repeats: int = 20) -> tuple[float, ...]:
    """Time `fn(*args)` `repeats` times after `warmup` untimed calls.

    The warmup absorbs compilation, which on a first call is orders of
    magnitude larger than the thing being measured. `block_until_ready` on
    every call is what makes the numbers wall-clock rather than dispatch-queue.
    """
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return tuple(times)


def operands(
    tokens: int, *, cfg: Gemma3Config = GEMMA3_1B, seed: int = 0
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Random fp32 operands at Gemma's MLP geometry, scaled like real weights.

    Same construction as `tests/test_mlp_kernel.py::_operands`, so a correctness
    check and a timing run are looking at the same numbers.
    """
    keys = jax.random.split(jax.random.key(seed), 4)
    x = jax.random.normal(keys[0], (tokens, cfg.embed_dim), jnp.float32)
    w_gate = jax.random.normal(keys[1], (cfg.embed_dim, cfg.hidden_dim), jnp.float32) * 0.02
    w_up = jax.random.normal(keys[2], (cfg.embed_dim, cfg.hidden_dim), jnp.float32) * 0.02
    w_down = jax.random.normal(keys[3], (cfg.hidden_dim, cfg.embed_dim), jnp.float32) * 0.02
    return x, w_gate, w_up, w_down


def precision_label(precision) -> str:
    """A short stable string for a `PrecisionLike`, for tables and JSON."""
    if precision is None:
        return "DEFAULT"
    if isinstance(precision, str):
        return precision.upper()
    return getattr(precision, "name", str(precision))


def _compiler_params(vmem_limit_bytes: int | None):
    """`pltpu.CompilerParams(vmem_limit_bytes=...)`, or None if not asked for.

    Imported lazily: the module is importable on CPU, but there is no reason to
    require it of a caller who never sets a VMEM limit.
    """
    if vmem_limit_bytes is None:
        return None
    from jax.experimental.pallas import tpu as pltpu

    return pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes)


def bench_kernel(
    tokens: int,
    block_t: int,
    block_h: int,
    *,
    precision,
    interpret,
    vmem_limit_bytes: int | None = None,
    cfg: Gemma3Config = GEMMA3_1B,
    seed: int = 0,
    warmup: int = 3,
    repeats: int = 20,
) -> BenchResult:
    """Time `fused_gated_mlp` at one geometry. Never raises for a bad config.

    A config that will not compile -- a ragged block, or a working set past the
    VMEM budget -- comes back as `status="failed"` carrying the compiler's own
    message. That is a data point, not an error: the boundary between the
    configs that compile and the ones that do not is precisely what a VMEM
    sweep is measuring. `KeyboardInterrupt` is *not* swallowed, so a wedged
    sweep can still be stopped.
    """
    result = dict(
        label="kernel",
        tokens=tokens,
        block_t=block_t,
        block_h=block_h,
        precision=precision_label(precision),
        vmem_limit_bytes=vmem_limit_bytes,
    )
    try:
        x, w_gate, w_up, w_down = operands(tokens, cfg=cfg, seed=seed)
        compiler_params = _compiler_params(vmem_limit_bytes)

        def run(x, w_gate, w_up, w_down):
            return fused_gated_mlp(
                x,
                w_gate,
                w_up,
                w_down,
                block_t=block_t,
                block_h=block_h,
                interpret=interpret,
                precision=precision,
                compiler_params=compiler_params,
            )

        fn = jax.jit(run)
        times = time_call(fn, x, w_gate, w_up, w_down, warmup=warmup, repeats=repeats)
    except Exception as exc:  # noqa: BLE001 -- a failure to compile is the measurement
        return BenchResult(**result, status="failed", times_s=(), error=f"{type(exc).__name__}: {exc}")
    return BenchResult(**result, status="ok", times_s=times, error=None)


def bench_xla(
    tokens: int,
    *,
    precision,
    cfg: Gemma3Config = GEMMA3_1B,
    seed: int = 0,
    warmup: int = 3,
    repeats: int = 20,
) -> BenchResult:
    """Time `jax.jit(reference.gated_mlp)` at the same geometry, as the baseline.

    `reference.gated_mlp` writes `@`, which carries no precision, so the
    precision is supplied by `jax.default_matmul_precision` around the trace --
    the dots are staged out under that context and inherit it. This is the
    number the kernel is compared against; the reference is not changed to
    accommodate the benchmark.
    """
    result = dict(
        label="xla",
        tokens=tokens,
        block_t=None,
        block_h=None,
        precision=precision_label(precision),
        vmem_limit_bytes=None,
    )
    try:
        x, w_gate, w_up, w_down = operands(tokens, cfg=cfg, seed=seed)

        def run(x, w_gate, w_up, w_down):
            if precision is None:
                return gated_mlp(x, w_gate, w_up, w_down)
            with jax.default_matmul_precision(precision):
                return gated_mlp(x, w_gate, w_up, w_down)

        fn = jax.jit(run)
        times = time_call(fn, x, w_gate, w_up, w_down, warmup=warmup, repeats=repeats)
    except Exception as exc:  # noqa: BLE001 -- symmetric with bench_kernel
        return BenchResult(**result, status="failed", times_s=(), error=f"{type(exc).__name__}: {exc}")
    return BenchResult(**result, status="ok", times_s=times, error=None)


def sweep(
    specs: Iterable[dict],
    *,
    interpret,
    on_result: Callable[[BenchResult], None] | None = None,
) -> list[BenchResult]:
    """Run each spec in order, calling `on_result` after every one.

    A spec is the keyword arguments for one benchmark, plus an optional
    `"label"` selecting `bench_xla` over `bench_kernel`. `on_result` fires
    *after each config rather than at the end* so that a driver can save
    incrementally -- a Colab runtime that disconnects mid-sweep then still
    leaves every config completed so far on disk.
    """
    results: list[BenchResult] = []
    for spec in specs:
        spec = dict(spec)
        label = spec.pop("label", "kernel")
        if label == "xla":
            result = bench_xla(**spec)
        else:
            result = bench_kernel(interpret=interpret, **spec)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def save(results: Sequence[BenchResult], path) -> None:
    """Write the whole result list as one JSON array.

    Rewrites the file from scratch every time, which makes it safe to call
    after every config: there is no append state to get out of step, and a
    process killed mid-write loses at most the configs since the last call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
    )


def load(path) -> list[BenchResult]:
    """Read back what `save` wrote."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchResult.from_dict(d) for d in raw]


def summarise(
    results: Sequence[BenchResult],
    *,
    passes: int,
    cfg: Gemma3Config = GEMMA3_1B,
    hw: TpuV5e = V5E,
    elide_weights: bool = False,
) -> str:
    """A text table: achieved GFLOP/s against the roof that binds at `passes`.

    `passes` is the number of bf16 passes the measured precision costs -- it
    selects the compute roof, and there is no default because inventing one is
    the mistake `roofline_bound`'s required keyword exists to prevent.

    A row whose intensity is within `RIDGE_TOLERANCE` of the ridge prints as
    **at ridge** rather than as a verdict: the two roofs are within the noise
    of each other there, so the measurement cannot distinguish them.

    XLA rows get a GFLOP/s but no intensity: the baseline materialises both
    `[tokens, hidden_dim]` intermediates in HBM, so `mlp_bytes` -- which is a
    model of *this kernel's* traffic -- does not describe it.
    """
    peak = hw.peak_flops(passes)
    ridge = hw.ridge_point(passes)

    header = (
        f"roof: {peak / 1e12:.1f} TFLOP/s at {passes} bf16 pass(es), "
        f"bandwidth {hw.peak_hbm_bandwidth / 1e9:.0f} GB/s, ridge {ridge:.1f} FLOP/byte"
    )
    columns = (
        f"{'label':<7} {'T':>5} {'blk_t':>6} {'blk_h':>6} {'prec':<8} "
        f"{'ms':>8} {'GFLOP/s':>9} {'I':>7} {'%roof':>7}  verdict"
    )
    lines = [header, "", columns, "-" * len(columns)]

    kernel_ms: dict[int, float] = {}
    xla_ms: dict[int, float] = {}

    for r in results:
        blk_t = "-" if r.block_t is None else str(r.block_t)
        blk_h = "-" if r.block_h is None else str(r.block_h)
        if r.status != "ok" or r.median() is None:
            lines.append(
                f"{r.label:<7} {r.tokens:>5} {blk_t:>6} {blk_h:>6} {r.precision:<8} "
                f"{'-':>8} {'-':>9} {'-':>7} {'-':>7}  FAILED"
            )
            continue

        seconds = r.median()
        flops = mlp_flops(r.tokens, cfg=cfg)
        achieved = flops / seconds
        (kernel_ms if r.label == "kernel" else xla_ms)[r.tokens] = seconds

        if r.block_t is None:
            intensity_s, percent_s, verdict = "-", "-", "baseline"
        else:
            moved = mlp_bytes(
                r.tokens, block_t=r.block_t, cfg=cfg, elide_weights=elide_weights
            )
            intensity = arithmetic_intensity(flops, moved)
            attainable = min(peak, intensity * hw.peak_hbm_bandwidth)
            intensity_s = f"{intensity:.1f}"
            percent_s = f"{100 * achieved / attainable:.1f}"
            if abs(intensity / ridge - 1.0) < RIDGE_TOLERANCE:
                verdict = "at ridge"
            elif intensity < ridge:
                verdict = "HBM-bound"
            else:
                verdict = "compute-bound"

        lines.append(
            f"{r.label:<7} {r.tokens:>5} {blk_t:>6} {blk_h:>6} {r.precision:<8} "
            f"{1e3 * seconds:>8.3f} {achieved / 1e9:>9.1f} {intensity_s:>7} "
            f"{percent_s:>7}  {verdict}"
        )

    shared = sorted(set(kernel_ms) & set(xla_ms))
    if shared:
        lines.append("")
        lines.append("kernel vs XLA (>1 means the kernel is faster):")
        for tokens in shared:
            lines.append(f"  T={tokens:<6} {xla_ms[tokens] / kernel_ms[tokens]:.2f}x")

    return "\n".join(lines)
