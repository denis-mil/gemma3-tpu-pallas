"""Pallas TPU kernels at Gemma 3 1B's shapes."""

from .mlp import fused_gated_mlp
from .shapes import (
    GEMMA3_1B,
    V5E,
    Gemma3Config,
    TpuV5e,
    arithmetic_intensity,
    attention_flops,
    mlp_bytes,
    mlp_flops,
    roofline_bound,
)

__all__ = [
    "GEMMA3_1B",
    "V5E",
    "Gemma3Config",
    "TpuV5e",
    "arithmetic_intensity",
    "attention_flops",
    "fused_gated_mlp",
    "mlp_bytes",
    "mlp_flops",
    "roofline_bound",
]
