# Kernels are benchmarked standalone, at Gemma 3 1B's real shapes

The model is never loaded. No checkpoint, no tokenizer, no Hugging Face
authentication, no sampler. Kernels are benchmarked in isolation, but exclusively at
the shapes Gemma 3 1B actually runs: 26 layers, 4 query heads, 1 KV head, head
dimension 256, window 512, sequence lengths from 512 to 32768.

A reader may reasonably wonder why a project about "optimising inference" never runs
the model, so the reasoning is recorded here. End-to-end integration is plumbing —
gated checkpoint access, weight conversion, KV cache wiring, sampler surgery — and
none of it is Pallas work. On a free Colab runtime with a 12-hour session cap, a
90-minute idle timeout and no guarantee of getting a TPU at all, that plumbing is the
most likely place for the project to stall. Worse, end-to-end timings bury a kernel's
effect underneath embedding lookups, normalisation and sampling, which is precisely
the signal the project exists to measure.

Constraining the shapes to the real model is what keeps the claim honest. "This
kernel is 12× faster at Gemma 3 1B's exact prefill shapes" is true and verifiable
without ever holding the weights.

## Consequences

The project cannot report tokens per second or time-to-first-token, and should not
imply it can. Its results are per-kernel latency, achieved FLOP/s and achieved
bandwidth. Any future end-to-end claim requires reopening this decision.
