# ProfilingConfig ownership comparison

## Current result — 2026-09-22

Eight runtime cases pass exactly against frozen main. Zero captured configuration
differences or execution/capture errors. The candidate reads profiling settings
through the global run config and retains the existing inline profiler lifecycle.

| Role | Revision |
| --- | --- |
| Candidate: PR7568 | `098a809c25df41907fca9eca9aa7c19eee6b27b2` |
| Baseline: frozen main, including merged PR7418 | `5e85e2b27fc29bcf938e8f3b38acfbdecffcc8f3` |
| Harness checkout | `bfd84ac314de0e6d75c75caa436e07017c8193ee` |
| Harness code and manifest | `40738bfed9b80a774eedeee4c1b0cd7e5ba716a4` |

The comparison used one H100 and `nvcr.io/nvidian/nemo:26.10.rc0`, with identical
harness/fixture hashes within the pair and a shared checkpoint produced by the
baseline. Main was not advanced between runs. All 17 harness self-tests passed.

| Case | Result |
| --- | --- |
| `runtime_fresh` | Exact pass |
| `runtime_resume` | Exact pass |
| `profiling_disabled` | Exact pass |
| `profiling_excluded_rank` | Exact pass |
| `profiling_nsys` | Exact pass |
| `profiling_pytorch` | Exact pass |
| `profiling_memory` | Exact pass |
| `profiling_resume` | Exact pass |

Enabled PyTorch and memory cases require real schedule/snapshot-call captures.
Both revisions produced nonempty PyTorch traces and memory snapshots.
The excluded-rank case deliberately supplies an otherwise invalid unused PyTorch
window; both revisions preserve the accepted no-op when no launched rank is selected.

Targeted GPU units: **325 candidate / 294 baseline passed**, with the same three
skips. Includes 31 profiling/global-config cases, initialization lifecycle, argument
conversion, checkpoint serialization, ModelOpt model-provider behavior, GPT/Hybrid
builders, and selected MiMo/batch compatibility checks. The new cases exercise the
actual inline profiler paths, config values replacing legacy args, guarded global
registration and cleanup, inference-provider compatibility, and detached checkpoint
metadata. Changed-file checks and independent full-diff review pass; unrelated
repository-wide formatting/Pylint findings remain separate.

## Reproduce and interpret

Follow [the harness run instructions](README.md) using the pinned revisions above
and the eight cases listed here. Use the same development environment and fixtures
for both sides. The harness remains fork-only, outside the product migration PR.

This validates exact captured configuration/consumer inputs and successful short
training, save/resume, and profiler execution. It does not establish full tensor-value
equality, convergence, performance equivalence, or every model/script combination.
The nsys case exercises CUDA-profiler/NVTX APIs without an external Nsight capture;
Chakra and OOM callbacks have targeted unit coverage only.

The manifest has 52 available cases; this report covers eight, not a full matrix
rerun. The separate [PR7418 report](RESULTS.md) retains its own pinned validation.
Raw artifacts and private environment paths are not published. This page keeps
only the current result; Git preserves previous revisions.
