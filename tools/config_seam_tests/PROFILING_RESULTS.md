# ProfilingConfig ownership comparison

## Current result — 2026-09-21

Eight runtime cases pass exactly against both the config-first foundation and its
fixed main ancestor. Zero configuration differences or execution/capture errors.

| Role | Revision |
| --- | --- |
| Candidate: C01 profiling ownership | `dc57fd4d85e4980d057945307c3c0e88cf96470a` |
| Incremental baseline: PR7418 parent | `1e2ddf82b18f7568a1813b320b76ad34bb222557` |
| Cumulative baseline: fixed main | `3226303b34002c6022c08035694da0b46e4110b1` |
| Harness code and manifest | `e140613d1226d80b120f2a65cf1e9aab69436f5c` |

Each comparison used one H100 and `nvcr.io/nvidian/nemo:26.10.rc0`, with identical
harness/fixture hashes within the pair and a shared checkpoint produced by that
pair's baseline. Main was not advanced between runs. The 13 harness self-tests
passed in each allocation.

| Case | PR7418 parent → C01 | Fixed main → C01 |
| --- | --- | --- |
| `runtime_fresh` | Exact pass | Exact pass |
| `runtime_resume` | Exact pass | Exact pass |
| `profiling_disabled` | Exact pass | Exact pass |
| `profiling_excluded_rank` | Exact pass | Exact pass |
| `profiling_nsys` | Exact pass | Exact pass |
| `profiling_pytorch` | Exact pass | Exact pass |
| `profiling_memory` | Exact pass | Exact pass |
| `profiling_resume` | Exact pass | Exact pass |

Enabled PyTorch and memory cases required real schedule/snapshot-call captures.
Both revisions also produced nonempty PyTorch traces and memory snapshots.
The excluded-rank case deliberately supplies an otherwise invalid unused PyTorch
window; both revisions preserve the accepted no-op when no launched rank is selected.

The exact candidate also passed 184 targeted GPU unit tests, versus 158 on the
PR7418 parent: initialization lifecycle, argument conversion, checkpointing,
ModelOpt model-builder behavior, and 26 profiling ownership cases. Coverage includes
missing YAML aliases, config values diverging from or replacing legacy args,
rank/window behavior, real checkpoint serialization in three formats, detached
save snapshots, and existing current-run profiling precedence after resume.
There were no test failures or skips. Changed-file lint and copyright checks pass;
unrelated repository-wide formatting/Pylint findings remain separate.

## Reproduce and interpret

Follow [the harness run instructions](README.md) using the pinned revisions above
and the eight profiling-pilot cases. Use the same development environment and
fixtures for both sides. The harness remains fork-only and is not part of a
product migration PR.

This validates exact captured configuration/consumer inputs and successful short
training, save/resume, and profiler execution. It does not establish full tensor-value
equality, convergence, performance equivalence, or every model/script combination.
The nsys case exercises CUDA-profiler/NVTX APIs without an external Nsight capture;
Chakra and OOM callbacks have targeted unit coverage only.

The manifest has 44 available cases; this report covers eight, not a full 44-case
rerun. The separate [PR7418 report](RESULTS.md) retains its own pinned 38-case
validation and must not be attributed to this candidate. Raw artifacts and private
environment paths are not published. Keep this page limited to the current validation;
Git preserves its revision history.
