# LoggerConfig ownership comparison

## Current result — 2026-09-22

Five runtime cases pass exactly: zero captured configuration/consumer differences
and zero execution/capture errors.

| Role | Revision |
| --- | --- |
| Frozen main baseline (merged #7418) | `5e85e2b27fc29bcf938e8f3b38acfbdecffcc8f3` |
| LoggerConfig candidate ([draft #7587](https://github.com/NVIDIA/Megatron-LM/pull/7587)) | `48e29aa871fdb9fd80d07d5083d15e316d56e826` |
| Fork-only harness code and manifest | `97758a57b0da28f0583f66c9ff7f625d33cbbe7b` |

One H100, `nvcr.io/nvidian/nemo:26.10.rc0`, identical harness and fixture hashes,
and a shared checkpoint produced by the baseline. Main was not advanced between
runs. All 15 harness self-tests pass.

| Case | Result |
| --- | --- |
| `runtime_fresh` | Exact pass |
| `runtime_resume` | Exact pass |
| `logging_metrics` | Exact pass |
| `logging_attention` | Exact pass |
| `logging_resume` | Exact pass |

The metrics case uses default enabled timing barriers; logging resume disables
them for the current run. Writer and timer constructor inputs match. The three
logging cases produced nonempty TensorBoard artifacts with identical scalar tag
sets. Loss, LR, gradient/parameter norms, zero-gradient counts and attention maxima
match exactly wherever emitted. Metrics has 24 exact scalar trajectories and 12
timing/throughput differences; attention has 13/13 exact and resume 18/18 exact.
Elapsed-time and throughput measurements are not equality targets.

Targeted units: candidate **331 passed**, baseline **285 passed**, three identical
skips each, no errors/failures. All 46 ownership cases ran, including native config
values diverging from or replacing legacy args, detached metadata, service inputs,
model/optimizer derivation, nested MIMO configs, teacher-YAML precedence and optional
inspector startup. The final candidate differs from the unit-tested source only by
wrapping changed calls and one docstring; executable AST equivalence was verified.

## Reproduce and interpret

Follow [the harness instructions](README.md) with the pinned revisions and these
five cases. The comparison harness stays in this fork, outside the product PR.
Its semantic observation adapter is not proof of ownership by itself; the separate
native/divergent-args behavioral tests provide that gate.

The manifest has 47 available cases; this report covers five, not a full rerun.
This validates short training/save/resume and captured configuration/consumer
inputs, not full tensor equality, convergence, performance, multirank execution
or every supported entrypoint. External telemetry services and workload-inspector
startup have controlled unit coverage, not live network integration coverage.

Independent product and harness reviews found no remaining blocking findings.
Baseline formatting/Pylint findings remain separate. Raw artifacts and private
environment paths are not published. Git preserves this report's history.
