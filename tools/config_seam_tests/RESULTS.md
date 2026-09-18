# PR7418 configuration comparison

## Fixed-main before/after comparison — 2026-09-18

**All 38 cases have the expected result: 37 exact passes and one expected CLI
rejection. Zero configuration differences and zero execution/capture errors.**

This rerun validates PR7418 after integration with merged PR7447. Both sides
contain the same upstream changes; only PR7418 differs. Main and the candidate
were frozen for the entire run.

| Role | Revision |
| --- | --- |
| Before: fixed main | `d564dd01de4549a4884be26b1d74e169798fd877` |
| After: PR7418 | `f0148a3600818588406d8ca6437a8de1f8377f38` |
| Harness code and manifest | `0ee5866bee35f2fb78d4368ab02466ae1c3f6485` |

| Allocation | Cases | Outcome |
| --- | --- | --- |
| One H100 | 33 | 32 exact passes; `invalid_batch_conflict` rejected as expected on both sides |
| Two H100s | 4 | 4 exact passes: MoE EP2, CP2, TP2/sequence-parallel overlap, interleaved PP2/VPP2 |
| Eight H100s | 1 | Exact pass: TP2/PP2/DP2 |

The expected rejection is `invalid_batch_conflict`. It deliberately supplies both
`--global-batch-size 4` and `--step-batch-size-schedule "0:2 256:4"`.
Both revisions reject the conflicting raw CLI flags with:

```text
Cannot specify both --step-batch-size-schedule and --global-batch-size
```

This is a passing negative test: it verifies that existing CLI validation is
preserved, not a training failure or a waived comparison.

This rerun used the unchanged 38-case manifest, capture code, comparison rules and
`nvcr.io/nvidian/nemo:26.10.rc0` environment. Each pair used identical harness and
fixture hashes; checkpoint-dependent cases shared a checkpoint created by the
baseline. No fields were ignored or waived.

The 37 positive cases comprise 27 builder cases and 10 runtime cases. Runtime
coverage includes fresh training, checkpoint resume, full recompute, fine-tuning,
frozen-vision VLM and the distributed scenarios above. This establishes equivalence
of the captured configurations/consumer inputs and completion of the short runtime
checks, **not full convergence, tensor-value equality or performance equivalence**.

The 13 harness self-tests pass in each allocation. Earlier on the same day, all
31 initialization-lifecycle unit cases passed on this exact candidate, including
TrainState initialization through config-first and legacy args-only startup.
Full repository pre-commit on these same baseline/candidate revisions reported
the same 30 unrelated Black-modified files and identical Pylint diagnostics;
isort passed and all 35 PR-file hashes remained unchanged. No baseline lint
changes were retained.

Detailed reports retain source trees, harness/fixture hashes and per-rank evidence;
private paths and raw artifacts are not published. Earlier results remain in Git
history rather than being accumulated on this page.
