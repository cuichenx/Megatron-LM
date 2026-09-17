# PR7418 configuration comparison

## Fixed-main before/after rebase — 2026-09-17

**All 38 cases have the expected result: 37 exact passes and one expected CLI
rejection. Zero configuration differences and zero execution/capture errors.**

The PR was rebased once onto the main commit already selected for testing. Both
sides now contain the same upstream changes; only the PR differs. Main was not
advanced again during the run.

| Role | Revision |
| --- | --- |
| Before: fixed main | `cfa9e20d2658844dd84a07f2ccd34d7e62ab5552` |
| After: rebased PR7418 | `c92551aa6f63c64a5a495a94b784c19ad0966a8c` |
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

During the run, PR head advanced to `f9add9679ed177b5a3c355639280e26beb9942a9`
solely to add the 2026 copyright header to the lifecycle unit-test file. Exact-text
and AST checks verified that this follow-up changed no executable code. The matrix
remained pinned to `c92551aa6`; it was not rerun or relabeled as testing `f9add9679`.
Both GitHub copyright checks pass for that header-only follow-up.

The 13 harness self-tests pass in each allocation. Full repository pre-commit was
run on both pinned revisions in disposable checkouts: both report the same 30
unrelated Black-modified files and identical Pylint diagnostics; isort passes.
All 35 PR-file hashes remained unchanged by those checks. No baseline lint changes
were retained. Detailed reports retain source trees, harness/fixture hashes and
per-rank evidence; private paths and raw artifacts are not published.
