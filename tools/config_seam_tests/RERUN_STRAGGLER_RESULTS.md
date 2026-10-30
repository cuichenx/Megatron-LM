# C14/C15 rerun and straggler migration

PR: [NVIDIA/Megatron-LM#8798](https://github.com/NVIDIA/Megatron-LM/pull/8798).

Tested on 2026-10-30:

- Baseline: `a2946a03b1c8f889ca79bf53adbbe6d0bc1de2c8` (pinned main).
- Candidate: `4dfc89869575a95c354034e941bdae0c79e0e536` (baseline plus migration).
- Environment: cached NeMo `26.08.rc9`, image ID `sha256:a6e707b53662a5d9efc7aad237b14ed6428cab2f3e8776725649de225c0624c0`, one RTX 6000 Ada GPU per process group, mock data, no network access.

## Comparison results

All six selected cases passed, with zero captured differences and no missing required observations:

| Case | Coverage |
| --- | --- |
| `defaults` | Default config construction and consumed inputs |
| `runtime_fresh` | Training, evaluation, and checkpoint save |
| `runtime_resume` | Restore the same baseline checkpoint and continue training |
| `rerun_replay` | Real transient numerical-error injection, replay, and continuation through iteration four |
| `straggler_enabled` | Enabled detector setup and reporting every two iterations |
| `straggler_start_disabled` | Detector setup with runtime enablement initially false |

The separate `rerun_deterministic_cases.json` manifest also passed both fresh/save and resume cases with `--deterministic-mode` enabled. Comparisons include exact loss/LR, rerun transitions and initialization inputs, config/consumer inputs, and captured RNG/checkpoint boundaries.

The replay case observed three replay passes and 12 replayed batches on each revision. Their hashes matched the corresponding original batches. The harness independently checks this relationship within each run, in addition to before/after equality; its negative self-test rejects identical corruption on both revisions.

## Reproduce

Use clean baseline/candidate checkouts in the same supported MLM environment. From this directory:

```bash
uv run --no-project python compare_config_seams.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" \
  --baseline-revision a2946a03b1c8f889ca79bf53adbbe6d0bc1de2c8 \
  --candidate-revision 4dfc89869575a95c354034e941bdae0c79e0e536 \
  --environment-id "$IMAGE_ID" --output "$RESULTS/selected" \
  --cases defaults runtime_fresh runtime_resume rerun_replay \
    straggler_enabled straggler_start_disabled

uv run --no-project python compare_config_seams.py \
  --baseline "$BASELINE" --candidate "$CANDIDATE" \
  --baseline-revision a2946a03b1c8f889ca79bf53adbbe6d0bc1de2c8 \
  --candidate-revision 4dfc89869575a95c354034e941bdae0c79e0e536 \
  --environment-id "$IMAGE_ID" --output "$RESULTS/deterministic" \
  --manifest rerun_deterministic_cases.json
```

Set the path/environment variables explicitly; use new output directories. The runner records exact revisions, trees, manifest/observer hashes and fixture hashes in `report.json`.

## Limits and separate checks

This is the focused C14/C15 selection, not a rerun of the entire scenario matrix. The harness does not execute MLM unit tests. Its 27 CPU-only self-checks passed separately.

Replay integrity covers the single-iterator probe, not interleaved/multi-iterator pipeline replay. Reading losses and hashing batches synchronize tensors: this is correctness evidence, not performance or straggler-timing evidence. Both resume sides consume one baseline-created checkpoint; this is not a candidate-checkpoint round-trip claim.

Separately, 383 targeted MLM tests passed at the candidate commit. Six hardware-specific parameters were deselected on SM89; four SM90-only failures were independently reproduced on the pinned baseline. No baseline hardware fix or unit-test invocation is included in this harness.

The initial six-case and deterministic captures preceded the additional replay-integrity assertion and import formatting. Their recorded observer hashes are retained in the private run reports. The strengthened assertion was also applied to both saved replay captures, and a fresh before/after `rerun_replay` run using the final harness passed with zero differences or errors. No comparison tolerance or ignored field was added.
