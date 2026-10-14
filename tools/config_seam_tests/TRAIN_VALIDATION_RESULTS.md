# TrainingConfig / ValidationConfig comparison — PR8149

Validated on 2026-10-14: **111 exact passes, one expected CLI rejection, and
one shared baseline failure across all 113 cases**. No configuration mismatches,
candidate-only failures, skipped cases or missing cases. This is not an all-green
result.

- Frozen main: `fb65908da91d6185590ace81e2aeb84760c0566b`.
- [PR8149](https://github.com/NVIDIA/Megatron-LM/pull/8149) full-matrix head:
  `fdf43f1d5fae513551f5be2e9faa2b7cb8676358`.
- Both sides: H100 GPUs, `nvcr.io/nvidian/nemo:26.10.rc0`.
- Manifest SHA256: `ed3a15db2eecfe3ca30e70b3e3a9bdab95576b5c935457abdc668d0350e26c32`.

The GPU matrix runs only the fork-only capture/compare harness, not MLM unit
tests. Runtime cases execute short real training runs. Resume cases share a
checkpoint produced by the pinned baseline.

## Full-matrix results

| Selection | Cases | Result |
|---|---:|---|
| One-rank captures | 93 | 91 exact passes, one expected rejection, one shared baseline failure |
| Two-rank captures | 18 | All exact passes on both ranks |
| Four-rank rank-mapping capture | 1 | Exact pass on all ranks |
| Eight-rank TP2/PP2/DP2 capture | 1 | Exact pass on all ranks |

All 282 rank snapshots were checked for case/rank identity and successful status
where required. Every checkpoint-seed run completed successfully and created a
checkpoint. The five captured harness implementation hashes match the published
files. The real `examples/multimodal/train.py` startup case is included; it does
not claim full multimodal data loading or training.

`invalid_batch_conflict` rejects simultaneous explicit `--global-batch-size`
and `--step-batch-size-schedule` flags on both revisions, as expected.

`coverage_checkpoint_integrity_resume` fails on both revisions with
`--dist-ckpt-strictness raise_all`: the checkpoint contains
`rerun_state_machine_state`, but the requested sharded state dictionary omits it.
This is the previously recorded strict-checkpoint baseline issue, retained as a
failure. No product fix or strictness relaxation is included. Ordinary checkpoint
resume cases pass.

## Observation changes

The manifest adds `coverage_model_test_mode`, a builder case checking that
`--test-mode` reaches the model transformer config. The legacy
`ValidationConfig.test_mode` duplicate must agree with the independently
captured model input before it is omitted from the comparable validation view.
All other validation fields remain strict comparisons, including unknown fields
and missing native values. Raw validation schemas are retained separately.

Native model containers supply `cfg.model.transformer.test_mode`. Legacy VLM
callbacks construct their transformer config later, so startup observes their
CLI input instead. An initial harness attempt incorrectly assumed every VLM
entrypoint already had a model container and failed on both revisions. That
attempt was stopped and retained separately; the observation was corrected before
the full matrix above. No product changes or relaxed comparison rules were needed.

Local harness self-checks pass (24). All-files pre-commit retains unrelated
baseline Core formatting/Pylint findings; those files are not modified.

## Checkpoint comparison follow-up

The subsequent head `5e36f7d20eb3c1b9a2f6b984782321d237de2ec1` shares one
config-aware checkpoint comparison helper between `train.global_batch_size`
and `rng.data_parallel_random_init`. The 113-case result above belongs to its
parent, not this later head.

Separate focused product verification passes all 23 unit cases: config ownership,
saved-config precedence, older-checkpoint fallbacks, explicit nulls, mismatch and
skip handling, existing RNG resume policy, and wide-residual compatibility.
These unit tests are not part of the fork-only suite.

All four selected fork-only follow-up comparisons pass exactly against the same
frozen main, using the same harness implementation:

- `runtime_resume` (one rank).
- `checkpoint_args` (one rank).
- `coverage_checkpoint_override_iteration` (one rank).
- `rng_dp_resume` (two ranks).

All ten baseline/candidate rank snapshots and both baseline checkpoint seeds
were verified. There are no differences or capture errors. This targeted rerun
does not claim a second full 113-case run on the later head.

## Reproduce and limits

Follow the [README](README.md), using the exact full-matrix revisions above and
grouping manifest cases by rank requirement: 93 one-rank, 18 two-rank, one
four-rank and one eight-rank case. Keep main frozen between comparisons.
Reports retain revision/tree IDs, environment, fixture and harness hashes,
per-rank outcomes, field differences and capture errors. Raw artifacts are
archived separately and not committed to this public branch.

This matrix compares configuration/consumer inputs and captured RNG state. It
does not establish tensor-weight, convergence, performance, exhaustive-entrypoint
or native-config-only ownership equivalence. External logging services remain
disabled; see [the scenario inventory](EXPANSION.md) for feature-specific limits.
