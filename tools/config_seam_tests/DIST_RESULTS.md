# DistributedInitConfig comparison — PR7778

Validated on 2026-10-13: **110 exact passes, one expected CLI rejection, and
one shared baseline failure across all 112 cases**. There are no configuration
mismatches or candidate-only failures. This is not an all-green result.

- Frozen main: `3bb482c487cd447a2b40a3bf076ab14cad514e13`.
- [PR7778](https://github.com/NVIDIA/Megatron-LM/pull/7778) head:
  `f994dea1253eae06852057ac93ce8042e9531bbe`.
- Both sides: H100 GPUs, `nvcr.io/nvidian/nemo:26.10.rc0`.
- The final manifest contains **112 cases**: 57 builder, 54 runtime, and one
  real multimodal-entrypoint startup case. Builder cases do not train.

Only the fork-only capture/compare matrix runs on GPUs. No MLM unit tests are
part of this run. Runtime cases execute short real training runs; checkpoint
resume cases share a checkpoint produced by the pinned baseline.

## Results

| Selection | Cases | Result |
|---|---:|---|
| One-rank captures | 92 | 90 exact passes, one expected rejection, one shared baseline failure |
| Two-rank captures | 18 | All exact passes on both ranks |
| Four-rank rank-mapping capture | 1 | Exact pass on all ranks |
| Eight-rank TP2/PP2/DP2 capture | 1 | Exact pass on all ranks |

`invalid_batch_conflict` rejects simultaneous explicit `--global-batch-size`
and `--step-batch-size-schedule` flags with the expected error on both revisions.
No cases are skipped or missing. All 288 rank snapshots, including follow-up
runs, were checked for case/rank identity and successful status where required.
The separate negative import control is excluded from that count.

The original frozen 111-case run recorded 106 passes, four errors and one
expected rejection. Three errors were invalid fixtures, corrected in the
separate runs below. The fourth remains a baseline error. The extra Hybrid
TE-tracker case brings the reconciled result to 112 unique cases.

Async checkpoint saves finalized iterations 1 and 2 on both revisions, with
nonempty checkpoint shards/metadata and latest iteration 2 verified afterward.
This extra artifact check does not claim checkpoint tensor-payload equality.
Local harness self-checks passed (23); these are separate from the GPU matrix.
Scoped import/undefined-name checks pass. All-files lint retains unrelated
baseline Core formatting/Pylint findings; those files were not changed.

## Baseline issue retained

`coverage_checkpoint_integrity_resume` fails identically on both revisions with
`--dist-ckpt-strictness raise_all`: `rerun_state_machine_state` exists in the
checkpoint but is missing from the requested sharded state dictionary.

The save path persists a rerun-state placeholder in `NOT_RUNNING_YET`, while
`validate_state_dict()` rejects that state and the load path omits its template.
The relevant `megatron/core/rerun_state_machine.py` and
`megatron/training/checkpointing.py` files are identical between these two
commits. The case and its strictness are unchanged. Equal failures do **not**
count as equivalence, and no product fix is included in this harness branch.

## Fixture corrections and follow-ups

The original 111-case run remains frozen. Separate reruns use the same product
commits and retain the original errors in the archived reports.

| Case | Original problem | Correction / separate result |
|---|---|---|
| `coverage_tokenizer_vocab_file` | The legacy BERT alias requires uncached Hub metadata even with a local vocabulary | Local BERT/WordPiece tokenizer metadata plus explicit vocabulary file; exact pass, including the expected 261-token vocabulary |
| `coverage_dataset_fim_config` | FIM rejects mock data during argument validation | Remove mock-data mode and supply a symbolic indexed-data prefix; exact builder-only pass, with no data loading or FIM augmentation claimed |
| `profiling_excluded_rank` | Start/end steps `2/2` are invalid for PyTorch profiling | Preserve the fork's existing `2/3` correction; exact pass |

The original expansion draft predated the fork's `hybrid_te_tracker` addition
and excluded-rank correction. Both existing changes and the prior RNG report
are preserved. The Hybrid TE-tracker case passes separately and raises the
final unique-case count from 111 to 112. The defaults control also passes with
the corrected offline tokenizer fixture generator; it is not an extra case.

The capture/comparison rules were not relaxed for these reruns. The only Python
change after the initial run is supplying local BERT tokenizer metadata in the
fixture generator. Other follow-up changes are the manifest corrections above.

## Real multimodal startup coverage

`multimodal_train_startup` executes the actual
`examples/multimodal/train.py` imports, parser, configuration construction,
tokenizer and runtime-service setup, then observes its call into `pretrain`.
Both pinned revisions pass. There are no stub imports.

A separate negative control uses the old PR head
`ac091bd84b7924160d94e66f1be7b8048b6d9a3d`: main passes, but that old head fails
specifically with `ImportError: cannot import name 'get_run_config' from
'megatron.training'`. This proves the new case catches the previously missed
package-export bug. The negative control is not counted among the 112 cases.

This is startup coverage, not real Energon data loading or multimodal training.
The existing `pretrain_vlm.py` mock-data runtime cases remain separate.

## Reproduce and limits

Follow the [README](README.md), using the exact product revisions above and
grouping cases by their manifest rank requirement: 92 one-rank, 18 two-rank,
one four-rank, and one eight-rank case. Do not advance main between comparisons.
Reports retain revision/tree IDs, environment, fixture and harness hashes,
per-rank outcomes, field differences and capture errors. Raw artifacts are
archived separately and are not committed to this public branch.

The distributed-config observation preserves the raw schema. It compares the
legacy `fake_process_group` CLI value only when that field is absent from the
dataclass, and checks the old FSDP duplicate against its DDP owner before
omitting the duplicate. Missing native values and unknown fields are not hidden.

This matrix compares configuration/consumer inputs and captured RNG state. It
does not establish tensor-weight, convergence, performance, exhaustive-entrypoint
or native-config-only ownership equivalence. External logging services remain
disabled; see [the scenario inventory](EXPANSION.md) for feature-specific limits.
