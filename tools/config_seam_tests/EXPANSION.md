# Configuration coverage expansion

2026-10-13 update: the 110-case draft is integrated with the fork's existing
Hybrid TE-tracker case and a real `examples/multimodal/train.py` startup capture
(112 total). See [the PR7778 validation report](DIST_RESULTS.md) for results
against frozen main `3bb482c48` and PR head `f994dea12`. The parameter-input
inventory below remains the historical 2026-09-23 source audit, not a fresh
spreadsheet audit or a behavior-coverage percentage.

The subsequent training/validation comparison adds `coverage_model_test_mode`
(113 current cases), checking the enabled model-owned flag after its redundant
ValidationConfig field is removed. This does not change the historical PR7778
result or the parameter-input counts below.

## Status and scope

Requested 2026-09-23. The isolated harness-only draft started from fork harness
`59e0123bbdff8c6a4ef06a709db10e4e9e449bca`; integration preserves the later fork
changes through `44b3e84c3145510d3125355ed4ca6c5c5a475548`. It does not alter
MLM production code or rewrite earlier test-result records.

There are 58 additional `coverage_*` cases. Local harness self-tests and static
source checks alone are not GPU evidence. The subsequent PR7778 validation
records the actual passing cases, fixture corrections, and retained baseline
failure separately; it does not retroactively validate older migration PRs.

Exclude new fault-tolerance, rerun, straggler, RL, distillation, and general
model-architecture coverage. Per Chen's scope clarification, include VLM,
inference, and tensor-parallel model communication-overlap settings. A few other
tiny-model settings are prerequisites for optimizer/DDP or routing-log tests,
not a new architecture matrix.

## Parameter-input inventory

Reference: [args mapping](https://docs.google.com/spreadsheets/d/1FpqpiP60RKAbz5o6WKt9plhc50BBavJYy4QouUDOMPU/edit?gid=1760900932),
read 2026-09-23: 609 distinct rows, including 161 marked "not used in training loop".
Counts are distinct arguments explicitly supplied in the materialized manifest,
including common inputs, bool-negation flags, and metadata aliases. An option
whose manifest value is false is omitted, not counted. Expected-rejection cases
are excluded. CLI names were traced through the pinned MLM parser and config
argparse metadata at `5e85e2b27fc29bcf938e8f3b38acfbdecffcc8f3`.

Explicit spreadsheet inputs increase from **148/609 to 315/609**, and from
**133/448 to 291/448** for the literal training-loop complement. These are
**draft input counts**, not passing-case counts or behavior-coverage percentages.
They include builder-only and dormant settings; some supplied values equal defaults.

| Sheet group | Original explicit inputs | Expanded explicit inputs |
| --- | ---: | ---: |
| cfg.ddp | 3/21 | 19/21 |
| cfg.dist | 1/22 | 16/22 |
| cfg.dataset | 7/31 | 29/31 |
| cfg.logger | 16/39 | 33/39 |
| cfg.checkpoint | 10/54 | 54/54 |
| cfg.optimizer | 7/14 | 12/14 |
| cfg.train | 6/19 | 19/19 |
| cfg.rng | 4/4 | 4/4 |
| cfg.vlm | 4/11 | 9/11 |
| cfg.ckpt | 0/1 | 0/1 |
| cfg.scheduler | 9/15 | 15/15 |
| cfg.validation | 2/9 | 9/9 |
| cfg.inference | 0/2 | 2/2 |
| cfg.profiling | 11/12 | 12/12 |
| cfg.tokenizer | 1/2 | 2/2 |
| cfg.comm_overlap | 0/1 | 1/1 |

The Muon scenario additionally asserts the derived
`use_layer_wise_distributed_optimizer=True` value. It is not a user-supplied flag
and is therefore not credited in the explicit-input count. Similarly, local rank
comes from the launcher and is observed in config rather than forced via CLI.

## Stronger observations

- Compare the complete distributed-init config, not just selected flags.
- Observe DataLoader numeric/policy inputs and single/cyclic/full-validation
  sampler constructor inputs without serializing datasets or runtime objects.
- Record configured and running batch size separately. A configured size of 10
  with microbatch size 3 and rounding enabled keeps configured size 10, runs 9,
  and produces three microbatches; evaluation has its own divisible size 9.
- Every added case includes expected-value assertions checked on each side.
  Missing values, wrong types, wrong aliases and ignored options fail even when
  the two snapshots would otherwise be equal. Expectations for one seam must
  match one complete event, not pieces from different events.
- Checkpoint iteration override is asserted at the actual load return value.
- Add an offline WordPiece vocabulary for the tokenizer's file-based input.
- Observe VLM freezing (language-only and projection-only training), separate data
  sequence length, class-token removal, CP packed format, and decoder-only overlap.
  Packed input uses CP=2 and microbatch=2; its unpadded sequence length is already
  CP-aligned so `use_packed_sequence` itself must select `thd` packing.
- Observe actual TE userbuffer initialization with default/file-based policies,
  and NCCL/Gloo bootstrap choices. The YAML-compatible JSON fixture contains
  distinct QKV/FC1 ring-exchange aggregation policies adapted from pinned MLM's
  `gpt3_weekly_mcore_tp2_pp2_current_scaling_native_fp8_tp_pp_sp_tp_overlap`
  functional recipe. These cases use BF16; no quantization-mode comparison is claimed.
  Bulk and collective switches are asserted on the model-constructor config.
- Explicitly invoke the real inference config factory in builder-only probes:
  static settings, dynamic batching with chunked prefill/prefix caching, and
  current-run inference settings alongside checkpoint-loaded model arguments.
  This does not claim pretraining normally constructs an inference config.

## Important limits / remaining work

- Full validation remains builder-only: evaluating the entire default mock corpus
  is not a bounded smoke test. Add tiny indexed-data fixtures before promoting it
  or multi-validation/FIM/cache scenarios to behavioral tests. Symbolic multi-data
  prefixes are intentionally only used by config construction, never data loading.
- Cache policies, FIM, debug tensor-dump intervals, checkpoint conversion,
  local checkpoint algorithm, FSDP registration/reshard policies, optimizer CUDA
  graphs, and Chakra have explicit builder checks, not runtime feature claims.
- Checkpoint worker/drop-extra-state CLI inputs are supplied but are not separately
  observed downstream yet. Parameter-input counts must not be called consumer coverage.
- External one-logger/W&B/OTel services remain disabled. Dormant metadata cases
  test mapping, not service activation. Inspector, energy monitoring, SHARP,
  NCCL config files, lazy/external distributed initialization, MSC, and backend-
  specific GTP/MXFP8 require dedicated fixtures or platform validation. New TP
  overlap validation is specific to the pinned H100/container environment,
  including the tested bootstrap backends, not arbitrary hardware/TE versions.
- VLM `vision_pretraining` and `vision_pretraining_type` remain uncovered: DINO
  invokes model-specific methods absent from LLaVA. It needs a separate vision
  entrypoint and fixture rather than setting dormant flags on a LLaVA case.
- Inference coverage is declarative, not generation or serving. The spreadsheet
  lists only two inference fields; additional dynamic-batching inputs are tested
  but do not inflate its denominator. KV-cache sizing is not runtime-validated.
- Deprecated TP split switches and `tp_comm_overlap_rs_dgrad` have builder checks
  only. Atomic AG/RS and QKV/FC1 per-operation disable fields have no CLI in the
  pinned MLM parser; native-config tests would be needed to vary those fields.
- Async save-hook completion does not itself prove durable writer completion.
  Validate final checkpoint artifacts and worker cleanup when running that case.
- The harness still does not compare tensor weights, convergence, performance,
  all checkpoint payload fields, or prove that migrated config is the only owner.

## Original draft validation (2026-09-23)

- CPU-only harness self-tests: 22 pass (including expected-value failure tests,
  optional probe tier restrictions, and the offline overlap fixture).
- Static flag inventory: every new option has a declaration in the pinned source.
- Actual stdlib-only config ArgumentGroupFactory/parser schema checks accept the
  new cases' non-model options; six already-existing model prerequisite flags
  were checked by declaration rather than imported through GPU dependencies.
- Executed the pinned stdlib-only microbatch calculator to verify the 10/9/3 oracle.
- Executed the pinned inference argument group and inference config factory in
  isolation: all three inference cases' config expectations pass. This check does
  not execute full argument validation or checkpoint loading.
- Executed the pinned TP field declarations and ArgumentGroupFactory with the
  parser's actual exclusion list: all six TP-overlap cases' CLI schemas pass.
  Help-text extraction alone was omitted for the in-memory class definitions.
- `git diff --check` passes. GPU dependency, construction, and runtime validation
  are still pending; optional NVRX/TE/Chakra capabilities must not be silently skipped.

For further expansion, first validate new fixtures/oracles against a pinned
baseline, then use the same harness against baseline/candidate. Group by rank
requirement and classify baseline errors separately from regressions. Do not
treat "both failed" as equivalent. Current GPU evidence is recorded in the
[PR7778 report](DIST_RESULTS.md), not in the historical checks above.

## Added cases

Use the existing `--cases` selection; do not run the entire manifest in a one-GPU
allocation. GPU requirements below include checkpoint seed scenarios.

| Case | Tier | Ranks |
| --- | --- | ---: |
| `coverage_vlm_freeze_language` | runtime | 1 |
| `coverage_vlm_projection_only` | runtime | 1 |
| `coverage_vlm_packed_cp` | runtime | 2 |
| `coverage_vlm_decoder_overlap` | runtime | 2 |
| `coverage_tp_overlap_default` | runtime | 2 |
| `coverage_tp_overlap_file` | runtime | 2 |
| `coverage_tp_overlap_bulk_off` | runtime | 2 |
| `coverage_tp_overlap_collectives_off` | runtime | 2 |
| `coverage_tp_overlap_legacy_config` | builder | 2 |
| `coverage_inference_static_config` | builder | 1 |
| `coverage_inference_dynamic_config` | builder | 1 |
| `coverage_inference_resume_config` | builder | 1 |
| `coverage_tokenizer_vocab_file` | builder | 1 |
| `coverage_checkpoint_override_iteration` | runtime | 1 |
| `coverage_checkpoint_local_config` | builder | 1 |
| `coverage_scheduler_wsd_iters` | builder | 1 |
| `coverage_scheduler_wsd_samples` | builder | 1 |
| `coverage_scheduler_fraction_inverse_sqrt` | builder | 1 |
| `coverage_scheduler_constant_cosine_wd` | builder | 1 |
| `coverage_scheduler_resume_checkpoint_policy` | runtime | 1 |
| `coverage_validation_batches` | runtime | 1 |
| `coverage_validation_full` | builder | 1 |
| `coverage_validation_only_config` | builder | 1 |
| `coverage_validation_multiple_config` | builder | 1 |
| `coverage_train_gc_memory` | runtime | 1 |
| `coverage_train_signal_duration_config` | builder | 1 |
| `coverage_train_decrease_batch` | builder | 1 |
| `coverage_train_dp_hash` | runtime | 2 |
| `coverage_ddp_average_buckets` | runtime | 2 |
| `coverage_ddp_bf16_local_fp32_config` | builder | 1 |
| `coverage_optimizer_precision_aware` | runtime | 1 |
| `coverage_optimizer_muon_layout_config` | builder | 1 |
| `coverage_optimizer_graph_config` | builder | 1 |
| `coverage_optimizer_interleaved_overlap_config` | builder | 2 |
| `coverage_dist_controls` | runtime | 1 |
| `coverage_dist_flight_recorder` | runtime | 1 |
| `coverage_dist_rank_mapping` | runtime | 4 |
| `coverage_fsdp_registration_config` | builder | 1 |
| `coverage_fsdp2_reshard_config` | builder | 1 |
| `coverage_dataset_workers_cyclic` | runtime | 1 |
| `coverage_dataset_cache_config` | builder | 1 |
| `coverage_dataset_masks_runtime` | runtime | 1 |
| `coverage_dataset_inter_document_config` | builder | 1 |
| `coverage_dataset_fim_config` | builder | 1 |
| `coverage_logger_local_controls` | runtime | 1 |
| `coverage_logger_disabled_service_metadata` | builder | 1 |
| `coverage_logger_moe_trace` | runtime | 1 |
| `coverage_checkpoint_async_worker` | runtime | 1 |
| `coverage_checkpoint_integrity_seed` | runtime | 1 |
| `coverage_checkpoint_integrity_resume` | runtime | 1 |
| `coverage_checkpoint_parallel_load` | runtime | 2 |
| `coverage_checkpoint_optimizer_reshard` | runtime | 2 |
| `coverage_checkpoint_nonpersistent_global` | runtime | 1 |
| `coverage_checkpoint_save_policy` | runtime | 1 |
| `coverage_checkpoint_pretrained_main_params` | runtime | 1 |
| `coverage_checkpoint_cache_config` | builder | 1 |
| `coverage_checkpoint_debug_intervals_config` | builder | 1 |
| `coverage_profiling_chakra_config` | builder | 1 |
