# RNGConfig ownership results

Validated on **2026-10-06** after pushing the Hybrid-MoE golden correction in
[#7591](https://github.com/NVIDIA/Megatron-LM/pull/7591).

- Frozen main baseline: `fbbe6eb5490de4c81c51bd8f715f333c91837fe4`.
- Candidate: `bc02ca029c89f6452b8b80d6744f95144b861566`.
- Both sides: H100 GPUs, `nvcr.io/nvidian/nemo:26.10.rc0`.
- Current capture-only harness: `4ace84a5befc2b6983bd67c2433aebe24b8efb54`.

Only the additional args/configuration capture-and-compare suite ran.
**No product unit tests or harness self-tests ran.**

## Results

All **53 unique manifest cases** completed: **50 exact passes, two explained
configuration mismatches, and one expected CLI rejection**. There are no remaining
capture errors after correcting and rerunning the invalid profiler fixture below.
The strict comparator retains the two mismatches; this is not an all-identical run.

| Selection | Cases | Result |
|---|---:|---|
| One-rank builder/configuration captures | 29 | 27 exact passes, one tracker mismatch, one expected rejection |
| One-rank runtime captures | 17 | 16 exact passes, one tracker mismatch |
| Two-rank runtime captures | 6 | All exact passes on both ranks |
| Eight-rank TP2/PP2/DP2 runtime capture | 1 | Exact pass on all ranks |

The two-rank cases are `rng_dp_fresh`, `rng_dp_resume`, `runtime_moe_ep2`,
`runtime_context_parallel2`, `runtime_tp2_sequence_overlap`, and
`runtime_interleaved_pipeline`. The eight-rank case is `runtime_tp2_pp2_dp2`.
Single-rank coverage includes tokenizer/vocabulary resolution, batch schedules,
checkpoint precedence and resume, logging, profiling, dense and MoE configurations,
Hybrid/Mamba, recomputation, FSDP2 configuration, fine-tuning, and VLM training.
See [cases.json](cases.json) for the complete selection and options.

### Reviewed tracker differences

Both cases explicitly enable `--te-rng-tracker`. Centralized propagation now
correctly sets the model's differently named `use_te_rng_tracker` field to `true`;
the baseline args adapter left that field `false`.

| Case | Exact differing paths, all `false` → `true` |
|---|---|
| `rng_te_tracker` | `rank[0].config.model[0].fields.transformer.fields.use_te_rng_tracker`; `rank[0].consumer.model[0].config.fields.use_te_rng_tracker`; `rank[0].consumer.ddp[0].config.fields.use_te_rng_tracker` |
| `hybrid_te_tracker` | `rank[0].config.model[0].fields.transformer.fields.use_te_rng_tracker` |

These are the only configuration/consumer differences. All captured seeding inputs
and RNG states match, including the runtime tracker case. RNG observations record
Python/NumPy state and hashes of CPU/CUDA/tracker state without drawing random values.
Resume comparisons use the same baseline-produced checkpoint on both sides.

The added `hybrid_te_tracker` builder capture covers the mapping inconsistency
exposed by the Hybrid-MoE golden. It does not construct the large unit-test model
or rerun that unit test. No mismatch was suppressed or converted into an exact pass.

### Expected rejection and fixture correction

`invalid_batch_conflict` rejects the simultaneous explicit `--global-batch-size`
and `--step-batch-size-schedule` flags on both revisions, with the expected error.

The original `profiling_excluded_rank` fixture used start/end steps `2/2`.
Both revisions rejected this with
`PyTorch profiling requires profile_step_end > profile_step_start`.
The harness-only correction changes the end to `3`, retaining excluded rank `1`.
Rerunning this case against the same frozen product commits gives an exact pass.
The original error remains in the archived first-run report; the summary above
uses its corrected rerun, not an additional unique case. No product fix was needed.

## Provenance and reproduction

The original 52 cases ran with harness `dda469fc5585eb2f5202a02395b98484c7b776e2`.
The added Hybrid capture ran with `fa6df0401e2f247954917fc9d6f02518c43b2e73`;
the corrected profiler capture ran with `4ace84a5befc2b6983bd67c2433aebe24b8efb54`.
The capture, comparison, and snapshot Python implementations are identical across
these revisions. Only the two manifest changes described above affect the captures.
The temporary unit-test runner was removed; it was never invoked in this validation.

Follow the [README](README.md), substituting the pinned baseline and candidate above.
Run the 46 one-rank, six two-rank, and one eight-rank cases in appropriately sized
allocations. Use only `compare_config_seams.py`; do not invoke unit-test runners.
Reports retain revision/tree IDs, environment, fixture and harness hashes,
per-case outcomes, raw field differences, and capture errors. Detailed rank logs,
snapshots, and shared baseline checkpoints are archived separately, not committed.

This is capture/compare validation, including short real training for runtime-tier
cases. Builder cases do not train. Full unit-suite, exhaustive model-family,
convergence, and performance validation are not claimed. Equivalent CLI captures
alone do not prove that every consumer treats native config as authoritative.
Earlier reports remain available in Git history.
