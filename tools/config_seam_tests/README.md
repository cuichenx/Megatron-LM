# Configuration equivalence across MLM revisions

This fork-only harness supports section 1.1 testing items 1–2: compare configuration
objects and resolved inputs across a representative CLI matrix. It is **not part of
PR7418 or the profiling ownership PR**, does not modify either tested checkout,
and does not need to land in MLM main.

## Run

Use the same supported MLM development environment for both revisions (Python 3.12,
PyTorch, Transformer Engine, tokenizer dependencies, and CUDA). No dependency installation
or remote model download is performed by the harness. `uv` should use that existing environment.

Freeze one main commit and compare it with that same commit plus the PR. Latest
main is not required; do not advance the baseline between runs. Prepare two clean
worktrees at exact commits, then run from this directory:

```bash
uv run --no-sync python -m unittest -v test_snapshot
uv run --no-sync python compare_config_seams.py \
  --baseline /path/to/main-worktree \
  --candidate /path/to/pr-worktree \
  --baseline-revision d564dd01de4549a4884be26b1d74e169798fd877 \
  --candidate-revision f0148a3600818588406d8ca6437a8de1f8377f38 \
  --environment-id YOUR_CONTAINER_REFERENCE_OR_DIGEST \
  --output /path/to/new-result-directory \
  --cases defaults known_padded known_unpadded hf_derived hf_unpadded
```

The driver launches each side through `python -m torch.distributed.run` in its own
process group. Allocate one GPU for builder and single-rank runtime cases. Allocate
the manifest's specified rank count for distributed runtime cases; do not hold idle
GPUs while running the one-rank matrix. The output directory must not already exist.
Use node-local storage for builds, caches and temporary results. Set `HF_HOME`,
`TORCH_EXTENSIONS_DIR`, `TRITON_CACHE_DIR`, `XDG_CACHE_HOME` and `TMPDIR` accordingly.
The tokenizer fixture is created locally and offline. All logger cases use local
TensorBoard only; external loggers and telemetry are disabled.
For Hopper TP/CP cases, set `CUDA_DEVICE_MAX_CONNECTIONS=1`. The FSDP2 builder case
explicitly overrides this to 32 for both child processes to satisfy MLM validation.

Select additional cases by name from `cases.json`. Omit `--cases` only when enough
GPUs are allocated for every case. Cases needing checkpoint arguments/resume first
produce a tiny checkpoint using **the baseline revision**, then supply that same
checkpoint to both sides. Never provide an untrusted checkpoint to MLM's legacy loader.
`checkpoint_cli` retains explicit CLI configuration without `--use-checkpoint-args`;
`checkpoint_args` exercises MLM's existing per-field precedence with that flag and
competing explicit values. `runtime_resume` actually restores the shared checkpoint.
The step-batch case samples immediately before/at/after 8 consumed samples (256 tokens
at sequence length 32); the legacy ramp case preserves its existing deprecated behavior.

## What is measured

For the profiling ownership pilot, use the same command above with the chosen pinned
baseline/candidate revisions and select these one-GPU runtime cases:

```bash
--cases runtime_fresh runtime_resume profiling_disabled profiling_excluded_rank \
  profiling_nsys profiling_pytorch profiling_memory profiling_resume
```

Use the exact #7418 parent to isolate C01, and the frozen main ancestor for the
cumulative comparison. Neither comparison requires a different capture implementation.

For logging ownership, select `runtime_fresh runtime_resume logging_metrics
logging_attention logging_resume`. The added one-GPU cases exercise local
TensorBoard/timer construction, parameter/gradient metrics, throughput/memory
logging, max-attention-logit instrumentation, save and current-run settings on
resume. They observe real writer/timer constructor inputs, not nondeterministic
elapsed times or GPU-memory measurements. External W&B/one-logger/OTel services
remain disabled and need separate behavioral tests with controlled service doubles.
The resume case disables level-1 timing barriers for the current run, while the
metrics case retains the default enabled policy. Inspector startup is covered by
controlled behavioral tests rather than opening a network listener in this suite.
See [the LoggerConfig report](LOGGER_RESULTS.md) for the pinned five-case result;
the older profiling and initialization reports do not validate these changes.

`capture_config_seams.py` runs the checkout's actual `pretrain_gpt.py` (or
`pretrain_hybrid.py` for Hybrid builder cases, or `pretrain_vlm.py` for the VLM case) as `__main__`.
It observes its existing initialization path instead of recreating the old/new order.
The same capture implementation is used for both revisions.

- **Builder tier:** actual parser/validation, tokenizer and microbatch services,
  container translation, the GPT dataset-config builder, and real scheduler
  derivation against a minimal parameter-group fixture. No model, optimizer or DDP
  object is constructed, and no training is claimed.
- **Runtime tier:** delegates to actual `pretrain`, observing model/DDP/optimizer
  constructor arguments, dataset configuration, scheduler inputs and real checkpoint
  save/load completion during short training runs. Every launched rank must report.
- **CLI rejection:** a named invalid flag combination must be rejected on both sides
  with the expected validation error; this is not a successful configuration capture.

All dataclass fields are encoded recursively. Tokenizers use type/vocabulary/special-ID
descriptors; callable names, defaults and closure values are retained. Bound methods
record their function and owner type, not the owner's mutable runtime state. Timer
services record logging settings, not elapsed wall-clock measurements. Unsupported
objects fail explicitly. Source/fixture/run roots are replaced only at path boundaries.
DDP layout parameter references use model-local names, shapes, dtypes and trainability,
not memory addresses or tensor values. Unregistered parameter references still fail.
Timing and provenance are not configuration values. Missing and null, tuples and lists,
integers and booleans remain distinct. No numeric tolerance or broad field ignore list.

For the logging ownership migration, `config.logger` retains every native field
and additionally observes ten formerly CLI-only one-logger/OTel/inspector settings from
their actual legacy owner. Once a field exists in the dataclass schema, its native
value is authoritative, including null; stale CLI values cannot replace it.
Missing legacy inputs fail rather than receiving invented defaults. The raw native
field list is recorded separately as `logger_schema_fields` provenance, making the
schema expansion visible without treating a changed storage location as a changed
setting. This observation adapter does not reproduce production normalization or
exclude unknown/new fields. Self-tests exercise changed values, missing inputs,
native nulls and future fields.

The configuration surfaces include model TransformerConfig, OptimizerConfig, DDPConfig,
SchedulerConfig, OptimizerParamScheduler, CheckpointConfig, LoggerConfig, TokenizerConfig
and GPTDatasetConfig, plus ProfilingConfig for the ownership pilot. Enabled profiling
cases also observe real profiler-schedule and memory-snapshot calls. These captures
work on both sides without reproducing production profiling calculations.
Checkpoint/logger configuration captures are not an exhaustive
audit of every downstream consumer. Constructor input equality does not prove complete
model/process-group/control-flow equivalence. This harness supplements behavioral unit
tests and training/resume/convergence/performance validation, not replaces them.

## Results and extension

See [the PR7418 results](RESULTS.md) for the current pinned before/after comparison.
See [the profiling ownership results](PROFILING_RESULTS.md) for the separate C01 pilot.
Earlier results remain available in Git history.
See [scenario coverage](SCENARIOS.md) for the expanded recipe matrix, pinned source
provenance, GPU requirements and limits of each testing tier.

`report.json` records the exact revisions/trees, harness and fixture hashes, environment
identifier, per-case outcome and field-level differences. Case subdirectories contain
expanded argv, per-rank snapshots and logs. A missing capture, timeout or failed baseline
is an error, not a pass. Inspect baseline failures separately from candidate regressions.
Both sides failing does not establish equivalence. Comparing a newer main against
an older PR head can include unrelated upstream differences. Use a shared fixed
base for the equivalence gate; retain any historical mismatches and their attribution,
never silently whitelist them.

Keep the harness/comparison rules separate from migration PRs. Add explicit capture
adapters if APIs move, without reproducing production calculations. Add a named manifest
case for new configuration semantics and required captures so omissions cannot silently
pass. Self-tests deliberately change vocabulary/scheduler/DDP values and omit captures.

Out of scope: full convergence/performance CI, exhaustive model families, real packed
multimodal data pipelines, RL/distillation configs, and TrainState-authority checks.
Do not commit raw run artifacts: they can contain private paths or environment metadata.
