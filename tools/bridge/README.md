# [DO NOT MERGE] Training-loop / Super VL integration experiment

This fork branch combines the following **pinned** changes. It is not a proposal
to merge these PRs as one patch, and does not change their original branches.

| Input | Pinned revision |
| --- | --- |
| MLM main | `7f510b60c04bdda411019f581f20bda78cc250a6` |
| #8205 — DDP/FSDP | `73cd9648c5850cb89b172c84bfce1765cf054f7c` |
| #8234 — model config | `872898a72482a37770dded87366823e7e2f49c41` |
| #8978 — dataset inputs | `436688a90eac46fbb0ad467cd1e31c2697f92e2e` |
| #8851 — fault tolerance | `b77007d5f92feed6d429be20f4cdb1b4bb8b63f0` |
| #8340 — ModelOpt builders | `8a9e22443ff30b5be187d3a40ee4719f68576aa5` |
| #8444 — VLM inputs | `b3d041b74c9b8047ea1890a2a2a1a5d4f5eef48e` |

## Narrow Bridge port

`megatron/bridge` contains the native `Nemotron35SuperVLBridge`, its Nemotron-H
text-only projection, model builders, AutoBridge and the shared conversion
dependency closure. The source is Apache-2.0 Megatron Bridge at
[`53c20f81`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/53c20f81cb575f19af34ff0ae1749fb13b07999b).
Copied files carry source links and notices.

Only Super VL and Nemotron-H are registered. No other Omni variant, audio
encoder, MIMO model, recipe library or Bridge training loop is imported. The
shared expanded-sequence model still contains dormant optional audio handling;
there is no supported audio construction path in this port. Shared conversion
code retains its existing provider/PEFT utilities, but PEFT checkpoint I/O is
explicitly unsupported here.

This package intentionally occupies `megatron.bridge`. Run from this checkout
with `PYTHONPATH=.`; do not combine this experiment with an independent Bridge
installation and assume both packages are active.

## Offline probe

Inside a compatible GPU development environment:

```bash
PYTHONPATH=. uv run --no-project python -m torch.distributed.run \
  --nproc-per-node=1 tools/bridge/smoke_super_vl.py

# Additionally build the full RADIO encoder with a tiny language model:
PYTHONPATH=. uv run --no-project python -m torch.distributed.run \
  --nproc-per-node=1 tools/bridge/smoke_super_vl.py --vision
```

The probe uses random reduced language weights, checks local import provenance,
native config serialization, forward/backward, native weight save/load, and
Super VL's text-only safetensors projection. It downloads no model artifacts.
It is **not** a full pretrained Super VL quality test, a convergence result,
or validation of the unified `pretrain()` lifecycle.

## Conversion entrypoint

`convert_checkpoint.py` uses the ported AutoBridge mappings and MCore
distributed checkpoint I/O. The native output stores `model`/`modelN` weights,
checkpoint version/iteration and MLM's `run_config.yaml`, publishing the tracker
only after successful save. It does not install global training args.

For an already accessible HF checkpoint, using an appropriately sized GPU job:

```bash
PYTHONPATH=. uv run --no-project python -m torch.distributed.run \
  --nproc-per-node=8 tools/bridge/convert_checkpoint.py import \
  --hf-model /models/super-vl --megatron-path /checkpoints/super-vl \
  --tensor-model-parallel-size 8

# Same source, but import only its language weights:
PYTHONPATH=. uv run --no-project python -m torch.distributed.run \
  --nproc-per-node=8 tools/bridge/convert_checkpoint.py import \
  --hf-model /models/super-vl --megatron-path /checkpoints/super-text \
  --tensor-model-parallel-size 8 --text-only

PYTHONPATH=. uv run --no-project python -m torch.distributed.run \
  --nproc-per-node=8 tools/bridge/convert_checkpoint.py export \
  --hf-model /models/super-vl --megatron-path /checkpoints/super-text \
  --hf-output /exports/super-text --tensor-model-parallel-size 8 --text-only
```

GPU counts above are illustrative, not a measured capacity recommendation.
The public source is `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`.
Use a pinned `--revision` for a Hub source. `--trust-remote-code` and
`--save-tokenizer` are explicit opt-ins. Native HF Super VL configs require
Transformers >=5.18; older environments need the original remote-code config.

Initial scope is builder-backed `torch_dist` weights only. Legacy args/provider
reconstruction, optimizer/resume state, ModelOpt checkpoint transforms, PEFT
checkpoint I/O and destructive low-memory saves are not supported. Loading
preserves saved topology unless explicit overrides are supplied. Full-scale
pretrained conversion, multi-rank conversion and training-loop integration
remain validation work, not claims made by the reduced probe.

## Cross-PR integration adjustments

- Preserve model-owned topology, tokenizer-owned padded vocabulary and the
  config finalization/validation separation.
- Use dataset sample length rather than position capacity or stale CLI values
  for the combined TP-buffer, JIT, FLOP and memory-estimate consumers.
- Preserve class-selected builders when deserializing configs.
- Keep ModelOpt transformations on stock GPT/Hybrid builders. Teacher checkpoint
  reads use an isolated operation policy, including the inner metadata/read
  calls, so student step/non-persistent selectors do not leak into them.
- Adapt existing targeted fixtures to the combined contracts; no per-config
  synthetic ownership-test files were introduced.

## Initial validation (2026-11-06)

Using the cached NeMo `26.08.rc9` environment with network access disabled:

- 913 targeted tests passed; 314 skipped in the single-GPU configuration. This
  selection covers config finalization/serialization, model builders, dataset
  ownership/batching, checkpoint policy, ModelOpt and runtime initialization.
  It is neither the full unit suite nor the fork-only argument seam suite.
- The offline probe passed native config round-trip, reduced Mamba/attention/MoE
  forward/backward and exact native save/load of all 46 parameter tensors.
- Super VL's text-only projection passed HF safetensors export/import with exact
  parameter comparison, using locally generated random weights.
- Full RADIO plus projector and a reduced language model (653,647,564 total
  parameters) passed a random-image forward with output shape `(1, 16, 256)`.
- Independent integration and port reviews were completed and findings fixed.
  Scoped import sorting, formatting and undefined-name checks pass. All-files
  pre-commit still reports unrelated baseline formatting/Core Pylint findings;
  these were not copied into this experiment.

These checks do not establish pretrained HF/MCore forward parity, full-size
120B conversion, multi-rank behavior, or unified `pretrain()` execution.
