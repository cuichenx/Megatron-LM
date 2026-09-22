# Scenario coverage and recipe provenance

The matrix contains 52 cases: the original 19 initialization/configuration cases,
19 recipe-inspired additions below, six profiling, three logging, and five RNG
ownership cases. These are small, offline configuration-seam tests,
not reproductions of production model quality or throughput. Keep the feature
combinations while reducing layers, hidden sizes, experts, sequence length and steps.

Sources are pinned in `cases.json` under `source_revisions`; each added case references
a `sources` entry. The sources are public
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/cfa9e20d2658844dd84a07f2ccd34d7e62ab5552)
and [Megatron-MoE-ModelZoo](https://github.com/yanring/Megatron-MoE-ModelZoo/tree/c2d0171e5d8c9f4da9b00bd44c05317fa20a4fcd).
These are source recipes, not an instruction to execute their launchers, download
their assets, or use their cluster settings. Only main-branch recipes were selected.

## Added builder cases

All use one GPU for real MLM initialization. They stop before model/DDP/optimizer
construction. Flags being accepted and translated do **not** establish kernel or
distributed-runtime support.

| Case | Retained configuration combination | Source ID |
|---|---|---|
| `dense_llama_gqa` | GQA, RoPE, SwiGLU, RMSNorm, untied embeddings, decoupled LR | `mlm-llama` |
| `fp16_static_scaling` | FP16, static loss scaling, local transformer, QK scaling | `mlm-fp16` |
| `fp8_delayed` | Hybrid FP8, delayed recipe, amax history, FP8 parameter gather | `mlm-llama` |
| `moe_mixtral_aux` | Top-2 routing, aux loss, grouped GEMM, permutation fusion, GQA | `zoo-mixtral` |
| `moe_qwen3_topk` | QK norm, FP32 top-k router, local HF tokenizer | `zoo-qwen3` |
| `moe_deepseek_mla` | MLA, Q/KV LoRA, grouped sigmoid routing, expert bias, shared experts | `zoo-deepseek` |
| `moe_capacity_drop` | Capacity factor, probability-based dropping, padding | `mlm-moe-capacity` |
| `moe_sinkhorn` | Sinkhorn top-1 balancing | `mlm-moe-overlap` |
| `qwen_next_gdn` | GDN, gated attention, shared expert gate, MTP, QK-norm weight decay | `zoo-qwen-next` |
| `hybrid_mamba` | Mamba/attention/MLP hybrid stack, no positional embeddings | `mlm-mamba` |
| `selective_recompute` | Selective core-attention/layernorm recomputation | `mlm-selective-recompute` |
| `torch_fsdp2_config` | Torch FSDP2 configuration and distributed checkpoint format | `mlm-fsdp2` |

The scheduler fixture checks derivation but does not construct the production parameter
groups. For example, decoupled LR and Qwen-Next QK-norm weight-decay settings are captured as
configuration, not validated as actual parameter-group assignments. Capacity/drop and
Sinkhorn cases are deliberate routing variants inspired by MLM test coverage, not
literal copies of a whole production recipe. Qwen top-k and expert counts are reduced.
The ModelZoo Qwen-Next recipe's obsolete `--no-weight-decay-cond-type qwen3_next`
is rejected by both pinned MLM revisions. Its documented QK-layernorm weight-decay
intent is expressed using the supported `--apply-wd-to-qk-layernorm` flag; this does
not claim equivalence with a historical optimizer implementation.
Likewise, the recipe's obsolete `--linear-attention-type gated_delta_net` becomes
the current `--experimental-attention-variant gdn` selector.
FSDP2 uses BF16 (these revisions reject FP16 with FSDP2), disables
gradient-accumulation fusion, unties embeddings, and sets
`CUDA_DEVICE_MAX_CONNECTIONS=32` for its isolated processes on both sides, as required
by MLM validation. It does not inherit the TP/CP lane's setting of 1. This environment
override is recorded in the case and snapshot; the driver rejects other override keys.

## Added runtime cases

These execute short training/evaluation through the real entrypoint and observe
constructor inputs. All must finish successfully on each side before comparison counts.

| Case | GPUs | Executed scenario | Source ID |
|---|---:|---|---|
| `runtime_full_recompute` | 1 | Full uniform activation recomputation | `mlm-full-recompute` |
| `runtime_finetune` | 1 | Load baseline weights, reset optimizer/RNG policy, override LR scheduler | `zoo-mixtral` |
| `runtime_vlm_frozen_vision` | 1 | Actual `pretrain_vlm.py`, mock images/text, frozen vision encoder | `mlm-vlm` |
| `runtime_moe_ep2` | 2 | Expert parallelism, grouped GEMM, all-to-all, distributed optimizer, DDP overlap | `mlm-moe-overlap` |
| `runtime_context_parallel2` | 2 | Context parallelism and flash attention | `mlm-context-parallel` |
| `runtime_tp2_sequence_overlap` | 2 | TP2 plus sequence parallelism and DDP gradient/parameter overlap | `mlm-moe-overlap` |
| `runtime_interleaved_pipeline` | 2 | PP2 with two virtual stages per rank | `mlm-interleaved` |

The fine-tuning case tests checkpoint/optimizer/scheduler policy on the tiny dense model;
it is not a Mixtral fine-tuning quality test. The sequence-parallel case does not enable
the separate `tp_comm_overlap` backend. The VLM case requires both VLM constructor and
freeze captures in addition to the usual model, dataset, optimizer and scheduler seams.
Mock multimodal training is not a validation of a real packed SFT data pipeline.

## Profiling ownership pilot

Six additional one-GPU runtime cases exercise the existing MLM profiler modes:
`profiling_disabled` (PyTorch option without `--profile`), `profiling_nsys`
(CUDA-profiler window, NVTX, shape recording), `profiling_pytorch` (shape/callstack
collection and trace export), `profiling_memory` (memory-history snapshots), and
`profiling_resume` (current-run profiling policy after loading the shared baseline checkpoint).
`profiling_excluded_rank` preserves the accepted no-op behavior when no launched rank
is selected, even if the unused PyTorch window would be invalid on a selected rank.
All captures include `ProfilingConfig`; the enabled PyTorch and memory cases also
require actual schedule or snapshot-call captures. CUDA-profiler API execution is
tested without an external Nsight capture. Chakra and
deliberately removed/divergent legacy args are covered by targeted product unit tests,
not these one-rank comparison cases. Case presence is not a validation result.

## RNG ownership

`rng_custom_seed`, `rng_te_tracker`, and `rng_inference_tracker` use one GPU
to exercise nondefault seeding and each optional tracker. `rng_dp_fresh` and
`rng_dp_resume` use two GPUs and data-parallel random initialization, including
rank-specific checkpoint RNG restore. The latter explicitly selects the former
as its baseline checkpoint seed scenario. All runtime cases capture exact RNG
state after seeding and at relevant save/load boundaries. No random draws are
added by the observer. Case presence is not a validation result.

## Deliberately separate follow-up coverage

- MIMO packed multimodal data and production SFT recipes need accessible datasets,
  pretrained assets and a separate dataset-content correctness oracle.
- DeepEP/HybridEP, MoE communication-overlap backends, Blackwell MXFP8/FP4, and full
  FSDP training require backend/hardware-specific runtime lanes.
- MLA/GDN/Mamba forward/backward correctness, MTP loss behavior, and FP8 numerics need
  actual runtime tests beyond the builder cases above.
- Longer convergence, performance, checkpoint resharding, and fault recovery remain
  separate from section 1.1 items 1–2.

Use `--cases` to select an allocation-sized subset; do not reserve eight GPUs while
running the one-GPU builder matrix. See [results](RESULTS.md) for what has actually
been validated, including failures and differences. Case presence alone is not a pass.
