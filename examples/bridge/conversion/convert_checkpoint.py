# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Experimental Super VL/text-only HF ↔ MCore checkpoint entrypoint.

Run with ``python -m torch.distributed.run``. No Bridge training package, Slurm
launcher, remote-code trust, or tokenizer download is implicitly enabled.
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

from megatron.bridge import AutoBridge
from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed


def main() -> None:
    """Import or export the supported builder-backed model using explicit paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["import", "export"])
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument("--hf-output")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--save-tokenizer", action="store_true")
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--expert-model-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.direction == "export" and args.hf_output is None:
        parser.error("export requires --hf-output")
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
    )
    model_parallel_cuda_manual_seed(1234)
    try:
        # lazy HF tensors, not a second materialized Hugging Face model.
        bridge = AutoBridge.from_hf_pretrained(
            args.hf_model,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            text_only=args.text_only,
        )
        overrides = dict(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
        )
        if args.direction == "import":
            config = bridge.get_model_config()
            for name, value in overrides.items():
                if not hasattr(config.transformer, name):
                    raise ValueError(f"Unknown transformer field: {name}")
                setattr(config.transformer, name, value)
            model = bridge.get_model(config, wrap_with_ddp=False)
            bridge.save_megatron_model(
                model,
                args.megatron_path,
                hf_tokenizer_path=args.hf_model if args.save_tokenizer else None,
                hf_tokenizer_kwargs={
                    "trust_remote_code": args.trust_remote_code,
                    "revision": args.revision,
                },
            )
        else:
            model = bridge.load_megatron_model(args.megatron_path, mp_overrides=overrides)
            bridge.save_hf_pretrained(model, args.hf_output, strict=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
