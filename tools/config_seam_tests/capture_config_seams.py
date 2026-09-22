# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Observe a checkout's real GPT entrypoint without changing its source files."""

import argparse
import dataclasses
import functools
import hashlib
import inspect
import json
import os
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

from snapshot import encode, logger_settings


def build_dataset_config(dataset_provider, args):
    """Call the observed checkout's factory through either callback signature."""
    factory = inspect.unwrap(dataset_provider).__globals__["core_gpt_dataset_config_from_args"]
    inputs = {}
    if "random_seed" in inspect.signature(factory).parameters:
        if not isinstance(dataset_provider, functools.partial) or "random_seed" not in dataset_provider.keywords:
            raise ValueError("Dataset callback is missing its explicit random_seed binding")
        inputs["random_seed"] = dataset_provider.keywords["random_seed"]
    return factory(args, **inputs)


def main() -> None:
    """Run one case/rank and write a snapshot even if initialization fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    options = parser.parse_args()
    case = json.loads(options.case_file.read_text())
    entrypoint = case.get("entrypoint", "pretrain_gpt.py")
    rank = int(os.environ.get("RANK", "0"))
    options.output.mkdir(parents=True, exist_ok=True)
    options.run_root.mkdir(parents=True, exist_ok=True)
    roots = {
        str(options.repo.resolve()): "<repo>",
        str(options.fixtures.resolve()): "<fixtures>",
        str(options.checkpoint.resolve()): "<checkpoint>",
        str(options.run_root.resolve()): "<run>",
    }
    result = {"schema": 1, "case": case["name"], "tier": case["tier"], "rank": rank, "captures": {}, "status": "error"}

    def record(name, value, references=None):
        result["captures"].setdefault(name, []).append(encode(value, roots, references))

    def observe_call(owner, name, capture_name, excluded=(), after=None):
        original = getattr(owner, name)
        signature = inspect.signature(original)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            references = None
            if capture_name == "consumer.ddp":
                # Layouts refer to live Parameters. Compare their model-local identity and
                # metadata, not weights, addresses, or just shapes (which can alias).
                references = {
                    id(parameter): {
                        "parameter": parameter_name,
                        "shape": list(parameter.shape),
                        "dtype": str(parameter.dtype),
                        "requires_grad": parameter.requires_grad,
                    }
                    for parameter_name, parameter in bound.arguments["module"].named_parameters()
                }
            record(
                capture_name, {key: value for key, value in bound.arguments.items() if key not in excluded}, references
            )
            output = original(*args, **kwargs)
            if after is not None:
                after(args, output)
            return output

        setattr(owner, name, wrapped)

    try:
        sys.path.insert(0, str(options.repo.resolve()))
        os.chdir(options.repo)
        import torch

        import megatron.training as package
        import megatron.training.global_vars as globals_
        from megatron.core import num_microbatches_calculator as batches
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
        from megatron.core.distributed import DistributedDataParallel
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.training import training
        from megatron.training.config import TokenizerConfig

        if not Path(training.__file__).resolve().is_relative_to(options.repo.resolve()):
            raise RuntimeError("Imported training code from the wrong checkout")
        result["environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_max_connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS"),
        }
        observe_call(globals_, "init_num_microbatches_calculator", "runtime.microbatch_inputs")

        original_tokenizer = globals_.build_tokenizer

        def tokenizer(args, **kwargs):
            record(
                "runtime.tokenizer_inputs",
                {f.name: getattr(args, f.name) for f in dataclasses.fields(TokenizerConfig) if hasattr(args, f.name)},
            )
            output = original_tokenizer(args, **kwargs)
            record("runtime.tokenizer", output)
            return output

        globals_.build_tokenizer = tokenizer
        observe_call(training, "OptimizerParamScheduler", "consumer.scheduler", excluded=("optimizer",))
        original_dataset_post_init = GPTDatasetConfig.__post_init__

        def dataset_post_init(self):
            original_dataset_post_init(self)
            record("consumer.dataset", self)

        GPTDatasetConfig.__post_init__ = dataset_post_init
        original_pretrain = training.pretrain

        def entry(cfg, dataset_provider, *args, **kwargs):
            cli = globals_.get_args()
            record("config.model", cfg.model)
            for field in (
                "optimizer",
                "ddp",
                "scheduler",
                "checkpoint",
                "logger",
                "tokenizer",
                "train",
                "validation",
                "profiling",
                "rng",
            ):
                value = getattr(cfg, field)
                if field == "logger":
                    result["logger_schema_fields"] = [field.name for field in dataclasses.fields(value)]
                    value = logger_settings(value, cli)
                record("config." + field, value)
            record(
                "runtime.services",
                {
                    "tokenizer": globals_.get_tokenizer(),
                    "tensorboard": globals_.get_tensorboard_writer() is not None,
                    "wandb": globals_.get_wandb_writer() is not None,
                    "timers": globals_.get_timers() is not None,
                    "energy": globals_.get_energy_monitor() is not None,
                },
            )
            record(
                "runtime.batch",
                {
                    "global": batches.get_current_global_batch_size(),
                    "microbatches": batches.get_num_microbatches(),
                    "resolved_global": cli.global_batch_size,
                    "eval_global": cli.eval_global_batch_size,
                    "eval_micro": cli.eval_micro_batch_size,
                },
            )
            if case["tier"] == "runtime":
                return original_pretrain(cfg, dataset_provider, *args, **kwargs)
            # Builder tier: execute the real dataset builder and scheduler derivation,
            # but do NOT claim model/optimizer/DDP construction or checkpoint I/O.
            build_dataset_config(dataset_provider, cli)
            optimizer = SimpleNamespace(param_groups=[{"default_config": True}, {"wd_mult": 0.0}])
            scheduler = training.get_optimizer_param_scheduler(optimizer)
            for target in sorted(
                {0, int(scheduler.lr_warmup_steps), int(scheduler.lr_decay_steps), int(scheduler.lr_decay_steps) + 1}
            ):
                scheduler.step(target - scheduler.num_steps)
                record("builder.scheduler_points", {"samples": target, "groups": optimizer.param_groups})
            for consumed in case.get("batch_points", [0, 4, 8, 16]):
                batches.update_num_microbatches(consumed, consistency_check=False)
                record(
                    "builder.batch_points",
                    {
                        "consumed": consumed,
                        "global": batches.get_current_global_batch_size(),
                        "microbatches": batches.get_num_microbatches(),
                    },
                )

        package.pretrain = entry
        training.pretrain = entry
        if case["tier"] == "runtime":
            from torch.utils.tensorboard import SummaryWriter
            from megatron.training import initialize
            from megatron.core import tensor_parallel
            import numpy as np
            import random

            def rng_snapshot():
                def tensor_digest(state):
                    state = tensor_parallel.convert_cuda_rng_state(state, to_graphable=False)
                    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()

                numpy_state = np.random.get_state()
                return {
                    "python": random.getstate(),
                    "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                    "cpu": tensor_digest(torch.get_rng_state()),
                    "cuda": tensor_digest(torch.cuda.get_rng_state()),
                    "trackers": {
                        name: tensor_digest(state)
                        for name, state in tensor_parallel.get_cuda_rng_tracker().get_states().items()
                    },
                }

            observe_call(
                initialize, "_set_random_seed", "consumer.rng_seed",
                after=lambda args, output: record("runtime.rng_after_seed", rng_snapshot()),
            )
            observe_call(SummaryWriter, "__init__", "consumer.tensorboard", excluded=("self",))
            observe_call(globals_.Timers, "__init__", "consumer.timers", excluded=("self",))
            observe_call(torch.profiler, "schedule", "consumer.profiler_schedule")
            observe_call(torch.cuda.memory, "_dump_snapshot", "consumer.memory_snapshot")
            observe_call(GPTModel, "__init__", "consumer.model", excluded=("self",))
            if entrypoint == "pretrain_vlm.py":
                from megatron.core.models.multimodal.llava_model import LLaVAModel

                observe_call(LLaVAModel, "__init__", "consumer.vlm", excluded=("self",))
                observe_call(LLaVAModel, "freeze", "consumer.vlm_freeze", excluded=("self",))
            observe_call(DistributedDataParallel, "__init__", "consumer.ddp", excluded=("self", "module"))
            observe_call(training, "get_megatron_optimizer", "consumer.optimizer", excluded=("model_chunks", "timers"))
            original_save = training.save_checkpoint

            @functools.wraps(original_save)
            def save(*args, **kwargs):
                record("runtime.rng_at_save", rng_snapshot())
                output = original_save(*args, **kwargs)
                record("runtime.checkpoint_save", {"iteration": args[0], "completed": True})
                return output

            training.save_checkpoint = save
            original_load = training.load_checkpoint

            @functools.wraps(original_load)
            def load(*args, **kwargs):
                output = original_load(*args, **kwargs)
                record("runtime.checkpoint_load", output)
                record("runtime.rng_after_load", rng_snapshot())
                return output

            training.load_checkpoint = load

        replacements = {
            "{fixtures}": str(options.fixtures),
            "{checkpoint}": str(options.checkpoint),
            "{run}": str(options.run_root),
        }
        argv = []
        for argument in case["argv"]:
            for token, value in replacements.items():
                argument = argument.replace(token, value)
            argv.append(argument)
        if entrypoint not in ("pretrain_gpt.py", "pretrain_hybrid.py", "pretrain_vlm.py"):
            raise ValueError("Unsupported entrypoint")
        sys.argv = [str(options.repo / entrypoint), *argv]
        runpy.run_path(sys.argv[0], run_name="__main__")
        result["status"] = "ok"
    except SystemExit as error:
        result["status"] = "ok" if error.code in (None, 0) else "error"
        result["error"] = {"type": "SystemExit", "message": str(error.code)}
    except Exception as error:  # noqa: BLE001 - record any tested-program failure, then exit nonzero.
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
    finally:
        (options.output / f"rank-{rank}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
