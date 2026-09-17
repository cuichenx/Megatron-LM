# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run a shared configuration matrix against two clean, pinned MLM checkouts."""

import argparse
import hashlib
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from snapshot import differences

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


def git(repo: Path, *args: str) -> str:
    """Read Git metadata without modifying the tested checkout."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def materialize_case(manifest: dict, case: dict) -> dict:
    """Expand a case's explicit CLI overrides without duplicating MLM derivation."""
    options = manifest["common"] | case["options"]
    argv = []
    for flag, value in options.items():
        if value is False:
            continue
        argv.append(flag)
        if value is not True:
            argv.extend(value if isinstance(value, list) else [value])
    return case | {"argv": argv}


def validate(snapshot: dict, required: list[str]) -> list[str]:
    """Require a successful capture and every expected nonempty seam."""
    errors = []
    if snapshot.get("schema") != 1 or snapshot.get("status") != "ok":
        errors.append("capture failed or unsupported schema")
    for name in required:
        if not snapshot.get("captures", {}).get(name):
            errors.append("missing capture: " + name)
    return errors


def run_case(repo: Path, case: dict, destination: Path, fixtures: Path, checkpoint: Path, timeout: int) -> dict:
    """Run an isolated process group and terminate the entire group on timeout."""
    destination.mkdir(parents=True)
    case_file = destination / "case.json"
    case_file.write_text(json.dumps(case, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={case.get('ranks', 1)}",
        "--log-dir",
        str(destination / "rank-logs"),
        "--redirects=3",
        str(HERE / "capture_config_seams.py"),
        "--repo",
        str(repo),
        "--case-file",
        str(case_file),
        "--output",
        str(destination / "snapshots"),
        "--fixtures",
        str(fixtures),
        "--checkpoint",
        str(checkpoint),
        "--run-root",
        str(destination / "run"),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(repo),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "MEGATRON_OTEL_ENABLED": "false",
            "NEMO_LENS_ENABLED": "false",
        }
    )
    with (destination / "launcher.log").open("w") as stream:
        process = subprocess.Popen(
            command, cwd=repo, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status = "timeout"
    snapshots = []
    for rank in range(case.get("ranks", 1)):
        path = destination / "snapshots" / f"rank-{rank}.json"
        snapshots.append(json.loads(path.read_text()) if path.exists() else {})
    return {"exit": status, "snapshots": snapshots}


def compare_runs(case: dict, baseline: dict, candidate: dict, required: list[str]) -> dict:
    """Compare snapshots only after exit, rank and completeness checks succeed."""
    result = {"case": case["name"], "tier": case["tier"], "status": "pass", "differences": [], "errors": {}}
    if "expected_error" in case:
        for label, run in (("baseline", baseline), ("candidate", candidate)):
            if (
                run["exit"] in (0, "timeout")
                or len(run["snapshots"]) != case.get("ranks", 1)
                or not all(
                    case["expected_error"] in snap.get("error", {}).get("message", "") for snap in run["snapshots"]
                )
            ):
                result["errors"][label] = ["expected CLI rejection was not observed"]
        result["status"] = "expected_rejection" if not result["errors"] else "error"
        return result
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        errors = []
        if len(run["snapshots"]) != case.get("ranks", 1):
            errors.append("wrong rank count")
        if run["exit"] != 0:
            errors.append(f"process exit: {run['exit']}")
        for rank, snapshot in enumerate(run["snapshots"]):
            errors.extend(f"rank {rank}: {item}" for item in validate(snapshot, required))
            if snapshot.get("status") != "ok" and snapshot.get("error"):
                errors.append(f"rank {rank}: {snapshot['error']}")
            if snapshot.get("rank") != rank or snapshot.get("case") != case["name"]:
                errors.append(f"rank {rank}: capture identity mismatch")
        if errors:
            result["errors"][label] = errors
    if result["errors"]:
        result["status"] = "error"
        return result
    for rank, (left, right) in enumerate(zip(baseline["snapshots"], candidate["snapshots"])):
        result["differences"].extend(differences(left["captures"], right["captures"], f"rank[{rank}]"))
        result["differences"].extend(differences(left["environment"], right["environment"], "environment"))
    if result["differences"]:
        result["status"] = "mismatch"
    return result


def create_tokenizer(destination: Path) -> None:
    """Create a deterministic local 261-token HF fixture; no network/model downloads."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3}
    vocabulary.update({f"t{index}": index for index in range(4, 261)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    ).save_pretrained(destination)


def main() -> None:
    """Execute selected cases, leaving logs/snapshots and a machine-readable report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "cases.json")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--environment-id", required=True, help="Container reference or immutable environment identifier"
    )
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    options.baseline = options.baseline.resolve()
    options.candidate = options.candidate.resolve()
    options.output = options.output.resolve()
    manifest = json.loads(options.manifest.read_text())
    cases = [
        materialize_case(manifest, case)
        for case in manifest["cases"]
        if options.cases is None or case["name"] in options.cases
    ]
    if not cases or (options.cases and set(options.cases) != {case["name"] for case in cases}):
        parser.error("Empty or unknown case selection")
    provenance = {
        "environment": options.environment_id,
        "harness_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir() if p.suffix in (".py", ".json")
        },
        "manifest_sha256": hashlib.sha256(options.manifest.read_bytes()).hexdigest(),
    }
    for label in ("baseline", "candidate"):
        repo = getattr(options, label)
        revision = git(repo, "rev-parse", "HEAD")
        if revision != getattr(options, label + "_revision") or git(
            repo, "status", "--porcelain", "--untracked-files=normal"
        ):
            parser.error(f"{label} must be clean at the exact supplied revision")
        provenance[label] = {"revision": revision, "tree": git(repo, "rev-parse", "HEAD^{tree}")}
    options.output.mkdir(parents=True, exist_ok=False)
    fixtures = options.output / "fixtures"
    create_tokenizer(fixtures / "tokenizer")
    provenance["fixtures"] = {
        str(p.relative_to(fixtures)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in fixtures.rglob("*")
        if p.is_file()
    }
    checkpoint = options.output / "seed" / "run" / "checkpoint"
    report = {"schema": 1, "provenance": provenance, "results": []}
    if any(case.get("needs_checkpoint") for case in cases):
        seed = materialize_case(manifest, next(case for case in manifest["cases"] if case["name"] == "runtime_fresh"))
        seed["name"] = "checkpoint_seed"
        run = run_case(options.baseline, seed, options.output / "seed", fixtures, checkpoint, options.timeout)
        seed_errors = [
            item
            for snapshot in run["snapshots"]
            for item in validate(snapshot, [*manifest["required"], "runtime.checkpoint_save"])
        ]
        report["checkpoint_seed"] = {
            "exit": run["exit"],
            "errors": seed_errors,
            "created": (checkpoint / "latest_checkpointed_iteration.txt").exists(),
            "files": {
                str(path.relative_to(checkpoint)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in checkpoint.rglob("*")
                if path.is_file()
            },
        }
    for case in cases:
        if case.get("needs_checkpoint") and (
            report["checkpoint_seed"]["exit"] != 0
            or report["checkpoint_seed"]["errors"]
            or not report["checkpoint_seed"]["created"]
        ):
            report["results"].append({"case": case["name"], "status": "missing_fixture"})
        else:
            runs = {
                label: run_case(
                    getattr(options, label),
                    case,
                    options.output / case["name"] / label,
                    fixtures,
                    checkpoint,
                    options.timeout,
                )
                for label in ("baseline", "candidate")
            }
            required = list(manifest["required"])
            if case["tier"] == "runtime":
                required += ["consumer.model", "consumer.ddp", "consumer.optimizer"]
                if "--save" in case["argv"]:
                    required += ["runtime.checkpoint_save"]
                if case.get("needs_checkpoint"):
                    required += ["runtime.checkpoint_load"]
            else:
                required += ["builder.scheduler_points", "builder.batch_points"]
            report["results"].append(compare_runs(case, runs["baseline"], runs["candidate"], required))
        LOGGER.info("%s: %s", case["name"], report["results"][-1]["status"])
        (options.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if any(item["status"] not in ("pass", "expected_rejection") for item in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
