# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only harness self-checks, independent of MLM's GPU pytest fixtures."""

import copy
import functools
import json
import unittest
from dataclasses import make_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from capture_config_seams import build_dataset_config
from compare_config_seams import (
    compare_runs,
    create_tp_overlap_config,
    materialize_case,
    validate,
    validate_expected,
)
from snapshot import (
    LEGACY_LOGGER_FIELDS,
    differences,
    dist_settings,
    encode,
    logger_settings,
    validation_settings,
)


class SnapshotTests(unittest.TestCase):
    """Prove exact comparisons and fail-closed capture handling."""

    def test_equal(self):
        self.assertEqual(differences(encode({"vocab": 256}), encode({"vocab": 256})), [])

    def test_dataset_callback_owner_binding(self):
        for owned in (False, True):
            namespace = {}
            factory = "def core_gpt_dataset_config_from_args(args, *, random_seed): return random_seed" if owned else "def core_gpt_dataset_config_from_args(args): return args.seed"
            exec(factory + "\ndef provider(*, random_seed=None): pass", namespace)
            provider = namespace["provider"]
            if owned:
                with self.assertRaises(ValueError):
                    build_dataset_config(provider, SimpleNamespace(seed=7))
                provider = functools.update_wrapper(functools.partial(provider, random_seed=9), provider)
            self.assertEqual(build_dataset_config(provider, SimpleNamespace(seed=7)), 9 if owned else 7)

    def test_checkpoint_seed_scenarios_are_self_contained(self):
        manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
        by_name = {case["name"]: case for case in manifest["cases"]}
        for case in by_name.values():
            if case.get("needs_checkpoint"):
                seed = by_name[case.get("checkpoint_seed_case", "runtime_fresh")]
                self.assertEqual(seed["tier"], "runtime")
                self.assertFalse(seed.get("needs_checkpoint"))
                self.assertIn("--save", seed["options"])
        self.assertEqual(by_name["rng_dp_resume"]["checkpoint_seed_case"], "rng_dp_fresh")

    def test_logger_owner_migration_is_exact(self):
        legacy = make_dataclass("LoggerConfig", [("log_interval", int)], namespace={"__module__": "fixture"})
        native = make_dataclass(
            "LoggerConfig",
            [("log_interval", int), *((name, object) for name in LEGACY_LOGGER_FIELDS)],
            namespace={"__module__": "fixture"},
        )
        args = SimpleNamespace(**{name: index for index, name in enumerate(LEGACY_LOGGER_FIELDS)})
        expected = encode(logger_settings(legacy(7), args))
        config = native(7, **vars(args))
        self.assertEqual(differences(expected, encode(logger_settings(config, SimpleNamespace()))), [])
        config.one_logger_project = "changed"
        self.assertTrue(differences(expected, encode(logger_settings(config, args))))
        config.one_logger_project = None
        self.assertIsNone(logger_settings(config, args)["fields"]["one_logger_project"])
        del config.one_logger_project
        self.assertEqual(logger_settings(config, args)["fields"]["one_logger_project"], {"missing_attribute": True})
        with self.assertRaises(AttributeError):
            logger_settings(legacy(7), SimpleNamespace())

    def test_logger_projection_retains_unknown_fields(self):
        schema = make_dataclass("LoggerConfig", [("future_setting", int)])
        args = SimpleNamespace(**dict.fromkeys(LEGACY_LOGGER_FIELDS))
        self.assertTrue(differences(logger_settings(schema(1), args), logger_settings(schema(2), args)))

    def test_validation_duplicate_removal_requires_model_agreement(self):
        legacy = make_dataclass("Validation", [("test_mode", bool), ("eval_iters", int)])
        native = make_dataclass("Validation", [("eval_iters", int)])
        for enabled in (False, True):
            model = SimpleNamespace(test_mode=enabled)
            expected = validation_settings(legacy(enabled, 7), model)
            self.assertEqual(expected, validation_settings(native(7), model))
            self.assertTrue(differences(expected, validation_settings(native(8), model)))
            with self.assertRaises(ValueError):
                validation_settings(legacy(not enabled, 7), model)
            with self.assertRaises(ValueError):
                validation_settings(legacy(int(enabled), 7), model)
            with self.assertRaises(AttributeError):
                validation_settings(legacy(enabled, 7), SimpleNamespace())
        future = make_dataclass("Validation", [("eval_iters", int), ("future_setting", int)])
        self.assertTrue(differences(validation_settings(future(7, 1), model), validation_settings(future(7, 2), model)))

    def test_dist_owner_schema_changes_do_not_hide_values(self):
        legacy = make_dataclass("Dist", [("use_megatron_fsdp", bool), ("future_setting", int)])
        native = make_dataclass("Dist", [("fake_process_group", bool), ("future_setting", int)])
        ddp = SimpleNamespace(use_megatron_fsdp=False)
        args = SimpleNamespace(fake_process_group=True)
        expected = dist_settings(legacy(False, 7), ddp, args)
        self.assertEqual(expected, dist_settings(native(True, 7), ddp, SimpleNamespace()))
        self.assertTrue(differences(expected, dist_settings(native(False, 7), ddp, args)))
        self.assertTrue(differences(expected, dist_settings(native(True, 8), ddp, args)))
        with self.assertRaises(ValueError):
            dist_settings(legacy(True, 7), ddp, args)
        with self.assertRaises(AttributeError):
            dist_settings(legacy(False, 7), ddp, SimpleNamespace())
        config = native(True, 7)
        del config.fake_process_group
        self.assertEqual(dist_settings(config, ddp, args)["fields"]["fake_process_group"], {"missing_attribute": True})

    def test_changed_values(self):
        for name in ("vocab_size", "lr_decay_steps", "bucket_size"):
            with self.subTest(name=name):
                self.assertTrue(differences({name: 256}, {name: 512}))

    def test_missing_null_and_types(self):
        self.assertTrue(differences({}, {"value": None}))
        self.assertTrue(differences({"value": 1}, {"value": True}))
        self.assertTrue(differences(encode((1,)), encode([1])))

    def test_unknown_and_nonfinite_rejected(self):
        with self.assertRaises(TypeError):
            encode(object())
        with self.assertRaises(ValueError):
            encode(float("nan"))

    def test_required_capture(self):
        self.assertTrue(validate({"schema": 1, "status": "ok", "captures": {}}, ["config.model"]))

    def test_expected_values_require_one_real_matching_event(self):
        snapshot = {"captures": {"loader": [{"workers": 0, "pin": True}, {"workers": 2, "pin": False}]}}
        self.assertEqual(validate_expected(snapshot, {"loader": {"workers": 2, "pin": False}}), [])
        self.assertTrue(validate_expected(snapshot, {"loader": {"workers": 2, "pin": True}}))
        self.assertTrue(validate_expected(snapshot, {"loader": {"workers": False}}))
        self.assertTrue(validate_expected(snapshot, {"loader": {"missing": None}}))
        self.assertTrue(validate_expected(snapshot, {"absent": {"workers": 2}}))
        nested = {"captures": {"config": [{"fields": {"optional": None}}]}}
        self.assertEqual(validate_expected(nested, {"config": {"fields.optional": None}}), [])
        checkpoint = {"captures": {"load": [{"tuple": [3, 0.0]}]}}
        self.assertEqual(validate_expected(checkpoint, {"load": {"tuple.0": 3}}), [])
        self.assertTrue(validate_expected(checkpoint, {"load": {"tuple.2": None}}))
        self.assertTrue(validate_expected(checkpoint, {"load": {"tuple.-1": 0.0}}))

    def test_both_sides_ignoring_a_flag_is_not_a_pass(self):
        case = {"name": "explicit", "tier": "builder", "expected": {"config.dist": {"fields.disable_jit_fuser": True}}}
        snapshot = {"schema": 1, "status": "ok", "case": "explicit", "rank": 0,
                    "environment": {}, "captures": {"config.dist": [{"fields": {"disable_jit_fuser": False}}]}}
        run = {"exit": 0, "snapshots": [snapshot]}
        self.assertEqual(compare_runs(case, run, run, ["config.dist"])["status"], "error")

    def test_manifest_sources_and_arguments(self):
        manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
        names = [case["name"] for case in manifest["cases"]]
        self.assertEqual(len(names), len(set(names)))
        for case in manifest["cases"]:
            with self.subTest(case=case["name"]):
                if "source" in case:
                    self.assertIn(case["source"], manifest["sources"])
                    self.assertIn(case["source"].split("-")[0], manifest["source_revisions"])
                self.assertIn(case["tier"], ("builder", "runtime", "startup"))
                self.assertGreater(case.get("ranks", 1), 0)
                self.assertTrue(all(isinstance(item, str) for item in materialize_case(manifest, case)["argv"]))

    def test_scenario_required_capture(self):
        snapshot = {"schema": 1, "status": "ok", "captures": {"consumer.model": [{}]}}
        self.assertTrue(validate(snapshot, ["consumer.model", "consumer.vlm_freeze"]))

    def test_expansion_cases_have_nonempty_expected_values(self):
        manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
        self.assertIn("config.dist", manifest["required"])
        for case in manifest["cases"]:
            if case["name"].startswith("coverage_"):
                with self.subTest(case=case["name"]):
                    self.assertTrue(case["expected"])
                    for seam, paths in case["expected"].items():
                        self.assertTrue(seam)
                        self.assertTrue(paths)
                        self.assertTrue(all(path and all(path.split(".")) for path in paths))

    def test_case_environment_is_bounded(self):
        manifest = {"common": {}}
        case = {"options": {}, "environment": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}}
        self.assertEqual(materialize_case(manifest, case)["environment"], case["environment"])
        with self.assertRaises(ValueError):
            materialize_case(manifest, case | {"environment": {"PYTHONPATH": "/another/checkout"}})

    def test_optional_probe_tiers(self):
        for probe, allowed, denied in (("build_inference_config", "builder", "runtime"),
                                       ("observe_tp_overlap", "runtime", "builder")):
            case = {"tier": allowed, "options": {}, probe: True}
            self.assertEqual(materialize_case({"common": {}}, case)["argv"], [])
            with self.assertRaises(ValueError):
                materialize_case({"common": {}}, case | {"tier": denied})

    def test_tp_overlap_fixture_contains_distinct_operation_policies(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.json"
            create_tp_overlap_config(path)
            self.assertEqual(json.loads(path.read_text()), {
                "qkv_fprop": {"method": "ring_exchange", "aggregate": 1},
                "fc1_fprop": {"method": "ring_exchange", "aggregate": 0},
            })

    def test_named_parameter_references(self):
        first, second = object(), object()
        references = {id(first): {"parameter": "layer.0"}, id(second): {"parameter": "layer.1"}}
        self.assertEqual(
            encode({first: [second]}, references=references),
            {"mapping": [[{"parameter": "layer.0"}, [{"parameter": "layer.1"}]]]},
        )
        self.assertTrue(differences(encode(first, references=references), encode(second, references=references)))
        with self.assertRaises(TypeError):
            encode(object(), references=references)

    def test_infinite_config_sentinel(self):
        self.assertEqual(encode(float("inf")), {"float": "+inf"})
        self.assertTrue(differences(encode(float("inf")), encode(float("-inf"))))

    def test_bound_method_identity(self):
        encoded = encode(self.test_equal)
        self.assertTrue(encoded["bound_method"].endswith("SnapshotTests.test_equal"))
        self.assertTrue(encoded["owner_type"].endswith("SnapshotTests"))

    def test_rank_count(self):
        case = {"name": "example", "tier": "builder", "ranks": 1}
        empty = {"exit": 0, "snapshots": []}
        self.assertEqual(compare_runs(case, empty, empty, ["config.model"])["status"], "error")

    def test_driver_mismatch_and_failure(self):
        case = {"name": "example", "tier": "builder"}
        snapshot = {
            "schema": 1,
            "case": "example",
            "status": "ok",
            "rank": 0,
            "environment": {},
            "captures": {"config.model": [{"vocab_size": 256}]},
        }
        good = {"exit": 0, "snapshots": [snapshot]}
        changed = copy.deepcopy(good)
        changed["snapshots"][0]["captures"]["config.model"][0]["vocab_size"] = 512
        self.assertEqual(compare_runs(case, good, good, ["config.model"])["status"], "pass")
        self.assertEqual(compare_runs(case, good, changed, ["config.model"])["status"], "mismatch")
        failed = {"exit": 1, "snapshots": [{}]}
        self.assertEqual(compare_runs(case, failed, failed, ["config.model"])["status"], "error")


if __name__ == "__main__":
    unittest.main()
