# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""CPU-only harness self-checks, independent of MLM's GPU pytest fixtures."""

import copy
import json
import unittest
from pathlib import Path

from compare_config_seams import compare_runs, materialize_case, validate
from snapshot import differences, encode


class SnapshotTests(unittest.TestCase):
    """Prove exact comparisons and fail-closed capture handling."""

    def test_equal(self):
        self.assertEqual(differences(encode({"vocab": 256}), encode({"vocab": 256})), [])

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

    def test_manifest_sources_and_arguments(self):
        manifest = json.loads(Path(__file__).with_name("cases.json").read_text())
        names = [case["name"] for case in manifest["cases"]]
        self.assertEqual(len(names), len(set(names)))
        for case in manifest["cases"]:
            with self.subTest(case=case["name"]):
                if "source" in case:
                    self.assertIn(case["source"], manifest["sources"])
                    self.assertIn(case["source"].split("-")[0], manifest["source_revisions"])
                self.assertIn(case["tier"], ("builder", "runtime"))
                self.assertGreater(case.get("ranks", 1), 0)
                self.assertTrue(all(isinstance(item, str) for item in materialize_case(manifest, case)["argv"]))

    def test_scenario_required_capture(self):
        snapshot = {"schema": 1, "status": "ok", "captures": {"consumer.model": [{}]}}
        self.assertTrue(validate(snapshot, ["consumer.model", "consumer.vlm_freeze"]))

    def test_case_environment_is_bounded(self):
        manifest = {"common": {}}
        case = {"options": {}, "environment": {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}}
        self.assertEqual(materialize_case(manifest, case)["environment"], case["environment"])
        with self.assertRaises(ValueError):
            materialize_case(manifest, case | {"environment": {"PYTHONPATH": "/another/checkout"}})

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
