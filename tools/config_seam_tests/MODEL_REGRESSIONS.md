# Model regression gate for training-loop migrations

Added on 2026-10-06 after RNG ownership PR #7591 exposed a coverage gap in the
focused two-GPU unit selection. This gate supplements the configuration/runtime
comparison matrix; it does not replace it or change its reported case count.

## Required selection

Run the complete existing `TestHybridMoEModel` class from
`tests/unit_tests/models/test_hybrid_moe_model.py` on both pinned revisions:

- `test_constructor`: full serialized-config golden, model flags, hybrid pattern,
  and parameter count.
- `test_set_input_tensor`: pipeline input tensor dimensions.
- `test_forward`: forward-pass output dimensions (not numerical parity).

The fixture uses TP=2, EP=4, and ETP=1. Run on eight GPUs, separately from the
one/two-GPU suite. Do not reduce its topology, model dimensions, or assertions
to fit a smaller allocation. This is an existing model-unit-test selection,
not a new full-suite or convergence run.

Inside an existing eight-GPU allocation and the same supported MLM environment
used for both revisions, with test dependencies/assets already available:

```bash
bash /path/to/harness/run_model_regressions.sh \
  /path/to/frozen-main /path/to/new-baseline-results
bash /path/to/harness/run_model_regressions.sh \
  /path/to/pr-head /path/to/new-candidate-results
```

Use clean, separate checkouts and node-local result/cache directories. Configure
`UV_PROJECT_ENVIRONMENT` to the existing container environment if necessary.
The runner records commit/tree IDs, the test-file hash, environment information,
rank logs, and exit status. It neither submits allocations nor modifies either
checkout. Both sides must be run and reported even if one fails.

## Failure this gate caught

The previous #7591 CI run tested `15aaaa5361869dbe005fc1e3ba27e1e9ed7b2603`:
[failed models job](https://github.com/NVIDIA/Megatron-LM/actions/runs/37503989784/job/112460525047).
It reported one failure, 1,012 passes, 20 skips, and 30 deselections.

The fixture explicitly sets `args.te_rng_tracker=True`, but its golden expects
`TransformerConfig.use_te_rng_tracker=False`. The old same-name field adapter
did not map those differently named fields. Centralized RNG ownership now
propagates True, so the golden reports a real configuration change.

This does not demonstrate a newly changed runtime tracker: normal training
already passed the CLI flag directly into runtime RNG initialization. This unit
fixture instead directly calls `model_parallel_cuda_manual_seed(123)` without
the tracker flag. Configuration correctness and runtime RNG parity therefore
need separate evidence. Keep the RNG runtime/checkpoint comparisons as well.

The golden must not be ignored or regenerated wholesale. Review the intentional
one-field change, then update that expectation in the product PR and rerun this
gate. Baseline and candidate may legitimately retain different golden values,
but that difference must be documented rather than hidden by the harness.

## Validation status

The known CI failure is evidence for the previous product head, not a passing
run of this new launcher. The latest teacher-overlay removal does not change
the failing propagation/golden pair. No new GPU run or golden update is claimed
by this suite addition.
