# RNGConfig ownership results

Validated on 2026-09-22 with H100 GPUs and the same `26.10.rc0` environment on
both sides. The baseline is frozen at
`5e85e2b27fc29bcf938e8f3b38acfbdecffcc8f3` (includes merged PR7418).

- Implementation: `e306f6f1f368c42155778798696a12d265ce486b`.
- Final head: `652ee266420ab6960a81c1d7a55f64659a065c9f`.
- Tested fork-only harness: `40738bfed9b80a774eedeee4c1b0cd7e5ba716a4`.

The final head differs from the implementation commit only in the checkpoint
unit test. Production code is identical. The one-GPU runtime lane tested the
implementation commit; final-head units and two-GPU runtime tested the final head.

## Results

| Selection | GPUs | Result |
|---|---:|---|
| `runtime_fresh`, `runtime_resume` | 1 | Exact pass |
| `rng_custom_seed`, `rng_te_tracker`, `rng_inference_tracker` | 1 | Exact pass |
| `rng_dp_fresh`, `rng_dp_resume` | 2 | Exact pass on both ranks |
| `hybrid_known` (configuration only) | 1 | Exact pass at final head |

All seven runtime comparisons have zero configuration/consumer differences and
zero capture errors. Each resume scenario uses the same baseline-produced
checkpoint for both revisions. RNG observations include exact Python/NumPy state
and hashes of CPU/CUDA/tracker state, captured without drawing random values.

The focused product unit selection passed **326 candidate / 294 baseline tests**,
with three identical skips and no failures/errors. This includes 31 new ownership
cases and an added checkpoint ownership variant. The legacy checkpoint test also
passed both shared-stream and DP-specific-stream variants on each of two ranks,
including exact CPU RNG restoration with legacy RNG fields removed.

All 17 harness self-tests pass. Independent product and harness reviews found no
outstanding blockers. Changed-file formatting, syntax and whitespace checks pass;
all-files lint retains only untouched baseline findings.

## Reproduce and limits

Follow the [README](README.md) with the pinned revisions above. Run one-GPU and
two-GPU case groups separately. The harness is fork-only and is not included in
the product PR. Report files record revision/tree, environment, harness and
fixture hashes; raw checkpoints and rank logs are not committed here.

This is a targeted eight-case result (seven runtime, one configuration-only),
not a rerun of all 52 manifest cases. The Hybrid case does not run training.
Full-suite, exhaustive model-family, convergence and performance validation are
not claimed. Deleted/divergent-args ownership is established by behavioral product
tests; equivalent CLI runs alone do not prove ownership.
