# Sequence-length ownership: pinned before/after comparison

Validated on 2026-11-12 against one frozen main revision:

- Baseline: `a7a6d2225700a8ee701453288894501d868c6bd4`.
- Candidate: `b068862478906fd53c37181130bb2381ef905c81`.
- Full matrix: H100, `nvcr.io/nvidian/nemo:26.10.rc5`.
- Manifest: 118 cases, including two new packed-SFT runtime cases.

After the four same-image follow-up reruns: **112 exact passes, one expected CLI
rejection, four intentional VLM configuration differences, and one shared
baseline/environment failure across all 118 cases**. No candidate-only failures
or missing cases. This is not an all-green equivalence result.

| Selection | Cases | Final result |
|---|---:|---|
| One rank | 97 | 93 exact passes, one expected rejection, three VLM differences |
| Two ranks | 19 | 17 exact passes, one VLM difference, one shared TE failure |
| Four ranks | 1 | Exact pass on all ranks |
| Eight ranks | 1 | Exact pass on all ranks |

`invalid_batch_conflict` rejects simultaneous explicit `--global-batch-size` and
`--step-batch-size-schedule` flags on both revisions, as expected. All baseline
checkpoint-seed runs completed and created their required checkpoint. Ordinary
and strict checkpoint-resume cases pass on this pinned main.

The comparison suite runs capture/compare scenarios only, not MLM unit tests.
Runtime cases execute real short training runs. Resume cases load the same
checkpoint produced by the pinned baseline. Builder cases do not claim training;
the real multimodal startup case does not claim Energon dataset loading.

## Intentional differences

The mock VLM entrypoint now resolves text + image + alignment width before
runtime initialization. The small fixtures with 24 text tokens and four image
tokens therefore expose a 28-token execution bound rather than the old CLI
value of 32. The frozen-vision fixture similarly resolves 32 text + five image
tokens to 37 rather than 32. These raw configuration and microbatch-input differences remain
visible; the harness does not suppress them. Training loss, learning rate, RNG,
and other captured consumers match exactly. The four cases are
`coverage_vlm_freeze_language`, `coverage_vlm_projection_only`,
`coverage_vlm_packed_cp`, and `runtime_vlm_frozen_vision`.

## Harness maintenance

Both tested revisions already contain newer dataset/tokenizer/DDP APIs. The
observation adapter now captures the actual dataset config passed to the dataset
builder instead of intercepting the removed `GPTDatasetConfig.__post_init__`.
Builder-only cases capture the config held by the container or returned by the
legacy factory. The comparable physical pack width is `packed_sequence_length`
when present, otherwise `sequence_length`; all other dataset fields remain strict.

Three stale cases were corrected after failing equally on both revisions:

- Tokenizer input is now a dataclass: the expected path is `fields.vocab_file`.
- Torch FSDP2 is identified by its DDP config type, not `cfg.dist.use_torch_fsdp2`.
- Manual FSDP registration uses supported FSDP v1 with NCCL user buffers and FP32
  gradient communication, matching current main's existing validation.

These are harness corrections, not production changes. Their original failed
attempts are retained in the validation artifacts. Corrected reruns pass on the
same H100/26.10.rc5 environment and pinned revisions. Separate earlier local
26.08.rc9 reruns also pass, but are not substituted for those H100 results.

Original manifest SHA256:
`13da3c332960da5b9a999133f3d7bc30154a8389a7329287ec2b734c9d899523`.
Corrected manifest SHA256:
`a42c1609ef689d59a920340a8ffc4adff2babeb0b389778a22296e55562c0685`.
The published observer differs from the full-run observer only in import
ordering; the four final H100 reruns use the published observer. No comparison
filter or production change was made in response to a failing case.

## Baseline/environment limitations

`coverage_vlm_decoder_overlap` fails on both revisions inside Transformer Engine
attention when PyTorch's `guard_scalar` rejects `max_seqlen_q` (`unrecognized
scalar 128`). Neither side reaches the required training metrics. This remains
a failed case, not a parity pass; no production workaround is included.

The initial H100 `coverage_checkpoint_async_worker` baseline timed out during its
first asynchronous save; the candidate completed. Isolated paired reruns pass
on both revisions, including H100/26.10.rc5. The final rerun uses a 600-second
per-side budget instead of the original 300 seconds. Retain the original
timeout in the record rather than claiming an entirely clean first attempt.

## New behavior checks

Separate candidate-only runtime probes exercise the newly independent lengths:

- Fixed GPT: content/row width 8, allocation bound 32; the runtime microbatch
  input remains 8 and training reaches iteration 2.
- Packed SFT: per-conversation content limit 8, physical pack width 32; the actual
  dataset and runtime microbatch input use 32 and training reaches iteration 2.

These are behavior checks, not before/after equivalence claims. Focused product
tests separately cover truncation, read-only validation, startup derivation,
serialization round trips, T5 encoder/decoder widths, and callback boundaries.
Those 13 focused product modules pass on two H100 ranks: 908 passed and 277
skipped per rank. They are not invoked by this fork-only comparison suite.

The harness's separate self-checks pass: 27 tests and 180 subtests. Independent
full-diff reviews of the production change and observation adapter found no
remaining blocking issue.

## Reproduce

Use the [README](README.md) instructions with the exact revisions above. Group
`cases.json` by rank requirement: 97 one-rank cases, 19 two-rank cases, one
four-rank case, and one eight-rank case. Freeze main for the whole comparison.
The corrected manifest is intended for these APIs; older historical reports
refer to their original pinned harness revisions.
