# Reproduce the released QASC endpoints

This guide uses the released, frozen QASC channel. No adapter training, HiddenBench data,
or original private run-directory layout is needed for the inference-only entry point.

## Verified replay (7 November 2026)

On an NVIDIA H100 80GB with PyTorch `2.12.0+cu130`, both requested endpoints reproduced
all 1,000 published labels exactly: `final_only` and four-exchange `matched`. The saved
messages also matched the original run's tensor hashes for the single exchange and all
four recurrent exchanges. Recording was enabled during this replay; no training was performed.

The machine-readable [verification receipt](../evidence/qasc_release_verification.json)
records source/input identities, decoder parity, prediction agreement, and trace checks.
The [package inventory](../evidence/qasc_release_environment.txt) records this replay's
installed versions. The additional intervention modes exposed by the driver were not rerun
in this two-endpoint release check.

## Files and provenance

- [`configs/qasc_original_evaluation.yaml`](../configs/qasc_original_evaluation.yaml) is the
  original endpoint-run configuration, including `surface`, the prompt, assistant prefix,
  model revision, channel settings, and intervention seeds. It retains the original run
  metadata. The older `configs/qasc.yaml` is a paper summary, not a runnable evaluation config.
- [`data/qasc/validation.jsonl`](../data/qasc/validation.jsonl) contains the exact 1,000 derived
  development examples in their original order. SHA-256:
  `a88c888c082611fe5ecdfd62563026357068b2e23dfb0914bd31012ea230e20a`.
- [`data/qasc/dataset_card.json`](../data/qasc/dataset_card.json) and
  [`source_files.json`](../data/qasc/source_files.json) preserve construction metadata.
- The checkpoint is `checkpoints/qasc_dense_latent_prefix_channel.pt`, SHA-256
  `83397168be971c5de6edb51e8d57ddeee87781cae13e77ba55bfb69df3b5f66d`.
- [`evidence/qasc_predictions.jsonl`](../evidence/qasc_predictions.jsonl) is the per-example
  reference, not output from a new run.
- The previously omitted `phase6_dream_*` helper modules are restored under `code/scripts/`,
  including their transitive imports. The original QASC evaluation function is unchanged.

The source evaluation run was
`phase8-qasc-hiddenbench-dense-prefix-posthoc-endpoints-20260901-ag-r8k4t2-r2`.
The original environment receipt records NVIDIA H100 80GB, PyTorch `2.12.0+cu130`, and CUDA
`13.0`. Package pins are in `code/pyproject.toml`; upstream commits are in
`code/upstreams/manifest.yaml`. These are historical environment records. Exact prediction
parity on a different GPU/software stack must be measured, not assumed.

## Install and download

Run from the repository root on a CUDA machine. Use Python 3.11 or 3.12 in a separate virtual
environment with a CUDA-compatible PyTorch installation. For the closest reproduction, match
the original H100/PyTorch/CUDA environment above.

```bash
unset PYTHONPATH
git clone https://github.com/ZHZisZZ/dllm.git third_party/dllm
git -C third_party/dllm checkout ca176752fbceec49c6b4777a2c18ae88e4eb10ed
git -C third_party/dllm submodule update --init --recursive
git -C third_party/dllm/lm-evaluation-harness checkout dca091dbee3d6ee1dab268e9f649aa64e8db97d8
python -m pip install -e third_party/dllm/lm-evaluation-harness
python -m pip install -e code
python -m pip install --no-deps -e third_party/dllm
export PYTHONPATH="$PWD/code/src:$PWD/code/scripts:$PWD/third_party/dllm"
```

Download the pinned Dream backbone into an HF cache, not a flattened `--local-dir` directory:

```bash
hf download Dream-org/Dream-v0-Instruct-7B \
  --revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae \
  --cache-dir /path/to/hf-cache
```

The channel file is included in this GitHub release. The identical file is also available at
[RobotLearningLab/hidden-messages-dream-qasc](https://huggingface.co/RobotLearningLab/hidden-messages-dream-qasc).
It is a state dictionary; the runner instantiates `DenseLatentPrefixChannel` and loads it with
`weights_only=True` and `strict=True`. It is not an `AutoModel` checkpoint.

## Evaluate the two published endpoints

```bash
python code/scripts/reproduce_qasc.py \
  --model-cache /path/to/hf-cache \
  --output-dir /path/to/new-qasc-reproduction \
  --conditions final_only matched \
  --expected-gpu H100
```

The driver checks the development-data, channel, and backbone-weight identities; uses the
original tokenization, padding, prefix insertion, batch sizes, and evaluation functions; and
compares every predicted label by example ID. It writes `predictions.jsonl`, `verification.json`,
and `mismatches.json`. A label mismatch produces a nonzero exit status after saving the results.
Matching the aggregate accuracy alone is not sufficient for an exact reproduction.

The published references contain 878/1,000 correct for `final_only` (one exchange) and
808/1,000 for `matched` (four exchanges). These are reference results, not a guarantee about
an untested environment. Existing output directories are rejected.

All released intervention endpoints can be requested with:

```bash
python code/scripts/reproduce_qasc.py \
  --model-cache /path/to/hf-cache \
  --output-dir /path/to/new-qasc-all-conditions \
  --conditions no_message matched final_only deranged zero random self
```

## Preserve tokenization and prefix behavior

The exact user prompt is:

```text
QASC multiple-choice task.
{question}
{evidence}
Use the supplied evidence when available. Return exactly one option label from A through H.
```

`question` already contains the formatted choices in the derived dataset. Each agent receives
its own `private_contexts` entry as `evidence`. The original tokenizer applies its chat template
with `add_generation_prompt=True`; the separately tokenized string `"Answer: "` (including the
trailing space) and one mask token are then appended. The maximum length is 768; overlength
inputs raise an error rather than being silently truncated.

Four continuous prefix embeddings are inserted immediately **before** the visible assistant
prefix. The answer-state position uses `output_position_offset: -1`. Predictions are constrained
to labels A through H. Agent 0 is the scored receiver; there is no voting.

`final_only` encodes the agents' unprefixed states and performs one final prefixed receiver
forward. `matched` repeats state extraction, message encoding, cross-agent aggregation, and
prefix projection four times before its final receiver readout. The original QASC implementation
keeps the input token rows fixed and replaces the continuous prefix between exchanges; do not
substitute a text-generation loop or append previously predicted answer tokens.

## Record intermediate messages

Add `--record-messages` to the endpoint command. The driver wraps the original `encode_all`
only to save its input and returned output; it returns the same tensor to the evaluator.
The original evaluation function and frozen weights are unchanged.

Each `<condition>_exchange_<NN>.pt` contains example IDs, agent order `[0, 1]`, sender states
`[1000, 2, model_dim]`, and encoded messages `[1000, 2, 4, 128]`. These are captured **before**
intervention, incoming aggregation, and prefix projection. For `matched` they are the sender
messages used by that exchange. For `deranged`, `zero`, or `random`, they are pre-intervention
messages, not the delivered payload. Exchange 1 uses the unprefixed states. `final_only`
produces one file; `matched` produces four. `token_rows.jsonl` records the exact inputs.

This instrumentation does not add intermediate receiver forward passes or new accuracy
experiments. Its I/O overhead makes recorded wall-clock times unsuitable for latency comparison.

## Rebuild the derived data (optional)

The included development file is sufficient for inference. To reconstruct all derived splits,
use the original builder with the pinned upstream dataset:

```bash
git ls-files -z code configs | xargs -0 sha256sum > /path/to/source-manifest.sha256
python code/scripts/phase7_build_qasc_distributed.py \
  --run-root /path/to/derived-data \
  --run-id qasc-rebuild-001 \
  --dataset-cache /path/to/qasc-source-cache \
  --project-source-revision "$(git rev-parse HEAD)" \
  --project-source-manifest /path/to/source-manifest.sha256 \
  --distractors-per-agent 4
```

The builder checks source Parquet hashes. It ranks the 8,134 official training examples by
salted SHA-256, holds out the first 1,000 for development, and uses the remaining 7,134 for
training. Distractors come only from the derived training fact pool. The official labeled
validation split becomes the separate 926-example held-out split; it is not the published
1,000-example evaluation set. The unlabeled official test is not emitted as an evaluation split.
Expected derived hashes are in `configs/qasc_original_evaluation.yaml`.

QASC attribution and the license for the redistributed derived file are in
[`LICENSES.md`](../LICENSES.md). No backbone weights are redistributed.
