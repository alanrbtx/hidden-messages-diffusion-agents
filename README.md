# Hidden Messages Between Diffusion Language Model Agents

Anonymous supplementary material for the paper
“Hidden Messages Between Diffusion Language Model Agents: Causal Evidence for Latent
Communication.”

This archive packages the implementation, configurations, channel weights, per-example
predictions, and aggregate evidence used in the paper. It contains no newly run experiment.

## Contents

- `code/src/hidden_messages/`: latent-channel, diffusion-sampling, and dataset code.
- `code/scripts/`: QASC construction, controlled-task evaluation, QASC evaluation, aggregation,
  and table-generation entry points.
- `configs/`: paper-facing configurations for distributed QASC and the Dream-7B rendezvous.
- `evidence/tiny_controlled_results.json`: Tiny-A2D HM-Synth accuracies, paired bootstrap
  intervals, and targeted-intervention outcomes selected from the immutable run artifact.
- `evidence/tiny_controlled_predictions.jsonl`: all 2,000 HM-Synth test examples under every
  intervention.
- `evidence/qasc_results.json`: QASC accuracies, paired bootstrap intervals, channel diagnostics,
  and integrity checks selected directly from the immutable run artifact.
- `evidence/qasc_predictions.jsonl`: all 1,000 QASC development examples under every intervention.
- `evidence/controlled_results.json`: aggregate controlled-task results and replicate metadata
  selected directly from the immutable aggregate artifact.
- `evidence/controlled_predictions_rep*.jsonl`: the locked 1,024-example evaluation rows for
  each controlled replicate.
- `checkpoints/qasc_dense_latent_prefix_channel.pt`: the learned QASC channel state.
- `checkpoints/tiny_a2d_channel.pt`: the learned Tiny-A2D HM-Synth communication module.
- `SHA256SUMS`: integrity inventory for the unpacked supplement.

## Reproducing paper values

From the archive root, generate the LaTeX result macros and table with:

```bash
python code/scripts/generate_results.py \
  --phase5 evidence/tiny_controlled_results.json \
  --phase8 evidence/qasc_results.json \
  --phase6 evidence/controlled_results.json \
  --output-dir generated
```

The command only transforms preserved JSON evidence; it does not rerun a model. Full model
execution requires Tiny-A2D and Dream-v0-Instruct-7B at the pinned revisions, the HM-Synth and QASC
configurations in `configs/`, and a CUDA environment with the package versions in
`code/pyproject.toml`. The experimental entry points expose their required paths through `--help`.

## Data and model access

The Tiny-A2D and Dream model weights and QASC source data are not redistributed. Their identifiers,
revisions, row counts, and content hashes are recorded in the configurations and evidence. See
`LICENSES.md` for attribution and license information.
