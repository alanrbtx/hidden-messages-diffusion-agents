# Third-party assets

- **Tiny-A2D / Qwen3-0.6B-diffusion-mdlm-v0.1** — dLLM, Apache License 2.0.
  Model card: https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1
- **Dream-v0-Instruct-7B** — Dream-org, Apache License 2.0.
  Paper: https://arxiv.org/abs/2508.15487
- **QASC** — Khot et al., CC BY 4.0.
  Paper: https://arxiv.org/abs/1910.11473

The archive does not redistribute the Tiny-A2D or Dream base-model weights.
`data/qasc/validation.jsonl` is a derived subset of QASC, by Tushar Khot, Peter Clark,
Michal Guerquin, Peter Jansen, and Ashish Sabharwal, distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source: [allenai/qasc](https://huggingface.co/datasets/allenai/qasc), revision
`a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`.
Changes: a deterministic development subset, two private evidence contexts per example,
four distractors per agent, formatted answer options, and derived identifiers. These are
the preserved experimental inputs, not newly generated examples.
Exact pinned revisions remain in the machine-readable configurations and upstream manifest.
