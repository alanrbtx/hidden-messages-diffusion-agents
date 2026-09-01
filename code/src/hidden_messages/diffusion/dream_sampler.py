"""Read-only hidden-state callbacks around the pinned upstream Dream sampler."""

from __future__ import annotations

from typing import Any

import torch
from dllm.core.samplers.base import BaseSamplerOutput
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig

from hidden_messages.diffusion.callbacks import DenoisingCallback, DenoisingStep


def _identity_tokens(
    _step: int | None, canvas: torch.Tensor, _logits: torch.Tensor | None
) -> torch.Tensor:
    return canvas


def _identity_logits(_step: int, _canvas: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return logits


class CallbackDreamSampler(DreamSampler):
    """Preserve upstream decoding and expose final hidden states at each Dream step."""

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: DreamSamplerConfig | None = None,
        generation_tokens_hook_func: Any = _identity_tokens,
        generation_logits_hook_func: Any = _identity_logits,
        *,
        callback: DenoisingCallback | None = None,
        **kwargs: Any,
    ) -> BaseSamplerOutput | torch.Tensor:
        if callback is None:
            return super().sample(
                inputs,
                config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                **kwargs,
            )

        resolved_config = config or DreamSamplerConfig()
        max_new_tokens = kwargs.get("max_new_tokens", resolved_config.max_new_tokens)
        max_length = kwargs.get("max_length", resolved_config.max_length)
        prompt_lens = [len(prompt) for prompt in inputs]
        if max_new_tokens:
            canvas_width = max_new_tokens + max(prompt_lens)
        elif max_length is not None:
            canvas_width = max_length
            max_new_tokens = canvas_width - max(prompt_lens)
        else:
            raise ValueError("either max_new_tokens or max_length must be set")

        batch_size = len(inputs)
        sequence_lens = [prompt_len + max_new_tokens for prompt_len in prompt_lens]
        attention_mask = torch.zeros(
            (batch_size, canvas_width),
            dtype=torch.long,
            device=self.model.device,
        )
        for batch_index, sequence_len in enumerate(sequence_lens):
            attention_mask[batch_index, -sequence_len:] = 1

        output_embeddings = self.model.get_output_embeddings()
        if not isinstance(output_embeddings, torch.nn.Module):
            raise TypeError("callback sampling requires a module returned by get_output_embeddings")
        captured: list[torch.Tensor] = []
        capture_enabled = True

        def capture_hidden(
            _module: torch.nn.Module,
            hook_inputs: tuple[Any, ...],
        ) -> None:
            if not capture_enabled:
                return
            if not hook_inputs or not isinstance(hook_inputs[0], torch.Tensor):
                raise RuntimeError("output head did not receive a hidden-state tensor")
            captured.append(hook_inputs[0])

        initial_mask_count = max(batch_size * max_new_tokens, 1)

        def callback_logits(step: int, canvas: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            nonlocal capture_enabled
            if len(captured) != 1:
                raise RuntimeError(f"expected one hidden capture, observed {len(captured)}")
            hidden_states = captured.pop()
            if hidden_states.shape[0] == 2 * batch_size:
                hidden_states = torch.chunk(hidden_states, 2, dim=0)[0]
            canvas_mask = canvas == self.tokenizer.mask_token_id
            capture_enabled = False
            try:
                fused_logits = callback(
                    DenoisingStep(
                        block_index=0,
                        step_index=step,
                        normalized_time=float(canvas_mask.sum().item() / initial_mask_count),
                        canvas=canvas.detach().clone(),
                        hidden_states=hidden_states,
                        pre_logits=logits,
                        attention_mask=attention_mask,
                        canvas_mask=canvas_mask,
                    )
                )
            finally:
                capture_enabled = True
            if fused_logits.shape != logits.shape:
                raise ValueError("callback logits must preserve [B, L, vocab] shape")
            return generation_logits_hook_func(step, canvas, fused_logits)

        handle = output_embeddings.register_forward_pre_hook(capture_hidden)
        try:
            result = super().sample(
                inputs,
                resolved_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=callback_logits,
                **kwargs,
            )
        finally:
            handle.remove()
        if captured:
            raise RuntimeError("unconsumed hidden-state capture after Dream sampling")
        return result
