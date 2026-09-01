"""Dream decoding with a persistent virtual-prefix cache between denoising passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig
from dllm.pipelines.dream.sampler import sample_tokens


class PrefixForward(Protocol):
    """One frozen Dream forward with an optional receiver-side virtual prefix."""

    def __call__(
        self,
        canvas: torch.Tensor,
        attention_mask: torch.Tensor,
        prefix_embeddings: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unshifted hidden states and logits on the original canvas width."""


@dataclass(frozen=True, slots=True)
class PrefixDenoisingStep:
    step_index: int
    normalized_time: float
    canvas: torch.Tensor
    hidden_states: torch.Tensor
    pre_logits: torch.Tensor
    attention_mask: torch.Tensor
    canvas_mask: torch.Tensor
    prefix_embeddings: torch.Tensor | None


class PrefixCacheCallback(Protocol):
    """Update the message-derived prefix cache for the next Dream pass."""

    def __call__(self, step: PrefixDenoisingStep) -> torch.Tensor | None:
        """Return the cache used by the next pass; never mutate the current canvas."""


@dataclass(slots=True)
class PersistentPrefixSamplerOutput:
    sequences: torch.Tensor
    histories: list[torch.Tensor]
    final_hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    answer_reveal_steps: torch.Tensor
    generation_reveal_steps: torch.Tensor
    prefix_present_steps: list[bool]
    effective_steps: int
    backbone_forward_calls: int
    answer_commit_not_before_steps: torch.Tensor | None = None
    answer_commit_scope: str = "first_token"
    deferred_transfer_counts_final: torch.Tensor | None = None


class PersistentPrefixDreamSampler:
    """Reproduce Dream MaskGIT updates while carrying prefixes across passes.

    A callback observes the current pass and returns a prefix for the *next* pass. This enforces a
    Jacobi boundary: newly produced messages cannot affect the states from which they were built.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prefix_forward: PrefixForward,
        *,
        scheduler: BaseAlphaScheduler | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.prefix_forward = prefix_forward
        self.scheduler = scheduler or LinearAlphaScheduler()

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor] | list[list[int]],
        config: DreamSamplerConfig,
        *,
        callback: PrefixCacheCallback | None = None,
        initial_prefix: torch.Tensor | None = None,
        answer_commit_not_before_steps: torch.Tensor | None = None,
        answer_commit_scope: str = "first_token",
    ) -> PersistentPrefixSamplerOutput:
        max_new_tokens = int(config.max_new_tokens)
        steps = int(config.steps)
        if not inputs or max_new_tokens <= 0 or steps <= 0:
            raise ValueError("inputs, max_new_tokens, and steps must be non-empty and positive")
        if config.cfg_scale != 0.0:
            raise ValueError("persistent-prefix sampler does not support classifier-free guidance")
        if answer_commit_scope not in {"first_token", "full_generation"}:
            raise ValueError(f"unknown answer commit scope: {answer_commit_scope}")
        mask_token_id = self.tokenizer.mask_token_id
        eos_token_id = self.tokenizer.eos_token_id
        if mask_token_id is None or eos_token_id is None:
            raise RuntimeError("Dream tokenizer requires mask and EOS token IDs")

        device = next(self.model.parameters()).device
        tensor_inputs = [
            torch.as_tensor(prompt, dtype=torch.long, device=device) for prompt in inputs
        ]
        prompt_lengths = [int(prompt.shape[0]) for prompt in tensor_inputs]
        batch_size = len(tensor_inputs)
        canvas_width = max(prompt_lengths) + max_new_tokens
        canvas = torch.full(
            (batch_size, canvas_width),
            int(eos_token_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(canvas, dtype=torch.long)
        for row_index, prompt in enumerate(tensor_inputs):
            total_length = prompt_lengths[row_index] + max_new_tokens
            start = canvas_width - total_length
            canvas[row_index, start : start + prompt_lengths[row_index]] = prompt
            canvas[row_index, start + prompt_lengths[row_index] :] = int(mask_token_id)
            attention_mask[row_index, -total_length:] = 1

        initial_mask = canvas == int(mask_token_id)
        transfer_counts = get_num_transfer_tokens(
            mask_index=initial_mask,
            steps=steps,
            scheduler=self.scheduler,
            stochastic=bool(config.stochastic_transfer),
        )
        effective_steps = int(transfer_counts.shape[1])
        histories = [canvas.clone()]
        answer_reveal_steps = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        answer_position = canvas_width - max_new_tokens
        generation_reveal_steps = torch.full(
            (batch_size, max_new_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )
        if callback is None and initial_prefix is None and answer_commit_not_before_steps is None:
            upstream = DreamSampler(
                model=self.model,
                tokenizer=self.tokenizer,
                scheduler=self.scheduler,
            ).sample(tensor_inputs, config, return_dict=True)
            if isinstance(upstream, torch.Tensor) or upstream.histories is None:
                raise RuntimeError("pinned Dream did not return trajectory histories")
            for step_index, history in enumerate(upstream.histories[1:]):
                newly_revealed_generation = torch.logical_and(
                    generation_reveal_steps < 0,
                    history[:, answer_position:] != int(mask_token_id),
                )
                generation_reveal_steps[newly_revealed_generation] = step_index
                newly_revealed = torch.logical_and(
                    answer_reveal_steps < 0,
                    history[:, answer_position] != int(mask_token_id),
                )
                answer_reveal_steps[newly_revealed] = step_index
            final_hidden, _ = self.prefix_forward(upstream.sequences, attention_mask, None)
            return PersistentPrefixSamplerOutput(
                sequences=upstream.sequences,
                histories=upstream.histories,
                final_hidden_states=final_hidden,
                attention_mask=attention_mask,
                answer_reveal_steps=answer_reveal_steps,
                generation_reveal_steps=generation_reveal_steps,
                prefix_present_steps=[False] * effective_steps,
                effective_steps=effective_steps,
                backbone_forward_calls=effective_steps + 1,
                answer_commit_scope=answer_commit_scope,
            )
        prefix = initial_prefix.detach().clone() if initial_prefix is not None else None
        if prefix is not None and prefix.shape[0] != batch_size:
            raise ValueError("initial prefix batch does not match the sampler batch")
        commit_steps: torch.Tensor | None = None
        if answer_commit_not_before_steps is not None:
            commit_steps = torch.as_tensor(
                answer_commit_not_before_steps,
                dtype=torch.long,
                device=device,
            ).reshape(-1)
            if commit_steps.shape != (batch_size,):
                raise ValueError("answer commit schedule must contain one step per sampler row")
            if bool((commit_steps < 0).any()) or bool((commit_steps >= effective_steps).any()):
                raise ValueError("answer commit steps must lie inside the denoising schedule")
        prefix_present_steps: list[bool] = []
        deferred_transfer_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        for step_index in range(effective_steps):
            mask_index = canvas == int(mask_token_id)
            normalized_time = float(
                mask_index[:, -max_new_tokens:].sum().item() / max(batch_size * max_new_tokens, 1)
            )
            hidden_states, logits = self.prefix_forward(canvas, attention_mask, prefix)
            if hidden_states.shape[:2] != canvas.shape or logits.shape[:2] != canvas.shape:
                raise ValueError("prefix forward must preserve batch and original canvas width")
            if bool(config.right_shift_logits):
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            prefix_present_steps.append(prefix is not None)

            if callback is not None:
                protected_canvas = canvas.detach().clone()
                prefix_checksum = prefix.detach().clone() if prefix is not None else None
                next_prefix = callback(
                    PrefixDenoisingStep(
                        step_index=step_index,
                        normalized_time=normalized_time,
                        canvas=protected_canvas,
                        hidden_states=hidden_states,
                        pre_logits=logits,
                        attention_mask=attention_mask,
                        canvas_mask=mask_index,
                        prefix_embeddings=prefix,
                    )
                )
                if not torch.equal(canvas, protected_canvas):
                    raise AssertionError("prefix callback mutated the sampler canvas")
                if prefix is not None:
                    if prefix_checksum is None:
                        raise AssertionError("current prefix checksum was not captured")
                    if not torch.equal(prefix, prefix_checksum):
                        raise AssertionError("prefix callback mutated the current cache")
                if next_prefix is not None:
                    if next_prefix.ndim != 3 or next_prefix.shape[0] != batch_size:
                        raise ValueError("callback prefix must have shape [B, K, model_dim]")
                    prefix = next_prefix.detach().clone()
                else:
                    prefix = None

            mask_logits = logits[mask_index]
            if str(config.alg) == "maskgit_plus":
                confidence, sampled = sample_tokens(
                    mask_logits,
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                    top_k=int(config.top_k) if config.top_k is not None else None,
                )
            elif str(config.alg) == "topk_margin":
                confidence, sampled = sample_tokens(
                    mask_logits,
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                    top_k=int(config.top_k) if config.top_k is not None else None,
                    margin_confidence=True,
                )
            elif str(config.alg) == "entropy":
                confidence, sampled = sample_tokens(
                    mask_logits,
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                    top_k=int(config.top_k) if config.top_k is not None else None,
                    neg_entropy=True,
                )
            else:
                raise RuntimeError(f"unknown Dream algorithm: {config.alg}")

            full_confidence = torch.full_like(canvas, -torch.inf, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            proposed = torch.full_like(canvas, int(mask_token_id))
            proposed[mask_index] = sampled
            for row_index in range(batch_size):
                requested_transfers = int(
                    transfer_counts[row_index, step_index].item()
                    + deferred_transfer_counts[row_index].item()
                )
                eligible = mask_index[row_index].clone()
                if commit_steps is not None and step_index < int(commit_steps[row_index].item()):
                    if answer_commit_scope == "full_generation":
                        eligible[answer_position:] = False
                    else:
                        eligible[answer_position] = False
                eligible_count = int(eligible.sum().item())
                transfer_count = min(requested_transfers, eligible_count)
                deferred_transfer_counts[row_index] = requested_transfers - transfer_count
                if transfer_count <= 0:
                    continue
                row_confidence = full_confidence[row_index].clone()
                row_confidence[~eligible] = -torch.inf
                if config.alg_temp is None or float(config.alg_temp) == 0.0:
                    selected = torch.topk(row_confidence, transfer_count).indices
                else:
                    probabilities = F.softmax(row_confidence / float(config.alg_temp), dim=-1)
                    selected = torch.multinomial(probabilities, num_samples=transfer_count)
                was_masked = bool(canvas[row_index, answer_position] == int(mask_token_id))
                canvas[row_index, selected] = proposed[row_index, selected]
                generated_selected = selected[selected >= answer_position] - answer_position
                if generated_selected.numel():
                    unrevealed = generation_reveal_steps[row_index, generated_selected] < 0
                    generation_reveal_steps[row_index, generated_selected[unrevealed]] = step_index
                if (
                    was_masked
                    and canvas[row_index, answer_position] != int(mask_token_id)
                    and answer_reveal_steps[row_index] < 0
                ):
                    answer_reveal_steps[row_index] = step_index
            histories.append(canvas.clone())

        if commit_steps is not None:
            if bool(deferred_transfer_counts.any()):
                raise AssertionError("protected answer transfers remained deferred after denoising")
            if bool((canvas[:, -max_new_tokens:] == int(mask_token_id)).any()):
                raise AssertionError("protected denoising left masked answer positions")

        final_hidden, _ = self.prefix_forward(canvas, attention_mask, prefix)
        return PersistentPrefixSamplerOutput(
            sequences=canvas,
            histories=histories,
            final_hidden_states=final_hidden,
            attention_mask=attention_mask,
            answer_reveal_steps=answer_reveal_steps,
            generation_reveal_steps=generation_reveal_steps,
            prefix_present_steps=prefix_present_steps,
            effective_steps=effective_steps,
            backbone_forward_calls=effective_steps + 1,
            answer_commit_not_before_steps=(
                commit_steps.detach().cpu() if commit_steps is not None else None
            ),
            answer_commit_scope=answer_commit_scope,
            deferred_transfer_counts_final=deferred_transfer_counts.detach().cpu(),
        )
