"""dLLM MDLM sampler with an optional hidden-state/logit callback.

The no-callback branch deliberately mirrors the pinned upstream implementation. The callback path
captures the exact tensor entering the model output head, which also works for dLLM backbones that
ignore the standard Transformers ``output_hidden_states`` argument.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from dllm.core.samplers.base import BaseSamplerOutput
from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

from hidden_messages.diffusion.callbacks import DenoisingCallback, DenoisingStep


class CallbackMDLMSampler(MDLMSampler):
    """Instrumented blockwise masked-diffusion sampler."""

    def _forward_with_hidden_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_embeddings = self.model.get_output_embeddings()
        if not isinstance(output_embeddings, torch.nn.Module):
            raise TypeError("callback sampling requires a module returned by get_output_embeddings")
        captured: list[torch.Tensor] = []

        def capture_hidden(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("output head did not receive a hidden-state tensor")
            captured.append(inputs[0])

        handle = output_embeddings.register_forward_pre_hook(capture_hidden)
        try:
            outputs = self.model(input_ids, attention_mask=attention_mask)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(f"expected one output-head call, observed {len(captured)}")
        return outputs.logits, captured[0]

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list[int]],
        config: MDLMSamplerConfig | None = None,
        *,
        callback: DenoisingCallback | None = None,
        **kwargs: Any,
    ) -> BaseSamplerOutput | torch.Tensor:
        if config is None:
            config = MDLMSamplerConfig()

        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        cfg_scale = kwargs.get("cfg_scale", config.cfg_scale)
        cfg_keep_tokens = kwargs.get("cfg_keep_tokens", config.cfg_keep_tokens)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        stochastic_transfer = kwargs.get("stochastic_transfer", config.stochastic_transfer)
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        begin_suppress_tokens = kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)

        if block_size < 1 or steps < 1:
            raise ValueError("block_size and steps must be positive")
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        if right_shift_logits:
            inputs = [
                [bos_id] if isinstance(prompt, list) and not prompt else prompt for prompt in inputs
            ]
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(prompt, dtype=torch.long, device=self.model.device)
                for prompt in inputs
            ]
        tensor_inputs = [torch.as_tensor(prompt, device=self.model.device) for prompt in inputs]
        prompt_lens = [prompt.shape[0] for prompt in tensor_inputs]

        if max_new_tokens:
            max_length = max_new_tokens + max(prompt_lens)
        elif max_length is not None:
            max_new_tokens = max_length - max(prompt_lens)
        else:
            raise ValueError("either max_new_tokens or max_length must be set")

        batch_size = len(tensor_inputs)
        canvas_width = int(max_length)
        x = torch.full(
            (batch_size, canvas_width),
            eos_id,
            dtype=torch.long,
            device=self.model.device,
        )
        for batch_index, prompt in enumerate(tensor_inputs):
            prompt_len = prompt_lens[batch_index]
            x[batch_index, :prompt_len] = prompt
            x[batch_index, prompt_len : prompt_len + max_new_tokens] = mask_id

        attention_mask = torch.zeros_like(x)
        for batch_index, prompt_len in enumerate(prompt_lens):
            valid_end = min(prompt_len + max_new_tokens, canvas_width)
            attention_mask[batch_index, :valid_end] = 1

        unmasked_index = (x != mask_id) & attention_mask.bool()
        if cfg_keep_tokens:
            keep_mask = torch.isin(x, torch.as_tensor(cfg_keep_tokens, device=x.device))
            unmasked_index = unmasked_index & ~keep_mask

        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)
        histories = [x.clone()] if return_dict else None
        initial_mask_count = max(int((x == mask_id).sum().item()), 1)

        for block_index in range(num_blocks):
            block_mask_index = torch.zeros(
                (batch_size, block_size),
                dtype=torch.bool,
                device=x.device,
            )
            for batch_index, prompt_len in enumerate(prompt_lens):
                start = prompt_len + block_index * block_size
                stop = min(start + block_size, prompt_len + max_new_tokens, canvas_width)
                if start < stop:
                    block_mask_index[batch_index, : stop - start] = (
                        x[batch_index, start:stop] == mask_id
                    )

            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            for step_index in range(effective_steps):
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[unmasked_index] = mask_id
                    model_input = torch.cat([x, un_x], dim=0)
                    model_attention = attention_mask.repeat(2, 1)
                    if callback is None:
                        model_logits = self.model(
                            model_input,
                            attention_mask=model_attention,
                        ).logits
                        model_hidden_states = None
                    else:
                        model_logits, model_hidden_states = self._forward_with_hidden_capture(
                            model_input,
                            model_attention,
                        )
                    logits, unconditional_logits = torch.chunk(model_logits, 2, dim=0)
                    logits = unconditional_logits + (cfg_scale + 1) * (
                        logits - unconditional_logits
                    )
                    hidden_states = (
                        torch.chunk(model_hidden_states, 2, dim=0)[0]
                        if model_hidden_states is not None
                        else None
                    )
                else:
                    if callback is None:
                        logits = self.model(x, attention_mask=attention_mask).logits
                        hidden_states = None
                    else:
                        logits, hidden_states = self._forward_with_hidden_capture(
                            x,
                            attention_mask,
                        )

                if suppress_tokens:
                    logits[:, :, suppress_tokens] = -torch.inf
                if right_shift_logits:
                    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                if callback is not None:
                    assert hidden_states is not None
                    normalized_time = float(mask_index.sum().item() / initial_mask_count)
                    logits = callback(
                        DenoisingStep(
                            block_index=block_index,
                            step_index=step_index,
                            normalized_time=normalized_time,
                            canvas=x.detach().clone(),
                            hidden_states=hidden_states,
                            pre_logits=logits,
                            attention_mask=attention_mask,
                            canvas_mask=mask_index,
                        )
                    )
                    if logits.shape[:2] != x.shape:
                        raise ValueError("callback logits must preserve [B, L, vocab] shape")

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if begin_suppress_tokens:
                    logits[:, :, begin_suppress_tokens] = -torch.inf

                if remasking == "low_confidence":
                    probabilities = functional.softmax(logits, dim=-1)
                    x0_probability = torch.gather(probabilities, -1, x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_probability = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                for batch_index, prompt_len in enumerate(prompt_lens):
                    x0_probability[
                        batch_index,
                        prompt_len + (block_index + 1) * block_size :,
                    ] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_probability, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for batch_index in range(batch_size):
                    count = int(num_transfer_tokens[batch_index, step_index].item())
                    if count:
                        selected = torch.topk(confidence[batch_index], k=count).indices
                        transfer_index[batch_index, selected] = True
                x[transfer_index] = x0[transfer_index]
                if histories is not None:
                    histories.append(x.clone())

        if return_dict:
            return BaseSamplerOutput(sequences=x, histories=histories)
        return x
