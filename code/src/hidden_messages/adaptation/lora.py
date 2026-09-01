"""Minimal LoRA wrappers that keep pretrained linear layers immutable."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn


class FrozenLinearLoRA(nn.Module):
    """Add a trainable low-rank residual to one frozen linear projection."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, dtype=torch.float32)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device)
        self.lora_b.to(device=base.weight.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_inputs = self.dropout(inputs).to(dtype=self.lora_a.weight.dtype)
        delta = self.lora_b(self.lora_a(adapter_inputs))
        return base_output + delta.to(dtype=base_output.dtype) * self.scaling


@dataclass(frozen=True, slots=True)
class LoRAAttachment:
    """One declared projection replaced by a low-rank wrapper."""

    layer_index: int
    module_name: str
    qualified_name: str
    rank: int
    alpha: float


def _projection_parent(layer: nn.Module, module_name: str) -> nn.Module:
    if module_name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        parent = getattr(layer, "self_attn", None)
    elif module_name in {"gate_proj", "up_proj", "down_proj"}:
        parent = getattr(layer, "mlp", None)
    else:
        raise ValueError(f"unsupported LoRA target module: {module_name}")
    if not isinstance(parent, nn.Module):
        raise TypeError(f"decoder layer has no module parent for {module_name}")
    return parent


def attach_lora_to_decoder_layers(
    decoder: nn.Module,
    *,
    layer_indices: list[int],
    target_modules: list[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[LoRAAttachment]:
    """Attach LoRA only to the explicitly listed decoder layers and projections."""

    layers: Any = getattr(decoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("decoder.layers must be a torch ModuleList")
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer_indices must be a non-empty unique list")
    if not target_modules or len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules must be a non-empty unique list")
    attachments: list[LoRAAttachment] = []
    for layer_index in layer_indices:
        if not 0 <= layer_index < len(layers):
            raise IndexError(f"LoRA layer index {layer_index} is outside the decoder")
        layer = layers[layer_index]
        for module_name in target_modules:
            parent = _projection_parent(layer, module_name)
            projection = getattr(parent, module_name, None)
            if isinstance(projection, FrozenLinearLoRA):
                raise RuntimeError(
                    f"decoder layer {layer_index} projection {module_name} already has LoRA"
                )
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"decoder layer {layer_index} projection {module_name} is not linear"
                )
            wrapped = FrozenLinearLoRA(
                projection,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(parent, module_name, wrapped)
            parent_name = (
                "self_attn" if module_name.endswith("_proj") and module_name[0] in "qkvo" else "mlp"
            )
            attachments.append(
                LoRAAttachment(
                    layer_index=layer_index,
                    module_name=module_name,
                    qualified_name=f"layers.{layer_index}.{parent_name}.{module_name}",
                    rank=rank,
                    alpha=float(alpha),
                )
            )
    return attachments


def lora_named_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return only trainable A/B parameters from attached LoRA wrappers."""

    parameters: list[tuple[str, nn.Parameter]] = []
    for module_name, child in module.named_modules():
        if not isinstance(child, FrozenLinearLoRA):
            continue
        parameters.extend(
            (
                (
                    f"{module_name}.lora_a.weight",
                    cast(nn.Parameter, child.lora_a.weight),
                ),
                (
                    f"{module_name}.lora_b.weight",
                    cast(nn.Parameter, child.lora_b.weight),
                ),
            )
        )
    return parameters


def lora_adapter_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU state dictionary containing no pretrained weights."""

    return {
        name: parameter.detach().to(device="cpu").clone()
        for name, parameter in lora_named_parameters(module)
    }


def load_lora_adapter_state_dict(
    module: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    """Load an exact adapter-only state dictionary into already attached wrappers."""

    parameters = dict(lora_named_parameters(module))
    if set(state) != set(parameters):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise RuntimeError(
            f"LoRA checkpoint keys differ; missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise RuntimeError(
                    f"LoRA checkpoint shape differs for {name}: {value.shape} != {parameter.shape}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
