"""Small, auditable backbone adaptations used only by explicitly labeled variants."""

from hidden_messages.adaptation.lora import (
    FrozenLinearLoRA,
    LoRAAttachment,
    attach_lora_to_decoder_layers,
    load_lora_adapter_state_dict,
    lora_adapter_state_dict,
    lora_named_parameters,
)

__all__ = [
    "FrozenLinearLoRA",
    "LoRAAttachment",
    "attach_lora_to_decoder_layers",
    "load_lora_adapter_state_dict",
    "lora_adapter_state_dict",
    "lora_named_parameters",
]
