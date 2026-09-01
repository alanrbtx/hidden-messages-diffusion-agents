"""Contamination-free controlled distributed-information generators."""

from hidden_messages.datasets.hm_synth.batching import pair_preserving_control_batches
from hidden_messages.datasets.hm_synth.generator import (
    HMSynthExample,
    generate_chain_lookup_pair,
    generate_modular_arithmetic_pair,
    generate_rendezvous_lookup_pair,
)

__all__ = [
    "HMSynthExample",
    "generate_chain_lookup_pair",
    "generate_modular_arithmetic_pair",
    "generate_rendezvous_lookup_pair",
    "pair_preserving_control_batches",
]
