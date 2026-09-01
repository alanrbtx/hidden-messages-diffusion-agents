"""Trainable hidden-message channel and causal interventions."""

from hidden_messages.communication.compressor import MessageCompressor
from hidden_messages.communication.dense_latent_prefix import DenseLatentPrefixChannel
from hidden_messages.communication.fusion import MessageFusion
from hidden_messages.communication.input_prefix import InputPrefixChannel
from hidden_messages.communication.interventions import apply_message_intervention
from hidden_messages.communication.regularization import batch_message_variance_loss
from hidden_messages.communication.semantic_prototype import SemanticPrototypeChannel
from hidden_messages.communication.transport import (
    all_to_all_graph_mask,
    all_to_all_messages,
    payload_bytes,
)

__all__ = [
    "DenseLatentPrefixChannel",
    "InputPrefixChannel",
    "MessageCompressor",
    "MessageFusion",
    "SemanticPrototypeChannel",
    "all_to_all_graph_mask",
    "all_to_all_messages",
    "apply_message_intervention",
    "batch_message_variance_loss",
    "payload_bytes",
]
