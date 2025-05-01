from dataclasses import dataclass
from typing import Optional

import torch
from transformers.utils import ModelOutput


@dataclass
class BiEncoderModelOutput(torch.nn.Module):
    """
    Outputs of bi-encoder entity linking models.
    Args:
        query_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Hidden states of the mention encoder.
        candidates_hidden_state (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Hidden states of the entity encoder.
        hard_negatives_hidden_state (`torch.FloatTensor` of shape `(config.negative_num, hidden_size)`):
            Hidden states of the entity encoder.
    """

    query_hidden_state: torch.FloatTensor
    candidates_hidden_state: torch.FloatTensor
    hard_negatives_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class EntityLinkingOutput(ModelOutput):
    """
    Outputs of bi-encoder entity linking models.

    Args:
        loss (`torch.Tensor` of shape `(1,)`):
            Classification loss.
        logits (`torch.Tensor` of shape `(batch_size, batch_size + config.negative_num)`):
            Classification scores (before SoftMax).
        mention_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True`):
            `torch.Tensor` (one for the output of the mention_encoder) of shape `(batch_size, hidden_size)`.
        candidate_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True`):
            `torch.Tensor` (one for the output of the entity_encoder) of shape `(batch_size, batch_size + config.negative_num, hidden_size)`.
    """

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    mention_hidden_states: Optional[torch.Tensor] = None
    candidate_hidden_states: Optional[torch.Tensor] = None
