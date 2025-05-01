import os
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from transformers.models.luke.modeling_luke import BaseModelOutputWithPooling

from mix_blink import MixBlinkConfig

from .model_output import BiEncoderModelOutput

TOKEN = os.environ.get("TOKEN", True)


class Transformer(nn.Module):
    def __init__(self, model_name: str, config: dict | PretrainedConfig | None, from_pretrained: bool = False) -> None:
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(model_name, token=TOKEN)

        if not isinstance(config, PretrainedConfig):
            raise ValueError(f"Unspecified types: {type(config)}. Expected: PretrainedConfig")

        if from_pretrained:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=TOKEN)
        else:
            self.model = AutoModel.from_config(config, trust_remote_code=True)

        self.config = config

    def forward(self, *args: int , **kwargs: str) -> BaseModelOutputWithPooling:
        output = self.model(*args, return_dict = True, **kwargs)
        return output


class Encoder(nn.Module):
    def __init__(self, config: MixBlinkConfig, from_pretrained: bool = False, entity_encoder: bool = False) -> None:
        super().__init__()
        if entity_encoder:
            self.bert_layer = Transformer(config.entity_encoder, config.entity_encoder_config, from_pretrained)
        else:
            self.bert_layer = Transformer(config.mention_encoder, config.mention_encoder_config, from_pretrained)

        self.config = self.bert_layer.config
        bert_hidden_size = self.bert_layer.model.config.hidden_size
        if config.hidden_size != bert_hidden_size:
            self.projection = nn.Linear(bert_hidden_size, config.hidden_size)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> nn.Embedding:
        self.bert_layer.config.vocab_size = new_num_tokens
        return self.bert_layer.model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    def get_input_embeddings(self) -> nn.Module:
        return self.bert_layer.model.get_input_embeddings()

    def freeze_parameters(self) -> None:
        for param in self.bert_layer.model.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None
        ) -> torch.Tensor:

        if token_type_ids is None:
            outputs = self.bert_layer(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.bert_layer(input_ids, attention_mask=attention_mask, token_type_ids = token_type_ids)
        logits = outputs.last_hidden_state[:, 0, :]

        if hasattr(self, "projection"):
            logits = self.projection(logits)
        return logits


class BiEncoder(nn.Module):
    def __init__(self, config: MixBlinkConfig, from_pretrained: bool = False) -> None:
        """
        Args:
            config (`MixBlinkConfig`):
                Configuration of the model.
            from_pretrained (bool):
                True if you use pretrained model else False
        """
        super().__init__()
        self.mention_encoder = Encoder(config, from_pretrained)
        self.entity_encoder = Encoder(config, from_pretrained, entity_encoder=True)
        if not config.mention_encoder_config:
            config.mention_encoder_config = self.mention_encoder.config
        if not config.entity_encoder_config:
            config.entity_encoder_config = self.entity_encoder.config
        self.config = config

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            candidates_input_ids: torch.Tensor,
            candidates_attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            candidates_token_type_ids: Optional[torch.Tensor] = None,
            hard_negatives_input_ids: Optional[torch.Tensor] = None,
            hard_negatives_attention_mask: Optional[torch.Tensor] = None,
            hard_negatives_token_type_ids: Optional[torch.Tensor] = None,
        ) -> BiEncoderModelOutput:

        queries = self.mention_encoder(input_ids, attention_mask, token_type_ids)
        candidates = self.entity_encoder(candidates_input_ids, candidates_attention_mask, candidates_token_type_ids)

        if isinstance(hard_negatives_input_ids, torch.Tensor) and isinstance(hard_negatives_attention_mask, torch.Tensor):
            hard_negatives = self.entity_encoder(hard_negatives_input_ids, hard_negatives_attention_mask, hard_negatives_token_type_ids)
        else:
            hard_negatives = None

        return BiEncoderModelOutput(queries, candidates, hard_negatives)
