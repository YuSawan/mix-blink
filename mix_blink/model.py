
import json
import os
import re
from pathlib import Path
from typing import Optional, Self, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers.models.luke.modeling_luke import BaseLukeModelOutputWithPooling

from .config import MixBlinkConfig
from .modeling import BiEncoder, Encoder


class MixBlink(nn.Module):
    # to suppress an AttributeError when training
    _keys_to_ignore_on_save = None

    def __init__(self, config: MixBlinkConfig, encoder_from_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.model = BiEncoder(config, encoder_from_pretrained)

        if config.mention_encoder_vocab_size == -1:
            config.mention_encoder_vocab_size = self.entity_encoder.config.vocab_size
        if config.mention_encoder_vocab_size != self.mention_encoder.config.vocab_size:
            self.mention_encoder.resize_token_embeddings(config.mention_encoder_vocab_size)

        if config.entity_encoder_vocab_size == -1:
            config.entity_encoder_vocab_size = self.model.entity_encoder.config.vocab_size
        if config.entity_encoder_vocab_size != self.model.entity_encoder.config.vocab_size:
            self.entity_encoder.resize_token_embeddings(config.entity_encoder_vocab_size)

        if config.freeze_entity_encoder:
            self.entity_encoder.freeze_parameters()
        if config.freeze_mention_encoder:
            self.mention_encoder.freeze_parameters()

    def forward(self, *args: int, **kwargs: str) -> BaseLukeModelOutputWithPooling:
        """Wrapper function for the model's forward pass."""
        output = self.model(*args, **kwargs)
        return output

    @property
    def mention_encoder(self) -> Encoder:
        return self.model.mention_encoder

    @property
    def entity_encoder(self) -> Encoder:
        return self.model.entity_encoder

    @property
    def device(self) -> torch.device:
        device = next(self.model.parameters()).device
        return device

    def prepare_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Prepare state dict in the case of torch.compile
        """
        new_state_dict = {}
        for key, tensor in state_dict.items():
            key = re.sub(r"_orig_mod\.", "", key)
            new_state_dict[key] = tensor
        return new_state_dict

    def save_pretrained(
        self,
        save_directory: Union[str, Path],
        *,
        config: Optional[MixBlinkConfig] = None,
        safe_serialization: bool = False,
    ) -> Optional[str]:
        """
        Save weights in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the model weights and configuration will be saved.
            safe_serialization (`bool`):
                Whether to save the model using `safetensors` or the traditional way for PyTorch.
            config (`dict` or `DataclassInstance`, *optional*):
                Model configuration specified as a key/value dictionary or a dataclass instance.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save model weights/files
        # model_state_dict = self.prepare_state_dict(self.model.state_dict())
        model_state_dict = self.model.state_dict()
        # save model weights using safetensors
        if safe_serialization:
            save_file(model_state_dict, os.path.join(save_directory, "model.safetensors"))
        else:
            torch.save(
                model_state_dict,
                os.path.join(save_directory, "pytorch_model.bin"),
            )

        # save config (if provided)
        if config is None:
            config = self.config
        if config is not None:
            config.to_json_file(save_directory / "config.json")

        return None

    @classmethod
    def from_pretrained(cls, model_id: str, map_location: str = "cpu", strict: bool = False) -> Self:
        """
        Load a pretrained model from a given model ID.

        Args:
            model_id (str): Identifier of the model to load.
            map_location (str): Device to map model to. Defaults to "cpu".
            strict (bool): Enforce strict state_dict loading.

        Returns:
            An instance of the model loaded from the pretrained weights.
        """

        model_dir = Path(model_id)  # / "pytorch_model.bin"
        model_file = os.path.join(model_dir, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(model_dir, "pytorch_model.bin")
        config_file = Path(model_dir) / "config.json"

        with open(config_file, "r") as f:
            config_ = json.load(f)
        config = MixBlinkConfig(**config_)
        mixblink = cls(config, encoder_from_pretrained=False)

        if model_file.endswith("safetensors"):
            state_dict = {}
            with safe_open(model_file, framework="pt", device=map_location) as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            state_dict = torch.load(model_file, map_location=torch.device(map_location), weights_only=True)
        mixblink.model.load_state_dict(state_dict, strict=strict)
        mixblink.model.to(map_location)

        mixblink.eval()

        return mixblink
