from typing import Optional

from transformers import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class MixBlinkConfig(PretrainedConfig):
    model_type = "mixblink"
    is_composition = True
    def __init__(self,
            mention_encoder: str = "studio-ousia/luke-base-lite",
            entity_encoder: str = "studio-ousia/luke-base-lite",
            mention_encoder_config: Optional[dict | PretrainedConfig] = None,
            entity_encoder_config: Optional[dict | PretrainedConfig] = None,
            hidden_size: int = 768,
            mention_encoder_vocab_size: int = -1,
            entity_encoder_vocab_size: int = -1,
            freeze_mention_encoder: bool = False,
            freeze_entity_encoder: bool = False,
            **kwargs: str
        ) -> None:
        super().__init__(**kwargs)
        if isinstance(mention_encoder_config, dict):
            mention_encoder_config["model_type"] = (mention_encoder_config["model_type"] if "model_type" in mention_encoder_config else "studio-ousia/mluke-base-lite")
            mention_encoder_config = CONFIG_MAPPING[mention_encoder_config["model_type"]](**mention_encoder_config)
        self.mention_encoder_config = mention_encoder_config

        if isinstance(entity_encoder_config, dict):
            entity_encoder_config["model_type"] = (entity_encoder_config["model_type"] if "model_type" in entity_encoder_config else "studio-ousia/mluke-base-lite")
            entity_encoder_config = CONFIG_MAPPING[entity_encoder_config["model_type"]](**entity_encoder_config)
        self.entity_encoder_config = entity_encoder_config

        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder
        self.hidden_size = hidden_size
        self.freeze_mention_encoder = freeze_mention_encoder
        self.freeze_entity_encoder = freeze_entity_encoder

        self.mention_encoder_vocab_size = mention_encoder_vocab_size
        self.entity_encoder_vocab_size = entity_encoder_vocab_size

CONFIG_MAPPING.update({"mixblink": MixBlinkConfig})
