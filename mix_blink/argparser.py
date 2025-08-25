import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Optional

import yaml


def load_config_as_namespace(config_file: str | os.PathLike) -> Namespace:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


@dataclass
class DatasetArguments:
    """Dataset arguments."""
    dictionary_file: Optional[str] = None
    train_file : Optional[str] = None
    validation_file : Optional[str] = None
    test_file: Optional[str] = None
    add_nil: bool = False
    nil_label: str = "[NIL]"
    nil_description: str = "[NIL] is an entity that does not exist in the dataset."
    start_mention_token: str = "[START_ENT]"
    end_mention_token: str = "[END_ENT]"
    entity_token: str = "[ENT]"
    language: str = "en"
    cache_dir: Optional[str] = None


@dataclass
class ModelArguments:
    """Model arguments."""
    mention_encoder: str
    entity_encoder: str
    hidden_size: int
    mention_context_length: int = 512
    entity_context_length: int = 512
    freeze_mention_encoder: bool = False
    freeze_entity_encoder: bool = False
    measure: str = 'ip'
    temperature: float = 1.0
    negative: bool = False
    model_path: Optional[str] = None
    top_k: int = 10
    retriever_path: Optional[str] = None
