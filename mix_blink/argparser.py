import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from typing import Optional

import yaml
from transformers import HfArgumentParser, TrainingArguments


def load_config_as_namespace(config_file: str | os.PathLike) -> Namespace:
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return Namespace(**config_dict)


@dataclass
class DatasetArguments:
    """Dataset arguments."""
    dictionary_file: str
    train_file : str
    validation_file : str
    test_file: str
    add_nil: bool
    nil_label: str
    nil_description: str
    start_mention_token: str
    end_mention_token: str
    entity_token: str
    task_description: str


@dataclass
class ModelArguments:
    """Model arguments."""
    mention_encoder: str
    entity_encoder: str
    mention_context_length: int
    entity_context_length: int
    hidden_size: int
    freeze_mention_encoder: bool
    freeze_entity_encoder: bool
    measure: str
    temperature: float
    negative: str
    top_k: int
    cache_dir: Optional[str]
    prev_path: Optional[str]


def parse_args() -> tuple[DatasetArguments, ModelArguments, TrainingArguments]:
    parser = ArgumentParser()
    hfparser = HfArgumentParser(TrainingArguments)

    parser.add_argument(
        "--config_file", metavar="FILE", required=True
    )
    parser.add_argument(
        "--measure", type=str, default='cos'
    )
    parser.add_argument(
        "--negative", type=str, default='inbatch'
    )
    parser.add_argument(
        '--prev_path', metavar="DIR", default=None
    )

    args, extras = parser.parse_known_args()
    config = vars(load_config_as_namespace(args.config_file))
    training_args = hfparser.parse_args_into_dataclasses(extras)[0]

    data_config = config.pop("dataset")
    model_config = config.pop("model")

    arguments = DatasetArguments(**data_config)
    model_args = ModelArguments(**model_config)
    training_args = replace(training_args, **config)

    model_args.measure = args.measure if args.measure else model_args.measure
    model_args.negative = args.negative if args.negative else model_args.negative
    model_args.prev_path = args.prev_path if args.prev_path else model_args.prev_path

    return arguments, model_args, training_args
