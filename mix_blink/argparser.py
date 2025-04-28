from argparse import ArgumentParser
from dataclasses import dataclass, replace
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments

from .utils import load_config_as_namespace


@dataclass
class Arguments:
    """Arguments."""
    dictionary_file: str
    train_file : str
    validation_file : str
    test_file: str
    config_file: str
    # measure: {'cos': cosine, 'ip': inner-product, 'l2': eucridean}
    measure: str
    # negative: {'inbatch': in-batch sampling, 'dense': inbatch+hard negative with DPR, 'bm25': inbatch+hard negative with BM25}
    negative: str
    # top_k: Number of hard negative samples
    top_k: int
    # Pretrained Model Path
    # Use None if no pretrained model is being used
    prev_path: Optional[str] = None
    cache_dir: Optional[str] = None

@dataclass
class ModelArguments:
    """Model arguments."""
    mention_encoder: str
    entity_encoder: str
    hidden_size: int
    freeze_mention_encoder: bool
    freeze_entity_encoder: bool
    mention_context_length: int
    entity_context_length: int
    add_nil: bool
    nil_label: str
    nil_description: str


def parse_args() -> tuple[Arguments, ModelArguments, TrainingArguments]:
    parser = ArgumentParser()
    hfparser = HfArgumentParser(TrainingArguments)

    parser.add_argument(
        "--config_file", metavar="FILE", required=True
    )
    parser.add_argument(
        "--dictionary_file", metavar="FILE", required=True
    )
    parser.add_argument(
        "--train_file", metavar="FILE", default=None
    )
    parser.add_argument(
        "--validation_file", metavar="FILE", default=None
    )
    parser.add_argument(
        "--test_file", metavar="FILE", default=None
    )
    parser.add_argument(
        "--cache_dir", metavar="FILE", default='./.cache'
    )
    parser.add_argument(
        "--measure", type=str, default='cos'
    )
    parser.add_argument(
        "--negative", type=str, default='inbatch'
    )
    parser.add_argument(
        "--top_k", type=int, default=10
    )
    parser.add_argument(
        '--prev_path', metavar="DIR", default=None
    )

    args, extras = parser.parse_known_args()
    training_args = hfparser.parse_args_into_dataclasses(extras)[0]

    arguments = Arguments(**vars(args))
    config = vars(load_config_as_namespace(arguments.config_file))
    model_config = config.pop("luke_el")
    model_args = ModelArguments(**model_config)
    training_args = replace(training_args, **config)

    return arguments, model_args, training_args
