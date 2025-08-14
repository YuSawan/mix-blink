import logging
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import torch
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from mix_blink import MixBlink, MixBlinkConfig
from mix_blink.argparser import (
    DatasetArguments,
    ModelArguments,
    load_config_as_namespace,
)
from mix_blink.data import (
    EntityDictionary,
)
from mix_blink.retriever import DenseRetriever

logger = logging.getLogger(__name__)


def main(model_args: ModelArguments, data_args: DatasetArguments, training_args: TrainingArguments) -> None:
    if data_args.dictionary_file is None:
        raise ValueError("Dictionary file is required.")
    if training_args.output_dir is None:
        raise ValueError("Output directory is required.")

    set_seed(training_args.seed)
    if model_args.model_path:
        mention_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.model_path, 'mention_tokenizer'))
        entity_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.model_path, 'entity_tokenizer'))
        model = MixBlink.from_pretrained(model_args.model_path)
        config = model.config
    else:
        mention_tokenizer = AutoTokenizer.from_pretrained(model_args.mention_encoder, model_max_length=model_args.mention_context_length, token=True)
        entity_tokenizer = AutoTokenizer.from_pretrained(model_args.entity_encoder, model_max_length=model_args.entity_context_length, token=True)
        mention_tokenizer.add_tokens(data_args.start_mention_token, data_args.end_mention_token)
        entity_tokenizer.add_tokens([data_args.entity_token, data_args.nil_label])
        config = MixBlinkConfig(
            model_args.mention_encoder,
            model_args.entity_encoder,
            hidden_size=model_args.hidden_size,
            mention_encoder_vocab_size=len(mention_tokenizer),
            entity_encoder_vocab_size=len(entity_tokenizer),
            freeze_entity_encoder=model_args.freeze_entity_encoder,
            freeze_mention_encoder=model_args.freeze_mention_encoder,
        )
        model = MixBlink(config)

    cache_dir = data_args.cache_dir or get_temporary_cache_files_directory()
    dictionary = EntityDictionary(
        tokenizer=entity_tokenizer,
        dictionary_path=data_args.dictionary_file,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"name": data_args.nil_label, "description": data_args.nil_description} if data_args.add_nil else None
    )

    dense_retriever = DenseRetriever(
        entity_tokenizer,
        mention_tokenizer,
        dictionary,
        measure=model_args.measure,
        batch_size=training_args.eval_batch_size*2,
        vector_size=config.hidden_size,
        device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
        training_args=training_args,
    )
    dense_retriever.dump(model.entity_encoder, training_args.output_dir)


def cli_main() -> None:
    parser = ArgumentParser(description="Build Indexer with Dual Encoder")
    hfparser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        '--model_name_or_path', "-m", metavar="DIR", type=str, default=None,
    )
    parser.add_argument(
        "--config_file", "-c", metavar="FILE", required=True
    )
    parser.add_argument(
        "--dictionary_file", "-d", type=str, required=True,
    )
    parser.add_argument(
        "--measure", type=str, default="ip", choices=["ip", "cos", "l2"]
    )

    args, extras = parser.parse_known_args()
    config = vars(load_config_as_namespace(args.config_file))
    training_args = hfparser.parse_args_into_dataclasses(extras)[0]

    data_config = config.pop("dataset")
    model_config = config.pop("model")

    data_args = DatasetArguments(**data_config)
    model_args = ModelArguments(**model_config)
    training_args = replace(training_args, **config)

    data_args.dictionary_file = args.dictionary_file
    if args.model_name_or_path:
        model_args.model_path = args.model_name_or_path
    model_args.measure = args.measure
    main(model_args, data_args, training_args)

if __name__ == '__main__':
    cli_main()
