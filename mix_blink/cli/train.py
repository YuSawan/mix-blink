import logging
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

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
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from mix_blink.training import EntityLinkingTrainer, LoggerCallback, setup_logger

logger = logging.getLogger(__name__)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"data_args: {data_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"training args: {training_args}")
    if data_args.dictionary_file is None:
        raise ValueError("Dictionary file is required.")

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
        entity_token=data_args.entity_token,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"name": data_args.nil_label, "description": data_args.nil_description} if data_args.add_nil else None
    )

    raw_datasets = read_dataset(
        data_args.train_file,
        data_args.validation_file,
        cache_dir=cache_dir
    )
    preprocessor = Preprocessor(
        mention_tokenizer,
        dictionary.entity_ids,
        negative=model_args.negative,
        start_mention_token=data_args.start_mention_token,
        end_mention_token=data_args.end_mention_token,
        remove_nil=False if data_args.add_nil else True,
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)

    trainer = EntityLinkingTrainer(
        model=model,
        measure=model_args.measure,
        temperature=model_args.temperature,
        args=training_args,
        train_dataset = splits['train'],
        eval_dataset = splits['validation'] if 'validation' in splits else None,
        data_collator = CollatorForEntityLinking(mention_tokenizer, dictionary)
    )
    trainer.add_callback(LoggerCallback(logger))

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", result.metrics)
    if training_args.save_strategy != "no":
        assert training_args.output_dir is not None
        mention_tokenizer.save_pretrained(Path(training_args.output_dir, 'mention_tokenizer'))
        entity_tokenizer.save_pretrained(Path(training_args.output_dir, 'entity_tokenizer'))
        trainer.save_model(training_args.output_dir)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)


def cli_main() -> None:
    parser = ArgumentParser()
    hfparser = HfArgumentParser(TrainingArguments)
    parser.add_argument(
        '--model_path', "-m", metavar="DIR", type=str, default=None,
    )
    parser.add_argument(
        "--config_file", "-c", metavar="FILE", required=True,
    )
    parser.add_argument(
        "--dictionary_file", "-d", type=str, required=True,
    )
    parser.add_argument(
        "--train_file", type=str, required=True,
    )
    parser.add_argument(
        "--validation_file", type=str, default=None,
    )
    parser.add_argument(
        "--measure", type=str, default="ip", choices=["ip", "cos", "l2"]
    )
    parser.add_argument(
        "--hard_negative", action='store_true', default=False
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
    data_args.train_file = args.train_file
    data_args.validation_file = args.validation_file

    if args.model_path:
        model_args.model_path = args.model_path
    model_args.measure = args.measure
    model_args.negative = args.hard_negative

    if data_args.validation_file is None:
        training_args.eval_strategy = "no"
    main(data_args, model_args, training_args)

if __name__ == '__main__':
    cli_main()
