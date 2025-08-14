import json
import logging
import os
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path

import torch
import wandb
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
    Preprocessor,
    get_splits,
    read_dataset,
)
from mix_blink.evaluation.eval import evaluate, submit_wandb_eval
from mix_blink.prediction.predict import predict, submit_wandb_predict
from mix_blink.retriever import DenseRetriever
from mix_blink.training import setup_logger

logger = logging.getLogger(__name__)
TOKEN = os.environ.get("TOKEN", True)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"data_args: {data_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"training args: {training_args}")

    if data_args.test_file is None:
        raise ValueError("Test file is required.")
    if data_args.dictionary_file is None:
        raise ValueError("Dictionary file is required.")
    if not model_args.retriever_path:
        raise ValueError("Retriever path is not specified")
    if training_args.output_dir is None:
        raise ValueError("Output directory is required.")

    set_seed(training_args.seed)
    if model_args.model_path:
        mention_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.model_path, 'mention_tokenizer'))
        entity_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.model_path, 'entity_tokenizer'))
        model = MixBlink.from_pretrained(model_args.model_path)
        config = model.config
    else:
        mention_tokenizer = AutoTokenizer.from_pretrained(model_args.mention_encoder, model_max_length=model_args.mention_context_length, token=TOKEN)
        entity_tokenizer = AutoTokenizer.from_pretrained(model_args.entity_encoder, model_max_length=model_args.entity_context_length, token=TOKEN)
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

    preprocessor = Preprocessor(
        mention_tokenizer,
        dictionary.entity_ids,
        start_mention_token=data_args.start_mention_token,
        end_mention_token=data_args.end_mention_token,
        remove_nil=False if data_args.add_nil else True
    )
    raw_datasets = read_dataset(test_file=data_args.test_file, cache_dir=cache_dir)
    splits = get_splits(raw_datasets, preprocessor, training_args)

    retriever = DenseRetriever(
        entity_tokenizer,
        mention_tokenizer,
        dictionary,
        measure=model_args.measure,
        batch_size=training_args.eval_batch_size*2,
        vector_size=config.hidden_size,
        device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
        training_args=training_args
    )
    retriever.deserialize_from(model_args.retriever_path)

    results = evaluate(model=model, dataset=splits['test'], retriever=retriever)
    logger.info(f"R@1: {round(results['tp_1']/results['true'], 5)} ({results['tp_1']}/{results['true']})")
    logger.info(f"R@10: {round(results['tp_10']/results['true'], 5)} ({results['tp_10']}/{results['true']})")
    logger.info(f"R@50: {round(results['tp_50']/results['true'], 5)} ({results['tp_50']}/{results['true']})")
    logger.info(f"R@100: {round(results['tp_100']/results['true'], 5)} ({results['tp_100']}/{results['true']})")
    logger.info(f"MRR: {round(results['reciprocal_rank']/results['true'], 5)}")
    with open(os.path.join(training_args.output_dir, "eval_results.json"), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    _ = wandb.init(project=training_args.run_name)
    submit_wandb_eval(results)

    if training_args.do_predict:
        predicts = predict(
            model=model,
            dataset=splits['test'],
            retriever=retriever,
        )
        with open(os.path.join(training_args.output_dir, "predicts.jsonl"), 'w') as f:
            for p in predicts:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")
        submit_wandb_predict(predicts)


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
        "--dictionary_file", "-d", metavar="FILE", type=str, required=True,
    )
    parser.add_argument(
        "--input_file", metavar="FILE",  type=str, required=True,
    )
    parser.add_argument(
        "--retriever_path", "-r", type=str, required=True,
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

    if args.model_path:
        model_args.model_path = args.model_name_or_path
    data_args.test_file = args.input_file
    data_args.dictionary_file = args.dictionary_file
    model_args.retriever_path = args.retriever_path
    main(data_args, model_args, training_args)


if __name__ == '__main__':
    cli_main()
