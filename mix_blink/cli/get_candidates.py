import json
import logging
import os
from pathlib import Path

import torch
from datasets import Dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from mix_blink import MixBlink, MixBlinkConfig, parse_args
from mix_blink.argparser import DatasetArguments, ModelArguments
from mix_blink.data import (
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from mix_blink.retriever import BM25Retriever, DenseRetriever
from mix_blink.training import setup_logger

logger = logging.getLogger(__name__)

def add_candidate_ids(dataset: Dataset, candidate_ids: list[list[str]]) -> list[dict]:
    added_dataset = []
    for data in dataset:
        examples = {"id": data["id"], "examples": []}
        for example in data["examples"]:
            entities = example["entities"]
            example = {"id": example["id"], "text": example["text"], "entities": []}
            for entity in entities:
                candidates = candidate_ids.pop(0)
                example["entities"].append({
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": entity["label"],
                    "title": entity["title"],
                    "text": entity["text"],
                    "candidates": candidates
                })
            examples["examples"].append(example)
        added_dataset.append(examples)
    assert not candidate_ids
    return added_dataset


def write_jsonl(dataset: list[dict], output_path: str) -> None:
    """
    Write a dataset to a JSONL file.

    Args:
        dataset (list[dict]): The dataset to write.
        output_path (str): The path to the output JSONL file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
    logger.warning(
        f"process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"data_args: {data_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"training args: {training_args}")

    set_seed(training_args.seed)
    if model_args.prev_path:
        mention_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.prev_path, 'mention_tokenizer'))
        entity_tokenizer = AutoTokenizer.from_pretrained(Path(model_args.prev_path, 'entity_tokenizer'))
        model = MixBlink.from_pretrained(model_args.prev_path)
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

    cache_dir = model_args.cache_dir or get_temporary_cache_files_directory()
    raw_datasets = read_dataset(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        cache_dir
    )
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
    splits = get_splits(raw_datasets, preprocessor, training_args)
    assert training_args.output_dir is not None
    if model_args.negative == 'bm25':
        bm25_retriever = BM25Retriever(
            dictionary,
            top_k=model_args.top_k,
            lang='en'
        )
        bm25_retriever.deserialize_from(os.path.join(training_args.output_dir, 'retriever_bm25'))
        if 'train' in splits:
            candidate_ids = bm25_retriever.get_hard_negatives(splits['train'], n_threads=training_args.dataloader_num_workers)
            train_data = add_candidate_ids(raw_datasets['train'], candidate_ids)
            write_jsonl(train_data, data_args.train_file.replace('.jsonl', '_candidates_bm25.jsonl'))
        if 'validation' in splits:
            candidate_ids = bm25_retriever.get_hard_negatives(splits['validation'], n_threads=training_args.dataloader_num_workers)
            dev_data = add_candidate_ids(raw_datasets['validation'], candidate_ids)
            write_jsonl(dev_data, data_args.validation_file.replace('.jsonl', '_candidates_bm25.jsonl'))
        if 'test' in splits:
            candidate_ids = bm25_retriever.get_hard_negatives(splits['test'], n_threads=training_args.dataloader_num_workers)
            test_data = add_candidate_ids(raw_datasets['test'], candidate_ids)
            write_jsonl(test_data, data_args.test_file.replace('.jsonl', '_candidates_bm25.jsonl'))
    elif model_args.negative == 'dense':
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure=model_args.measure,
            batch_size=training_args.eval_batch_size*2,
            top_k=model_args.top_k,
            vector_size=config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        if 'train' in splits:
            train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
            train_data = add_candidate_ids(raw_datasets['train'], train_candidate_ids)
            write_jsonl(train_data, data_args.train_file.replace('.jsonl', '_candidates_dense.jsonl'))
        if 'validation' in splits:
            dev_candidate_ids = retriever.get_hard_negatives(model, splits['validation'])
            dev_data = add_candidate_ids(raw_datasets['validation'], dev_candidate_ids)
            write_jsonl(dev_data, data_args.validation_file.replace('.jsonl', '_candidates_dense.jsonl'))
        if 'test' in splits:
            test_candidate_ids = retriever.get_hard_negatives(model, splits['test'])
            test_data = add_candidate_ids(raw_datasets['test'], test_candidate_ids)
            write_jsonl(test_data, data_args.test_file.replace('.jsonl', '_candidates_dense.jsonl'))


def cli_main() -> None:
    data_args, model_args, training_args = parse_args()
    main(data_args, model_args, training_args)

if __name__ == '__main__':
    cli_main()
