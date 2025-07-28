from importlib.resources import files

import pytest
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    TrainingArguments,
)

import tests.test_data as test_data
from mix_blink import MixBlink, MixBlinkConfig
from mix_blink.data import (
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
training_args = TrainingArguments(output_dir=".tmp/")

@pytest.mark.parametrize("model_name", ["google-bert/bert-base-uncased"])
@pytest.mark.parametrize("sampling", ["inbatch", "dense", "bm25"])
def test_CollatorForEntityLinking(model_name: str, sampling: str) -> None:
    entity_tokenizer = AutoTokenizer.from_pretrained(model_name)
    entity_tokenizer.add_tokens('[NIL]')
    mention_tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = MixBlinkConfig(model_name, model_name)
    model = MixBlink(config)
    model.entity_encoder.resize_token_embeddings(len(entity_tokenizer))

    dictionary = EntityDictionary(
        tokenizer=entity_tokenizer,
        dictionary_path=dictionary_path,
        nil={'name': '[NIL]'}
    )
    raw_datasets = read_dataset(train_file=dataset_path)
    preprocessor = Preprocessor(
        mention_tokenizer,
        dictionary.entity_ids,
        negative=False if sampling == 'inbatch' else True,
    )
    splits = get_splits(raw_datasets, preprocessor, training_args)
    collator = CollatorForEntityLinking(mention_tokenizer, dictionary)
    dataloader_params = {
        "batch_size": 2,
        "collate_fn": collator,
        "num_workers": training_args.dataloader_num_workers,
        "pin_memory": training_args.dataloader_pin_memory,
        "persistent_workers": training_args.dataloader_persistent_workers,
    }
    dataloader_params["sampler"] = SequentialSampler(splits['train'])
    dataloader_params["drop_last"] = training_args.dataloader_drop_last
    dataloader_params["prefetch_factor"] = training_args.dataloader_prefetch_factor
    dataloader = DataLoader(splits['train'], **dataloader_params)

    for batch in dataloader:
        assert isinstance(batch, BatchEncoding)
        keys = list(batch.keys())
        assert "input_ids" in keys and isinstance(batch['input_ids'], torch.Tensor)
        assert "attention_mask" in keys and isinstance(batch['attention_mask'], torch.Tensor)
        assert "candidates_input_ids" in keys and isinstance(batch['candidates_input_ids'], torch.Tensor)
        assert "candidates_attention_mask" in keys and isinstance(batch['candidates_attention_mask'], torch.Tensor)
        assert "labels" in keys and isinstance(batch['labels'], torch.Tensor)
        assert batch['input_ids'].size(0) == batch['attention_mask'].size(0) == 2
        assert batch['candidates_input_ids'].size(0) == batch['candidates_attention_mask'].size(0) == 2
        assert batch['labels'].size(0) == 2

        hard_negatives_input_ids = batch.get("hard_negatives_input_ids", None)
        hard_negatives_attention_mask = batch.get("hard_negatives_attention_mask", None)
        if sampling != "inbatch":
            assert hard_negatives_input_ids is not None
            assert hard_negatives_attention_mask is not None
            assert isinstance(hard_negatives_input_ids, torch.Tensor)
            assert isinstance(hard_negatives_attention_mask, torch.Tensor)
            assert hard_negatives_input_ids.size(0) == hard_negatives_attention_mask.size(0) == 6
        else:
            assert hard_negatives_input_ids is None
            assert hard_negatives_attention_mask is None
