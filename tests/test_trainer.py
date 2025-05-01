from importlib.resources import files

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    AutoTokenizer,
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
from mix_blink.retriever import DenseRetriever

model_name = 'google-bert/bert-base-uncased'
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
mention_tokenizer = AutoTokenizer.from_pretrained(model_name)
entity_tokenizer = AutoTokenizer.from_pretrained(model_name)
config = MixBlinkConfig(model_name, model_name)
model = MixBlink(config, encoder_from_pretrained=True)

training_args = TrainingArguments(output_dir=".tmp/")
dictionary = EntityDictionary(
    tokenizer=entity_tokenizer,
    dictionary_path=dictionary_path,
    nil={"name": "[NIL]"}
)
preprocessor = Preprocessor(
    mention_tokenizer,
    dictionary.entity_ids
)

@pytest.mark.parametrize('sampling', ['inbatch', 'dense'])
@pytest.mark.parametrize('measure', ['ip', 'cos', 'l2'])
def test_compute_loss(sampling: str, measure: str) -> None:
    raw_datasets = read_dataset(train_file=dataset_path)
    splits = get_splits(raw_datasets, preprocessor, training_args)
    if sampling == 'dense':
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure=measure,
            batch_size=2,
            top_k=1,
            vector_size=config.hidden_size,
            device=torch.device('cpu'),
            training_args=training_args
        )
        train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
        splits['train'] = splits['train'].add_column("negatives", train_candidate_ids)

    collator = CollatorForEntityLinking(mention_tokenizer, dictionary, negative_sample=sampling)
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

    for inputs in dataloader:
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        queries = outputs.query_hidden_state
        candidates = outputs.candidates_hidden_state
        hard_negatives = outputs.hard_negatives_hidden_state
        bs, hs = candidates.size(0), candidates.size(-1)
        candidates = candidates.unsqueeze(0).repeat(bs, 1, 1)

        assert isinstance(labels, Tensor)
        assert labels.size(0) == 2
        assert queries.size() == (2, config.hidden_size)
        assert candidates.size() == (2, 2, config.hidden_size)

        if hard_negatives is not None:
            hard_negatives = hard_negatives.reshape([bs, -1, hs])
            candidates = torch.concat([candidates, hard_negatives], dim=1)

        if sampling == 'inbatch':
            assert hard_negatives is None
        else:
            assert hard_negatives is not None
            assert candidates.size() == (2, 3, config.hidden_size)

        if measure == 'ip':
            scores = torch.bmm(queries.unsqueeze(1), candidates.transpose(1, -1)).squeeze(1)
        elif measure == 'cos':
            queries_norm = queries.unsqueeze(1) / torch.norm(queries.unsqueeze(1), dim=2, keepdim=True)
            candidates_norm = candidates / torch.norm(candidates, dim=2, keepdim=True)
            scores = torch.bmm(queries_norm, candidates_norm.transpose(1, -1)).squeeze(1)
        else:
            assert measure == 'l2'
            scores = torch.cdist(queries.unsqueeze(1), candidates).squeeze(1)

        loss = nn.functional.cross_entropy(scores, labels, reduction="mean")
        assert isinstance(loss, Tensor)
        assert isinstance(scores, Tensor)
