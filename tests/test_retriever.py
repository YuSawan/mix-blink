from importlib.resources import files

import numpy as np
import pytest
import torch
from faiss import IndexFlatIP, IndexFlatL2
from transformers import AutoTokenizer, TrainingArguments

import tests.test_data as test_data
from mix_blink import MixBlink, MixBlinkConfig
from mix_blink.data import (
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from mix_blink.retriever import DenseRetriever

model_name = 'google-bert/bert-base-uncased'
test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))

mention_tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
entity_tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
config = MixBlinkConfig(model_name, model_name)
model = MixBlink(config, encoder_from_pretrained=False)
training_args = TrainingArguments(output_dir=".tmp/")

dictionary = EntityDictionary(
    tokenizer=entity_tokenizer,
    dictionary_path=dictionary_path,
    training_arguments=training_args,
)

raw_datasets = read_dataset(test_file=test_dataset_path)
preprocessor = Preprocessor(mention_tokenizer, dictionary.entity_ids, remove_nil=True)
dataset = get_splits(raw_datasets, preprocessor, training_args)['test']


class TestDenseRetriever:
    @pytest.mark.parametrize("measure", ["M", "cos", "ip", "l2"])
    def test___init__(self, measure: str) -> None:
        if measure == "M":
            with pytest.raises(NotImplementedError) as nie:
                retriever = DenseRetriever(
                    entity_tokenizer,
                    mention_tokenizer,
                    dictionary,
                    measure=measure,
                    batch_size=training_args.eval_batch_size*2,
                    vector_size=model.config.hidden_size,
                    device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                    training_args=training_args
                )
            assert isinstance(nie.value, NotImplementedError)
            assert str(nie.value) == f"{measure} is not supported"
        else:
            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure=measure,
                batch_size=training_args.eval_batch_size*2,
                vector_size=model.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
            if measure == "l2":
                assert isinstance(retriever.index, IndexFlatL2)
            else:
                assert isinstance(retriever.index, IndexFlatIP)

    @pytest.mark.parametrize("measure",["ip", "cos", "l2"])
    def test_build_index(self, measure: str) -> None:
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure=measure,
            batch_size=training_args.eval_batch_size*2,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)
        assert len(retriever) == len(dictionary)
        assert len(list(retriever.meta_ids_to_keys.keys())) == len(dictionary)

        retriever.build_index(model.entity_encoder)
        assert len(retriever) == len(dictionary)
        assert len(list(retriever.meta_ids_to_keys.keys())) == len(dictionary)

    def test_reset_index(self) -> None:
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure='cos',
            batch_size=training_args.eval_batch_size*2,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)
        retriever.reset_index()
        assert len(retriever) == 0

    @pytest.mark.parametrize("top_k", [0, 2, 5, 10])
    def test_search_knn(self, top_k: int) -> None:
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure='cos',
            batch_size=training_args.eval_batch_size*2,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)

        query_vector = torch.randn(2, model.config.hidden_size).detach().numpy().copy()
        if top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                retriever.search_knn(query_vector, top_k)
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        else:
            distances, indices = retriever.search_knn(query_vector, top_k)
            assert isinstance(distances, np.ndarray) and isinstance(indices, list) and isinstance(indices[0], list)
            if top_k > len(dictionary):
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == len(dictionary)
            else:
                assert distances.shape[0] == len(indices) == 2
                assert distances.shape[1] == len(indices[0]) == top_k

            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure='l2',
                batch_size=training_args.eval_batch_size*2,
                vector_size=model.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
            retriever.build_index(model.entity_encoder)
            l2_distances, l2_indices = retriever.search_knn(query_vector, top_k)
            assert distances.shape == l2_distances.shape and len(indices) == len(l2_indices) and len(indices[0]) == len(l2_indices[0])
            assert not np.all(l2_distances == distances)

            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure='ip',
                batch_size=training_args.eval_batch_size*2,
                vector_size=model.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
            retriever.build_index(model.entity_encoder)
            ip_distances, ip_indices = retriever.search_knn(query_vector, top_k)
            assert distances.shape == ip_distances.shape and len(indices) == len(ip_indices) and len(indices[0]) == len(ip_indices[0])
            assert not np.all(ip_distances == distances)
            assert not np.all(ip_distances == l2_distances)

    @pytest.mark.parametrize("top_k", [2, 4])
    def test_get_hard_negatives(self, top_k: int) -> None:
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure='cos',
            batch_size=training_args.eval_batch_size*2,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)
        candidate_ids, hard_negative_ids = retriever.get_hard_negatives(model, dataset, top_k)
        assert isinstance(candidate_ids, list) and isinstance(hard_negative_ids, list)
        assert len(dataset) == len(candidate_ids) == len(hard_negative_ids)
        for data, ids in zip(dataset, hard_negative_ids):
            assert isinstance(ids, list) and len(ids) == top_k
            labels = data.copy().pop('labels')
            for label in labels:
                target = dictionary[label].id
                assert target not in ids
