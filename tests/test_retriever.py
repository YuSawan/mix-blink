import os
from importlib.resources import files

import bm25s
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
from mix_blink.retriever import BM25Retriever, DenseRetriever
from mix_blink.retriever.bm25 import sudachi_tokenize, whitespace_tokenize

TOKEN = os.environ.get("TOKEN", True)

model_name = 'google-bert/bert-base-uncased'
test_dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))
dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))

mention_tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
entity_tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
config = MixBlinkConfig(model_name, model_name)
model = MixBlink(config, encoder_from_pretrained=False)
training_args = TrainingArguments(output_dir=".tmp/")

dictionary = EntityDictionary(
    tokenizer=entity_tokenizer,
    dictionary_path=dictionary_path,
    training_arguments=training_args
)

raw_datasets = read_dataset(test_file=test_dataset_path)
preprocessor = Preprocessor(mention_tokenizer, dictionary.entity_ids, remove_nil=True)
dataset = get_splits(raw_datasets, preprocessor, training_args)['test']


class TestDenseRetriever:
    @pytest.mark.parametrize("measure", ["M", "cos", "ip", "l2"])
    @pytest.mark.parametrize("top_k",[-1, 0, 2, 6])
    def test___init__(self, measure: str, top_k: int) -> None:
        if measure == "M":
            with pytest.raises(NotImplementedError) as nie:
                retriever = DenseRetriever(
                    entity_tokenizer,
                    mention_tokenizer,
                    dictionary,
                    measure=measure,
                    batch_size=training_args.eval_batch_size*2,
                    top_k=top_k,
                    vector_size=model.config.hidden_size,
                    device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                    training_args=training_args
                )
            assert isinstance(nie.value, NotImplementedError)
            assert str(nie.value) == f"{measure} is not supported"
        elif top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                retriever = DenseRetriever(
                    entity_tokenizer,
                    mention_tokenizer,
                    dictionary,
                    measure=measure,
                    batch_size=training_args.eval_batch_size*2,
                    top_k=top_k,
                    vector_size=model.config.hidden_size,
                    device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                    training_args=training_args
                )
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        elif top_k == 6:
            with pytest.raises(RuntimeError) as re:
                retriever = DenseRetriever(
                    entity_tokenizer,
                    mention_tokenizer,
                    dictionary,
                    measure=measure,
                    batch_size=training_args.eval_batch_size*2,
                    top_k=top_k,
                    vector_size=model.config.hidden_size,
                    device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                    training_args=training_args
                )
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is same or over the size of dictionary"
        else:
            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure=measure,
                batch_size=training_args.eval_batch_size*2,
                top_k=top_k,
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
            top_k=1,
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
            top_k=1,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)
        retriever.reset_index()
        assert len(retriever) == 0

    @pytest.mark.parametrize("top_k", [None, 0, 2, 5, 10])
    def test_search_knn(self, top_k: int|None) -> None:
        retriever = DenseRetriever(
            entity_tokenizer,
            mention_tokenizer,
            dictionary,
            measure='cos',
            batch_size=training_args.eval_batch_size*2,
            top_k=1,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        retriever.build_index(model.entity_encoder)

        query_vector = torch.randn(2, model.config.hidden_size).detach().numpy().copy()
        if not top_k:
            distances, indices = retriever.search_knn(query_vector)
            assert isinstance(distances, np.ndarray) and isinstance(indices, list) and isinstance(indices[0], list)
            assert distances.shape[0] == len(indices) == 2
            assert distances.shape[1] == len(indices[0]) == retriever.top_k
        else:
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

        if not top_k or top_k <= 0:
            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure='l2',
                batch_size=training_args.eval_batch_size*2,
                top_k=1,
                vector_size=model.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
            retriever.build_index(model.entity_encoder)
            if not top_k:
                l2_distances, l2_indices = retriever.search_knn(query_vector)
            else:
                l2_distances, l2_indices = retriever.search_knn(query_vector, top_k)
            assert distances.shape == l2_distances.shape and len(indices) == len(l2_indices) and len(indices[0]) == len(l2_indices[0])
            assert not np.all(l2_distances == distances)

        if not top_k or top_k <= 0:
            retriever = DenseRetriever(
                entity_tokenizer,
                mention_tokenizer,
                dictionary,
                measure='ip',
                batch_size=training_args.eval_batch_size*2,
                top_k=1,
                vector_size=model.config.hidden_size,
                device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
                training_args=training_args
            )
            retriever.build_index(model.entity_encoder)
            if not top_k:
                ip_distances, ip_indices = retriever.search_knn(query_vector)
            else:
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
            top_k=top_k,
            vector_size=model.config.hidden_size,
            device=torch.device(training_args.device) if torch.cuda.is_available() else torch.device('cpu'),
            training_args=training_args
        )
        candidate_ids = retriever.get_hard_negatives(model, dataset)
        assert isinstance(candidate_ids, list)
        assert len(dataset) == len(candidate_ids)
        for data, ids in zip(dataset, candidate_ids):
            assert isinstance(ids, list) and len(ids) == top_k
            labels = data.copy().pop('labels')
            target = dictionary[labels[0]].id
            assert target not in ids


class TestBM25Retriever:
    @pytest.mark.parametrize("lang", ["en", "ja"])
    @pytest.mark.parametrize("top_k",[-1, 0, 2, 6])
    def test___init__(self, lang: str, top_k: int) -> None:
        if top_k <= 0:
            with pytest.raises(RuntimeError) as re:
                BM25Retriever(
                    dictionary=dictionary,
                    top_k=top_k,
                    lang=lang
                )
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is zero or under zero."
        elif top_k == 6:
            with pytest.raises(RuntimeError) as re:
                BM25Retriever(
                    dictionary=dictionary,
                    top_k=top_k,
                    lang=lang
                )
            assert isinstance(re.value, RuntimeError)
            assert str(re.value) == "K is same or over the size of dictionary"
        else:
            retriever = BM25Retriever(
                    dictionary=dictionary,
                    top_k=top_k,
                    lang=lang
            )
            if lang == 'en':
                assert retriever.tokenize_func is whitespace_tokenize
            if lang == 'ja':
                assert retriever.tokenize_func is sudachi_tokenize
            assert retriever.top_k == top_k

    @pytest.mark.parametrize("top_k",[2, 3])
    def test_build_index(self, top_k: int) -> None:
        retriever = BM25Retriever(
                dictionary=dictionary,
                top_k=top_k
        )
        retriever.build_index()
        assert isinstance(retriever.index, bm25s.BM25)
        assert len(list(retriever.meta_ids_to_keys.keys())) == 5

    @pytest.mark.parametrize("query", ["amazon is established by"])
    @pytest.mark.parametrize("top_k", [None, 0, 1, 2, 5, 10])
    def test_search_knn(self, query: str, top_k: int) -> None:
        retriever = BM25Retriever(
            dictionary=dictionary,
            top_k=2
        )
        retriever.build_index()
        if top_k is None:
            scores, indices = retriever.search_knn(query)
            assert isinstance(scores, np.ndarray) and isinstance(indices, np.ndarray)
            assert scores.shape[0] == indices.shape[0] == 1
            assert scores.shape[1] == indices.shape[1] == retriever.top_k
        else:
            if top_k <= 0:
                with pytest.raises(RuntimeError) as re:
                    retriever.search_knn(query, top_k)
                assert isinstance(re.value, RuntimeError)
                assert str(re.value) == "K is zero or under zero."
            else:
                scores, indices = retriever.search_knn(query, top_k)
                assert isinstance(scores, np.ndarray) and isinstance(indices, np.ndarray)
                if top_k > len(dictionary):
                    assert scores.shape[0] == indices.shape[0] == 1
                    assert scores.shape[1] == indices.shape[1] == len(dictionary)
                else:
                    assert scores.shape[0] == indices.shape[0] == 1
                    assert scores.shape[1] == indices.shape[1] == top_k
                results = [retriever.dictionary[j].id for j in indices[0].tolist()]
                assert "000014" in results
                assert results[0] == "000014"


    @pytest.mark.parametrize("top_k", [1, 2, 4])
    @pytest.mark.parametrize("lang", ["en", "ja"])
    def test_get_hard_negatives(self, top_k: int, lang: str) -> None:
        retriever = BM25Retriever(
            dictionary=dictionary,
            top_k=top_k,
            lang=lang
        )
        retriever.build_index()
        candidate_ids = retriever.get_hard_negatives(dataset)
        assert isinstance(candidate_ids, list)
        assert len(dataset) == len(candidate_ids)
        for data, cids in zip(dataset, candidate_ids):
            labels = data.copy().pop('labels')
            target = dictionary[labels[0]].id
            assert isinstance(cids, list)
            assert target not in cids
            assert top_k == len(cids)
