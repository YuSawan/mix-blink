from importlib.resources import files

import pytest
from transformers import (
    AutoTokenizer,
    BatchEncoding,
)

import tests.test_data as test_data
from mix_blink.data import (
    EntityDictionary,
    Preprocessor,
    read_dataset,
)

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))
dataset_path = str(files(test_data).joinpath("dataset_toy.jsonl"))


class TestPreprocessor:
    @pytest.mark.parametrize("model_name", ["google-bert/bert-base-uncased"])
    def test___init__(self, model_name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(['[START_ENT]', '[END_ENT]'])
        dictionary = EntityDictionary(tokenizer, dictionary_path, nil={"name": "[NIL]"})

        preprocessor = Preprocessor(
            tokenizer,
            dictionary.entity_ids,
        )
        assert preprocessor.tokenizer == tokenizer
        assert preprocessor.labels == dictionary.entity_ids
        assert len(preprocessor.label2id.keys()) == 6
        assert len(preprocessor.id2label.keys()) == 6


    @pytest.mark.parametrize("model_name", ["google-bert/bert-base-uncased"])
    @pytest.mark.parametrize("remove_nil", [True, False])
    @pytest.mark.parametrize("negative", [True, False])
    def test___call__(self, model_name: str, negative: bool, remove_nil: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(['[START_ENT]', '[END_ENT]'])

        dictionary = EntityDictionary(tokenizer, dictionary_path, nil = None if remove_nil else {"id": "-1"})
        preprocessor = Preprocessor(
            tokenizer,
            dictionary.entity_ids,
            negative=negative,
            remove_nil=remove_nil
        )

        raw_dataset = read_dataset(test_file=dataset_path)
        for document in raw_dataset['test']["examples"]:
            for example in document:
                for ent in example['entities']:
                    encodings = tokenizer(example["text"], truncation=True)
                    assert isinstance(encodings, BatchEncoding)

        features: list[BatchEncoding] = []
        for document in raw_dataset['test']["examples"]:
            features.extend(preprocessor(document))

        assert isinstance(features, list)
        if remove_nil:
            assert len(features) == 6
        else:
            assert len(features) == 8
        assert isinstance(features[0], BatchEncoding)

        outputs: dict = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]

        assert "input_ids" in outputs.keys()
        assert "labels" in outputs.keys()
        assert "id" in outputs.keys()
        assert "entity_span" in outputs.keys()
        assert "candidates" in outputs.keys()
        assert "text" in outputs.keys()
