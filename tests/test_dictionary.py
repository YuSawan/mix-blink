from importlib.resources import files

import pytest
from transformers import AutoTokenizer

import tests.test_data as test_data
from mix_blink.data.dictionary import Entity, EntityDictionary

dictionary_path = str(files(test_data).joinpath("dictionary_toy.jsonl"))


@pytest.mark.parametrize("model_name", [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base"
])
def test_EntityDictionary(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens('[NIL]')

    dictionary = EntityDictionary(tokenizer, dictionary_path)
    assert len(dictionary) == 5
    assert isinstance(dictionary[0], Entity)
    assert dictionary[0].id == "000011"
    assert dictionary("000012").name == "Apple"
    label_id = dictionary("000012").label_id
    assert dictionary[label_id].name == "Apple"

    nil_description = "[NIL] is an entity that does not exist in the dataset."
    dictionary = EntityDictionary(tokenizer, dictionary_path, nil={"description": nil_description})
    assert len(dictionary) == 6
    assert dictionary[5].id == "-1"
    assert isinstance(dictionary[5], Entity)
    assert dictionary("-1").name == "[NIL]"
    label_id = dictionary("-1").label_id
    assert dictionary[label_id].name == "[NIL]"
    assert dictionary("-1").description == "[NIL]"+dictionary.entity_token+"[NIL] is an entity that does not exist in the dataset."

    for entity in dictionary:
        assert isinstance(entity, Entity)
        if "e5" in model_name:
            prefix = 'passages: ' if "instruct" not in model_name else ''
            assert entity.name == entity.description[len(prefix): len(prefix)+len(entity.name)]
        else:
            assert entity.name == entity.description[0: len(entity.name)]

    nil_description = ""
    dictionary = EntityDictionary(tokenizer, dictionary_path, nil={"description": nil_description})
    assert dictionary("-1").description == "[NIL]"+dictionary.entity_token+"[NIL] is an entity in this dictionary."
