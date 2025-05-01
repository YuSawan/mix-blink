import os

import pytest
import torch
from transformers import AutoTokenizer, PretrainedConfig

from mix_blink import MixBlinkConfig
from mix_blink.modeling.encoder import BiEncoder, Encoder, Transformer
from mix_blink.modeling.model_output import BiEncoderModelOutput

TOKEN = os.environ.get('TOKEN', True)

@pytest.mark.parametrize("model_name", [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
])
@pytest.mark.parametrize("from_pretrained", [True, False])
def test_Transformer(model_name: str, from_pretrained: bool) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=TOKEN)
    config = MixBlinkConfig(model_name, "google-bert/bert-base-uncased")
    bert = Transformer(config.mention_encoder, config.mention_encoder_config, from_pretrained)
    assert bert.model.name_or_path == model_name
    assert isinstance(bert, Transformer)
    assert hasattr(bert, "config") and isinstance(bert.config, PretrainedConfig)

    encodings = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = bert(**encodings)
    assert hasattr(output, "last_hidden_state")


@pytest.mark.parametrize("model_name", [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base",
    "google-bert/bert-large-uncased"
])
def test_Encoder(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = MixBlinkConfig(model_name, "google-bert/bert-base-uncased")
    encoder = Encoder(config)
    assert isinstance(encoder, Encoder)
    assert hasattr(encoder, "config") and isinstance(encoder.config, PretrainedConfig)
    assert isinstance(encoder.bert_layer, Transformer)
    assert encoder.bert_layer.model.name_or_path == model_name
    encodings = tokenizer("Hello, my dog is cute", return_tensors="pt")
    output = encoder(**encodings)
    assert isinstance(output, torch.Tensor)
    assert output.size() == (1, config.hidden_size)
    if model_name == "google-bert/bert-large-uncased":
        assert hasattr(encoder, "projection")
    del encoder

    encoder = Encoder(config, entity_encoder=True)
    assert isinstance(encoder, Encoder)
    assert hasattr(encoder, "config")
    assert isinstance(encoder.bert_layer, Transformer)
    assert encoder.bert_layer.model.name_or_path == "google-bert/bert-base-uncased"


@pytest.mark.parametrize("model_name", [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base"
])
@pytest.mark.parametrize("from_pretrained", [True, False])
def test_BiEncoder(model_name: str, from_pretrained: bool) -> None:
    mention_tokenizer = AutoTokenizer.from_pretrained(model_name)
    entity_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    config = MixBlinkConfig(model_name, "google-bert/bert-base-uncased")
    biencoder = BiEncoder(config, from_pretrained)
    assert isinstance(biencoder, BiEncoder)
    assert isinstance(biencoder.mention_encoder, Encoder)
    assert biencoder.mention_encoder.bert_layer.model.name_or_path == model_name
    assert biencoder.config.mention_encoder_config
    assert isinstance(biencoder.entity_encoder, Encoder)
    assert biencoder.entity_encoder.bert_layer.model.name_or_path == "google-bert/bert-base-uncased"
    assert biencoder.config.entity_encoder_config

    mention_encodings = mention_tokenizer("Jobs found Apple.", return_tensors="pt")
    entity_encodings = entity_tokenizer(["Apple is a favorite company", "Microsoft is a favorite company"], return_tensors="pt")
    output = biencoder(
        input_ids=mention_encodings.get("input_ids"),
        attention_mask=mention_encodings.get("attention_mask"),
        token_type_ids=mention_encodings.get("token_type_ids"),
        candidates_input_ids=entity_encodings.get("input_ids"),
        candidates_attention_mask=entity_encodings["attention_mask"],
        candidates_token_type_ids=entity_encodings.get("token_type_ids")
    )
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert not output.hard_negatives_hidden_state

    hard_negative_encodings = entity_tokenizer("Meta is a favorite company", return_tensors="pt")
    output = biencoder(
        input_ids=mention_encodings.get("input_ids"),
        attention_mask=mention_encodings.get("attention_mask"),
        token_type_ids=mention_encodings.get("token_type_ids"),
        candidates_input_ids=entity_encodings.get("input_ids"),
        candidates_attention_mask=entity_encodings["attention_mask"],
        candidates_token_type_ids=entity_encodings.get("token_type_ids"),
        hard_negatives_input_ids=hard_negative_encodings.get("input_ids"),
        hard_negatives_attention_mask=hard_negative_encodings.get("attention_mask"),
        hard_negatives_token_type_ids=hard_negative_encodings.get("token_type_ids"),
    )
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert output.hard_negatives_hidden_state is not None
    assert output.hard_negatives_hidden_state.size() == (1, config.hidden_size)


@pytest.mark.parametrize("model_name", ["google-bert/bert-base-uncased"])
def test_resize_embedding(model_name: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = MixBlinkConfig(model_name, model_name)
    biencoder = BiEncoder(config)
    assert biencoder.config.mention_encoder_config
    assert biencoder.config.mention_encoder_config.vocab_size == len(tokenizer)
    assert biencoder.mention_encoder.config.vocab_size == len(tokenizer)
    assert biencoder.mention_encoder.bert_layer.config.vocab_size == len(tokenizer)

    tokenizer.add_tokens('[NIL]')
    biencoder.mention_encoder.resize_token_embeddings(len(tokenizer))
    assert biencoder.config.mention_encoder_config.vocab_size == len(tokenizer)
    assert biencoder.mention_encoder.config.vocab_size == len(tokenizer)
    assert biencoder.mention_encoder.bert_layer.config.vocab_size == len(tokenizer)


@pytest.mark.parametrize("model_name", ["google-bert/bert-base-uncased", "google-bert/bert-large-uncased"])
def test_freeze_parameters(model_name: str) -> None:
    config = MixBlinkConfig(model_name, model_name)
    biencoder = BiEncoder(config)
    for param in biencoder.parameters():
        assert param.requires_grad

    biencoder.entity_encoder.freeze_parameters()
    for param in biencoder.mention_encoder.parameters():
        assert param.requires_grad
    if model_name == "google-bert/bert-large-uncased":
        for param in biencoder.entity_encoder.bert_layer.parameters():
            assert not param.requires_grad
        for param in biencoder.entity_encoder.projection.parameters():
            assert param.requires_grad
    else:
        for param in biencoder.entity_encoder.parameters():
            assert not param.requires_grad
