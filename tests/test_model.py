import pytest
from transformers import AutoTokenizer

from mix_blink import MixBlink, MixBlinkConfig
from mix_blink.modeling.encoder import BiEncoder, Encoder
from mix_blink.modeling.model_output import BiEncoderModelOutput


class TestMixBlink:
    @pytest.mark.parametrize("model_name", [
        "google-bert/bert-base-uncased",
        "FacebookAI/xlm-roberta-base",
        "microsoft/deberta-v3-base",
        "FacebookAI/roberta-base",
        "answerdotai/ModernBERT-base"
    ])
    def test__init__(self, model_name: str) -> None:
        mention_tokenizer = AutoTokenizer.from_pretrained(model_name)
        entity_tokenizer = AutoTokenizer.from_pretrained(model_name)
        entity_tokenizer.add_tokens(["[NIL]"])
        assert len(mention_tokenizer) + 1 == len(entity_tokenizer)
        config = MixBlinkConfig(
            model_name, model_name,
            entity_encoder_vocab_size=len(entity_tokenizer),
            mention_encoder_vocab_size=len(mention_tokenizer),
            freeze_entity_encoder=True,
            freeze_mention_encoder=False
        )
        mixblink = MixBlink(config, encoder_from_pretrained=False)
        assert isinstance(mixblink, MixBlink)
        assert isinstance(mixblink.model, BiEncoder)
        assert isinstance(mixblink.mention_encoder, Encoder)
        assert isinstance(mixblink.entity_encoder, Encoder)
        assert mixblink.config.mention_encoder_config and mixblink.config.mention_encoder_config
        assert mixblink.config.mention_encoder_vocab_size == len(mention_tokenizer)
        assert mixblink.config.entity_encoder_vocab_size == len(entity_tokenizer)
        assert mixblink.config.mention_encoder_vocab_size + 1 == mixblink.config.entity_encoder_vocab_size

        for param in mixblink.mention_encoder.parameters():
            assert param.requires_grad
        for param in mixblink.entity_encoder.parameters():
            assert not param.requires_grad


@pytest.mark.parametrize("model_name", [
    "google-bert/bert-base-uncased",
    "FacebookAI/xlm-roberta-base",
    "microsoft/deberta-v3-base",
    "FacebookAI/roberta-base",
    "answerdotai/ModernBERT-base"
])
def test_MixBlink(model_name: str) -> None:
    mention_tokenizer = AutoTokenizer.from_pretrained(model_name)
    entity_tokenizer = AutoTokenizer.from_pretrained(model_name)
    entity_tokenizer.add_tokens(["[NIL]"])
    config = MixBlinkConfig(
        model_name, model_name,
        entity_encoder_vocab_size=len(entity_tokenizer),
        mention_encoder_vocab_size=len(mention_tokenizer),
        freeze_entity_encoder=True,
        freeze_mention_encoder=False
    )
    mixblink = MixBlink(config, encoder_from_pretrained=False)

    mention_encodings = mention_tokenizer("Jobs found Apple.", return_tensors="pt")
    entity_encodings = entity_tokenizer(["Apple is a favorite company", "Microsoft is a favorite company"], return_tensors="pt")
    entity_encodings = { f"candidates_{k}": v for k, v in entity_encodings.items() }
    output = mixblink(**mention_encodings, **entity_encodings)
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert not output.hard_negatives_hidden_state

    hard_negative_encodings = entity_tokenizer("Meta is a favorite company", return_tensors="pt")
    hard_negative_encodings = { f"hard_negatives_{k}": v for k, v in hard_negative_encodings.items() }
    output = mixblink(**mention_encodings, **entity_encodings, **hard_negative_encodings)
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert output.hard_negatives_hidden_state is not None
    assert output.hard_negatives_hidden_state.size() == (1, config.hidden_size)

    mixblink.save_pretrained("test_model")
    del mixblink, config
    mixblink = MixBlink.from_pretrained("test_model")
    config = mixblink.config
    assert isinstance(config, MixBlinkConfig)
    assert isinstance(mixblink, MixBlink)
    assert isinstance(mixblink.model, BiEncoder)

    output = mixblink(**mention_encodings, **entity_encodings)
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert not output.hard_negatives_hidden_state

    output = mixblink(**mention_encodings, **entity_encodings, **hard_negative_encodings)
    assert isinstance(output, BiEncoderModelOutput)
    assert output.query_hidden_state.size() == (1, config.hidden_size)
    assert output.candidates_hidden_state.size() == (2, config.hidden_size)
    assert output.hard_negatives_hidden_state is not None
    assert output.hard_negatives_hidden_state.size() == (1, config.hidden_size)
