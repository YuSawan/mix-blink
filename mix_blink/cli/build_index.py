import logging
import os
from pathlib import Path

import torch
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
)
from mix_blink.retriever import BM25Retriever, DenseRetriever
from mix_blink.training import setup_logger

logger = logging.getLogger(__name__)
TOKEN = os.environ.get("TOKEN", True)


def main(data_args: DatasetArguments, model_args: ModelArguments, training_args: TrainingArguments) -> None:
    setup_logger(training_args)
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
        mention_tokenizer = AutoTokenizer.from_pretrained(model_args.mention_encoder, model_max_length=model_args.mention_context_length, token=TOKEN)
        entity_tokenizer = AutoTokenizer.from_pretrained(model_args.entity_encoder, model_max_length=model_args.entity_context_length, token=TOKEN)
        mention_tokenizer.add_tokens(data_args.start_mention_token, data_args.end_mention_token)
        entity_tokenizer.add_tokens([data_args.entity_token, data_args.nil_label])

    cache_dir = model_args.cache_dir or get_temporary_cache_files_directory()
    dictionary = EntityDictionary(
        tokenizer=entity_tokenizer,
        dictionary_path=data_args.dictionary_file,
        entity_token=data_args.entity_token,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"name": data_args.nil_label, "description": data_args.nil_description} if data_args.add_nil else None
    )
    if model_args.negative == 'dense':
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

        dense_retriever = DenseRetriever(
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
        assert training_args.output_dir is not None
        dense_retriever.dump(model.entity_encoder, os.path.join(training_args.output_dir, 'retriever'))
    elif model_args.negative == 'bm25':
        bm25_retriever = BM25Retriever(
            dictionary,
            top_k=model_args.top_k,
            lang='en'
        )
        assert training_args.output_dir is not None
        bm25_retriever.build_index()
        bm25_retriever.dump(os.path.join(training_args.output_dir, 'retriever'))
    else:
        raise ValueError(f"Unknown negative sampling method: {model_args.negative}")


def cli_main() -> None:
    data_args, model_args, training_args = parse_args()
    main(data_args, model_args, training_args)


if __name__ == '__main__':
    cli_main()
