import json
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
    CollatorForEntityLinking,
    EntityDictionary,
    Preprocessor,
    get_splits,
    read_dataset,
)
from mix_blink.evaluation.eval import evaluate, submit_wandb_eval
from mix_blink.prediction.predict import predict, submit_wandb_predict
from mix_blink.retriever import DenseRetriever
from mix_blink.training import EntityLinkingTrainer, LoggerCallback, setup_logger

logger = logging.getLogger(__name__)
TOKEN = os.environ.get("TOKEN", True)


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
    else:
        mention_tokenizer = AutoTokenizer.from_pretrained(model_args.mention_encoder, model_max_length=model_args.mention_context_length, token=TOKEN)
        entity_tokenizer = AutoTokenizer.from_pretrained(model_args.entity_encoder, model_max_length=model_args.entity_context_length, token=TOKEN)
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

    cache_dir = model_args.cache_dir or get_temporary_cache_files_directory()
    dictionary = EntityDictionary(
        tokenizer=entity_tokenizer,
        dictionary_path=data_args.dictionary_file,
        cache_dir=cache_dir,
        training_arguments=training_args,
        nil={"name": data_args.nil_label, "description": data_args.nil_description} if data_args.add_nil else None
    )

    raw_datasets = read_dataset(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        cache_dir
    )
    preprocessor = Preprocessor(mention_tokenizer, dictionary.entity_ids, remove_nil=False if data_args.add_nil else True)
    splits = get_splits(raw_datasets, preprocessor, training_args)

    if training_args.do_train:
        if model_args.negative == 'dense':
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
        elif model_args.negative == 'bm25':
            raise NotImplementedError
        else:
            assert model_args.negative == 'inbatch'
            retriever = None

        if retriever is not None:
            train_candidate_ids = retriever.get_hard_negatives(model, splits['train'])
            splits['train'] = splits['train'].add_column("negatives", train_candidate_ids)
            dev_candidate_ids = retriever.get_hard_negatives(model, splits['validation'], reset_index=False)
            splits['validation'] = splits['validation'].add_column("negatives", dev_candidate_ids)

        trainer = EntityLinkingTrainer(
            model = model,
            measure=model_args.measure,
            temperature=model_args.temperature,
            args=training_args,
            train_dataset = splits['train'],
            eval_dataset = splits['validation'],
            data_collator = CollatorForEntityLinking(mention_tokenizer, dictionary, negative_sample=model_args.negative)
        )
        trainer.add_callback(LoggerCallback(logger))

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.log_metrics("train", result.metrics)
        if training_args.save_strategy != "no":
            assert training_args.output_dir
            mention_tokenizer.save_pretrained(Path(training_args.output_dir, 'mention_tokenizer'))
            entity_tokenizer.save_pretrained(Path(training_args.output_dir, 'entity_tokenizer'))
            trainer.save_model()
            trainer.save_state()
            trainer.save_metrics("train", result.metrics)

    if training_args.do_eval:
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
        results = evaluate(model=model, dataset=splits['test'], retriever=retriever)
        if training_args.do_train:
            submit_wandb_eval(results)
        logger.info(f"R@1: {round(results['tp_1']/results['true'], 5)} ({results['tp_1']}/{results['true']})")
        logger.info(f"R@5: {round(results['tp_5']/results['true'], 5)} ({results['tp_5']}/{results['true']})")
        logger.info(f"R@10: {round(results['tp_10']/results['true'], 5)} ({results['tp_10']}/{results['true']})")
        logger.info(f"R@20: {round(results['tp_20']/results['true'], 5)} ({results['tp_20']}/{results['true']})")
        logger.info(f"R@50: {round(results['tp_50']/results['true'], 5)} ({results['tp_50']}/{results['true']})")
        logger.info(f"R@100: {round(results['tp_100']/results['true'], 5)} ({results['tp_100']}/{results['true']})")

    if training_args.do_predict:
        assert training_args.output_dir
        if not training_args.do_eval:
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
        assert isinstance(retriever, DenseRetriever)
        predicts = predict(
            model=model,
            dataset=splits['validation'].remove_columns('negatives') if model_args.negative != 'inbatch' else splits['validation'],
            retriever=retriever,
            reset_index=False if training_args.do_eval else True
        )
        if training_args.do_train:
            submit_wandb_predict(predicts)
        with open(Path(training_args.output_dir, "predicts.jsonl"), 'w') as f:
            for p in predicts:
                json.dump(p, f, ensure_ascii=False)
                f.write("\n")


def cli_main() -> None:
    data_args, model_args, training_args = parse_args()
    if data_args.validation_file is None:
        training_args.eval_strategy = "no"
    main(data_args, model_args, training_args)


if __name__ == '__main__':
    cli_main()
