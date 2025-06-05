from collections.abc import Iterable
from typing import Any, Optional, TypedDict

from datasets import Dataset, DatasetDict, load_dataset
from datasets.fingerprint import get_temporary_cache_files_directory
from transformers import (
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding


class Mention(TypedDict):
    start: int
    end: int
    label: list[str]


class Example(TypedDict):
    id: str
    text: str
    entities: list[Mention]


def read_dataset(
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        test_file: Optional[str] = None,
        cache_dir: Optional[str] = None
        ) -> DatasetDict:
    """
    DatasetReader is for processing
    Input:
        train_file: dataset path for training
        validation_file: dataset path for validation
        test_file: dataset path for test
    Output: DatasetDict
    """
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
    cache_dir = cache_dir or get_temporary_cache_files_directory()
    raw_datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

    return raw_datasets


class Preprocessor:
    """
    Base processor for Prepare of models.
    The preprocessing is differ by Models such as BERT, LUKE
    """
    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            labels: list[str],
            start_mention_token: str = '[START_ENT]',
            end_mention_token: str = '[END_ENT]',
            remove_nil: bool = False
            ) -> None:
        self.tokenizer = tokenizer
        self.model_name = tokenizer.name_or_path
        self.labels = labels
        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for i, label in enumerate(labels)}
        self.start_mention_token = start_mention_token
        self.end_mention_token = end_mention_token
        self.remove_nil = remove_nil
        if not self.tokenizer.sep_token:
            raise ValueError(f'This token does not includes sep token in special tokens: {self.tokenizer.special_tokens_map}')

    def __call__(self, document: list[dict[str, Any]]) -> Iterable[BatchEncoding]:
        """
        Input: list[dict[str, Any]]
        Output: Iterable[BatchEncoding]
        """
        for example in document:
            for ent in example["entities"]:
                text = example["text"][:ent["start"]] + self.start_mention_token + example["text"][ent["start"]:ent["end"]] + self.end_mention_token + example["text"][ent["end"]:]
                encodings  = self.tokenizer(
                    text,
                    padding=True,
                    truncation=True
                )
                encodings["text"] = example["text"]
                encodings["entity_span"] = (ent["start"], ent["end"])
                encodings["id"] = example['id']
                encodings["label"] = []
                for label in ent["label"]:
                    if label in self.label2id:
                        encodings["label"].append(self.label2id[label])
                    else:
                        if self.remove_nil:
                            continue
                        else:
                            raise KeyError(f"Label {label} not found in label2id mapping.")
                if encodings["label"]:
                    yield encodings


def get_splits(
        raw_datasets: DatasetDict,
        preprocessor: Preprocessor,
        training_arguments: Optional[TrainingArguments]=None
        ) -> dict[str, Dataset]:
    def preprocess(documents: Dataset) -> Any:
        features: list[BatchEncoding] = []
        for document in documents["examples"]:
            features.extend(preprocessor(document))
        outputs = {}
        for k in list(features[0].keys()):
            outputs[k] = [f[k] for f in features]
        return outputs

    if training_arguments:
        with training_arguments.main_process_first(desc="dataset map pre-processing"):
            column_names = next(iter(raw_datasets.values())).column_names
            splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)
    else:
        column_names = next(iter(raw_datasets.values())).column_names
        splits = raw_datasets.map(preprocess, batched=True, remove_columns=column_names)

    return splits


