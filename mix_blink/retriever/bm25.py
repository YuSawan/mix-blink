import os
from logging import INFO, getLogger
from typing import Literal, Optional, Union

import bm25s
import numpy as np
from bm25s.hf import BM25HF
from bm25s.tokenization import Tokenized
from datasets import Dataset

from ..data import EntityDictionary
from .stopwords import JAPANESE_STOP_WORDS

logger = getLogger(__name__)
logger.setLevel(INFO)


def whitespace_tokenize(texts: Union[str, list[str]]) -> Tokenized:
    def _tokenize(text: list[str]) -> list[str]:
        return [token for token in text]

    corpus_token = bm25s.tokenize(texts, token_pattern=r"(?u)\b[\w#]+\b", stopwords="en", stemmer=_tokenize)
    return corpus_token


def sudachi_tokenize(
        texts: Union[str, list[str]],
        mode: Literal["A", "B", "C"] = "C",
        sudachi_dict: Literal["small", "core", "full"] = "core",
        stopwords: list[str] = list(JAPANESE_STOP_WORDS),
        pos_filter: list[str] = ["名詞","動詞","形容詞"],
    ) -> Tokenized:
    from sudachipy import dictionary, tokenizer

    if isinstance(texts, str):
        texts = [texts]

    if mode not in ["A", "B", "C"]:
        raise ValueError(f"Mode must be one of 'A', 'B', or 'C', but got {mode}")
    mode = getattr(tokenizer.Tokenizer.SplitMode, mode)

    if sudachi_dict not in ["small", "core", "full"]:
        raise ValueError(f"SUDACHI dictionary must be one of 'small', 'core', or 'full', but got {sudachi_dict}")
    tokenizer_obj = dictionary.Dictionary(dict=sudachi_dict).create()

    corpus_ids = []
    token_to_idx: dict[str, int] = {}
    for text  in texts:
        morphs = tokenizer_obj.tokenize(text, mode)
        tokens = []
        for m in morphs:
            if pos_filter is None or any(m.part_of_speech()[0].startswith(pos) for pos in pos_filter):
                token = m.normalized_form()
                if token not in stopwords:
                    tokens.append(token)
        doc_ids = []
        for token in tokens:
            if token not in token_to_idx:
                token_to_idx[token] = len(token_to_idx)
            token_id = token_to_idx[token]
            doc_ids.append(token_id)
        corpus_ids.append(doc_ids)

    return Tokenized(ids=corpus_ids, vocab=token_to_idx)


class BM25Retriever:
    def __init__(
            self,
            dictionary: EntityDictionary,
            top_k: int,
            lang: str = 'en'
        ) -> None:
            self.dictionary = dictionary
            self.top_k = top_k
            if self.top_k <= 0:
                raise RuntimeError("K is zero or under zero.")
            if self.top_k >= len(self.dictionary):
                raise RuntimeError("K is same or over the size of dictionary")
            self.meta_ids_to_keys: dict[int, str] = {}
            if lang == 'en':
                self.tokenize_func = whitespace_tokenize
            elif lang == 'ja':
                self.tokenize_func = sudachi_tokenize
            else:
                raise ValueError(f"Language {lang} is not supported. Supported languages are 'en' and 'ja'.")

    def build_index(self) -> None:
        self.index = BM25HF()
        self.meta_ids_to_keys = {i: idx for i, idx in enumerate(self.dictionary.entity_ids)}
        self.corpus_tokens = self.tokenize_func(self.dictionary.entity_dict["description"])
        self.index.index(self.corpus_tokens)

    def search_knn(self, query: str|list[str], top_k: Optional[int] = None, n_threads: int = -1) -> tuple[np.ndarray, np.ndarray]:
        K = self.top_k if top_k is None else top_k
        if K <= 0:
            raise RuntimeError("K is zero or under zero.")
        if K > len(self.dictionary):
            K = len(self.dictionary)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.dictionary)}")

        query_tokens = self.tokenize_func(query)
        results, scores = self.index.retrieve(query_tokens, k=K, n_threads=n_threads)

        return scores, results

    def get_hard_negatives(self, dataset: Dataset, n_threads: int = -1) -> list[list[str]]:
        # queries = [text[span[0]: span[1]] for text, span in zip(dataset['text'], dataset['entity_span'])]
        queries = dataset['text']
        print(queries[0])
        print(len(queries))
        _, results = self.search_knn(queries, top_k=self.top_k+1, n_threads=n_threads)
        labels = dataset['labels']

        candidate_ids = []
        for i in range(results.shape[0]):
            indices = [self.dictionary[j].id for j in results[i].tolist()]
            if self.dictionary[labels[i][0]].id in indices:
                indices.remove(self.dictionary[labels[i][0]].id)
            else:
                indices = indices[:self.top_k]
            candidate_ids.append(indices)
        return candidate_ids

    def dump(self, index_path: str) -> None:
        self.serialize(index_path)

    def serialize(self, output_dir: str) -> None:
        index_dir = os.path.join(output_dir, 'retriever_bm25')
        logger.info("Serializing index to %s", index_dir)
        self.index.save(index_dir)

    def deserialize_from(self, index_dir: str) -> None:
        logger.info("Deserializing index from %s", index_dir)
        self.meta_ids_to_keys = {i: idx for i, idx in enumerate(self.dictionary.entity_ids)}
        self.index = BM25HF.load(index_dir, load_corpus=True)
