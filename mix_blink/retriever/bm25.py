import os
from logging import INFO, getLogger
from typing import Optional

import bm25s
import numpy as np
from bm25s.hf import BM25HF
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from ..data import EntityDictionary
from .stopwords import ENGLISH_STOP_WORDS

logger = getLogger(__name__)
logger.setLevel(INFO)



class BM25Retriever:
    def __init__(
            self,
            mention_tokenizer: PreTrainedTokenizerBase,
            entity_tokenizer: PreTrainedTokenizerBase,
            dictionary: EntityDictionary,
            top_k: int,
            lang: str = 'en'
        ) -> None:
            self.mention_tokenizer = mention_tokenizer
            self.entity_tokenizer = entity_tokenizer
            self.dictionary = dictionary
            self.top_k = top_k
            if self.top_k <= 0:
                raise RuntimeError("K is zero or under zero.")
            if self.top_k >= len(self.dictionary):
                raise RuntimeError("K is same or over the size of dictionary")
            self.meta_ids_to_keys: dict[int, str] = {}
            self.lang = lang
            self.stopwords = ENGLISH_STOP_WORDS
            self.build_index()

    @staticmethod
    def bm25_tokenize(text: list[str]) -> list[str]:
        # return [token.strip(string.punctuation) for token in text]
        return [token for token in text]

    def build_index(self) -> None:
        self.index = BM25HF()
        self.meta_ids_to_keys = {i: idx for i, idx in enumerate(self.dictionary.entity_ids)}
        self.corpus_tokens = bm25s.tokenize(self.dictionary.entity_dict['description'], token_pattern=r"(?u)\b[\w#]+\b", stemmer=self.bm25_tokenize)
        self.index.index(self.corpus_tokens)

    def search_knn(self, query: str|list[str], top_k: Optional[int] = None, n_threads: int = -1) -> tuple[np.ndarray, np.ndarray]:
        K = self.top_k if top_k is None else top_k
        if K <= 0:
            raise RuntimeError("K is zero or under zero.")
        if K > len(self.dictionary):
            K = len(self.dictionary)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.dictionary)}")

        query_tokens = bm25s.tokenize(query, token_pattern=r"(?u)\b[\w#]+\b", stemmer=self.bm25_tokenize)
        results, scores = self.index.retrieve(query_tokens, k=K, n_threads=n_threads)

        return scores, results

    def get_hard_negatives(self, dataset: Dataset, n_threads: int = -1) -> list[list[str]]:
        # queries = [text[span[0]: span[1]] for text, span in zip(dataset['text'], dataset['entity_span'])]
        queries = dataset['text']
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

    def serialize(self, index_path: str) -> None:
        logger.info("Serializing index to %s", index_path)
        if os.path.isdir(index_path):
            self.index.save(index_path)
        else:
            raise RuntimeError('Please specify the directory path')

    def deserialize_from(self, index_path: str) -> None:
        if os.path.isdir(index_path):
            self.index.load(index_path)
        else:
            raise RuntimeError('Please specify the directory path')
