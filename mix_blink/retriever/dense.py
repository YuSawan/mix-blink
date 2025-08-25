# import json
import json
import os
from logging import INFO, getLogger

import faiss
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase, TrainingArguments

from ..data import Collator, EntityDictionary
from ..model import MixBlink
from ..modeling import Encoder

logger = getLogger(__name__)
logger.setLevel(INFO)


class DenseRetriever:
    def __init__(
            self,
            entity_tokenizer: PreTrainedTokenizerBase,
            mention_tokenizer: PreTrainedTokenizerBase,
            dictionary: EntityDictionary,
            measure: str,
            batch_size: int,
            vector_size: int,
            device: torch.device,
            training_args: TrainingArguments,
        ) -> None:
            self.entity_tokenizer = entity_tokenizer
            self.mention_tokenizer = mention_tokenizer
            self.dictionary = dictionary
            self.measure = measure
            self.batch_size = batch_size
            self.device = device
            self.vector_size = vector_size
            if self.measure not in ["cos", "ip", "l2"]:
                raise NotImplementedError(f"{measure} is not supported")
            if self.measure == 'l2':
                self.index = faiss.IndexFlatL2(vector_size)
            else:
                self.index = faiss.IndexFlatIP(vector_size)
            self.meta_ids_to_keys: dict[int, str] = {}
            self.args = training_args

    def reset_index(self) -> None:
        self.index.reset()
        self.meta_ids_to_keys = {}

    def get_dataloader(self, dataset: Dataset, tokenizer: PreTrainedTokenizerBase) -> DataLoader:
        collator = Collator(tokenizer)
        dataloader_params = {
            "batch_size": self.batch_size,
            "collate_fn": collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        dataloader_params["sampler"] = SequentialSampler(dataset)
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(dataset, **dataloader_params)

    @torch.no_grad()
    def build_index(self, model: Encoder) -> None:
        self.reset_index()
        dataloader = self.get_dataloader(self.dictionary.entity_dict, self.entity_tokenizer)
        self.meta_ids_to_keys = {i: idx for i, idx in enumerate(self.dictionary.entity_ids)}

        model.eval()
        model.to(self.device)
        pbar = tqdm(total=len(dataloader), desc='Load Entity Emb')
        for batch in dataloader:
            pbar.update()
            batch = batch.to(self.device)
            entity_embedding = model(**batch).to('cpu').detach().numpy().copy()
            if self.measure == 'cos':
                faiss.normalize_L2(entity_embedding)
            self.index.add(entity_embedding)
        pbar.close()

    def search_knn(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, list[list[str]]]:
        if top_k <= 0:
            raise RuntimeError("K is zero or under zero.")
        if top_k > len(self.dictionary):
            top_k = len(self.dictionary)
            logger.warning(f"K is over size of dictionary. K is modified to size of dictionary to {len(self.dictionary)}")
        if self.measure == "cos":
            faiss.normalize_L2(query_vectors)
        distances, indices = self.index.search(query_vectors, top_k)
        indices_keys = []
        for indice in indices:
            indices_keys.append([self.meta_ids_to_keys[ind] for ind in indice])

        return distances, indices_keys

    @torch.no_grad()
    def get_hard_negatives(self, model: MixBlink, dataset: Dataset, top_k: int) -> tuple[list[list[str]], list[list[str]]]:
        model.to(self.device)
        model.eval()

        dataloader = self.get_dataloader(dataset, self.mention_tokenizer)
        pbar = tqdm(total=(len(dataloader)), desc='Hard Negative Search')
        hard_negative_ids, candidate_ids = [], []
        for batch, labels in dataloader:
            pbar.update()
            batch = batch.to(self.device)
            outputs = model.mention_encoder(**batch).to('cpu').detach().numpy().copy()
            _, batch_indices = self.search_knn(outputs, top_k=top_k+max([len(label) for label in labels]))
            for idxs, indices in zip(labels, batch_indices):
                candidate_ids.append(indices[:top_k])
                for idx in idxs:
                    dic_id = self.dictionary[idx].id
                    if dic_id in indices:
                        indices.remove(dic_id)
                hard_negative_ids.append(indices[:top_k])
        pbar.close()
        return candidate_ids, hard_negative_ids

    def dump(self, model: Encoder, index_path: str, ensure_ascii: bool = False) -> None:
        self.build_index(model)
        self.serialize(index_path, ensure_ascii)

    def serialize(self, index_path: str, ensure_ascii: bool = False) -> None:
        logger.info("Serializing index to %s", index_path)
        if os.path.isdir(index_path):
            index_file = os.path.join(index_path, "index.dpr")
            meta_file = os.path.join(index_path, "meta.json")
        else:
            index_file = index_path + ".index.dpr"
            meta_file = index_path + ".meta.json"
        faiss.write_index(self.index, index_file)
        json.dump(self.meta_ids_to_keys, open(meta_file, 'w'), ensure_ascii=ensure_ascii)

    def deserialize_from(self, index_path: str) -> None:
        if os.path.isdir(index_path):
            index_file = os.path.join(index_path, "index.dpr")
            meta_file = os.path.join(index_path, "meta.json")
        else:
            index_file = index_path + ".index.dpr"
            meta_file = index_path + ".meta.json"
        logger.info("Loading index from %s", index_file)
        self.index = faiss.read_index(index_file)
        self.meta_ids_to_keys = {int(k): v for k, v in json.load(open(meta_file)).items()}
        logger.info(
            "Loaded index of type %s and size %d", type(self.index), self.index.ntotal
        )

    def __len__(self) -> int:
        return self.index.ntotal
