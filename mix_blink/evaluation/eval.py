
import torch
import wandb
from datasets import Dataset
from tqdm.auto import tqdm

from ..model import MixBlink
from ..retriever import DenseRetriever


@torch.no_grad()
def evaluate(model: MixBlink, dataset: Dataset, retriever: DenseRetriever) -> dict[str, int|float]:
    dataloader = retriever.get_dataloader(dataset, retriever.mention_tokenizer)
    dictionary = retriever.dictionary
    model.to(retriever.device)

    true, tp_1, tp_10, tp_50, tp_100, reciprocal_rank = 0, 0, 0, 0, 0, 0.
    pbar = tqdm(total=len(dataloader), desc="Eval")
    for batch, labels in dataloader:
        pbar.update()
        batch = batch.to(retriever.device)
        query = model.mention_encoder(**batch).to('cpu').detach().numpy().copy()
        _, batch_indices = retriever.search_knn(query, top_k=100)
        for idxs, indices in zip(labels, batch_indices):
            true += 1
            best_rank = 0
            for idx in idxs:
                dic_id = dictionary[idx].id
                if dic_id in indices:
                    rank = indices.index(dic_id) + 1
                    if rank < best_rank or best_rank == 0:
                        best_rank = rank
            if best_rank > 0:
                if best_rank == 1:
                    tp_1 += 1
                if best_rank <= 10:
                    tp_10 += 1
                if best_rank <= 50:
                    tp_50 += 1
                if best_rank <= 100:
                    tp_100 += 1
                reciprocal_rank += 1 / best_rank
    pbar.close()

    return {
        "tp_1": tp_1,
        "tp_10": tp_10,
        "tp_50": tp_50,
        "tp_100": tp_100,
        "true": true,
        "reciprocal_rank": reciprocal_rank
    }

def submit_wandb_eval(metrics: dict[str, int | float]) -> None:
    wandb.log({"R@1": metrics["tp_1"]/metrics["true"]})
    wandb.log({"R@10": metrics["tp_10"]/metrics["true"]})
    wandb.log({"R@50": metrics["tp_50"]/metrics["true"]})
    wandb.log({"R@100": metrics["tp_100"]/metrics["true"]})
    wandb.log({"MRR": metrics["reciprocal_rank"]/metrics["true"]})
