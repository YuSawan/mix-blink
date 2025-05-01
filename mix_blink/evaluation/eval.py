import torch
import wandb
from datasets import Dataset
from tqdm.auto import tqdm

from ..model import MixBlink
from ..retriever import DenseRetriever


@torch.no_grad()
def evaluate(model: MixBlink, dataset: Dataset, retriever: DenseRetriever) -> dict[str, int]:
    retriever.build_index(model.entity_encoder)
    dataloader = retriever.get_dataloader(dataset, retriever.mention_tokenizer)
    dictionary = retriever.dictionary
    model.to(retriever.device)

    true, tp_1, tp_5, tp_10, tp_20, tp_50, tp_100 = 0, 0, 0, 0, 0, 0, 0
    pbar = tqdm(total=len(dataloader), desc="Eval")
    for batch, labels in dataloader:
        pbar.update()
        batch = batch.to(retriever.device)
        query = model.mention_encoder(**batch).to('cpu').detach().numpy().copy()
        _, batch_indices = retriever.search_knn(query, top_k=100)
        for idxs, indices in zip(labels, batch_indices):
            true += 1
            if dictionary[idxs[0]].id in indices:
                tp_100 += 1
                if dictionary[idxs[0]].id in indices[:50]:
                    tp_50 += 1
                    if dictionary[idxs[0]].id in indices[:20]:
                        tp_20 += 1
                        if dictionary[idxs[0]].id in indices[:10]:
                            tp_10 += 1
                            if dictionary[idxs[0]].id in indices[:5]:
                                tp_5 += 1
                                if dictionary[idxs[0]].id in indices[:1]:
                                    tp_1 += 1
    pbar.close()

    return {
        "tp_1": tp_1,
        "tp_5": tp_5,
        "tp_10": tp_10,
        "tp_20": tp_20,
        "tp_50": tp_50,
        "tp_100": tp_100,
        "true": true
    }

def submit_wandb_eval(metrics: dict[str, int]) -> None:
    wandb.log({"R@1": metrics["tp_1"]/metrics["true"]})
    wandb.log({"R@5": metrics["tp_1"]/metrics["true"]})
    wandb.log({"R@10": metrics["tp_10"]/metrics["true"]})
    wandb.log({"R@20": metrics["tp_20"]/metrics["true"]})
    wandb.log({"R@50": metrics["tp_50"]/metrics["true"]})
    wandb.log({"R@100": metrics["tp_100"]/metrics["true"]})
