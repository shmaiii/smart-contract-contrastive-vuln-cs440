from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actual_model.contrastive_dataset import ChunkInferenceDataset, ChunkTripletDataset
from actual_model.codebert_contrastive import CodeBERTContrastiveEncoder


@dataclass
class TrainConfig:
    train_path: str = "datasets/train_chunks_subset.pt"
    val_path: str | None = None
    output_dir: str = "outputs/contrastive"
    model_name: str = "microsoft/codebert-base"
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    projection_dim: int = 128
    dropout: float = 0.1
    max_length: int = 512
    margin: float = 1.0
    lambda_clf: float = 0.5
    num_workers: int = 0
    seed: int = 42
    freeze_lower_layers: int = 0
    threshold: float = 0.5
    selection_metric: str = "pr_auc"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_first_layers(model: CodeBERTContrastiveEncoder, num_layers: int) -> None:
    if num_layers <= 0:
        return
    embeddings = getattr(model.encoder, "embeddings", None)
    if embeddings is not None:
        for param in embeddings.parameters():
            param.requires_grad = False
    encoder_layers = getattr(getattr(model.encoder, "encoder", None), "layer", None)
    if encoder_layers is None:
        return
    for layer in encoder_layers[:num_layers]:
        for param in layer.parameters():
            param.requires_grad = False


def save_checkpoint(
    checkpoint_dir: Path,
    model: CodeBERTContrastiveEncoder,
    config: TrainConfig,
    epoch: int,
    metrics: Dict[str, float] | None = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    payload = asdict(config)
    payload["saved_epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    with (checkpoint_dir / "train_config.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Checkpoint saved → {checkpoint_dir}")


def evaluate(
    model: CodeBERTContrastiveEncoder,
    dataloader: DataLoader,
    *,
    device: torch.device,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    contract_scores: Dict[int, List[float]] = defaultdict(list)
    contract_labels: Dict[int, int] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            contract_ids = batch["contract_id"]
            contract_label_batch = batch["contract_label"]

            embeddings = model.encode_chunks(input_ids, attention_mask)
            probs = torch.sigmoid(model.classifier_head(embeddings))

            for i, contract_id in enumerate(contract_ids.tolist()):
                contract_scores[contract_id].append(float(probs[i].item()))
                contract_labels[contract_id] = int(contract_label_batch[i].item())

    sorted_contract_ids = sorted(contract_scores.keys())
    contract_probs = np.array(
        [max(contract_scores[contract_id]) for contract_id in sorted_contract_ids]
    )
    contract_true = np.array(
        [contract_labels[contract_id] for contract_id in sorted_contract_ids]
    )
    contract_pred = (contract_probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(contract_true, contract_pred)),
        "precision": float(precision_score(contract_true, contract_pred, zero_division=0)),
        "recall": float(recall_score(contract_true, contract_pred, zero_division=0)),
        "f1": float(f1_score(contract_true, contract_pred, zero_division=0)),
    }

    try:
        metrics["pr_auc"] = float(average_precision_score(contract_true, contract_probs))
    except ValueError:
        metrics["pr_auc"] = 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(contract_true, contract_probs))
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ChunkTripletDataset.from_pt(
        config.train_path,
        max_length=config.max_length,
        seed=config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dataloader = None
    if config.val_path:
        val_dataset = ChunkInferenceDataset.from_pt(
            config.val_path,
            max_length=config.max_length,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    model = CodeBERTContrastiveEncoder(
        model_name=config.model_name,
        projection_dim=config.projection_dim,
        dropout=config.dropout,
    )
    freeze_first_layers(model, config.freeze_lower_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device} | dataset size={len(dataset)} | batches={len(dataloader)}")
    if val_dataloader is not None:
        print(f"Validation set loaded | size={len(val_dataset)} | batches={len(val_dataloader)}")

    triplet_loss_fn = nn.TripletMarginLoss(margin=config.margin, p=2)
    # BCEWithLogitsLoss handles per-sample weights via the weight argument
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
    )
    best_metric = float("-inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = total_triplet = total_clf = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in progress:
            optimizer.zero_grad()

            anchor_ids  = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            pos_ids     = batch["positive_input_ids"].to(device)
            pos_mask    = batch["positive_attention_mask"].to(device)
            neg_ids     = batch["negative_input_ids"].to(device)
            neg_mask    = batch["negative_attention_mask"].to(device)
            labels      = batch["label"].to(device)
            weights     = batch["weight"].to(device)

            # encode all three roles — classifier head only needed for anchor
            anchor_emb, anchor_scores = model(anchor_ids, anchor_mask)
            pos_emb, _                = model(pos_ids, pos_mask)
            neg_emb, _                = model(neg_ids, neg_mask)

            # contrastive loss (triplet)
            # anchor_scores are raw logits — feed directly to BCEWithLogitsLoss
            l_triplet = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            l_clf = (bce_loss_fn(anchor_scores, labels) * weights).mean()

            loss = l_triplet + config.lambda_clf * l_clf
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            total_triplet += l_triplet.item()
            total_clf     += l_clf.item()
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                triplet=f"{l_triplet.item():.4f}",
                clf=f"{l_clf.item():.4f}",
            )

        n = max(len(dataloader), 1)
        print(
            f"epoch={epoch} "
            f"avg_loss={total_loss/n:.4f} "
            f"triplet={total_triplet/n:.4f} "
            f"clf={total_clf/n:.4f}"
        )

        epoch_metrics = None
        if val_dataloader is not None:
            epoch_metrics = evaluate(
                model,
                val_dataloader,
                device=device,
                threshold=config.threshold,
            )
            print("validation " + " ".join(f"{k}={v:.4f}" for k, v in epoch_metrics.items()))

            selected_value = epoch_metrics[config.selection_metric]
            if selected_value > best_metric:
                best_metric = selected_value
                save_checkpoint(
                    output_dir / "best-checkpoint",
                    model,
                    config,
                    epoch,
                    metrics=epoch_metrics,
                )
                print(
                    f"  Best checkpoint updated on {config.selection_metric}={selected_value:.4f}"
                )

        save_checkpoint(
            output_dir / f"checkpoint-epoch-{epoch}",
            model,
            config,
            epoch,
            metrics=epoch_metrics,
        )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the CodeBERT contrastive encoder.")
    parser.add_argument("--train-path", type=str, default="datasets/train_chunks_subset.pt",
                        help=".pt file produced by the preprocessing pipeline.")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Optional .pt validation file for model selection.")
    parser.add_argument("--output-dir", type=str, default="outputs/contrastive")
    parser.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Triplet loss margin.")
    parser.add_argument("--lambda-clf", type=float, default=0.5,
                        help="Weight of classification loss in joint objective.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-lower-layers", type=int, default=0,
                        help="Number of encoder transformer layers to freeze.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="MIL threshold used during validation.")
    parser.add_argument("--selection-metric", type=str, default="pr_auc",
                        choices=["accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"],
                        help="Validation metric used to choose the best checkpoint.")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
