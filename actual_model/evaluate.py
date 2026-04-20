from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actual_model.contrastive_dataset import ChunkInferenceDataset
from actual_model.codebert_contrastive import CodeBERTContrastiveEncoder


def load_model(
    checkpoint_dir: Path,
    *,
    model_name: str = "microsoft/codebert-base",
    projection_dim: int = 128,
    dropout: float = 0.1,
    device: torch.device,
) -> CodeBERTContrastiveEncoder:
    model = CodeBERTContrastiveEncoder(
        model_name=model_name,
        projection_dim=projection_dim,
        dropout=dropout,
    )
    state = torch.load(str(checkpoint_dir / "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: CodeBERTContrastiveEncoder,
    dataset: ChunkInferenceDataset,
    *,
    batch_size: int,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings: List[np.ndarray] = []
    contract_scores: Dict[int, List[float]] = defaultdict(list)
    contract_label_map: Dict[int, int] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            contract_ids = batch["contract_id"]
            contract_labels = batch["contract_label"]

            embeddings = model.encode_chunks(input_ids, attention_mask)
            scores = model.classifier_head(embeddings)

            all_embeddings.append(embeddings.cpu().numpy())

            for i, contract_id in enumerate(contract_ids.tolist()):
                contract_scores[contract_id].append(float(scores[i].item()))
                contract_label_map[contract_id] = int(contract_labels[i].item())

    sorted_contract_ids = sorted(contract_scores.keys())
    contract_preds = np.array(
        [int(max(contract_scores[contract_id]) >= threshold) for contract_id in sorted_contract_ids]
    )
    contract_labels = np.array(
        [contract_label_map[contract_id] for contract_id in sorted_contract_ids]
    )
    contract_probs = np.array(
        [max(contract_scores[contract_id]) for contract_id in sorted_contract_ids]
    )
    chunk_embeddings = np.concatenate(all_embeddings, axis=0)

    return contract_preds, contract_labels, chunk_embeddings, contract_probs


def compute_classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

    try:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics["pr_auc"] = 0.0

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics


def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    hits = sum(
        any(labels[j] == labels[i] for j in np.argsort(sim[i])[::-1][:k])
        for i in range(len(embeddings))
    )
    return hits / len(embeddings)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"Loading model from: {checkpoint_dir}")
    model = load_model(
        checkpoint_dir,
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        device=device,
    )

    print(f"Loading test data from: {args.test_path}")
    dataset = ChunkInferenceDataset.from_pt(args.test_path, max_length=args.max_length)
    print(f"Test chunks: {len(dataset)}")

    contract_preds, contract_labels, chunk_embeddings, contract_probs = run_inference(
        model,
        dataset,
        batch_size=args.batch_size,
        device=device,
        threshold=args.threshold,
    )
    metrics = compute_classification_metrics(
        contract_labels,
        contract_preds,
        contract_probs,
    )

    print("\n=== Contract-Level Classification Report ===")
    print(classification_report(contract_labels, contract_preds, digits=4))
    print("\n=== Contract-Level Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    chunk_labels = np.array([dataset[i]["label"].item() for i in range(len(dataset))])
    print("\n=== Retrieval Metrics (chunk-level embeddings) ===")
    for k in (1, 5, 10):
        if k < len(chunk_embeddings):
            print(f"Recall@{k:<3}: {recall_at_k(chunk_embeddings, chunk_labels, k):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained CodeBERT contrastive model."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing model.pt and train_config.json.")
    parser.add_argument("--test-path", type=str, required=True,
                        help=".pt test file with pre-chunked contracts.")
    parser.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold for binary prediction.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
