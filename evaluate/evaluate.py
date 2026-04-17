from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.codebert_contrastive import ClassifierHead, CodeBERTContrastiveEncoder


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_records(path: str | Path) -> List[Dict]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_encoder(
    checkpoint_dir: Path,
    *,
    projection_dim: int = 128,
    dropout: float = 0.1,
) -> Tuple[CodeBERTContrastiveEncoder, object]:
    """Load the frozen contrastive encoder from a training checkpoint."""
    encoder_path = checkpoint_dir / "encoder"
    tokenizer_path = checkpoint_dir / "tokenizer"
    proj_path = checkpoint_dir / "projection_head.pt"

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    model = CodeBERTContrastiveEncoder(
        model_name=str(encoder_path),
        projection_dim=projection_dim,
        dropout=dropout,
    )
    state = torch.load(str(proj_path), map_location="cpu")
    model.projection_head.load_state_dict(state)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_records(
    model: CodeBERTContrastiveEncoder,
    tokenizer,
    records: List[Dict],
    *,
    max_length: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for record in tqdm(records, desc="Embedding", leave=False):
            emb = model.encode_contract(
                record["code"],
                tokenizer,
                max_length=max_length,
                stride=stride,
            )
            embeddings.append(emb.cpu().numpy())
    return np.stack(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of queries where at least one top-k neighbour shares the same label."""
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    hits = 0
    for i in range(len(embeddings)):
        top_k = np.argsort(sim[i])[::-1][:k]
        if any(labels[j] == labels[i] for j in top_k):
            hits += 1
    return hits / len(embeddings)


# ---------------------------------------------------------------------------
# Classifier head — train & evaluate
# ---------------------------------------------------------------------------

def train_classifier(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    projection_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> ClassifierHead:
    """Train MLP#2 (ClassifierHead with sigmoid) on top of frozen embeddings."""
    X = torch.tensor(train_embeddings, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    head = ClassifierHead(input_dim=projection_dim, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

    head.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  [classifier] epoch {epoch}/{epochs}  loss={running_loss / max(len(loader), 1):.4f}")

    return head


def evaluate_classifier(
    head: ClassifierHead,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    *,
    device: torch.device,
    threshold: float = 0.5,
) -> None:
    X = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    head.eval()
    with torch.no_grad():
        probs = head(X).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    print("\n=== Classifier Head — Classification Report ===")
    print(classification_report(test_labels, preds, digits=4))
    print(f"Accuracy : {accuracy_score(test_labels, preds):.4f}")


def save_classifier_head(head: ClassifierHead, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), output_dir / "classifier_head.pt")
    print(f"Classifier head saved → {output_dir / 'classifier_head.pt'}")


def load_classifier_head(
    path: Path, *, projection_dim: int, dropout: float, device: torch.device
) -> ClassifierHead:
    head = ClassifierHead(input_dim=projection_dim, dropout=dropout).to(device)
    head.load_state_dict(torch.load(str(path), map_location=device))
    return head


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)

    print(f"Loading encoder from: {checkpoint_dir}")
    encoder, tokenizer = load_encoder(
        checkpoint_dir,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    )
    encoder.to(device)

    # ---- embed test set ----
    test_records = load_records(args.test_path)
    test_labeled = [r for r in test_records if r.get("label") is not None]
    if not test_labeled:
        raise ValueError("No labeled records found in --test-path.")
    print(f"Test records  : {len(test_labeled)}")

    test_embeddings = embed_records(
        encoder, tokenizer, test_labeled,
        max_length=args.max_length, stride=args.stride, device=device,
    )
    test_labels = np.array([r["label"] for r in test_labeled])

    # ---- classifier head ----
    if args.classifier_head_path:
        head = load_classifier_head(
            Path(args.classifier_head_path),
            projection_dim=args.projection_dim,
            dropout=args.dropout,
            device=device,
        )
        print("Loaded classifier head from:", args.classifier_head_path)
    elif args.train_path:
        train_records = load_records(args.train_path)
        train_labeled = [r for r in train_records if r.get("label") is not None]
        if not train_labeled:
            raise ValueError("No labeled records found in --train-path.")
        print(f"Train records : {len(train_labeled)}")

        train_embeddings = embed_records(
            encoder, tokenizer, train_labeled,
            max_length=args.max_length, stride=args.stride, device=device,
        )
        train_labels = np.array([r["label"] for r in train_labeled])

        print("\nTraining classifier head (encoder frozen)...")
        head = train_classifier(
            train_embeddings, train_labels,
            projection_dim=args.projection_dim,
            dropout=args.dropout,
            epochs=args.clf_epochs,
            lr=args.clf_lr,
            batch_size=args.clf_batch_size,
            device=device,
        )
        if args.output_dir:
            save_classifier_head(head, Path(args.output_dir))
    else:
        raise ValueError(
            "Provide --train-path to train a classifier head, "
            "or --classifier-head-path to load a saved one."
        )

    evaluate_classifier(head, test_embeddings, test_labels, device=device)

    # ---- retrieval metrics ----
    print("\n=== Retrieval Metrics (test set) ===")
    for k in (1, 5, 10):
        if k < len(test_labeled):
            print(f"Recall@{k:<3}: {recall_at_k(test_embeddings, test_labels, k):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CodeBERT contrastive encoder with a classifier head."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Checkpoint dir from train_contrastive.py.")
    parser.add_argument("--test-path", type=str, required=True,
                        help="JSONL test file with {code, label, ...}.")
    parser.add_argument("--train-path", type=str, default=None,
                        help="JSONL train file to fit the classifier head.")
    parser.add_argument("--classifier-head-path", type=str, default=None,
                        help="Path to a saved classifier_head.pt (skips training).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save the trained classifier head.")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--clf-epochs", type=int, default=10,
                        help="Epochs to train the classifier head.")
    parser.add_argument("--clf-lr", type=float, default=1e-3,
                        help="Learning rate for the classifier head.")
    parser.add_argument("--clf-batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
