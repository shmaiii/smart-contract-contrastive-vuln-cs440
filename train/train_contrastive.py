from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.contrastive_dataset import ContrastiveDataset
from models.codebert_contrastive import CodeBERTContrastiveEncoder


@dataclass
class TrainConfig:
    train_path: str | None = None
    output_dir: str = "outputs/contrastive"
    model_name: str = "microsoft/codebert-base"
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 2e-5
    projection_dim: int = 128
    dropout: float = 0.1
    max_length: int = 512
    stride: int = 256
    margin: float = 1.0
    num_workers: int = 0
    seed: int = 42
    freeze_lower_layers: int = 0
    smoke_test: bool = False


def collate_triplets(batch: List[Dict]) -> Dict[str, List[str]]:
    return {
        "anchor": [item["anchor"] for item in batch],
        "positive": [item["positive"] for item in batch],
        "negative": [item["negative"] for item in batch],
    }


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


def build_smoke_records() -> List[Dict]:
    return [
        {
            "parent_id": "safe_1",
            "label": 0,
            "code": "pragma solidity ^0.8.0; contract A { function x() public pure returns (uint) { return 1; } }",
        },
        {
            "parent_id": "safe_1",
            "label": 0,
            "code": "pragma solidity ^0.8.0;\ncontract A {\n    function x() public pure returns (uint) {\n        return 1;\n    }\n}",
        },
        {
            "parent_id": "vuln_1",
            "label": 1,
            "code": "pragma solidity ^0.8.0; contract B { uint public x; function set(uint v) public { x = v; } }",
        },
        {
            "parent_id": "vuln_1",
            "label": 1,
            "code": "pragma solidity ^0.8.0;\ncontract B {\n    uint public x;\n    function set(uint value) public {\n        x = value;\n    }\n}",
        },
    ]


def save_checkpoint(
    output_dir: Path,
    model: CodeBERTContrastiveEncoder,
    tokenizer,
    config: TrainConfig,
    epoch: int,
) -> None:
    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(checkpoint_dir / "encoder")
    tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
    torch.save(model.projection_head.state_dict(), checkpoint_dir / "projection_head.pt")
    with (checkpoint_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.smoke_test:
        dataset = ContrastiveDataset(build_smoke_records(), seed=config.seed)
    elif config.train_path:
        dataset = ContrastiveDataset.from_jsonl(config.train_path, seed=config.seed)
    else:
        raise ValueError("Provide --train-path or enable --smoke-test.")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = CodeBERTContrastiveEncoder(
        model_name=config.model_name,
        projection_dim=config.projection_dim,
        dropout=config.dropout,
    )
    freeze_first_layers(model, config.freeze_lower_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_triplets,
    )

    criterion = nn.TripletMarginLoss(margin=config.margin, p=2)
    optimizer = AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
    )

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}")
        for batch in progress:
            optimizer.zero_grad()

            anchor_embeddings = model(
                batch["anchor"],
                tokenizer,
                max_length=config.max_length,
                stride=config.stride,
            )
            positive_embeddings = model(
                batch["positive"],
                tokenizer,
                max_length=config.max_length,
                stride=config.stride,
            )
            negative_embeddings = model(
                batch["negative"],
                tokenizer,
                max_length=config.max_length,
                stride=config.stride,
            )

            loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(len(dataloader), 1)
        print(f"epoch={epoch} avg_loss={epoch_loss:.4f}")
        save_checkpoint(output_dir, model, tokenizer, config, epoch)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a CodeBERT contrastive encoder.")
    parser.add_argument("--train-path", type=str, default=None, help="JSONL path with {parent_id, code, ...}.")
    parser.add_argument("--output-dir", type=str, default="outputs/contrastive")
    parser.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-lower-layers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
