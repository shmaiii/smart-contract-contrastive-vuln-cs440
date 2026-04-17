from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from utils.chunking import chunk_code


class ProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
class ClassifierHead(nn.Module):
    """MLP#2 — binary classifier that sits on top of frozen contrastive embeddings."""

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)

class CodeBERTContrastiveEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection_head = ProjectionHead(
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            dropout=dropout,
        )

    def encode_contract(
        self,
        code: str,
        tokenizer,
        *,
        max_length: int = 512,
        stride: int = 256,
    ) -> torch.Tensor:
        chunk_batch = chunk_code(
            code,
            tokenizer,
            max_length=max_length,
            stride=stride,
        )
        device = next(self.parameters()).device
        input_ids = chunk_batch["input_ids"].to(device)
        attention_mask = chunk_batch["attention_mask"].to(device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        chunk_representations = outputs.last_hidden_state[:, 0, :]
        contract_representation = chunk_representations.mean(dim=0)
        embedding = self.projection_head(contract_representation)
        return F.normalize(embedding, p=2, dim=-1)

    def forward(
        self,
        codes: Sequence[str],
        tokenizer,
        *,
        max_length: int = 512,
        stride: int = 256,
    ) -> torch.Tensor:
        embeddings = [
            self.encode_contract(
                code,
                tokenizer,
                max_length=max_length,
                stride=stride,
            )
            for code in codes
        ]
        return torch.stack(embeddings, dim=0)
