from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class MeanPooling(nn.Module):
    """Attention-mask weighted mean pooling over token embeddings."""

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class ProjectionHead(nn.Module):
    """MLP#1 — maps encoder hidden states into contrastive embedding space."""

    def __init__(self, hidden_size: int, projection_dim: int = 128, dropout: float = 0.1) -> None:
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
    """MLP#2 — binary vulnerability classifier on top of projected embeddings."""

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
        self.pooler = MeanPooling()
        self.projection_head = ProjectionHead(
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            dropout=dropout,
        )
        self.classifier_head = ClassifierHead(
            input_dim=projection_dim,
            dropout=dropout,
        )

    def encode_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of pre-tokenized chunks.

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            L2-normalised embeddings of shape (batch, projection_dim)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        projected = self.projection_head(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def classify_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return per-chunk vulnerability probability scores (0-1).

        Args:
            input_ids:      (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            scores of shape (batch,)
        """
        embeddings = self.encode_chunks(input_ids, attention_mask)
        return self.classifier_head(embeddings)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        contract_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass used by the training loop.

        Returns:
            embeddings  — (batch, projection_dim) — for contrastive loss
            chunk_scores — (batch,) or (num_contracts,) — for classification loss

        If contract_ids is provided, chunk_scores are aggregated per contract
        via max-pooling (MIL), matching the baseline's approach.
        """
        embeddings = self.encode_chunks(input_ids, attention_mask)
        chunk_scores = self.classifier_head(embeddings)

        if contract_ids is not None:
            unique_ids = torch.unique(contract_ids)
            chunk_scores = torch.stack(
                [chunk_scores[contract_ids == cid].max() for cid in unique_ids]
            )

        return embeddings, chunk_scores

