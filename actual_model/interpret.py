from __future__ import annotations

"""
Interpretability and invariance evaluation for CodeBERTContrastiveEncoder.

Two analyses:
  1. Invariance metric  — cosine distance between original and augmented chunk
                          embeddings. Lower = encoder is robust to augmentation.
  2. Token attributions — Integrated Gradients via Captum to identify which
                          tokens drive vulnerability predictions most strongly.
                          Also computes Jaccard overlap between top-k attributed
                          tokens for original vs augmented chunks.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from actual_model.codebert_contrastive import CodeBERTContrastiveEncoder


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors evaluate.py)
# ---------------------------------------------------------------------------

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
    state = torch.load(str(checkpoint_dir / "model.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_pt(path: str | Path) -> list:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def pad_or_truncate(token_ids: list, max_length: int = 512) -> torch.Tensor:
    t = torch.tensor(token_ids, dtype=torch.long)
    if len(t) > max_length:
        t = t[:max_length]
    elif len(t) < max_length:
        pad = torch.full((max_length - len(t),), 1, dtype=torch.long)
        t = torch.cat([t, pad])
    return t


# ---------------------------------------------------------------------------
# 1. Invariance metric
# ---------------------------------------------------------------------------

def invariance_metric(
    model: CodeBERTContrastiveEncoder,
    data: list,
    *,
    max_length: int = 512,
    batch_size: int = 32,
    device: torch.device,
    n_samples: int | None = None,
) -> None:
    """
    For each chunk, embed the original (anchor) and augmented (pos) version.
    Report cosine distance between them.

    Lower mean distance → model produces similar representations for
    semantically equivalent code → good invariance.
    """
    samples = data[:n_samples] if n_samples else data

    distances: List[float] = []

    # process in batches
    for start in tqdm(range(0, len(samples), batch_size), desc="Invariance"):
        batch = samples[start : start + batch_size]

        anchor_ids  = torch.stack([pad_or_truncate(x["anchor_input_ids"],  max_length) for x in batch]).to(device)
        pos_ids     = torch.stack([pad_or_truncate(x["pos_input_ids"],     max_length) for x in batch]).to(device)
        anchor_mask = anchor_ids.ne(1).long()
        pos_mask    = pos_ids.ne(1).long()

        with torch.no_grad():
            anchor_emb = model.encode_chunks(anchor_ids, anchor_mask)  # (B, 128) L2-normed
            pos_emb    = model.encode_chunks(pos_ids,    pos_mask)

        # cosine distance = 1 - cosine_similarity (embeddings already L2-normed)
        cos_sim  = (anchor_emb * pos_emb).sum(dim=-1)          # (B,)
        cos_dist = (1 - cos_sim).cpu().numpy()
        distances.extend(cos_dist.tolist())

    distances = np.array(distances)
    print("\n=== Invariance Metric (original vs augmented chunk) ===")
    print(f"Cosine distance  mean : {distances.mean():.4f}")
    print(f"Cosine distance  std  : {distances.std():.4f}")
    print(f"Cosine distance  min  : {distances.min():.4f}")
    print(f"Cosine distance  max  : {distances.max():.4f}")
    print("(Lower mean → more invariant representations)")


# ---------------------------------------------------------------------------
# 2. Token attributions via Integrated Gradients (Captum)
# ---------------------------------------------------------------------------

def _forward_from_embeds(
    model: CodeBERTContrastiveEncoder,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass that takes word embeddings directly instead of token IDs.
    Required by Captum because token IDs are discrete and non-differentiable.
    """
    outputs = model.encoder(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
    )
    from torch.nn import functional as F
    pooled    = model.pooler(outputs.last_hidden_state, attention_mask)
    projected = model.projection_head(pooled)
    normed    = F.normalize(projected, p=2, dim=-1)
    score     = torch.sigmoid(model.classifier_head(normed))  # (batch,) probabilities
    return score


def token_attributions(
    model: CodeBERTContrastiveEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution scores for each token position.

    Returns:
        attributions — (seq_len,) absolute attribution magnitude per token
    """
    from captum.attr import IntegratedGradients

    word_embeddings = model.encoder.embeddings.word_embeddings

    input_ids     = input_ids.unsqueeze(0)      # (1, seq_len)
    attention_mask = attention_mask.unsqueeze(0)

    input_embeds   = word_embeddings(input_ids)                          # (1, seq_len, 768)
    baseline_embeds = torch.zeros_like(input_embeds)                     # all-zero baseline

    ig = IntegratedGradients(
        lambda embeds: _forward_from_embeds(model, embeds, attention_mask)
    )

    attributions, _ = ig.attribute(
        input_embeds,
        baselines=baseline_embeds,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # L2 norm across the embedding dimension → one score per token
    token_scores = attributions.squeeze(0).norm(dim=-1).detach().cpu().numpy()  # (seq_len,)
    return token_scores


def attribution_analysis(
    model: CodeBERTContrastiveEncoder,
    data: list,
    *,
    max_length: int = 512,
    top_k: int = 10,
    n_samples: int = 20,
    n_steps: int = 50,
    device: torch.device,
) -> None:
    """
    For n_samples chunks:
      - Compute token attributions for both anchor and augmented (pos) version
      - Report top-k tokens by attribution for the anchor
      - Compute Jaccard overlap of top-k token sets between anchor and pos
        (higher Jaccard → explanations are stable under augmentation)
    """
    try:
        from captum.attr import IntegratedGradients  # noqa: F401
    except ImportError:
        print("captum not installed. Run: pip install captum")
        return

    samples = data[:n_samples]
    jaccard_scores: List[float] = []

    print(f"\n=== Token Attribution Analysis ({n_samples} samples, top-k={top_k}) ===")

    for idx, item in enumerate(tqdm(samples, desc="Attributions")):
        anchor_ids = pad_or_truncate(item["anchor_input_ids"], max_length).to(device)
        pos_ids    = pad_or_truncate(item["pos_input_ids"],    max_length).to(device)
        anchor_mask = anchor_ids.ne(1).long()
        pos_mask    = pos_ids.ne(1).long()

        anchor_attr = token_attributions(model, anchor_ids, anchor_mask, n_steps=n_steps)
        pos_attr    = token_attributions(model, pos_ids,    pos_mask,    n_steps=n_steps)

        # top-k token positions by attribution magnitude
        anchor_topk = set(np.argsort(anchor_attr)[::-1][:top_k].tolist())
        pos_topk    = set(np.argsort(pos_attr)[::-1][:top_k].tolist())

        intersection = len(anchor_topk & pos_topk)
        union        = len(anchor_topk | pos_topk)
        jaccard      = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

        if idx < 3:  # print detail for first 3 samples
            print(f"\n  Sample {idx} (contract_id={item['contract_id']}, label={item['label']})")
            print(f"  Top-{top_k} anchor token positions : {sorted(anchor_topk)}")
            print(f"  Top-{top_k} augmented token positions: {sorted(pos_topk)}")
            print(f"  Jaccard overlap                    : {jaccard:.4f}")

    jaccard_arr = np.array(jaccard_scores)
    print(f"\n--- Jaccard overlap summary (top-{top_k} tokens, {n_samples} samples) ---")
    print(f"Mean : {jaccard_arr.mean():.4f}")
    print(f"Std  : {jaccard_arr.std():.4f}")
    print("(Higher mean → attribution explanations are stable under augmentation)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpretability and invariance analysis for the contrastive model."
    )
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing model.pt.")
    parser.add_argument("--data-path", type=str, required=True,
                        help=".pt file with pre-chunked data (anchor + pos pairs).")
    parser.add_argument("--model-name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-invariance-samples", type=int, default=None,
                        help="Number of chunks to use for invariance metric (default: all).")
    parser.add_argument("--n-attribution-samples", type=int, default=20,
                        help="Number of chunks to run Integrated Gradients on.")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k tokens to compare for Jaccard overlap.")
    parser.add_argument("--ig-steps", type=int, default=50,
                        help="Number of integration steps for Integrated Gradients.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint_dir}")
    model = load_model(
        Path(args.checkpoint_dir),
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        device=device,
    )

    print(f"Loading data from: {args.data_path}")
    data = load_pt(args.data_path)
    print(f"Total chunks: {len(data)}")

    invariance_metric(
        model, data,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
        n_samples=args.n_invariance_samples,
    )

    attribution_analysis(
        model, data,
        max_length=args.max_length,
        top_k=args.top_k,
        n_samples=args.n_attribution_samples,
        n_steps=args.ig_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
