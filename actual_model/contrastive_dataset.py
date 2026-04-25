from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class ChunkTripletDataset(Dataset):
    """
    Training dataset — loads pre-chunked data from a .pt file.

    Each item in the .pt list is expected to have:
        contract_id       (int)   — groups chunks belonging to the same contract
        anchor_input_ids  (list)  — token IDs for the original chunk
        pos_input_ids     (list)  — token IDs for the augmented (positive) chunk
        label             (int)   — chunk-level vulnerability label (0/1)
        weight            (float) — class-imbalance weight
        contract_label    (int)   — contract-level label
        chunk_line_labels (list)  — per-line labels within the chunk

    __getitem__ returns a triplet:
        anchor   = original chunk
        positive = pre-augmented version of the same chunk  (from pos_input_ids)
        negative = a random chunk from a *different* contract_id
    """

    def __init__(self, data: List[Dict], *, max_length: int = 512, seed: int = 42) -> None:
        if not data:
            raise ValueError("ChunkTripletDataset requires at least one record.")

        self.data = data
        self.max_length = max_length
        self.rng = random.Random(seed)

        # index chunks by contract_id and by label for negative sampling
        self.contract_to_indices: Dict[int, List[int]] = defaultdict(list)
        # label_to_indices[label] = list of dataset indices with that chunk label
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        # contract_label_to_indices[(contract_id, label)] = indices
        self.contract_label_to_indices: Dict[tuple, List[int]] = defaultdict(list)

        for idx, item in enumerate(self.data):
            cid   = item["contract_id"]
            label = int(item["label"])
            self.contract_to_indices[cid].append(idx)
            self.label_to_indices[label].append(idx)
            self.contract_label_to_indices[(cid, label)].append(idx)

        self.contract_ids = list(self.contract_to_indices.keys())
        if len(self.contract_ids) < 2:
            raise ValueError("ChunkTripletDataset needs at least two distinct contract_ids.")

    @classmethod
    def from_pt(cls, path: str | Path, *, max_length: int = 512, seed: int = 42) -> "ChunkTripletDataset":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(data, max_length=max_length, seed=seed)

    def __len__(self) -> int:
        return len(self.data)

    def _pad_or_truncate(self, token_ids: list) -> torch.Tensor:
        t = torch.tensor(token_ids, dtype=torch.long)
        if len(t) > self.max_length:
            t = t[: self.max_length]
        elif len(t) < self.max_length:
            pad = torch.full((self.max_length - len(t),), 1, dtype=torch.long)
            t = torch.cat([t, pad])
        return t

    def _sample_negative(self, anchor_contract_id: int, anchor_label: int) -> Dict:
        """
        Hard-negative mining with fallback chain:

        1. Hard negative   — opposite-label chunk from the SAME contract
           (e.g. safe chunk inside the same vulnerable contract)
        2. Medium negative — opposite-label chunk from a DIFFERENT contract
        3. Easy negative   — any chunk from a different contract (original behaviour)

        The opposite label forces the model to separate vulnerability signal
        rather than just contract identity.
        """
        opposite_label = 1 - anchor_label

        # 1. Hard: opposite label, same contract
        hard_pool = self.contract_label_to_indices.get((anchor_contract_id, opposite_label), [])
        if hard_pool:
            return self.data[self.rng.choice(hard_pool)]

        # 2. Medium: opposite label, different contract
        medium_pool = [
            idx for idx in self.label_to_indices.get(opposite_label, [])
            if self.data[idx]["contract_id"] != anchor_contract_id
        ]
        if medium_pool:
            return self.data[self.rng.choice(medium_pool)]

        # 3. Easy fallback: any chunk from a different contract
        other_contracts = [cid for cid in self.contract_ids if cid != anchor_contract_id]
        neg_contract_id = self.rng.choice(other_contracts)
        neg_idx = self.rng.choice(self.contract_to_indices[neg_contract_id])
        return self.data[neg_idx]

    def __getitem__(self, index: int) -> Dict:
        item = self.data[index]
        neg = self._sample_negative(item["contract_id"], int(item["label"]))

        anchor_ids = self._pad_or_truncate(item["anchor_input_ids"])
        pos_ids = self._pad_or_truncate(item["pos_input_ids"])
        neg_ids = self._pad_or_truncate(neg["anchor_input_ids"])

        return {
            "anchor_input_ids": anchor_ids,
            "anchor_attention_mask": anchor_ids.ne(1).long(),
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_ids.ne(1).long(),
            "negative_input_ids": neg_ids,
            "negative_attention_mask": neg_ids.ne(1).long(),
            "label": torch.tensor(item["label"], dtype=torch.float),
            "weight": torch.tensor(item["weight"], dtype=torch.float),
            "contract_id": torch.tensor(item["contract_id"], dtype=torch.long),
            "contract_label": torch.tensor(item["contract_label"], dtype=torch.float),
        }


class ChunkInferenceDataset(Dataset):
    """
    Eval / inference dataset — no triplets, no negatives.
    Returns one chunk at a time with its contract_id for MIL aggregation.
    Mirrors baseline_model/eval_datasets.py SmartContractEvalDataset.
    """

    def __init__(self, data: List[Dict], *, max_length: int = 512) -> None:
        self.data = data
        self.max_length = max_length

    @classmethod
    def from_pt(cls, path: str | Path, *, max_length: int = 512) -> "ChunkInferenceDataset":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(data, max_length=max_length)

    def __len__(self) -> int:
        return len(self.data)

    def _pad_or_truncate(self, token_ids: list) -> torch.Tensor:
        t = torch.tensor(token_ids, dtype=torch.long)
        if len(t) > self.max_length:
            t = t[: self.max_length]
        elif len(t) < self.max_length:
            pad = torch.full((self.max_length - len(t),), 1, dtype=torch.long)
            t = torch.cat([t, pad])
        return t

    def __getitem__(self, index: int) -> Dict:
        item = self.data[index]
        ids = self._pad_or_truncate(item["anchor_input_ids"])
        return {
            "input_ids": ids,
            "attention_mask": ids.ne(1).long(),
            "label": torch.tensor(item["label"], dtype=torch.float),
            "contract_id": torch.tensor(item["contract_id"], dtype=torch.long),
            "contract_label": torch.tensor(item["contract_label"], dtype=torch.float),
        }
