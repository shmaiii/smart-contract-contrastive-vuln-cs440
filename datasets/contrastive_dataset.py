from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Dict],
        *,
        seed: int = 42,
    ) -> None:
        if not records:
            raise ValueError("ContrastiveDataset requires at least one record.")

        self.records = list(records)
        self.random = random.Random(seed)
        self.parent_to_indices: Dict[str, List[int]] = defaultdict(list)

        for idx, record in enumerate(self.records):
            parent_id = record.get("parent_id")
            code = record.get("code")
            if parent_id is None:
                raise ValueError(f"Record {idx} is missing required key 'parent_id'.")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(f"Record {idx} is missing a non-empty 'code' string.")
            self.parent_to_indices[str(parent_id)].append(idx)

        self.parent_ids = list(self.parent_to_indices.keys())
        if len(self.parent_ids) < 2:
            raise ValueError("ContrastiveDataset needs at least two distinct parent_id values.")

    @classmethod
    def from_jsonl(cls, path: str | Path, *, seed: int = 42) -> "ContrastiveDataset":
        file_path = Path(path)
        records = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records, seed=seed)

    def __len__(self) -> int:
        return len(self.records)

    def _sample_positive_pair(self, parent_id: str) -> tuple[Dict, Dict]:
        indices = self.parent_to_indices[parent_id]
        if len(indices) == 1:
            record = self.records[indices[0]]
            return record, record

        anchor_idx, positive_idx = self.random.sample(indices, 2)
        return self.records[anchor_idx], self.records[positive_idx]

    def _sample_negative(self, excluded_parent_id: str) -> Dict:
        negative_parent_choices = [pid for pid in self.parent_ids if pid != excluded_parent_id]
        negative_parent_id = self.random.choice(negative_parent_choices)
        negative_idx = self.random.choice(self.parent_to_indices[negative_parent_id])
        return self.records[negative_idx]

    def __getitem__(self, index: int) -> Dict:
        parent_id = self.parent_ids[index % len(self.parent_ids)]
        anchor_record, positive_record = self._sample_positive_pair(parent_id)
        negative_record = self._sample_negative(parent_id)

        return {
            "anchor": anchor_record["code"],
            "positive": positive_record["code"],
            "negative": negative_record["code"],
            "anchor_parent_id": str(anchor_record["parent_id"]),
            "negative_parent_id": str(negative_record["parent_id"]),
            "anchor_label": anchor_record.get("label"),
            "positive_label": positive_record.get("label"),
            "negative_label": negative_record.get("label"),
        }
