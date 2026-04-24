import torch
from torch.utils.data import Dataset

class SmartContractDataset(Dataset):
    def __init__(self, data, max_length=512):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _pad_or_truncate(self, tokens):
        # Convert to tensor if it isn't one
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # 1. Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # 2. Pad with 1 (CodeBERT pad token) if too short
        elif len(tokens) < self.max_length:
            padding = torch.full((self.max_length - len(tokens),), 1, dtype=torch.long)
            tokens = torch.cat([tokens, padding])
            
        return tokens

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process both Anchor and Positive
        anchor_ids = self._pad_or_truncate(item["anchor_input_ids"])
        pos_ids = self._pad_or_truncate(item["pos_input_ids"])
    
        return {
            "anchor_input_ids": anchor_ids,
            "anchor_attention_mask": anchor_ids.ne(1).to(torch.long),

            "pos_input_ids": pos_ids,
            "pos_attention_mask": pos_ids.ne(1).to(torch.long),

            "label": torch.tensor(item["label"], dtype=torch.float),
            "weight": torch.tensor(item["weight"], dtype=torch.float),
            "contract_id": torch.tensor(item["contract_id"], dtype=torch.long)
        }
