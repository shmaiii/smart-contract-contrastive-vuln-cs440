import torch
from torch.utils.data import Dataset

class SmartContractEvalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def _pad_or_truncate(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long)

        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            padding = torch.full(
                (self.max_length - len(tokens),),
                self.pad_token_id,
                dtype=torch.long
            )
            tokens = torch.cat([tokens, padding])

        return tokens

    def __getitem__(self, idx):
        item = self.data[idx]

        input_key = "input_ids" if "input_ids" in item else "anchor_input_ids"
        tokens = self._pad_or_truncate(item[input_key])

        return {
            "input_ids": tokens,
            "attention_mask": tokens.ne(self.pad_token_id).long(),
            "label": torch.tensor(item["label"], dtype=torch.float32),
            "contract_id": item.get("contract_id", -1),
            "contract_label": item.get("contract_label", item["label"]),
        }
