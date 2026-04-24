import torch
import torch.nn as nn
from transformers import AutoModel

class MeanPooling(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

class BaselineModel(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooler = MeanPooling()
        hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask, contract_ids=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        chunk_logits = self.classifier(pooled).squeeze(-1)

        # MIL Max-Pooling Aggregation
        if contract_ids is not None:
            unique_ids = torch.unique(contract_ids)
            return torch.stack([torch.max(chunk_logits[contract_ids == cid]) for cid in unique_ids])
        
        return chunk_logits

    def forward_for_captum(self, input_ids, attention_mask):
        return self.forward(input_ids, attention_mask)
