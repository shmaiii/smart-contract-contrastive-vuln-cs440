from __future__ import annotations

from typing import Dict, List

import torch


def _pad_chunk(
    token_ids: List[int],
    *,
    max_length: int,
    pad_token_id: int,
) -> List[int]:
    if len(token_ids) > max_length:
        raise ValueError(f"Chunk length {len(token_ids)} exceeds max_length={max_length}.")
    return token_ids + [pad_token_id] * (max_length - len(token_ids))


def chunk_code(
    code: str,
    tokenizer,
    *,
    max_length: int = 512,
    stride: int = 256,
) -> Dict[str, torch.Tensor]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if stride >= max_length:
        raise ValueError("stride must be smaller than max_length.")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    chunk_payload = max_length - special_tokens
    if chunk_payload <= 0:
        raise ValueError("max_length is too small for tokenizer special tokens.")

    raw_token_ids = tokenizer.encode(code, add_special_tokens=False)
    if not raw_token_ids:
        raw_token_ids = [tokenizer.unk_token_id or tokenizer.pad_token_id or 0]

    step = chunk_payload - stride
    if step <= 0:
        raise ValueError(
            "Chunk step must be positive. Reduce stride or increase max_length."
        )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    input_id_chunks: List[List[int]] = []
    attention_mask_chunks: List[List[int]] = []

    start = 0
    while start < len(raw_token_ids):
        window = raw_token_ids[start : start + chunk_payload]
        chunk_ids = tokenizer.build_inputs_with_special_tokens(window)
        padded_chunk_ids = _pad_chunk(
            chunk_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
        )
        attention_mask = [1] * len(chunk_ids) + [0] * (max_length - len(chunk_ids))
        input_id_chunks.append(padded_chunk_ids)
        attention_mask_chunks.append(attention_mask)

        if start + chunk_payload >= len(raw_token_ids):
            break
        start += step

    return {
        "input_ids": torch.tensor(input_id_chunks, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_chunks, dtype=torch.long),
    }
