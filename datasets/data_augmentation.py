import random
import hashlib
from typing import Optional
from pygments import lexers, token


class SemanticDataAugmentor:
    def __init__(
        self,
        enable_whitespace: bool = True,
        enable_comments: bool = True,
        enable_noop_injection: bool = True,
        max_noops: int = 2,
    ):
        self.enable_whitespace = enable_whitespace
        self.enable_comments = enable_comments
        self.enable_noop_injection = enable_noop_injection
        self.max_noops = max_noops
        self.lexer = lexers.get_lexer_by_name("solidity")

    def _get_seed(self, contract_id: Optional[str]) -> int:
        if contract_id is None:
            return random.randint(0, 2**32 - 1)
        return int(hashlib.md5(str(contract_id).encode()).hexdigest(), 16) % (2**32)

    def _normalize_whitespace(self, code: str) -> str:
        lines = [line.rstrip() for line in code.splitlines()]
        cleaned = []
        prev_empty = False

        for line in lines:
            if not line:
                if not prev_empty:
                    cleaned.append("")
                prev_empty = True
            else:
                cleaned.append(line)
                prev_empty = False

        return "\n".join(cleaned)

    def __call__(self, contract_text: str, contract_id: Optional[str] = None) -> str:
        rng = random.Random(self._get_seed(contract_id))

        toks = list(self.lexer.get_tokens(contract_text))
        augmented_tokens = []

        for i, (ttype, value) in enumerate(toks):

            # --- Comment augmentation ---
            if self.enable_comments and ttype in token.Comment:
                if rng.random() < 0.3:
                    continue
                augmented_tokens.append((ttype, value))
                continue

            augmented_tokens.append((ttype, value))

            # --- Safer noop injection ---
            if self.enable_noop_injection and value == "{":
                # Look back for function-like keywords
                context_tokens = toks[max(0, i - 15):i]
                context_str = "".join(t[1] for t in context_tokens)

                if any(k in context_str for k in ["function", "constructor", "modifier"]):
                    num_noops = rng.randint(0, self.max_noops)

                    for _ in range(num_noops):
                        # SAFE noop (no storage impact)
                        noop = " if (false) { uint256(0); } "
                        augmented_tokens.append((token.Text, noop))

        transformed = "".join(t[1] for t in augmented_tokens)

        if self.enable_whitespace:
            transformed = self._normalize_whitespace(transformed)

        # Guarantee difference
        if transformed == contract_text:
            transformed += " "

        return transformed