from dataclasses import dataclass
import math
from typing import Optional

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(
            1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim)
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(
            config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(seq_len, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(
                batch_size * seq_len, -1), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


if __name__ == "__main__":
    config = TransformerConfig(vocab_size=100, block_size=32)
    model = TransformerLM(config)

    batch = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(batch, targets)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))
