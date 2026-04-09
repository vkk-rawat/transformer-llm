from dataclasses import dataclass
import math
from typing import Optional

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]


def format_tensor_preview(tensor: torch.Tensor, max_values: int = 6) -> str:
    flat = tensor.detach().flatten().float()
    if flat.numel() == 0:
        return "[]"

    preview_values = ", ".join(
        f"{value:.4f}" for value in flat[:max_values].tolist())
    if flat.numel() > max_values:
        preview_values += ", ..."
    return f"[{preview_values}]"


def print_model_report(model: nn.Module, preview_values: int = 6) -> None:
    print("model architecture:")
    print(model)
    print()
    print("parameter summary:")

    total_parameters = 0
    trainable_parameters = 0

    for name, parameter in model.named_parameters():
        parameter_count = parameter.numel()
        total_parameters += parameter_count
        if parameter.requires_grad:
            trainable_parameters += parameter_count

        data = parameter.detach().float()
        mean = float(data.mean())
        std = float(data.std(unbiased=False)) if data.numel() > 1 else 0.0
        preview = format_tensor_preview(parameter, max_values=preview_values)
        print(
            f"- {name}: shape={tuple(parameter.shape)}, params={parameter_count}, "
            f"mean={mean:.6f}, std={std:.6f}, preview={preview}"
        )

    print()
    print(f"total parameters: {total_parameters}")
    print(f"trainable parameters: {trainable_parameters}")


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.stoi = {char: idx for idx, char in enumerate(vocab)}
        self.itos = {idx: char for idx, char in enumerate(vocab)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        unknown_chars = sorted(
            {char for char in text if char not in self.stoi})
        if unknown_chars:
            raise ValueError(
                f"Input text contains unknown characters: {unknown_chars}")
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)


def build_training_sequences(text: str, tokenizer: CharTokenizer, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.encode(text)
    if len(encoded) <= block_size:
        raise ValueError("Training text must be longer than block_size")

    x_data: list[list[int]] = []
    y_data: list[list[int]] = []
    for start in range(len(encoded) - block_size):
        chunk = encoded[start:start + block_size + 1]
        x_data.append(chunk[:-1])
        y_data.append(chunk[1:])

    x = torch.tensor(x_data, dtype=torch.long)
    y = torch.tensor(y_data, dtype=torch.long)
    return x, y


def train_language_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 250,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    sample_count = x.size(0)
    batch_size = min(batch_size, sample_count)
    log_interval = max(1, steps // 10)

    for step in range(1, steps + 1):
        batch_indices = torch.randint(
            0, sample_count, (batch_size,), device=x.device)
        batch_x = x[batch_indices]
        batch_y = y[batch_indices]

        _, loss = model(batch_x, batch_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_interval == 0 or step == steps:
            print(f"step {step:4d}/{steps}  loss={float(loss.detach()):.6f}")

    model.eval()


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
        positions = positions.expand(batch_size, seq_len)
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
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(42)

    training_text = (
        "transformers are sequence models. "
        "large language models use transformer blocks for next token prediction. "
        "attention helps the model focus on relevant previous tokens. "
    ) * 30

    tokenizer = CharTokenizer.from_text(training_text)
    config = TransformerConfig(vocab_size=tokenizer.vocab_size, block_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerLM(config).to(device)

    print_model_report(model)
    print()

    x_train, y_train = build_training_sequences(
        training_text, tokenizer, config.block_size)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    print(f"device: {device}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"training sequences: {x_train.size(0)}")
    print()
    print("training...")
    train_language_model(model, x_train, y_train, steps=250,
                         batch_size=32, learning_rate=3e-4)
    print()

    demo_batch = x_train[:2]
    demo_targets = y_train[:2]

    with torch.no_grad():
        logits, loss = model(demo_batch, demo_targets)
        prompt = "transformers "
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=120, temperature=0.9)
        generated_text = tokenizer.decode(generated_ids[0].tolist())

    print("demo batch shape:", demo_batch.shape)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))
    print("generated tokens:", generated_ids)
    print("generated text:")
    print(generated_text)
