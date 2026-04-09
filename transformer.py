import argparse
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Optional

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]


DEFAULT_TRAINING_TEXT = (
    "transformers are sequence models. "
    "large language models use transformer blocks for next token prediction. "
    "attention helps the model focus on relevant previous tokens. "
) * 30


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


def split_train_val(
    x: torch.Tensor,
    y: torch.Tensor,
    val_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0.0, 1.0)")

    sample_count = x.size(0)
    if sample_count < 2 or val_ratio == 0.0:
        return x, y, x[:0], y[:0]

    val_size = max(1, int(sample_count * val_ratio))
    if val_size >= sample_count:
        val_size = sample_count - 1

    return x[:-val_size], y[:-val_size], x[-val_size:], y[-val_size:]


@torch.no_grad()
def evaluate_language_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")

    was_training = model.training
    model.eval()
    _, loss = model(x, y)
    if was_training:
        model.train()

    if loss is None:
        raise RuntimeError("Expected loss when evaluating language model")
    return float(loss.detach())


def train_language_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
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
            train_loss = float(loss.detach())
            message = f"step {step:4d}/{steps}  train_loss={train_loss:.6f}"

            if x_val is not None and y_val is not None and x_val.numel() > 0 and y_val.numel() > 0:
                val_loss = evaluate_language_model(model, x_val, y_val)
                if math.isfinite(val_loss):
                    perplexity = math.exp(min(val_loss, 20.0))
                    message += f"  val_loss={val_loss:.6f}  val_ppl={perplexity:.3f}"

            print(message)

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


def load_training_text(path: Optional[str]) -> str:
    if not path:
        return DEFAULT_TRAINING_TEXT

    text_file = Path(path)
    if not text_file.exists():
        raise FileNotFoundError(f"Training text file not found: {text_file}")

    text = text_file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Training text file is empty")
    return text


def save_checkpoint(
    model: TransformerLM,
    tokenizer: CharTokenizer,
    config: TransformerConfig,
    checkpoint_path: str,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "vocab": tokenizer.vocab,
    }
    torch.save(checkpoint, path)
    print(f"checkpoint saved: {path}")


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TransformerLM, CharTokenizer, TransformerConfig]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    config = TransformerConfig(**checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["vocab"])
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"checkpoint loaded: {path}")
    return model, tokenizer, config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and sample from a small decoder-only Transformer LM",
    )
    parser.add_argument("--text-file", type=str, default=None,
                        help="Path to a UTF-8 text file for training data")
    parser.add_argument("--steps", type=int, default=250,
                        help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="AdamW learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio in [0.0, 1.0)")

    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--prompt", type=str,
                        default="transformers ", help="Prompt for generation")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.9)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-checkpoint", type=str, default=None,
                        help="Path to save model checkpoint after training")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to load model checkpoint before training")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and only run generation")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)

    training_text = load_training_text(args.text_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load_checkpoint:
        model, tokenizer, config = load_checkpoint(
            args.load_checkpoint, device)
    else:
        tokenizer = CharTokenizer.from_text(training_text)
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        )
        model = TransformerLM(config).to(device)

    print_model_report(model)
    print()

    x_all, y_all = build_training_sequences(
        training_text, tokenizer, config.block_size)
    x_train, y_train, x_val, y_val = split_train_val(
        x_all, y_all, val_ratio=args.val_ratio)

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    print(f"device: {device}")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"training sequences: {x_train.size(0)}")
    print(f"validation sequences: {x_val.size(0)}")
    print()
    if args.skip_training:
        print("training skipped (--skip-training set)")
    else:
        print("training...")
        train_language_model(
            model,
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    if args.save_checkpoint:
        save_checkpoint(model, tokenizer, config, args.save_checkpoint)

    print()

    demo_batch = x_train[:2] if x_train.size(0) >= 2 else x_all[:2].to(device)
    demo_targets = y_train[:2] if y_train.size(
        0) >= 2 else y_all[:2].to(device)

    with torch.no_grad():
        logits, loss = model(demo_batch, demo_targets)

        prompt = args.prompt
        try:
            prompt_tokens = tokenizer.encode(prompt)
        except ValueError:
            fallback_prompt = training_text[: min(16, len(training_text))]
            print(
                "prompt contains unknown characters for this tokenizer; "
                f"using fallback prompt: {fallback_prompt!r}"
            )
            prompt = fallback_prompt
            prompt_tokens = tokenizer.encode(prompt)

        prompt_ids = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=device)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        generated_text = tokenizer.decode(generated_ids[0].tolist())

    print("demo batch shape:", demo_batch.shape)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))
    print("generated tokens:", generated_ids)
    print("generated text:")
    print(generated_text)
