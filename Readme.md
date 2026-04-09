# Transformer-LLM

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GitHub stars](https://img.shields.io/github/stars/vkk-rawat/transformer-llm?style=social)](https://github.com/vkk-rawat/transformer-llm/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/vkk-rawat/transformer-llm)](https://github.com/vkk-rawat/transformer-llm/commits/main)

A compact, educational decoder-only Transformer language model built with PyTorch.

This project demonstrates a full mini AI/ML workflow in a single script:

- tokenizer creation from raw text
- sequence dataset preparation for next-token prediction
- GPT-style Transformer implementation
- short training loop with loss logging
- autoregressive text generation

It is designed for learning and experimentation, not for production-scale training.

## Out-of-Scope Implementation Branch

This implementation is available on the feature branch:

- `feature/out-of-scope-checkpointing`

Added out-of-scope capabilities in this branch:

- command-line configurable training and model hyperparameters
- optional external text dataset input (`--text-file`)
- train/validation split with validation loss + perplexity logging
- model checkpoint save/load support
- optional generation-only mode (`--skip-training`)

## Project Overview

The model is a decoder-only Transformer (GPT-style) trained on a small built-in text corpus.
It predicts the next character token given previous tokens.

Implemented components:

- token embedding + positional embedding
- multi-head causal self-attention with triangular mask
- feed-forward MLP blocks with GELU
- residual connections and LayerNorm
- dropout regularization
- language modeling head for logits over vocabulary

## Repository Structure

```text
transformer-llm/
|-- transformer.py      # model, tokenizer, data prep, training, generation
|-- requirements.txt    # Python dependencies
|-- Readme.md           # project documentation
```

## Tech Stack

- Python
- PyTorch
- NumPy (listed in requirements)

## Installation

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python transformer.py
```

Advanced run examples:

```bash
# Train with custom steps and save a checkpoint
python transformer.py --steps 500 --learning-rate 3e-4 --save-checkpoint checkpoints/model.pt

# Load checkpoint and generate only
python transformer.py --load-checkpoint checkpoints/model.pt --skip-training --prompt "transformers " --max-new-tokens 150

# Train from your own text file
python transformer.py --text-file data/corpus.txt --val-ratio 0.1
```

## What the Script Does

When you run `transformer.py`, it executes the following pipeline:

1. Build a character-level tokenizer from training text.
2. Create input/target sequences of length `block_size` for next-token training.
3. Initialize a decoder-only Transformer model.
4. Print model architecture and parameter statistics.
5. Train for a short number of steps using AdamW.
6. Optionally track validation loss/perplexity during training.
7. Optionally save/load checkpoints.
8. Generate text from a prompt using autoregressive sampling.

## Model Configuration (Default)

The default configuration is defined in `TransformerConfig`:

- `n_embd = 128`
- `n_head = 4`
- `n_layer = 4`
- `block_size = 32`
- `dropout = 0.1`

## Example Console Output

Output will vary due to random sampling and hardware, but follows this pattern:

```text
model architecture:
TransformerLM(...)

parameter summary:
- token_embedding.weight: shape=(..., 128), params=..., mean=..., std=...
...
total parameters: ...
trainable parameters: ...

device: cpu
vocab size: ...
training sequences: ...

training...
step    1/250  train_loss=...  val_loss=...  val_ppl=...
step   25/250  train_loss=...  val_loss=...  val_ppl=...
...
step  250/250  train_loss=...  val_loss=...  val_ppl=...

demo batch shape: torch.Size([2, 32])
logits shape: torch.Size([2, 32, vocab_size])
loss: ...
generated text:
transformers ...
```

## Implementation Notes

- Causal masking prevents attending to future tokens during training.
- The `generate` method uses temperature-scaled softmax + multinomial sampling.
- The training data is intentionally small for fast local experimentation.

## Limitations

- Character-level tokenizer only (no subword/BPE tokenizer).
- Tiny built-in dataset and short training duration.
- No experiment tracking dashboard (TensorBoard/W&B).
- No distributed or mixed-precision large-scale training.

## How to Extend This Project

Potential next improvements:

- switch to token-level tokenizer (BPE/WordPiece)
- add unit tests for tokenizer and data pipeline
- add benchmark scripts and reproducible experiment configs

## Reference

Video inspiration:
https://www.youtube.com/watch?v=p3sij8QzONQ
