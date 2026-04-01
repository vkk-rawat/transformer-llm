LLMs have gotten a major breakthrough with the help of the now popular Transformer architecture. This repository now includes a small, working decoder-only Transformer implementation based on the GPT-style architecture.

## What is included

- Causal self-attention with a triangular mask
- Multi-head attention
- Feed-forward MLP blocks
- Residual connections and LayerNorm
- Autoregressive token generation

## Files

- [transformer.py](transformer.py): model implementation and a small demo

## Install

```bash
pip install torch
```

## Run

```bash
python transformer.py
```

## Model shape
