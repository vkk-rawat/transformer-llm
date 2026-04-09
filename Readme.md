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

The script prints the model architecture, a compact weight summary, and a small synthetic forward pass in the terminal.

It now also includes a tiny character tokenizer and a short training loop on built-in text so you can see loss logs and generated text output directly in the terminal.

## Model shape

The implementation is a decoder-only Transformer for next-token prediction. If you want, the next step can be turning this into:

- a full encoder-decoder Transformer
- a training script on text data
- a tokenizer pipeline
- a smaller educational version with more comments

Reference video followed:
https://www.youtube.com/watch?v=p3sij8QzONQ
