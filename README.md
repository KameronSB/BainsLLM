# BainsLLM

BainsLLM is a from-scratch experimental decoder-only transformer lab built in PyTorch.

The goal is to rebuild a small GPT-style language model from first principles, then use the repo to experiment with tokenizers, architecture changes, training behaviour, and inference optimisations.

This is a fresh project, not a refactor of my earlier GPT-1-style repo.

## Current Structure

```text
BainsLLM/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── runs/
├── scripts/
│   └── smoke_test_rocm.py
├── src/
│   └── bainsllm/
│       ├── data/
│       ├── model/
│       │   └── architectures/
│       ├── training/
│       ├── eval/
│       └── utils/
└── tests/
```

## Milestones

### Milestone 0 — Environment and ROCm

Prove PyTorch can see and use the AMD GPU through ROCm.

Command:

```bash
python scripts/smoke_test_rocm.py
```

Success means:

- `torch.cuda.is_available()` is `True`
- the selected device is the AMD GPU
- tensors are created on `cuda`
- matmul runs on GPU
- backward pass works on GPU

### Milestone 1 — Config System

Create dataclass/YAML-based config loading for repeatable experiments.

This avoids hard-coding model and training settings throughout the codebase.

### Milestone 2 — Tokenizer Baseline

Implement a tokenizer interface and a character-level tokenizer first.

The character tokenizer is the simplest baseline. It lets the model train before adding BPE complexity.

Later tokenizer upgrades:

- BPE tokenizer
- byte-level BPE tokenizer

### Milestone 3 — Dataset Loader

Create next-token prediction batches.

```text
x = tokens[t : t + block_size]
y = tokens[t + 1 : t + block_size + 1]
```

The model receives `x` and learns to predict `y`.

### Milestone 4 — Dataset Shift Test

Test that `y` is really `x` shifted by one token.

Example check:

```python
assert torch.equal(y[:, :-1], x[:, 1:])
```

This is a critical test because next-token prediction depends on the target being shifted correctly.

### Milestone 5 — Causal Self-Attention

Implement scaled dot-product self-attention with a causal mask.

The causal mask prevents the model from looking at future tokens.

### Milestone 6 — Multi-Head Attention

Add multiple attention heads and an output projection.

Multi-head attention lets the model learn different token relationships in parallel.

### Milestone 7 — Transformer Block

Add:

- LayerNorm
- multi-head causal self-attention
- MLP
- residual connections
- dropout

Initial block design:

```text
x = x + attention(layer_norm(x))
x = x + mlp(layer_norm(x))
```

### Milestone 8 — Full Decoder-Only Model

Assemble:

- token embeddings
- positional embeddings
- transformer blocks
- final LayerNorm
- language modelling head

The model should return:

```python
logits, loss = model(x, y)
```

### Milestone 9 — Training Loop

Train the first tiny model and confirm the loss decreases.

The training loop should include:

- AdamW
- evaluation loop
- gradient clipping
- checkpoint saving
- periodic sample generation

### Milestone 10 — Text Generation

Generate text from the trained model.

This proves the model can be used autoregressively.

### Milestone 11 — Tokenizer Experiments

Compare:

- character tokenizer
- BPE tokenizer
- byte-level BPE tokenizer

The goal is to measure how tokenization changes sequence length, training speed, and generation quality.

### Milestone 12 — Architecture Experiments

Add and compare:

- pre-LayerNorm
- RoPE
- SwiGLU
- KV cache

## Design Principle

Build everything slowly, test each component, and only add complexity after the baseline works.