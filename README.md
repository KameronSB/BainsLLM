# BainsLLM

BainsLLM is a from-scratch experimental decoder-only transformer lab built in PyTorch.

The goal is to rebuild a small GPT-style language model from first principles, then use the repo to experiment with tokenizers, architecture changes, training behaviour, and inference optimisations.

This is a fresh project, not a refactor of my earlier GPT-1-style repo.

## Current Status

Early scaffold stage.

Implemented:

- repository skeleton
- Python package layout
- ROCm/PyTorch smoke test
- data/model/training folder structure
- initial tiny config

Not implemented yet:

- tokenizer
- dataset loader
- attention
- transformer block
- full language model
- training loop
- generation
- architecture variants

## Project Goals

BainsLLM will eventually include:

- decoder-only transformer
- causal self-attention
- multi-head attention
- MLP blocks
- LayerNorm
- residual connections
- character tokenizer baseline
- BPE tokenizer
- byte-level BPE tokenizer
- RoPE
- pre-LayerNorm architecture
- KV cache for inference
- configurable experiments
- training instrumentation
- dataset comparisons

The priority is understanding and clean engineering, not scale.

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

#Milestones
##Milestone 0 — Environment and ROCm

Prove PyTorch can see and use the AMD GPU through ROCm.

Command:

python scripts/smoke_test_rocm.py

Success means:

torch.cuda.is_available() is True
device is the AMD GPU
tensors are created on cuda
matmul runs on GPU
backward pass works on GPU

##Milestone 1 — Config System

Create dataclass/YAML-based config loading for repeatable experiments.

##Milestone 2 — Tokenizer Baseline

Implement a character-level tokenizer first.

##Milestone 3 — Dataset Loader

Create next-token prediction batches:

x = tokens[t : t + block_size]
y = tokens[t + 1 : t + block_size + 1]

##Milestone 4 — Causal Self-Attention

Implement scaled dot-product attention with a causal mask.

##Milestone 5 — Multi-Head Attention

Add multiple attention heads and output projection.

##Milestone 6 — Transformer Block

Add LayerNorm, MLP, residual connections, and dropout.

##Milestone 7 — Full Decoder-Only Model

Assemble embeddings, transformer blocks, final norm, and LM head.

##Milestone 8 — Training Loop

Train the first tiny model and confirm the loss decreases.

##Milestone 9 — Text Generation

Sample from the trained model.

##Milestone 10 — Architecture Experiments

Add and compare:

pre-LayerNorm
RoPE
SwiGLU
KV cache

##Milestone 11 — Tokenizer Experiments

Compare:

character tokenizer
BPE
byte-level BPE
Design Principle

Build everything slowly, test each component, and only add complexity after the baseline works.
EOF


## 10. Test the package import

```bash
python -c "import bainsllm; print('bainsllm import OK')"