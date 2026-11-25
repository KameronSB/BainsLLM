from bainsllm.config import load_config
from bainsllm.data.dataset import TokenDataset

cfg = load_config("configs/tiny.yaml")
text = "banana banana\n" * 100

dataset = TokenDataset.from_text(
    text,
    train_split=cfg.data.train_split,
    kind=cfg.data.tokenizer,
    vocab_size=cfg.data.vocab_size,
)

print("Tokenizer used:", type(dataset.tokenizer).__name__)
print("Requested vocabulary size:", cfg.data.vocab_size)
print("Actual vocabulary size:", dataset.vocab_size)