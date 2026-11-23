"""TinyLM: a compact, from-scratch implementation of a modern language model.

This package implements byte-pair tokenization, a RoPE + SwiGLU + RMSNorm
Transformer language model, its training loop and optimizers, and checkpointed
text generation. It is intended as a readable, self-contained reference for how
a small LLM is built end to end.
"""

import importlib.metadata

from .config import ModelConfig, TokenizerConfig, TrainConfig
from .model import (
    Embedding,
    Linear,
    MultiheadSelfAttention,
    RMSNorm,
    RoPE,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    infer,
    predict,
    scaled_dot_product_attention,
    silu,
    softmax,
)
from .tokenization import NUM_CHUNKS, PAT, Tokenizer, bpe_train, bpe_train_naive, run_bpe_train
from .training import (
    AdamW,
    SGD,
    ExperimentLogger,
    cross_entropy,
    evaluate,
    get_batch,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    plot_loss_curve,
    save_checkpoint,
    train,
)

try:
    __version__ = importlib.metadata.version("tinylm")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # config
    "ModelConfig",
    "TrainConfig",
    "TokenizerConfig",
    # layers
    "Embedding",
    "Linear",
    "RMSNorm",
    # attention
    "RoPE",
    "MultiheadSelfAttention",
    "scaled_dot_product_attention",
    "softmax",
    # transformer
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "silu",
    # inference
    "infer",
    "predict",
    # tokenization
    "NUM_CHUNKS",
    "PAT",
    "Tokenizer",
    "bpe_train",
    "bpe_train_naive",
    "run_bpe_train",
    # training
    "AdamW",
    "SGD",
    "ExperimentLogger",
    "cross_entropy",
    "evaluate",
    "get_batch",
    "gradient_clipping",
    "learning_rate_schedule",
    "load_checkpoint",
    "plot_loss_curve",
    "save_checkpoint",
    "train",
]
