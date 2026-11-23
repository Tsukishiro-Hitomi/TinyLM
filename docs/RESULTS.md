# Results

Trained checkpoints and experiment sweeps from CS336 assignment 1, reproduced
with the `tinylm` library. Unless noted, every run uses the same 22.7M-parameter
model, seed 336, and the TinyStories corpus (10k BPE vocabulary).

## Setup

| Item | Value |
| --- | --- |
| Model | 22.7M params; `d_model=512`, 16 heads, 4 layers, `d_ff=1344`, RoPE θ=10000, context 256 |
| Optimizer | AdamW (from-scratch), weight decay 0.1 |
| Schedule | cosine decay with 5% linear warmup |
| Data | TinyStories (bin-packed `uint16`); OpenWebText for the cross-domain run |
| Hardware | 1× NVIDIA A800-80GB |

## Baseline

| Run | Steps | Train loss | Val loss | PPL |
| --- | ---: | ---: | ---: | ---: |
| `20261028-080941` | 40,000 | 1.432 | **1.375** | **3.96** |

![Training loss curve](figures/main_loss_curve.png)

Sample continuation from `checkpoint40000` (`temperature=0.8, top_p=0.9`):

> Once upon a time, there was a little girl named Lily. She loved to eat
> apples. One day, she went to the park with her mom. They saw a big tree with
> lots of apples. Lily wanted to pick some apples, but her mom said, "Be careful,
> it's not good to eat."

## Learning-rate sweep

All runs at 40k steps except the last two, which diverged within 1k steps.

| Learning rate | Steps | Val loss | PPL |
| --- | ---: | ---: | ---: |
| 1e-4 | 40,000 | 1.598 | 4.94 |
| **1e-3** | 40,000 | **1.378** | **3.97** |
| 3e-3 | 40,000 | 1.384 | 3.99 |
| 1e-2 | 40,000 | 1.486 | 4.42 |
| 3e-2 | 40,000 | 1.866 | 6.46 |
| 1e-1 | 40,000 | 2.132 | 8.43 |
| 3e-1 | 1,000 | 5.322 | 204.85 |
| 1.0 | 500 | 5.901 | 365.46 |

![Validation loss vs steps across learning rates](figures/lr_validation_vs_steps.png)

## Batch-size sweep

These runs are **token-equalized** (≈40k sequences each), so a larger batch means
fewer optimizer steps. Loss reflects that trade-off, not a longer baseline run.

| Batch size | Steps | Val loss | PPL |
| ---: | ---: | ---: | ---: |
| 1 | 40,000 | 2.021 | 7.55 |
| 8 | 5,000 | 1.974 | 7.20 |
| 32 | 1,250 | 2.071 | 7.93 |
| 64 | 625 | 2.227 | 9.27 |
| 128 | 313 | 2.519 | 12.42 |
| 256 | 157 | 2.962 | 19.33 |
| 512 | 79 | 3.481 | 32.50 |

![Validation loss vs tokens processed](figures/batch_validation_vs_tokens.png)

## Ablation

Baseline is pre-norm + RoPE + SwiGLU (val 1.375).

| Variant | Steps | Val loss | Δ PPL |
| --- | ---: | ---: | ---: |
| Post-norm instead of pre-norm | 40,000 | 1.378 | +0.01 |
| SiLU FFN instead of SwiGLU | 40,000 | 1.387 | +0.04 |
| No positional encoding (no RoPE) | 40,000 | 1.430 | +0.22 |
| No normalization (low LR) | 40,000 | 1.670 | +1.35 |
| No normalization (default LR) | 8,500 | 1.850 | — (stopped early) |

| Normalization placement | Activation function | Position encoding |
| --- | --- | --- |
| ![normalization](figures/ablation_normalization.png) | ![activation](figures/ablation_activation.png) | ![position](figures/ablation_position.png) |

RoPE has the largest single-component effect among the positional/normalization
ablations, consistent with the small-scale setting where the model relies on
positional structure to fit TinyStories.

## OpenWebText (cross-domain)

| Run | Vocab | Steps | Train loss | Val loss | PPL |
| --- | ---: | ---: | ---: | ---: | ---: |
| `owt_baseline_seed336` | 32k | 40,000 | 3.972 | 3.939 | 51.36 |
| `owt_tied_seed336` | 32k | 20,004 | 4.131 | 4.085 | 59.42 |

Web text is markedly harder than TinyStories, as expected from the higher
entropy and much larger vocabulary.

## Evaluation beyond loss

Perplexity depends on the tokenizer's vocabulary size, so it is **not comparable
across tokenizers**. Bits-per-byte (BPB) divides the negative log-likelihood by
the number of UTF-8 bytes in the targets and is tokenizer-independent.

| Model | Data | Vocab | Val loss | PPL | **BPB** |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyStories | TinyStories valid | 10k | 1.431 | 4.18 | **0.4996** |
| OpenWebText | OWT valid | 32k | 3.919 | 50.34 | **1.2929** |

The PPL ratio (4.18 vs 50.34, ≈12×) overstates the gap because the OWT tokenizer
has a 3.2× larger vocabulary; in bits-per-byte the gap is ≈2.6×, which is the
honest cross-domain difficulty. Reproduce with `scripts/evaluate.py`.

### Generation quality

`scripts/eval_generation.py` samples 8 prompts × 3 continuations (120 tokens,
temperature 0.8, top-p 0.9):

| Metric | Value |
| --- | ---: |
| distinct-1 | 0.267 |
| distinct-2 | 0.676 |
| distinct-3 | 0.881 |
| repetition-4 | 0.044 |
| unique samples | 24 / 24 |

Repetition is low and all 24 continuations are unique. Samples are written to
`docs/generations.jsonl` for optional LLM-as-judge scoring.

## Inference performance

KV cache and Flash Attention (via `F.scaled_dot_product_attention`) are both
implemented; see `scripts/benchmark_inference.py`.

| Change | Effect |
| --- | --- |
| `einops.einsum` → `matmul` in the hot path | ~1.9× generation throughput (151 → 297 tok/s) |
| SDPA (Flash Attention) vs manual attention | ~1.25× |
| KV cache vs full-prefix recompute | 1.1–1.2× end-to-end |

The KV cache is algorithmically O(L) instead of O(L²) per step (≈66× fewer FLOPs
at context 128), but wall-clock is only ~1.1× here because a 22M model at batch 1
is **CPU-dispatch-bound**: a single forward takes ≈3.2 ms whether it processes 1
or 128 tokens, so the GPU stays idle (0% utilization) waiting on launches. The
compute win would surface with a larger model, batching, or kernel fusion
(`torch.compile` / CUDA graphs).

## Notes

- PPL is token-level perplexity, `exp(val_loss)` on the fixed validation split.
- Partial runs (steps < 40k) are reported as-is and are not directly comparable
  to full-length runs.
- The batch sweep uses a fixed number of sequences; compare within the sweep,
  not against the 40k-step baseline.
