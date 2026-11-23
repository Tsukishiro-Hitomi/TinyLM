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

## Notes

- PPL is token-level perplexity, `exp(val_loss)` on the fixed validation split.
- Partial runs (steps < 40k) are reported as-is and are not directly comparable
  to full-length runs.
- The batch sweep uses a fixed number of sequences; compare within the sweep,
  not against the 40k-step baseline.
