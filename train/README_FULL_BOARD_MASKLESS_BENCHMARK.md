# 15x15 full-board maskless training report

This report records the implementation, correctness checks, and measured
throughput of the full-board maskless training path added in commit `955a992`.
It is deliberately separate from `README_TRAINING_THROUGHPUT.md`: the latter
describes all generic training optimizations, while this document isolates the
effect of removing masks and changing the training activation layout.

## Benchmark environment

- two NVIDIA RTX 4090 D GPUs, using physical devices 0 and 3;
- PyTorch 2.12.1 with CUDA 13.0;
- FP16 AMP and two-process DDP;
- 15x15 Gomoku data;
- default `torch.compile` mode;
- model compilation, loss compilation, DDP, Muon, and all other runtime
  optimizations held constant between each masked/maskless comparison;
- the first 100-step timing interval excluded because it includes compilation.

The two representative models were:

- `b24c256h8tflrs-bng-silu-v102`, batch 416 per GPU;
- `b40c384h12tflrs-bng-silu-v102`, batch 172 per GPU.

## Full-board dataset

`filter_full_board_npz.py` filtered every full 15x15 row into a separate data
tree. Of 3,013,632 input rows, 2,703,710 were retained:

```text
retained fraction = 89.715997%
removed rows      = 309,922
output NPZ files  = 11
validation files  = 0
```

The source tree was opened read-only and was not modified. The filter:

- applies one row selection to every row-aligned NPZ field, including unknown
  future fields;
- checks the exact packed spatial width and ignores only the seven padding bits
  after the 225th board point;
- updates `num_rows` and `num_batches` sidecars;
- writes and verifies a sibling staging tree before atomically publishing it;
- refuses to replace an existing destination or publish zero training rows;
- writes `full_board_filter_manifest.json` and supports `--verify-only`.

Example:

```bash
cd train
python filter_full_board_npz.py /path/to/shuffled /path/to/shuffled_full15 \
  --pos-len 15 --workers 2
python filter_full_board_npz.py --verify-only /path/to/shuffled_full15 \
  --pos-len 15
```

The loader independently validates every NPZ before transferring its first
batch. If any row is not full-board, `-disable-mask` fails with the filename,
invalid-row count, and first invalid row index.

## Implementation

### Model and loss fast paths

`-disable-mask` is fixed for the entire training process so that
`torch.compile` sees a static graph. On the training path it:

- passes `mask=None` to the model;
- passes a true `attn_mask=None` to scaled dot-product attention;
- avoids constructing the spatial mask, mask sum, and policy mask in the loss;
- uses the fixed board area for ownership, scoring, future-position, and seki
  loss normalization.

Validation remains masked and accepts an unfiltered validation set. Full-one
masks and no masks implement the same equations, although fused kernels and
reduction order make AMP results non-bitwise-identical.

### Channels-last training activations

Maskless training automatically stores only `binaryInputNCHW` in channels-last
memory format. Parameters, optimizer tensors, value targets, ordinary
validation, and SWA validation retain their existing layouts.

For b24, the logical BSC view at the start of every Transformer block changes
from:

```text
NCHW storage:          stride (57600, 1, 225)
channels-last storage: stride (57600, 256, 1)
```

The original `view` and `permute` operations are metadata-only and can be
removed by the compiler, but that does not make the channel dimension
contiguous. Channels-last allows RMSNorm, QKV projections, and FFN projections
to consume a BSC tensor whose final dimension has stride 1, and the residual
path preserves that layout across all Transformer blocks.

Runtime behavior is:

```text
environment variable absent:
  -disable-mask absent -> NCHW
  -disable-mask present -> channels-last

KATAGO_INPUT_CHANNELS_LAST=0 -> force NCHW
KATAGO_INPUT_CHANNELS_LAST=1 -> force channels-last
```

The optional attention-pool implementation was also made layout-safe by using
`reshape` where merging batch and split-head dimensions cannot be represented
as a channels-last view. The legacy `TransformerBlock` also passes
`key_padding_mask=None` when masking is disabled.

### SDPA backend control

`KATAGO_SDPA_BACKEND=auto|flash|cudnn|efficient|math` can force one backend for
diagnostic benchmarks. The production default remains `auto`.

## Attention microbenchmark

These are eager BF16 forward-plus-backward SDPA timings on one RTX 4090 D. They
measure attention only, not a full training step.

| Model shape | Backend and mask | Time | Relative to additive mask |
|---|---|---:|---:|
| b24: B416, H8, S225, D32 | auto, `attn_mask=None` (Flash) | 1.797 ms | 1.71x faster |
| b24 | forced Flash, no mask | 1.797 ms | 1.71x faster |
| b24 | forced cuDNN, no mask | 2.608 ms | 1.18x faster |
| b24 | forced Efficient, no mask | 2.831 ms | 1.09x faster |
| b24 | auto, additive mask (Efficient) | 3.079 ms | baseline |
| b40: B172, H12, S225, D32 | auto, `attn_mask=None` (Flash) | 1.135 ms | 1.70x faster |
| b40 | forced Flash, no mask | 1.134 ms | 1.70x faster |
| b40 | forced cuDNN, no mask | 1.615 ms | 1.19x faster |
| b40 | forced Efficient, no mask | 1.772 ms | 1.09x faster |
| b40 | auto, additive mask (Efficient) | 1.924 ms | baseline |

PyTorch `auto` already selected the fastest measured backend, so forcing Flash
is unnecessary. The approximately 1.7x attention-only gain becomes a smaller
end-to-end gain because attention is only one part of a training step.

## End-to-end DDP results

All rates below are stable intervals after excluding the first compilation
interval. "Model-only maskless" predates the maskless loss specialization and
is included to separate the two effects.

| Model and path | Batch/GPU | Stable intervals | Mean interval | Samples/s | Versus masked |
|---|---:|---:|---:|---:|---:|
| b24 masked NCHW | 416 | 4 | 29.845 s / 83,200 | 2,787.7 | baseline |
| b24 model-only maskless NCHW | 416 | 4 | 27.595 s / 83,200 | 3,015.0 | +8.15% |
| b24 model+loss maskless NCHW | 416 | 7 | 27.414 s / 83,200 | 3,034.9 | +8.87% |
| b24 maskless channels-last | 416 | 6 | 23.755 s / 83,200 | 3,502.4 | +25.64% |
| b24 maskless channels-last | 424 | 4 | 24.313 s / 84,800 | 3,487.9 | +25.12% |
| b40 masked NCHW | 172 | 9 | 36.518 s / 34,400 | 942.0 | baseline |
| b40 model+loss maskless NCHW | 172 | 6 | 33.462 s / 34,400 | 1,028.0 | +9.13% |
| b40 maskless channels-last | 172 | 6 | 29.380 s / 34,400 | 1,170.9 | +24.29% |

The stable interval coefficient of variation was below 0.9% for the final
channels-last runs. The controlled conclusions are:

- removing model and loss masks improves full-step throughput by about 9%;
- channels-last adds another 13.9% to 15.4% over maskless NCHW;
- the complete maskless path is 24.3% to 25.6% faster than the masked control.

Batch 424 was 0.4% slower than batch 416 for b24 and used about 0.65 GiB more
memory, so batch 416 is preferred. An earlier maskless-NCHW batch-432 attempt ran
out of memory by approximately 72 MiB; channels-last batch 432 was not tested.
For b40, batch 172 was stable; a batch-176 attempt reached approximately 24.05
GiB on one rank but encountered a shape-specific Inductor compilation long tail
and was stopped after 12 minutes without producing a training interval. It is
not counted as a throughput result.

## Correctness and integration checks

- 81 local unit tests passed.
- Full-one masks and `mask=None` were compared for numerical agreement of
  outputs and gradients.
- Full-one-masked and maskless spatial losses were compared for numerical
  agreement of values and gradients.
- A small complete Transformer model matched between NCHW and channels-last for
  outputs, input gradients, global-input gradients, and parameter gradients,
  with both masked and maskless forwards.
- The optional attention-pool path matched for NCHW and channels-last outputs
  and gradients.
- The filter was tested for row alignment, dtype/shape preservation, source
  immutability, atomic publication, packed padding bits, malformed inputs,
  existing destinations, zero retained rows, and verify-only operation.
- Compiled b24 and b40 two-GPU runs completed stable intervals without DDP
  parameter/bucket stride warnings.
- A final two-GPU smoke test confirmed that `-disable-mask` automatically selects
  channels-last when the environment override is absent.
- The remote filtered tree passed `--verify-only` with 2,703,710 rows.

## Recommendation and limitations

Use `-disable-mask` for a training tree that is guaranteed to contain only full
boards. Recommended starting batches for the measured 24 GiB cards are 416 for
b24 and 172 for b40. Use `KATAGO_INPUT_CHANNELS_LAST=0` only for regression
comparison or if a previously untested custom block has a layout constraint.

The measurements are specific to two RTX 4090 D cards, PyTorch 2.12.1, FP16,
15x15 sequence length 225, and the listed model shapes. They do not establish
the same percentage on another GPU generation or sequence length. Randomly
initialized benchmark runs were used for throughput, so their loss curves are
not model-quality comparisons; no long-convergence or final-playing-strength
A/B test was performed. The third-party `flash-attn` package and Transformer
Engine/FP8 were not tested here. All masked/maskless timing comparisons held
the Muon optimizer implementation and its settings constant.
