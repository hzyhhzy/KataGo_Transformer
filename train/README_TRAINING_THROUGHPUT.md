# Training throughput changes

The default `train_muon_ki.py` path enables a set of generic optimizations for
compiled FP16/BF16 training:

- learned RoPE keeps `sin`/`cos` in FP32, then casts the small tables before the
  batch-sized rotation;
- model parameter norms are computed only on logging steps;
- the seki moving average stays on the model device;
- the postprocessed training loss is compiled separately from the model;
- Muon Newton-Schulz updates with identical matrix shapes are batched (up to 32
  by default), and auxiliary Adam updates use PyTorch foreach kernels;
- DDP uses a static graph and gradient bucket views, skips redundant per-forward
  buffer broadcasts for ordinary batch norm, and aligns 1x1 convolution weight
  strides with the layout produced by CUDA backward;
- distributed Muon parameters are synchronized in reusable flat buckets rather
  than one collective per parameter.

The distributed Muon layout supports arbitrary world sizes and parameters that
cross bucket boundaries. Rank 0 checkpoint ownership is preserved, so a
checkpoint can be resumed with a different world size as long as the parameter
groups and ordering are unchanged. Rank-0-only validation bypasses the DDP
wrapper, and rank-0 stop/retry decisions are broadcast so all workers follow the
same collective control flow.

## Measured result

The following controlled comparison was measured on two RTX 4090 D cards with
PyTorch 2.12.1, the same filtered full15 Gomoku data, FP16, and physical GPUs 0
and 3. Batch size is fixed within each model. Every rate uses timing windows 3
through 15 (13 stable 100-step windows), excluding two compilation and warm-up
windows. The baseline is an exact copy of `main` commit `6c8f0c8`.

| Model | Batch/GPU | Exact `main`, masked | Optimized, masked | Generic gain | Optimized, maskless CL | Maskless CL over optimized masked | Total over `main` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b24c256h8tflrs-bng-silu-v102` | 384 | 2,295.6 samples/s | 2,757.1 samples/s | +20.10% | 3,453.2 samples/s | +25.25% | +50.43% |
| `b40c384h12tflrs-bng-silu-v102` | 160 | 761.4 samples/s | 931.7 samples/s | +22.37% | 1,143.5 samples/s | +22.74% | +50.20% |

The maskless gain column includes both removal of masks and the NHWC
activation layout. The total column also includes all generic training and
optimizer changes; these percentages must not be added to one another again.

The highest observed throughput used more aggressive batches, but the fixed
batches leave about 1.5 GiB more memory headroom and are better long-run
starting points:

| Model | Recommended batch | Samples/s | Best observed batch | Samples/s | Throughput cost |
|---|---:|---:|---:|---:|---:|
| b24 | 384 | 3,453.2 | 416 | 3,502.4 | -1.40% |
| b40 | 160 | 1,143.5 | 172 | 1,170.9 | -2.33% |

The b24 batch-416 and b40 batch-172 runs used about 22.9 and 23.4 GiB per card.
A later b24 batch-416 repeat using the final Muon A/B/C protocol reached 3,475.4
samples/s, against which batch 384 is only 0.64% slower. Both comparisons remain
well inside the requested 5% threshold. These are benchmark-specific starting
points, not universal defaults: allocator state, validation, checkpointing,
drivers, and other processes can require a smaller batch.

### Single-GPU versus two-GPU scaling

The final maskless NHWC configuration was also measured with an
ordinary single process: no `-multi-gpus`, no process group, and no DDP wrapper.
The model, data, per-GPU batch, and runtime flags match the fixed-batch two-GPU
runs; global-batch-dependent training scalars naturally change with world size.
Rates again aggregate timing windows 3 through 15; a 100-step single-GPU window
contains 38,400 b24 or 16,000 b40 samples.

| Model | Batch/GPU | Single GPU | Two-GPU total | Speedup | Two-GPU efficiency | Peak `nvidia-smi` memory |
|---|---:|---:|---:|---:|---:|---:|
| b24 | 384 | 1,823.0 samples/s | 3,453.2 samples/s | 1.894x | 94.71% | 21,627 MiB |
| b40 | 160 | 637.2 samples/s | 1,143.5 samples/s | 1.795x | 89.73% | 21,821 MiB |

Both requested batches fit without adjustment. The two-GPU per-card throughput
penalty relative to single GPU is 5.29% for b24 and 10.27% for b40. This is
whole-training scaling rather than a pure DDP communication measurement: in
DDP, each rank computes only its assigned share of Muon matrix updates before
flat parameter synchronization, while a single process updates every Muon
matrix. DDP-only hooks, gradient buckets, and 1x1-convolution stride alignment
also change the execution path.

BF16 AMP and BF16 DDP gradient compression were also measured. BF16 AMP was
effectively tied with FP16; BF16 gradient compression improved b40 by only about
0.3% while changing reduction precision. FP16 remains the script default and
gradient communication remains FP32.

## Muon before/after benchmark

The optimizer was isolated with the same maskless NHWC model, data,
batch, compile mode, and two-GPU setup. As above, each rate is aggregate samples
divided by aggregate time over timing windows 3 through 15. The variants are:

- A: the `main` Muon file, with scalar Newton--Schulz, scalar auxiliary Adam,
  and the old owner-round synchronization (one `all_gather` per round, with at
  most one parameter owned by each rank in a round);
- B: reusable flat DDP synchronization, with batched Newton--Schulz and foreach
  Adam explicitly disabled;
- C: the current defaults, adding batched Newton--Schulz (groups of at most 32)
  and foreach auxiliary Adam to B.

| Model | A: old | B: flat + scalar kernels | C: current | B/A | C/B | C/A |
|---|---:|---:|---:|---:|---:|---:|
| b24, batch 384 | 2,912.6 | 2,964.4 | 3,453.2 | +1.78% | +16.49% | +18.56% |
| b40, batch 160 | 963.0 | 930.9 | 1,143.5 | -3.34% | +22.85% | +18.74% |

The flat implementation is designed and unit-tested for arbitrary world sizes,
different parameter shapes, and parameters crossing bucket boundaries, but is
not by itself a universal speedup: it helped b24 slightly and hurt b40 in this
test. The defensible performance conclusion is that the complete current Muon
path is about 18.6% to 18.7% faster than the old implementation on both
representative models. Variant C is the same observation as the final maskless
row in the fixed-batch table, so its gain is not additive with the total
training gain.

The update equations and checkpoint state schema are retained, but batched
Newton--Schulz and foreach Adam change floating-point operation order and are
not bitwise identical to their scalar versions. Unit tests cover scalar/default
checkpoint continuation in both directions, world-size 2/3 redistribution,
empty owners, heterogeneous shapes, and cross-bucket parameters. A real
two-process NCCL integration test isolating scalar flat synchronization also
completed six buckets successfully. Independent random initialization makes the
short-run losses useful only for detecting NaNs or crashes, not for comparing
training quality.

## Full-board mask-free training

See `README_FULL_BOARD_MASKLESS_BENCHMARK.md` for the complete implementation,
correctness, SDPA microbenchmark, and end-to-end measurement record.

When every sample uses the complete `pos_len` by `pos_len` board, the model and
loss can omit spatial masks. This lets CUDA SDPA select an unmasked fused
attention kernel and removes redundant mask operations from normalization,
pooling, heads, and spatial losses. Filter shuffled data into a separate tree:

```bash
python filter_full_board_npz.py /path/to/shuffled /path/to/shuffled_full15 \
  --pos-len 15 --workers 2
python filter_full_board_npz.py --verify-only /path/to/shuffled_full15 --pos-len 15
```

The filter reads the source only, applies the same row selection to every NPZ
field, ignores packed padding bits after the board, verifies the completed
staging tree, and atomically publishes the destination. It also updates training
sidecars and writes `full_board_filter_manifest.json`. In the benchmark dataset,
2,703,710 of 3,013,632 rows were full 15x15 boards (89.716%).

Pass `-disable-mask` when training on the filtered tree. Every NPZ is checked on
the CPU before its first batch is transferred; a single non-full-board row makes
the run fail with the filename and first bad row. The flag is fixed for the
whole training run so `torch.compile` sees one static graph. Validation keeps
the normal masked NCHW path. When on-load filtering is enabled, validation uses
the retained full-board rows by default; add
`-disable-validation-full-board-filter` to keep the original mixed validation
set. Full-one masks and no masks are mathematically equivalent, but fused
attention and reduction order mean AMP results are not bitwise identical.

As a convenience for a mixed source tree, add `-filter-full-board-on-load`
alongside `-disable-mask` to discard non-full-board training and validation rows
in memory.
Filtering is deterministic and applies the same row selection to every NPZ
field. Batching remains file-local: if a file retains fewer than
`batch_size * world_size` rows, rank 0 logs a warning and that file yields no
training batches. Offline filtering remains preferable when data-loading
throughput matters because it avoids repeatedly reading discarded rows.

Training stores the binary spatial input in NHWC format by default. This makes
the BSC view used by every Transformer block
contiguous in its channel dimension; model parameters, optimizer tensors,
targets, validation, and SWA validation keep their existing layouts. Set
`-input-memory-format nchw` for an NCHW compatibility or regression comparison.
NHWC applies to both masked and mask-free training.

The following historical effect-decomposition sweep uses the same filtered
data, FP16, default compile mode, and per-GPU batches 416 (b24) or 172 (b40).
The masked control still computes full-one masks. Each rate excludes the first
100-step compilation interval.

| Model | Masked NCHW | Mask-free NCHW | Mask-free NHWC | Versus optimized masked |
|---|---:|---:|---:|---:|
| `b24c256h8tflrs-bng-silu-v102` | 2,787.7 samples/s | 3,034.9 samples/s | 3,502.4 samples/s | 25.6% |
| `b40c384h12tflrs-bng-silu-v102` | 942.0 samples/s | 1,028.0 samples/s | 1,170.9 samples/s | 24.3% |

Removing model and loss masks contributed about 9%; NHWC added another
13.9--15.4% over mask-free NCHW. With a true `attn_mask=None`, PyTorch `auto`
selected Flash SDPA on both models; forcing another backend was slower. For b24,
batch 424 was 0.4% slower than batch 416 while using about 0.65 GiB more memory,
so 416 was the peak-throughput choice in that sweep. Batch 384 is the current
recommendation when retaining about 1.5 GiB of memory headroom matters.

## Learned RoPE layout benchmark

The current learned and fixed RoPE implementations use adjacent pairs:
channels `(0,1)`, `(2,3)`, and so on. A half-split layout pairs the first half
of each head with the second half. The layouts are related by a channel
permutation and have the same representational capacity, but may compile to
different rotation kernels.

One RTX 4090 D measured compiled FP16 forward plus backward for both the full
learned-RoPE chain and a complete production-shaped maskless Transformer block.
The effective batches match the recommended training batches. Medians are over
seven alternating trials for the chain and five for the block.

| Model | RoPE chain: adjacent / half | Half throughput gain | Full block: adjacent / half | Half throughput gain |
|---|---:|---:|---:|---:|
| b24, batch 384 | 1.2431 / 1.1254 ms | +10.45% | 7.8895 / 7.8531 ms | +0.46% |
| b40, batch 160 | 1.2767 / 1.2202 ms | +4.62% | 5.6146 / 5.5548 ms | +1.08% |

The b40 chain trials drifted substantially, so its isolated 4.62% number is not
a stable end-to-end claim. Both models passed FP32 output, input-gradient,
frequency-gradient, and all full-block parameter-gradient equivalence checks
after permuting Q/K projection rows; full-block peak allocated memory was
identical between layouts.

An end-to-end b24 check then measured the complete mask-free, NHWC,
FP16 Muon training loop with two RTX 4090 D GPUs, DDP, and batch 384 per GPU.
Eight independent runs used the balanced order `AB|BA|BA|AB`, where A was the
adjacent layout and B was half-split. Each run produced 56 logging windows of
100 steps; the first eight were discarded and the remaining 48 represented
3,686,400 samples. Functionally equivalent initial checkpoints differed only
by a per-head permutation of Q/K projection rows.

| Pair | Half-split throughput change |
|---:|---:|
| 1 (`AB`) | +0.1026% |
| 2 (`BA`) | -0.0821% |
| 3 (`BA`) | +0.1736% |
| 4 (`AB`) | +0.0512% |

The paired geometric mean was +0.0613%, with a two-sided 95% confidence
interval of -0.1104% to +0.2333%. Pooling all measured time gave 3,479.53
samples/s for adjacent and 3,481.66 samples/s for half-split, while GPU clocks,
power, temperature, and utilization were closely matched. A shorter fixed-RoPE
b24 ABBA check estimated +0.3300%, but its two-pair interval was likewise
inconclusive (-3.6625% to +4.4879%). The b40 end-to-end suite was not run after
the b24 result met the rejection criterion.

The layout conversion itself remained numerically sound: per-layer FP32 RMS
differences grew smoothly from floating-point dot-product reduction order, and
learned/fixed, MHA/GQA, masked/unmasked FP64 output and gradient checks had a
global maximum error of 2.66e-15. The issue is solely that the isolated kernel
gain is lost in complete training. Half-split is therefore not enabled in
production and no model-config switch is added.

Changing an existing model would also require an explicit layout field and a
per-head row permutation for every Q/K projection in the raw model, SWA/EMA,
and optimizer momentum, followed by ONNX re-export and TensorRT engine rebuild.
Old checkpoints must continue to default to adjacent layout; silently changing
the pairing would change attention despite identical tensor shapes. RoPE
frequencies, V projections, and output projections do not require conversion.

A separate short b24 control found that retaining angle/`sin`/`cos` computation
in FP32 but casting the small table before rotating the batch-sized FP16 Q/K
tensors improved end-to-end throughput by about 1.04%. That estimate has only
two stable baseline windows, so it is supporting evidence rather than a precise
standalone speedup. This table cast remains enabled by default.

## Runtime controls

The optimized settings are on by default. Set an environment variable to `0`
to disable an individual low-level optimization for debugging or regression
comparison:

| Environment variable | Default | Effect |
|---|---:|---|
| `KATAGO_LEARNED_ROPE_CAST_TO_INPUT_DTYPE` | `1` | Cast RoPE tables before batch-sized AMP rotation |
| `KATAGO_MODEL_NORMS_ONLY_AT_PRINT` | `1` | Compute model norms only when logging |
| `KATAGO_SEKI_EMA_ON_DEVICE` | `1` | Keep the seki moving average on device |
| `KATAGO_COMPILE_TRAINING_LOSS` | `1` | Compile postprocessed loss and metrics |
| `KATAGO_MUON_BATCHED_NS` | `1` | Batch same-shape Muon Newton-Schulz updates |
| `KATAGO_MUON_NS_BATCH_SIZE` | `32` | Maximum matrices in one Newton-Schulz batch |
| `KATAGO_AUX_ADAM_FOREACH` | `1` | Use foreach auxiliary Adam updates |
| `KATAGO_DDP_STATIC_GRAPH` | `1` | Enable DDP static-graph mode |
| `KATAGO_DDP_GRADIENT_AS_BUCKET_VIEW` | `1` | Make gradients views of DDP buckets |
| `KATAGO_DDP_BROADCAST_BUFFERS` | norm-dependent | Disabled for ordinary batch norm; enabled for batch renorm and QAT |
| `KATAGO_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES` | `1` | Match 1x1 convolution parameter and backward-gradient strides under DDP |

The commonly changed runtime choices are command-line arguments:

| Argument | Default | Choices/effect |
|---|---|---|
| `-compile-mode` | `default` | `default`, `max-autotune-no-cudagraphs`, or `max-autotune` |
| `-sdpa-backend` | `auto` | `auto`, `flash`, `cudnn`, `efficient`, or `math` |
| `-input-memory-format` | `nhwc` | `nhwc` or `nchw`; independent of masking |
| `-disable-flex-attention` | absent | Use masked SDPA instead of the default compatible Transformer FlexAttention path |
| `-disable-validation-full-board-filter` | absent | Keep mixed validation rows when on-load full-board filtering is enabled |

`-no-compile` disables both model and loss compilation. QAT also disables the
compiled loss. Muon's Newton-Schulz kernel retains its existing local
`torch.compile` implementation; `-no-compile` has historically controlled the
model rather than that optimizer kernel. If either model-norm deferral or the
on-device seki average is disabled, set `KATAGO_COMPILE_TRAINING_LOSS=0` as
well.

`-compile-mode` accepts `default`, `max-autotune-no-cudagraphs`, or
`max-autotune`. The main table uses `default`. On an earlier masked-NCHW b24
batch-416 path,
`max-autotune-no-cudagraphs` took about 757 seconds for the first 100-step
interval, then sustained about 2,990--3,023 samples/s (roughly another 8% over
the default compiler mode). It has not been remeasured on the current maskless
NHWC path and should not be applied as an extra 8% to the final table.
At the old measured rates it breaks even after roughly 27 million samples, so
it is useful for a long uninterrupted run but is not the global default. The
CUDA-Graph-enabled `max-autotune` mode was also attempted, but Inductor skipped
graph capture because the training graph mutates running state; use
`max-autotune-no-cudagraphs` for these models. Autotune modes should be measured
on the exact model and batch.

The batched Muon and foreach Adam paths implement the same update equations but
are not bitwise identical to the scalar-launch implementations. The RoPE cast
also slightly changes AMP rounding. All three can be disabled independently
when exact regression against an older run matters.
