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

The following numbers were measured on two RTX 4090 D cards with PyTorch
2.12.1, 15x15 Gomoku data, and physical GPUs 0 and 3. Each number excludes the
first 100-step interval, which includes model/loss/optimizer compilation.
Batch size is per GPU and was selected for maximum samples/s rather than for a
fixed memory target.

| Model | Baseline batch | Baseline samples/s | Optimized batch | Optimized samples/s | Gain |
|---|---:|---:|---:|---:|---:|
| `b24c256h8tflrs-bng-silu-v102` | 384 | about 2,315 | 416 | about 2,800 | about 21% |
| `b40c384h12tflrs-bng-silu-v102` | 160 | about 760 | 172 | about 947 | about 25% |

The final batch-416 b24 smoke run used about 22.9 GiB per card; batch-172 b40
peaked near 23.4 GiB. Batch 168 was only about 0.5% slower and leaves more
headroom. These
are benchmark-specific starting points, not universal defaults:
allocator state, validation, checkpointing, drivers, and other processes can
require a smaller batch.

BF16 AMP and BF16 DDP gradient compression were also measured. BF16 AMP was
effectively tied with FP16; BF16 gradient compression improved b40 by only about
0.3% while changing reduction precision. FP16 remains the script default and
gradient communication remains FP32.

## Runtime controls

The optimized settings are on by default. Set an environment variable to `0`
to disable an individual optimization for debugging or regression comparison:

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

`-no-compile` disables both model and loss compilation. QAT also disables the
compiled loss. Muon's Newton-Schulz kernel retains its existing local
`torch.compile` implementation; `-no-compile` has historically controlled the
model rather than that optimizer kernel. If either model-norm deferral or the
on-device seki average is disabled, set `KATAGO_COMPILE_TRAINING_LOSS=0` as
well.

`KATAGO_COMPILE_MODE` accepts `default`, `max-autotune-no-cudagraphs`, or
`max-autotune`. The main table uses `default`. On b24 batch 416,
`max-autotune-no-cudagraphs` took about 757 seconds for the first 100-step
interval, then sustained about 2,990--3,023 samples/s (roughly another 8% over
the default compiler mode). At the measured rates it breaks even after roughly
27 million samples, so it is useful for a long uninterrupted run but is not the
global default. The CUDA-Graph-enabled `max-autotune` mode was also attempted,
but Inductor skipped graph capture because the training graph mutates running
state; use `max-autotune-no-cudagraphs` for these models. Autotune modes should
be measured on the exact model and batch.

The batched Muon and foreach Adam paths implement the same update equations but
are not bitwise identical to the scalar-launch implementations. The RoPE cast
also slightly changes AMP rounding. All three can be disabled independently
when exact regression against an older run matters.
