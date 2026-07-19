# FlexAttention mixed-board training benchmark

## Scope and method

- Server: 4 x RTX 4090 D, PyTorch 2.12.1+cu130.
- Sequence length: 15 x 15 (225 tokens), FP16 training, four-process DDP.
- Models: `b24c256h8tflrs-bng-silu-v102` at batch 384/GPU and
  `b40c384h12tflrs-bng-silu-v102` at batch 160/GPU.
- Every mixed-data rank batch actually contained both full and smaller boards.
  At batch 384, all 7,832 inspected rank batches were mixed and contained
  16-62 smaller-board rows. At batch 160, all 18,812 inspected rank batches
  were mixed and contained 2-34 smaller-board rows.
- Each reported value is the median of six stable log windows. The first
  compile-heavy window was discarded. All other runtime settings, including
  compiled loss, batched Muon and foreach auxiliary Adam, were held constant.

## Results

| Model | Samples/window | Mixed masked SDPA | Mixed FlexAttention | Flex vs SDPA | Full-15 no mask | Flex gap to no mask |
|---|---:|---:|---:|---:|---:|---:|
| b24c256h8 | 153,600 | 27.695 s / 5,546 samples/s | 24.630 s / 6,236 samples/s | +12.4% | 22.130 s / 6,941 samples/s | -10.2% |
| b40c384h12 | 64,000 | 33.895 s / 1,888 samples/s | 30.565 s / 2,094 samples/s | +10.9% | 27.530 s / 2,325 samples/s | -9.9% |

FlexAttention recovered about 49.5% of the b24 mask-related throughput gap and
47.1% of the b40 gap. It also used slightly less peak memory than masked SDPA:
about 22.45 GiB/GPU versus 22.53 GiB/GPU for b40 at batch 160.

## Kept implementation

- `-use-flex-attention` is a runtime training flag, not a ModelConfig field and
  not part of a checkpoint.
- A per-sample, KV-only `BlockMask` exactly preserves the old additive SDPA mask
  semantics, including evaluating off-board query rows. One mask is built per
  model forward and reused by every transformer layer.
- The mask uses `H=1` so it broadcasts over all heads. Each DDP rank builds it
  from its local batch; no new collective or cross-rank routing is needed.
- FlexAttention requires `torch.compile` and is rejected with `-no-compile`,
  `-disable-mask`, or QAT. Without the flag, the old SDPA paths are unchanged.
- SWA validation deliberately uses masked SDPA. `AveragedModel` is evaluated
  eagerly, where eager FlexAttention materializes the full score matrix; the
  runtime fallback does not alter averaged parameters or checkpoint keys.

## Kernel tuning conclusions

- Keep PyTorch's default `BLOCK_SIZE=128` and default kernel heuristics.
- `BLOCK_SIZE=64` cannot compile backward on RTX 4090 with the PyTorch 2.12
  default SM89 backward tile (`NoValidChoicesError`). Enabling full-model
  max-autotune only to obtain additional Flex candidates is not worthwhile.
- Separately compiling `create_block_mask` added roughly 69 seconds of startup
  work and did not change steady throughput, so the deprecated private
  `_compile=True` path was removed.
- `PRESCALE_QK`, `WRITE_DQ=False`, forward/backward tile overrides, and
  contiguous-block hints had no repeatable full-training gain. The best manual
  forward tile was only 0.2-0.6% faster and was within run-to-run noise.
- `ROWS_GUARANTEED_SAFE=True` produced NaN in the first training window because
  225 is not block-aligned; it must not be used. A 128x128 forward tile with
  three stages exceeded the 4090 shared-memory limit.
- A general O(B*S) direct BlockMask metadata builder exactly matched the dense
  reference for 9x9, 15x15 and 19x19 masks, but its full b24 median/mean were
  indistinguishable from the simpler official `create_block_mask` path. It was
  removed to keep the production code smaller.

## Numerical checks

For both target `(batch, heads)` shapes, real compiled FlexAttention was compared
with masked SDPA on identical FP16 Q/K/V tensors and output gradients:

- all outputs and dQ/dK/dV values were finite;
- maximum output difference was at most 9.77e-4;
- maximum dQ/dK difference was 1.95e-3, with mean absolute error about 3.1e-5;
- maximum dV difference was 4.88e-4.

A four-rank b24 end-to-end boundary run also completed FlexAttention training,
ordinary validation, SWA validation, checkpoint saving and the final state-dict
NaN check without an NCCL or process failure. Ordinary validation took 37.56 s
including its compiled FlexAttention startup; SWA validation used the deliberate
masked-SDPA fallback and took 8.27 s.

The complete training-code unit-test suite passed on the server (89 tests).
