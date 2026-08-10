# KataGo_Transformer

Transformer-based models for board games, designed for use with KataGo.

*   **Original KataGo**: [GitHub](https://github.com/lightvector/KataGo) | [Website](https://katagotraining.org/)
*   **KataGomo (Fork for various games)**: [GitHub](https://github.com/hzyhhzy/KataGomo)

---

## Technical Details

### Muon Optimizer
This project utilizes the **Muon optimizer**, which has demonstrated strong performance with KataGo models.
The implementation can be found in `./train/muon_kissin.py` (adapted for KataGo by @loker404 and the author).

### Transformer Architecture
The transformer architecture shares similarities with **QWen3**.
*   **Components**: Uses **RoPE** (Rotary Positional Embeddings), **SwiGLU**, and **RMSNorm**. These components have been verified to provide stable performance.
    *   Use `tflrs` model aliases or add `-lrope` to transformer RoPE model names for learnable 2D RoPE frequencies, e.g. `b14c192h6tflrs-bng-silu` or `b14c192h6tfrs-bng-silu-lrope`.
    *   `nbttflrs` models use nested bottleneck transformer blocks: 1x1 channel reduction, transformer blocks in the bottleneck, then 1x1 expansion back to trunk channels.
*   **GQA**: Grouped Query Attention (GQA) is currently **disabled** by default due to the lack of a highly optimized implementation.

**Source Code**: `TransformerRoPEGQABlock` class in `./train/model_pytorch.py`.

**Configurations**:
Pre-defined model configurations are available in `./train/modelconfigs.py`.
*   Example `b14c192h6tfrs`: 14 layers, 192 hidden size, 6 QKV heads, 512 feedforward size, with RoPE and SwiGLU.
*   Example `b4c256h4nbttflrs`: 4 nested bottleneck transformer blocks, 256 trunk channels, 128 bottleneck channels, 4 heads, learnable RoPE, and SwiGLU.

---

## Training

**Prerequisites**: PyTorch **2.7+** is recommended.
> **Note**: `NaN` issues have been reported when using PyTorch 2.5 with transformer models.

### Command
```bash
bash train_muon_ki.sh {save_dir} {data_dir} {save_name} {model_type} {batch_size} {"extra"/"main"/"trainonly"} {other arguments}
```

### Example
```bash
bash train_muon_ki.sh ../data ../data/shuffleddata/current b14c192h6tfrs_1 b14c192h6tfrs-bng-silu 384 extra -multi-gpus 0,1,2,3 -lr-scale-auto-type custom
```

### Parameters
Parameters can be modified in `./train/train_muon_ki.sh` or passed as arguments (arguments override file settings).

*   `save_dir`: Directory where the model will be saved (`{save_dir}/train/{save_name}`).
*   `data_dir`: Directory containing shuffled data (KataGo format).
*   `save_name`: Name for the saved model.
*   `model_type`: Model architecture type (e.g., `b14c192h6tfrs-bng-silu`).
*   `batch_size`: Training batch size.
*   `"extra"/"main"/"trainonly"`: Determines where to export `.bin.gz` models:
    *   `extra`: Exports to `"{save_dir}/models_extra"`
    *   `main`: Exports to `"{save_dir}/models"`
    *   `trainonly`: Does not automatically export models (manual export via `export_bin.sh` is possible).

### Extra Arguments
*   `-multi-gpus {gpus}`: Specify GPUs to use, e.g., `0,1,2,3`.
*   `-lr-scale-auto-type {type}`: Use a custom learning rate schedule defined in `./train/train_muon_ki.py`.
    *   `custom`: Predefined schedule where `lr-scale ~ 1/sqrt(step)`.
*   `-lr-scale {scale}`: Fixed learning rate scale (e.g., `1.0`). Cannot be used with `-lr-scale-auto-type`.
*   `-enable-history-matrices`: Enables history matrices transformation (enabled by default in `./train/train_muon_ki.sh`).
    *   **Note**: This is primarily for Go. **Remove or disable this flag when training for other games.**
*   `-symmetry-type {type}`: Data augmentation symmetry type. Default is `xyt` in `./train/train_muon_ki.sh`.
    *   `xyt`: x-flip, y-flip, or transpose (8-fold symmetry). Suitable for Go, Gomoku, etc.
    *   `xy`: 4-fold symmetry.
    *   `x`: 2-fold symmetry (x-flip). Suitable for chess-like games.
    *   `x+y`: Simultaneous x and y flip (2-fold symmetry). Suitable for Hex.
    *   `none`: No symmetry.
*   `-disable-mask`: Use the mask-free model and loss path. Training data must
    contain only complete `pos-len` by `pos-len` boards; the loader verifies
    every file before use. See `train/README_TRAINING_THROUGHPUT.md` for the
    full-board filtering command and runtime tuning controls.
*   `-filter-full-board-on-load`: With `-disable-mask`, discard non-full-board
    training and validation rows while loading instead of rejecting a mixed
    NPZ file. Files retaining fewer than one applicable batch produce a warning
    and no batches.
*   `-disable-validation-full-board-filter`: Keep the original mixed validation
    rows when `-filter-full-board-on-load` is enabled. Validation remains masked
    and NCHW either way.
*   `-input-memory-format {nhwc,nchw}`: Spatial input memory format. Defaults
    to `nhwc`; pass `nchw` only for compatibility or regression comparison.
*   `-compile-mode {default,max-autotune-no-cudagraphs,max-autotune}`: Select
    the `torch.compile` mode. Defaults to `default`.
*   `-sdpa-backend {auto,flash,cudnn,efficient,math}`: Select the CUDA SDPA
    backend. Defaults to `auto`.
*   `-use-flex-attention`: Opt in to FlexAttention for compiled, masked
    Transformer training. It can reduce mask overhead and improve throughput,
    but some model shapes or mixed-board masks have made all losses stall and
    then rapidly diverge to NaN. Verify short-run overall convergence, model
    norms, and finite losses before a long run. Without this flag, masked SDPA
    is used by default; it is generally more robust but can be slower.

### Model Type Settings
*   **Model Structure**: `b14c192h6tfrs` is a pre-defined structure in `./train/modelconfigs.py`. You can modify this file to define custom architectures.
*   **Postfixes**:
    *   `-bng-silu`: Recommended. Enables Batch Normalization in Conv layers and SiLU activation in Transformer layers.
    *   `-lrope` / `tflrs`: Uses learnable 2D RoPE frequencies instead of fixed axis-aligned RoPE tables.
    *   `-qkn`: Enables QK Norm in each supported transformer block. Q and K are reshaped into heads, independently RMS-normalized over `head_dim`, and only then passed through RoPE and attention. Q and K have separate learnable gamma vectors (initialized to one, shared across heads within each projection, and with no beta); their parameters are assigned to the `noreg` optimizer group, which uses the training code's strongly reduced `noreg` weight decay rather than the regular weight decay. Despite the group name, this decay is small but nonzero under the current schedules. This postfix is generated for the `TransformerRoPEGQABlock` family, not for non-transformer models or the legacy block whose kind is exactly `transformer`.
    *   `-clip4` / `-clip7`: For SwiGLU transformer configurations only, clamps both factors of every SwiGLU product to `[-4, 4]` or `[-7, 7]`: the activated up branch and the linear gate branch. It does not add clipping elsewhere, and these aliases are not generated for non-SwiGLU transformers.
    *   `-fullclip4` / `-fullclip7`: Simulates bounded INT8 activations throughout each supported transformer block by clamping the block and normalization inputs/outputs, Q/K/V projection and post-RoPE tensors, attention and output-projection tensors, residual outputs, and FFN projection, activation, gate, product, and output tensors to the selected symmetric range. It applies to both SwiGLU and non-SwiGLU transformer configurations, but does not clip the input stem, global projection, or policy/value heads.
    *   QK Norm may be combined with either clipping mode, for example `b14c192h6tfrs-bng-silu-qkn-clip7` or `b14c192h6tfrs-bng-silu-qkn-fullclip4`. The generated order is `-qkn` followed by the clipping postfix. `-clipN` and `-fullclipN` are mutually exclusive; mixed aliases such as `-clip4-fullclip4` are intentionally not generated.
    *   `-v11`: Use version 11 of the model input features (common for games other than Go).

---

## Inference with KataGo Engine

To use these models in KataGo, you must export them to ONNX format and use a modified engine that supports ONNX inference.

### 1. Export ONNX Model
Use `./train/export_onnx.py` to convert a checkpoint to ONNX.

**Command**:
```bash
python export_onnx.py -checkpoint {checkpoint_file} -export-dir {export_dir} -model-name {model_name} -pos-len {pos_len} -batch-size 8 -use-swa -disable-mask
```

**Example**:
```bash
python export_onnx.py -checkpoint ../data/train/b14c192h6tfrs_1/checkpoint.ckpt -export-dir ../data/models_onnx -model-name b14c192h6tfrs_1 -pos-len 19 -batch-size 8 -use-swa -disable-mask
```

**Arguments**:
*   `-checkpoint`: Path to the checkpoint file (usually `{save_dir}/train/{save_name}/checkpoint.ckpt`).
*   `-export-dir`: Directory to save the ONNX model.
*   `-model-name`: Filename for the exported model.
*   `-pos-len`: Board size (e.g., `19` for Go, `15` for Gomoku).
    *   *Note*: Rectangular boards and dynamic board sizes are **not supported**. You must export separate models for different board sizes.
*   `-batch-size`: Batch size used during export (has no effect on inference, `8` is standard).
*   `-use-swa`: Whether to use the SWA (Stochastic Weight Averaging) model if available.
*   `-disable-mask`: Disables masking. This can slightly improve performance.

### 2. ONNX Runtime CPU Optimization and INT8

Two command-line tools build ONNX Runtime CPU-specific inference graphs:

*   `./train/ort_cpu_optimize_fp32.py` rewrites transformer attention and RoPE to ONNX Runtime `com.microsoft` `MultiHeadAttention` and `RotaryEmbedding` operators, replaces RMSNorm with `SimplifiedLayerNormalization`, and fuses residual additions with normalization as `SkipSimplifiedLayerNormalization` where applicable.
*   `./train/ort_cpu_quantize_int8.py` converts the optimized FP32 graph to dynamic-trunk W8A8. Activations are quantized dynamically to UINT8 at inference time and weights use symmetric per-tensor QInt8. Only the seven logical transformer projections per block (Q, K, V, attention output, SwiGLU up, gate, and FFN output) are quantized; the spatial stem, global projection, and all policy/value heads remain FP32. Fused and unfused QKV and SwiGLU projection layouts are detected and audited. The current seven-role quantizer therefore requires SwiGLU transformer blocks and rejects an incomplete or mixed block selection.

Install `numpy`, `onnx`, `onnxruntime`, and `onnxsim`, then first export a fixed-batch-1, mask-free, simplified FP32 model. In addition to the normal export arguments, use `-batch-size 1 -fix-batchsize -simplify -disable-mask`; the optimizer requires the simplified topology and `has_mask=false` metadata.

```bash
python train/ort_cpu_optimize_fp32.py \
    --input model_fixedb1_simplified.onnx \
    --output model_ort_cpu_fp32.onnx

python train/ort_cpu_quantize_int8.py \
    --input model_ort_cpu_fp32.onnx \
    --output model_ort_cpu_int8.onnx
```

Both commands write an audit report beside the output by default and fail closed when the expected graph structure cannot be proven. Each model and report file is fully staged beside its own destination and atomically replaces that destination; the two files do not form a cross-file transaction. `--data validation_file.npz` is optional and is used only for numerical validation of all five KataGo outputs; it is not needed to optimize FP32 and is not calibration data for dynamic INT8. For a multi-sample INT8 check, add `--data validation_file.npz --validation-samples 2`. The INT8 tool expects a self-contained ONNX file and rejects external-data models; the normal FP32 optimizer output satisfies this requirement.

The FP32 rewrite supports fixed `tfrs`, per-head learnable-RoPE `tflrs`, QK-Norm fixed-RoPE models, and ordinary SwiGLU `-clip4`/`-clip7` models. It currently rejects `-fullclip4`/`-fullclip7` graphs and the combination of learnable RoPE with QK Norm because those topologies require separate semantics-preserving matchers.

These outputs contain ONNX Runtime-specific contributed operators and target `CPUExecutionProvider`. They are not portable TensorRT graphs; keep the original exported ONNX and use a separate standard ONNX/QDQ calibration pipeline for TensorRT.

### 3. TensorRT-ONNX Engine
A modified KataGo engine supporting ONNX models is available here (source code only, compilation required):
[KataGomo (branch: go_onnx_test)](https://github.com/hzyhhzy/KataGomo/tree/go_onnx_test)
*(Mostly developed by @yehu3d)*

**Usage Notes**:
This is an experimental engine.

1.  **Static Board Size**: The engine does not support dynamic board sizes. The `Board::MAX_LEN` constant in the engine code must match the `-pos-len` used when exporting the ONNX model. To support a different board size, you must recompile the engine.
2.  **Placeholder Model File**:
    To load an ONNX model (e.g., `model.onnx`), you must currently provide a "dummy" placeholder file named `model.bin.gz` in the same directory.
    *   This file is required solely to bypass the engine's initialization checks.
    *   It is **not** used for inference.
    *   Any valid KataGo model file (e.g., a small untrained `b6c96` model) can be used, **but its version should match the ONNX model**.
