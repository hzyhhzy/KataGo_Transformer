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
*   **GQA**: Grouped Query Attention (GQA) is currently **disabled** by default due to the lack of a highly optimized implementation.

**Source Code**: `TransformerRoPEGQABlock` class in `./train/model_pytorch.py`.

**Configurations**:
Pre-defined model configurations are available in `./train/modelconfigs.py`.
*   Example `b14c192h6tfrs`: 14 layers, 192 hidden size, 6 QKV heads, 512 feedforward size, with RoPE and SwiGLU.

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

### Model Type Settings
*   **Model Structure**: `b14c192h6tfrs` is a pre-defined structure in `./train/modelconfigs.py`. You can modify this file to define custom architectures.
*   **Postfixes**:
    *   `-bng-silu`: Recommended. Enables Batch Normalization in Conv layers and SiLU activation in Transformer layers.
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

### 2. TensorRT-ONNX Engine
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
