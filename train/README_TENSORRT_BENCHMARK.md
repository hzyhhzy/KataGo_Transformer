# TensorRT inference benchmark

`benchmark_tensorrt.py` builds or loads a TensorRT 10.x serialized engine and
measures fixed-batch inference using CUDA events. All inputs and outputs remain
in PyTorch CUDA tensors, and the measured loop calls TensorRT's
`execute_async_v3` directly, so it contains no host/device copies.

Export a static batch-64 ONNX model first. For a full 15x15 board, the relevant
export options are:

```bash
cd train
python export_onnx.py \
  -checkpoint /path/to/checkpoint.ckpt \
  -export-dir /path/to/export \
  -model-name benchmark-model \
  -pos-len 15 \
  -batch-size 64 \
  -fix-batchsize \
  -disable-mask \
  -use-swa
```

Run the benchmark in an environment with CUDA-enabled PyTorch and TensorRT
10.13 installed:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_tensorrt.py /path/to/export/model.onnx \
  --engine /path/to/export/model.bs64.fp16.plan \
  --expected-batch-size 64 \
  --workspace-gib 16 \
  --warmup 20 \
  --iterations 100 \
  --repeats 10 \
  --json-output /path/to/export/model.bs64.benchmark.json
```

The first invocation parses the ONNX file, enables FP16 builder tactics, builds
the engine, and saves both the plan and a sibling build manifest named
`MODEL.plan.build.json`. The manifest records SHA256 and byte size for the ONNX
and plan, every builder option used by this script, the exact TensorRT version,
and the build GPU name and compute capability. Hashing and provenance checks are
outside the measured region.

On later invocations that include the ONNX path, the script validates the plan
hash, ONNX hash, FP16 setting, workspace limit, builder optimization level,
TensorRT version, and GPU. A missing, damaged, or mismatched manifest, a changed
ONNX file, or changed builder options automatically rebuilds the engine. Thus
`--rebuild` is only needed to force a fresh tactic search when all recorded
inputs still match. TensorRT engine plans depend on the TensorRT version and GPU,
so build the plan on the machine where it will be measured. Load only engine
plans that you built yourself or obtained from a trusted source; serialized
plans contain executable code.

When the ONNX path is supplied, repeat the intended builder options on every
invocation. For example, a plan built with `--workspace-gib 16` will be rebuilt
with the default 8 GiB limit if that option is omitted later. When only
`--engine` is supplied, the recorded builder options are accepted as-is because
there is no ONNX input available for rebuilding.

Optionally verify the plan against ONNX Runtime on CPU before benchmarking:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_tensorrt.py /path/to/export/model.onnx \
  --engine /path/to/export/model.bs64.fp16.plan \
  --expected-batch-size 64 \
  --workspace-gib 16 \
  --verify-onnxruntime \
  --verify-atol 1e-2 \
  --verify-rtol 1e-2
```

Verification copies the already-filled TensorRT input buffers to CPU, performs
one synchronized TensorRT launch, and compares every same-named output with an
ONNX Runtime `CPUExecutionProvider` run. It reports `max_abs`, `max_rel`, and
`mean_abs` for each output and fails if any output does not satisfy NumPy-style
`allclose` with the selected tolerances. The verification launch, device/host
copies, and CPU execution all occur outside the timed region. The option is off
by default, so ONNX Runtime is not imported and no validation copies or launches
occur during a normal benchmark. If enabled, the ONNX path is required even
when `--engine` names an existing plan; this prevents silently checking against
no reference model. Install the CPU `onnxruntime` package in the benchmark
environment before using this option. Verification results are also included in
`--json-output`.

The script rejects dynamic tensor shapes and infers batch size from the leading
dimension shared by all engine inputs. Its reported latency is batch latency;
`samples/s` is `fixed_batch_size / mean_batch_latency`. Input buffers are filled
once before warmup. In the default path output buffers are never copied back;
optional ONNX Runtime verification copies them only before timing. Engine
execution is synchronized only outside each measured repeat. The implementation
also requires linear device I/O, which is the normal format for these exported
models.

Useful options:

- `--device N` selects a logical device after `CUDA_VISIBLE_DEVICES` filtering.
- `--no-fp16` builds an engine without enabling FP16 tactics.
- `--builder-optimization-level 0..5` controls TensorRT build effort.
- `--verify-onnxruntime` enables the unmeasured ONNX Runtime CPU comparison;
  `--verify-atol` and `--verify-rtol` default to `1e-2`.
- `--cuda-graph` performs the requested ordinary TensorRT warmup, captures one
  `execute_async_v3` launch on the benchmark's PyTorch CUDA stream, and measures
  CUDA Graph replay instead of direct enqueue calls. Capture, graph
  instantiation, and one initial replay are synchronized before timing starts.
- `--verbose` enables verbose TensorRT parser/builder logging.
- An existing engine can be measured without its ONNX file using
  `python benchmark_tensorrt.py --engine /path/to/model.plan`, but its sibling
  `.build.json` manifest must be present and its plan hash, TensorRT version, and
  GPU must validate. Builder options are read from the manifest in this mode,
  since there is no ONNX file from which to rebuild. If validation fails, supply
  the matching ONNX path and the script will rebuild automatically.

`--json-output` records the ONNX and engine SHA256 values and sizes, manifest
path, complete builder configuration, original build environment, current
runtime environment, engine I/O, and timing results. This makes benchmark files
self-contained enough to detect accidental plan reuse in later comparisons.

CUDA Graph mode is disabled by default, so the direct `execute_async_v3` path
remains the baseline. The engine is static-shape and its PyTorch I/O buffers stay
alive at fixed device addresses for the entire benchmark, as TensorRT graph
capture requires. At least one ordinary warmup launch is required before
capture. If TensorRT tactics or plugins in an engine are not capture-compatible,
the script stops with a capture error instead of silently falling back to the
direct path. ONNX Runtime verification, when selected, runs independently before
the benchmark creates or captures its timing stream.

## RTX 4090 D reference measurements

With TensorRT 10.13.3, FP16, builder optimization level 3, fixed batch 64, and
15x15 inputs, direct enqueue measured about 6.74 ms (9,500 samples/s) for
`b24c256h8tflrs` and 19.64 ms (3,259 samples/s) for `b40c384h12tflrs`.
CUDA Graph replay reduced these to about 6.59 ms (9,714 samples/s) and 19.33 ms
(3,310 samples/s), respectively. These timings include engine execution only,
not host/device copies.

All five outputs were compared against ONNX Runtime CPU for both models. The
largest absolute error was about 0.0027 for b24 and 0.0046 for b40; both passed
with `atol=0.02, rtol=0.02`.

Two graph rewrites were rejected based on direct TensorRT measurements: freezing
learned RoPE trigonometric tables was about 0.2% slower, and replacing exported
SDPA with explicit matmul/softmax attention was about 1% slower. TensorRT was
already optimizing the original graph effectively. Builder optimization level 5
took roughly nine minutes for b24 and did not produce a repeatable latency gain
over level 3.
