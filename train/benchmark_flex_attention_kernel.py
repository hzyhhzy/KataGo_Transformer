"""Microbenchmark FlexAttention kernel options for KataGo transformer shapes.

This is intentionally separate from model configuration. It measures a single
forward/backward attention operation and compares every candidate against the
masked SDPA path on identical FP16 inputs and gradients.
"""

import argparse
import json
import statistics

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def make_block_mask(batch_size: int, seq_len: int, device: torch.device):
    board_len = int(seq_len**0.5)
    if board_len * board_len != seq_len:
        raise ValueError(f"seq_len must be a square board, got {seq_len}")

    # Approximate the benchmark dataset: predominantly full 15x15 boards, with
    # a small board in every tenth row. Only KV positions are masked, matching
    # model_pytorch.py's existing SDPA semantics.
    valid_kv = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    for batch_idx in range(0, batch_size, 10):
        small_len = 9 + (batch_idx // 10) % max(1, board_len - 8)
        small_len = min(small_len, board_len - 1)
        spatial = torch.zeros((board_len, board_len), dtype=torch.bool, device=device)
        spatial[:small_len, :small_len] = True
        valid_kv[batch_idx] = spatial.flatten()

    def mask_mod(b, h, q_idx, kv_idx):
        return valid_kv[b, kv_idx]

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=1,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        BLOCK_SIZE=128,
    )
    return block_mask, valid_kv


def make_attention(block_mask, kernel_options):
    def attention(q, k, v):
        return flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            kernel_options=(kernel_options or None),
        )

    return torch.compile(attention, fullgraph=True)


def make_sdpa_attention(valid_kv):
    def attention(q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=valid_kv.reshape(valid_kv.shape[0], 1, 1, -1),
            dropout_p=0.0,
        )

    return torch.compile(attention, fullgraph=True)


def run_once(attention, q, k, v, grad_out):
    q.grad = None
    k.grad = None
    v.grad = None
    out = attention(q, k, v)
    out.backward(grad_out)
    return out


def tensor_error(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def benchmark_candidate(
    name,
    kernel_options,
    block_mask,
    q,
    k,
    v,
    grad_out,
    references,
    warmup_iterations,
    iterations,
):
    attention = make_attention(block_mask, kernel_options)
    for _ in range(warmup_iterations):
        run_once(attention, q, k, v, grad_out)
    torch.cuda.synchronize()

    elapsed_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run_once(attention, q, k, v, grad_out)
        end.record()
        end.synchronize()
        elapsed_ms.append(start.elapsed_time(end))

    result = {
        "name": name,
        "kernel_options": kernel_options,
        "median_ms": statistics.median(elapsed_ms),
        "mean_ms": statistics.fmean(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "output_error": tensor_error(out, references[0]),
        "dq_error": tensor_error(q.grad, references[1]),
        "dk_error": tensor_error(k.grad, references[2]),
        "dv_error": tensor_error(v.grad, references[3]),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=225)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument(
        "--candidates-json",
        required=True,
        help='JSON list of {"name": str, "kernel_options": object}',
    )
    args = parser.parse_args()
    candidates = json.loads(args.candidates_json)
    if not isinstance(candidates, list):
        raise ValueError("--candidates-json must be a JSON list")

    torch.manual_seed(12345)
    device = torch.device("cuda")
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
    k = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
    v = torch.randn(shape, dtype=torch.float16, device=device, requires_grad=True)
    grad_out = torch.randn(shape, dtype=torch.float16, device=device)
    block_mask, valid_kv = make_block_mask(args.batch_size, args.seq_len, device)

    sdpa_attention = make_sdpa_attention(valid_kv)
    reference_out = run_once(sdpa_attention, q, k, v, grad_out)
    references = (
        reference_out.detach().clone(),
        q.grad.detach().clone(),
        k.grad.detach().clone(),
        v.grad.detach().clone(),
    )

    results = []
    for candidate in candidates:
        try:
            results.append(
                benchmark_candidate(
                    candidate["name"],
                    candidate.get("kernel_options", {}),
                    block_mask,
                    q,
                    k,
                    v,
                    grad_out,
                    references,
                    args.warmup_iterations,
                    args.iterations,
                )
            )
        except Exception as error:
            print(
                json.dumps(
                    {
                        "name": candidate.get("name", "<missing>"),
                        "kernel_options": candidate.get("kernel_options", {}),
                        "error": f"{type(error).__name__}: {error}",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if results:
        best = min(results, key=lambda result: result["median_ms"])
        print("BEST=" + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
