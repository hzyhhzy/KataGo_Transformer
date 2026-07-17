"""Real-collective integration smoke test for Muon's flat DDP buckets.

Run with:

    torchrun --standalone --nproc-per-node=2 train/test_muon_ddp_bucketing_integration.py

Importing this module, including through unittest discovery, does not initialize a
process group. CUDA/NCCL is preferred; CPU/Gloo is used when two CUDA devices are
not available.
"""

import datetime
import os
import sys

import torch
import torch.distributed as dist

try:
    from .muon_kissin import MuonWithAuxAdamKimi, muon_update_kimi
except ImportError:
    from muon_kissin import MuonWithAuxAdamKimi, muon_update_kimi


def _make_parameter(shape, value_offset, device):
    numel = 1
    for size in shape:
        numel *= size
    values = torch.arange(numel, dtype=torch.float32, device=device)
    values = values.reshape(shape).mul_(0.002).add_(value_offset)
    return torch.nn.Parameter(values)


def _make_gradient(param, value_offset):
    values = torch.arange(param.numel(), dtype=torch.float32, device=param.device)
    return values.reshape_as(param).mul_(0.001).add_(value_offset)


def _reference_step(param_specs):
    references = {}
    with torch.no_grad():
        for param, momentum, learning_rate, weight_decay in param_specs:
            reference = param.detach().clone()
            reference_grad = param.grad.detach().clone()
            reference_momentum = torch.zeros_like(reference)
            update = muon_update_kimi(
                reference_grad,
                reference_momentum,
                beta=momentum,
            )
            reference.mul_(1.0 - learning_rate * weight_decay)
            reference.add_(update, alpha=-learning_rate)
            references[id(param)] = reference
    return references


def _select_backend_and_device(local_rank, world_size):
    if (
        dist.is_nccl_available()
        and torch.cuda.is_available()
        and torch.cuda.device_count() >= world_size
    ):
        torch.cuda.set_device(local_rank)
        return "nccl", torch.device("cuda", local_rank)
    if dist.is_gloo_available():
        return "gloo", torch.device("cpu")
    return None, None


def main():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(
            "SKIP: no torchrun rank environment; run with "
            "`torchrun --standalone --nproc-per-node=2 "
            "train/test_muon_ddp_bucketing_integration.py`"
        )
        return 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        if rank == 0:
            print(f"SKIP: this integration test requires world_size=2, got {world_size}")
        return 0

    backend, device = _select_backend_and_device(local_rank, world_size)
    if backend is None:
        if rank == 0:
            print("SKIP: neither two-device NCCL nor CPU Gloo is available")
        return 0

    dist.init_process_group(
        backend=backend,
        timeout=datetime.timedelta(seconds=120),
    )
    previous_batched_ns = os.environ.get("KATAGO_MUON_BATCHED_NS")
    # This test isolates flat-bucket ownership and collectives. The batched
    # Newton-Schulz path has its own numerical tests and need not be bitwise
    # identical to independent GEMMs across CUDA/cuBLAS versions.
    os.environ["KATAGO_MUON_BATCHED_NS"] = "0"
    try:
        # Every rank constructs exactly the same initial parameters and gradients.
        # The 128-byte cap is intentionally tiny, forcing several parameters to
        # span buckets and exercising padding for unequal owner payloads.
        group_one_params = [
            _make_parameter((10, 6), 0.10, device),
            _make_parameter((9, 7), -0.20, device),
            _make_parameter((7, 5), 0.30, device),
        ]
        group_two_params = [
            _make_parameter((12, 8), -0.40, device),
            _make_parameter((11, 9), 0.50, device),
        ]
        all_params = group_one_params + group_two_params
        for param_index, param in enumerate(all_params):
            param.grad = _make_gradient(param, 0.01 * (param_index + 1))

        group_specs = [
            (group_one_params, 0.80, 0.025, 0.030),
            (group_two_params, 0.90, 0.015, 0.070),
        ]
        reference_specs = [
            (param, momentum, learning_rate, weight_decay)
            for params, momentum, learning_rate, weight_decay in group_specs
            for param in params
        ]
        references = _reference_step(reference_specs)

        optimizer = MuonWithAuxAdamKimi(
            [
                {
                    "params": group_one_params,
                    "group_name": "normal",
                    "use_muon": True,
                    "momentum": group_specs[0][1],
                    "weight_decay": group_specs[0][3],
                },
                {
                    "params": group_two_params,
                    "group_name": "normal_attn",
                    "use_muon": True,
                    "momentum": group_specs[1][1],
                    "weight_decay": group_specs[1][3],
                },
            ],
            distributed_bucket_cap_bytes=128,
        )
        optimizer.param_groups[0]["lr"] = group_specs[0][2]
        optimizer.param_groups[1]["lr"] = group_specs[1][2]
        optimizer.step()

        bucket_count = sum(
            len(layout.buckets) for layout in optimizer._muon_distributed_layouts
        )
        local_errors = []
        if bucket_count <= 1:
            local_errors.append(f"small bucket cap did not create multiple buckets: {bucket_count}")

        # Finish every verification collective before raising, so a mismatch on
        # one rank cannot strand the peer in a later all-gather.
        for param_index, param in enumerate(all_params):
            reference = references[id(param)]
            gathered = torch.empty(
                world_size * param.numel(),
                dtype=param.dtype,
                device=param.device,
            )
            dist.all_gather_into_tensor(gathered, param.detach().view(-1))
            gathered = gathered.view(world_size, param.numel())
            for source_rank in range(world_size):
                if not torch.equal(gathered[source_rank], gathered[0]):
                    max_diff = (
                        gathered[source_rank] - gathered[0]
                    ).abs().max().item()
                    local_errors.append(
                        f"param {param_index} rank {source_rank} differs across ranks; "
                        f"max_abs_diff={max_diff}"
                    )
                if not torch.allclose(
                    gathered[source_rank],
                    reference.view(-1),
                    rtol=5e-3,
                    atol=5e-4,
                ):
                    max_diff = (
                        gathered[source_rank] - reference.view(-1)
                    ).abs().max().item()
                    local_errors.append(
                        f"param {param_index} rank {source_rank} differs from reference; "
                        f"max_abs_diff={max_diff}"
                    )

        local_ok = torch.tensor(
            0 if local_errors else 1,
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        if local_ok.item() != 1:
            if local_errors:
                print(f"rank {rank} failures:\n  " + "\n  ".join(local_errors), file=sys.stderr)
            raise AssertionError("Muon DDP flat-bucket integration test failed")

        if rank == 0:
            print(
                "MUON_DDP_FLAT_BUCKET_INTEGRATION_OK "
                f"backend={backend} world_size={world_size} buckets={bucket_count}"
            )
        return 0
    finally:
        if previous_batched_ns is None:
            os.environ.pop("KATAGO_MUON_BATCHED_NS", None)
        else:
            os.environ["KATAGO_MUON_BATCHED_NS"] = previous_batched_ns
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
