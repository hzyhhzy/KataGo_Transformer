import torch
import torch.distributed as dist

import logging
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple


DEFAULT_DISTRIBUTED_BUCKET_CAP_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class _MuonBucketSegment:
    param_index: int
    param_offset: int
    packed_offset: int
    numel: int


@dataclass(frozen=True)
class _MuonFlatBucketPlan:
    collective_numel: int
    owner_numels: Tuple[int, ...]
    segments_by_owner: Tuple[Tuple[_MuonBucketSegment, ...], ...]


@dataclass
class _MuonDistributedLayout:
    params: Tuple[torch.Tensor, ...]
    buckets: Tuple[_MuonFlatBucketPlan, ...]
    send_buffer: torch.Tensor
    gathered_buffer: torch.Tensor


def _build_muon_flat_bucket_plan(
    owner_param_numels: Sequence[Sequence[Tuple[int, int]]],
    bucket_cap_numel: int,
) -> Tuple[_MuonFlatBucketPlan, ...]:
    """Build equal-sized all-gather buckets from one parameter stream per owner.

    Each input item is ``(param_index, numel)``. A parameter may be split across
    buckets, but every element appears exactly once and in parameter-stream order.
    The collective size of a bucket is the largest owner payload in that bucket;
    shorter owner payloads are padded by the caller.
    """
    if bucket_cap_numel <= 0:
        raise ValueError(f"bucket_cap_numel must be positive, got {bucket_cap_numel}")
    if len(owner_param_numels) <= 0:
        raise ValueError("owner_param_numels must contain at least one owner")

    normalized_streams: List[Tuple[Tuple[int, int], ...]] = []
    for stream in owner_param_numels:
        normalized_stream = []
        for param_index, numel in stream:
            if param_index < 0:
                raise ValueError(f"param_index must be nonnegative, got {param_index}")
            if numel < 0:
                raise ValueError(f"parameter numel must be nonnegative, got {numel}")
            if numel > 0:
                normalized_stream.append((param_index, numel))
        normalized_streams.append(tuple(normalized_stream))

    stream_indices = [0 for _ in normalized_streams]
    param_offsets = [0 for _ in normalized_streams]
    buckets: List[_MuonFlatBucketPlan] = []

    while any(stream_indices[owner] < len(normalized_streams[owner]) for owner in range(len(normalized_streams))):
        owner_numels: List[int] = []
        segments_by_owner: List[Tuple[_MuonBucketSegment, ...]] = []

        for owner, stream in enumerate(normalized_streams):
            packed_offset = 0
            owner_segments: List[_MuonBucketSegment] = []
            while packed_offset < bucket_cap_numel and stream_indices[owner] < len(stream):
                param_index, param_numel = stream[stream_indices[owner]]
                param_offset = param_offsets[owner]
                take = min(param_numel - param_offset, bucket_cap_numel - packed_offset)
                assert take > 0
                owner_segments.append(_MuonBucketSegment(
                    param_index=param_index,
                    param_offset=param_offset,
                    packed_offset=packed_offset,
                    numel=take,
                ))
                packed_offset += take
                param_offset += take
                if param_offset == param_numel:
                    stream_indices[owner] += 1
                    param_offsets[owner] = 0
                else:
                    param_offsets[owner] = param_offset

            owner_numels.append(packed_offset)
            segments_by_owner.append(tuple(owner_segments))

        collective_numel = max(owner_numels)
        assert collective_numel > 0
        buckets.append(_MuonFlatBucketPlan(
            collective_numel=collective_numel,
            owner_numels=tuple(owner_numels),
            segments_by_owner=tuple(segments_by_owner),
        ))

    return tuple(buckets)


def _validate_muon_parameter_shape(param):
    """Validate Muon's static matrix interpretation before rank sharding."""
    original_shape = tuple(param.shape)
    dims_gt_4 = sum(1 for size in original_shape if size > 4)
    if dims_gt_4 <= 1:
        raise ValueError(
            f"Muon shape check failed: original_shape {original_shape} has only "
            f"{dims_gt_4} dimensions greater than 4 (must be > 1)."
        )

    if param.ndim == 2:
        matrix_shape = original_shape
    elif param.ndim == 4:
        if original_shape[0] <= 0:
            raise ValueError(
                f"Muon shape check failed: original_shape {original_shape} has an empty leading dimension"
            )
        matrix_shape = (original_shape[0], param.numel() // original_shape[0])
    else:
        raise ValueError(
            f"Muon shape check failed: original_shape {original_shape} is neither 2D nor 4D"
        )
    if matrix_shape[0] <= 4 or matrix_shape[1] <= 4:
        raise ValueError(
            f"Muon shape check failed: original_shape {original_shape} became {matrix_shape}"
        )

# Modified from "Kimi"'s Muon:  https://github.com/MoonshotAI/Moonlight
# Adapted for KataGo by LK (aka. loker404/Joe7/Kissin) and HZY (aka. Sigmoid/hzyhhzy)

@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    #assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    assert G.ndim == 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16() # this will be ignored if not supported
    #if torch.cuda.is_bf16_supported():
    #    X = G.bfloat16()
    #else:
    #    X = G.clone() # Fallback
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def zeropower_via_newtonschulz5_batched(G, steps: int):
    """Batched Newton-Schulz for matrices already oriented as rows <= columns."""
    assert G.ndim == 3
    assert G.size(-2) <= G.size(-1)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X



def muon_update_kimi(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    
    # 保存原始形状
    original_shape = update.shape

    dims_gt_4 = sum(1 for s in original_shape if s > 4)
    if dims_gt_4 <= 1:
        raise ValueError(f"Muon 形状检查失败: original_shape {original_shape} 中大于4的维度只有 {dims_gt_4} 个 (必须 > 1)。这通常意味着该参数不适合使用 Muon。")

    #print(update.shape)
    if update.ndim == 4:  # 对于卷积滤波器的情况
        update = update.view(len(update), -1)
        #print(update.shape)
    if update.shape[0] <= 4 or update.shape[1] <= 4 :
        raise ValueError(f"Muon 形状检查失败: original_shape {original_shape} 被reshape成 {update.shape} ")

    
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, max(grad.size()))**0.5
    
    # 恢复原始形状
    if len(original_shape) == 4:
        update = update.view(original_shape)
    else:
        assert original_shape==update.shape, f"original_shape={original_shape}, update.shape={update.shape}"
    
    return update




def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



class MuonWithAuxAdamKimi(torch.optim.Optimizer):
    def __init__(
        self,
        param_groups,
        momentum_default=0.95,
        distributed_bucket_cap_bytes=DEFAULT_DISTRIBUTED_BUCKET_CAP_BYTES,
    ):
        self.is_distributed = dist.is_initialized()
        self.momentum_default=momentum_default
        self.distributed_bucket_cap_bytes = int(distributed_bucket_cap_bytes)
        if self.distributed_bucket_cap_bytes <= 0:
            raise ValueError(
                f"distributed_bucket_cap_bytes must be positive, got {distributed_bucket_cap_bytes}"
            )
        self.use_batched_muon_ns = os.environ.get("KATAGO_MUON_BATCHED_NS", "1") == "1"
        self.use_foreach_aux_adam = os.environ.get("KATAGO_AUX_ADAM_FOREACH", "1") == "1"
        self.muon_ns_batch_size = 32
        self._distributed_muon_gradients_validated = False
        if self.use_batched_muon_ns:
            self.muon_ns_batch_size = int(os.environ.get("KATAGO_MUON_NS_BATCH_SIZE", "32"))
            if self.muon_ns_batch_size <= 0:
                raise ValueError(
                    "KATAGO_MUON_NS_BATCH_SIZE must be positive, "
                    f"got {self.muon_ns_batch_size}"
                )
            logging.info(
                "Using batched Muon Newton-Schulz with batch size %d",
                self.muon_ns_batch_size,
            )
        if self.use_foreach_aux_adam:
            logging.info("Using foreach kernels for auxiliary Adam parameter groups")
        self._muon_distributed_layouts = None
        for group in param_groups:
            # 确保所有参数组都有group_name
            #group["group_name"] = group.get("group_name", "")
            group["lr"] = 0 #会被update_and_return_lr_and_wd()覆盖
            
            
            # 设置默认学习率倍数 (adam学习率 = lr / muon_lr_multiplier)
            group["muon_lr_multiplier"] = group.get("muon_lr_multiplier", 8.0)
            
            if "use_muon" not in group:
                group["use_muon"] = self.is_muon_group(group["group_name"])
            
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                if self.is_distributed:
                    # Validate every parameter on every rank before ownership is
                    # sharded. A rank-local error during step would strand peers
                    # in the final parameter all-gather.
                    for param in group["params"]:
                        _validate_muon_parameter_shape(param)
                # Muon参数组的默认值
                group["momentum"] = group.get("momentum", momentum_default)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) <= set(["params", "lr", "momentum", "weight_decay", 
                                                "use_muon", "group_name", "muon_lr_multiplier"])
            else:
                # Adam参数组的默认值
                group["betas"] = group.get("betas", (self.momentum_default,  0.995))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) <= set(["params", "lr", "betas", "eps", "weight_decay", 
                                                "use_muon", "group_name", "muon_lr_multiplier"])
        super().__init__(param_groups, dict())

    def is_muon_group(self, group_name: str) -> bool:
        """自动判断参数是否应该使用Muon优化"""
        # 根据组名判断
        if "output" in group_name.lower():
            return False
        if "gamma" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "noreg" in group_name.lower():
            #logging.info(f"{group_name} is output")
            return False
        if "normal" in group_name.lower():
            assert group_name=="normal" or group_name=="normal_attn", f"Unknown group_name: {group_name}, you should add it to is_muon_group()"
            return True
        assert False, f"Unknown group_name: {group_name}, you should add it to is_muon_group()"
        # 默认情况下根据参数维度判断
        #return param.ndim >= 2
        
        return False

    def _local_muon_state_for_checkpoint(self, optimizer_state_dict):
        local_muon_state = {}
        if not self.is_distributed:
            return local_muon_state
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for group in optimizer_state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    continue
                if param_id not in optimizer_state_dict["state"]:
                    continue
                state = optimizer_state_dict["state"][param_id]
                if "momentum_buffer" not in state:
                    continue
                local_muon_state[param_id] = {
                    "momentum_buffer": state["momentum_buffer"].detach().cpu()
                }
        return local_muon_state

    def state_dict_for_checkpoint(self):
        optimizer_state_dict = super().state_dict()
        if not self.is_distributed:
            return optimizer_state_dict
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_muon_state = self._local_muon_state_for_checkpoint(optimizer_state_dict)
        gathered_muon_state = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_muon_state, gathered_muon_state, dst=0)
        if rank != 0:
            return None
        for rank_state in gathered_muon_state:
            if rank_state is None:
                continue
            for param_id, state in rank_state.items():
                optimizer_state_dict["state"][param_id] = state
        return optimizer_state_dict

    def load_state_dict_for_checkpoint(self, state_dict):
        if not self.is_distributed:
            super().load_state_dict(state_dict)
            return
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        muon_non_local_param_ids = set()
        for group in state_dict["param_groups"]:
            if not group.get("use_muon", False):
                continue
            for local_index, param_id in enumerate(group["params"]):
                if local_index % world_size != rank:
                    muon_non_local_param_ids.add(param_id)
        filtered_state = dict(state_dict["state"])
        for param_id in muon_non_local_param_ids:
            if param_id in filtered_state:
                del filtered_state[param_id]
        filtered_state_dict = {
            "state": filtered_state,
            "param_groups": state_dict["param_groups"],
        }
        super().load_state_dict(filtered_state_dict)

    def _initialize_muon_distributed_layouts(self):
        assert self.is_distributed
        world_size = dist.get_world_size()

        # Insertion order follows parameter traversal and is therefore identical
        # across ranks even though each rank's CUDA device index is different.
        layout_builders = {}
        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            for local_index, param in enumerate(group["params"]):
                if not param.is_contiguous():
                    raise ValueError(
                        "Distributed Muon parameter synchronization requires contiguous parameters, "
                        f"got shape={tuple(param.shape)} stride={param.stride()}"
                    )
                key = (param.device, param.dtype)
                if key not in layout_builders:
                    layout_builders[key] = {
                        "params": [],
                        "owner_param_numels": [[] for _ in range(world_size)],
                    }
                builder = layout_builders[key]
                param_index = len(builder["params"])
                builder["params"].append(param)
                owner = local_index % world_size
                builder["owner_param_numels"][owner].append((param_index, param.numel()))

        layouts = []
        total_buckets = 0
        total_workspace_bytes = 0
        for builder in layout_builders.values():
            params = tuple(builder["params"])
            if len(params) <= 0:
                continue
            element_size = params[0].element_size()
            bucket_cap_numel = max(1, self.distributed_bucket_cap_bytes // element_size)
            buckets = _build_muon_flat_bucket_plan(
                builder["owner_param_numels"],
                bucket_cap_numel,
            )
            if len(buckets) <= 0:
                continue
            max_collective_numel = max(bucket.collective_numel for bucket in buckets)
            send_buffer = torch.empty(
                max_collective_numel,
                dtype=params[0].dtype,
                device=params[0].device,
            )
            gathered_buffer = torch.empty(
                world_size * max_collective_numel,
                dtype=params[0].dtype,
                device=params[0].device,
            )
            layouts.append(_MuonDistributedLayout(
                params=params,
                buckets=buckets,
                send_buffer=send_buffer,
                gathered_buffer=gathered_buffer,
            ))
            total_buckets += len(buckets)
            total_workspace_bytes += (world_size + 1) * max_collective_numel * element_size

        self._muon_distributed_layouts = tuple(layouts)
        logging.info(
            "Muon DDP flat parameter synchronization: %d bucket(s), %.1f MiB reusable workspace per rank",
            total_buckets,
            total_workspace_bytes / (1024.0 * 1024.0),
        )

    def _synchronize_muon_parameters(self):
        assert self.is_distributed
        if self._muon_distributed_layouts is None:
            self._initialize_muon_distributed_layouts()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        for layout in self._muon_distributed_layouts:
            for bucket in layout.buckets:
                collective_numel = bucket.collective_numel
                owner_numel = bucket.owner_numels[rank]
                send = layout.send_buffer[:collective_numel]

                local_parts = [
                    layout.params[segment.param_index].detach().view(-1)[
                        segment.param_offset:segment.param_offset + segment.numel
                    ]
                    for segment in bucket.segments_by_owner[rank]
                ]
                if len(local_parts) == 1:
                    send[:owner_numel].copy_(local_parts[0])
                elif len(local_parts) > 1:
                    torch.cat(local_parts, dim=0, out=send[:owner_numel])
                else:
                    assert owner_numel == 0
                if owner_numel < collective_numel:
                    send[owner_numel:].zero_()

                gathered = layout.gathered_buffer[:world_size * collective_numel]
                dist.all_gather_into_tensor(gathered, send)
                gathered_by_owner = gathered.view(world_size, collective_numel)

                destination_parts = []
                source_parts = []
                for owner in range(world_size):
                    if owner == rank:
                        continue
                    for segment in bucket.segments_by_owner[owner]:
                        destination_parts.append(
                            layout.params[segment.param_index].view(-1)[
                                segment.param_offset:segment.param_offset + segment.numel
                            ]
                        )
                        source_parts.append(
                            gathered_by_owner[owner, segment.packed_offset:segment.packed_offset + segment.numel]
                        )
                if len(destination_parts) > 0:
                    torch._foreach_copy_(destination_parts, source_parts)

    def _validate_distributed_muon_gradients_once(self):
        if not self.is_distributed or self._distributed_muon_gradients_validated:
            return

        muon_params = [
            param
            for group in self.param_groups
            if group["use_muon"]
            for param in group["params"]
        ]
        if len(muon_params) <= 0:
            self._distributed_muon_gradients_validated = True
            return

        missing_params = [param for param in muon_params if param.grad is None]
        missing_on_any_rank = torch.tensor(
            [1 if missing_params else 0],
            dtype=torch.int32,
            device=muon_params[0].device,
        )
        dist.all_reduce(missing_on_any_rank, op=dist.ReduceOp.MAX)
        if missing_on_any_rank.item() != 0:
            local_shapes = [tuple(param.shape) for param in missing_params]
            raise RuntimeError(
                "Distributed Muon requires gradients for every Muon parameter on every rank; "
                f"missing local parameter shapes: {local_shapes}"
            )
        self._distributed_muon_gradients_validated = True

    def _step_muon_group_batched(self, group, param_indices, muon_lr):
        entries_by_normalized_shape = {}

        # Preserve the original per-parameter momentum and Nesterov mutation
        # order. Only the independent Newton-Schulz computations are regrouped.
        for param_index in param_indices:
            param = group["params"][param_index]
            state = self.state[param]
            if len(state) == 0:
                state["momentum_buffer"] = torch.zeros_like(param)
            momentum = state["momentum_buffer"]
            momentum.lerp_(param.grad, 1 - group["momentum"])
            update = param.grad.lerp_(momentum, group["momentum"])

            original_shape = update.shape
            dims_gt_4 = sum(1 for size in original_shape if size > 4)
            if dims_gt_4 <= 1:
                raise ValueError(
                    f"Muon shape check failed: original_shape {original_shape} has only "
                    f"{dims_gt_4} dimensions greater than 4 (must be > 1)."
                )
            matrix = update
            if matrix.ndim == 4:
                matrix = matrix.view(len(matrix), -1)
            assert matrix.ndim == 2
            if matrix.shape[0] <= 4 or matrix.shape[1] <= 4:
                raise ValueError(
                    f"Muon shape check failed: original_shape {original_shape} became {matrix.shape}"
                )

            was_transposed = matrix.shape[0] > matrix.shape[1]
            normalized_matrix = matrix.mT if was_transposed else matrix
            key = (
                normalized_matrix.device,
                normalized_matrix.dtype,
                normalized_matrix.shape[0],
                normalized_matrix.shape[1],
            )
            entries_by_normalized_shape.setdefault(key, []).append((
                param,
                original_shape,
                normalized_matrix,
                was_transposed,
                max(1, max(param.grad.size())) ** 0.5,
            ))

        for entries in entries_by_normalized_shape.values():
            for chunk_begin in range(0, len(entries), self.muon_ns_batch_size):
                chunk = entries[chunk_begin:chunk_begin + self.muon_ns_batch_size]
                matrices = torch.stack([entry[2] for entry in chunk], dim=0)
                updates = zeropower_via_newtonschulz5_batched(matrices, steps=5)
                for entry, update in zip(chunk, updates.unbind(dim=0)):
                    param, original_shape, _, was_transposed, scale = entry
                    if was_transposed:
                        update = update.mT
                    update *= scale
                    if len(original_shape) == 4:
                        update = update.view(original_shape)
                    else:
                        assert update.shape == original_shape
                    assert update.shape == param.shape
                    param.mul_(1 - muon_lr * group["weight_decay"])
                    param.add_(update, alpha=-muon_lr)

    def _step_aux_adam_group_foreach(self, group, adam_lr):
        """Update an auxiliary Adam group with one multi-tensor launch per operation."""
        entries_by_step = {}
        for param in group["params"]:
            state = self.state[param]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            if "step" not in state:
                state["step"] = 100000
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)

            state["step"] += 1
            step = state["step"]
            if not isinstance(step, int):
                raise TypeError(
                    "Foreach auxiliary Adam requires integer optimizer steps, "
                    f"got {type(step).__name__}"
                )
            entries_by_step.setdefault(step, []).append((
                param,
                param.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
            ))

        beta1, beta2 = group["betas"]
        for step, entries in entries_by_step.items():
            params, grads, exp_avgs, exp_avg_sqs = map(list, zip(*entries))
            torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(
                exp_avg_sqs,
                grads,
                grads,
                value=1 - beta2,
            )

            bias_correction1 = 1 - beta1**step
            bias_correction2_sqrt = (1 - beta2**step) ** 0.5
            denominators = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denominators, bias_correction2_sqrt)
            torch._foreach_add_(denominators, group["eps"])
            updates = torch._foreach_div(exp_avgs, denominators)

            torch._foreach_mul_(
                params,
                1 - adam_lr * group["weight_decay"],
            )
            torch._foreach_add_(
                params,
                updates,
                alpha=-adam_lr / bias_correction1,
            )


    @torch.no_grad()
    def step(self):
        self._validate_distributed_muon_gradients_once()
        for group in self.param_groups:
            muon_lr = group["lr"] 
            adam_lr = group["lr"] / group["muon_lr_multiplier"]

        
            # 原有逻辑保持不变 (显式指定use_muon的情况)
            if group["use_muon"]:
                params = group["params"]
                if self.is_distributed:
                    param_indices = range(dist.get_rank(), len(params), dist.get_world_size())
                else:
                    param_indices = range(len(params))
                if self.use_batched_muon_ns:
                    self._step_muon_group_batched(group, param_indices, muon_lr)
                    continue
                for param_index in param_indices:
                    p = params[param_index]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update_kimi(p.grad, state["momentum_buffer"], beta=group["momentum"])

                    # 确保update和p的形状一致
                    assert update.shape == p.shape

                    p.mul_(1 - muon_lr * group["weight_decay"])
                    p.add_(update, alpha=-muon_lr)
            else:
                if "betas" not in group:
                    group["betas"] = group.get("betas", (self.momentum_default,  0.995))
                if self.use_foreach_aux_adam:
                    self._step_aux_adam_group_foreach(group, adam_lr)
                    continue
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    
                    if "step" not in state:
                        state["step"] = 100000 
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                        state["step"], group["betas"], group["eps"])
                    p.mul_(1 - adam_lr * group["weight_decay"])
                    p.add_(update, alpha=-adam_lr)
        if self.is_distributed:
            self._synchronize_muon_parameters()
