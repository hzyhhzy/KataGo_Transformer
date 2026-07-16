import os
import socket
import unittest
from unittest import mock

import torch

from model_pytorch import Model, SparseMoE, TransformerRoPEGQABlock


def _run_moe_ddp_worker(rank: int, world_size: int, master_port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=world_size
    )
    try:
        torch.manual_seed(0)
        moe = SparseMoE(
            c_main=2,
            ffn_dim=4,
            num_experts=4,
            top_k=1,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        moe.balance_bias_update_rate = 0.001
        moe.defer_balance_bias_update = True
        with torch.no_grad():
            moe.router.weight.copy_(torch.tensor([
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]))
        ddp_moe = torch.nn.parallel.DistributedDataParallel(moe)
        direction = 1.0 if rank == 0 else -1.0
        x = torch.tensor([[[direction, 0.0], [direction, 0.0]]])
        output, load_balance_loss = ddp_moe(x, torch.ones(1, 2))
        (output.square().mean() + 0.01 * load_balance_loss).backward()

        Model._synchronize_moe_assignment_load([moe], True)

        torch.testing.assert_close(
            moe.last_assignment_fraction,
            torch.tensor([0.5, 0.5, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            moe.router_selection_bias,
            torch.tensor([-0.001, -0.001, 0.001, 0.001]),
        )
        gathered_biases = [torch.empty_like(moe.router_selection_bias) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_biases, moe.router_selection_bias)
        for bias in gathered_biases[1:]:
            torch.testing.assert_close(bias, gathered_biases[0])
    finally:
        torch.distributed.destroy_process_group()


class SparseMoETest(unittest.TestCase):
    def _run_moe(self, routing_mode: str):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=routing_mode,
            activation="silu",
            use_swiglu=True,
        )
        x = torch.randn(3, 9, 8, requires_grad=True)
        mask = torch.ones(3, 1, 3, 3)
        mask[1, 0, 2, :] = 0

        output, load_balance_loss = moe(x, mask)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(load_balance_loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(load_balance_loss))
        self.assertTrue(torch.all(output[1, 6:] == 0))

        (output.square().mean() + 0.01 * load_balance_loss).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in moe.parameters()))

    def test_token_routing_forward_and_backward(self):
        self._run_moe(SparseMoE.TOKEN_ROUTING)

    def test_board_routing_forward_and_backward(self):
        self._run_moe(SparseMoE.BOARD_ROUTING)

    def test_board_routing_reuses_one_route_for_all_board_tokens(self):
        moe = SparseMoE(
            c_main=2,
            ffn_dim=4,
            num_experts=3,
            top_k=2,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        x = torch.tensor([[[3.0, 0.0], [-3.0, 0.0], [1.0, 0.0]]])
        mask = torch.ones(1, 1, 1, 3)
        valid = mask.reshape(1, 3).bool()
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)

        board_indices, board_weights, _ = moe._route(pooled)
        expanded_indices = board_indices.unsqueeze(1).expand(-1, 3, -1)
        expanded_weights = board_weights.unsqueeze(1).expand(-1, 3, -1)

        self.assertTrue(torch.equal(expanded_indices[:, 0], expanded_indices[:, 2]))
        self.assertTrue(torch.equal(expanded_weights[:, 0], expanded_weights[:, 2]))

    def test_batched_board_runtime_switch_matches_sequential_dispatch(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        x = torch.randn(5, 4, 8)
        mask = torch.ones(5, 4)
        sequential_output, sequential_loss = moe(x, mask)
        moe.batched_experts_per_group = 4
        batched_output, batched_loss = moe(x, mask)
        torch.testing.assert_close(batched_output, sequential_output)
        torch.testing.assert_close(batched_loss, sequential_loss)

    def test_moe_load_stats_collection(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        moe.collect_load_stats = True
        x = torch.randn(4, 4, 8)
        mask = torch.ones(4, 4)
        moe(x, mask)
        self.assertAlmostEqual(
            moe.last_assignment_fraction.sum().item(), 1.0, places=6
        )

    def test_selection_bias_tracks_actual_top_k_load(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        moe.balance_bias_update_rate = 0.001
        with torch.no_grad():
            moe.router.weight.zero_()

        first_indices, _, _ = moe._route(torch.zeros(16, 8))
        first_assignment_fraction = torch.nn.functional.one_hot(
            first_indices, num_classes=moe.num_experts
        ).float().mean(dim=(0, 1))
        expected_bias = 0.001 * torch.sign(
            1.0 / moe.num_experts - first_assignment_fraction
        )
        expected_bias = expected_bias - expected_bias.mean()
        torch.testing.assert_close(moe.router_selection_bias, expected_bias)

        second_indices, _, _ = moe._route(torch.zeros(16, 8))

        self.assertFalse(torch.equal(first_indices, second_indices))
        self.assertAlmostEqual(moe.router_selection_bias.sum().item(), 0.0, places=6)

    def test_deferred_selection_bias_uses_globally_weighted_ddp_load(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=1,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        moe.balance_bias_update_rate = 0.001
        moe.defer_balance_bias_update = True
        moe.last_assignment_fraction.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        moe.last_assignment_weight.fill_(2.0)

        # Simulate another rank with six assignments, all routed to expert 1.
        other_rank_packed_load = torch.tensor([0.0, 6.0, 0.0, 0.0, 6.0])

        def add_other_rank_load(packed_load, op):
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            packed_load.add_(other_rank_packed_load)

        with mock.patch(
            "torch.distributed.all_reduce", side_effect=add_other_rank_load
        ) as all_reduce:
            Model._synchronize_moe_assignment_load([moe], True)

        all_reduce.assert_called_once()
        torch.testing.assert_close(
            moe.last_assignment_fraction,
            torch.tensor([0.25, 0.75, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            moe.router_selection_bias,
            torch.tensor([-0.00025, -0.00125, 0.00075, 0.00075]),
        )
        self.assertAlmostEqual(moe.router_selection_bias.sum().item(), 0.0, places=7)

    @unittest.skipIf(
        os.name == "nt",
        "This PyTorch Windows build cannot create a Gloo process-group device",
    )
    def test_two_rank_ddp_forward_backward_and_bias_sync(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            master_port = sock.getsockname()[1]
        torch.multiprocessing.spawn(
            _run_moe_ddp_worker,
            args=(2, master_port),
            nprocs=2,
            join=True,
        )

    def test_old_checkpoint_without_selection_bias_loads_strictly(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        old_state_dict = moe.state_dict()
        del old_state_dict["router_selection_bias"]

        moe.load_state_dict(old_state_dict, strict=True)
        torch.testing.assert_close(
            moe.router_selection_bias, torch.zeros_like(moe.router_selection_bias)
        )

    def test_router_stays_float32_under_autocast(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=2,
            routing_mode=SparseMoE.TOKEN_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, weights, load_balance_loss = moe._route(torch.randn(6, 8))

        self.assertEqual(weights.dtype, torch.float32)
        self.assertEqual(load_balance_loss.dtype, torch.float32)

    def test_top_one_gate_is_one_with_task_gradient(self):
        moe = SparseMoE(
            c_main=8,
            ffn_dim=16,
            num_experts=4,
            top_k=1,
            routing_mode=SparseMoE.BOARD_ROUTING,
            activation="silu",
            use_swiglu=True,
        )
        _, weights, _ = moe._route(torch.randn(32, 8))

        torch.testing.assert_close(weights, torch.ones_like(weights))
        weights.sum().backward()
        self.assertIsNotNone(moe.router.weight.grad)
        self.assertGreater(moe.router.weight.grad.norm().item(), 0.0)

    def test_tflrs_block_returns_auxiliary_loss(self):
        config = {
            "norm_kind": "fixup",
            "transformer_ffn_channels": 16,
            "transformer_heads": 2,
            "transformer_kv_heads": 2,
            "learnable_rope": True,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_routing_mode": "token",
        }
        block = TransformerRoPEGQABlock(
            name="test",
            c_main=8,
            config=config,
            activation="silu",
            pos_len=3,
            use_swiglu=True,
            use_rope=True,
        )
        x = torch.randn(2, 8, 3, 3)
        mask = torch.ones(2, 1, 3, 3)

        output, load_balance_loss = block(x, mask, None, mask.sum(), None)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(load_balance_loss))

        reg_dict = {
            "normal": [],
            "normal_attn": [],
            "normal_router": [],
            "noreg": [],
        }
        block.add_reg_dict(reg_dict)
        router_ids = {id(parameter) for parameter in reg_dict["normal_router"]}
        normal_ids = {id(parameter) for parameter in reg_dict["normal"]}
        self.assertIn(id(block.moe.router.weight), router_ids)
        self.assertNotIn(id(block.moe.router.weight), normal_ids)
        self.assertIn(id(block.moe.experts[0].linear1.weight), normal_ids)


if __name__ == "__main__":
    unittest.main()
