import io
import math
import os
import unittest
from unittest import mock

import torch

try:
    from . import muon_kissin
except ImportError:
    import muon_kissin

_build_muon_flat_bucket_plan = muon_kissin._build_muon_flat_bucket_plan


class MuonFlatBucketPlanTest(unittest.TestCase):
    def test_performance_paths_are_enabled_by_default(self):
        param = torch.nn.Parameter(torch.ones(5, 7))
        env = dict(os.environ)
        for name in (
            "KATAGO_MUON_BATCHED_NS",
            "KATAGO_MUON_NS_BATCH_SIZE",
            "KATAGO_AUX_ADAM_FOREACH",
        ):
            env.pop(name, None)
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                optimizer = muon_kissin.MuonWithAuxAdamKimi([{
                    "params": [param],
                    "group_name": "output",
                    "use_muon": False,
                }])

        self.assertTrue(optimizer.use_batched_muon_ns)
        self.assertEqual(optimizer.muon_ns_batch_size, 32)
        self.assertTrue(optimizer.use_foreach_aux_adam)

    def test_full_checkpoint_filters_muon_state_for_new_world_size(self):
        def make_groups():
            muon_params = [
                torch.nn.Parameter(torch.full((size, size), float(size)))
                for size in (9, 8, 7, 6)
            ]
            aux_param = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 5))
            groups = [
                {
                    "params": muon_params,
                    "group_name": "normal",
                    "use_muon": True,
                },
                {
                    "params": [aux_param],
                    "group_name": "output",
                    "use_muon": False,
                },
            ]
            return groups, muon_params, aux_param

        reference_groups, reference_muon_params, reference_aux_param = make_groups()
        with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
            reference = muon_kissin.MuonWithAuxAdamKimi(reference_groups)
        for index, param in enumerate(reference_muon_params):
            reference.state[param]["momentum_buffer"] = torch.full_like(
                param, 0.1 * (index + 1)
            )
        reference.state[reference_aux_param] = {
            "exp_avg": torch.full_like(reference_aux_param, 0.25),
            "exp_avg_sq": torch.full_like(reference_aux_param, 0.5),
            "step": 11,
        }
        full_checkpoint = reference.state_dict_for_checkpoint()

        for world_size, rank in ((2, 1), (3, 2)):
            with self.subTest(world_size=world_size, rank=rank):
                groups, muon_params, aux_param = make_groups()
                with mock.patch.object(
                    muon_kissin.dist, "is_initialized", return_value=True
                ), mock.patch.object(
                    muon_kissin.dist, "get_world_size", return_value=world_size
                ), mock.patch.object(
                    muon_kissin.dist, "get_rank", return_value=rank
                ):
                    optimizer = muon_kissin.MuonWithAuxAdamKimi(groups)
                    optimizer.load_state_dict_for_checkpoint(full_checkpoint)

                for index, param in enumerate(muon_params):
                    if index % world_size == rank:
                        expected = torch.full_like(param, 0.1 * (index + 1))
                        self.assertTrue(torch.equal(
                            optimizer.state[param]["momentum_buffer"], expected
                        ))
                    else:
                        self.assertNotIn(param, optimizer.state)
                self.assertEqual(optimizer.state[aux_param]["step"], 11)
                self.assertTrue(torch.equal(
                    optimizer.state[aux_param]["exp_avg"],
                    torch.full_like(aux_param, 0.25),
                ))

    def assert_valid_plan(self, owner_streams, bucket_cap_numel):
        plan = _build_muon_flat_bucket_plan(owner_streams, bucket_cap_numel)
        world_size = len(owner_streams)
        expected_num_buckets = math.ceil(
            max((sum(numel for _, numel in stream) for stream in owner_streams), default=0)
            / bucket_cap_numel
        )
        self.assertEqual(len(plan), expected_num_buckets)

        expected_sizes = {
            (owner, param_index): numel
            for owner, stream in enumerate(owner_streams)
            for param_index, numel in stream
            if numel > 0
        }
        covered_ranges = {key: [] for key in expected_sizes}

        for bucket in plan:
            self.assertEqual(len(bucket.owner_numels), world_size)
            self.assertEqual(len(bucket.segments_by_owner), world_size)
            self.assertEqual(bucket.collective_numel, max(bucket.owner_numels))
            self.assertGreater(bucket.collective_numel, 0)
            self.assertLessEqual(bucket.collective_numel, bucket_cap_numel)

            for owner in range(world_size):
                packed_offset = 0
                for segment in bucket.segments_by_owner[owner]:
                    self.assertEqual(segment.packed_offset, packed_offset)
                    self.assertGreater(segment.numel, 0)
                    packed_offset += segment.numel
                    key = (owner, segment.param_index)
                    self.assertIn(key, expected_sizes)
                    self.assertLessEqual(
                        segment.param_offset + segment.numel,
                        expected_sizes[key],
                    )
                    covered_ranges[key].append(
                        (segment.param_offset, segment.param_offset + segment.numel)
                    )
                self.assertEqual(packed_offset, bucket.owner_numels[owner])
                self.assertLessEqual(packed_offset, bucket_cap_numel)

        for key, numel in expected_sizes.items():
            ranges = sorted(covered_ranges[key])
            cursor = 0
            for begin, end in ranges:
                self.assertEqual(begin, cursor)
                cursor = end
            self.assertEqual(cursor, numel)

        return plan

    def test_world_sizes_one_through_four(self):
        param_sizes = [17, 3, 11, 8, 2, 23, 5]
        for world_size in range(1, 5):
            with self.subTest(world_size=world_size):
                owner_streams = [[] for _ in range(world_size)]
                for param_index, numel in enumerate(param_sizes):
                    owner_streams[param_index % world_size].append((param_index, numel))
                self.assert_valid_plan(owner_streams, bucket_cap_numel=7)

    def test_n_less_than_world_size_and_empty_owners(self):
        owner_streams = [
            [(0, 9)],
            [(1, 2)],
            [],
            [],
        ]
        plan = self.assert_valid_plan(owner_streams, bucket_cap_numel=4)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[-1].owner_numels, (1, 0, 0, 0))

    def test_parameter_can_span_multiple_buckets(self):
        owner_streams = [
            [(0, 13), (2, 2)],
            [(1, 4), (3, 8)],
            [(4, 1)],
        ]
        plan = self.assert_valid_plan(owner_streams, bucket_cap_numel=5)
        param_zero_segments = [
            segment
            for bucket in plan
            for segment in bucket.segments_by_owner[0]
            if segment.param_index == 0
        ]
        self.assertEqual(
            [(segment.param_offset, segment.numel) for segment in param_zero_segments],
            [(0, 5), (5, 5), (10, 3)],
        )

    def test_plan_reconstructs_heterogeneous_owner_payloads(self):
        owner_streams = [
            [(0, 3), (3, 12)],
            [(1, 14)],
            [(2, 1), (4, 4)],
            [],
        ]
        plan = self.assert_valid_plan(owner_streams, bucket_cap_numel=5)
        sources = {
            param_index: torch.arange(numel, dtype=torch.float32) + 1000 * param_index
            for stream in owner_streams
            for param_index, numel in stream
        }
        reconstructed_by_rank = [
            {param_index: torch.empty_like(source) for param_index, source in sources.items()}
            for _ in owner_streams
        ]

        for bucket in plan:
            gathered = torch.zeros(len(owner_streams), bucket.collective_numel)
            for owner, segments in enumerate(bucket.segments_by_owner):
                for segment in segments:
                    gathered[
                        owner,
                        segment.packed_offset:segment.packed_offset + segment.numel,
                    ].copy_(sources[segment.param_index][
                        segment.param_offset:segment.param_offset + segment.numel
                    ])
            for rank_params in reconstructed_by_rank:
                for owner, segments in enumerate(bucket.segments_by_owner):
                    for segment in segments:
                        rank_params[segment.param_index][
                            segment.param_offset:segment.param_offset + segment.numel
                        ].copy_(gathered[
                            owner,
                            segment.packed_offset:segment.packed_offset + segment.numel,
                        ])

        for rank_params in reconstructed_by_rank:
            for param_index, source in sources.items():
                self.assertTrue(torch.equal(rank_params[param_index], source))

    def test_empty_and_zero_sized_parameters(self):
        self.assertEqual(
            _build_muon_flat_bucket_plan([[(0, 0)], [], []], bucket_cap_numel=4),
            (),
        )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            _build_muon_flat_bucket_plan([], bucket_cap_numel=4)
        with self.assertRaises(ValueError):
            _build_muon_flat_bucket_plan([[]], bucket_cap_numel=0)
        with self.assertRaises(ValueError):
            _build_muon_flat_bucket_plan([[(-1, 2)]], bucket_cap_numel=4)
        with self.assertRaises(ValueError):
            _build_muon_flat_bucket_plan([[(0, -2)]], bucket_cap_numel=4)

    def test_runtime_layout_preserves_group_local_owner_mapping(self):
        first_group = [torch.nn.Parameter(torch.zeros(6, size)) for size in (9, 8, 7)]
        second_group = [torch.nn.Parameter(torch.zeros(6, size)) for size in (6, 5)]
        with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=True):
            optimizer = muon_kissin.MuonWithAuxAdamKimi(
                [
                    {"params": first_group, "group_name": "normal", "use_muon": True},
                    {"params": second_group, "group_name": "normal_attn", "use_muon": True},
                ],
                distributed_bucket_cap_bytes=16,
            )
        with mock.patch.object(muon_kissin.dist, "get_world_size", return_value=4):
            optimizer._initialize_muon_distributed_layouts()

        self.assertEqual(len(optimizer._muon_distributed_layouts), 1)
        layout = optimizer._muon_distributed_layouts[0]
        owner_by_param_index = {}
        for bucket in layout.buckets:
            for owner, segments in enumerate(bucket.segments_by_owner):
                for segment in segments:
                    if segment.param_index in owner_by_param_index:
                        self.assertEqual(owner_by_param_index[segment.param_index], owner)
                    owner_by_param_index[segment.param_index] = owner

        param_index_by_identity = {id(param): index for index, param in enumerate(layout.params)}
        for group in optimizer.param_groups:
            for local_index, param in enumerate(group["params"]):
                self.assertEqual(
                    owner_by_param_index[param_index_by_identity[id(param)]],
                    local_index % 4,
                )

    def test_distributed_constructor_validates_every_muon_shape_before_sharding(self):
        invalid_param = torch.nn.Parameter(torch.zeros(4, 9))
        with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=True):
            with self.assertRaisesRegex(ValueError, "Muon shape check failed"):
                muon_kissin.MuonWithAuxAdamKimi([{
                    "params": [invalid_param],
                    "group_name": "normal",
                    "use_muon": True,
                }])

    def test_distributed_step_reports_missing_muon_grad_before_owner_work(self):
        params = [
            torch.nn.Parameter(torch.zeros(7, 6)),
            torch.nn.Parameter(torch.zeros(8, 6)),
        ]
        params[0].grad = torch.ones_like(params[0])
        with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=True):
            optimizer = muon_kissin.MuonWithAuxAdamKimi([{
                "params": params,
                "group_name": "normal",
                "use_muon": True,
            }])

        with mock.patch.object(muon_kissin.dist, "all_reduce") as all_reduce:
            with mock.patch.object(
                muon_kissin.dist,
                "get_rank",
                side_effect=AssertionError("owner work should not begin"),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires gradients"):
                    optimizer.step()
        all_reduce.assert_called_once()

    def test_distributed_step_propagates_remote_missing_grad_before_owner_work(self):
        params = [torch.nn.Parameter(torch.zeros(7, 6))]
        params[0].grad = torch.ones_like(params[0])
        with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=True):
            optimizer = muon_kissin.MuonWithAuxAdamKimi([{
                "params": params,
                "group_name": "normal",
                "use_muon": True,
            }])

        def report_remote_missing(flag, op):
            self.assertEqual(op, muon_kissin.dist.ReduceOp.MAX)
            flag.fill_(1)

        with mock.patch.object(
            muon_kissin.dist,
            "all_reduce",
            side_effect=report_remote_missing,
        ):
            with mock.patch.object(
                muon_kissin.dist,
                "get_rank",
                side_effect=AssertionError("owner work should not begin"),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires gradients"):
                    optimizer.step()

    def test_single_process_step_updates_every_muon_parameter_without_layout(self):
        params = [
            torch.nn.Parameter(torch.full((6, 7), 2.0)),
            torch.nn.Parameter(torch.full((8, 5), -1.0)),
        ]
        grads = [torch.full_like(params[0], 0.25), torch.full_like(params[1], -0.5)]
        for param, grad in zip(params, grads):
            param.grad = grad

        with mock.patch.dict(os.environ, {"KATAGO_MUON_BATCHED_NS": "0"}):
            with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                optimizer = muon_kissin.MuonWithAuxAdamKimi([
                    {"params": params, "group_name": "normal", "use_muon": True},
                ])
        optimizer.param_groups[0]["lr"] = 0.2
        optimizer.param_groups[0]["weight_decay"] = 0.1
        originals = {id(param): param.detach().clone() for param in params}

        def fake_muon_update(grad, momentum, beta):
            momentum.copy_(grad)
            return grad * 2.0

        with mock.patch.object(muon_kissin, "muon_update_kimi", side_effect=fake_muon_update):
            optimizer.step()

        for param in params:
            expected = originals[id(param)].mul_(1.0 - 0.2 * 0.1)
            expected.add_(param.grad * 2.0, alpha=-0.2)
            self.assertTrue(torch.equal(param, expected))
            self.assertTrue(torch.equal(optimizer.state[param]["momentum_buffer"], param.grad))
        self.assertIsNone(optimizer._muon_distributed_layouts)

    def test_batched_ns_matches_sequential_for_transposes_and_4d(self):
        group_shapes = [
            [(9, 6), (6, 9), (9, 6), (6, 5, 2, 2), (20, 6)],
            [(6, 9), (9, 6)],
        ]
        group_hyperparameters = [
            ("normal", 0.83, 0.020, 0.040),
            ("normal_attn", 0.91, 0.015, 0.025),
        ]

        def make_optimizer(use_batched):
            logical_groups = []
            flat_param_index = 0
            for shapes in group_shapes:
                params = []
                for shape in shapes:
                    numel = math.prod(shape)
                    values = torch.arange(numel, dtype=torch.float32).reshape(shape)
                    values = values.mul(0.001).add(0.1 * (flat_param_index + 1))
                    param = torch.nn.Parameter(values)
                    grad = torch.arange(numel, dtype=torch.float32).reshape(shape)
                    param.grad = grad.mul(0.0007).add(-0.03 + 0.01 * flat_param_index)
                    params.append(param)
                    flat_param_index += 1
                logical_groups.append(params)

            optimizer_groups = []
            for params, (group_name, momentum, _, weight_decay) in zip(
                logical_groups, group_hyperparameters
            ):
                optimizer_groups.append({
                    "params": params,
                    "group_name": group_name,
                    "use_muon": True,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                })
            environment = {
                "KATAGO_MUON_BATCHED_NS": "1" if use_batched else "0",
                "KATAGO_MUON_NS_BATCH_SIZE": "2",
            }
            with mock.patch.dict(os.environ, environment):
                with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                    optimizer = muon_kissin.MuonWithAuxAdamKimi(optimizer_groups)
            for group, (_, _, learning_rate, _) in zip(
                optimizer.param_groups, group_hyperparameters
            ):
                group["lr"] = learning_rate
            return optimizer, logical_groups

        sequential_optimizer, sequential_groups = make_optimizer(use_batched=False)
        batched_optimizer, batched_groups = make_optimizer(use_batched=True)
        sequential_ns = getattr(
            muon_kissin.zeropower_via_newtonschulz5,
            "_torchdynamo_orig_callable",
            muon_kissin.zeropower_via_newtonschulz5,
        )
        batched_ns = getattr(
            muon_kissin.zeropower_via_newtonschulz5_batched,
            "_torchdynamo_orig_callable",
            muon_kissin.zeropower_via_newtonschulz5_batched,
        )
        with mock.patch.object(muon_kissin, "zeropower_via_newtonschulz5", sequential_ns):
            with mock.patch.object(
                muon_kissin, "zeropower_via_newtonschulz5_batched", batched_ns
            ):
                sequential_optimizer.step()
                batched_optimizer.step()

        self.assertFalse(sequential_optimizer.use_batched_muon_ns)
        self.assertTrue(batched_optimizer.use_batched_muon_ns)
        self.assertEqual(batched_optimizer.muon_ns_batch_size, 2)
        for sequential_params, batched_params in zip(sequential_groups, batched_groups):
            for sequential_param, batched_param in zip(sequential_params, batched_params):
                torch.testing.assert_close(
                    batched_param,
                    sequential_param,
                    rtol=5e-3,
                    atol=5e-4,
                )
                self.assertTrue(torch.equal(batched_param.grad, sequential_param.grad))
                self.assertTrue(torch.equal(
                    batched_optimizer.state[batched_param]["momentum_buffer"],
                    sequential_optimizer.state[sequential_param]["momentum_buffer"],
                ))

    def test_foreach_aux_adam_matches_sequential_across_steps(self):
        def make_optimizer(use_foreach):
            params = [
                torch.nn.Parameter(torch.linspace(-0.4, 0.7, 35).reshape(5, 7)),
                torch.nn.Parameter(torch.linspace(0.2, 0.8, 11)),
                torch.nn.Parameter(torch.tensor(-0.3)),
            ]
            environment = {
                "KATAGO_AUX_ADAM_FOREACH": "1" if use_foreach else "0",
            }
            with mock.patch.dict(os.environ, environment):
                with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                    optimizer = muon_kissin.MuonWithAuxAdamKimi([{
                        "params": params,
                        "group_name": "output",
                        "use_muon": False,
                        "betas": (0.87, 0.993),
                        "eps": 3e-7,
                        "weight_decay": 0.04,
                        "muon_lr_multiplier": 5.0,
                    }])
            optimizer.param_groups[0]["lr"] = 0.035
            return optimizer, params

        sequential_optimizer, sequential_params = make_optimizer(use_foreach=False)
        foreach_optimizer, foreach_params = make_optimizer(use_foreach=True)
        for step in range(4):
            for param_index, (sequential_param, foreach_param) in enumerate(
                zip(sequential_params, foreach_params)
            ):
                grad = torch.full_like(
                    sequential_param,
                    -0.08 + 0.03 * step + 0.01 * param_index,
                )
                sequential_param.grad = grad.clone()
                foreach_param.grad = grad.clone()
            sequential_optimizer.step()
            foreach_optimizer.step()

        self.assertFalse(sequential_optimizer.use_foreach_aux_adam)
        self.assertTrue(foreach_optimizer.use_foreach_aux_adam)
        for sequential_param, foreach_param in zip(sequential_params, foreach_params):
            torch.testing.assert_close(foreach_param, sequential_param, rtol=2e-6, atol=2e-7)
            sequential_state = sequential_optimizer.state[sequential_param]
            foreach_state = foreach_optimizer.state[foreach_param]
            self.assertEqual(foreach_state["step"], sequential_state["step"])
            torch.testing.assert_close(
                foreach_state["exp_avg"], sequential_state["exp_avg"], rtol=2e-6, atol=2e-7
            )
            torch.testing.assert_close(
                foreach_state["exp_avg_sq"], sequential_state["exp_avg_sq"], rtol=2e-6, atol=2e-7
            )

    def test_foreach_aux_adam_handles_loaded_parameters_at_different_steps(self):
        def make_optimizer(use_foreach):
            params = [
                torch.nn.Parameter(torch.linspace(-0.5, 0.6, 20).reshape(4, 5)),
                torch.nn.Parameter(torch.linspace(0.1, 0.9, 9)),
            ]
            with mock.patch.dict(
                os.environ,
                {"KATAGO_AUX_ADAM_FOREACH": "1" if use_foreach else "0"},
            ):
                with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                    optimizer = muon_kissin.MuonWithAuxAdamKimi([{
                        "params": params,
                        "group_name": "output",
                        "use_muon": False,
                        "betas": (0.9, 0.997),
                        "eps": 1e-7,
                        "weight_decay": 0.03,
                    }])
            optimizer.param_groups[0]["lr"] = 0.02
            return optimizer, params

        sequential_optimizer, sequential_params = make_optimizer(False)
        foreach_optimizer, foreach_params = make_optimizer(True)
        for index, (sequential_param, foreach_param, step) in enumerate(
            zip(sequential_params, foreach_params, (3, 17))
        ):
            grad = torch.full_like(sequential_param, 0.04 + 0.01 * index)
            sequential_param.grad = grad.clone()
            foreach_param.grad = grad.clone()
            exp_avg = torch.full_like(sequential_param, -0.02 + 0.005 * index)
            exp_avg_sq = torch.full_like(sequential_param, 0.03 + 0.002 * index)
            sequential_optimizer.state[sequential_param] = {
                "exp_avg": exp_avg.clone(),
                "exp_avg_sq": exp_avg_sq.clone(),
                "step": step,
            }
            foreach_optimizer.state[foreach_param] = {
                "exp_avg": exp_avg.clone(),
                "exp_avg_sq": exp_avg_sq.clone(),
                "step": step,
            }

        sequential_optimizer.step()
        foreach_optimizer.step()

        for sequential_param, foreach_param in zip(sequential_params, foreach_params):
            torch.testing.assert_close(foreach_param, sequential_param, rtol=2e-6, atol=2e-7)
            sequential_state = sequential_optimizer.state[sequential_param]
            foreach_state = foreach_optimizer.state[foreach_param]
            self.assertEqual(foreach_state["step"], sequential_state["step"])
            torch.testing.assert_close(
                foreach_state["exp_avg"], sequential_state["exp_avg"], rtol=2e-6, atol=2e-7
            )
            torch.testing.assert_close(
                foreach_state["exp_avg_sq"], sequential_state["exp_avg_sq"], rtol=2e-6, atol=2e-7
            )

    def test_checkpoint_continues_across_sequential_and_default_update_paths(self):
        optimized_env_names = (
            "KATAGO_MUON_BATCHED_NS",
            "KATAGO_MUON_NS_BATCH_SIZE",
            "KATAGO_AUX_ADAM_FOREACH",
        )
        clean_env = dict(os.environ)
        for name in optimized_env_names:
            clean_env.pop(name, None)

        def make_optimizer(use_defaults):
            muon_params = [
                torch.nn.Parameter(torch.linspace(-0.4, 0.6, 54).reshape(9, 6)),
                torch.nn.Parameter(torch.linspace(0.7, -0.3, 54).reshape(9, 6)),
            ]
            aux_params = [
                torch.nn.Parameter(torch.linspace(-0.2, 0.5, 35).reshape(5, 7)),
                torch.nn.Parameter(torch.linspace(0.1, 0.8, 11)),
            ]
            environment = {} if use_defaults else {
                "KATAGO_MUON_BATCHED_NS": "0",
                "KATAGO_AUX_ADAM_FOREACH": "0",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                with mock.patch.object(muon_kissin.dist, "is_initialized", return_value=False):
                    optimizer = muon_kissin.MuonWithAuxAdamKimi([
                        {
                            "params": muon_params,
                            "group_name": "normal",
                            "use_muon": True,
                            "momentum": 0.87,
                            "weight_decay": 0.03,
                        },
                        {
                            "params": aux_params,
                            "group_name": "output",
                            "use_muon": False,
                            "betas": (0.9, 0.997),
                            "eps": 2e-7,
                            "weight_decay": 0.04,
                            "muon_lr_multiplier": 5.0,
                        },
                    ])
            optimizer.param_groups[0]["lr"] = 0.02
            optimizer.param_groups[1]["lr"] = 0.035
            return optimizer, muon_params, aux_params

        def set_gradients(muon_params, aux_params, step):
            for index, param in enumerate(muon_params + aux_params):
                values = torch.arange(param.numel(), dtype=param.dtype).reshape(param.shape)
                param.grad = values.mul(0.0003).add(-0.04 + 0.01 * index + 0.005 * step)

        sequential_ns = getattr(
            muon_kissin.zeropower_via_newtonschulz5,
            "_torchdynamo_orig_callable",
            muon_kissin.zeropower_via_newtonschulz5,
        )
        batched_ns = getattr(
            muon_kissin.zeropower_via_newtonschulz5_batched,
            "_torchdynamo_orig_callable",
            muon_kissin.zeropower_via_newtonschulz5_batched,
        )
        with mock.patch.dict(os.environ, clean_env, clear=True):
            with mock.patch.object(
                muon_kissin, "zeropower_via_newtonschulz5", sequential_ns
            ), mock.patch.object(
                muon_kissin, "zeropower_via_newtonschulz5_batched", batched_ns
            ):
                for source_defaults, destination_defaults in ((False, True), (True, False)):
                    with self.subTest(
                        source_defaults=source_defaults,
                        destination_defaults=destination_defaults,
                    ):
                        source, source_muon, source_aux = make_optimizer(source_defaults)
                        for step in range(2):
                            set_gradients(source_muon, source_aux, step)
                            source.step()

                        buffer = io.BytesIO()
                        torch.save(source.state_dict_for_checkpoint(), buffer)
                        buffer.seek(0)
                        checkpoint = torch.load(buffer, weights_only=False)

                        destination, destination_muon, destination_aux = make_optimizer(
                            destination_defaults
                        )
                        with torch.no_grad():
                            for destination_param, source_param in zip(
                                destination_muon + destination_aux,
                                source_muon + source_aux,
                            ):
                                destination_param.copy_(source_param)
                        destination.load_state_dict_for_checkpoint(checkpoint)

                        self.assertEqual(destination.use_batched_muon_ns, destination_defaults)
                        self.assertEqual(destination.use_foreach_aux_adam, destination_defaults)
                        for source_param, destination_param in zip(
                            source_muon + source_aux,
                            destination_muon + destination_aux,
                        ):
                            source_state = source.state[source_param]
                            destination_state = destination.state[destination_param]
                            self.assertEqual(set(destination_state), set(source_state))
                            for key in source_state:
                                if isinstance(source_state[key], torch.Tensor):
                                    self.assertTrue(torch.equal(destination_state[key], source_state[key]))
                                else:
                                    self.assertEqual(destination_state[key], source_state[key])

                        set_gradients(source_muon, source_aux, step=2)
                        set_gradients(destination_muon, destination_aux, step=2)
                        source.step()
                        destination.step()

                        for source_param, destination_param in zip(
                            source_muon, destination_muon
                        ):
                            torch.testing.assert_close(
                                destination_param,
                                source_param,
                                rtol=5e-3,
                                atol=5e-4,
                            )
                            self.assertTrue(torch.equal(
                                destination.state[destination_param]["momentum_buffer"],
                                source.state[source_param]["momentum_buffer"],
                            ))
                        for source_param, destination_param in zip(
                            source_aux, destination_aux
                        ):
                            torch.testing.assert_close(
                                destination_param,
                                source_param,
                                rtol=2e-6,
                                atol=2e-7,
                            )
                            source_state = source.state[source_param]
                            destination_state = destination.state[destination_param]
                            self.assertEqual(destination_state["step"], source_state["step"])
                            torch.testing.assert_close(
                                destination_state["exp_avg"],
                                source_state["exp_avg"],
                                rtol=2e-6,
                                atol=2e-7,
                            )
                            torch.testing.assert_close(
                                destination_state["exp_avg_sq"],
                                source_state["exp_avg_sq"],
                                rtol=2e-6,
                                atol=2e-7,
                            )


if __name__ == "__main__":
    unittest.main()
