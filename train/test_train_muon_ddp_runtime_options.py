import os
import sys
import unittest
from unittest import mock


sys.path.insert(0, os.path.dirname(__file__))
import train_muon_ki


_DDP_ENV_NAMES = (
    "KATAGO_DDP_STATIC_GRAPH",
    "KATAGO_DDP_GRADIENT_AS_BUCKET_VIEW",
    "KATAGO_DDP_BROADCAST_BUFFERS",
    "KATAGO_DDP_ALIGN_CONV1X1_WEIGHT_STRIDES",
)
_ENV_NAMES = _DDP_ENV_NAMES


class DdpRuntimeOptionsTests(unittest.TestCase):
    @staticmethod
    def _raw_model(norm_kind="bnorm"):
        model = object.__new__(type("FakeModel", (), {}))
        model.get_norm_kind = lambda: norm_kind
        model.modules = lambda: []
        return model

    def _clean_env(self):
        env = dict(os.environ)
        for name in _ENV_NAMES:
            env.pop(name, None)
        return mock.patch.dict(os.environ, env, clear=True)

    def _mock_wrappers(self):
        events = []

        def compile_model(model, **kwargs):
            events.append(("compile", model, kwargs))
            return ("compiled", model)

        def make_ddp(model, **kwargs):
            events.append(("ddp", model, kwargs))
            return ("ddp", model)

        return (
            events,
            mock.patch.object(train_muon_ki.torch, "compile", side_effect=compile_model),
            mock.patch.object(
                train_muon_ki,
                "DistributedDataParallel",
                side_effect=make_ddp,
            ),
        )

    def test_default_compiles_raw_model_before_optimized_ddp(self):
        raw_model = self._raw_model()
        device = object()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), compile_patch, ddp_patch:
            result = train_muon_ki.wrap_model_for_training(
                raw_model, device, world_size=2, no_compile=False
            )

        self.assertEqual([event[0] for event in events], ["compile", "ddp"])
        self.assertIs(events[0][1], raw_model)
        self.assertEqual(events[0][2], {"mode": "default"})
        self.assertEqual(events[1][1], ("compiled", raw_model))
        self.assertEqual(
            events[1][2],
            {
                "device_ids": [device],
                "broadcast_buffers": False,
                "static_graph": True,
                "gradient_as_bucket_view": True,
            },
        )
        self.assertEqual(result, ("ddp", ("compiled", raw_model)))

    def test_explicit_zero_disables_optimized_ddp_kwargs(self):
        raw_model = self._raw_model()
        device = object()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        env = {name: "0" for name in _DDP_ENV_NAMES}
        with self._clean_env(), mock.patch.dict(os.environ, env), compile_patch, ddp_patch:
            result = train_muon_ki.wrap_model_for_training(
                raw_model, device, world_size=2, no_compile=True
            )

        self.assertEqual(
            events,
            [("ddp", raw_model, {"device_ids": [device], "broadcast_buffers": False})],
        )
        self.assertEqual(result, ("ddp", raw_model))

    def test_single_gpu_ignores_ddp_environment_flags(self):
        raw_model = self._raw_model()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        invalid_env = {name: "not-a-bool" for name in _DDP_ENV_NAMES}
        with self._clean_env(), mock.patch.dict(
            os.environ, invalid_env
        ), compile_patch, ddp_patch:
            result = train_muon_ki.wrap_model_for_training(
                raw_model, object(), world_size=1, no_compile=False
            )

        self.assertEqual(events, [("compile", raw_model, {"mode": "default"})])
        self.assertEqual(result, ("compiled", raw_model))

    def test_single_gpu_no_compile_returns_raw_model(self):
        raw_model = self._raw_model()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), compile_patch, ddp_patch:
            result = train_muon_ki.wrap_model_for_training(
                raw_model, object(), world_size=1, no_compile=True
            )

        self.assertIs(result, raw_model)
        self.assertEqual(events, [])

    def test_distributed_environment_flags_are_strict(self):
        for name in _DDP_ENV_NAMES:
            with self.subTest(name=name):
                with self._clean_env(), mock.patch.dict(os.environ, {name: "true"}):
                    with self.assertRaisesRegex(ValueError, name):
                        train_muon_ki.wrap_model_for_training(
                            self._raw_model(), object(), world_size=2, no_compile=True
                        )

    def test_valid_compile_modes_are_used(self):
        for mode in ("max-autotune-no-cudagraphs", "max-autotune"):
            for world_size in (1, 2):
                with self.subTest(mode=mode, world_size=world_size):
                    raw_model = self._raw_model()
                    events, compile_patch, ddp_patch = self._mock_wrappers()
                    with self._clean_env(), compile_patch, ddp_patch:
                        train_muon_ki.wrap_model_for_training(
                            raw_model,
                            object(),
                            world_size=world_size,
                            no_compile=False,
                            compile_mode=mode,
                        )

                    compile_events = [event for event in events if event[0] == "compile"]
                    self.assertEqual(len(compile_events), 1)
                    self.assertEqual(compile_events[0][2], {"mode": mode})

    def test_invalid_compile_mode_is_rejected(self):
        with self._clean_env():
            with self.assertRaisesRegex(ValueError, "-compile-mode"):
                train_muon_ki.wrap_model_for_training(
                    self._raw_model(),
                    object(),
                    world_size=1,
                    no_compile=False,
                    compile_mode="reduce-overhead",
                )

    def test_no_compile_ignores_compile_mode(self):
        raw_model = self._raw_model()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), compile_patch, ddp_patch:
            result = train_muon_ki.wrap_model_for_training(
                raw_model,
                object(),
                world_size=1,
                no_compile=True,
                compile_mode="not-used",
            )

        self.assertIs(result, raw_model)
        self.assertEqual(events, [])

    def test_sdpa_backend_can_be_forced_for_benchmarks(self):
        backend_calls = {}
        patches = []
        for name in ("flash", "cudnn", "mem_efficient", "math"):
            patcher = mock.patch.object(
                train_muon_ki.torch.backends.cuda,
                f"enable_{name}_sdp",
                side_effect=lambda enabled, name=name: backend_calls.__setitem__(
                    name, enabled
                ),
            )
            patches.append(patcher)

        with self._clean_env():
            for patcher in patches:
                patcher.start()
            try:
                selected = train_muon_ki.configure_sdpa_backend("cudnn")
            finally:
                for patcher in reversed(patches):
                    patcher.stop()

        self.assertEqual(selected, "cudnn")
        self.assertEqual(
            backend_calls,
            {"flash": False, "cudnn": True, "mem_efficient": False, "math": False},
        )

    def test_invalid_sdpa_backend_is_rejected(self):
        with self._clean_env():
            with self.assertRaisesRegex(ValueError, "-sdpa-backend"):
                train_muon_ki.configure_sdpa_backend("unknown")

    def test_input_memory_format_uses_nhwc_by_default_and_allows_nchw(self):
        self.assertEqual(train_muon_ki._ALLOWED_INPUT_MEMORY_FORMATS[0], "nhwc")
        self.assertTrue(train_muon_ki.resolve_input_nhwc("nhwc"))
        self.assertFalse(train_muon_ki.resolve_input_nhwc("nchw"))

    def test_invalid_input_memory_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "-input-memory-format"):
            train_muon_ki.resolve_input_nhwc("auto")

    def test_on_load_full_board_filter_requires_maskless_training(self):
        train_muon_ki.validate_full_board_filter_options(
            disable_mask=True,
            filter_full_board_on_load=True,
        )
        with self.assertRaisesRegex(ValueError, "requires -disable-mask"):
            train_muon_ki.validate_full_board_filter_options(
                disable_mask=False,
                filter_full_board_on_load=True,
            )

    def test_flex_attention_defaults_on_for_compatible_masked_training(self):
        self.assertTrue(
            train_muon_ki.resolve_flex_attention_enabled(
                requested=None,
                disable_mask=False,
                no_compile=False,
                qat_int8=False,
            )
        )
        self.assertFalse(
            train_muon_ki.resolve_flex_attention_enabled(
                requested=False,
                disable_mask=False,
                no_compile=False,
                qat_int8=False,
            )
        )

    def test_flex_attention_auto_falls_back_for_incompatible_modes(self):
        cases = (
            {"disable_mask": True},
            {"no_compile": True},
            {"qat_int8": True},
        )
        defaults = dict(
            requested=None,
            disable_mask=False,
            no_compile=False,
            qat_int8=False,
        )
        for updates in cases:
            with self.subTest(updates=updates):
                options = dict(defaults)
                options.update(updates)
                self.assertFalse(
                    train_muon_ki.resolve_flex_attention_enabled(**options)
                )

    def test_explicit_flex_attention_is_validated(self):
        invalid_cases = (
            ({"disable_mask": True}, "disable-mask"),
            ({"no_compile": True}, "torch.compile"),
            ({"qat_int8": True}, "qat-int8"),
        )
        defaults = dict(
            requested=True,
            disable_mask=False,
            no_compile=False,
            qat_int8=False,
        )
        for updates, pattern in invalid_cases:
            with self.subTest(updates=updates):
                options = dict(defaults)
                options.update(updates)
                with self.assertRaisesRegex(ValueError, pattern):
                    train_muon_ki.resolve_flex_attention_enabled(**options)

    def test_flex_attention_auto_skips_unsupported_cnn_models(self):
        class FakeModel:
            def __init__(self, supported):
                self.supported = supported
                self.enabled = None

            def supports_flex_attention(self):
                return self.supported

            def configure_flex_attention(self, enabled):
                self.enabled = enabled

        cnn_model = FakeModel(supported=False)
        actual = train_muon_ki.configure_model_flex_attention(
            cnn_model,
            requested=None,
            enabled=True,
        )
        self.assertFalse(actual)
        self.assertFalse(cnn_model.enabled)

        transformer_model = FakeModel(supported=True)
        actual = train_muon_ki.configure_model_flex_attention(
            transformer_model,
            requested=None,
            enabled=True,
        )
        self.assertTrue(actual)
        self.assertTrue(transformer_model.enabled)

        with self.assertRaisesRegex(ValueError, "no supported transformer"):
            train_muon_ki.configure_model_flex_attention(
                cnn_model,
                requested=True,
                enabled=True,
            )

    def test_batch_renorm_keeps_buffer_broadcast_by_default(self):
        raw_model = self._raw_model("brenorm")
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), compile_patch, ddp_patch:
            train_muon_ki.wrap_model_for_training(
                raw_model, object(), world_size=2, no_compile=True
            )

        self.assertTrue(events[0][2]["broadcast_buffers"])

    def test_qat_keeps_buffer_broadcast_by_default(self):
        raw_model = self._raw_model("bnorm")
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), compile_patch, ddp_patch:
            train_muon_ki.wrap_model_for_training(
                raw_model,
                object(),
                world_size=2,
                no_compile=True,
                qat_int8=True,
            )

        self.assertTrue(events[0][2]["broadcast_buffers"])

    def test_buffer_broadcast_can_be_explicitly_restored(self):
        raw_model = self._raw_model()
        events, compile_patch, ddp_patch = self._mock_wrappers()
        with self._clean_env(), mock.patch.dict(
            os.environ, {"KATAGO_DDP_BROADCAST_BUFFERS": "1"}
        ), compile_patch, ddp_patch:
            train_muon_ki.wrap_model_for_training(
                raw_model, object(), world_size=2, no_compile=True
            )

        self.assertTrue(events[0][2]["broadcast_buffers"])

    def test_conv1x1_stride_alignment_preserves_weights(self):
        model = train_muon_ki.torch.nn.Sequential(
            train_muon_ki.torch.nn.Conv2d(7, 11, kernel_size=1, bias=False),
            train_muon_ki.torch.nn.Conv2d(7, 11, kernel_size=3, bias=False),
        )
        original = model[0].weight.detach().clone()

        count = train_muon_ki.align_conv1x1_weight_strides_for_ddp(model)

        self.assertEqual(count, 1)
        self.assertEqual(model[0].weight.stride(), (7, 1, 7, 7))
        self.assertTrue(model[0].weight.is_contiguous())
        self.assertTrue(train_muon_ki.torch.equal(model[0].weight, original))
        self.assertEqual(model[1].weight.stride(), (63, 9, 3, 1))
        self.assertEqual(
            train_muon_ki.align_conv1x1_weight_strides_for_ddp(model), 0
        )


if __name__ == "__main__":
    unittest.main()
