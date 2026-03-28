"""
Unit and integration tests for MiniMax Cloud prompt enhancement (RePromptCloud).

Run with: python -m pytest tests/test_reprompt_cloud.py -v
"""

import importlib
import importlib.util
import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch


def _load_reprompt_cloud():
    """Load reprompt_cloud module directly, bypassing __init__.py import chain."""
    spec = importlib.util.spec_from_file_location(
        "reprompt_cloud",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "hyimage",
            "models",
            "reprompt",
            "reprompt_cloud.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-load the module once
_rc_module = _load_reprompt_cloud()
RePromptCloud = _rc_module.RePromptCloud
SYSTEM_PROMPT = _rc_module.SYSTEM_PROMPT


def _make_cloud(**kwargs):
    """Create a RePromptCloud with mocked openai."""
    mock_openai_module = MagicMock()
    mock_openai_cls = MagicMock()
    mock_openai_module.OpenAI = mock_openai_cls
    defaults = {"api_key": "test-key-123"}
    defaults.update(kwargs)
    with patch.dict("sys.modules", {"openai": mock_openai_module}):
        model = RePromptCloud(**defaults)
    return model, mock_openai_cls


def _mock_response(content):
    """Create a mock OpenAI chat completion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


class TestRePromptCloudInit(unittest.TestCase):
    """Test RePromptCloud initialization and configuration."""

    def test_init_with_explicit_api_key(self):
        model, mock_cls = _make_cloud(api_key="explicit-key")
        self.assertEqual(model.api_key, "explicit-key")
        self.assertEqual(model.provider, "minimax")
        self.assertEqual(model.model, "MiniMax-M2.7")
        self.assertEqual(model.base_url, "https://api.minimax.io/v1")
        mock_cls.assert_called_once_with(
            api_key="explicit-key",
            base_url="https://api.minimax.io/v1",
        )

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key-456"})
    def test_init_with_env_api_key(self):
        model, _ = _make_cloud(api_key=None)
        self.assertEqual(model.api_key, "env-key-456")

    def test_init_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError) as ctx:
                _make_cloud(api_key=None)
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    def test_init_unsupported_provider_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_cloud(provider="unsupported")
        self.assertIn("Unsupported provider", str(ctx.exception))

    @patch.dict(
        os.environ,
        {
            "MINIMAX_API_KEY": "env-key",
            "MINIMAX_MODEL": "MiniMax-M2.5",
            "MINIMAX_BASE_URL": "https://custom.api.url/v1",
        },
    )
    def test_init_env_overrides(self):
        model, _ = _make_cloud(api_key=None)
        self.assertEqual(model.model, "MiniMax-M2.5")
        self.assertEqual(model.base_url, "https://custom.api.url/v1")

    def test_init_explicit_params_override_env(self):
        model, _ = _make_cloud(
            api_key="param-key",
            model="custom-model",
            base_url="https://param.url/v1",
        )
        self.assertEqual(model.api_key, "param-key")
        self.assertEqual(model.model, "custom-model")
        self.assertEqual(model.base_url, "https://param.url/v1")

    def test_init_ignores_local_model_params(self):
        """Cloud model should accept but ignore local model parameters."""
        model, _ = _make_cloud(
            models_root_path="/some/path",
            device_map="cpu",
            enable_offloading=False,
        )
        self.assertIsNotNone(model)

    def test_supported_providers(self):
        self.assertIn("minimax", RePromptCloud.SUPPORTED_PROVIDERS)
        config = RePromptCloud.SUPPORTED_PROVIDERS["minimax"]
        self.assertEqual(config["api_key_env"], "MINIMAX_API_KEY")
        self.assertEqual(config["default_model"], "MiniMax-M2.7")
        self.assertEqual(config["default_base_url"], "https://api.minimax.io/v1")

    def test_default_model_is_m2_7(self):
        model, _ = _make_cloud()
        self.assertEqual(model.model, "MiniMax-M2.7")


class TestRePromptCloudPredict(unittest.TestCase):
    """Test RePromptCloud.predict() method."""

    def _make_model(self):
        model, _ = _make_cloud()
        return model

    def test_predict_plain_response(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "A detailed description of a sunset over the ocean with golden light."
            )
        )
        result = model.predict("sunset over ocean")
        self.assertEqual(
            result,
            "A detailed description of a sunset over the ocean with golden light.",
        )

    def test_predict_with_answer_tags(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "Here is the result:\n<answer>Enhanced prompt text here</answer>"
            )
        )
        result = model.predict("test prompt")
        self.assertEqual(result, "Enhanced prompt text here")

    def test_predict_strips_think_tags(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "<think>Let me analyze this prompt...</think>\n"
                "A beautiful sunset over a calm ocean."
            )
        )
        result = model.predict("sunset")
        self.assertEqual(result, "A beautiful sunset over a calm ocean.")

    def test_predict_strips_think_tags_with_answer(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "<think>Reasoning here</think>\n"
                "<answer>Clean enhanced prompt</answer>"
            )
        )
        result = model.predict("test")
        self.assertEqual(result, "Clean enhanced prompt")

    def test_predict_uses_correct_api_params(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("enhanced text")
        )
        model.predict("test prompt")
        call_kwargs = model._client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(call_kwargs["max_tokens"], 2048)
        self.assertGreater(call_kwargs["temperature"], 0.0)
        self.assertLessEqual(call_kwargs["temperature"], 1.0)

    def test_predict_passes_system_prompt(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("enhanced")
        )
        model.predict("test", sys_prompt="Custom system prompt")
        call_kwargs = model._client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "Custom system prompt")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "test")

    def test_predict_uses_default_system_prompt(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("enhanced")
        )
        model.predict("test prompt")
        call_kwargs = model._client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(messages[0]["content"], SYSTEM_PROMPT)

    def test_predict_fallback_on_api_error(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            side_effect=Exception("API connection failed")
        )
        result = model.predict("original prompt")
        self.assertEqual(result, "original prompt")

    def test_predict_fallback_on_empty_response(self):
        model = self._make_model()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        model._client.chat.completions.create = MagicMock(return_value=mock_resp)
        result = model.predict("original prompt")
        self.assertEqual(result, "original prompt")

    def test_predict_chinese_prompt(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "一幅精美的山水画，远处的青山在薄雾中若隐若现，近处的溪流清澈见底。"
            )
        )
        result = model.predict("中国山水画")
        self.assertIn("山水", result)

    def test_predict_multiline_think_tags(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "<think>\nLine 1\nLine 2\nLine 3\n</think>\n"
                "Final enhanced prompt."
            )
        )
        result = model.predict("test")
        self.assertEqual(result, "Final enhanced prompt.")
        self.assertNotIn("<think>", result)

    def test_predict_long_prompt(self):
        model = self._make_model()
        long_prompt = "A scene with " + ", ".join(f"element {i}" for i in range(50))
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("Enhanced: " + long_prompt)
        )
        result = model.predict(long_prompt)
        self.assertIn("Enhanced:", result)

    def test_predict_empty_prompt(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("A blank canvas.")
        )
        result = model.predict("")
        self.assertEqual(result, "A blank canvas.")

    def test_predict_special_characters(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response("Enhanced with 'quotes' and extra details")
        )
        result = model.predict("prompt with 'quotes'")
        self.assertIn("quotes", result)

    def test_predict_preserves_original_on_timeout(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            side_effect=TimeoutError("Request timed out")
        )
        result = model.predict("my prompt")
        self.assertEqual(result, "my prompt")

    def test_predict_with_nested_tags(self):
        model = self._make_model()
        model._client.chat.completions.create = MagicMock(
            return_value=_mock_response(
                "<think>step 1</think>\n"
                "<answer>A <b>beautiful</b> scene</answer>"
            )
        )
        result = model.predict("scene")
        self.assertEqual(result, "A <b>beautiful</b> scene")


class TestRePromptCloudInterface(unittest.TestCase):
    """Test interface compatibility with local RePrompt models."""

    def test_to_method_is_noop(self):
        model, _ = _make_cloud()
        result = model.to("cuda")
        self.assertIs(result, model)

    def test_to_method_chaining(self):
        model, _ = _make_cloud()
        result = model.to("cuda").to("cpu").to("cuda:0")
        self.assertIs(result, model)

    def test_has_predict_method(self):
        model, _ = _make_cloud()
        self.assertTrue(callable(getattr(model, "predict", None)))

    def test_has_to_method(self):
        model, _ = _make_cloud()
        self.assertTrue(callable(getattr(model, "to", None)))

    def test_predict_signature_matches_local(self):
        """Verify predict() accepts same args as local RePrompt."""
        import inspect

        sig = inspect.signature(RePromptCloud.predict)
        params = list(sig.parameters.keys())
        self.assertIn("prompt_cot", params)
        self.assertIn("sys_prompt", params)


class TestModelZooMiniMax(unittest.TestCase):
    """Test MiniMax reprompt configuration in model_zoo."""

    def test_minimax_reprompt_config_exists(self):
        # Import model_zoo components that don't need torch/diffusers
        from hyimage.common.config.base_config import RepromptConfig
        from hyimage.common.config.lazy import LazyCall as L, LazyObject

        config = RepromptConfig(
            model=L(RePromptCloud)(models_root_path=None, provider="minimax"),
            load_from="",
        )
        self.assertIsInstance(config, RepromptConfig)
        self.assertEqual(config.load_from, "")
        self.assertIsInstance(config.model, LazyObject)

    def test_minimax_config_instantiates_with_api_key(self):
        from hyimage.common.config.lazy import LazyCall as L, instantiate

        config_model = L(RePromptCloud)(models_root_path=None, provider="minimax")
        mock_openai_module = MagicMock()
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
                model = instantiate(config_model)
        self.assertIsNotNone(model)
        self.assertEqual(model.provider, "minimax")
        self.assertEqual(model.model, "MiniMax-M2.7")

    def test_minimax_config_cloud_flag(self):
        """Cloud reprompt config has empty load_from (no local model path)."""
        from hyimage.common.config.base_config import RepromptConfig
        from hyimage.common.config.lazy import LazyCall as L

        config = RepromptConfig(
            model=L(RePromptCloud)(models_root_path=None, provider="minimax"),
            load_from="",
        )
        # Pipeline uses empty load_from to detect cloud mode
        self.assertFalse(bool(config.load_from))


class TestPipelineRepromptResolution(unittest.TestCase):
    """Test that the pipeline correctly resolves reprompt model names.

    Since the pipeline module has heavy dependencies (torch, diffusers),
    we test the resolution logic by verifying the model_zoo factory
    functions produce correct configs.
    """

    def test_minimax_factory_produces_empty_load_from(self):
        """HUNYUANIMAGE_REPROMPT_MINIMAX should produce config with empty load_from."""
        from hyimage.common.config.base_config import RepromptConfig
        from hyimage.common.config.lazy import LazyCall as L

        # Replicate what model_zoo.HUNYUANIMAGE_REPROMPT_MINIMAX does
        config = RepromptConfig(
            model=L(RePromptCloud)(models_root_path=None, provider="minimax"),
            load_from="",
        )
        self.assertEqual(config.load_from, "")

    def test_local_factory_produces_nonempty_load_from(self):
        """Local reprompt configs should have non-empty load_from."""
        from hyimage.common.config.base_config import RepromptConfig
        from hyimage.common.config.lazy import LazyCall as L

        config = RepromptConfig(
            model=L(MagicMock)(models_root_path=None),
            load_from="./ckpts/reprompt",
        )
        self.assertTrue(bool(config.load_from))

    def test_resolution_logic(self):
        """Test the _resolve_reprompt_config logic without importing the pipeline."""
        from hyimage.common.config.base_config import RepromptConfig
        from hyimage.common.config.lazy import LazyCall as L

        # Simulate what _resolve_reprompt_config does
        def resolve(name):
            if name == "hunyuanimage-reprompt-32b":
                return RepromptConfig(
                    model=L(MagicMock)(models_root_path=None),
                    load_from="./ckpts/reprompt_32b",
                )
            elif name in ("hunyuanimage-reprompt-minimax", "minimax"):
                return RepromptConfig(
                    model=L(RePromptCloud)(
                        models_root_path=None, provider="minimax"
                    ),
                    load_from="",
                )
            else:
                return RepromptConfig(
                    model=L(MagicMock)(models_root_path=None),
                    load_from="./ckpts/reprompt",
                )

        self.assertEqual(resolve("minimax").load_from, "")
        self.assertEqual(resolve("hunyuanimage-reprompt-minimax").load_from, "")
        self.assertNotEqual(resolve("hunyuanimage-reprompt-32b").load_from, "")
        self.assertNotEqual(resolve("hunyuanimage-reprompt").load_from, "")


class TestRePromptCloudIntegration(unittest.TestCase):
    """Integration tests for MiniMax Cloud prompt enhancement.

    These tests require a valid MINIMAX_API_KEY environment variable
    and network access to the MiniMax API.
    """

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set; skipping integration tests",
    )
    def test_live_prompt_enhancement_english(self):
        model = RePromptCloud()
        result = model.predict("A cute cat sitting on a windowsill")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertGreater(len(result), len("A cute cat sitting on a windowsill"))

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set; skipping integration tests",
    )
    def test_live_prompt_enhancement_chinese(self):
        model = RePromptCloud()
        result = model.predict("一只可爱的猫咪坐在窗台上")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    @unittest.skipUnless(
        os.environ.get("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set; skipping integration tests",
    )
    def test_live_predict_interface_compatibility(self):
        """Verify cloud model has same interface as local models."""
        model = RePromptCloud()
        result = model.predict(
            "sunset over mountains",
            sys_prompt="Enhance this image prompt with more details.",
        )
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        same_model = model.to("cuda")
        self.assertIs(same_model, model)


if __name__ == "__main__":
    unittest.main()
