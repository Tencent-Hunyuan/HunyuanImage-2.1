import os
import re
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求："
    "1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； "
    "2 在宏观上遵循“总-分-总”的结构，确保信息的层次清晰；"
    "3 客观中立，避免主观臆断和情感评价；"
    "4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；"
    "5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；"
    "6 结尾点题，必须用一句话总结图像的整体风格或类型。"
)


class RePromptCloud:
    """
    Cloud-based prompt enhancement using MiniMax API (OpenAI-compatible).

    This class provides the same interface as the local RePrompt model but
    uses MiniMax's cloud LLM API for prompt enhancement, eliminating the
    need for local GPU resources for the reprompt model.

    Environment variables:
        MINIMAX_API_KEY: Required. Your MiniMax API key.
        MINIMAX_MODEL: Optional. Model name (default: MiniMax-M2.7).
        MINIMAX_BASE_URL: Optional. API base URL
            (default: https://api.minimax.io/v1).
    """

    SUPPORTED_PROVIDERS = {
        "minimax": {
            "default_base_url": "https://api.minimax.io/v1",
            "default_model": "MiniMax-M2.7",
            "api_key_env": "MINIMAX_API_KEY",
        },
    }

    def __init__(
        self,
        models_root_path=None,
        device_map="auto",
        enable_offloading=True,
        provider="minimax",
        api_key=None,
        base_url=None,
        model=None,
    ):
        """
        Initialize the cloud-based reprompt model.

        Args:
            models_root_path: Ignored (kept for interface compatibility).
            device_map: Ignored (kept for interface compatibility).
            enable_offloading: Ignored (kept for interface compatibility).
            provider: Cloud LLM provider name (default: "minimax").
            api_key: API key. If None, reads from environment variable.
            base_url: API base URL. If None, uses provider default.
            model: Model name. If None, uses provider default.
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )

        provider_config = self.SUPPORTED_PROVIDERS[provider]
        self.provider = provider

        self.api_key = api_key or os.environ.get(provider_config["api_key_env"])
        if not self.api_key:
            raise ValueError(
                f"API key required. Set {provider_config['api_key_env']} "
                f"environment variable or pass api_key parameter."
            )

        self.base_url = (
            base_url
            or os.environ.get("MINIMAX_BASE_URL")
            or provider_config["default_base_url"]
        )
        self.model = (
            model
            or os.environ.get("MINIMAX_MODEL")
            or provider_config["default_model"]
        )

        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for cloud-based prompt "
                "enhancement. Install it with: pip install openai"
            )

        logger.info(
            f"✓ Cloud reprompt model initialized "
            f"(provider={self.provider}, model={self.model})"
        )

    def predict(
        self,
        prompt_cot,
        sys_prompt=SYSTEM_PROMPT,
    ):
        """
        Generate a rewritten prompt using the cloud LLM API.

        Args:
            prompt_cot: The original prompt to be rewritten.
            sys_prompt: System prompt to guide the rewriting.

        Returns:
            str: The rewritten prompt, or the original if generation fails.
        """
        org_prompt_cot = prompt_cot
        try:
            # MiniMax temperature must be in (0.0, 1.0]
            temperature = 0.1

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": org_prompt_cot},
                ],
                max_tokens=2048,
                temperature=temperature,
            )

            output_res = response.choices[0].message.content

            # Strip thinking tags if present (MiniMax M2.7/M2.5 may include them)
            output_res = re.sub(
                r"<think>.*?</think>\s*", "", output_res, flags=re.DOTALL
            )

            # Try to extract from <answer> tags (matches local model format)
            answer_pattern = r"<answer>(.*?)</answer>"
            answer_matches = re.findall(answer_pattern, output_res, re.DOTALL)
            if answer_matches:
                prompt_cot = answer_matches[0].strip()
            else:
                # Use the full response if no <answer> tags
                prompt_cot = output_res.strip()

        except Exception as e:
            prompt_cot = org_prompt_cot
            logger.error(
                f"✗ Cloud re-prompting failed, fall back to original prompt. "
                f"Cause: {e}"
            )

        return prompt_cot

    def to(self, device, *args, **kwargs):
        """No-op for cloud models (kept for interface compatibility)."""
        return self
