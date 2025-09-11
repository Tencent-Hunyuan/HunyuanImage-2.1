import re
import loguru
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import cpu_offload_with_hook

"""
English translation of the System prompt:
----------------------------------------
You are an expert in writing image generation prompts. Please rewrite the user's prompt according to the following requirements:
1. The main subject/action/quantity/style/layout/relationship/attribute/text in the rewritten prompt must be consistent with the original intention;
2. The rewritten prompt should follow the "overall-detail-conclusion" structure, ensuring the clarity of information hierarchy;
3. The rewritten prompt should be objective and neutral, avoiding subjective judgment and emotional evaluation;
4. The rewritten prompt should be from the main to the secondary, always describing the most important elements first, and then the secondary and background elements;
5. The rewritten prompt should be logically clear, strictly follow the spatial logic or main-secondary logic, allowing the reader to reconstruct the image in the brain;
6. The rewritten prompt should end with a summary sentence, summarizing the overall style or type of the image.
"""

SYSTEM_PROMPT = (
    "你是一位图像生成提示词撰写专家，请根据用户输入的提示词，改写生成新的提示词，改写后的提示词要求："
    "1 改写后提示词包含的主体/动作/数量/风格/布局/关系/属性/文字等 必须和改写前的意图一致； "
    "2 在宏观上遵循“总-分-总”的结构，确保信息的层次清晰；"
    "3 客观中立，避免主观臆断和情感评价；"
    "4 由主到次，始终先描述最重要的元素，再描述次要和背景元素；"
    "5 逻辑清晰，严格遵循空间逻辑或主次逻辑，使读者能在大脑中重建画面；"
    "6 结尾点题，必须用一句话总结图像的整体风格或类型。"
)


def replace_single_quotes(text):
    """
    Replace single quotes within words with double quotes, and convert
    curly single quotes to curly double quotes for consistency.
    """
    pattern = r"\B'([^']*)'\B"
    replaced_text = re.sub(pattern, r'"\1"', text)
    replaced_text = replaced_text.replace("’", "”")
    replaced_text = replaced_text.replace("‘", "“")
    return replaced_text


class RePrompt:

    def __init__(self, models_root_path, device_map="auto", enable_offloading=True):
        """
        Initialize the RePrompt class with model and processor.

        Args:
            models_root_path (str): Path to the pretrained model.
            device_map (str): Device mapping for model loading.
        """
        if enable_offloading:
            device_map = None
        self.model = AutoModelForCausalLM.from_pretrained(models_root_path, device_map=device_map, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(models_root_path, trust_remote_code=True)
        self.enable_offloading = enable_offloading

        if enable_offloading:
            _, self.offload_hook = cpu_offload_with_hook(self.model, execution_device=torch.device('cuda'))
        self.device_map = device_map
        self.original_device_map = getattr(self.model, 'hf_device_map', None)

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt=SYSTEM_PROMPT,
    ):
        """
        Generate a rewritten prompt using the model.

        Args:
            prompt_cot (str): The original prompt to be rewritten.
            sys_prompt (str): System prompt to guide the rewriting.
            temperature (float): Sampling temperature.
            device (str): Device for inference.

        Returns:
            str: The rewritten prompt, or the original if generation fails.
        """
        org_prompt_cot = prompt_cot
        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": org_prompt_cot},
            ]
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", enable_thinking=False  # Toggle thinking mode (default: True)
            )
            if self.model.device != torch.device('meta'):
                tokenized_chat = tokenized_chat.to(self.model.device)
            outputs = self.model.generate(tokenized_chat, max_new_tokens=2048)
            if self.enable_offloading:
                self.offload_hook.offload()
            output_res = self.tokenizer.decode(outputs[0])
            answer_pattern = r'<answer>(.*?)</answer>'
            answer_matches = re.findall(answer_pattern, output_res, re.DOTALL)
            prompt_cot = [match.strip() for match in answer_matches][0]
            prompt_cot = replace_single_quotes(prompt_cot)
        except Exception as e:
            prompt_cot = org_prompt_cot
            loguru.logger.error(f"✗ Re-prompting failed, fall back to generate prompt. Cause: {e}")

        return prompt_cot

    def to(self, device, *args, **kwargs):
        self.model = self.model.to(device, *args, **kwargs)
        return self