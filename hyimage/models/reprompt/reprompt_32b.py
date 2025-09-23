import re
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            models_root_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(models_root_path)
        self.enable_offloading = enable_offloading

        if enable_offloading:
            from accelerate import cpu_offload_with_hook
            _, self.offload_hook = cpu_offload_with_hook(self.model, execution_device=torch.device('cuda'))
        self.device_map = device_map
        self.original_device_map = getattr(self.model, 'hf_device_map', None)

    @torch.inference_mode()
    def predict(
        self,
        prompt_cot,
        sys_prompt="请根据用户的输入，生成思考过程的思维链并改写提示词：",
        temperature=0,
        device="cuda",
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
            user_prompt_format = sys_prompt + "\n" + org_prompt_cot
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_format},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=float(temperature),
                do_sample=False,
                top_k=5,
                top_p=0.9
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_res = output_text[0]
            assert output_res.count("think>") == 2
            prompt_cot = output_res.split("think>")[-1]
            if prompt_cot.startswith("\n"):
                prompt_cot = prompt_cot[1:]
            prompt_cot = replace_single_quotes(prompt_cot)
        except Exception:
            prompt_cot = org_prompt_cot
            print("✗ Re-prompting failed, so we are using the original prompt")

        return prompt_cot

    def to(self, device, *args, **kwargs):
        self.model = self.model.to(device, *args, **kwargs)
        return self