import copy
import re
import os
from dataclasses import dataclass
from typing import Optional
from einops import rearrange

from tqdm import tqdm
import loguru
import torch
from hyimage.common.config.lazy import DictConfig
from PIL import Image

from hyimage.common.config import instantiate
from hyimage.common.constants import PRECISION_TO_TYPE
from hyimage.common.format_prompt import MultilingualPromptFormat
from hyimage.models.text_encoder import PROMPT_TEMPLATE
from hyimage.models.model_zoo import HUNYUANIMAGE_REPROMPT
from hyimage.models.text_encoder.byT5 import load_glyph_byT5_v2
from hyimage.models.hunyuan.modules.hunyuanimage_dit import load_hunyuan_dit_state_dict
from hyimage.diffusion.cfg_utils import AdaptiveProjectedGuidance, rescale_noise_cfg


@dataclass
class HunyuanImagePipelineConfig:
    """
    Configuration class for HunyuanImage diffusion pipeline.

    This dataclass consolidates all configuration parameters for the pipeline,
    including model configurations (DiT, VAE, text encoder) and pipeline
    parameters (sampling steps, guidance scale, etc.).
    """

    # Model configurations
    dit_config: DictConfig
    vae_config: DictConfig
    text_encoder_config: DictConfig
    reprompt_config: DictConfig
    refiner_model_name: str = "hunyuanimage-refiner"

    enable_stage1_offloading: bool = True # offload models in stage1 pipeline when reprompt or refiner is working
    enable_reprompt_model_offloading: bool = True # offload reprompt model after finishing
    enable_refiner_offloading: bool = True # offload refiner model after finishing
    enable_text_encoder_offloading: bool = True # offload text encoder after finishing
    enable_full_dit_offloading: bool = True  # offload during text encoding and latent decoding
    enable_vae_offloading: bool = True # offload vae after finishing
    enable_byt5_offloading: bool = True # offload byt5 after finishing

    cfg_mode: str = "MIX_mode_0"
    guidance_rescale: float = 0.0

    # Pipeline parameters
    default_sampling_steps: int = 50
    # Default guidance scale, will be overridden by the guidance_scale parameter in __call__
    default_guidance_scale: float = 3.5
    # Inference shift
    shift: int = 4
    torch_dtype: str = "bf16"
    device: str = "cuda"
    version: str = ""

    @classmethod
    def create_default(cls, version: str = "v2.1", use_distilled: bool = False, **kwargs):
        """
        Create a default configuration for specified HunyuanImage version.

        Args:
            version: HunyuanImage version, only "v2.1" is supported
            use_distilled: Whether to use distilled model
            **kwargs: Additional configuration options
        """
        if version == "v2.1":
            from hyimage.models.model_zoo import (
                HUNYUANIMAGE_V2_1_DIT,
                HUNYUANIMAGE_V2_1_DIT_CFG_DISTILL,
                HUNYUANIMAGE_V2_1_VAE_32x,
                HUNYUANIMAGE_V2_1_TEXT_ENCODER,
            )
            dit_config = HUNYUANIMAGE_V2_1_DIT_CFG_DISTILL() if use_distilled else HUNYUANIMAGE_V2_1_DIT()
            return cls(
                dit_config=dit_config,
                vae_config=HUNYUANIMAGE_V2_1_VAE_32x(),
                text_encoder_config=HUNYUANIMAGE_V2_1_TEXT_ENCODER(),
                reprompt_config=HUNYUANIMAGE_REPROMPT(),
                version=version,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported HunyuanImage version: {version}. Only 'v2.1' is supported")


class HunyuanImagePipeline:
    """
    User-friendly pipeline for HunyuanImage text-to-image generation.

    This pipeline provides a simple interface similar to diffusers library
    for generating high-quality images from text prompts.

    Supports HunyuanImage 2.1 version with automatic configuration.
    Both default and distilled (CFG distillation) models are supported.
    """

    def __init__(
        self,
        config: HunyuanImagePipelineConfig,
        **kwargs
    ):
        """
        Initialize the HunyuanImage diffusion pipeline.

        Args:
            config: Configuration object containing all model and pipeline settings
            **kwargs: Additional configuration options
        """
        self.config = copy.deepcopy(config)
        self.default_sampling_steps = config.default_sampling_steps
        self.default_guidance_scale = config.default_guidance_scale
        self.shift = config.shift
        self.torch_dtype = PRECISION_TO_TYPE[config.torch_dtype]
        self.device = config.device
        self.execution_device = config.device

        self.dit = None
        self.text_encoder = None
        self.vae = None
        self.byt5_kwargs = None
        self.prompt_format = None


        self.cfg_mode = config.cfg_mode
        self.guidance_rescale = config.guidance_rescale

        if self.cfg_mode == "APG_mode_0":
            self.cfg_guider = AdaptiveProjectedGuidance(guidance_scale=10.0, eta=0.0,
                                                        adaptive_projected_guidance_rescale=10.0,
                                                        adaptive_projected_guidance_momentum=-0.5)
            self.apg_start_step = 10
        elif self.cfg_mode == "MIX_mode_0":
            self.cfg_guider_ocr = AdaptiveProjectedGuidance(guidance_scale=10.0, eta=0.0,
                                                            adaptive_projected_guidance_rescale=10.0,
                                                            adaptive_projected_guidance_momentum=-0.5)
            self.apg_start_step_ocr = 75

            self.cfg_guider_general = AdaptiveProjectedGuidance(guidance_scale=10.0, eta=0.0,
                                                                adaptive_projected_guidance_rescale=10.0,
                                                                adaptive_projected_guidance_momentum=-0.5)
            self.apg_start_step_general = 10

        self.ocr_mask = []


        self._load_models()

    def _load_dit(self):
        dit_device = None
        if self.config.enable_full_dit_offloading:
            dit_device = 'cpu'
        else:
            dit_device = self.device
            
        try:
            dit_config = self.config.dit_config
            self.dit = instantiate(dit_config.model, dtype=self.torch_dtype, device=dit_device)
            if dit_config.load_from:
                load_hunyuan_dit_state_dict(self.dit, dit_config.load_from, strict=True)
            else:
                raise ValueError("Must provide checkpoint path for DiT model")
            self.dit = self.dit.to(dit_device, dtype=self.torch_dtype)
            self.dit.eval()
            if getattr(dit_config, "use_compile", False):
                self.dit = torch.compile(self.dit)
            loguru.logger.info("✓ DiT model loaded")
        except Exception as e:
            raise RuntimeError(f"Error loading DiT model: {e}") from e

    def _load_text_encoder(self):
        try:
            if self.config.enable_full_dit_offloading:
                self.dit.to('cpu')

            if self.config.enable_full_dit_offloading:
                text_encoder_device = 'cpu'
            else:
                text_encoder_device = self.device


            text_encoder_config = self.config.text_encoder_config
            if not text_encoder_config.load_from:
                raise ValueError("Must provide checkpoint path for text encoder")

            if text_encoder_config.prompt_template is not None:
                prompt_template = PROMPT_TEMPLATE[text_encoder_config.prompt_template]
                crop_start = prompt_template.get("crop_start", 0)
            else:
                crop_start = 0
                prompt_template = None
            max_length = text_encoder_config.text_len + crop_start

            self.text_encoder = instantiate(
                text_encoder_config.model,
                max_length=max_length,
                text_encoder_path=os.path.join(text_encoder_config.load_from, "llm"),
                prompt_template=prompt_template,
                logger=None,
                device=text_encoder_device,
            )
            loguru.logger.info("✓ HunyuanImage text encoder loaded")
        except Exception as e:
            raise RuntimeError(f"Error loading text encoder: {e}") from e

    def _load_vae(self):
        try:
            vae_config = self.config.vae_config
            self.vae = instantiate(
                vae_config.model,
                vae_path=vae_config.load_from,
            )
            self.vae = self.vae.to(self.device)
            loguru.logger.info("✓ VAE loaded")
        except Exception as e:
            raise RuntimeError(f"Error loading VAE: {e}") from e

    def _load_reprompt_model(self):
        try:
            if self.config.enable_stage1_offloading:
                self.offload()
            reprompt_config = self.config.reprompt_config
            self._reprompt_model = instantiate(reprompt_config.model, models_root_path=reprompt_config.load_from, enable_offloading=self.config.enable_reprompt_model_offloading)
            loguru.logger.info("✓ Reprompt model loaded")
        except Exception as e:
            raise RuntimeError(f"Error loading reprompt model: {e}") from e

    @property
    def refiner_pipeline(self):
        """
        As the refiner model is an optional component, we load it on demand.
        """
        if hasattr(self, '_refiner_pipeline') and self._refiner_pipeline is not None:
            return self._refiner_pipeline
        from hyimage.diffusion.pipelines.hunyuanimage_refiner_pipeline import HunYuanImageRefinerPipeline
        if self.config.enable_stage1_offloading:
            self.offload()
        self._refiner_pipeline = HunYuanImageRefinerPipeline.from_pretrained(self.config.refiner_model_name)
        return self._refiner_pipeline

    @property
    def reprompt_model(self):
        """
        As the reprompt model is an optional component, we load it on demand.
        """
        if hasattr(self, '_reprompt_model') and self._reprompt_model is not None:
            return self._reprompt_model
        self._load_reprompt_model()
        return self._reprompt_model

    def _load_byt5(self):

        assert self.dit is not None, "DiT model must be loaded before byT5"

        if not self.use_byt5:
            self.byt5_kwargs = None
            self.prompt_format = None
            return

        try:

            text_encoder_config = self.config.text_encoder_config

            glyph_root = os.path.join(self.config.text_encoder_config.load_from, "Glyph-SDXL-v2")
            if not os.path.exists(glyph_root):
                raise RuntimeError(
                    f"Glyph checkpoint not found from '{glyph_root}'. \n"
                    "Please download from https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files.\n\n"
                    "- Required files:\n"
                    "    Glyph-SDXL-v2\n"
                    "    ├── assets\n"
                    "    │   ├── color_idx.json\n"
                    "    │   └── multilingual_10-lang_idx.json\n"
                    "    └── checkpoints\n"
                    "        └── byt5_model.pt\n"
                )
                    

            byT5_google_path = os.path.join(text_encoder_config.load_from, "byt5-small")
            if not os.path.exists(byT5_google_path):
                loguru.logger.warning(f"ByT5 google path not found from: {byT5_google_path}. Try downloading from https://huggingface.co/google/byt5-small.")
                byT5_google_path = "google/byt5-small"


            multilingual_prompt_format_color_path = os.path.join(glyph_root, "assets/color_idx.json")
            multilingual_prompt_format_font_path = os.path.join(glyph_root, "assets/multilingual_10-lang_idx.json")

            byt5_args = dict(
                byT5_google_path=byT5_google_path,
                byT5_ckpt_path=os.path.join(glyph_root, "checkpoints/byt5_model.pt"),
                multilingual_prompt_format_color_path=multilingual_prompt_format_color_path,
                multilingual_prompt_format_font_path=multilingual_prompt_format_font_path,
                byt5_max_length=128
            )

            self.byt5_kwargs = load_glyph_byT5_v2(byt5_args, device=self.device)
            self.prompt_format = MultilingualPromptFormat(
                font_path=multilingual_prompt_format_font_path,
                color_path=multilingual_prompt_format_color_path
            )
            loguru.logger.info("✓ byT5 glyph processor loaded")
        except Exception as e:
            raise RuntimeError("Error loading byT5 glyph processor") from e

    def _load_models(self):
        """
        Load all model components.
        """
        loguru.logger.info("Loading HunyuanImage models...")
        self._load_vae()
        self._load_dit()
        self._load_byt5()
        self._load_text_encoder()


    def _encode_text(self, prompt: str, data_type: str = "image"):
        """
        Encode text prompt to embeddings.

        Args:
            prompt: The text prompt
            data_type: The type of data ("image" by default)

        Returns:
            Tuple of (text_emb, text_mask)
        """
        self.text_encoder.to(self.execution_device)
        text_inputs = self.text_encoder.text2tokens(prompt)
        with torch.no_grad():
            text_outputs = self.text_encoder.encode(
                text_inputs,
                data_type=data_type,
            )
            text_emb = text_outputs.hidden_state
            text_mask = text_outputs.attention_mask
        return text_emb, text_mask

    def _encode_glyph(self, prompt: str):
        """
        Encode glyph information using byT5.

        Args:
            prompt: The text prompt

        Returns:
            Tuple of (byt5_emb, byt5_mask)
        """
        if not self.use_byt5:
            return None, None

        if not prompt:
            return (
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=self.device),
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=self.device, dtype=torch.int64)
            )

        try:
            text_prompt_texts = []
            pattern_quote_double = r'\"(.*?)\"'
            pattern_quote_chinese_single = r'‘(.*?)’'
            pattern_quote_chinese_double = r'“(.*?)”'

            matches_quote_double = re.findall(pattern_quote_double, prompt)
            matches_quote_chinese_single = re.findall(pattern_quote_chinese_single, prompt)
            matches_quote_chinese_double = re.findall(pattern_quote_chinese_double, prompt)

            text_prompt_texts.extend(matches_quote_double)
            text_prompt_texts.extend(matches_quote_chinese_single)
            text_prompt_texts.extend(matches_quote_chinese_double)

            if not text_prompt_texts:
                self.ocr_mask = [False]
                return (
                    torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=self.device),
                    torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=self.device, dtype=torch.int64)
                )
            self.ocr_mask = [True]

            text_prompt_style_list = [{'color': None, 'font-family': None} for _ in range(len(text_prompt_texts))]
            glyph_text_formatted = self.prompt_format.format_prompt(text_prompt_texts, text_prompt_style_list)

            byt5_text_ids, byt5_text_mask = self._get_byt5_text_tokens(
                self.byt5_kwargs["byt5_tokenizer"],
                self.byt5_kwargs["byt5_max_length"],
                glyph_text_formatted
            )

            byt5_text_ids = byt5_text_ids.to(device=self.device)
            byt5_text_mask = byt5_text_mask.to(device=self.device)

            byt5_prompt_embeds = self.byt5_kwargs["byt5_model"](
                byt5_text_ids, attention_mask=byt5_text_mask.float()
            )
            byt5_emb = byt5_prompt_embeds[0]

            return byt5_emb, byt5_text_mask
        except Exception as e:
            loguru.logger.warning(f"Warning: Error in glyph encoding, using fallback: {e}")
            return (
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"], 1472), device=self.device),
                torch.zeros((1, self.byt5_kwargs["byt5_max_length"]), device=self.device, dtype=torch.int64)
            )

    def _get_byt5_text_tokens(self, tokenizer, max_length, text_list):
        """
        Get byT5 text tokens.

        Args:
            tokenizer: The tokenizer object
            max_length: Maximum token length
            text_list: List or string of text

        Returns:
            Tuple of (byt5_text_ids, byt5_text_mask)
        """
        if isinstance(text_list, list):
            text_prompt = " ".join(text_list)
        else:
            text_prompt = text_list

        byt5_text_inputs = tokenizer(
            text_prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        byt5_text_ids = byt5_text_inputs.input_ids
        byt5_text_mask = byt5_text_inputs.attention_mask

        return byt5_text_ids, byt5_text_mask

    def _prepare_latents(self, width: int, height: int, generator: torch.Generator, batch_size: int = 1, vae_downsampling_factor: int = 32):
        """
        Prepare initial noise latents.

        Args:
            width: Image width
            height: Image height
            generator: Torch random generator
            batch_size: Batch size

        Returns:
            Latent tensor
        """
        assert width % vae_downsampling_factor == 0 and height % vae_downsampling_factor == 0, (
            f"width and height must be divisible by {vae_downsampling_factor}, but got {width} and {height}"
        )
        latent_width = width // vae_downsampling_factor
        latent_height = height // vae_downsampling_factor
        latent_channels = 64

        if len(self.dit.patch_size) == 3:
            latent_shape = (batch_size, latent_channels, 1, latent_height, latent_width)
        elif len(self.dit.patch_size) == 2:
            latent_shape = (batch_size, latent_channels, latent_height, latent_width)
        else:
            raise ValueError(f"Unsupported patch_size: {self.dit.patch_size}")

        
        # Generate random noise with shape latent_shape
        latents = torch.randn(
            latent_shape,
            device=generator.device,
            dtype=self.torch_dtype,
            generator=generator,
        ).to(device=self.device)

        return latents

    def _denoise_step(self, latents, timesteps, text_emb, text_mask, byt5_emb, byt5_mask, guidance_scale: float = 1.0, timesteps_r=None):
        """
        Perform one denoising step.

        Args:
            latents: Latent tensor
            timesteps: Timesteps tensor
            text_emb: Text embedding
            text_mask: Text mask
            byt5_emb: byT5 embedding
            byt5_mask: byT5 mask
            guidance_scale: Guidance scale
            timesteps_r: Optional next timestep

        Returns:
            Noise prediction tensor
        """
        if byt5_emb is not None and byt5_mask is not None:
            extra_kwargs = {
                "byt5_text_states": byt5_emb,
                "byt5_text_mask": byt5_mask,
            }
        else:
            if self.use_byt5:
                raise ValueError("Must provide byt5_emb and byt5_mask for HunyuanImage 2.1")
            extra_kwargs = {}

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            if hasattr(self.dit, 'guidance_embed') and self.dit.guidance_embed:
                guidance_expand = torch.tensor(
                    [guidance_scale] * latents.shape[0],
                    dtype=torch.float32,
                    device=latents.device
                ).to(latents.dtype) * 1000
            else:
                guidance_expand = None

            noise_pred = self.dit(
                latents,
                timesteps,
                text_states=text_emb,
                encoder_attention_mask=text_mask,
                guidance=guidance_expand,
                return_dict=False,
                extra_kwargs=extra_kwargs,
                timesteps_r=timesteps_r,
            )[0]

        return noise_pred

    def _apply_classifier_free_guidance(self, noise_pred, guidance_scale: float, i: int):
        """
        Apply classifier-free guidance.

        Args:
            noise_pred: Noise prediction tensor
            guidance_scale: Guidance scale

        Returns:
            Guided noise prediction tensor
        """
        if guidance_scale == 1.0:
            return noise_pred

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)


        if self.cfg_mode.startswith("APG_mode_"):
            if i <= self.apg_start_step:
                noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                )
                _ = self.cfg_guider(noise_pred_text, noise_pred_uncond, step=i)
            else:
                noise_pred = self.cfg_guider(noise_pred_text, noise_pred_uncond, step=i)
        elif self.cfg_mode.startswith("MIX_mode_"):

            ocr_mask_bool = torch.tensor(self.ocr_mask, dtype=torch.bool)

            true_idx = torch.where(ocr_mask_bool)[0]
            false_idx = torch.where(~ocr_mask_bool)[0]

            noise_pred_text_true = noise_pred_text[true_idx] if len(true_idx) > 0 else \
                torch.empty((0, noise_pred_text.size(1)), dtype=noise_pred_text.dtype, device=noise_pred_text.device)
            noise_pred_text_false = noise_pred_text[false_idx] if len(false_idx) > 0 else \
                torch.empty((0, noise_pred_text.size(1)), dtype=noise_pred_text.dtype, device=noise_pred_text.device)

            noise_pred_uncond_true = noise_pred_uncond[true_idx] if len(true_idx) > 0 else \
                torch.empty((0, noise_pred_uncond.size(1)), dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)
            noise_pred_uncond_false = noise_pred_uncond[false_idx] if len(false_idx) > 0 else \
                torch.empty((0, noise_pred_uncond.size(1)), dtype=noise_pred_uncond.dtype, device=noise_pred_uncond.device)

            if len(noise_pred_text_true) > 0:
                if i <= self.apg_start_step_ocr:
                    noise_pred_true = noise_pred_uncond_true + guidance_scale * (
                            noise_pred_text_true - noise_pred_uncond_true
                    )
                    _ = self.cfg_guider_ocr(noise_pred_text_true, noise_pred_uncond_true, step=i)
                else:
                    noise_pred_true = self.cfg_guider_ocr(noise_pred_text_true, noise_pred_uncond_true, step=i)
            else:
                noise_pred_true = noise_pred_text_true

            if len(noise_pred_text_false) > 0:
                if i <= self.apg_start_step_general:
                    noise_pred_false = noise_pred_uncond_false + guidance_scale * (
                            noise_pred_text_false - noise_pred_uncond_false
                    )
                    _ = self.cfg_guider_general(noise_pred_text_false, noise_pred_uncond_false, step=i)
                else:
                    noise_pred_false = self.cfg_guider_general(noise_pred_text_false, noise_pred_uncond_false, step=i)
            else:
                noise_pred_false = noise_pred_text_false

            noise_pred = torch.empty_like(noise_pred_text)
            if len(true_idx) > 0:
                noise_pred[true_idx] = noise_pred_true
            if len(false_idx) > 0:
                noise_pred[false_idx] = noise_pred_false

        else:
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=self.guidance_rescale,
                )


        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def _decode_latents(self, latents, reorg_tokens=False):
        """
        Decode latents to images using VAE.

        Args:
            latents: Latent tensor

        Returns:
            Image tensor
        """
        if hasattr(self.vae.config, "shift_factor") and self.vae.config.shift_factor:
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            latents = latents / self.vae.config.scaling_factor

        if reorg_tokens:
            latents = rearrange(latents, "b c f h w -> b f c h w")
            latents = rearrange(latents, "b f (n c) h w -> b (f n) c h w", n=2)
            latents = rearrange(latents, "b f c h w -> b c f h w")
            latents = latents[:, :, 1:]

        if latents.ndim == 5:
            latents = latents.squeeze(2)
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            image = self.vae.decode(latents, return_dict=False)[0]
            
        # Post-process image - remove frame dimension and normalize
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image[:, :, 0]  # Remove frame dimension for images
        image = image.cpu().float()
        
        return image

    def get_timesteps_sigmas(self, sampling_steps: int, shift):
        sigmas = torch.linspace(1, 0, sampling_steps + 1)
        sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
        sigmas = sigmas.to(torch.float32)
        timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=self.device)
        return timesteps, sigmas

    def step(self, latents, noise_pred, sigmas, step_i):
        return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        shift: int = 4,
        negative_prompt: str = "",
        width: int = 2048,
        height: int = 2048,
        use_reprompt: bool = False,
        use_refiner: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = 42,
        **kwargs
    ) -> Image.Image:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the image
            negative_prompt: Negative prompt for guidance
            width: Image width
            height: Image height
            use_reprompt: Whether to use reprompt model
            use_refiner: Whether to use refiner pipeline
            num_inference_steps: Number of denoising steps (overrides config if provided)
            guidance_scale: Strength of classifier-free guidance (overrides config if provided)
            seed: Random seed for reproducibility
            **kwargs: Additional arguments

        Returns:
            Generated PIL Image
        """
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        sampling_steps = num_inference_steps if num_inference_steps is not None else self.default_sampling_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.default_guidance_scale
        shift = shift if shift is not None else self.shift

        user_prompt = prompt
        if use_reprompt:
            if self.config.enable_stage1_offloading:
                self.offload()
            prompt = self.reprompt_model.predict(prompt)


        print("=" * 60)
        print("🖼️  HunyuanImage Generation Task")
        print("-" * 60)
        print(f"Prompt:           {user_prompt}")
        if use_reprompt:
            print(f"Reprompt:         {prompt}")
        if not self.cfg_distilled:
            print(f"Negative Prompt:  {negative_prompt if negative_prompt else '(none)'}")
        print(f"Guidance Scale:   {guidance_scale}")
        print(f"CFG Mode:         {self.cfg_mode}")
        print(f"Guidance Rescale: {self.guidance_rescale}")
        print(f"Shift:            {self.shift}")
        print(f"Seed:             {seed}")
        print(f"Use MeanFlow:     {self.use_meanflow}")
        print(f"Use byT5:         {self.use_byt5}")
        print(f"Image Size:       {width} x {height}")
        print(f"Sampling Steps:   {sampling_steps}")
        print("=" * 60)

        pos_text_emb, pos_text_mask = self._encode_text(prompt)
        neg_text_emb, neg_text_mask = self._encode_text(negative_prompt)

        if self.config.enable_text_encoder_offloading:
            self.text_encoder.to('cpu')

        self.byt5_kwargs['byt5_model'].to(self.execution_device)
        pos_byt5_emb, pos_byt5_mask = self._encode_glyph(prompt)
        neg_byt5_emb, neg_byt5_mask = self._encode_glyph(negative_prompt)
        if self.config.enable_byt5_offloading:
            self.byt5_kwargs['byt5_model'].to('cpu')

        latents = self._prepare_latents(width, height, generator=generator)

        do_classifier_free_guidance = (not self.cfg_distilled) and guidance_scale > 1
        if do_classifier_free_guidance:
            text_emb = torch.cat([neg_text_emb, pos_text_emb])
            text_mask = torch.cat([neg_text_mask, pos_text_mask])

            if self.use_byt5 and pos_byt5_emb is not None and neg_byt5_emb is not None:
                byt5_emb = torch.cat([neg_byt5_emb, pos_byt5_emb])
                byt5_mask = torch.cat([neg_byt5_mask, pos_byt5_mask])
            else:
                byt5_emb = pos_byt5_emb
                byt5_mask = pos_byt5_mask
        else:
            text_emb = pos_text_emb
            text_mask = pos_text_mask
            byt5_emb = pos_byt5_emb
            byt5_mask = pos_byt5_mask

        timesteps, sigmas = self.get_timesteps_sigmas(sampling_steps, shift)

        self.dit.to(self.execution_device)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising", total=len(timesteps))):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            t_expand = t.repeat(latent_model_input.shape[0])
            if self.use_meanflow:
                if i == len(timesteps) - 1:
                    timesteps_r = torch.tensor([0.0], device=self.device)
                else:
                    timesteps_r = timesteps[i + 1]
                timesteps_r = timesteps_r.repeat(latent_model_input.shape[0])
            else:
                timesteps_r = None

            if self.cfg_distilled:
                noise_pred = self._denoise_step(
                    latent_model_input, t_expand, text_emb, text_mask, byt5_emb, byt5_mask, guidance_scale, timesteps_r=timesteps_r,
                )
            else:
                noise_pred = self._denoise_step(
                    latent_model_input, t_expand, text_emb, text_mask, byt5_emb, byt5_mask, timesteps_r=timesteps_r,
                )

            if do_classifier_free_guidance:
                noise_pred = self._apply_classifier_free_guidance(noise_pred, guidance_scale, i)

            latents = self.step(latents, noise_pred, sigmas, i)


        if self.config.enable_full_dit_offloading:
            self.dit.to('cpu')
        self.vae.to(self.execution_device)
        image = self._decode_latents(latents)
        if self.config.enable_vae_offloading:
            self.vae.to('cpu')
        image = (image.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
        pil_image = Image.fromarray(image)

        if use_refiner:
            if self.config.enable_stage1_offloading:
                self.offload()
            pil_image = self.refiner_pipeline(
                image=pil_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                use_reprompt=False,
                use_refiner=False,
                num_inference_steps=4,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            if self.config.enable_refiner_offloading:
                self.refiner_pipeline.offload()

        return pil_image

    @property
    def use_meanflow(self):
        return getattr(self.dit, 'use_meanflow', False)

    @property
    def use_byt5(self):
        return getattr(self.dit, 'glyph_byT5_v2', False)

    @property
    def cfg_distilled(self):
        return getattr(self.dit, 'guidance_embed', False)

    def to(self, device: str | torch.device):
        """
        Move pipeline to specified device.

        Args:
            device: Target device string

        Returns:
            Self
        """
        self.device = device
        if not self.config.enable_full_dit_offloading and not self.config.enable_stage1_offloading:
            if self.dit is not None:
                self.dit = self.dit.to(device, non_blocking=True)
        if not self.config.enable_text_encoder_offloading:
            if self.text_encoder is not None:
                self.text_encoder = self.text_encoder.to(device, non_blocking=True)
        if self.vae is not None:
            self.vae = self.vae.to(device, non_blocking=True)
        return self
    
    def offload(self):
        if self.dit is not None:
            self.dit = self.dit.to('cpu', non_blocking=True)
        if self.text_encoder is not None:
            self.text_encoder = self.text_encoder.to('cpu', non_blocking=True)
        if self.vae is not None:
            self.vae = self.vae.to('cpu', non_blocking=True)
        return self

    def update_config(self, **kwargs):
        """
        Update configuration parameters.

        Args:
            **kwargs: Key-value pairs to update

        Returns:
            Self
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if hasattr(self, key):
                    setattr(self, key, value)
        return self

    @classmethod
    def from_pretrained(cls, model_name: str = "hunyuanimage-v2.1", use_distilled: bool = False, **kwargs):
        """
        Create pipeline from pretrained model.

        Args:
            model_name: Model name, supports "hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"
            use_distilled: Whether to use distilled model (overrides model_name if specified)
            **kwargs: Additional configuration options

        Returns:
            HunyuanImagePipeline instance
        """
        if model_name == "hunyuanimage-v2.1":
            version = "v2.1"
            use_distilled = False
        elif model_name == "hunyuanimage-v2.1-distilled":
            version = "v2.1"
            use_distilled = True
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}. Supported names: 'hunyuanimage-v2.1', 'hunyuanimage-v2.1-distilled'"
            )

        config = HunyuanImagePipelineConfig.create_default(
            version=version, use_distilled=use_distilled, **kwargs
        )
        return cls(config=config)

    @classmethod
    def from_config(cls, config: HunyuanImagePipelineConfig):
        """
        Create pipeline from configuration object.

        Args:
            config: HunyuanImagePipelineConfig instance

        Returns:
            HunyuanImagePipeline instance
        """
        return cls(config=config)


def DiffusionPipeline(model_name: str = "hunyuanimage-v2.1", use_distilled: bool = False, **kwargs):
    """
    Factory function to create HunyuanImagePipeline.

    Args:
        model_name: Model name, supports "hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"
        use_distilled: Whether to use distilled model (overrides model_name if specified)
        **kwargs: Additional configuration options

    Returns:
        HunyuanImagePipeline instance
    """
    return HunyuanImagePipeline.from_pretrained(model_name, use_distilled=use_distilled, **kwargs)
