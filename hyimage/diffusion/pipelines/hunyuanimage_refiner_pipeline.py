from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from einops import rearrange

from .hunyuanimage_pipeline import HunyuanImagePipeline, HunyuanImagePipelineConfig

from hyimage.models.model_zoo import (
    HUNYUANIMAGE_REFINER_DIT,
    HUNYUANIMAGE_REFINER_VAE_16x,
    HUNYUANIMAGE_REFINER_TEXT_ENCODER,
)


@dataclass
class HunYuanImageRefinerPipelineConfig(HunyuanImagePipelineConfig):
    """
    Configuration class for HunyuanImage refiner pipeline.
    
    Inherits from HunyuanImagePipelineConfig and overrides specific parameters
    for the refiner functionality.
    """
    
    default_sampling_steps: int = 4
    shift: int = 1
    version: str = "v1.0"
    cfg_mode: str = ""

    @classmethod
    def create_default(
        cls,
        version: str = "v1.0",
        use_distilled: bool = False,
        **kwargs,
    ):
        dit_config = HUNYUANIMAGE_REFINER_DIT()
        vae_config = HUNYUANIMAGE_REFINER_VAE_16x()
        text_encoder_config = HUNYUANIMAGE_REFINER_TEXT_ENCODER()

        return cls(
            dit_config=dit_config,
            vae_config=vae_config,
            text_encoder_config=text_encoder_config,
            reprompt_config=None,
            version=version,
            **kwargs,
        )


class HunYuanImageRefinerPipeline(HunyuanImagePipeline):
    """A refiner pipeline for HunyuanImage that inherits from the main pipeline.
    
    This pipeline refines existing images using the same model architecture
    but with different default parameters and an image input.
    """
    
    def __init__(self, config: HunYuanImageRefinerPipelineConfig, **kwargs):
        """Initialize the refiner pipeline.
        
        Args:
            config: Refiner-specific configuration
            **kwargs: Additional arguments passed to parent class
        """
        assert isinstance(config, HunYuanImageRefinerPipelineConfig)
        super().__init__(config, **kwargs)
        assert self.cfg_distilled
        
    def _condition_aug(self, latents, noise=None, strength=0.25):
        """Apply conditioning augmentation for refiner.
        
        Args:
            latents: Input latents tensor
            noise: Optional noise tensor, if None will be generated
            strength: Augmentation strength factor
            
        Returns:
            Augmented latents tensor
        """
        if noise is None:
            noise = torch.randn_like(latents)
        return strength * noise + (1 - strength) * latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 2048,
        height: int = 2048,
        use_reprompt: bool = False,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        shift: int = 4,
        seed: Optional[int] = 42,
        image: Optional[Image.Image] = None,
        **kwargs,
    ) -> Image.Image:
        """Refine an existing image using text guidance.

        Args:
            prompt: Text prompt describing the desired refinement
            negative_prompt: Negative prompt for guidance
            width: Image width
            height: Image height
            use_reprompt: Whether to use reprompt (ignored for refiner)
            num_inference_steps: Number of denoising steps (overrides config if provided)
            guidance_scale: Strength of classifier-free guidance (overrides config if provided)
            seed: Random seed for reproducibility
            image: Image to be refined (required for refiner)
            **kwargs: Additional arguments

        Returns:
            Refined PIL Image
        """
        if image is None:
            raise ValueError("Image parameter is required for refiner pipeline")
            
        if seed is not None:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            torch.manual_seed(seed)
        else:
            generator = None

        sampling_steps = (
            num_inference_steps
            if num_inference_steps is not None
            else self.default_sampling_steps
        )
        guidance_scale = (
            guidance_scale if guidance_scale is not None else self.default_guidance_scale
        )
        shift = shift if shift is not None else self.shift

        # Print log about current refinement task
        print("=" * 60)
        print("🔧 HunyuanImage Refinement Task")
        print("-" * 60)
        print(f"Prompt:           {prompt}")
        print(f"Guidance Scale:   {guidance_scale}")
        print(f"Shift:            {self.shift}")
        print(f"Seed:             {seed}")
        print(f"Image Size:       {width} x {height}")
        print(f"Sampling Steps:   {sampling_steps}")
        print("=" * 60)

        # Encode prompts
        self.text_encoder.to(self.execution_device)
        pos_text_emb, pos_text_mask = self._encode_text(prompt)
        if self.config.enable_text_encoder_offloading:
            self.text_encoder.to('cpu')

        latents = self._prepare_latents(width, height, generator=generator, vae_downsampling_factor=16)

        _pil_to_tensor = T.Compose(
            [
                T.ToTensor(),  # convert to tensor and normalize to [0, 1]
                T.Normalize([0.5], [0.5]),  # transform to [-1, 1]
            ]
        )

        image_tensor = (
            _pil_to_tensor(image).unsqueeze(0).to("cuda", dtype=self.vae.dtype)
        )
        image_tensor = image_tensor.unsqueeze(2)

        with torch.no_grad():
            self.vae.to(self.execution_device)
            cond_latents = self.vae.encode(
                image_tensor.to(self.execution_device, dtype=self.vae.dtype)
            ).latent_dist.sample()
            if self.config.enable_vae_offloading:
                self.vae.to('cpu')

        # reorg tokens
        cond_latents = torch.cat((cond_latents[:, :, :1], cond_latents), dim=2)
        cond_latents = rearrange(cond_latents, "b c f h w -> b f c h w")
        cond_latents = rearrange(cond_latents, "b (f n) c h w -> b f (n c) h w", n=2)
        cond_latents = rearrange(cond_latents, "b f c h w -> b c f h w").contiguous()

        if (
            hasattr(self.vae.config, "shift_factor")
            and self.vae.config.shift_factor
        ):
            cond_latents.sub_(self.vae.config.shift_factor).mul_(
                self.vae.config.scaling_factor
            )
        else:
            cond_latents.mul_(self.vae.config.scaling_factor)
        
        
        # Apply conditioning augmentation
        cond_latents = self._condition_aug(cond_latents)

        timesteps, sigmas = self.get_timesteps_sigmas(sampling_steps, shift)

        text_emb = pos_text_emb
        text_mask = pos_text_mask

        self.dit.to(self.execution_device)
        for i, t in enumerate(tqdm(timesteps, desc="Refining", total=len(timesteps))):
            # Concatenate noise latents with condition latents for refiner input
            latent_model_input = torch.cat([latents, cond_latents], dim=1)
            t_expand = t.repeat(latent_model_input.shape[0])

            # Predict noise with guidance
            noise_pred = self._denoise_step(
                latent_model_input,
                t_expand,
                text_emb,
                text_mask,
                None,
                None,
                guidance_scale,
                timesteps_r=None,
            )

            latents = self.step(latents, noise_pred, sigmas, i)
        if self.config.enable_full_dit_offloading:
            self.dit.to('cpu')
        
        self.vae.to(self.execution_device)
        refined_image = self._decode_latents(latents, reorg_tokens=True)
        if self.config.enable_vae_offloading:
            self.vae.to('cpu')

        # Convert to PIL Image
        refined_image = (refined_image.squeeze(0).permute(1, 2, 0) * 255).byte().numpy()
        pil_image = Image.fromarray(refined_image)

        return pil_image

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "hunyuanimage-refiner",
        use_distilled: bool = False,
        **kwargs,
    ):
        """Create refiner pipeline from pretrained model.
        
        Args:
            model_name: Model name, currently only supports "hunyuanimage-refiner"
            use_distilled: Whether to use distilled model (unused for refiner)
            **kwargs: Additional configuration options
        """
        if model_name == "hunyuanimage-refiner":
            version = "v1.0"
        else:
            raise ValueError(
                f"Unsupported refiner model name: {model_name}. Supported names: 'hunyuanimage-refiner'"
            )

        config = HunYuanImageRefinerPipelineConfig.create_default(
            version=version, **kwargs
        )

        return cls(config=config)

    @classmethod
    def from_config(cls, config: Union[HunYuanImageRefinerPipelineConfig, HunyuanImagePipelineConfig]):
        """Create refiner pipeline from configuration object.
        
        Args:
            config: Configuration object for the pipeline
            
        Returns:
            Initialized refiner pipeline instance
        """
        return cls(config=config)


# Convenience function for easy access
def RefinerPipeline(
    model_name: str = "hunyuanimage-refiner",
    **kwargs,
):
    """Factory function to create HunYuanImageRefinerPipeline.

    Args:
        model_name: Model name, currently only supports "hunyuanimage-refiner"
        **kwargs: Additional configuration options
        
    Returns:
        Initialized refiner pipeline instance
    """
    return HunYuanImageRefinerPipeline.from_pretrained(
        model_name, **kwargs
    )
