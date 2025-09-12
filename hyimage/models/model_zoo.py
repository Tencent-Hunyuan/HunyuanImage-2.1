import os
import copy

from hyimage.common.config import LazyCall as L
from hyimage.models.hunyuan.configs.hunyuanimage_config import (
    hunyuanimage_v2_1_cfg,
    hunyuanimage_v2_1_distilled_cfg,
    hunyuanimage_refiner_cfg,
)
from hyimage.models.vae import load_refiner_vae, load_vae
from hyimage.common.config.base_config import (
    DiTConfig,
    RepromptConfig,
    TextEncoderConfig,
    VAEConfig,
)
from hyimage.models.text_encoder import TextEncoder

HUNYUANIMAGE_V2_1_MODEL_ROOT = os.environ.get("HUNYUANIMAGE_V2_1_MODEL_ROOT", "./ckpts")

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# =============================================================================
# V2.1 MODELS
# =============================================================================

def HUNYUANIMAGE_V2_1_TEXT_ENCODER(**kwargs):
    return TextEncoderConfig(
        model=L(TextEncoder)(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision='fp16',
            tokenizer_type="llm",
            text_encoder_path=None,
            prompt_template=None,
            prompt_template_video=None,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=None,
        ),
        prompt_template="dit-llm-encode-v2",
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/text_encoder",
        text_len=1000,
    )


def HUNYUANIMAGE_V2_1_VAE_32x(**kwargs):
    return VAEConfig(
        model=L(load_vae)(
            vae_path=None,
            device="cuda",
        ),
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/vae/vae_2_1",
        cpu_offload=False,
    )


def HUNYUANIMAGE_V2_1_DIT(**kwargs):
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_v2_1_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1.safetensors",
        fp8_scale=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1_fp8_scale.safetensors",
        fp8_load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1_fp8.safetensors",
        use_compile=False,
    )


def HUNYUANIMAGE_V2_1_DIT_CFG_DISTILL(**kwargs):
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_v2_1_distilled_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1-distilled.safetensors",
        fp8_scale=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1-distilled_fp8_scale.safetensors",
        fp8_load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage2.1-distilled_fp8.safetensors",
        use_compile=False,
    )


# =============================================================================
# REFINER MODELS
# =============================================================================

def HUNYUANIMAGE_REFINER_DIT(**kwargs):
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_refiner_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage-refiner.safetensors",
        fp8_scale=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage-refiner_fp8_scale.safetensors",
        fp8_load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/dit/hunyuanimage-refiner_fp8.safetensors",
        use_compile=False,
    )


def HUNYUANIMAGE_REFINER_VAE_16x(**kwargs):
    return VAEConfig(
        model=L(load_refiner_vae)(
            vae_path=None,
            device="cuda",
        ),
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/vae/vae_refiner",
        cpu_offload=False,
    )


def HUNYUANIMAGE_REFINER_TEXT_ENCODER(**kwargs):
    return TextEncoderConfig(
        model=L(TextEncoder)(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision='fp16',
            tokenizer_type="llm",
            text_encoder_path=None,
            prompt_template=None,
            prompt_template_video=None,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=None,
        ),
        prompt_template="dit-llm-encode",
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/text_encoder",
        text_len=256,
    )


# =============================================================================
# SPECIALIZED MODELS
# =============================================================================

def HUNYUANIMAGE_REPROMPT(**kwargs):
    from hyimage.models.reprompt import RePrompt
    
    return RepromptConfig(
        model=L(RePrompt)(
            models_root_path=None,
            device_map="auto",
        ),
        load_from=f"{HUNYUANIMAGE_V2_1_MODEL_ROOT}/reprompt",
    )