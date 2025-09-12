from dataclasses import dataclass
from typing import Optional

from hyimage.common.config.lazy import DictConfig


@dataclass
class DiTConfig:
    model: DictConfig
    use_lora: bool = False
    use_cpu_offload: bool = False
    gradient_checkpointing: bool = False
    load_from: Optional[str] = None
    fp8_scale: Optional[str] = None
    fp8_load_from: Optional[str] = None
    use_compile: bool = False


@dataclass
class VAEConfig:
    model: DictConfig
    load_from: str
    cpu_offload: bool = False
    enable_tiling: bool = False


@dataclass
class TextEncoderConfig:
    model: DictConfig
    load_from: str
    prompt_template: Optional[str] = None
    text_len: Optional[int] = None


@dataclass
class RepromptConfig:
    model: DictConfig
    load_from: str