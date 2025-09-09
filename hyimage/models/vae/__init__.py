import torch
from pathlib import Path
from hyimage.common.constants import PRECISION_TO_TYPE
from .hunyuanimage_vae import HunyuanVAE2D
from .refiner_vae import AutoencoderKLConv3D

def load_vae(device, vae_path: str = None, vae_precision: str = None):
    config = HunyuanVAE2D.load_config(vae_path)
    vae = HunyuanVAE2D.from_config(config)

    ckpt_path = Path(vae_path) / "pytorch_model.ckpt"
    if not ckpt_path.exists():
        ckpt_path = Path(vae_path) / "pytorch_model.pt"

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    vae_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("vae."):
            vae_ckpt[k.replace("vae.", "")] = v
    vae.load_state_dict(vae_ckpt)

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if device is not None:
        vae = vae.to(device)

    vae.eval()
    return vae


def load_refiner_vae(device, vae_path: str = None, vae_precision: str = "fp16"):
    config = AutoencoderKLConv3D.load_config(vae_path)
    vae = AutoencoderKLConv3D.from_config(config)

    ckpt_path = Path(vae_path) / "pytorch_model.ckpt"
    if not ckpt_path.exists():
        ckpt_path = Path(vae_path) / "pytorch_model.pt"

    ckpt = torch.load(ckpt_path, map_location='cpu')
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    vae_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("vae."):
            vae_ckpt[k.replace("vae.", "")] = v
    vae.load_state_dict(vae_ckpt)

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if device is not None:
        vae = vae.to(device)

    vae.eval()
    return vae