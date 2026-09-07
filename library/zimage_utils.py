import os
from typing import Optional, Tuple

import torch

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Z-Image (Alibaba Tongyi-MAI) is not supported by Kohya's sd-scripts model loaders, so unlike
# flux_utils.py / sd3_utils.py this module loads the official `diffusers` implementation directly
# instead of a hand-rolled state-dict parser. This means Z-Image training uses diffusers' model
# classes end-to-end (transformer, VAE, text encoder) rather than Kohya's custom ones.
#
# Requires diffusers>=0.36 (ZImageTransformer2DModel / ZImagePipeline) and a transformers version
# with Qwen3 support.

DEFAULT_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


def _is_diffusers_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "model_index.json"))


def load_transformer(
    pretrained_model_name_or_path: str,
    weight_dtype: Optional[torch.dtype],
    device: str = "cpu",
    subfolder: str = "transformer",
):
    from diffusers import ZImageTransformer2DModel

    if _is_diffusers_dir(pretrained_model_name_or_path):
        transformer = ZImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, torch_dtype=weight_dtype
        )
    elif os.path.isdir(pretrained_model_name_or_path) or pretrained_model_name_or_path.endswith(
        (".safetensors", ".bin", ".pt")
    ):
        # single transformer subfolder/checkpoint (no model_index.json at the root)
        transformer = ZImageTransformer2DModel.from_pretrained(pretrained_model_name_or_path, torch_dtype=weight_dtype)
    else:
        # treat as a HF hub repo id
        transformer = ZImageTransformer2DModel.from_pretrained(
            pretrained_model_name_or_path, subfolder=subfolder, torch_dtype=weight_dtype
        )

    transformer.to(device)
    return transformer


def load_text_encoder(
    pretrained_model_name_or_path: str,
    weight_dtype: Optional[torch.dtype],
    device: str = "cpu",
    subfolder: str = "text_encoder",
):
    """Z-Image uses a Qwen3-4B causal LM as its text encoder. Tokenizer is loaded separately by
    ZImageTokenizeStrategy since it's needed before the model itself in the training setup order."""
    from transformers import AutoModel

    load_path, kwargs = _resolve_component_path(pretrained_model_name_or_path, subfolder)
    text_encoder = AutoModel.from_pretrained(load_path, torch_dtype=weight_dtype, **kwargs)
    text_encoder.to(device)
    text_encoder.eval()
    return text_encoder


def load_vae(
    pretrained_model_name_or_path: str,
    weight_dtype: Optional[torch.dtype],
    device: str = "cpu",
    subfolder: str = "vae",
):
    """Z-Image reuses the Flux VAE (AutoencoderKL)."""
    from diffusers import AutoencoderKL

    load_path, kwargs = _resolve_component_path(pretrained_model_name_or_path, subfolder)
    vae = AutoencoderKL.from_pretrained(load_path, torch_dtype=weight_dtype, **kwargs)
    vae.to(device)
    vae.eval()
    return vae


def _resolve_component_path(pretrained_model_name_or_path: str, subfolder: str) -> Tuple[str, dict]:
    """
    Returns (load_path, kwargs) so a component can be loaded either from a diffusers-layout
    directory/hub repo (root + subfolder=...) or from a path that already points directly at the
    component's own folder (e.g. a separately-downloaded text_encoder/ dir).
    """
    if _is_diffusers_dir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path, {"subfolder": subfolder}
    if os.path.isdir(pretrained_model_name_or_path):
        candidate = os.path.join(pretrained_model_name_or_path, subfolder)
        if os.path.isdir(candidate):
            return pretrained_model_name_or_path, {"subfolder": subfolder}
        return pretrained_model_name_or_path, {}
    # HF hub repo id
    return pretrained_model_name_or_path, {"subfolder": subfolder}


def encode_images_to_latents(vae, images: torch.Tensor) -> torch.Tensor:
    latents = vae.encode(images).latent_dist.sample()
    return latents


def shift_scale_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    # standard diffusers AutoencoderKL scaling, same convention Flux uses
    return (latents - vae.config.shift_factor) * vae.config.scaling_factor


def unshift_unscale_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    return latents / vae.config.scaling_factor + vae.config.shift_factor
