# LoRA network for Z-Image (S3-DiT). No hand-rolled Z-Image model class exists in this repo (unlike
# lora_flux.py/lora_sd3.py, which target Kohya's own model classes) -- this targets the linear layers
# inside diffusers' `ZImageTransformerBlock` directly (attention to_q/to_k/to_v/to_out and the
# feed_forward w1/w2/w3 gated-MLP layers), reusing the same generic LoRAModule Kohya uses for Flux/SD3.
#
# Transformer-only: unlike lora_sd3.py this does not support training the text encoder (Qwen3-4B),
# since it's a different architecture (decoder LM, not CLIP/T5) that would need its own targeting.

import os
from typing import Dict, List, Optional, Type, Union

import torch

from .lora_flux import LoRAModule, LoRAInfModule

import logging

logger = logging.getLogger(__name__)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: List,
    transformer,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    train_block_indices = kwargs.get("train_block_indices", None)  # "all" / "none" / "0,2,5-10" against `layers` only
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    network = LoRANetwork(
        transformer,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_block_indices=train_block_indices,
        verbose=verbose,
    )
    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, transformer, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        transformer,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    # any Linear nested under one of these block classes gets a LoRA adapter. `layers`,
    # `noise_refiner` and `context_refiner` are all made of ZImageTransformerBlock instances, so this
    # single class-name match covers the whole transformer.
    ZIMAGE_TARGET_REPLACE_MODULE = ["ZImageTransformerBlock"]
    LORA_PREFIX_ZIMAGE = "lora_unet"  # ComfyUI-style prefix, matches lora_flux/lora_sd3 convention

    def __init__(
        self,
        transformer,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_block_indices: Optional[str] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        train_block_set = None
        if train_block_indices is not None and train_block_indices not in ("all", ""):
            train_block_set = set()
            for r in train_block_indices.split(","):
                r = r.strip()
                if "-" in r:
                    start, end = map(int, r.split("-"))
                    train_block_set.update(range(start, end + 1))
                else:
                    train_block_set.add(int(r))

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network for Z-Image. base dim (rank): {lora_dim}, alpha: {alpha}")

        loras = []
        skipped = []
        for name, module in transformer.named_modules():
            if module.__class__.__name__ not in LoRANetwork.ZIMAGE_TARGET_REPLACE_MODULE:
                continue

            # only `layers` (the main stack) has a stable, trainable block index; refiners are left
            # alone by train_block_indices filtering and always included
            block_index = None
            if ".layers." in "." + name:
                try:
                    block_index = int(name.split(".layers.")[1].split(".")[0])
                except (IndexError, ValueError):
                    block_index = None

            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ != "Linear":
                    continue

                lora_name = (LoRANetwork.LORA_PREFIX_ZIMAGE + "." + name + "." + child_name).replace(".", "_")

                dim = None
                a = None
                if modules_dim is not None:
                    if lora_name in modules_dim:
                        dim = modules_dim[lora_name]
                        a = modules_alpha[lora_name]
                else:
                    if train_block_set is not None and block_index is not None and block_index not in train_block_set:
                        skipped.append(lora_name)
                        continue
                    dim = lora_dim
                    a = alpha

                if dim is None or dim == 0:
                    skipped.append(lora_name)
                    continue

                lora = module_class(
                    lora_name,
                    child_module,
                    multiplier,
                    dim,
                    a,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                )
                loras.append(lora)

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]] = loras
        self.text_encoder_loras: List = []  # text encoder (Qwen3) LoRA not supported yet

        logger.info(f"create LoRA for Z-Image transformer: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")
        if verbose and len(skipped) > 0:
            logger.info(f"{len(skipped)} modules skipped (dim=0 or excluded block index)")

        names = set()
        for lora in self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        return self.load_state_dict(weights_sd, False)

    def apply_to(self, text_encoders, transformer, apply_text_encoder=False, apply_unet=True):
        if apply_unet:
            logger.info(f"enable LoRA for Z-Image transformer: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, transformer, weights_sd, dtype=None, device=None):
        for lora in self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)
        logger.info("weights are merged")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        params = {"params": [p for lora in self.unet_loras for p in lora.parameters()]}
        lr = unet_lr if unet_lr is not None else default_lr
        if lr is not None and lr != 0:
            params["lr"] = lr
            all_params.append(params)
            lr_descriptions.append("unet")

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, transformer):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, transformer):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from ..library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        for lora in self.unet_loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        for lora in self.unet_loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        for lora in self.unet_loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()
            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)
            org_module._lora_restored = False
            lora.enabled = False
