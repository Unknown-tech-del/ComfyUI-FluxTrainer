# LoRA network for Z-Image (S3-DiT + Qwen3-4B text encoder). No hand-rolled Z-Image model class
# exists in this repo (unlike lora_flux.py/lora_sd3.py, which target Kohya's own model classes) --
# this targets the linear layers inside diffusers' `ZImageTransformerBlock` directly (attention
# to_q/to_k/to_v/to_out and the feed_forward w1/w2/w3 gated-MLP layers) for the transformer, and
# `Qwen3DecoderLayer` (q/k/v/o_proj, gate/up/down_proj) for the text encoder, reusing the same
# generic LoRAModule Kohya uses for Flux/SD3.

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

    # block selection, same convention as lora_flux.py: a comma-separated list of substrings; a
    # lora is kept only if its full generated name contains at least one of them. Produced by the
    # existing "Flux Train Block Select" node, e.g. only_if_contains=lora_unet_layers_(0-15)_...
    only_if_contains = kwargs.get("only_if_contains", None)
    if only_if_contains is not None:
        only_if_contains = [w.strip() for w in only_if_contains.split(",")]
    exclude_if_contains = kwargs.get("exclude_if_contains", None)
    if exclude_if_contains is not None:
        exclude_if_contains = [w.strip() for w in exclude_if_contains.split(",")]

    train_text_encoder = kwargs.get("train_text_encoder", False)
    if train_text_encoder is not None:
        train_text_encoder = True if train_text_encoder == "True" else bool(train_text_encoder)

    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    network = LoRANetwork(
        text_encoders,
        transformer,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        only_if_contains=only_if_contains,
        exclude_if_contains=exclude_if_contains,
        train_text_encoder=train_text_encoder,
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
    train_text_encoder = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]
        if lora_name.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
            train_text_encoder = True

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders,
        transformer,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_text_encoder=train_text_encoder,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    # any Linear nested under one of these block classes gets a LoRA adapter. `layers`,
    # `noise_refiner` and `context_refiner` are all made of ZImageTransformerBlock instances, so this
    # single class-name match covers the whole transformer.
    ZIMAGE_TARGET_REPLACE_MODULE = ["ZImageTransformerBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["Qwen3DecoderLayer"]
    LORA_PREFIX_ZIMAGE = "lora_unet"  # ComfyUI-style prefix, matches lora_flux/lora_sd3 convention
    LORA_PREFIX_TEXT_ENCODER = "lora_te1"

    def __init__(
        self,
        text_encoders: List,
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
        only_if_contains: Optional[List[str]] = None,
        exclude_if_contains: Optional[List[str]] = None,
        train_text_encoder: bool = False,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_text_encoder = train_text_encoder

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network for Z-Image. base dim (rank): {lora_dim}, alpha: {alpha}")

        def create_modules(root_module, target_replace_modules, prefix) -> (List, List):
            found = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ not in target_replace_modules:
                    continue

                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ != "Linear":
                        continue

                    lora_name = (prefix + "." + name + "." + child_name).replace(".", "_")

                    if only_if_contains is not None and not any(w in lora_name for w in only_if_contains):
                        skipped.append(lora_name)
                        continue
                    if exclude_if_contains is not None and any(w in lora_name for w in exclude_if_contains):
                        skipped.append(lora_name)
                        continue

                    dim = None
                    a = None
                    if modules_dim is not None:
                        if lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            a = modules_alpha[lora_name]
                    else:
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
                    found.append(lora)
            return found, skipped

        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_unet = create_modules(
            transformer, LoRANetwork.ZIMAGE_TARGET_REPLACE_MODULE, LoRANetwork.LORA_PREFIX_ZIMAGE
        )

        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_te = []
        if train_text_encoder or (modules_dim is not None and self._has_text_encoder_weights(modules_dim)):
            (text_encoder,) = text_encoders
            self.text_encoder_loras, skipped_te = create_modules(
                text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE, LoRANetwork.LORA_PREFIX_TEXT_ENCODER
            )

        logger.info(f"create LoRA for Z-Image transformer: {len(self.unet_loras)} modules.")
        if self.text_encoder_loras:
            logger.info(f"create LoRA for Z-Image text encoder (Qwen3): {len(self.text_encoder_loras)} modules.")
        if verbose:
            for lora in self.unet_loras + self.text_encoder_loras:
                logger.info(f"\t{lora.lora_name:60} {lora.lora_dim}, {lora.alpha}")
        skipped = skipped_unet + skipped_te
        if verbose and len(skipped) > 0:
            logger.info(f"{len(skipped)} modules skipped (excluded by block selection or dim=0)")

        names = set()
        for lora in self.unet_loras + self.text_encoder_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    @staticmethod
    def _has_text_encoder_weights(modules_dim: Dict[str, int]) -> bool:
        return any(name.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER) for name in modules_dim)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.unet_loras + self.text_encoder_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.unet_loras + self.text_encoder_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")
        return self.load_state_dict(weights_sd, False)

    def apply_to(self, text_encoders, transformer, apply_text_encoder=True, apply_unet=True):
        if apply_unet:
            logger.info(f"enable LoRA for Z-Image transformer: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        if apply_text_encoder:
            if self.text_encoder_loras:
                logger.info(f"enable LoRA for Z-Image text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        for lora in self.unet_loras + self.text_encoder_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, transformer, weights_sd, dtype=None, device=None):
        for lora in self.unet_loras + self.text_encoder_loras:
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

        if self.text_encoder_loras:
            te_lr = text_encoder_lr[0] if isinstance(text_encoder_lr, list) else text_encoder_lr
            te_lr = te_lr if te_lr is not None and te_lr != 0 else default_lr
            if te_lr is not None and te_lr != 0:
                all_params.append({"params": [p for lora in self.text_encoder_loras for p in lora.parameters()], "lr": te_lr})
                lr_descriptions.append("textencoder")

        if self.unet_loras:
            unet_lr = unet_lr if unet_lr is not None else default_lr
            if unet_lr is not None and unet_lr != 0:
                all_params.append({"params": [p for lora in self.unet_loras for p in lora.parameters()], "lr": unet_lr})
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
        for lora in self.unet_loras + self.text_encoder_loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        for lora in self.unet_loras + self.text_encoder_loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        for lora in self.unet_loras + self.text_encoder_loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()
            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)
            org_module._lora_restored = False
            lora.enabled = False
