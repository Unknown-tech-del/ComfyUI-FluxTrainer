import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from . import train_util
from .strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# Z-Image's text conditioning ("cap_feats") comes from a Qwen3-4B causal LM, prompted through its
# chat template rather than encoded as a bare string like CLIP/T5. This mirrors diffusers'
# ZImagePipeline._get_qwen_prompt_embeds: apply_chat_template(..., enable_thinking=True), take
# hidden_states[-2], then mask out the template preamble with the attention mask.
DEFAULT_MAX_SEQUENCE_LENGTH = 512


class ZImageTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        tokenizer_cache_dir: Optional[str] = None,
    ) -> None:
        from transformers import AutoTokenizer
        from .zimage_utils import _resolve_component_path

        self.max_sequence_length = max_sequence_length
        # Z-Image's tokenizer files live in their own top-level `tokenizer/` folder, separate from
        # `text_encoder/` (which holds only the model weights/config) -- diffusers convention.
        load_path, kwargs = _resolve_component_path(pretrained_model_name_or_path, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=tokenizer_cache_dir, **kwargs)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        assert self.tokenizer is not None, "tokenizer must be set with set_tokenizer() before calling tokenize()"

        texts = [text] if isinstance(text, str) else text
        templated = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            for t in texts
        ]

        tokens = self.tokenizer(
            templated,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return [tokens["input_ids"], tokens["attention_mask"]]


class ZImageTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, dropout_rate: float = 0.0) -> None:
        self.dropout_rate = dropout_rate

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        enable_dropout: bool = True,
    ) -> List[torch.Tensor]:
        """
        Returns [cap_feats, attn_mask]. cap_feats is a padded (batch, seq, dim) tensor; callers that
        need the ragged per-sample sequences the transformer expects should mask with attn_mask
        (see zimage_train_network.build_cap_feats_list).
        """
        (text_encoder,) = models
        input_ids, attn_mask = tokens

        import random

        batch_size = input_ids.shape[0]
        non_drop_indices = []
        for i in range(batch_size):
            drop = enable_dropout and self.dropout_rate > 0.0 and random.random() < self.dropout_rate
            if not drop:
                non_drop_indices.append(i)

        if len(non_drop_indices) > 0:
            nd_input_ids = input_ids[non_drop_indices].to(text_encoder.device)
            nd_attn_mask = attn_mask[non_drop_indices].to(text_encoder.device)
            out = text_encoder(nd_input_ids, attention_mask=nd_attn_mask, output_hidden_states=True)
            nd_cap_feats = out.hidden_states[-2]

        seq_len = input_ids.shape[1]
        dim = nd_cap_feats.shape[-1] if len(non_drop_indices) > 0 else text_encoder.config.hidden_size
        if len(non_drop_indices) == batch_size:
            cap_feats = nd_cap_feats
        else:
            cap_feats = torch.zeros((batch_size, seq_len, dim), device=text_encoder.device, dtype=torch.float32)
            if len(non_drop_indices) > 0:
                cap_feats[non_drop_indices] = nd_cap_feats
            attn_mask = attn_mask.clone()
            drop_indices = [i for i in range(batch_size) if i not in non_drop_indices]
            attn_mask[drop_indices] = 0

        return [cap_feats, attn_mask]

    def drop_cached_text_encoder_outputs(self, cap_feats: torch.Tensor, attn_mask: torch.Tensor):
        if self.dropout_rate <= 0.0:
            return [cap_feats, attn_mask]
        import random

        cap_feats = cap_feats.clone()
        attn_mask = attn_mask.clone()
        for i in range(cap_feats.shape[0]):
            if random.random() < self.dropout_rate:
                cap_feats[i] = torch.zeros_like(cap_feats[i])
                attn_mask[i] = torch.zeros_like(attn_mask[i])
        return [cap_feats, attn_mask]


class ZImageTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    ZIMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_zimage_te.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + ZImageTextEncoderOutputsCachingStrategy.ZIMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            npz = np.load(npz_path)
            if "cap_feats" not in npz or "attn_mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e
        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        return [data["cap_feats"], data["attn_mask"]]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        zimage_text_encoding_strategy: ZImageTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            cap_feats, attn_mask = zimage_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens, enable_dropout=False)

        if cap_feats.dtype == torch.bfloat16:
            cap_feats = cap_feats.float()
        cap_feats = cap_feats.cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()

        for i, info in enumerate(infos):
            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, cap_feats=cap_feats[i], attn_mask=attn_mask[i])
            else:
                info.text_encoder_outputs = (cap_feats[i], attn_mask[i])


class ZImageLatentsCachingStrategy(LatentsCachingStrategy):
    ZIMAGE_LATENTS_NPZ_SUFFIX = "_zimage.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return ZImageLatentsCachingStrategy.ZIMAGE_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + ZImageLatentsCachingStrategy.ZIMAGE_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(self, npz_path: str, bucket_reso: Tuple[int, int]):
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        # cache raw (unshifted/unscaled) latents; the trainer's shift_scale_latents() applies
        # Z-Image/Flux VAE normalization afterward, same convention as sd3_train_network.py
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).latent_dist.sample().to("cpu")

        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
