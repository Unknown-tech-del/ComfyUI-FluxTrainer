import argparse
from typing import Any, Optional

import torch
from accelerate import Accelerator

from .library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .library import strategy_base, strategy_zimage, sd3_train_utils, train_util, zimage_utils
from . import train_network
from .library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def build_cap_feats_list(cap_feats: torch.Tensor, attn_mask: torch.Tensor):
    """
    ZImageTransformer2DModel.forward() takes `cap_feats` as a list of variable-length per-sample
    tensors (see diffusers' ZImagePipeline, which masks out padding before calling the model),
    not a padded batch tensor. Convert the cached/encoded padded batch to that ragged list here,
    right before the model call.
    """
    return [cap_feats[i][attn_mask[i].bool()] for i in range(cap_feats.shape[0])]


class ZImageNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.train_text_encoder = False

    def assert_extra_args(self, args, train_dataset_group: train_util.DatasetGroup):
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        self.train_text_encoder = not args.network_train_unet_only

        if self.train_text_encoder and args.cache_text_encoder_outputs:
            raise ValueError(
                "cache_text_encoder_outputs cannot be used when the text encoder is trained "
                "(disable network_train_unet_only=False to train the Qwen3 text encoder, or turn off caching)"
            )

        train_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype

        # library/zimage_models.py is our own patched vendored copy of diffusers' model (not the
        # diffusers package's own class) -- it fixes the two ops that broke under a blanket fp8
        # cast (RMSNorm's elementwise multiply, and _prepare_sequence's torch.where), matching the
        # defensive dtype-casting Kohya's hand-written Flux/SD3 model classes already do.
        transformer = zimage_utils.load_transformer(args.pretrained_model_name_or_path, loading_dtype, "cpu")
        if args.fp8_base:
            fp8_dtype = torch.float8_e4m3fn if args.fp8_dtype == "e4m3" else torch.float8_e5m2
            transformer.to(fp8_dtype)

        text_encoder = zimage_utils.load_text_encoder(
            args.text_encoder if args.text_encoder else args.pretrained_model_name_or_path, weight_dtype, "cpu"
        )
        vae = zimage_utils.load_vae(args.vae if args.vae else args.pretrained_model_name_or_path, weight_dtype, "cpu")

        return "zimage", [text_encoder], vae, transformer

    def get_tokenize_strategy(self, args):
        return strategy_zimage.ZImageTokenizeStrategy(
            args.text_encoder if args.text_encoder else args.pretrained_model_name_or_path,
            args.max_sequence_length,
            args.tokenizer_cache_dir,
        )

    def get_tokenizers(self, tokenize_strategy: strategy_zimage.ZImageTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_zimage.ZImageLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_zimage.ZImageTextEncodingStrategy(args.caption_dropout_rate)

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_zimage.ZImageTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check
            )
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs and not self.train_text_encoder:
            return None  # text encoder output is fully cached
        return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_text_encoder]

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # network's actual text_encoder_loras may end up empty even if train_text_encoder was
        # requested (e.g. loading pre-trained transformer-only weights), so trust the network here
        self.train_text_encoder = self.train_text_encoder and len(network.text_encoder_loras) > 0

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if not args.cache_text_encoder_outputs:
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            return

        if not args.lowram:
            logger.info("move vae and transformer to cpu to save memory")
            org_vae_device = vae.device
            org_unet_device = unet.device
            vae.to("cpu")
            unet.to("cpu")
            clean_memory_on_device(accelerator.device)

        text_encoders[0].to(accelerator.device, dtype=weight_dtype)

        with accelerator.autocast():
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

        if args.sample_prompts is not None:
            tokenize_strategy: strategy_zimage.ZImageTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy: strategy_zimage.ZImageTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

            prompts = []
            for line in args.sample_prompts:
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    prompts.append(line)

            for i in range(len(prompts)):
                prompt_dict = prompts[i]
                if isinstance(prompt_dict, str):
                    from .library.train_util import line_to_prompt_dict

                    prompt_dict = line_to_prompt_dict(prompt_dict)
                    prompts[i] = prompt_dict
                prompt_dict["enum"] = i
                prompt_dict.pop("subset", None)

            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            tokens = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, text_encoders, tokens, enable_dropout=False
                            )
            self.sample_prompts_te_outputs = sample_prompts_te_outputs

        accelerator.wait_for_everyone()

        logger.info("move text encoder back to cpu")
        text_encoders[0].to("cpu")
        clean_memory_on_device(accelerator.device)

        if not args.lowram:
            vae.to(org_vae_device)
            unet.to(org_unet_device)

    def sample_images(self, epoch, global_step, validation_settings):
        # TODO: wire up diffusers' ZImagePipeline for validation samples; not implemented in this
        # first pass. Training and checkpoint saving work without it.
        return None

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.training_shift)

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return zimage_utils.encode_images_to_latents(vae, images)

    def shift_scale_latents(self, args, latents):
        vae = self.vae
        return zimage_utils.shift_scale_latents(vae, latents)

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
    ):
        noise = torch.randn_like(latents)

        noisy_model_input, timesteps, sigmas = sd3_train_utils.get_noisy_model_input_and_timesteps(
            args, latents, noise, accelerator.device, weight_dtype
        )

        cap_feats, attn_mask = text_encoder_conds
        cap_feats = cap_feats.to(accelerator.device)
        attn_mask = attn_mask.to(accelerator.device)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            cap_feats.requires_grad_(True)

        cap_feats_list = build_cap_feats_list(cap_feats, attn_mask)

        # ZImageTransformer2DModel takes x as a list of (C, F, H, W) tensors (one per sample --
        # F is a frame/temporal dim, 1 for plain images) and timesteps normalized to [0, 1],
        # matching diffusers' ZImagePipeline call convention.
        x_list = [noisy_model_input[i].unsqueeze(1) for i in range(noisy_model_input.shape[0])]
        t_norm = (timesteps / 1000.0).to(noisy_model_input.dtype)

        with accelerator.autocast():
            model_pred = unet(x_list, t_norm, cap_feats_list, return_dict=False)[0]
            model_pred = torch.stack([p.squeeze(1) for p in model_pred], dim=0)

        # rectified-flow / flow-matching target: velocity from data to noise
        target = noise - latents

        weighting = sd3_train_utils.compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False)

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_training_shift"] = args.training_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.train_text_encoder

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # set top parameter requires_grad = True for gradient checkpointing to work, same reasoning
        # as T5XXL in sd3_train_network.py
        text_encoder.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.to(te_weight_dtype)

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoding_strategy: strategy_zimage.ZImageTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()
            batch["text_encoder_outputs_list"] = text_encoding_strategy.drop_cached_text_encoder_outputs(*text_encoder_outputs_list)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    # train_network.setup_parser() already calls train_util.add_dit_training_arguments() internally
    # (cache_text_encoder_outputs, etc.) -- calling it again here would raise a duplicate-argument error.
    sd3_train_utils.add_sd3_training_arguments(parser)  # reuse: weighting_scheme, logit_mean/std, mode_scale, training_shift, min/max_timestep
    parser.add_argument("--text_encoder", type=str, default=None, help="path to Z-Image's Qwen3 text encoder, if separate from the transformer checkpoint")
    # --vae already exists on the base parser (added generically for SDXL/SD3-style trainers)
    parser.add_argument("--max_sequence_length", type=int, default=512, help="max token length for the Qwen3 text encoder")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = ZImageNetworkTrainer()
    trainer.train(args)
