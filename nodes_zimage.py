import os
import torch

import folder_paths
import comfy.model_management as mm
import comfy.utils
import toml
import json
import time
import shutil
import shlex

script_directory = os.path.dirname(os.path.abspath(__file__))

from .zimage_train_network import ZImageNetworkTrainer
from .library.device_utils import init_ipex
init_ipex()

from .library import train_util
from .train_network import setup_parser as train_network_setup_parser

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ZImageModelSelect:
    """
    Z-Image ships as a diffusers-format checkpoint (transformer/text_encoder/vae subfolders), not a
    single safetensors file, so this takes plain paths (a local diffusers directory or a HF hub repo
    id like Tongyi-MAI/Z-Image-Turbo) rather than folder_paths dropdowns like the Flux/SD3 selectors.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model_path": ("STRING", {"default": "Tongyi-MAI/Z-Image-Turbo", "multiline": False, "tooltip": "local diffusers-format directory, or a HF hub repo id"}),
                },
                "optional": {
                    "text_encoder_path": ("STRING", {"default": "", "multiline": False, "tooltip": "override path for the Qwen3 text encoder, if not bundled with model_path"}),
                    "vae_path": ("STRING", {"default": "", "multiline": False, "tooltip": "override path for the VAE, if not bundled with model_path"}),
                    "lora_path": ("STRING", {"multiline": True, "forceInput": True, "default": "", "tooltip": "pre-trained LoRA path to load (network_weights)"}),
                }
        }

    RETURN_TYPES = ("TRAIN_ZIMAGE_MODELS",)
    RETURN_NAMES = ("zimage_models",)
    FUNCTION = "loadmodel"
    CATEGORY = "FluxTrainer/ZImage"

    def loadmodel(self, model_path, text_encoder_path="", vae_path="", lora_path=""):
        zimage_models = {
            "model_path": model_path,
            "text_encoder_path": text_encoder_path,
            "vae_path": vae_path,
            "lora_path": lora_path,
        }
        return (zimage_models,)


class InitZImageLoRATraining:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "zimage_models": ("TRAIN_ZIMAGE_MODELS",),
            "dataset": ("JSON",),
            "optimizer_settings": ("ARGS",),
            "output_name": ("STRING", {"default": "zimage_lora", "multiline": False}),
            "output_dir": ("STRING", {"default": "zimage_trainer_output", "multiline": False, "tooltip": "path to dataset, root is the 'ComfyUI' folder, with windows portable 'ComfyUI_windows_portable'"}),
            "network_dim": ("INT", {"default": 16, "min": 1, "max": 2048, "step": 1, "tooltip": "network dim"}),
            "network_alpha": ("FLOAT", {"default": 16, "min": 0.0, "max": 2048.0, "step": 0.01, "tooltip": "network alpha"}),
            "learning_rate": ("FLOAT", {"default": 1e-4, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "learning rate"}),
            "max_train_steps": ("INT", {"default": 1500, "min": 1, "max": 100000, "step": 1, "tooltip": "max number of training steps"}),
            "cache_latents": (["disk", "memory", "disabled"], {"tooltip": "caches latents"}),
            "cache_text_encoder_outputs": (["disk", "memory", "disabled"], {"tooltip": "caches text encoder outputs"}),
            "training_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.0001, "tooltip": "shift value for the training distribution of timesteps"}),
            "max_sequence_length": ("INT", {"default": 512, "min": 32, "max": 1024, "step": 8, "tooltip": "max token length for the Qwen3 text encoder"}),
            "fp8_base": ("BOOLEAN", {"default": False, "tooltip": "use fp8 for base model"}),
            "gradient_dtype": (["fp32", "fp16", "bf16"], {"default": "bf16", "tooltip": "the actual dtype training uses"}),
            "save_dtype": (["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16", "tooltip": "the dtype to save checkpoints as"}),
            "sample_prompts": ("STRING", {"multiline": True, "default": "illustration of a kitten | photograph of a turtle", "tooltip": "validation sample prompts, for multiple prompts, separate by `|`"}),
            "gradient_checkpointing": (["enabled", "disabled"], {"default": "enabled", "tooltip": "use gradient checkpointing"}),
            "train_text_encoder": (["disabled", "qwen3"], {"default": "disabled", "tooltip": "also train the Qwen3 text encoder as LoRA; incompatible with cache_text_encoder_outputs"}),
            "text_encoder_lr": ("FLOAT", {"default": 0, "min": 0.0, "max": 10.0, "step": 0.000001, "tooltip": "text encoder learning rate, 0 = use learning_rate"}),
            },
            "optional": {
                "additional_args": ("STRING", {"multiline": True, "default": "", "tooltip": "additional args to pass to the training command"}),
                "resume_args": ("ARGS", {"default": "", "tooltip": "resume args to pass to the training command"}),
                "block_args": ("ARGS", {"default": "", "tooltip": "limit which transformer blocks get a LoRA, from a Flux Train Block Select node. Z-Image module names look like 'lora_unet_layers_5_attention_to_q' -- note the block-select node's '(start-end)' range shorthand only expands names containing '_blocks_', so for Z-Image list block names individually (comma-separated), e.g. 'lora_unet_layers_0,lora_unet_layers_1'"}),
                "loss_args": ("ARGS", {"default": "", "tooltip": "loss args"}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT", "KOHYA_ARGS",)
    RETURN_NAMES = ("network_trainer", "epochs_count", "args",)
    FUNCTION = "init_training"
    CATEGORY = "FluxTrainer/ZImage"

    def init_training(self, zimage_models, dataset, optimizer_settings, sample_prompts, output_name,
                      gradient_dtype, save_dtype, additional_args=None, resume_args=None,
                      block_args=None, gradient_checkpointing="enabled", train_text_encoder="disabled", text_encoder_lr=0,
                      prompt=None, extra_pnginfo=None, loss_args=None, **kwargs):
        mm.soft_empty_cache()

        output_dir = os.path.abspath(kwargs.get("output_dir"))
        os.makedirs(output_dir, exist_ok=True)

        total, used, free = shutil.disk_usage(output_dir)
        required_free_space = 2 * (2**30)
        if free <= required_free_space:
            raise ValueError(f"Insufficient disk space. Required: {required_free_space/2**30}GB. Available: {free/2**30}GB")

        dataset_config = dataset["datasets"]
        dataset_toml = toml.dumps(json.loads(dataset_config))

        import importlib
        zimage_train_network = importlib.import_module(".zimage_train_network", package=__name__.split(".")[0])
        parser = zimage_train_network.setup_parser()

        if additional_args is not None:
            print(f"additional_args: {additional_args}")
            args, _ = parser.parse_known_args(args=shlex.split(additional_args))
        else:
            args, _ = parser.parse_known_args(args=[])

        if kwargs.get("cache_latents") == "memory":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = False
        elif kwargs.get("cache_latents") == "disk":
            kwargs["cache_latents"] = True
            kwargs["cache_latents_to_disk"] = True
            kwargs["caption_dropout_rate"] = 0.0
            kwargs["shuffle_caption"] = False
            kwargs["token_warmup_step"] = 0.0
            kwargs["caption_tag_dropout_rate"] = 0.0
        else:
            kwargs["cache_latents"] = False
            kwargs["cache_latents_to_disk"] = False

        if kwargs.get("cache_text_encoder_outputs") == "memory":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = False
        elif kwargs.get("cache_text_encoder_outputs") == "disk":
            kwargs["cache_text_encoder_outputs"] = True
            kwargs["cache_text_encoder_outputs_to_disk"] = True
        else:
            kwargs["cache_text_encoder_outputs"] = False
            kwargs["cache_text_encoder_outputs_to_disk"] = False

        if '|' in sample_prompts:
            prompts = sample_prompts.split('|')
        else:
            prompts = [sample_prompts]

        config_dict = {
            "sample_prompts": prompts,
            "save_precision": save_dtype,
            "mixed_precision": "bf16",
            "num_cpu_threads_per_process": 1,
            "pretrained_model_name_or_path": zimage_models["model_path"],
            "text_encoder": zimage_models["text_encoder_path"] or None,
            "vae": zimage_models["vae_path"] or None,
            "save_model_as": "safetensors",
            "persistent_data_loader_workers": False,
            "max_data_loader_n_workers": 0,
            "seed": 42,
            "network_module": ".networks.lora_zimage",
            "dataset_config": dataset_toml,
            "output_name": f"{output_name}_rank{kwargs.get('network_dim')}_{save_dtype}",
            "loss_type": "l2",
            "alpha_mask": dataset["alpha_mask"],
            "network_train_unet_only": train_text_encoder == "disabled",
            "disable_mmap_load_safetensors": False,
        }
        if train_text_encoder != "disabled":
            config_dict["text_encoder_lr"] = text_encoder_lr

        gradient_dtype_settings = {
            "fp16": {"full_fp16": True, "full_bf16": False, "mixed_precision": "fp16"},
            "bf16": {"full_bf16": True, "full_fp16": False, "mixed_precision": "bf16"}
        }
        config_dict.update(gradient_dtype_settings.get(gradient_dtype, {}))

        additional_network_args = []
        if block_args:
            additional_network_args.append(block_args["include"])
        if train_text_encoder != "disabled":
            additional_network_args.append("train_text_encoder=True")
        if hasattr(args, 'network_args') and isinstance(args.network_args, list):
            args.network_args.extend(additional_network_args)
        else:
            setattr(args, 'network_args', additional_network_args)

        if gradient_checkpointing == "disabled":
            config_dict["gradient_checkpointing"] = False
        else:
            config_dict["gradient_checkpointing"] = True

        if zimage_models["lora_path"]:
            config_dict["network_weights"] = zimage_models["lora_path"]

        config_dict.update(kwargs)
        config_dict.update(optimizer_settings)

        if loss_args:
            config_dict.update(loss_args)
        if resume_args:
            config_dict.update(resume_args)

        for key, value in config_dict.items():
            setattr(args, key, value)

        saved_args_file_path = os.path.join(output_dir, f"{output_name}_args.json")
        with open(saved_args_file_path, 'w') as f:
            json.dump(vars(args), f, indent=4)

        metadata = {}
        if extra_pnginfo is not None:
            metadata.update(extra_pnginfo["workflow"])
        saved_workflow_file_path = os.path.join(output_dir, f"{output_name}_workflow.json")
        with open(saved_workflow_file_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        with torch.inference_mode(False):
            network_trainer = ZImageNetworkTrainer()
            training_loop = network_trainer.init_train(args)

        epochs_count = network_trainer.num_train_epochs

        trainer = {
            "network_trainer": network_trainer,
            "training_loop": training_loop,
        }
        return (trainer, epochs_count, args)


class ZImageTrainLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "steps": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1, "tooltip": "the step point in training to validate/save"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "INT",)
    RETURN_NAMES = ("network_trainer", "steps",)
    FUNCTION = "train"
    CATEGORY = "FluxTrainer/ZImage"

    def train(self, network_trainer, steps):
        with torch.inference_mode(False):
            training_loop = network_trainer["training_loop"]
            network_trainer = network_trainer["network_trainer"]

            target_global_step = network_trainer.global_step + steps
            comfy_pbar = comfy.utils.ProgressBar(steps)
            network_trainer.comfy_pbar = comfy_pbar

            network_trainer.optimizer_train_fn()

            while network_trainer.global_step < target_global_step:
                training_loop(
                    break_at_steps=target_global_step,
                    epoch=network_trainer.current_epoch.value,
                )
                if network_trainer.global_step >= network_trainer.args.max_train_steps:
                    break

            trainer = {
                "network_trainer": network_trainer,
                "training_loop": training_loop,
            }
        return (trainer, network_trainer.global_step)


class ZImageTrainLoRASave:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": False, "tooltip": "save the whole model state as well"}),
            "copy_to_comfy_lora_folder": ("BOOLEAN", {"default": False, "tooltip": "copy the lora model to the comfy lora folder"}),
             },
        }

    RETURN_TYPES = ("NETWORKTRAINER", "STRING", "INT",)
    RETURN_NAMES = ("network_trainer", "lora_path", "steps",)
    FUNCTION = "save"
    CATEGORY = "FluxTrainer/ZImage"

    def save(self, network_trainer, save_state, copy_to_comfy_lora_folder):
        with torch.inference_mode(False):
            trainer = network_trainer["network_trainer"]
            global_step = trainer.global_step

            ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, global_step)
            trainer.save_model(ckpt_name, trainer.accelerator.unwrap_model(trainer.network), global_step, trainer.current_epoch.value + 1)

            remove_step_no = train_util.get_remove_step_no(trainer.args, global_step)
            if remove_step_no is not None:
                remove_ckpt_name = train_util.get_step_ckpt_name(trainer.args, "." + trainer.args.save_model_as, remove_step_no)
                trainer.remove_model(remove_ckpt_name)

            if save_state:
                train_util.save_and_remove_state_stepwise(trainer.args, trainer.accelerator, global_step)

            lora_path = os.path.join(trainer.args.output_dir, ckpt_name)
            if copy_to_comfy_lora_folder:
                destination_dir = os.path.join(folder_paths.models_dir, "loras", "flux_trainer")
                os.makedirs(destination_dir, exist_ok=True)
                shutil.copy(lora_path, os.path.join(destination_dir, ckpt_name))

        return (network_trainer, lora_path, global_step)


class ZImageTrainEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "network_trainer": ("NETWORKTRAINER",),
            "save_state": ("BOOLEAN", {"default": True}),
             },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lora_name", "metadata", "lora_path",)
    FUNCTION = "endtrain"
    CATEGORY = "FluxTrainer/ZImage"
    OUTPUT_NODE = True

    def endtrain(self, network_trainer, save_state):
        with torch.inference_mode(False):
            network_trainer = network_trainer["network_trainer"]

            network_trainer.metadata["ss_epoch"] = str(network_trainer.num_train_epochs)
            network_trainer.metadata["ss_training_finished_at"] = str(time.time())

            network = network_trainer.accelerator.unwrap_model(network_trainer.network)

            network_trainer.accelerator.end_training()
            network_trainer.optimizer_eval_fn()

            if save_state:
                train_util.save_state_on_train_end(network_trainer.args, network_trainer.accelerator)

            ckpt_name = train_util.get_last_ckpt_name(network_trainer.args, "." + network_trainer.args.save_model_as)
            network_trainer.save_model(ckpt_name, network, network_trainer.global_step, network_trainer.num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")

            final_lora_name = str(network_trainer.args.output_name)
            final_lora_path = os.path.join(network_trainer.args.output_dir, ckpt_name)

            metadata = json.dumps(network_trainer.metadata, indent=2)

            network_trainer = None
            mm.soft_empty_cache()

        return (final_lora_name, metadata, final_lora_path)


NODE_CLASS_MAPPINGS = {
    "ZImageModelSelect": ZImageModelSelect,
    "InitZImageLoRATraining": InitZImageLoRATraining,
    "ZImageTrainLoop": ZImageTrainLoop,
    "ZImageTrainLoRASave": ZImageTrainLoRASave,
    "ZImageTrainEnd": ZImageTrainEnd,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageModelSelect": "Z-Image Model Select",
    "InitZImageLoRATraining": "Init Z-Image LoRA Training",
    "ZImageTrainLoop": "Z-Image Train Loop",
    "ZImageTrainLoRASave": "Z-Image Train LoRA Save",
    "ZImageTrainEnd": "Z-Image Train End",
}
