from accelerate.utils import write_basic_config
write_basic_config()
import warnings

import logging
import os
import math
import shutil
from pathlib import Path
import numpy as np
from einops import rearrange
import accelerate
from collections import defaultdict
import time
import cv2
from PIL import Image, ImageDraw, ImageFont

from tqdm.auto import tqdm

import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from packaging import version

from diffusers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from src.utils import parse_args, encode_video_image, get_add_time_ids, get_samples
from src.datasets.dataset_utils import get_dataloader
from src.models import UNetSpatioTemporalConditionModel, ControlNetModel
from src.pipelines import StableVideoControlNullModelPipeline

if not is_wandb_available():
    warnings.warn("Make sure to install wandb if you want to use it for logging during training.")
else: 
    import wandb
logger = get_logger(__name__, log_level="INFO")


def get_latest_checkpoint(checkpoint_dir):
    dirs = os.listdir(checkpoint_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None

    return path

def load_model_hook(models, input_dir):
    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()
        if isinstance(model, UNetSpatioTemporalConditionModel):
            load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
        elif isinstance(model, ControlNetModel):
            load_model = ControlNetModel.from_pretrained(input_dir, subfolder="control_net")
        else:
            raise Exception("Only UNetSpatioTemporalConditionModel and ControlNetModel are supported for loading.")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model


# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_models_accelerate(models, weights, output_dir, vae, feature_extractor, image_encoder, noise_scheduler):
    for i, model in enumerate(models):
        if isinstance(model, UNetSpatioTemporalConditionModel):
            model.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=False)
        elif isinstance(model, ControlNetModel):
            model.save_pretrained(os.path.join(output_dir, "control_net"), safe_serialization=False)
        else:
            raise Exception("Only UNetSpatioTemporalConditionModel and ControlNetModel are supported for saving.")

        # Also save other (frozen) components, just so they are found in the same checkpoint
        vae.save_pretrained(os.path.join(output_dir, "vae"), safe_serialization=False)
        feature_extractor.save_pretrained(os.path.join(output_dir, "feature_extractor"), safe_serialization=False)
        image_encoder.save_pretrained(os.path.join(output_dir, "image_encoder"), safe_serialization=False)
        noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"), safe_serialization=False)
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


class TrainVideoControlnet:

    def __init__(self, args):
        self.args = args


    def get_accelerator(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        accelerator_project_config = ProjectConfiguration(project_dir=self.args.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )

        self.accelerator = accelerator


    def log_setup(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Handle the repository creation
        if self.accelerator.is_main_process and self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)


    def load_models_from_pretrained(self):
        print("HARDCODING UNET VARIANT TO NONE, set to 'fp16' when training from stability's model")
        unet_variant = "fp16" if "stabilityai" in self.args.pretrained_model_name_or_path else None
        
        # Load scheduler, tokenizer and models.
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            variant=unet_variant,
            low_cpu_mem_usage=True,
            num_frames=self.args.clip_length
        )
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=self.args.revision,
            variant=unet_variant
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="image_encoder",
            revision=self.args.revision,
            variant=unet_variant
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="feature_extractor",
            revision=self.args.revision
        )
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="scheduler"
        )

        null_model_unet = UNetSpatioTemporalConditionModel.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            subfolder="unet",
            variant=unet_variant,
            low_cpu_mem_usage=True,
            num_frames=self.args.clip_length
        )
        
        return unet, vae, image_encoder, feature_extractor, noise_scheduler, null_model_unet

    

    def get_dataloaders(self):
        train_dataset, train_loader = get_dataloader(self.args.data_root, self.args.dataset_name, if_train=True, clip_length=self.args.clip_length,
                                                     batch_size=self.args.train_batch_size, num_workers=self.args.dataloader_num_workers, shuffle=True, 
                                                     image_height=self.args.train_H, image_width=self.args.train_W,
                                                     non_overlapping_clips=self.args.non_overlapping_clips,
                                                     bbox_masking_prob=self.args.bbox_masking_prob
                                                    )

        val_dataset, val_loader = get_dataloader(self.args.data_root, self.args.dataset_name, if_train=False, clip_length=self.args.clip_length,
                                                 batch_size=self.args.num_demo_samples, num_workers=self.args.dataloader_num_workers, shuffle=True, 
                                                 image_height=self.args.train_H, image_width=self.args.train_W,
                                                 non_overlapping_clips=True,
                                                )
        
        # demo_samples = get_samples(val_loader, self.args.num_demo_samples, show_progress=True)

        return train_dataset, train_loader, val_dataset, val_loader

    def get_sigmas(self, timesteps, n_dim=5, dtype=torch.float32):
        sigmas = self.noise_scheduler.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    def setup_training(self):
        self.get_accelerator()
        self.log_setup()

        self.last_save_time = 0
        if self.accelerator.is_main_process:
            self.last_save_time = time.time()

        # Setup devices
        accelerator_device = self.accelerator.device

        unet, vae, image_encoder, feature_extractor, noise_scheduler, null_model_unet = self.load_models_from_pretrained()

        # freeze parameters of models to save more memory
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        null_model_unet.requires_grad_(False)
        # Load the model
        assert self.args.train_H % 8 == 0 and self.args.train_W % 8 == 0
        self.bbox_embedding_shape = (4, self.args.train_H // 8, self.args.train_W // 8)

        ctrlnet = ControlNetModel.from_unet(unet, action_dim=5, bbox_embedding_shape=self.bbox_embedding_shape)
        ctrlnet.requires_grad_(True)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = self.train_weights_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator_device, dtype=self.weight_dtype)
        image_encoder.to(accelerator_device, dtype=self.weight_dtype)
        unet.to(accelerator_device, dtype=self.weight_dtype)
        null_model_unet.to(accelerator_device, dtype=self.weight_dtype)
        # `accelerate` 0.16.0 will have better support for customized saving
        assert version.parse(accelerate.__version__) >= version.parse("0.16.0")
        
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                save_models_accelerate(models, weights, output_dir, vae, feature_extractor, image_encoder, noise_scheduler)

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

        if self.args.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if self.args.scale_lr:
            self.args.learning_rate = (self.args.learning_rate * self.args.gradient_accumulation_steps * self.args.per_gpu_batch_size * self.accelerator.num_processes)

        optimizer = torch.optim.AdamW(
            list(ctrlnet.parameters()),  # Include all trainable parameters
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        train_dataset, train_loader, _, val_loader = self.get_dataloaders()

        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps * self.accelerator.num_processes,
        )
        
        # Prepare everything with our `accelerator`.
        unet, ctrlnet, optimizer, train_loader, lr_scheduler, null_model_unet = self.accelerator.prepare(
            unet, ctrlnet, optimizer, train_loader, lr_scheduler, null_model_unet
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            if not self.args.disable_wandb:
                self.accelerator.init_trackers(self.args.project_name, config=vars(self.args), init_kwargs={"wandb": {"dir": self.args.output_dir, "name": self.args.run_name, "entity": self.args.wandb_entity}})
            else:
                print("WANDB LOGGING DISABLED")
        
        self.unet = unet
        self.ctrlnet = ctrlnet
        self.vae = vae
        self.image_encoder = image_encoder
        self.feature_extractor = feature_extractor
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.train_dataset = train_dataset
        self.null_model_unet = null_model_unet
    
    def print_train_info(self):
        total_batch_size = self.args.train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        logger.info(f"  Number of processes = {self.accelerator.num_processes}")
    
    def load_checkpoint(self):
        self.first_epoch = 0
        self.global_step = 0

        if not self.args.resume_from_checkpoint:
            self.initial_global_step = 0
            return

        if self.args.resume_from_checkpoint != "latest":
            path = os.path.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            path = get_latest_checkpoint(self.args.output_dir)
        
        if path is None:
            self.accelerator.print(f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            self.args.resume_from_checkpoint = None
            self.initial_global_step = 0
        else:
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(os.path.join(self.args.output_dir, path))
            self.initial_global_step = self.global_step = int(path.split("-")[1])

            # self.first_epoch = global_step // num_update_steps_per_epoch # Not calculating first epoch right when using multiple processes. Let's default to using more epochs

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    def log_step(self, step_loss, train_loss):
        logs = {"step_loss": step_loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
        self.progress_bar.set_postfix(**logs)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            self.progress_bar.update(1)
            self.global_step += 1
            log_plot = {"train_loss": train_loss, "lr": self.lr_scheduler.get_last_lr()[0],}
            if self.args.add_bbox_frame_conditioning:
                log_plot["|attn_rz_weight|"] = self.unet.get_attention_rz_weight()
            self.accelerator.log(log_plot, step=self.global_step)
            train_loss = 0.0

    def save_checkpoint(self):
        
        # Checks if the accelerator has performed an optimization step behind the scenes (only checkpoint after the gradient accumulation steps)
        if not self.accelerator.sync_gradients:
            return
        
        args = self.args

        # Save if checkpointing step reached or job is about to expire
        save_checkpoint_time = args.checkpointing_time > 0 and (time.time() - self.last_save_time >= args.checkpointing_time)

        # Save if number of steps for checkpointing reached
        save_checkpoint_steps = self.global_step % args.checkpointing_steps == 0 or save_checkpoint_time

        if self.accelerator.is_main_process and (save_checkpoint_time or save_checkpoint_steps):
            
            if save_checkpoint_time:
                print("Saving checkpoint due to time. Time elapsed:", time.time() - self.last_save_time)
                self.last_save_time = time.time()
            
            # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
            if args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]

                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)

            save_path = os.path.join(args.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    def run_inference_with_pipeline(self, pipeline, demo_samples, log_dict):
        args = self.args

        for sample_i, sample in tqdm(enumerate(demo_samples), desc="Validation", total=args.num_demo_samples):
            action_type = sample['action_type'].unsqueeze(0)

            frames = pipeline(sample['image_init'] if not args.generate_bbox else sample['bbox_init'], 
                            cond_images=sample['bbox_images'].unsqueeze(0) if not args.generate_bbox else sample['gt_clip'].unsqueeze(0),
                            action_type=action_type,
                            height=self.train_dataset.resize_height, width=self.train_dataset.resize_width, 
                            decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                            num_inference_steps=args.num_inference_steps,
                            num_frames=args.clip_length,
                            control_condition_scale=args.conditioning_scale,
                            min_guidance_scale=args.min_guidance_scale,
                            max_guidance_scale=args.max_guidance_scale,
                            noise_aug_strength=args.noise_aug_strength,
                            generator=self.generator, output_type='pt').frames[0]
            #frames = F.interpolate(frames, (train_dataset.orig_H, train_dataset.orig_W)).detach().cpu().numpy()*255
            frames = frames.detach().cpu().numpy()*255
            frames = frames.astype(np.uint8)
            
            log_dict["generated_videos"].append(wandb.Video(frames, fps=args.fps))
            
            if sample.get('bbox_images_np') is not None:
                # Add action type text to ground truth bounding box frames
                bbox_frames = sample['bbox_images_np'].copy()
                action_id = action_type.item()
                action_name = {0: "Normal", 1: "Ego", 2: "Ego/Veh", 3: "Veh", 4: "Veh/Veh"}
                action_text = f"Action: {action_name[action_id]} ({action_id})"
                for i in range(bbox_frames.shape[0]):
                    # Convert numpy array to PIL Image
                    frame = Image.fromarray(bbox_frames[i].transpose(1, 2, 0))
                    draw = ImageDraw.Draw(frame)
                    
                    # Add text in top right corner
                    text_position = (frame.width - 10, 10)  # 10 pixels from top, 10 pixels from right
                    draw.text(text_position, action_text, fill=(255, 255, 255), anchor="ra")

                    # Add video name in top left corner
                    text_position = (10, 10)  # 10 pixels from top, 10 pixels from right
                    draw.text(text_position, sample['vid_name'], fill=(255, 255, 255), anchor="la")
                    
                    # Convert back to numpy array
                    bbox_frames[i] = np.array(frame).transpose(2, 0, 1)
                
                log_dict["gt_bbox_frames"].append(wandb.Video(bbox_frames, fps=args.fps))
            
            log_dict["gt_videos"].append(wandb.Video(sample['gt_clip_np'], fps=args.fps))
            # frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (train_dataset.orig_W, train_dataset.orig_H))
            # log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes

        return log_dict


    def load_pipeline(self):
        # NOTE: Pipeline used for inference at validation step, can change for different pipelines
        pipeline = StableVideoControlNullModelPipeline.from_pretrained(self.args.pretrained_model_name_or_path,
                                                        unet=self.unwrap_model(self.unet), 
                                                        image_encoder=self.unwrap_model(self.image_encoder),
                                                        vae=self.unwrap_model(self.vae),
                                                        controlnet=self.unwrap_model(self.ctrlnet),
                                                        null_model=self.unwrap_model(self.null_model_unet),
                                                        feature_extractor=self.feature_extractor,
                                                        revision=self.args.revision, 
                                                        variant=self.args.variant, 
                                                        torch_dtype=self.weight_dtype,
                                                        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        return pipeline

    def validation_step(self, save_pipeline=False):
        logger.info("Running validation... ")

        log_dict = defaultdict(list)
        with torch.autocast(str(self.accelerator.device).replace(":0", ""), enabled=self.accelerator.mixed_precision == "fp16"):
            
            pipeline = self.load_pipeline()
            if self.demo_samples is None:
                self.demo_samples = get_samples(self.val_loader, self.args.num_demo_samples, show_progress=True)

            self.ctrlnet.eval()
            log_dict = self.run_inference_with_pipeline(pipeline, self.demo_samples, log_dict)

        for tracker in self.accelerator.trackers:
            if tracker.name == "wandb":
                tracker.log(log_dict)

        if save_pipeline:
            pipeline.save_pretrained(self.args.output_dir)

        del pipeline, log_dict
        # torch.cuda.empty_cache()
        logger.info("Validation complete. ")
              
    def train_step(self, batch):
        # Aliases
        args = self.args
        accelerator = self.accelerator
        accelerator_device = self.accelerator.device
        ctrlnet, unet, vae, image_encoder, feature_extractor = self.ctrlnet, self.unet, self.vae, self.image_encoder, self.feature_extractor
        optimizer, noise_scheduler, lr_scheduler = self.optimizer, self.noise_scheduler, self.lr_scheduler
        weight_dtype = self.weight_dtype
        train_weights_dtype = self.train_weights_dtype

        train_loss = 0.0
        self.ctrlnet.train()
        with accelerator.accumulate(self.ctrlnet):
            # Forward pass
            batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]
            initial_images = batch['clips'][:,0,:,:,:] if not self.args.generate_bbox else batch['bbox_images'][:,0,:,:,:] # only use the first frame
            # check device
            if vae.device != accelerator_device:
                vae.to(accelerator_device)
                image_encoder.to(accelerator_device)
                initial_images.to(accelerator_device)
            
            # Encode input image
            encoder_hidden_states = encode_video_image(initial_images, feature_extractor, weight_dtype, image_encoder).unsqueeze(1)
            encoder_hidden_states = encoder_hidden_states.to(dtype=train_weights_dtype).to(accelerator_device)

            # Encode input image using VAE
            conditional_latents = vae.encode(initial_images.to(accelerator_device).to(weight_dtype)).latent_dist.sample()
            conditional_latents = conditional_latents.to(accelerator_device).to(train_weights_dtype)

            # Encode bbox image using VAE
            # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
            bbox_frames = rearrange(batch['bbox_images'] if not args.generate_bbox else batch['clips'], 'b f c h w -> (b f) c h w').to(vae.device).to(weight_dtype)
            
            # Get the selected option from the batch (This is the accident type flag)
            if self.args.use_action_conditioning:
                action_type = batch['action_type']  # Shape: [batch_size, 1]
            else:
                action_type = None

            # Encode using VAE (now with standard 3 channels)
            bbox_em = vae.encode(bbox_frames).latent_dist.sample()
            bbox_em = rearrange(bbox_em, '(b f) c h w -> b f c h w', f=video_length).to(accelerator_device).to(train_weights_dtype)

            # Encode clip frames using VAE
            # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
            frames = rearrange(batch['clips'] if not args.generate_bbox else batch['bbox_images'], 'b f c h w -> (b f) c h w').to(vae.device).to(weight_dtype)
            latents = vae.encode(frames).latent_dist.sample()
            latents = rearrange(latents, '(b f) c h w -> b f c h w', f=video_length).to(accelerator_device).to(train_weights_dtype)
            target_latents = latents = latents * vae.config.scaling_factor

            del batch, frames
            noise = torch.randn_like(latents)
            
            indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=noise_scheduler.timesteps.device).long()
            timesteps = noise_scheduler.timesteps[indices].to(accelerator_device)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Scale the noisy latents for the UNet
            sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
            # inp_noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps)
            inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

            added_time_ids = get_add_time_ids(
                fps=args.fps-1,
                motion_bucket_id=127,
                noise_aug_strength=args.noise_aug_strength,
                dtype=weight_dtype,
                batch_size=batch_size,
                unet=unet
            ).to(accelerator_device)
            
            # Conditioning dropout to support classifier-free guidance during inference. For more details
            # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
            # Addapted from https://github.com/huggingface/diffusers/blob/0d2d424fbef933e4b81bea20a660ee6fc8b75ab0/docs/source/en/training/instructpix2pix.md
            if args.conditioning_dropout_prob is not None:
                random_p = torch.rand(batch_size, device=accelerator_device, generator=self.generator)

                # Sample masks for the edit prompts.
                prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
                # Final text conditioning (initial image CLIP embedding)
                null_conditioning = torch.zeros_like(encoder_hidden_states)
                encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                # Sample masks for the original images.
                image_mask_dtype = conditional_latents.dtype
                image_mask = 1 - (
                    (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                    * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                )
                image_mask = image_mask.reshape(batch_size, 1, 1, 1)
                # Final image conditioning.
                conditional_latents = image_mask * conditional_latents
            
            # Bbox conditioning masking
            if args.contiguous_bbox_masking_prob is not None and args.contiguous_bbox_masking_prob > 0:
                
                random_p = torch.rand(batch_size, device=accelerator_device, generator=self.generator)
                if args.contiguous_bbox_masking_start_ratio is not None and args.contiguous_bbox_masking_start_ratio > 0:
                    # Among the masked samples, randomly select some to mask from the start of the video (and the rest from the end)
                    random_threshold = args.contiguous_bbox_masking_prob * args.contiguous_bbox_masking_start_ratio
                    sample_mask_start = (random_p < random_threshold).view(batch_size, 1, 1, 1, 1)
                    sample_mask_end = ((random_p >= random_threshold) & (random_p < args.contiguous_bbox_masking_prob)).view(batch_size, 1, 1, 1, 1)

                    min_bbox_mask_idx_start, max_bbox_mask_idx_start = 0, (video_length + 1) # TODO: Determine schedule (decrease min mask idx over time)
                    bbox_mask_idx_start = torch.randint(min_bbox_mask_idx_start, max_bbox_mask_idx_start, (batch_size,), device=accelerator_device)
                    mask_cond_start = bbox_mask_idx_start > torch.arange(args.clip_length, device=accelerator_device).view(1, args.clip_length, 1, 1, 1)
                    bbox_em = torch.where(mask_cond_start & sample_mask_start, self.unwrap_model(self.ctrlnet).bbox_null_embedding, bbox_em)

                else:
                    sample_mask_end = (random_p < args.contiguous_bbox_masking_prob).view(batch_size, 1, 1, 1, 1)
                    
                min_bbox_mask_idx, max_bbox_mask_idx = 0, (video_length + 1) # TODO: Determine schedule (decrease min mask idx over time)
                bbox_mask_idx = torch.randint(min_bbox_mask_idx, max_bbox_mask_idx, (batch_size,), device=accelerator_device)
                mask_cond = bbox_mask_idx <= torch.arange(args.clip_length, device=accelerator_device).view(1, args.clip_length, 1, 1, 1)
                bbox_em = torch.where(mask_cond & sample_mask_end, self.unwrap_model(self.ctrlnet).bbox_null_embedding, bbox_em)

            # Concatenate the `original_image_embeds` with the `noisy_latents`.
            # conditional_latents = unet.encode_bbox_frame(conditional_latents, None)
            conditional_latents = conditional_latents.unsqueeze(1).repeat(1, self.args.clip_length, 1, 1, 1)
            concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
            added_time_ids = added_time_ids.to(dtype=train_weights_dtype)
          
            down_block_additional_residuals, mid_block_additional_residuals = ctrlnet(
                concatenated_noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_time_ids=added_time_ids,
                control_cond=bbox_em,
                action_type=action_type, 
                conditioning_scale=args.conditioning_scale,
                return_dict=False,
            )

            if args.empty_cuda_cache:
                torch.cuda.empty_cache()

            model_pred = unet(sample=concatenated_noisy_latents,
                            timestep=timesteps,
                            encoder_hidden_states=encoder_hidden_states, 
                            added_time_ids=added_time_ids,
                            down_block_additional_residuals=down_block_additional_residuals,
                            mid_block_additional_residuals=mid_block_additional_residuals,).sample

            # Denoise the latents
            c_out = -sigmas / ((sigmas**2 + 1)**0.5)
            c_skip = 1 / (sigmas**2 + 1)
            denoised_latents = model_pred * c_out + c_skip * noisy_latents
            weighting = (1 + sigmas ** 2) * (sigmas**-2.0)

            # MSE loss
            step_loss = torch.mean((weighting.float() * (denoised_latents.float() - target_latents.float()) ** 2).reshape(target_latents.shape[0], -1), dim=1,)
            step_loss = step_loss.mean()

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(step_loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(step_loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        return step_loss, train_loss
    

    def train(self):
        """
        Main training loop
        """

        self.print_train_info()

        # Potentially load in the weights and states from a previous save
        self.load_checkpoint()

        args = self.args
        accelerator = self.accelerator

        self.progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=self.initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        self.generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
        self.demo_samples = None # Lazy load for samples

        test_valid = False
        for _ in range(self.first_epoch, args.num_train_epochs):

            for _, batch in enumerate(self.train_loader):
                
                # Check for for validation
                if accelerator.sync_gradients:
                    if test_valid or (self.global_step % args.validation_steps == 0 and self.global_step != 0) or (args.val_on_first_step and self.global_step == 0):
                        if accelerator.is_main_process:
                            self.validation_step()
                        accelerator.wait_for_everyone()
                        test_valid = False
            
                # Training step
                step_loss, train_loss = self.train_step(batch)

                # Log info
                self.log_step(step_loss, train_loss)

                # Potentially save checkpoint
                self.save_checkpoint()

                # if args.empty_cuda_cache:
                #     torch.cuda.empty_cache()

                # if global_step >= args.max_train_steps:
                #     break
        
        accelerator.wait_for_everyone()

        # Run a final round of inference
        if accelerator.is_main_process:
            logger.info("Running inference before terminating...")
            self.validation_step()

        logging.info("Finished training.")
        accelerator.end_training()
                
            
def main():
    args = parse_args()

    try:
        train_controlnet = TrainVideoControlnet(args)
        train_controlnet.setup_training()  # Load models, setup logging and define training config
        train_controlnet.train()  # Train!

    except KeyboardInterrupt:
        if hasattr(train_controlnet, "accelerator"):
            train_controlnet.accelerator.end_training()

        if is_wandb_available():
            wandb.finish()

        print("Keyboard interrupt: shutdown requested... Exiting.")
        exit()
    except Exception:
        import sys, traceback
        if is_wandb_available():
            wandb.finish()
        traceback.print_exc(file=sys.stdout)
        sys.exit(0)


if __name__ == '__main__':
    main()