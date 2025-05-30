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

from tqdm.auto import tqdm
from peft import LoraConfig

import torch
torch.cuda.empty_cache()
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from packaging import version
from diffusers.utils.import_utils import is_xformers_available

from diffusers import EulerDiscreteScheduler
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.training_utils import EMAModel

from src.utils import parse_args, encode_video_image, get_add_time_ids, get_samples, get_model_attr
from src.datasets.dataset_utils import get_dataloader
from src.models import UNetSpatioTemporalConditionModel
from src.pipelines import VideoDiffusionPipeline

if not is_wandb_available():
    warnings.warn("Make sure to install wandb if you want to use it for logging during training.")
else: 
    import wandb
logger = get_logger(__name__, log_level="INFO")


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    last_save_time = 0
    if accelerator.is_main_process:
        last_save_time = time.time()

    try:
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)
                plot_dir = os.path.join(args.output_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

        # Load scheduler, tokenizer and models.
        noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        # Load scheduler, tokenizer and models.
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16",
        )
        # in_channels = 12 if args.add_bbox_frame_conditioning else 8
        # in_channels = 8
        # if args.add_bbox_frame_conditioning:
        #     unet = UNetSpatioTemporalConditionModel_with_bbox_cond.from_pretrained(
        #     args.pretrained_model_name_or_path, subfolder="unet", variant="fp16", revision=args.non_ema_revision,
        #     low_cpu_mem_usage=True, in_channels=in_channels, ignore_mismatched_sizes=True
        # )
        # else:
        unet_variant = "fp16"
        # print("HARDCODING UNET VARIANT TO NONE, set to 'fp16' when training from stability's model")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", variant=unet_variant,
            low_cpu_mem_usage=True, num_frames=args.clip_length
        )

        feature_extractor = CLIPImageProcessor.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision,
        )

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision, variant="fp16"
        )

        # freeze parameters of models to save more memory
        vae.requires_grad_(False)
        image_encoder.requires_grad_(False)

        # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        vae.to(accelerator.device, dtype=weight_dtype)
        image_encoder.to(accelerator.device, dtype=weight_dtype)
        
        # Add LoRA adapter
        if args.enable_lora:
            unet.requires_grad_(False)
            unet_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            # Add adapter and make sure the trainable params are in float32.
            unet.add_adapter(unet_lora_config) # this line of code disable the unet add_embedding's gradient

        # # Create EMA for the unet.
        if args.use_ema:
            ema_unet = EMAModel(unet.parameters(), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)
        
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # `accelerate` 0.16.0 will have better support for customized saving
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                    for i, model in enumerate(models):
                        model.save_pretrained(os.path.join(output_dir, "unet"), safe_serialization=False)

                        # Also save other (frozen) components, just so they are found in the same checkpoint
                        # vae.save_pretrained(os.path.join(output_dir, "vae"), safe_serialization=False)
                        # feature_extractor.save_pretrained(os.path.join(output_dir, "feature_extractor"), safe_serialization=False)
                        # image_encoder.save_pretrained(os.path.join(output_dir, "image_encoder"), safe_serialization=False)
                        # noise_scheduler.save_pretrained(os.path.join(output_dir, "scheduler"), safe_serialization=False)

                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            
            def load_model_hook(models, input_dir):
                if args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                    ema_unet.load_state_dict(load_model.state_dict())
                    ema_unet.to(accelerator.device)
                    del load_model

                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    # if args.add_bbox_frame_conditioning:
                    #     load_model = UNetSpatioTemporalConditionModel_with_bbox_cond.from_pretrained(input_dir, subfolder="unet")
                    # else:
                    load_model = UNetSpatioTemporalConditionModel.from_pretrained(input_dir, subfolder="unet")
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)


        if args.enable_gradient_checkpointing:
            unet.enable_gradient_checkpointing()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )
        
        if_only_temporal_backprop = False
        if args.add_bbox_frame_conditioning:
            unet.enable_grad_bbox_frame_embedder()
        if args.enable_lora:
            pass
        elif args.backprop_temporal_blocks_start_iter == 0:
            parameters_list = unet.enable_grad(temporal_transformer_block=True, all=False)
            if_only_temporal_backprop = True
        else:
            parameters_list = unet.enable_grad(all=True)
        parameters_list = unet.get_parameters_with_grad()

        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)
                
        optimizer = torch.optim.AdamW(
            parameters_list, 
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
         
        train_dataset, train_loader = get_dataloader(args.data_root, args.dataset_name, if_train=True, clip_length=args.clip_length,
                                                     batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers, shuffle=True, 
                                                     image_height=args.train_H, image_width=args.train_W,
                                                     non_overlapping_clips=args.non_overlapping_clips,
                                                     bbox_masking_prob=args.bbox_masking_prob
                                                    )

        _, val_loader = get_dataloader(args.data_root, args.dataset_name, if_train=False, clip_length=args.clip_length,
                                                 batch_size=args.num_demo_samples, num_workers=args.dataloader_num_workers, shuffle=True, 
                                                 image_height=args.train_H, image_width=args.train_W,
                                                 non_overlapping_clips=True,
                                                )      
                            
        # _, test_loader = get_dataloader(args.dataset_name, if_train=True, 
        #                                 batch_size=1, num_workers=args.dataloader_num_workers, 
        #                                 data_type='clip', use_default_collate=True, tokenizer=None, shuffle=True)
        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_loader, lr_scheduler
        )
        if args.use_ema:
            ema_unet.to(accelerator.device)

        # unet, optimizer, train_loader, test_loader, lr_scheduler = accelerator.prepare(
        #     unet, optimizer, train_loader, test_loader, lr_scheduler
        # )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            if not args.disable_wandb:
                accelerator.init_trackers(args.project_name, config=vars(args), init_kwargs={"wandb": {"dir": args.output_dir, "name": args.run_name, "entity": args.wandb_entity}})
            else:
                print("WANDB LOGGING DISABLED")

        def get_sigmas(timesteps, n_dim=5, dtype=torch.float32):
            sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)

            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        
        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Number of processes = {accelerator.num_processes}")
        initial_global_step = global_step = 0
        first_epoch = 0

        # # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                initial_global_step = global_step = int(path.split("-")[1])

                # first_epoch = global_step // num_update_steps_per_epoch # Not calculating first epoch right when using multiple processes. Let's default to using more epochs
        else:
            initial_global_step = 0

        progress_bar = tqdm(
            range(0, args.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )

        def unwrap_model(model):
            model = accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        demo_samples = None

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        def run_inference_with_pipeline(pipeline, demo_samples, log_dict):

            for sample_i, sample in tqdm(enumerate(demo_samples), desc="Validation", total=args.num_demo_samples):
                if args.if_last_frame_trajectory:
                    sample_bbox = sample['bbox_images'][:args.clip_length]
                    sample_bbox[-1] = sample['bbox_images'][args.clip_length]
                frames = pipeline(sample['image_init'], 
                                height=train_dataset.resize_height, width=train_dataset.resize_width, 
                                # bbox_conditions=sample['objects_tensors'], original_size=(train_dataset.orig_W, train_dataset.orig_H),
                                bbox_images=sample['bbox_images'].unsqueeze(0) if args.predict_bbox else None,
                                decode_chunk_size=8, motion_bucket_id=127, fps=args.fps, 
                                num_inference_steps=args.num_inference_steps,
                                num_frames=args.clip_length,
                                min_guidance_scale=args.min_guidance_scale,
                                max_guidance_scale=args.max_guidance_scale,
                                noise_aug_strength=args.noise_aug_strength,
                                generator=generator, output_type='pt',
                                num_cond_bbox_frames=args.num_cond_bbox_frames).frames[0]

                #frames = F.interpolate(frames, (train_dataset.orig_H, train_dataset.orig_W)).detach().cpu().numpy()*255
                frames = frames.detach().cpu().numpy()*255
                frames = frames.astype(np.uint8)
                log_dict["generated_videos"].append(wandb.Video(frames, fps=args.fps))
                # if sample.get('bbox_images_np') is not None:
                #     log_dict["gt_bbox_frames"].append(wandb.Video(sample['bbox_images_np'], fps=args.fps))
                log_dict["gt_videos"].append(wandb.Video(sample['gt_clip_np'], fps=args.fps))
                # frame_bboxes = wandb_frames_with_bbox(frames, sample['objects_tensors'], (train_dataset.orig_W, train_dataset.orig_H))
                # log_dict["frames_with_bboxes_{}".format(sample_i)] = frame_bboxes
            return log_dict

        first_step = True
        for epoch in range(first_epoch, args.num_train_epochs):
            train_loss = 0.0
            for i, batch in enumerate(train_loader):

                unet_dtype = get_model_attr(unet, 'dtype') 
                
                ## update optimizer's parameters
                if args.backprop_temporal_blocks_start_iter >= 0:
                    if not if_only_temporal_backprop and global_step >= args.backprop_temporal_blocks_start_iter:
                        optimizer.param_groups.clear()
                        optimizer.state.clear()
                        parameters_list = unet.enable_grad(temporal_transformer_block=True, all=False)
                        parameters_list = unet.get_parameters_with_grad()
                        optimizer.add_param_group({"params": parameters_list})
                        if_only_temporal_backprop = True
                        logger.info("Start only backpropagating the temporal layers.")

                if accelerator.sync_gradients:
                    if global_step % args.validation_steps == 0 and not first_step:
                        
                        if accelerator.is_main_process:
                            logger.info("Running validation... ")

                            log_dict = defaultdict(list)
                            with torch.autocast(
                                    str(accelerator.device).replace(":0", ""), enabled=accelerator.mixed_precision == "fp16"
                                ):
                                # create pipeline
                                if args.use_ema:
                                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                                    ema_unet.store(unet.parameters())
                                    ema_unet.copy_to(unet.parameters())
                                pipeline = VideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                            unet=unwrap_model(unet), 
                                                                            revision=args.revision, 
                                                                            variant=args.variant, 
                                                                            torch_dtype=weight_dtype,
                                                                            feature_extractor=feature_extractor,
                                                                            image_encoder=unwrap_model(image_encoder),
                                                                            vae=unwrap_model(vae),)
                                unet.eval()
                                pipeline = pipeline.to(accelerator.device)
                                pipeline.set_progress_bar_config(disable=True)
                                if demo_samples is None:
                                    demo_samples = get_samples(val_loader, args.num_demo_samples, no_bboxes=True)
                                log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)

                            for tracker in accelerator.trackers:
                                if tracker.name == "wandb":
                                    tracker.log(log_dict)
                            if args.use_ema:
                                # Switch back to the original UNet parameters.
                                ema_unet.restore(unet.parameters())
                                
                            del pipeline, log_dict
                            torch.cuda.empty_cache()
                            logger.info("Validation complete. ")
                            # val_done_event.set()
                        else:
                            # We don't want the other threads to timeout during the time the main thread performs validation (they must wait for each other at optimizer.step() anyway)
                            # while not val_done_event.is_set():
                            #     # No-op poke every 10s
                            #     time.sleep(5)
                            pass

                        accelerator.wait_for_everyone() 
                        # val_done_event.clear()
                
                unet.train()
                with accelerator.accumulate(unet):
                    # Forward pass
                    batch_size, video_length = batch['clips'].shape[0], batch['clips'].shape[1]
                    initial_images = batch['clips'][:,0,:,:,:] # only use the first frame

                    # Encode input image
                    encoder_hidden_states = encode_video_image(initial_images, feature_extractor, weight_dtype, image_encoder).unsqueeze(1)
                    
                    # Encode clip frames using VAE
                    # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
                    frames = rearrange(batch['clips'] if not args.predict_bbox else batch['bbox_images'], "b f c h w -> (b f) c h w")
                    latents = vae.encode(frames.to(dtype=weight_dtype)).latent_dist.sample()
                    latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size)
                    initial_frame_latent = vae.encode(initial_images.to(weight_dtype)).latent_dist.sample()
                    if not args.predict_bbox:
                        # Encode input image using VAE
                        conditional_latents = initial_frame_latent.to(dtype=unet_dtype)
                    else:
                        if args.if_last_frame_trajectory:
                            conditional_latents = latents.to(dtype=unet_dtype)
                            last_conditional_latents = conditional_latents[:,-1,::]
                            latents = latents[:,:-1,::]
                            conditional_latents = conditional_latents[:,:-1,::]
                            conditional_latents[:,args.num_cond_bbox_frames:-1,::] = initial_frame_latent.unsqueeze(1).repeat(1, video_length-args.num_cond_bbox_frames-1, 1, 1, 1)
                            conditional_latents[:,-1,::] = last_conditional_latents
                        else:
                            # TODO: (?) The latents for the initial frame are repeated ~20 times in the conditioning?
                            conditional_latents = latents.to(dtype=unet_dtype)
                            conditional_latents[:,args.num_cond_bbox_frames:-1,::] = initial_frame_latent.unsqueeze(1).repeat(1, video_length-args.num_cond_bbox_frames-1, 1, 1, 1)

                    target_latents = latents = latents * vae.config.scaling_factor

                    del batch, frames
                    noise = torch.randn_like(latents)
                    
                    indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).to(noise_scheduler.timesteps).long() # NOTE: Double type conversion here??
                    timesteps = noise_scheduler.timesteps[indices].to(accelerator.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Scale the noisy latents for the UNet
                    sigmas = get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # inp_noisy_latents = noise_scheduler.scale_model_input(noisy_latents, timesteps)
                    inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                    added_time_ids = get_add_time_ids(
                        fps=args.fps-1, # NOTE: Why do we have a -1 here???
                        motion_bucket_id=127, # NOTE: This could be increased for more dynamic scenes? (higher number means more motion)
                        # NOTE: We set this really low (noise_aug_strength)... but it would seem to have more impact when using high motion bucket ids... can influence motion (https://www.reddit.com/r/StableDiffusion/comments/184s07o/xy_chart_for_svd_comparing_motion_bucket_id_and/). 
                            # According to huggingface docs: "noise_aug_strength: the amount of noise added to the conditioning image. The higher the values the less the video resembles the conditioning image. Increasing this value also increases the motion of the generated video."
                        noise_aug_strength=args.noise_aug_strength, 
                        dtype=weight_dtype,
                        batch_size=batch_size,
                        unet=unet
                    ).to(accelerator.device)
                    
                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    # Addapted from https://github.com/huggingface/diffusers/blob/0d2d424fbef933e4b81bea20a660ee6fc8b75ab0/docs/source/en/training/instructpix2pix.md
                    if args.conditioning_dropout_prob is not None:
                        random_p = torch.rand(
                            batch_size, device=accelerator.device, generator=generator)
                        # Sample masks for the edit prompts.
                        prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                        prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
                        # Final text conditioning.
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(
                            prompt_mask, null_conditioning, encoder_hidden_states)
                        # Sample masks for the original images.
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - (
                            (random_p >= args.conditioning_dropout_prob).to(
                                image_mask_dtype)
                            * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                        )
                        image_mask = image_mask.reshape(batch_size, 1, 1, 1)
                        # Final image conditioning.
                        conditional_latents = image_mask * conditional_latents

                    # Concatenate the `original_image_embeds` with the `noisy_latents`.
                    if not args.predict_bbox:
                        conditional_latents = conditional_latents.unsqueeze(1).repeat(1, args.clip_length, 1, 1, 1)
                    
                    concatenated_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
                    model_pred = unet(concatenated_noisy_latents,
                                    timestep=timesteps,
                                    encoder_hidden_states=encoder_hidden_states.to(dtype=unet_dtype), 
                                    added_time_ids=added_time_ids.to(dtype=unet_dtype)).sample

                    # Denoise the latents
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighting = (1 + sigmas ** 2) * (sigmas**-2.0)

                    # # MSE loss
                    loss = torch.mean(
                        (weighting.float() * (denoised_latents.float() - target_latents.float()) ** 2).reshape(target_latents.shape[0], -1),
                        dim=1,
                    )
                    loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                del loss, latents, concatenated_noisy_latents, model_pred, weighting, inp_noisy_latents, noisy_latents, timesteps, indices, sigmas, added_time_ids

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if args.use_ema:
                        ema_unet.step(get_model_attr(unet, 'parameters')())
                    progress_bar.update(1)
                    global_step += 1
                    # print(f"Batch: {i}, proc: {accelerator.process_index}, global step: {global_step}")
                    first_step = False
                    log_plot = {
                                "train_loss": train_loss, 
                                "lr": lr_scheduler.get_last_lr()[0],
                            }
                    if args.add_bbox_frame_conditioning:
                        log_plot["|attn_rz_weight|"] = get_model_attr(unet, 'get_attention_rz_weight')()
                    accelerator.log(log_plot, step=global_step)
                    train_loss = 0.0

                    # Save if checkpointing step reached or job is about to expire
                    save_checkpoint_time = args.checkpointing_time > 0 and (time.time() - last_save_time >= args.checkpointing_time)
                    save_checkpoint_steps = global_step % args.checkpointing_steps == 0 or save_checkpoint_time
                    if accelerator.is_main_process and (save_checkpoint_time or save_checkpoint_steps):
                        
                        if save_checkpoint_time:
                            print("Saving checkpoint due to time. Time elapsed:", time.time() - last_save_time)
                            last_save_time = time.time()

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                # if global_step >= args.max_train_steps:
                #     print("BREAK", global_step, ">=", args.max_train_steps)
                #     break
            
        print("Exited training loop")
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unwrap_model(unet)
            if args.use_ema:
                ema_unet.copy_to(unet.parameters())
            pipeline = VideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                unet=unet, 
                                                                revision=args.revision, 
                                                                variant=args.variant, 
                                                                torch_dtype=weight_dtype,
                                                                feature_extractor=feature_extractor,
                                                                image_encoder=unwrap_model(image_encoder),
                                                                vae=unwrap_model(vae),)
            pipeline.save_pretrained(args.output_dir)

            # Run a final round of inference
            logger.info("Running inference before terminating...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            log_dict = defaultdict(list)
            log_dict = run_inference_with_pipeline(pipeline, demo_samples, log_dict)
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log(log_dict)
        
        accelerator.end_training()

    except KeyboardInterrupt:
        accelerator.end_training()
        if is_wandb_available():
            wandb.finish()
        torch.cuda.empty_cache()
        print("Keboard interrupt: shutdown requested... Exiting.")
        exit()
    except Exception:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        if is_wandb_available():
            wandb.finish()
        torch.cuda.empty_cache()
        sys.exit(0)

if __name__ == '__main__':
    
    main()
    