import torch
from typing import Callable, Dict, List, Optional, Union
import PIL.Image
from einops import rearrange

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    tensor2vid,
    StableVideoDiffusionPipelineOutput,
    _append_dims,
    EXAMPLE_DOC_STRING
)
from diffusers import StableVideoDiffusionPipeline as StableVideoDiffusionPipeline_original
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

from src.models import UNetSpatioTemporalConditionModel, ControlNetModel
from diffusers.models import AutoencoderKLTemporalDecoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class StableVideoControlNullModelPipeline(StableVideoDiffusionPipeline_original):

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        controlnet: ControlNetModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        null_model: UNetSpatioTemporalConditionModel,
    ):
        # calling the super class constructors without calling StableVideoDiffusionPipeline_original's
        DiffusionPipeline.__init__(self)
    
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            controlnet=controlnet,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            null_model=null_model,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
    
    def check_inputs(self, image, cond_images, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )
        if not isinstance(cond_images, torch.Tensor):
            raise ValueError(
                "`cond_images` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(cond_images)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    
    def _encode_vae_condition(
        self,
        cond_image: torch.tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
        bbox_mask_frames: List[bool] = None
    ):
        video_length = cond_image.shape[1]
        cond_image = cond_image.to(device=device)
        cond_image = cond_image.to(dtype=self.vae.dtype)

        if cond_image.shape[2] == 3:
            cond_image = rearrange(cond_image, "b f c h w -> (b f) c h w")
            cond_em = self.vae.encode(cond_image).latent_dist.mode()
            cond_em = rearrange(cond_em, "(b f) c h w -> b f c h w", f=video_length)
        else:
            assert cond_image.shape[2] == 4, "The input tensor should have 3 or 4 channels. 3 for frames and 4 for latents."
            cond_em = cond_image

        # duplicate cond_em for each generation per prompt, using mps friendly method
        cond_em = cond_em.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        # Bbox conditioning masking during inference (requiring the model to predict behaviour instead)
        if bbox_mask_frames is not None:
            mask_cond = torch.tensor(bbox_mask_frames, device=cond_em.device).view(num_videos_per_prompt, video_length, 1, 1, 1)
            null_embedding = self.controlnet.bbox_null_embedding.repeat(num_videos_per_prompt, video_length, 1, 1, 1)
            cond_em = torch.where(mask_cond, null_embedding, cond_em)

        if do_classifier_free_guidance:
            # negative_cond_em = torch.zeros_like(cond_em)
            negative_cond_em = self.controlnet.bbox_null_embedding.repeat(num_videos_per_prompt, video_length, 1, 1, 1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            cond_em = torch.cat([negative_cond_em, cond_em])

        return cond_em
    
    @property
    def do_classifier_free_guidance(self):
        return False
        # if isinstance(self.guidance_scale, (int, float)):
        #     return self.guidance_scale > 1
        # return self.guidance_scale.max() > 1
    
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        cond_images: torch.FloatTensor = None,
        bbox_mask_frames: List[bool] = None,
        action_type: torch.FloatTensor = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        control_condition_scale: float=1.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image(s) to guide image generation. If you provide a tensor, the expected value range is between `[0,
                1]`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames` (14 for
                `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`).
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            action_type (`torch.FloatTensor`, *optional*, defaults to None):
                The action type to condition the generation. These features are used by the ControlNet
                to influence the generation process. The features should be of shape `[batch_size, 1]`.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage. By default, the decoder decodes all frames at once for maximal quality.
                For lower memory usage, reduce `decode_chunk_size`.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference. The function is called
                with the following arguments:
                    `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`.
                `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.FloatTensor`)
                is returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image, cond_images, height, width)

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        vae_device = self.vae.device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # 3. Encode input image
        image_embeddings = self._encode_image(image, device, num_videos_per_prompt, self.do_classifier_free_guidance)

        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        image = self.image_processor.preprocess(image, height=height, width=width).to(device)
        noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
        image = image + noise_aug_strength * noise

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        image_latents = self._encode_vae_image(
            image,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] -> [batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7a. Prepare latent variables
        num_channels_latents = self.unet.config.out_channels*2
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )

        # 7b. Prepare control latent embeds
        if not cond_images is None:
            cond_em = self._encode_vae_condition(cond_images,
                                                device, 
                                                num_videos_per_prompt, 
                                                self.do_classifier_free_guidance,
                                                bbox_mask_frames=bbox_mask_frames)
            cond_em = cond_em.to(image_embeddings.dtype)
        else:
            cond_em = None

        # 7c. Prepare action features
        if not action_type is None:
            if self.do_classifier_free_guidance:
                action_type = torch.cat([torch.zeros_like(action_type).unsqueeze(0), action_type.unsqueeze(0)])
        else:
            action_type = None

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # print(latent_model_input.shape, image_latents.shape, self.do_classifier_free_guidance)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
                # latent_model_input_null_model = latent_model_input.clone().detach()
                down_block_additional_residuals, mid_block_additional_residuals = self.controlnet(
                    latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    control_cond=cond_em,
                    action_type=action_type,
                    conditioning_scale=control_condition_scale,
                    return_dict=False,
                )

                # predict the noise residual
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residuals=mid_block_additional_residuals,
                    return_dict=False,
                )[0]

                # Predict unconditional noise
                noise_pred_uncond = self.null_model(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    _, noise_pred_cond = noise_pred.chunk(2) # NOTE: Currently discarding the unconditional noise prediction from the finetuned model
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                # print("latents", latents.shape)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)