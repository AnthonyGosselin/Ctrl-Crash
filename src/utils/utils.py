import torch
from torchvision import transforms
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing

import numpy as np
from tqdm import tqdm


def encode_video_image(pixel_values, feature_extractor, weight_dtype, image_encoder):

    def resize_video_image(pixel_values):        
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        # # unnormalize
        pixel_values = (pixel_values+1.0)*0.5
        pixel_values = torch.clamp(pixel_values, min=0., max=1.)
        return pixel_values

    pixel_values = resize_video_image(pixel_values=pixel_values)
    
    # Normalize the image with for CLIP input
    pixel_values = feature_extractor(
        images=pixel_values,
        do_normalize=True,
        do_center_crop=False,
        do_resize=False,
        do_rescale=False,
        return_tensors="pt",
    ).pixel_values

    pixel_values = pixel_values.to(device=image_encoder.device).to(dtype=weight_dtype)
    image_embeddings = image_encoder(pixel_values).image_embeds
    return image_embeddings


def get_model_attr(model, attribute):
    """
    Get an attribute or call a method of the model.
    
    Parameters:
    - model: The model instance (potentially wrapped in DDP or DP).
    - attribute: The name of the attribute or method to access or call.
    
    Returns:
    - The attribute value
    """
    if hasattr(model, 'module'):
        model = model.module

    attr = getattr(model, attribute)
    
    return attr

def get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        unet
    ):
    """"https://github.com/huggingface/diffusers/blob/56bd7e67c2e01122cc93d98f5bd114f9312a5cce/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L215"""
    add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

    unet_config = get_model_attr(unet, 'config')
    passed_add_embed_dim = unet_config.addition_time_embed_dim * len(add_time_ids)
    
    expected_add_embed_dim = get_model_attr(unet, 'add_embedding').linear_1.in_features

    if expected_add_embed_dim != passed_add_embed_dim:
        raise ValueError(
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
        )

    add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
    add_time_ids = add_time_ids.repeat(batch_size, 1)
    return add_time_ids
 

def get_samples(data_loader, n_samples, show_progress=False, no_bboxes=False):
    samples = []

    dataset = data_loader.dataset
    for batch_idx, batch in tqdm(enumerate(data_loader), total=n_samples, disable=not show_progress, desc="Fetching samples"):
        for sample_idx in range(batch["indices"].shape[0]):
            if len(samples) >= n_samples:
                return samples
            
            gt_clip = batch["clips"][sample_idx,::].detach().cpu()
            gt_clip_np = (dataset.revert_transform(gt_clip).numpy() * 255).astype(np.uint8)
            image_init = transforms.ToPILImage()(dataset.revert_transform(gt_clip[0])).resize((dataset.resize_width, dataset.resize_height))

            if not no_bboxes:
                bbox_images = batch["bbox_images"][sample_idx, ::].detach().cpu()
                bbox_images_np = (dataset.revert_transform(bbox_images).numpy() * 255).astype(np.uint8)
                bbox_init = transforms.ToPILImage()(dataset.revert_transform(bbox_images[0])).resize((dataset.resize_width, dataset.resize_height))

                # object = dataset.__getitem__(sample_idx)
                
                sample = dict(
                        gt_clip             =           gt_clip,
                        gt_clip_np          =           gt_clip_np,
                        image_init          =           image_init,
                        image_paths         =           batch["image_paths"],

                        bbox_images         =           bbox_images,
                        bbox_images_np      =           bbox_images_np,
                        bbox_init           =           bbox_init,

                        action_type         =           batch["action_type"][sample_idx],
                        vid_name            =           batch["vid_name"][sample_idx]
                    )
            else:

                sample = dict(
                        gt_clip             =           gt_clip,
                        gt_clip_np          =           gt_clip_np,
                        image_init          =           image_init,
                        image_paths         =           batch["image_paths"],
                    )
                
            samples.append(sample)
        
    return samples


def rescale_bbox(bbox, image_size=(1242, 375), target_size=(1, 1)):
    """Rescales bounding boxes to the target size."""

    orig_shape = bbox.shape
    bbox = bbox.clone().reshape(-1, 4)
    bbox[:, 0] = bbox[:, 0] * target_size[0] / image_size[0]
    bbox[:, 1] = bbox[:, 1] * target_size[1] / image_size[1]
    bbox[:, 2] = bbox[:, 2] * target_size[0] / image_size[0]
    bbox[:, 3] = bbox[:, 3] * target_size[1] / image_size[1]

    return bbox.reshape(*orig_shape)