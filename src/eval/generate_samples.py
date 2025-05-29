import os
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import json
import argparse

import warnings
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.utils.checkpoint
from accelerate.utils import set_seed

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from src.pipelines import StableVideoControlPipeline
    from src.pipelines import StableVideoControlNullModelPipeline
    from src.pipelines import StableVideoControlFactorGuidancePipeline

from src.models import UNetSpatioTemporalConditionModel, ControlNetModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers.models import AutoencoderKLTemporalDecoder

from src.datasets.dataset_utils import get_dataloader
from src.utils import get_samples

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

generator = None #torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
CLIP_LENGTH = 25


def create_video_from_np(sample, video_path, fps=6):
    video_filename = f"{video_path}.mp4"
    frame_size = (512, 320)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer_out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

    for img in sample:
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer_out.write(img)

    video_writer_out.release()
    print(f"Video saved: {video_filename}")


def export_to_video(video_frames, output_video_path=None, fps=6):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path


def label_frames_with_action_id(bbox_frames, action_id, masked_idx=None):
    action_name = {0: "Normal", 1: "Ego", 2: "Ego/Veh", 3: "Veh", 4: "Veh/Veh"}
    action_text = f"Action: {action_name[action_id]} ({action_id})"
    for i in range(bbox_frames.shape[0]):
        # Convert numpy array to PIL Image
        frame = Image.fromarray(bbox_frames[i].transpose(1, 2, 0))
        draw = ImageDraw.Draw(frame)
        
        # Add text in top right corner
        text_position = (frame.width - 10, 10)  # 10 pixels from top, 10 pixels from right
        if masked_idx is not None and masked_idx <= i:
            text_color = (0, 0, 0) 
            action_text = f"Action: {action_name[action_id]} ({action_id}) [masked]"
        else:
            text_color = (255, 255, 255)
            action_text = action_text

        draw.text(text_position, action_text, fill=text_color, anchor="ra")
        
        # Convert back to numpy array
        bbox_frames[i] = np.array(frame).transpose(2, 0, 1)

    return bbox_frames

def load_ctrlv_pipelines(model_dir, use_null_model=False, use_factor_guidance=False):
    unet_variant = "fp16" if "stabilityai" in model_dir else None

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        model_dir,
        subfolder="unet",
        variant=unet_variant,
        low_cpu_mem_usage=True,
        num_frames=CLIP_LENGTH
    )    
    ctrlnet = ControlNetModel.from_pretrained(
        model_dir, 
        subfolder="control_net", 
        variant=unet_variant, 
        num_frames=25
    )

    if not use_null_model and not use_factor_guidance:
        pipeline = StableVideoControlPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", 
            controlnet=ctrlnet, 
            unet=unet, 
            variant=unet_variant
        )
    
    else:   
        
        # For null model prediction of uncond noise
        null_model_path = "stabilityai/stable-video-diffusion-img2vid-xt"
        null_model_unet = UNetSpatioTemporalConditionModel.from_pretrained(
            null_model_path,
            subfolder="unet",
            variant=unet_variant,
            low_cpu_mem_usage=True,
            num_frames=CLIP_LENGTH
        )

        if use_null_model and not use_factor_guidance:
            pipeline = StableVideoControlNullModelPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                controlnet=ctrlnet, 
                unet=unet,
                null_model=null_model_unet,
                variant=unet_variant
            )
        elif use_factor_guidance:
            pipeline = StableVideoControlFactorGuidancePipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                controlnet=ctrlnet, 
                unet=unet,
                null_model=null_model_unet,
                variant=unet_variant
            )
    
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    unet.eval()
    ctrlnet.eval()

    return pipeline


def generate_video_ctrlv(sample, pipeline, video_path="video_out/genvid", json_path="video_out/gt_frames", bbox_mask_frames=None, action_type=None, use_factor_guidance=False, guidance=[1.0, 3.0], video_path2=None):
    frame_size = (512, 320)
    FPS = 6
    CLIP_LENGTH = sample['bbox_images'].shape[0]
    
    init_image = sample['image_init']
    bbox_images = sample['bbox_images'].unsqueeze(0)
    action_type = sample['action_type'].unsqueeze(0) if action_type is None else action_type

    sample['bbox_images'].to(device)

    # Save GT frame paths to json file
    gt_frame_paths = [file_path[0] for file_path in sample['image_paths']]
    with open(json_path, "w") as file:
        json.dump(gt_frame_paths, file, indent=1)
    print("Saved GT frames json file:", json_path)

    if not use_factor_guidance:
        frames = pipeline(init_image, 
                        cond_images=bbox_images,
                        bbox_mask_frames=bbox_mask_frames,
                        action_type=action_type,
                        height=frame_size[1], width=frame_size[0], 
                        decode_chunk_size=8, motion_bucket_id=127, fps=FPS, 
                        num_inference_steps=30,
                        num_frames=CLIP_LENGTH,
                        control_condition_scale=1.0,
                        min_guidance_scale=guidance[0],
                        max_guidance_scale=guidance[1],
                        noise_aug_strength=0.01,
                        generator=generator, output_type='pt').frames[0]
    else:
        frames = pipeline(init_image, 
                        cond_images=bbox_images,
                        bbox_mask_frames=bbox_mask_frames,
                        action_type=action_type,
                        height=frame_size[1], width=frame_size[0], 
                        decode_chunk_size=8, motion_bucket_id=127, fps=FPS, 
                        num_inference_steps=30,
                        num_frames=CLIP_LENGTH,
                        control_condition_scale=1.0,
                        min_guidance_scale_img=1.0,
                        max_guidance_scale_img=3.0,
                        min_guidance_scale_action=6.0,
                        max_guidance_scale_action=12.0,
                        min_guidance_scale_bbox=1.0,
                        max_guidance_scale_bbox=3.0,
                        noise_aug_strength=0.01,
                        generator=generator, output_type='pt').frames[0]
    
    frames = frames.detach().cpu().numpy()*255
    frames = frames.astype(np.uint8)

    tmp = np.moveaxis(np.transpose(frames, (0, 2, 3, 1)), 0, 0)
    output_video_path = f"{video_path}.mp4"
    export_to_video(tmp, output_video_path, fps=FPS)
    print(f"Video saved:", output_video_path)

    if video_path2 is not None:
        output_video_path2 = f"{video_path2}.mp4"
        export_to_video(tmp, output_video_path2, fps=FPS)


def generate_samples(args):
    model_path = args.model_path
    print("Model path:", model_path)

    if args.seed is not None:
        set_seed(args.seed)
        print("Set seed:", args.seed)

    # LOAD PIPELINE
    use_null_model = not args.disable_null_model
    use_factor_guidance = args.use_factor_guidance
    pipeline = load_ctrlv_pipelines(model_path, use_null_model=use_null_model, use_factor_guidance=use_factor_guidance)

    # LOAD DATASET
    data_root = args.data_root
    dataset_name = args.dataset
    train_set = False
    val_dataset, val_loader = get_dataloader(
                                            data_root, dataset_name, if_train=train_set, clip_length=CLIP_LENGTH,
                                            batch_size=1, num_workers=0, shuffle=True, 
                                            image_height=320, image_width=512,
                                            non_overlapping_clips=True, #specific_samples=specific_samples
                                            )
    if train_set:
        print("WARNING: Currently using training split")

    # COLLECT SAMPLES
    num_demo_samples = args.num_demo_samples
    demo_samples = get_samples(val_loader, num_demo_samples, show_progress=True) 

    sample_range = range(0, num_demo_samples)
    num_samples = len(sample_range)

    # video_dir_path = os.path.join(os.getcwd(), "video_out", "video_out_box2video_may1_eval_test")
    video_dir_path = args.output_path
    os.makedirs(video_dir_path, exist_ok=True)
    video_counter = 0

    # GENERATION PARAMETERS

    # Set the bbox masking
    bbox_mask_idx_batch = args.bbox_mask_idx_batch
    condition_on_last_bbox = False

    # Set the action type
    force_action_type = None #1 # 0: Normal, 1: Ego, 2: Ego/Veh, 3: Veh, 4: Veh/Veh
    force_action_type_batch = args.force_action_type_batch 

    num_gens_per_sample = args.num_gens_per_sample
    guidance_scales = args.guidance_scales
    eval_output = args.eval_output

    # GENERATE VIDEOS

    # Check for samples that were already done and do not compute them again
    skip_samples = {}
    out_video_path = f"{video_dir_path}/gt_ref"
    if os.path.exists(out_video_path):
        all_videos = os.listdir(out_video_path)
        video_counter = len(all_videos)
        for sample_name in all_videos:
            vid_name = "_".join(sample_name.split("_")[1:])
            skip_samples[vid_name] = True

    print("SKIP SAMPLES:", skip_samples)

    for guidance in guidance_scales or [-1]:

        if guidance != -1:
            print("Guidance:", force_action_type)
        else:
            guidance = [1, 3]

        for _ in range(num_gens_per_sample):
            for force_action_type in force_action_type_batch or [-1]:

                if force_action_type != -1:
                    print("Force action type:", force_action_type)
                else:
                    force_action_type = None

                for bbox_mask_idx in bbox_mask_idx_batch or [-1]:

                    if bbox_mask_idx != -1:
                        print("Bbox masking:", bbox_mask_idx)
                    else:
                        bbox_mask_idx = None
                    
                    for i, sample in tqdm(enumerate(demo_samples), desc="Generating samples", total=num_samples):
                        if i >= list(sample_range)[-1] + 1:
                            break
                        if i not in sample_range:
                            continue

                        if video_counter > args.max_output_vids:
                            print(f"MAX OUTPUT VIDS REACHED: {video_counter} >= {args.max_output_vids}")
                            exit()
                        
                        vid_name = sample["vid_name"]

                        mask_hint = "" if bbox_mask_idx is None else f"_bframes:{str(bbox_mask_idx)}"
                        action_hint = "" if force_action_type is None else f"_action:{str(force_action_type)}"
                        guidance_hint = "" if guidance_scales is None else f"_guide{guidance[0]}:{guidance[1]}"
                        scene_name = f"{video_counter}_{vid_name}{mask_hint}{action_hint}{guidance_hint}"

                        scene_name_no_counter =  "_".join(scene_name.split("_")[1:])
                        if scene_name_no_counter in skip_samples:
                            print(f"Skipping sample that was already computed: {vid_name}")
                            continue

                        print("Generating video for:", scene_name)

                        if eval_output:
                            os.makedirs(f"{video_dir_path}/gen_videos", exist_ok=True)
                            os.makedirs(f"{video_dir_path}/gt_frames", exist_ok=True)
                            os.makedirs(f"{video_dir_path}/gt_ref", exist_ok=True)

                            gt_vid_path = f"{video_dir_path}/gt_ref/{scene_name}/(1)gt_video_{scene_name}"
                            bbox_out_path_root = f"{video_dir_path}/gt_ref/{scene_name}"
                            out_video_path = f"{video_dir_path}/gen_videos/genvid_{video_counter}_{vid_name}"
                            out_json_path = os.path.join(video_dir_path, "gt_frames", f"gt_frames_{video_counter}_{vid_name}.json")

                            out_video_path2 = f"{bbox_out_path_root}/(3)genvid_adv_{scene_name}"

                            os.makedirs(bbox_out_path_root, exist_ok=True)
                        else:
                            os.makedirs(f"{video_dir_path}/{scene_name}", exist_ok=True)

                            gt_vid_path = f"{video_dir_path}/{scene_name}/(1)gt_video_{scene_name}"
                            bbox_out_path_root = f"{video_dir_path}/{scene_name}"
                            out_video_path = f"{video_dir_path}/{scene_name}/(3)genvid_adv_{scene_name}"
                            out_json_path = os.path.join(video_dir_path, scene_name, f"gt_frames_{sample['vid_name']}.json")
                            out_video_path2 = None

                        create_video_from_np(sample['gt_clip_np'], video_path=gt_vid_path)

                        # Add action type text to ground truth bounding box frames # TODO: Make sure the action type aligns if we change it for generation
                        action_type = sample['action_type'].unsqueeze(0)
                        og_action_type = action_type.item()
                        if force_action_type is not None:
                            action_type = torch.ones_like(action_type) * force_action_type

                        action_id = action_type.item()
                        bbox_frames = sample['bbox_images_np'].copy()
                        if bbox_mask_idx is not None:
                            # print(f"Masking bboxes after index {bbox_mask_idx}")

                            # Let's save a copy of the original bboxes for reference
                            bbox_frames_ref = sample['bbox_images_np'].copy()
                            label_frames_with_action_id(bbox_frames_ref, og_action_type)
                            create_video_from_np(bbox_frames_ref, video_path=f"{bbox_out_path_root}/(2)video_2dbboxes_{scene_name}_nomask")

                            # For display, let's mask with white
                            mask_cond = bbox_mask_idx <= np.arange(CLIP_LENGTH).reshape(CLIP_LENGTH, 1, 1, 1)
                            if condition_on_last_bbox:
                                mask_cond[-1, 0, 0, 0] = False
                            bbox_frames = np.where(mask_cond, np.ones_like(bbox_frames)*255, bbox_frames)
                            label_frames_with_action_id(bbox_frames, action_id, masked_idx=bbox_mask_idx)
                        else:
                            label_frames_with_action_id(bbox_frames, action_id)
                        
                        create_video_from_np(bbox_frames, video_path=f"{bbox_out_path_root}/(2)video_2dbboxes_{scene_name}")
                        
                        bbox_mask_frames = [False] * CLIP_LENGTH
                        if bbox_mask_idx is not None:
                            bbox_mask_frames[bbox_mask_idx:] = [True] * (len(bbox_mask_frames) - bbox_mask_idx)
                        if condition_on_last_bbox:
                            bbox_mask_frames[-1] = False
                            
                        generate_video_ctrlv(
                            sample, 
                            pipeline, 
                            video_path=out_video_path,
                            json_path=out_json_path,
                            bbox_mask_frames=bbox_mask_frames,
                            action_type=action_type,
                            use_factor_guidance=use_factor_guidance,
                            guidance=guidance,
                            video_path2=out_video_path2
                        )

                        video_counter += 1

    print("DONE")

