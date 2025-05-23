import os
import cv2
import json
import numpy as np
from time import time
import glob
from tqdm import tqdm
import scenedetect as sd
import argparse

# Load existing JSON data if available
def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return []

def save_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def estimate_upsizing_factor(image_path):
    """Estimate how much an image was upsized before being resized to 720x1280"""
    
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Compute the 2D Fourier Transform
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # Center the low frequencies
    magnitude_spectrum = np.abs(fshift)
    
    # Compute high-frequency energy
    h, w = img.shape
    cx, cy = w // 2, h // 2  # Center of the image
    radius = min(cx, cy) // 4  # Define a region for high frequencies
    
    # Mask low frequencies (keep only high frequencies)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
    high_freq_energy = np.sum(magnitude_spectrum * (1 - mask))
    
    # Normalize energy by image size
    energy_score = high_freq_energy / (h * w)
    
    # Estimate how much the image was upscaled
    upsize_factor = 1 / (1 + energy_score)  # Inverse relation: lower energy → more upscaling
    
    return upsize_factor

def check_upsample(video_paths, output_root, use_cache=True):
    t = time()

    if use_cache:
        cache_file = f"{output_root}/upsample_scores.json"
        cached_data = load_json(cache_file)

    results = {}
    num_frames = 5
    for src_images in tqdm(video_paths, desc="Computing upscale"):
        vid_name = src_images.split('/')[-1]
        if use_cache and vid_name in cached_data:
            results[src_images] = cached_data[vid_name]
            continue

        all_images = sorted(glob.glob(f"{src_images}/*.jpg"))

        if len(all_images) < 5:
            continue

        frame_indices = np.linspace(0, len(all_images) - 1, num_frames).astype(int)

        vid_scores = []
        for frame_idx in frame_indices:
            image_path = all_images[frame_idx]

            upsize_factor = estimate_upsizing_factor(image_path)
            # print(image_dir, upsize_factor)
            vid_scores.append(upsize_factor)

        results[src_images] = np.median(vid_scores).item()

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    sorted_results = {k: v for k, v in sorted_results}

    if use_cache:
        sorted_vids_by_names = {k.split('/')[-1]: v for k, v in sorted_results.items()}
        save_json(cache_file, sorted_vids_by_names)

    # print(f"Done in {time()-t:.2f}s")
    return sorted_results

def detect_scenes(image_folder, threshold=27.0):
    """Detects scene changes in a folder of images using PySceneDetect."""
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder)) if f.lower().endswith(('.jpg', '.jpeg'))]
    detector = sd.detectors.ContentDetector(threshold=threshold)
    scene_list = []
    prev_frame = None
    frame_num = 0

    for image_idx in range(0, len(image_files), 2): # Skip frames to go faster
        image_file = image_files[image_idx]
        frame = cv2.imread(image_file)
        if frame is None:
            continue
        
        frame_num += 1
        if prev_frame is not None:
            if detector.process_frame(frame_num, frame):
                scene_list.append(frame_num)
        
        prev_frame = frame
    
    return scene_list

def scan_scene_changes(video_paths, output_root, use_cache=True):

    if use_cache:
        cache_file = f"{output_root}/scene_changes.json"
        cached_data = load_json(cache_file)

    all_scene_change_vids = []
    scene_changes_by_vid_name = {}
    for folder_path in tqdm(video_paths, desc="Detecting scene changes"):
        
        vid_name = folder_path.split('/')[-1]
        if use_cache and vid_name in cached_data:
            scene_changes = cached_data[vid_name]
        else:
            scene_changes = detect_scenes(folder_path)
        
        scene_changes_by_vid_name[vid_name] = scene_changes
            
        if len(scene_changes) > 0:
            # print(f"{folder_path.split('/')[-1]} scene changes:", scene_changes)
            all_scene_change_vids.append(folder_path)
            
    if use_cache:
        save_json(cache_file, scene_changes_by_vid_name)

    print("Scene change vids:", len(all_scene_change_vids))
    return all_scene_change_vids
            

def sort_tool(video_folders, rejected_file, highquality_file, start_idx=0):

    rejected_videos = load_json(rejected_file)
    highquality_videos = load_json(highquality_file)

    rejected_videos_count = 0
    for video_path in video_folders:
        video_name = video_path.split("/")[-1]
        if video_name in rejected_videos:
            rejected_videos_count += 1
    print(f"{rejected_videos_count}/{len(video_folders)} videos already rejected in this set")

    video_idx = start_idx
    frame_idx = 0
    fps = 12
    last_action_next = True

    while True:
        video_path = video_folders[video_idx]
        video_name = video_path.split("/")[-1]
        image_files = sorted([f for f in os.listdir(video_path) if f.endswith(".jpg")])

        if not image_files:
            print(f"No images found in {video_name}")
            if last_action_next:
                video_idx = (video_idx + 1) % len(video_folders)
            else:
                video_idx = (video_idx - 1) % len(video_folders)
            continue
        
        if video_name in rejected_videos or video_name in highquality_videos:
            print(f"{video_name} already filtered")
            if last_action_next:
                video_idx = (video_idx + 1) % len(video_folders)
            else:
                video_idx = (video_idx - 1) % len(video_folders)
            continue

        frame_idx = 0
        playing = True
        paused = False
        
        while playing:
            frame_path = os.path.join(video_path, image_files[frame_idx])
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Failed to load {frame_path}")
                continue
            
            display_text = f"Video: {video_name} ({video_idx}/{len(video_folders)}) | Frame: {frame_idx + 1}/{len(image_files)} | fps: {fps}"
            cv2.putText(frame, display_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Video Reviewer", frame)
            
            key = cv2.waitKey(int(1000 / fps))  # 12 FPS
            
            if key == ord('w'):  # Next frame
                frame_idx = min(len(image_files)-1, frame_idx + 1)
                paused = True
            elif key == ord('s'):  # Previous frame
                frame_idx = max(0, frame_idx - 1)
                paused = True
            elif key == ord('d'):  # Next video
                video_idx = (video_idx + 1) % len(video_folders)
                last_action_next = True
                break
            elif key == ord('a'):  # Previous video
                video_idx = (video_idx - 1) % len(video_folders)
                last_action_next = False
                break
            elif key == ord('r'):  # Reject video
                if video_name not in rejected_videos:
                    rejected_videos.append(video_name)
                    save_json(rejected_file, rejected_videos)
                print(f"Rejected: {video_name}")
                video_idx = (video_idx + 1) % len(video_folders)
                break
            elif key == ord('h'):  # Mark as high quality
                if video_name not in highquality_videos:
                    highquality_videos.append(video_name)
                    save_json(highquality_file, highquality_videos)
                print(f"High Quality: {video_name}")
                video_idx = (video_idx + 1) % len(video_folders)
                break
            elif key == ord('p'):  # Increase fps
                fps += 1
            elif key == ord('l'): # Lower fps
                fps = max(1, fps - 1)
            elif key == 27:  # ESC to exit
                playing = False
                break
            
            if not paused:
                frame_idx = (frame_idx + 1) % len(image_files)

        if key == 27:
            print(f"Last video: {video_name} ({video_idx})")
            break

    cv2.destroyAllWindows()

def collect_all_videos(data_dir, single_category=False):
    all_video_paths = []
    if single_category:
        all_video_paths = sorted(glob.glob(f"{data_dir}/*"))
    else:
        for category in sorted(os.listdir(data_dir)):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                all_video_paths.extend(sorted(glob.glob(f"{category_path}/*")))
    return all_video_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter and sort video dataset')
    parser.add_argument('--dataset_name', type=str, required=True,
                      help='Name of the dataset directory')
    parser.add_argument('--start_idx', type=int, default=0,
                      help='Starting index for video review')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Custom data directory path (defaults to ./{dataset_name}/images)')
    parser.add_argument('--output_root', type=str, default=None,
                      help='Custom output root directory (defaults to ./{dataset_name})')
    parser.add_argument('--disable_sort_by_upsample', action='store_true',
                      help='Disable sorting videos by upsampling factor')
    parser.add_argument('--disable_check_scene_changes', action='store_true',
                      help='Disable checking for scene changes in videos')
    parser.add_argument('--single_category', action='store_true',
                      help='Process videos from a single category directory')
    parser.add_argument('--use_cache', action='store_true',
                      help='Use cache to speed up processing')
    args = parser.parse_args()
    
    # Set default paths if not specified
    if args.data_dir is None:
        args.data_dir = f"./{args.dataset_name}/images"
    if args.output_root is None:
        args.output_root = f"./{args.dataset_name}"
    
    # Output JSON files
    rejected_file = f"{args.output_root}/rejected.json"
    auto_low_quality = f"{args.output_root}/auto_low_quality.json"
    highquality_file = f"{args.output_root}/highquality.json"
    
    all_video_paths = collect_all_videos(args.data_dir, args.single_category)
    
    if not args.disable_sort_by_upsample:
        sorted_vids = check_upsample(all_video_paths, args.output_root, use_cache=args.use_cache)
        video_folders = list(sorted_vids.keys())

        # Save the worst to file
        # auto_reject_vids = [v.split('/')[-1] for v in video_folders[:2000]]
        # save_json(auto_low_quality, auto_reject_vids)
    else:
        video_folders = all_video_paths
    
    # Prepend scene change samples
    if not args.disable_check_scene_changes:
        new_video_folders = []
        scene_change_vids = scan_scene_changes(all_video_paths, args.output_root, use_cache=args.use_cache)
        new_video_folders.extend(scene_change_vids)
        for vid_name in video_folders:
            if vid_name not in new_video_folders:
                new_video_folders.append(vid_name)
        video_folders = new_video_folders
    
    # Start tool
    sort_tool(video_folders, rejected_file, highquality_file, start_idx=args.start_idx)