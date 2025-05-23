import os
import cv2
import json
from tqdm import tqdm
import argparse

from yolo_sam import YoloSamProcessor


def create_video(sample, video_path):
    video_filename = f"{video_path}.mp4"
    FPS = 12
    frame_size = (1056, 660)#(512, 320)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer_out = cv2.VideoWriter(video_filename, fourcc, FPS, frame_size)

    for img in sample:
        video_writer_out.write(img)

    video_writer_out.release()
    print(f"Video saved: {video_filename}")


def downsample_and_crop_vid(video_path, output_dir, out_fps=12, crop_extents=None):
    """
    Downsample fps and crop frames
    """

    # Load video
    cap = cv2.VideoCapture(video_path)
    org_fps = int(cap.get(cv2.CAP_PROP_FPS))
    src_width, src_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_sample_name = video_path.split('/')[-1].split('.')[0]
    category = original_sample_name.split("_")[0]
    vid_num = '90'+str(int(original_sample_name.split("_")[-1])).zfill(3) # NOTE: We prepend a '90' to differentiate between DADA and CAP samples
    sample_name = f"{category}_{vid_num}"
    if crop_extents:
        src_width, src_height = (src_width + crop_extents[3]) - crop_extents[2], (src_height + crop_extents[1]) - crop_extents[0]

    print(f"Source video '{sample_name}': {src_width}x{src_height}, fps={org_fps}")

    total_frames = 0
    target_period = 1/out_fps
    last_frame_time = target_period
    out_frame_count = 0

    image_output_folder = os.path.join(output_dir, "images", category, sample_name)
    os.makedirs(image_output_folder, exist_ok=True)

    # sample_test = []
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        
        # Extract frames according to desired fps
        if last_frame_time >= target_period:
            out_frame_count += 1
            last_frame_time = (last_frame_time - target_period)

            # Crop frame
            if crop_extents:
                frame = frame[:, crop_extents[2] : crop_extents[3]]
            
            # Save frame
            out_image_name = f"{sample_name}_{str(total_frames).zfill(4)}.jpg"
            out_image_path = os.path.join(image_output_folder, out_image_name)
            cv2.imwrite(out_image_path, frame)

            # sample_test.append(frame)

        total_frames += 1
        last_frame_time += 1/org_fps

    print(f"Done '{sample_name}': {out_frame_count} frames, fps: {out_frame_count / (total_frames*1/org_fps)}")
    cap.release()
    # create_video(sample_test, "/home/mila/a/anthony.gosselin/dev/Ctrl-Merge/dev/sample_test_vid")


def extract_frames(dataset_dir, out_directory, crop_extents=None, out_fps=12):
    dataset_video_dir = os.path.join(dataset_dir)

    # NOTE: We are excluding all crashes that involve visible humans (pedestrians, cyclists, motorbikes...)
    video_types_to_exclude = [1, 2, 3, 4, 5, 6, 37, 38, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60] 

    for filename in tqdm(os.listdir(dataset_video_dir)):
        if filename.split('.')[-1] != "mp4":
            continue

        # Let's filter the videos we want right away
        vid_type = filename.split('_')[0]
        if int(vid_type) in video_types_to_exclude:
            continue

        video_path = os.path.join(dataset_video_dir, filename)
        downsample_and_crop_vid(video_path, out_directory, out_fps=out_fps, crop_extents=crop_extents)
    print("Extraction complete.")


def generate_labels(out_directory, vid_names=None, subdir=""):
    label_output_folder = os.path.join(out_directory, "labels", subdir)
    os.makedirs(label_output_folder, exist_ok=True)

    # Checkpoint paths
    yolo_ckpt = "yolov8x.pt" # Will auto download with utltralytics
    sam2_ckpt = "/network/scratch/x/xuolga/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
    yolo_sam = YoloSamProcessor(yolo_ckpt, sam2_ckpt, sam2_cfg)

    samples_run = 0

    video_dir_root = os.path.join(out_directory, "images", subdir)
    for category in sorted(os.listdir(video_dir_root), reverse=True):
        category_root = os.path.join(video_dir_root, category)
        for video_name in tqdm(os.listdir(category_root)):
            if vid_names is not None and video_name not in vid_names:
                continue

            video_dir = os.path.join(category_root, video_name)
            if len(os.listdir(video_dir)) == 0:
                print("Empty video dir:", video_dir)
                continue

            if len(os.listdir(video_dir)) > 300:
                print(f"SKIPPING LONG VIDEO {video_name}")
                continue
            
            # Skip if label file already exists
            out_label_path = os.path.join(label_output_folder, f"{video_name}.json")
            if os.path.exists(out_label_path):
                print(f"Skipping {video_name} - label file already exists")
                continue

            video_data = yolo_sam(video_dir, rel_bbox=True)

            if video_data is None:
                print("COMPUTED VIDEO DATA IS NULL for video:", video_dir)

            # Add metadata
            vid_type = int(video_name.split('_')[0])
            ego_involved = vid_type < 19 or vid_type == 61
            final_out_data = {
                "video_source": f"{video_name}.mp4",
                "metadata": {
                    "ego_involved": ego_involved,
                    "accident_type": vid_type
                    },
                "data": video_data
            }
            
            with open(out_label_path, 'w') as json_file:
                json.dump(final_out_data, json_file, indent=1)
            
            print("Saved label:", out_label_path)

            samples_run += 1
            if samples_run > 50:
                print("Resetting Yolo_Sam in case of memory leak")
                del yolo_sam
                yolo_sam = YoloSamProcessor(yolo_ckpt, sam2_ckpt, sam2_cfg)
                samples_run = 0


def make_train_val_split(out_directory):
    image_folder = os.path.join(out_directory, "images")
    label_folder = os.path.join(out_directory, "labels")

    all_image_folders = os.listdir(image_folder)
    split_idx = int(len(all_image_folders) * 0.9)

    train_split = all_image_folders[:split_idx]
    val_split = all_image_folders[split_idx:]

    os.makedirs(os.path.join(image_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(image_folder, "val"), exist_ok=True)
    os.makedirs(os.path.join(label_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(label_folder, "val"), exist_ok=True)

    for filename in train_split:
        os.rename(os.path.join(image_folder, filename), os.path.join(image_folder, "train", filename))
        os.rename(os.path.join(label_folder, f"{filename}.json"), os.path.join(label_folder, "train", f"{filename}.json"))

    for filename in val_split:
        os.rename(os.path.join(image_folder, filename), os.path.join(image_folder, "val", filename))
        os.rename(os.path.join(label_folder, f"{filename}.json"), os.path.join(label_folder, "val", f"{filename}.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process DADA2000 dataset')
    parser.add_argument('--dataset_root', type=str, default="/network/scratch/a/anthony.gosselin/Datasets",
                      help='Root directory for datasets')
    parser.add_argument('--dataset_dir', type=str, default="/network/scratch/a/anthony.gosselin/Datasets/DADA2000_full",
                      help='Directory containing the DADA2000 dataset')
    parser.add_argument('--out_directory', type=str, default=None,
                      help='Output directory (defaults to {dataset_root}/dada2000_images_12fps)')
    parser.add_argument('--skip_extraction', action='store_true',
                      help='Skip frame extraction step')
    parser.add_argument('--skip_labels', action='store_true',
                      help='Skip label generation step')
    parser.add_argument('--skip_split', action='store_true',
                      help='Skip train/val split step')
    parser.add_argument('--out_fps', type=int, default=12,
                      help='Output frames per second (default: 12)')
    args = parser.parse_args()

    # Set default output directory if not specified
    if args.out_directory is None:
        args.out_directory = os.path.join(args.dataset_root, "dada2000_images_12fps")

    # Extract frames from videos
    if not args.skip_extraction:
        dada_crop_extents = [0, -0, 264, -264]  # Custom crop for DADA2000 dataset (get ratio right)
        extract_frames(args.dataset_dir, args.out_directory, crop_extents=dada_crop_extents, out_fps=args.out_fps)

    # Create labels (run bbox detector)
    if not args.skip_labels:
        generate_labels(args.out_directory, vid_names=None)

    # Split into train and val sets
    if not args.skip_split:
        make_train_val_split(args.out_directory)
