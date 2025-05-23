import os
import cv2
import json
from tqdm import tqdm
import argparse

from yolo_sam import YoloSamProcessor

def downsample_and_crop_vid(video_path, output_dir, out_fps=7, crop_extents=None):
    """
    Downsample fps and crop frames
    """

    # Load video
    cap = cv2.VideoCapture(video_path)
    org_fps = int(cap.get(cv2.CAP_PROP_FPS))
    src_width, src_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_name = video_path.split('/')[-1].split('.')[0]
    if crop_extents:
        src_width, src_height = (src_width + crop_extents[3]) - crop_extents[2], (src_height + crop_extents[1]) - crop_extents[0]

    print(f"Source video '{sample_name}': {src_width}x{src_height}, fps={org_fps}")

    total_frames = 0
    target_period = 1/out_fps
    last_frame_time = target_period
    out_frame_count = 0

    image_output_folder = os.path.join(output_dir, "images", sample_name)
    os.makedirs(image_output_folder, exist_ok=True)

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
                frame = frame[crop_extents[0] : crop_extents[1], crop_extents[2] : crop_extents[3]]
            
            # Save frame
            out_image_name = f"{sample_name}_{str(total_frames).zfill(4)}.jpg"
            out_image_path = os.path.join(image_output_folder, out_image_name)
            cv2.imwrite(out_image_path, frame)

        total_frames += 1
        last_frame_time += 1/org_fps

    print(f"Done '{sample_name}': {out_frame_count} frames, fps: {out_frame_count / (total_frames*1/org_fps)}")
    cap.release()

def extract_frames(dataset_dir, out_directory, crop_extents=None):
    dataset_video_dir = os.path.join(dataset_dir, "video")
    for filename in tqdm(os.listdir(dataset_video_dir)):
        video_path = os.path.join(dataset_video_dir, filename)
        fps = 7
        downsample_and_crop_vid(video_path, out_directory, out_fps=fps, crop_extents=crop_extents)
    print("Extraction complete.")


def generate_labels(dataset_dir, out_directory, video_subdir=''):
    label_output_folder = os.path.join(out_directory, "labels", video_subdir)
    os.makedirs(label_output_folder, exist_ok=True)

    # Checkpoint paths
    yolo_ckpt = "yolov8x.pt" # Will auto download with utltralytics
    sam2_ckpt = "/network/scratch/x/xuolga/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
    yolo_sam = YoloSamProcessor(yolo_ckpt, sam2_ckpt, sam2_cfg)

    video_dir_root = os.path.join(out_directory, "images", video_subdir)
    # for video_name in tqdm(os.listdir(video_dir_root)):
    for video_name in tqdm(["w10_138", "w10_94", "w1_10", "w1_46", "w2_79", "w3_17", "w6_14", "w6_44", "w6_78", "w6_94", "w7_1", "w7_14"]):
        video_dir = os.path.join(video_dir_root, video_name)
        if len(os.listdir(video_dir)) == 0:
            print("Empty video dir:", video_dir)
            continue

        video_data = yolo_sam(video_dir, rel_bbox=True)

        # Add metadata
        org_dataset_labels = os.path.join(dataset_dir, "label", "json")
        orig_label_path = os.path.join(org_dataset_labels, f"{video_name}.json")
        with open(orig_label_path, 'r') as json_file:
            metadata = json.load(json_file)[0]['meta_data']
        
        final_out_data = {
            "video_source": f"{video_name}.mp4",
            "metadata": metadata,
            "data": video_data
        }
        
        out_label_path = os.path.join(label_output_folder, f"{video_name}.json")
        with open(out_label_path, 'w') as json_file:
            json.dump(final_out_data, json_file, indent=1)
        
        print("Saved label:", out_label_path)


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
    parser = argparse.ArgumentParser(description='Process Russia Car Crash dataset')
    parser.add_argument('--dataset_root', type=str, default="/network/scratch/a/anthony.gosselin/Datasets",
                      help='Root directory for datasets')
    parser.add_argument('--dataset_dir', type=str, default="/network/scratch/a/anthony.gosselin/Datasets/CarCrashRussia/dataset",
                      help='Directory containing the Russia Car Crash dataset')
    parser.add_argument('--out_directory', type=str, default=None,
                      help='Output directory (defaults to {dataset_root}/preprocess_russia_crash)')
    parser.add_argument('--skip_extraction', action='store_true',
                      help='Skip frame extraction step')
    parser.add_argument('--skip_labels', action='store_true',
                      help='Skip label generation step')
    parser.add_argument('--skip_split', action='store_true',
                      help='Skip train/val split step')
    parser.add_argument('--process_train', action='store_true',
                      help='Process training set (default is validation set only)')
    args = parser.parse_args()

    # Set default output directory if not specified
    if args.out_directory is None:
        args.out_directory = os.path.join(args.dataset_root, "preprocess_russia_crash")

    # Custom crop for Russia dataset (hide largest watermarks)
    src_height, src_width = 986, 555
    russia_crop_extents = [int(0.032*src_height), -int(0.198*src_height), int(0.115*src_width), -int(0.115*src_width)] 

    # Extract frames from videos
    if not args.skip_extraction:
        extract_frames(args.dataset_dir, args.out_directory, crop_extents=russia_crop_extents)

    # Create labels (run bbox detector)
    if not args.skip_labels:
        generate_labels(args.dataset_dir, args.out_directory, video_subdir='val')
        if args.process_train:
            generate_labels(args.dataset_dir, args.out_directory, video_subdir='train')

    # Split into train and val sets
    if not args.skip_split:
        make_train_val_split(args.out_directory)
