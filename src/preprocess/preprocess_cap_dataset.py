import os
import cv2
import json
from tqdm import tqdm
from glob import glob
import argparse

from yolo_sam import YoloSamProcessor

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    print(filename, "not found")
    return []

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


def crop_images(images_dir_path, output_dir_path, crop_extents=None):
    """
    Crop frames
    """

    all_images = sorted(glob(f"{images_dir_path}/*.jpg"))
    total_frames = len(all_images)
    sample_image = cv2.imread(all_images[0])
    sample_name = str(int(images_dir_path.split('/')[-2])).zfill(5)
    sample_category = images_dir_path.split('/')[-3]
    out_vid_name = f"{sample_category}_{sample_name}"
    src_height, src_width = sample_image.shape[:2]

    if crop_extents:
        src_height, src_width = (src_height + crop_extents[1]) - crop_extents[0], (src_width + crop_extents[3]) - crop_extents[2]

    # print(f"Source images '{out_vid_name}': {src_height}x{src_width}")

    image_output_folder = os.path.join(output_dir_path, out_vid_name)
    os.makedirs(image_output_folder, exist_ok=True)

    out_frame_count = 0
    # sample_test = []
    for frame_idx in range(total_frames):
            
        frame_path = all_images[frame_idx]
        frame = cv2.imread(frame_path)

        # Crop frame
        if crop_extents:
            frame = frame[crop_extents[0]:crop_extents[1], crop_extents[2]:crop_extents[3]]
        
        # Save frame
        out_image_name = f"{out_vid_name}_{str(frame_idx).zfill(4)}.jpg"
        out_image_path = os.path.join(image_output_folder, out_image_name)
        cv2.imwrite(out_image_path, frame)

        # sample_test.append(frame)

        out_frame_count += 1

    print(f"Done '{out_vid_name}': {src_height}x{src_width}, {out_frame_count} frames")

    # create_video(sample_test, "path/to/sample_test_vid", fps=6)


def extract_frames(dataset_dir, out_directory, crop_extents=None, specific_videos=None):

    # NOTE: We are excluding all crashes that involve visible humans (pedestrians, cyclists, motorbikes...)
    video_types_to_exclude = [1, 2, 3, 4, 5, 6, 37, 38, 45, 46, 47, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

    for category_dir in sorted(os.listdir(dataset_dir)):
        category_dir_path = os.path.join(dataset_dir, category_dir)
        if not os.path.isdir(category_dir_path):
            continue

        # Let's filter the videos we want right away
        vid_type = int(category_dir)
        if int(vid_type) in video_types_to_exclude:
            continue

        for vid_name in tqdm(sorted(os.listdir(category_dir_path))):
            if specific_videos is not None and vid_name not in specific_videos:
                continue

            images_dir_path = os.path.join(category_dir_path, vid_name, "images")
            out_path = os.path.join(out_directory, "images", category_dir)
            crop_images(images_dir_path, out_path, crop_extents=crop_extents)

    print("Extraction complete.")


def generate_labels(out_directory, vid_names=None, subdir="", in_directory=None, reverse_order=False):
    label_output_folder = os.path.join(out_directory, "labels", subdir)
    os.makedirs(label_output_folder, exist_ok=True)

    # Checkpoint paths
    yolo_ckpt = "yolov8x.pt" # Will auto download with utltralytics
    sam2_ckpt = "/network/scratch/x/xuolga/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
    yolo_sam = YoloSamProcessor(yolo_ckpt, sam2_ckpt, sam2_cfg)

    samples_run = 0

    src_directory = in_directory if in_directory is not None else out_directory
    video_dir_root = os.path.join(src_directory, "images", subdir)
    for category in sorted(os.listdir(video_dir_root), reverse=reverse_order):
        category_root = os.path.join(video_dir_root, category)
        for video_name in tqdm(sorted(os.listdir(category_root), reverse=reverse_order)):
            if vid_names is not None and video_name not in vid_names:
                continue

            video_dir = os.path.join(category_root, video_name)
            if len(os.listdir(video_dir)) == 0:
                print("Empty video dir:", video_dir)
                continue
            
            # Skip if label file already exists
            out_label_path = os.path.join(label_output_folder, f"{video_name}.json")
            if os.path.exists(out_label_path):
                print(f"Skipping {video_name} - label file already exists")
                continue

            if len(os.listdir(video_dir)) > 300:
                print(f"SKIPPING LONG VIDEO {video_name}")
                continue
            
            print(f"Computing bboxes for {video_name}...")
            video_data = yolo_sam(video_dir, rel_bbox=True)

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
    parser = argparse.ArgumentParser(description='Process CAP dataset')
    parser.add_argument('--dataset_root', type=str, required=True,
                      help='Root directory for datasets')
    parser.add_argument('--dataset_dir', type=str, default="/network/scratch/l/luis.lara/dev/MM-AU/CAP-DATA",
                      help='Directory containing the CAP dataset')
    parser.add_argument('--out_directory', type=str, default=None,
                      help='Output directory (defaults to {dataset_root}/cap_images_12fps)')
    parser.add_argument('--reverse', action='store_true',
                      help='Process samples in reverse order')
    parser.add_argument('--skip_extraction', action='store_true',
                      help='Skip frame extraction step')
    parser.add_argument('--skip_labels', action='store_true',
                      help='Skip label generation step')
    parser.add_argument('--skip_split', action='store_true',
                      help='Skip train/val split step')
    args = parser.parse_args()

    # Set default output directory if not specified
    if args.out_directory is None:
        args.out_directory = os.path.join(args.dataset_root, "cap_images_12fps")

    # Extract frames from videos
    if not args.skip_extraction:
        cap_crop_extents = [40, -40, 128, -128]  # Custom crop for CAP dataset (get ratio right)
        extract_frames(args.dataset_dir, args.out_directory, crop_extents=cap_crop_extents, specific_videos=None)

    # Create labels (run bbox detector)
    if not args.skip_labels:
        in_directory = os.path.join(args.dataset_root, "cap_images_12fps")
        generate_labels(args.out_directory, vid_names=None, reverse_order=args.reverse)

    # Split into train and val sets
    if not args.skip_split:
        make_train_val_split(args.out_directory)
