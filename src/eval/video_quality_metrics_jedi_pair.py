import torch
import os
import argparse
from torch.utils.data import DataLoader

from videojedi import JEDiMetric
from .video_dataset import VideoDataset
from src.datasets.dataset_utils import get_dataloader

def custom_collate(batch):
    videos, targets = [], []
    for sample in batch:
        clips = sample["clips"]
        videos.append(clips)
    return torch.utils.data.dataloader.default_collate(videos), targets

def evaluate_vids(vid_root, samples=200, downsample_int=1, num_frames=25, gt_samples=500, test_feature_path=None, action_type=None):
    """Evaluate JEDi metric between generated and ground truth videos."""
    
    # Initialize JEDi metric
    jedi = JEDiMetric(feature_path=vid_root, 
                      test_feature_path=test_feature_path, 
                      model_dir="/path/to/Models")

    # Create dataset and dataloader for generated videos
    gen_dataset = VideoDataset(vid_root, num_frames=num_frames, downsample_int=downsample_int)
    gen_loader = DataLoader(gen_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Set up category filtering if specified
    specific_categories = None
    force_clip_type = None
    if action_type is not None:
        if action_type == 0:
            force_clip_type = "normal"
            print("Collecting normal samples only")
        else:
            classes_by_action_type = {
                1: [61, 62, 13, 14, 15, 16, 17, 18],
                2: list(range(1, 12 + 1)),
                3: [37, 39, 41, 42, 44] + list(range(19, 36 + 1)) + list(range(52, 60 + 1)),
                4: [38, 40, 43, 45, 46, 47, 48, 49, 50, 51]
            }
            specific_categories = classes_by_action_type[action_type]
            force_clip_type = "crash"
            print("Collecting crash samples from categories:", specific_categories)

    # Get specific samples to evaluate
    specific_samples = []
    gen_videos = os.path.join(vid_root, "gen_videos") if os.path.exists(f"{vid_root}/gen_videos") else vid_root
    for fname in os.listdir(gen_videos):
        vid_name = fname.strip("genvid_").split(".")[0]
        gt_vid_name = "_".join(vid_name.split("_")[1:])
        specific_samples.append(gt_vid_name)

    # Create dataset and dataloader for ground truth videos
    dataset_name = "mmau"
    train_set = False
    val_dataset, _ = get_dataloader("/path/to/Datasets", dataset_name, 
                                   if_train=train_set, clip_length=num_frames,
                                   batch_size=1, num_workers=0, shuffle=True, 
                                   image_height=320, image_width=512,
                                   non_overlapping_clips=True, 
                                   specific_categories=specific_categories,
                                   force_clip_type=force_clip_type,
                                   specific_samples=specific_samples)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
    
    # Compute JEDi metric
    jedi.load_features(train_loader=gen_loader, test_loader=val_loader, 
                      num_samples=samples, num_test_samples=gt_samples)
    jedi_metric = jedi.compute_metric()
    print(f"JEDi Metric: {jedi_metric}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate JEDi metric between generated and ground truth videos')
    parser.add_argument('--vid_root', type=str, required=True,
                      help='Root directory containing generated videos')
    parser.add_argument('--samples', type=int, default=200,
                      help='Number of samples to evaluate (default: 200)')
    parser.add_argument('--gt_samples', type=int, default=500,
                      help='Number of ground truth samples to use (default: 500)')
    parser.add_argument('--num_frames', type=int, default=25,
                      help='Number of frames per video (default: 25)')
    parser.add_argument('--downsample_int', type=int, default=1,
                      help='Downsample interval for frames (default: 1)')
    parser.add_argument('--test_feature_path', type=str, default=None,
                      help='Path to test features (optional)')
    parser.add_argument('--action_type', type=int, default=None,
                      help='Action type to filter videos (0: normal, 1-4: crash types)')
    args = parser.parse_args()

    evaluate_vids(args.vid_root, args.samples, args.downsample_int, args.num_frames, args.gt_samples, args.test_feature_path, args.action_type)
