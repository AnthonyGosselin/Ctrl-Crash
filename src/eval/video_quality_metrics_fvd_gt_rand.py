import numpy as np
import torch
import scipy.linalg
from typing import Tuple
import torch.nn.functional as F
import math
import cv2
import json
import random
import os
import argparse
from tqdm import tqdm

import numpy as np
import io
import re
import requests
import html
import hashlib
import urllib
import urllib.request
import uuid

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Dict

from src.datasets.dataset_utils import get_dataloader
from src.utils import get_samples


def get_frames_from_path_list(path_list):
    frames = []
    for path in path_list:
        img = cv2.imread(path)
        img = cv2.resize(img, [512, 320])
        frames.append(img)
    return np.array(frames)

def get_frames_mp4(video_path: str, frame_interval: int = 1) -> None:

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    saved_count = 0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (512, 320))
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frames.append(frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return np.array(frames)


def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    print(filename, "not found")
    return []


def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)

"""
Modified from https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
class FVD:
    def __init__(self, device,
                 detector_url='https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1',
                 rescale=False, resize=False, return_features=True):
        
        self.device = device
        self.detector_kwargs = dict(rescale=False, resize=False, return_features=True)
        
        with open_url(detector_url, verbose=False) as f:
            self.detector = torch.jit.load(f).eval().to(device)
        
        # Initialize ground truth statistics
        self.mu_real = None
        self.sigma_real = None
    
    def to_device(self, device):
        self.device = device
        self.detector = self.detector.to(self.device)
    
    def _compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0) # [d]
        sigma = np.cov(feats, rowvar=False) # [d, d]
        return mu, sigma
    
    def save_gt_stats(self, save_path: str):
        """Save ground truth statistics to a file."""
        if self.mu_real is None or self.sigma_real is None:
            raise ValueError("Ground truth statistics not computed yet")
        
        stats = {
            'mu_real': self.mu_real,
            'sigma_real': self.sigma_real
        }
        np.savez(save_path, **stats)
    
    def load_gt_stats(self, load_path: str):
        """Load ground truth statistics from a file."""
        stats = np.load(load_path)
        self.mu_real = stats['mu_real']
        self.sigma_real = stats['sigma_real']
    
    def preprocess_videos(self, videos, resolution=224, sequence_length=None):
        
        b, t, c, h, w = videos.shape
        
        # temporal crop
        if sequence_length is not None:
            assert sequence_length <= t
            videos = videos[:, :sequence_length, ::]
        
        # b*t x c x h x w
        videos = videos.reshape(-1, c, h, w)
        if c == 1:
            videos = torch.cat([videos, videos, videos], 1)
            c = 3
        
        # scale shorter side to resolution
        scale = resolution / min(h, w)
        # import pdb; pdb.set_trace()
        if h < w:
            target_size = (resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), resolution)
        
        videos = F.interpolate(videos, size=target_size).clamp(min=-1, max=1)
        
        # center crop
        _, c, h, w = videos.shape
        
        h_start = (h - resolution) // 2
        w_start = (w - resolution) // 2
        videos = videos[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
        
        # b, c, t, w, h
        videos = videos.reshape(b, t, c, resolution, resolution).permute(0, 2, 1, 3, 4)
        
        return videos.contiguous()
    
    @torch.no_grad()
    def evaluate(self, video_fake, video_real=None, res=224, use_saved_stats=False, save_stats_path=None):
        """Evaluate FVD score.
        
        Args:
            video_fake: Generated videos
            video_real: Ground truth videos (optional if use_saved_stats=True)
            res: Resolution for preprocessing
            use_saved_stats: Whether to use saved ground truth statistics
        """
        video_fake = self.preprocess_videos(video_fake, resolution=res)
        feats_fake = self.detector(video_fake, **self.detector_kwargs).cpu().numpy()
        
        if use_saved_stats:
            if self.mu_real is None or self.sigma_real is None:
                raise ValueError("Ground truth statistics not loaded. Call load_gt_stats() first.")
            mu_real = self.mu_real
            sigma_real = self.sigma_real
        else:
            if video_real is None:
                raise ValueError("video_real must be provided when use_saved_stats=False")
            video_real = self.preprocess_videos(video_real, resolution=res)
            feats_real = self.detector(video_real, **self.detector_kwargs).cpu().numpy()
            mu_real, sigma_real = self._compute_stats(feats_real)
            # Save the computed statistics
            self.mu_real = mu_real
            self.sigma_real = sigma_real
            if save_stats_path is not None:
                self.save_gt_stats(save_stats_path) #"/network/scratch/a/anthony.gosselin/CarCrashEvalSamples/gt_fvd_stats")
        
        mu_gen, sigma_gen = self._compute_stats(feats_fake)
        
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return fid
    

def collect_fvd_stats(data_root, samples=200, downsample_int=1, num_frames=25, save_path=None, action_type=None):
    """Collect and save ground truth statistics for FVD evaluation."""
    
    if save_path is None:
        save_path = os.path.join(data_root, "gt_fvd_stats.npz")

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

    # Create dataset and dataloader
    dataset_name = "mmau"
    train_set = True
    val_dataset, val_loader = get_dataloader(data_root, dataset_name, 
                                            if_train=train_set, clip_length=num_frames,
                                            batch_size=1, num_workers=0, shuffle=True,
                                            image_height=320, image_width=512,
                                            non_overlapping_clips=True, 
                                            specific_categories=specific_categories,
                                            force_clip_type=force_clip_type)
    
    # Collect video paths
    gt_videos = []
    for sample in tqdm(val_loader, desc="Collecting samples", total=samples):
        vid_path = os.path.dirname(sample["image_paths"][0][0])
        gt_videos.append(vid_path)
        if len(gt_videos) >= samples:
            break
    
    random.shuffle(gt_videos)
    
    num_found_samples = len(gt_videos)
    print(f"Found {num_found_samples} ground truth video directories")

    # Initialize array for all videos
    all_videos = torch.zeros((num_found_samples, num_frames, 3, 320, 512), device="cuda")
    
    # Load and process videos
    valid = 0
    for idx, video_path in tqdm(enumerate(gt_videos), desc="Processing videos", total=num_found_samples):
        if valid == num_found_samples:
            break
        
        # Get list of jpg files in directory
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        
        if len(frame_files) < num_frames:
            print(f"Skipping {video_path.split('/')[-1]}, insufficient frames: {len(frame_files)}")
            continue
        
        # Load frames
        frames = []
        for frame_file in frame_files[0:num_frames:downsample_int]:
            frame_path = os.path.join(video_path, frame_file)
            img = cv2.imread(frame_path)
            img = cv2.resize(img, (512, 320))
            frames.append(img)
        
        frames = torch.tensor(np.array(frames), device="cuda")
        
        # Process frames
        frames = frames.unsqueeze(0).permute(0, 1, 4, 2, 3)
        all_videos[valid] = frames[:, :num_frames, ::]
        valid += 1
    
    if valid == 0:
        raise ValueError("No valid videos found")
    
    # Convert to torch tensor and normalize
    all_videos = all_videos.float()
    all_videos.div_(255/2.0).sub_(1.0)
    
    # Initialize FVD and compute statistics
    with torch.no_grad():
        fvd = FVD(device='cuda')
        video_real = fvd.preprocess_videos(all_videos)
        feats_real = fvd.detector(video_real, **fvd.detector_kwargs).cpu().numpy()
        mu_real, sigma_real = fvd._compute_stats(feats_real)
    
    # Save statistics
    stats = {
        'mu_real': mu_real,
        'sigma_real': sigma_real,
        'num_videos': valid,
        'num_frames': num_frames,
        'resolution': 320
    }
    np.savez(save_path, **stats)
    print(f"Saved ground truth statistics to {save_path}")
    
    # Clean up
    del fvd, all_videos, video_real, feats_real
    torch.cuda.empty_cache()
    
    return save_path

def evaluate_vids(vid_root, samples=200, downsample_int=1, num_frames=25, gt_stats=None, shuffle=False):
    """Evaluate FVD score for generated videos using pre-computed ground truth statistics."""
    
    # Initialize FVD and load ground truth statistics
    fvd = FVD(device='cuda')
    if gt_stats is not None:
        fvd.load_gt_stats(gt_stats)

    # Collect generated video paths
    f_gen_vid = []
    gen_videos = os.path.join(vid_root, "gen_videos") if os.path.exists(os.path.join(vid_root, "gen_videos")) else vid_root
    for fname in os.listdir(gen_videos):
        f_gen_vid.append(fname)

    print(f"Number of generated videos: {len(f_gen_vid)}")

    if not shuffle:
        f_gen_vid.sort()
    else:
        random.shuffle(f_gen_vid)

    f_gen_vid = f_gen_vid[:samples]
 
    # Initialize array for all videos
    all_gen = np.zeros((samples, num_frames, 3, 320, 512))

    # Load and process videos
    valid = 0
    for idx, fgen in tqdm(enumerate(f_gen_vid)):
        if valid == samples:
            break
        
        gen_vid_path = os.path.join(gen_videos, fgen)
        gen_vid = get_frames_mp4(gen_vid_path, frame_interval=downsample_int)
          
        if gen_vid.shape[0] < num_frames:
            print("Skipping, wrong size:", gen_vid.shape[0])
            continue

        gen_vid = np.expand_dims(gen_vid, 0).transpose(0, 1, 4, 2, 3)
        all_gen[valid] = gen_vid[:, :num_frames, ::]
        valid += 1

    # Convert to torch tensor and normalize
    all_gen = torch.from_numpy(all_gen).cuda().float()
    all_gen /= 255/2.0
    all_gen -= 1.0

    # Compute FVD score
    fvd_score = fvd.evaluate(all_gen, video_real=None, use_saved_stats=True)
    del fvd

    print(f'FVD Score: {fvd_score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate FVD score using pre-computed ground truth statistics')
    parser.add_argument('--vid_root', type=str, required=True,
                      help='Root directory containing generated videos')
    parser.add_argument('--samples', type=int, default=200,
                      help='Number of samples to evaluate (default: 200)')
    parser.add_argument('--num_frames', type=int, default=25,
                      help='Number of frames per video (default: 25)')
    parser.add_argument('--downsample_int', type=int, default=1,
                      help='Downsample interval for frames (default: 1)')
    parser.add_argument('--gt_stats', type=str, default=None,
                      help='Path to ground truth statistics file (optional)')
    parser.add_argument('--shuffle', action='store_true',
                      help='Shuffle videos before evaluation')
    parser.add_argument('--collect_stats', action='store_true',
                      help='Collect and save ground truth statistics')
    parser.add_argument('--data_root', type=str, default="/network/scratch/a/anthony.gosselin/Datasets",
                      help='Root directory for datasets (default: /network/scratch/a/anthony.gosselin/Datasets)')
    parser.add_argument('--action_type', type=int, default=None,
                      help='Action type to filter videos (0: normal, 1-4: crash types)')
    args = parser.parse_args()

    if args.collect_stats:
        stats_path = collect_fvd_stats(args.data_root, args.samples, args.downsample_int, args.num_frames, args.gt_stats, args.action_type)
        args.gt_stats = stats_path

    evaluate_vids(args.vid_root, args.samples, args.downsample_int, args.num_frames, args.gt_stats, args.shuffle)
