import numpy as np
import torch
import scipy.linalg
from typing import Tuple
import torch.nn.functional as F
import math
from torchvision import transforms
import cv2
import json
import argparse
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

"""
Copy-pasted from Copy-pasted from https://github.com/NVlabs/stylegan2-ada-pytorch
"""
import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid
from tqdm import tqdm

from distutils.util import strtobool
from typing import Any, List, Tuple, Union, Dict


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
    
    def to_device(self, device):
        self.device = device
        self.detector = self.detector.to(self.device)
    
    def _compute_stats(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu = feats.mean(axis=0) # [d]
        sigma = np.cov(feats, rowvar=False) # [d, d]
        return mu, sigma
    
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
    def evaluate(self, video_fake, video_real, res=224):
        
        video_fake = self.preprocess_videos(video_fake,resolution=res)
        video_real = self.preprocess_videos(video_real,resolution=res)
        feats_fake = self.detector(video_fake, **self.detector_kwargs).cpu().numpy()
        feats_real = self.detector(video_real, **self.detector_kwargs).cpu().numpy()
        
        mu_gen, sigma_gen = self._compute_stats(feats_fake)
        mu_real, sigma_real = self._compute_stats(feats_real)
        
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return fid

def evaluate_vids(vid_root, samples=200, downsample=False, num_frames=25):
    """Evaluate video quality metrics between generated and ground truth videos."""
    
    # Collect video paths
    vid_name_to_gt_frames = {}
    gt_videos_refs = os.path.join(vid_root, "gt_frames")
    for fname in os.listdir(gt_videos_refs):
        vid_name = fname.strip("gt_frames_").split(".")[0]
        vid_name_to_gt_frames[vid_name] = fname

    f_gen_vid = []
    gen_videos = os.path.join(vid_root, "gen_videos")
    for fname in os.listdir(gen_videos):
        f_gen_vid.append(fname)
        vid_name = fname.strip("genvid_").split(".")[0]
        assert vid_name_to_gt_frames.get(vid_name) is not None, f"{fname} has no matching gt frames"

    print(f"Number of generated videos: {len(f_gen_vid)}")
 
    # Initialize arrays for all videos
    all_gt = np.zeros((samples, num_frames, 3, 320, 512))
    all_gen = np.zeros((samples, num_frames, 3, 320, 512))

    # Load and process videos
    valid = 0
    for idx, fgen in tqdm(enumerate(f_gen_vid), desc="Collecting video frames"):
        if valid == samples:
            break

        vid_name = fgen.strip("genvid_").split(".")[0]
        fgt = vid_name_to_gt_frames[vid_name]

        gen_vid_path = os.path.join(gen_videos, fgen)
        gen_vid = get_frames_mp4(gen_vid_path)
        
        with open(os.path.join(gt_videos_refs, fgt)) as gt_json:
            gt_vid = get_frames_from_path_list(json.load(gt_json))
        
        if gt_vid.shape[0] < num_frames or gen_vid.shape[0] < num_frames:
            print("Skipping, wrong size:", gt_vid.shape[0], gen_vid.shape[0])
            continue

        gt_vid = np.expand_dims(gt_vid, 0).transpose(0, 1, 4, 2, 3)
        gen_vid = np.expand_dims(gen_vid, 0).transpose(0, 1, 4, 2, 3)

        all_gt[valid] = gt_vid[:, :num_frames, ::]
        all_gen[valid] = gen_vid[:, :num_frames, ::]
        valid += 1

    # Convert to torch tensors and normalize
    all_gt = torch.from_numpy(all_gt).cuda().float()
    all_gt /= 255/2.0
    all_gt -= 1.0
    all_gen = torch.from_numpy(all_gen).cuda().float()
    all_gen /= 255/2.0
    all_gen -= 1.0

    # Compute FVD score
    fvd = FVD(device='cuda')
    fvd_score = fvd.evaluate(all_gt, all_gen)
    del fvd

    # Compute LPIPS score
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    lpips_score = 0
    for idx in range(all_gen.shape[0]): 
        lpips_score += loss_fn_alex(all_gt[idx], all_gen[idx])/all_gen.shape[0]
    lpips_score = lpips_score.mean().item()
    del loss_fn_alex

    # Compute SSIM and PSNR scores
    all_gen = all_gen.detach().cpu().numpy()
    all_gt = all_gt.detach().cpu().numpy()

    ssim_score_vid = np.zeros(samples)
    ssim_score_image = np.zeros((samples, num_frames))
    psnr_score_vid = np.zeros(samples)
    psnr_score_image = np.zeros((samples, num_frames))
    psnr_score_all = psnr(all_gt, all_gen)

    for vid_idx in tqdm(range(all_gen.shape[0]), desc="Computing SSIM and PSNR"):
        for f_idx in range(all_gen.shape[1]):
            img_gt = all_gt[vid_idx, f_idx]
            img_gen = all_gen[vid_idx, f_idx]
            data_range = max(img_gt.max(), img_gen.max()) - min(img_gt.min(), img_gen.min())
            ssim_score_image[vid_idx, f_idx] = ssim(img_gt, img_gen, channel_axis=0, data_range=data_range, gaussian_weights=True, sigma=1.5)
            psnr_score_image[vid_idx, f_idx] = psnr(img_gt, img_gen, data_range=data_range)

        vid_gt = all_gt[vid_idx]
        vid_gen = all_gen[vid_idx]
        data_range = max(vid_gt.max(), vid_gen.max()) - min(vid_gt.min(), vid_gen.min())
        ssim_score_vid[vid_idx] = ssim(vid_gt, vid_gen, channel_axis=1, data_range=data_range, gaussian_weights=True, sigma=1.5)
        psnr_score_vid[vid_idx] = psnr(vid_gt, vid_gen, data_range=data_range)
    
    ssim_score_image_error = np.sqrt(((ssim_score_image - ssim_score_image.mean())**2).sum()/200)
    psnr_score_image_error = np.sqrt(((psnr_score_image - psnr_score_image.mean())**2).sum()/200)

    # Print results
    print(f'FVD Score: {fvd_score}')
    print(f'LPIPS Score: {lpips_score}')
    print(f'SSIM Score (per image): {ssim_score_image.mean()}')
    print(f'SSIM Score Error: {ssim_score_image_error}')
    print(f'PSNR Score (per image): {psnr_score_image.mean()}')
    print(f'PSNR Score Error: {psnr_score_image_error}')
    
    # Print copy-friendly format
    print("\nCopy friendly format:")
    print(f"{fvd_score}, {lpips_score}, {ssim_score_image.mean()}, {psnr_score_image.mean()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate video quality metrics between generated and ground truth videos')
    parser.add_argument('--vid_root', type=str, required=True,
                      help='Root directory containing generated and ground truth videos')
    parser.add_argument('--samples', type=int, default=200,
                      help='Number of samples to evaluate (default: 200)')
    parser.add_argument('--num_frames', type=int, default=25,
                      help='Number of frames per video (default: 25)')
    parser.add_argument('--downsample', action='store_true',
                      help='Downsample videos during evaluation')
    args = parser.parse_args()

    evaluate_vids(args.vid_root, args.samples, args.downsample, args.num_frames)

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
            
        # Save frame if it's the right interval
        if frame_count % frame_interval == 0:
            frames.append(frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    return np.array(frames)