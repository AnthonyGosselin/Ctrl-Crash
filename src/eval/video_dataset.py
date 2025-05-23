import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, video_root, num_frames=25, downsample_int=1, transform=None):
        """
        Args:
            video_root (str): Directory with all the video files
            num_frames (int): Number of frames to extract from each video
            downsample_int (int): Interval between frames to extract
            transform (callable, optional): Optional transform to be applied on frames
        """
        self.video_root = video_root
        self.num_frames = num_frames
        self.downsample_int = downsample_int
        self.transform = transform
        
        # Get list of video files
        self.video_files = []
        gen_videos = os.path.join(video_root, "gen_videos") if os.path.exists(os.path.join(video_root, "gen_videos")) else video_root
        for fname in os.listdir(gen_videos):
            if fname.endswith('.mp4'):
                self.video_files.append(os.path.join(gen_videos, fname))
        
        self.video_files.sort()
        
    def __len__(self):
        return len(self.video_files)
    
    def get_frames_mp4(self, video_path):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (512, 320))
            
            if frame_count % self.downsample_int == 0:
                frames.append(frame)
                
            frame_count += 1
            
            if len(frames) >= self.num_frames:
                break
                
        cap.release()
        
        if len(frames) < self.num_frames:
            # Pad with last frame if we don't have enough frames
            last_frame = frames[-1] if frames else np.zeros((320, 512, 3), dtype=np.uint8)
            while len(frames) < self.num_frames:
                frames.append(last_frame)
                
        return np.array(frames[:self.num_frames])
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        frames = self.get_frames_mp4(video_path)
        
        # Convert to torch tensor and normalize
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(0, 3, 1, 2)  # Change from (T, H, W, C) to (T, C, H, W)
        frames = frames / (255/2.0) - 1.0  # Normalize to [-1, 1]
        
        if self.transform:
            frames = self.transform(frames)
            
        return frames, []