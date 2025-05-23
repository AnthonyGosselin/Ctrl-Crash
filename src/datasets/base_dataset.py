import os
import torch
from torchvision import transforms
from PIL import Image
import random

from src.datasets.bbox_utils import plot_2d_bbox


class BaseDataset:

    def __init__(self,
                 root='./datasets', 
                 train=True,
                 clip_length=25,
                 # orig_width=None, orig_height=None,
                 resize_width=512, resize_height=320,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 sample_clip_from_end=True,
                 ego_only=False,
                 ignore_labels=False):
        
        self.root = root
        self.train = train
        self.clip_length = clip_length
        # self.orig_width = orig_width
        # self.orig_height = orig_height
        self.resize_width = resize_width
        self.resize_height = resize_height

        self.non_overlapping_clips = non_overlapping_clips
        self.bbox_masking_prob = bbox_masking_prob
        self.sample_clip_from_end = sample_clip_from_end
        self.ego_only = ego_only
        self.ignore_labels = ignore_labels

        self.data_split = 'train' if self.train else 'val'
        
        # Image transforms
        self.transform = transforms.Compose([
                         transforms.Resize((self.resize_height, self.resize_width)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # map from [0,1] to [-1,1]
                        ])
        self.revert_transform = transforms.Compose([
                         transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
                        ])
        
        self.image_files = []  # Contains the paths of all the images in the dataset
        self.clip_list = []  # Contains a list of image indices for each clip
        self.frame_labels = [] # For each image file, contains a list of dicts (labels of each object in the frame)

        self.disable_cache = True
        if self.disable_cache:
            print("Bbox image caching disabled")
    
    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, index):
        return self._getclipitem(index)

    def _getclipitem(self, index):
        frames_indices = self.clip_list[index]

        images, labels, bboxes, image_paths = [], [], [], []
        masked_track_ids = self._get_masked_track_ids(frames_indices)   
        for frame_idx in frames_indices:
            ret_dict = self._getimageitem(frame_idx, masked_track_ids=masked_track_ids)
            images.append(ret_dict["image"])
            labels.append(ret_dict["labels"])
            bboxes.append(ret_dict["bbox_image"])
            image_paths.append(ret_dict["image_path"])
        images = torch.stack(images)
        prompt = "" # NOTE: Currently not supporting prompts
        
        action_type = 0  # Assume "normal" driving when unspecified
        if hasattr(self, "action_type_list"):
            action_type = self.action_type_list[index]

        vid_name = self.image_files[frames_indices[0]].split("/")[-1].split(".")[0][:-5]

        if not self.ignore_labels:
            bboxes = torch.stack(bboxes)

            # NOTE: Keys are plural because this makes more sense when batches get collated
            ret_dict = {"clips": images, 
                        "prompts": prompt, 
                        "indices": index, 
                        "bbox_images": bboxes,
                        "action_type": action_type,
                        "vid_name": vid_name,
                        "image_paths": image_paths
                        } 
        else:
            ret_dict = {"clips": images, 
                        "prompts": prompt, 
                        "indices": index}

        return ret_dict
    
    
    def _getimageitem(self, frame_index, masked_track_ids=None):
        # Get the image
        image_file = self.image_files[frame_index]
        image = Image.open(image_file)
        image = self.transform(image)

        if not self.ignore_labels:

            # Get the labels
            labels = self.frame_labels[frame_index]
            
            # Get the bbox image (from cache or draw new one)
            image_filename = image_file.split('/')[-1].split('.')[0]
            cache_filename = f"{image_filename}_bboxes"
            cache_file = os.path.join(self.bbox_image_dir, f"{cache_filename}.jpg")
            redraw_for_masked_agents = masked_track_ids is not None and len(masked_track_ids) > 0
            if not os.path.exists(cache_file) or redraw_for_masked_agents or self.disable_cache:
                bbox_im = self._draw_bbox(labels, cache_img_name=cache_filename, masked_track_ids=masked_track_ids, disable_cache=redraw_for_masked_agents or self.disable_cache)
            else:
                bbox_im = Image.open(cache_file)
                bbox_im = self.transform(bbox_im)
        else:
            labels = None
            bbox_im = None
            
        ret_dict = {"image": image, 
                    "image_path": image_file,
                    "labels": labels, 
                    "frame_index": frame_index, 
                    "bbox_image": bbox_im}
        
        return ret_dict
    
    def _draw_bbox(self, frame_labels, cache_img_name=None, masked_track_ids=None, disable_cache=False):
        canvas = torch.zeros((3, self.orig_height, self.orig_width))
        bbox_im = plot_2d_bbox(canvas, frame_labels, show_track_color=True, masked_track_ids=masked_track_ids)
        transform = transforms.Compose([transforms.ToPILImage()])
        bbox_pil = transform(bbox_im)

        if cache_img_name is not None and not disable_cache:
            if not os.path.exists(self.bbox_image_dir):
                os.makedirs(self.bbox_image_dir, exist_ok=True)
            image_path = os.path.join(self.bbox_image_dir, f"{cache_img_name}.jpg")
            bbox_pil.save(image_path)
            print("Cached bbox file:", image_path)

        bbox_im = self.transform(bbox_pil)
        return bbox_im

    def _get_masked_track_ids(self, frames_indices):
        masked_track_ids = []
        if self.bbox_masking_prob > 0:
            # Find all the trackIDs in the clip, randomly select some to mask and exclude from the bbox rendering
            all_track_ids = set()
            for frame_idx in frames_indices:
                frame_labels = self.frame_labels[frame_idx] #self._parse_label(self.image_files[frame])
                for label in frame_labels:
                    track_id = label['track_id']
                    if track_id not in all_track_ids and random.random() <= self.bbox_masking_prob:
                        # Mask out this agent
                        masked_track_ids.append(track_id)
                    all_track_ids.add(label['track_id'])
        
        return masked_track_ids
    
    def get_frame_file_by_index(self, index, timestep=0):
        frames = self.clip_list[index]
        if timestep is None:
            ret = []
            for frame in frames:
                ret.append(self.image_files[frame])
            return ret
        return self.image_files[frames[timestep]]
    
    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_frame_file_by_index(index)
        
        clip_name = image_file.split("/")[-2]
        return image_file.replace(self.image_dir, self.bbox_image_dir).replace('/'+clip_name+'/', '/').replace(".jpg", "_bboxes.jpg")






