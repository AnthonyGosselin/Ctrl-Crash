from .base_dataset import BaseDataset

from PIL import Image
import os
import json


class BDD100KDataset(BaseDataset):
    CLASS_NAME_TO_ID = {
        'pedestrian': 1,
        'rider': 2,
        'car': 3,
        'truck': 4,
        'bus': 5,
        'train': 6,
        'motorcycle': 7,
        'bicycle': 8,
        'traffic light': 9,
        'traffic sign': 10,
    }

    TO_COCO_LABELS = {
        1: 0,
        2: 0,
        3: 2,
        4: 7,
        5: 5,
        6: 6,
    }

    TO_IMAGE_DIR = 'images/track'
    TO_BBOX_DIR = 'bboxes/track'
    TO_LABEL_DIR = 'labels'
    TO_BBOX_LABELS = 'labels/box_track_20'
    TO_SEG_LABELS = 'labels/seg_track_20/colormaps'
    TO_POSE_LABELS = 'labels/pose_21'

    def __init__(self,
                 root='./datasets', 
                 train=True,
                 clip_length=25,
                 #orig_height=720, orig_width=1280, # TODO: Define this (and use it)
                 resize_height=320, resize_width=512,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 sample_clip_from_end=True,
                 ego_only=False,
                 ignore_labels=False,
                 use_preplotted_bbox=True,
                 specific_samples=None,
                 specific_categories=None,
                 force_clip_type=None):

        super(BDD100KDataset, self).__init__(root=root, 
                                            train=train, 
                                            clip_length=clip_length,
                                            resize_height=resize_height, 
                                            resize_width=resize_width,
                                            non_overlapping_clips=non_overlapping_clips,
                                            bbox_masking_prob=bbox_masking_prob,
                                            sample_clip_from_end=sample_clip_from_end,
                                            ego_only=ego_only,
                                            ignore_labels=ignore_labels)
        
        self.MAX_BOXES_PER_DATA = 30
        self._location = 'train' if self.train else 'val'
        self.version = 'bdd100k'
        self.use_preplotted_bbox = use_preplotted_bbox

        self.image_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_IMAGE_DIR, self._location)
        self.bbox_label_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_BBOX_LABELS, self._location)
        self.bbox_image_dir = os.path.join(self.root, self.version, BDD100KDataset.TO_BBOX_DIR, self._location)

        if specific_categories is not None:
            print("BDD100k does not support `specific_categories`")
        if force_clip_type is not None:
            print("BDD100k does not support `force_clip_type`")
        self.specific_samples = specific_samples
        if self.specific_samples is not None:
            print("Only loading specific samples:", self.specific_samples)

        listed_image_dir = os.listdir(self.image_dir)
        try:
            listed_image_dir.remove('pred')
        except:
            pass
        self.clip_folders = sorted(listed_image_dir)
        self.clip_folder_lengths = {k:len(os.listdir(os.path.join(self.image_dir, k))) for k in self.clip_folders}

        for l in self.clip_folder_lengths.values():
            assert l >= self.clip_length, f'clip length {self.clip_length} is too long for clip folder length {l}'

        self._collect_clips()

    def _collect_clips(self):
        print("Collecting dataset clips...")
        
        for clip_folder in self.clip_folders:
            clip_path = os.path.join(self.image_dir, clip_folder)
            clip_frames = sorted(os.listdir(clip_path))

            if self.specific_samples is not None and clip_folder not in self.specific_samples:
                continue
            
            # Add all images to image_files
            image_indices = []
            for frame in clip_frames:
                self.image_files.append(os.path.join(clip_path, frame))
                image_indices.append(len(self.image_files)-1)
            
            # Create clips of length clip_length
            if self.clip_length is not None:
                # Collect clips as overlapping clips (i.e. A video with 30 frames will yield 5 25-frame clips)
                for start_image_idx in range(0, len(clip_frames) - self.clip_length + 1):
                    end_image_idx = start_image_idx + self.clip_length
                    clip_indices = image_indices[start_image_idx:end_image_idx]
                    self.clip_list.append(clip_indices)

    def _parse_label(self, label_file, frame_id):
        target = []
        with open(label_file, 'r') as f:
            label = json.load(f)
            frame_i = int(frame_id[-11:-4])-1
            assert frame_id == label[frame_i]['name']
            for obj in label[frame_i-1]['labels']:
                if obj['category'] not in BDD100KDataset.CLASS_NAME_TO_ID:
                    continue
                target.append({
                    'frame_name': frame_id,
                    'track_id': int(obj['id']),
                    'bbox': [obj['box2d']['x1'], obj['box2d']['y1'], obj['box2d']['x2'], obj['box2d']['y2']],
                    'class_id': BDD100KDataset.CLASS_NAME_TO_ID[obj['category']],
                    'class_name': obj['category'],
                })
                if len(target) >= self.MAX_BOXES_PER_DATA:
                    break
        return target

    def _getimageitem(self, frame_index, masked_track_ids=None):
        # Get the image
        image_file = self.image_files[frame_index]
        image = Image.open(image_file)
        image = self.transform(image)

        if not self.ignore_labels:
            # Get the labels
            clip_id = image_file[:image_file.rfind('/')]
            clip_id = clip_id[clip_id.rfind('/')+1:]
            label_file = os.path.join(self.bbox_label_dir, f'{clip_id}.json')
            frame_id = image_file[image_file.rfind('/')+1:]
            labels = self._parse_label(label_file, frame_id)
            
            # Get the bbox image
            if self.use_preplotted_bbox:
                bbox_file = self.get_bbox_image_file_by_index(image_file=image_file)
                bbox_im = Image.open(bbox_file)
                bbox_im = self.transform(bbox_im)
            else:
                bbox_im = self._draw_bbox(labels, masked_track_ids=masked_track_ids)
        else:
            labels = None
            bbox_im = None
            
        ret_dict = {"image": image, 
                    "image_path": image_file,
                    "labels": labels, 
                    "frame_index": frame_index, 
                    "bbox_image": bbox_im}
        
        return ret_dict

    def get_bbox_image_file_by_index(self, index=None, image_file=None):
        if image_file is None:
            image_file = self.get_image_file_by_index(index)

        return image_file.replace(BDD100KDataset.TO_IMAGE_DIR, BDD100KDataset.TO_BBOX_DIR)

    def get_image_file_by_index(self, index):
        return self.image_files[index]

    def __len__(self):
        return len(self.clip_list) if self.clip_length is not None else len(self.image_files)

if __name__ == "__init__":
    dataset = BDD100KDataset()
