import os
import json

from src.datasets.base_dataset import BaseDataset


class RussiaCrashDataset(BaseDataset):

    CLASS_NAME_TO_ID = {
            'person': 1,
            'car': 3,
            'truck': 4,
            'bus': 5,
            'train': 6,
            'motorcycle': 7,
            'bicycle': 8,
        }

    def __init__(self,
                 root='./datasets', 
                 train=True,
                 clip_length=25,
                 orig_height=555, orig_width=986,
                 resize_height=320, resize_width=512,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 sample_clip_from_end=True,
                 ego_only=False,
                 specific_samples=None):
    
        super(RussiaCrashDataset, self).__init__(root=root, 
                                                 train=train, 
                                                 clip_length=clip_length,
                                                 resize_height=resize_height, 
                                                 resize_width=resize_width,
                                                 non_overlapping_clips=non_overlapping_clips,
                                                 bbox_masking_prob=bbox_masking_prob,
                                                 sample_clip_from_end=sample_clip_from_end,
                                                 ego_only=ego_only)

        self.dataset_name = "preprocess_russia_crash"

        self.orig_width = orig_width
        self.orig_height = orig_height
        self.image_dir = os.path.join(self.root, self.dataset_name, "images", self.data_split)
        self.label_dir = os.path.join(self.root, self.dataset_name, "labels", self.data_split)
        self.bbox_image_dir = os.path.join(self.root, self.dataset_name, "bbox_images", self.data_split)

        self.specific_samples = specific_samples

        self._collect_clips()
    
    def _collect_clips(self):
        image_indices_by_clip = {}
        for label_file in sorted(os.listdir(self.label_dir)):
            if not label_file.endswith('.json'):
                continue

            full_filename = os.path.join(self.label_dir, label_file)
            with open(full_filename) as json_file:
                all_data = json.load(json_file)
                metadata = all_data['metadata']

                # Only include dashcam samples 
                if metadata['camera'] != "Dashcam":
                    continue
                # Exclude animal and "other" accidents
                if metadata['accident_type'] == "Risk of collision/collision with an animal":
                    continue
                if metadata['accident_type'] == 'Other types of traffic accidents':
                    continue
                # NOTE uncomment to only include actual car collision (no close misses and dangerous events)
                # if metadata['collision_type'] == "No Collision": 
                #     continue
                if self.ego_only:
                    print("Ego collisions only activated!")
                    if metadata['collision_type'] == "No Collision" or metadata["ego_car_involved"] != "Yes": 
                        continue

            clip_filename = label_file.split('.')[0]
            clip_file = os.path.join(self.image_dir, clip_filename)

            if self.specific_samples is not None and clip_filename not in self.specific_samples:
                continue

            if len(os.listdir(clip_file)) < self.clip_length:
                # print(f"{clip_filename} does not have enough frames: has {len(os.listdir(clip_file))} expected at least {self.clip_length}")
                continue
            
            clip_label_data = self._parse_clip_labels(all_data["data"]) 
            self.frame_labels.extend(clip_label_data) # In this case labels are already sorted so they will match up to the image indices
           
            image_indices_by_clip[clip_filename] = []
            for image_file in sorted(os.listdir(clip_file)):
                self.image_files.append(os.path.join(clip_file, image_file))
                image_indices_by_clip[clip_filename].append(len(self.image_files)-1)

            assert len(self.frame_labels) == len(self.image_files) # We assume a one-to-one association between images and labels

            # Cut the videos in clips of the correct length according to the strategies chosen
            if not self.non_overlapping_clips:
                for image_idx in range(len(image_indices_by_clip[clip_filename]) - self.clip_length + 1):
                    self.clip_list.append(image_indices_by_clip[clip_filename][image_idx:image_idx+self.clip_length])
            else:
                if self.sample_clip_from_end:
                    # In case self.clip_length << actual video sample length, we can create multiple non-overlapping clips for each sample  
                    # Prioritize selecting clips from the end, to make sur the accident is included (which tends to be at the end of the videos)  
                    total_frames = len(image_indices_by_clip[clip_filename]) 
                    for clip_i in range(total_frames // self.clip_length):
                        start_image_idx = total_frames - (self.clip_length * (clip_i + 1))
                        end_image_idx = total_frames - (self.clip_length * clip_i)
                        self.clip_list.append(image_indices_by_clip[clip_filename][start_image_idx:end_image_idx])
                else:
                    total_frames = len(image_indices_by_clip[clip_filename])
                    for clip_i in range(total_frames // self.clip_length):
                        start_image_idx = clip_i * self.clip_length
                        end_image_idx = start_image_idx + self.clip_length
                        self.clip_list.append(image_indices_by_clip[clip_filename][start_image_idx:end_image_idx])
        
        print("Number of clips Russia_crash:", len(self.clip_list), f"({self.data_split})")
    
    def _parse_clip_labels(self, clip_data):
        frame_labels = []
        for frame_data in clip_data:
            obj_data = frame_data['labels']

            object_labels = []
            for label in obj_data:
                # Only keep the classes of interest
                class_id = RussiaCrashDataset.CLASS_NAME_TO_ID.get(label['name'])
                if class_id is None:
                    continue

                # Convert bbox coordinates to pixel space wrt to image size
                bbox = label['box']
                bbox_coords_pixel = [int(bbox[0] * self.orig_width), # x1
                                     int(bbox[1] * self.orig_height), # y1
                                     int(bbox[2] * self.orig_width), # x2
                                     int(bbox[3] * self.orig_height)] # y2

                object_labels.append({
                    'frame_name': frame_data["image_source"],
                    'track_id': int(label['track_id']),
                    'bbox': bbox_coords_pixel,
                    'class_id': class_id,
                    'class_name': label['name'], # Class name of the object
                    })
            
            frame_labels.append(object_labels)
        
        return frame_labels


def pre_cache_dataset(dataset_root):
    # Trigger label and bbox image cache generation
    dataset_val = RussiaCrashDataset(root=dataset_root, train=False, clip_length=25, non_overlapping_clips=True)
    for i in tqdm(range(len(dataset_val))):
        d = dataset_val[i]

    dataset_train = RussiaCrashDataset(root=dataset_root, train=True, clip_length=25, non_overlapping_clips=True)
    for i in tqdm(range(len(dataset_train))):
        d = dataset_train[i]
    
    print("Done.")
    
if __name__ == "__main__":
    from tqdm import tqdm
    
    dataset_root = "/network/scratch/a/anthony.gosselin/Datasets"
    pre_cache_dataset(dataset_root)



