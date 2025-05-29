import os
import json
from tqdm import tqdm
import csv

from src.datasets.base_dataset import BaseDataset


class DADA2000Dataset(BaseDataset):

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
                 orig_height=660, orig_width=1056,
                 resize_height=320, resize_width=512,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 sample_clip_from_end=True,
                 ego_only=False,
                 specific_samples=None):
    
        super(DADA2000Dataset, self).__init__(root=root, 
                                                 train=train, 
                                                 clip_length=clip_length,
                                                 resize_height=resize_height, 
                                                 resize_width=resize_width,
                                                 non_overlapping_clips=non_overlapping_clips,
                                                 bbox_masking_prob=bbox_masking_prob,
                                                 sample_clip_from_end=sample_clip_from_end,
                                                 ego_only=ego_only)

        self.dataset_name = "preprocess_dada2000"

        self.orig_width = orig_width
        self.orig_height = orig_height
        self.image_dir = os.path.join(self.root, self.dataset_name, "images", self.data_split)
        self.label_dir = os.path.join(self.root, self.dataset_name, "labels", self.data_split)
        self.bbox_image_dir = os.path.join(self.root, self.dataset_name, "bbox_images", self.data_split)
        self.metadata_csv_path = os.path.join(self.root, self.dataset_name, "metadata.csv") # TODO: This information could be transfered into each individual label file

        self.strict_collision_filter = True
        if self.strict_collision_filter:
            print("Strict collision filter set for DADA2000")

        self.specific_samples = specific_samples
        if self.specific_samples is not None:
            print("Only loading specific samples:", self.specific_samples)

        self._collect_clips()
    
    def _collect_clips(self):

        accident_frame_metadata = {}
        with open(self.metadata_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue

                video_num = row[0]
                video_type = row[5]
                abnormal_start_frame_idx = int(row[7])
                accident_frame_idx = int(row[8])
                abnormal_end_frame_idx = int(row[9])

                video_name = f"{video_type}_{video_num.rjust(3, '0')}"

                if accident_frame_idx == "-1":
                    # print("Skipping video:", video_name)
                    continue
                
                # Need to convert the original frame idx to closest downsampled frame index
                downsample_factor = 30/7 # Because we downsampled from 30fps to 7fps
                accident_frame_metadata[video_name] = (int(abnormal_start_frame_idx / downsample_factor), int(accident_frame_idx / downsample_factor + 0.5), int(abnormal_end_frame_idx / downsample_factor + 0.5))

        self.clip_type_list = [] # crash or normal or abnormal (abnormal is a scene that has abnormal driving but doesn't contain the actual crash moment)
        image_indices_by_clip = {}
        for label_file in sorted(os.listdir(self.label_dir)):
            if not label_file.endswith('.json'):
                continue

            full_filename = os.path.join(self.label_dir, label_file)
            with open(full_filename) as json_file:
                all_data = json.load(json_file)
                metadata = all_data['metadata']

                if self.ego_only:
                    print("Ego collisions only activated!")
                    if metadata['ego_involved'] == False:
                        continue
                
                if self.strict_collision_filter and metadata["accident_type"] in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 28, 29, 31, 32, 34, 35, 36]:
                    # Reject video where the collision is with "static" agents
                    continue

                # Some rejected clips
                if all_data["video_source"] in ["10_001.mp4"]:
                    continue

                if self.specific_samples is not None and all_data["video_source"].split(".")[0] not in self.specific_samples:
                    continue


            clip_filename = label_file.split('.')[0]
            clip_file = os.path.join(self.image_dir, clip_filename)
            clip_data = all_data["data"]
            clip_frames = sorted(os.listdir(clip_file))

            num_frames = len(clip_frames)
            if num_frames < self.clip_length:
                # print(f"{clip_filename} does not have enough frames: has {num_frames} expected at least {self.clip_length}")
                continue

            accident_metadata = accident_frame_metadata.get(clip_filename)
            if accident_metadata is None:
                print(clip_filename, "no accident metadata found")
                continue
            
            # Frames within the abnormal range are considered accidents and outside are considered normal driving
            # ab_start_idx, acc_idx, ab_end_idx = accident_metadata
            # if ab_end_idx - ab_start_idx >= self.clip_length:
            #     # We can just feed abnormal clip frames
            #     clip_data = clip_data[ab_start_idx:ab_end_idx]
            #     clip_frames = clip_frames[ab_start_idx:ab_end_idx]
            #     num_frames = len(clip_frames)
            # else:
            #     # print(clip_filename, "no enough abnormal frames:", ab_end_idx - ab_start_idx)
            #     continue

            # #NOTE: Let's trim really long videos: videos over 75 frames will get the first frames trimmed
            # if num_frames > 75:
            #     # print("Long video:", clip_filename, len(num_frames), "frames")
            #     frames_to_trim = num_frames - 75
            #     clip_data = clip_data[frames_to_trim:]
            #     clip_frames = clip_frames[frames_to_trim:]
            
            clip_label_data = self._parse_clip_labels(clip_data) 
            self.frame_labels.extend(clip_label_data) # In this case labels are already sorted so they will match up to the image indices
           
            image_indices_by_clip[clip_filename] = []
            for image_file in clip_frames:
                self.image_files.append(os.path.join(clip_file, image_file))
                image_indices_by_clip[clip_filename].append(len(self.image_files)-1)

            assert len(self.frame_labels) == len(self.image_files) # We assume a one-to-one association between images and labels

            ab_start_idx, acc_idx, ab_end_idx = accident_metadata
            def get_clip_type(image_idx, end_image_idx):
                clip_type = "normal"
                if image_idx <= acc_idx and end_image_idx > acc_idx: 
                    # Contains accident frame
                    clip_type = "crash"
                elif (image_idx >= ab_start_idx and image_idx <= ab_end_idx) or (end_image_idx > ab_start_idx and end_image_idx < ab_end_idx):
                    # Does not contain accident frame, but contains abnormal driving (moment before and after accident)
                    clip_type = "abnormal"
                
                return clip_type

            # Cut the videos in clips of the correct length according to the strategies chosen
            if not self.non_overlapping_clips:
                for image_idx in range(len(image_indices_by_clip[clip_filename]) - self.clip_length + 1):
                    end_image_idx = image_idx+self.clip_length 
                    
                    clip_type = get_clip_type(image_idx, end_image_idx)
                    if clip_type == "abnormal":
                        # Let's just reject the abnormal clips
                        continue

                    self.clip_list.append(image_indices_by_clip[clip_filename][image_idx:end_image_idx])
                    self.clip_type_list.append(clip_type)

            else:
                if self.sample_clip_from_end:
                    # In case self.clip_length << actual video sample length, we can create multiple non-overlapping clips for each sample  
                    # Prioritize selecting clips from the end, to make sur the accident is included (which tends to be at the end of the videos)  
                    total_frames = len(image_indices_by_clip[clip_filename]) 
                    for clip_i in range(total_frames // self.clip_length):
                        start_image_idx = total_frames - (self.clip_length * (clip_i + 1))
                        end_image_idx = total_frames - (self.clip_length * clip_i)

                        clip_type = get_clip_type(start_image_idx, end_image_idx)
                        if clip_type == "abnormal":
                            # Let's just reject the abnormal clips
                            continue

                        self.clip_list.append(image_indices_by_clip[clip_filename][start_image_idx:end_image_idx])
                        self.clip_type_list.append(clip_type)
                else:
                    total_frames = len(image_indices_by_clip[clip_filename])
                    for clip_i in range(total_frames // self.clip_length):
                        start_image_idx = clip_i * self.clip_length
                        end_image_idx = start_image_idx + self.clip_length

                        clip_type = get_clip_type(start_image_idx, end_image_idx)
                        if clip_type == "abnormal":
                            # Let's just reject the abnormal clips
                            continue

                        self.clip_list.append(image_indices_by_clip[clip_filename][start_image_idx:end_image_idx])
                        self.clip_type_list.append(clip_type)
        
        print("Number of clips DADA2000:", len(self.clip_list), f"({self.data_split})")
        crash_clip_count = 0
        normal_clip_count = 0
        for clip_type in self.clip_type_list:
            if clip_type == "crash":
                crash_clip_count += 1
            elif clip_type == "normal":
                normal_clip_count += 1
        print(crash_clip_count, "crash clips", normal_clip_count, "normal clips")
    
    def _parse_clip_labels(self, clip_data):
        frame_labels = []
        for frame_data in clip_data:
            obj_data = frame_data['labels']

            object_labels = []
            for label in obj_data:
                # Only keep the classes of interest
                class_id = DADA2000Dataset.CLASS_NAME_TO_ID.get(label['name'])
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
    from time import time
    dataset_train = DADA2000Dataset(root=dataset_root, train=True, clip_length=25, non_overlapping_clips=False)
    t = time()
    for i in tqdm(range(len(dataset_train))):
        d = dataset_train[i]
        if i >= 100:
            print("Time:", time() - t)
            print("break")

    dataset_val = DADA2000Dataset(root=dataset_root, train=False, clip_length=25, non_overlapping_clips=True)
    for i in tqdm(range(len(dataset_val))):
        d = dataset_val[i]
    
    print("Done.")
    
if __name__ == "__main__":
    dataset_root = "/path/to/Datasets"
    pre_cache_dataset(dataset_root)


"""
ACCIDENT TYPES
{
    "ego_car_involved": {
        "self_initiated": {
            "out_of_control": [61]
        },
        "dynamic_participants": {
            "person_centric": {
                "pedestrian": [1, 2],
                "cyclist": [3, 4]
            },
            "vehicle_centric": {
                "motorbike": [5, 6],
                "truck": [7, 8, 9],
                "car": [10, 11, 12]
            }
        },
        "static_participants": {
            "road_crentric": {
                "large_roadblocks": [13],
                "curb": [14],
                "small_roadblocks_potholes": [15]
            },
            "other_semantics_centric": {
                "trees": [16],
                "telegraph_poles": [17],
                "other_road_facilities": [18]
            }
        }
    },
    "ego_car_uninvolved": {
        "dynamic_participants": {
            "vehicle_centric": {
                "motorbike_motorbike": [37, 38],
                "truck_truck": [39, 40, 41],
                "car_car": [42, 43, 44],
                "motorbike_truck": [45, 46, 47],
                "truck_car": [48, 49],
                "car_motorbike": [50, 51]
            },
            "person_centric": [52, 53, 54, 55, 56, 57, 58, 59, 60]
        },
        "static_participants" : [19, 20, 21, 22, 25, 26, 28, 29, 31, 32, 34, 35, 36]
    },
    "summary": {
        "ego_car_involved": {
            "person_centric": [1, 2, 3, 4],
            "vehicle_centric": [5, 6, 7, 8, 9, 10, 11, 12],
            "static_participants": [13, 14, 15, 16, 17, 18],
            "out_of_control": [61]
        },
        "ego_car_uninvolved": {
            "static_participants": [19, 20, 21, 22, 25, 26, 28, 29, 31, 32, 34, 35, 36],
            "vehicle_centric": [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51],
            "person_centric": [52, 53, 54, 55, 56, 57, 58, 59, 60]
        }
    }
}
"""



