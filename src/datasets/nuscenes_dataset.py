from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points

import numpy as np
from pyquaternion import Quaternion
import os
from typing import Tuple
from nuscenes.utils.splits import create_splits_scenes
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
import json
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


# "Singleton" that holds the data so we only have to load once for training & validation
nusc_data = None

class NuScenesDataset(BaseDataset):

    CLASS_NAME_TO_ID = {
        "animal": 1,
        "human.pedestrian.adult": 1,
        "human.pedestrian.child":  1,
        "human.pedestrian.construction_worker": 1,
        "human.pedestrian.personal_mobility": 1,
        "human.pedestrian.police_officer": 1,
        "human.pedestrian.stroller": 1,
        "human.pedestrian.wheelchair": 1,

        # "movable_object.barrier": 10,
        # "movable_object.debris":  10,
        # "movable_object.pushable_pullable": 10,
        # "movable_object.trafficcone": 10,
        # "static_object.bicycle_rack":  10,

        "vehicle.bicycle":  8,

        "vehicle.bus.bendy":  5,
        "vehicle.bus.rigid":  5,

        "vehicle.car": 3,
        "vehicle.emergency.police": 3,

        "vehicle.construction":  4,
        "vehicle.emergency.ambulance": 4,
        "vehicle.trailer":  4,
        "vehicle.truck":  4,

        "vehicle.motorcycle":  7, 
        
        "None": 10,
    }

    def __init__(self,
                 root='./datasets', 
                 train=True,
                 clip_length=25,
                 orig_height=900, orig_width=1600,
                 resize_height=320, resize_width=512,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 test_split=False,
                 ego_only=False):

        super(NuScenesDataset, self).__init__(root=root, 
                                                train=train, 
                                                clip_length=clip_length,
                                                resize_height=resize_height, 
                                                resize_width=resize_width,
                                                non_overlapping_clips=non_overlapping_clips,
                                                bbox_masking_prob=bbox_masking_prob)
        
        self.dataset_name = 'nuscenes'
        self.train = train
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.non_overlapping_clips = non_overlapping_clips
        self.bbox_image_dir = os.path.join(self.root, self.dataset_name, "bbox_images", self.data_split)
        self.label_dir = os.path.join(self.root, self.dataset_name, "labels", self.data_split)
        os.makedirs(self.label_dir, exist_ok=True)

        self.inst_token_to_track_id = {}

        split_scenes = self._load_nusc(test_split)
        self._collect_clips(split_scenes)


    def _load_nusc(self, test_split):
        global nusc_data
        if nusc_data is None:
            data_split = 'v1.0-trainval' if not test_split else 'v1.0-test' 
            # data_split = 'v1.0-mini' # Or: 'v1.0-mini' for testing
            nusc_data = NuScenes(version=data_split, 
                                dataroot=os.path.join(self.root, self.dataset_name),
                                verbose=True)
        self.nusc = nusc_data

        dataset_split = 'train' if self.train else 'val'
        if test_split:
            dataset_split = 'test'

        split_scene_names = create_splits_scenes()[dataset_split]  # [train: 700, val: 150, test: 150]
        split_scenes = [scene for scene in nusc_data.scene if scene['name'] in split_scene_names]

        return split_scenes


    def _collect_clips(self, split_scenes):
        image_indices_by_scene = {}

        def collect_frame(scene_idx, sample_data):
            # Get image
            image_path = os.path.join(self.root, self.dataset_name, sample_data['filename'])
            self.image_files.append(image_path)
            if image_indices_by_scene.get(scene_idx) is None:
                image_indices_by_scene[scene_idx] = []
            image_indices_by_scene[scene_idx].append(len(self.image_files) - 1)

            # Parse label
            labels = self._parse_label(sample_data["token"])
            self.frame_labels.append(labels)

        # Interpolating annotations to increase the frame rate (nuscenes annotation fps=2Hz, video data fps=12Hz)
        self.fps = 7
        target_period = 1/self.fps # For fps downsampling
        max_frames_per_scene = 75
        print("Collecting nuscenes clips...")
        for scene_i, scene in enumerate(split_scenes):
            
            curr_data_token = self.nusc.get('sample', scene['first_sample_token'])['data']["CAM_FRONT"] 
            curr_sample_data = self.nusc.get('sample_data', curr_data_token)
            collect_frame(scene_i, curr_sample_data)

            cumul_delta = 0
            total_delta = 0
            t = 0
            while curr_data_token:
                curr_sample_data = self.nusc.get('sample_data', curr_data_token)
                
                next_sample_data_token = curr_sample_data['next']
                if not next_sample_data_token:
                    break
                next_sample_data = self.nusc.get('sample_data', next_sample_data_token)

                # FPS downsampling: only select certain frames based on elapsed times 
                delta = (next_sample_data['timestamp'] - curr_sample_data['timestamp']) / 1e6
                cumul_delta += delta
                total_delta += delta
                if cumul_delta >= target_period:
                    collect_frame(scene_i, next_sample_data)
                    t += 1
                    cumul_delta = cumul_delta - target_period

                curr_data_token = next_sample_data_token

                if len(image_indices_by_scene[scene_i]) > max_frames_per_scene:
                    break
            
            # print(f"Fps: {len(image_indices_by_scene[scene_i]) / total_delta:.4f}")

            if not self.non_overlapping_clips:
                for image_idx in range(len(image_indices_by_scene[scene_i]) - self.clip_length + 1):
                    self.clip_list.append(image_indices_by_scene[scene_i][image_idx:image_idx+self.clip_length])
            else:
                # In case self.clip_length << actual video sample length (~20s), we can create multiple non-overlapping clips for each sample
                total_frames = len(image_indices_by_scene[scene_i])
                for clip_i in range(total_frames // self.clip_length):
                    start_image_idx = clip_i * self.clip_length
                    self.clip_list.append(image_indices_by_scene[scene_i][start_image_idx:start_image_idx+self.clip_length])
                        
        print("Number of nuScenes clips:", len(self.clip_list), f"({'train' if self.train else 'val'})")


    def _parse_label(self, token):

        cam_front_data = self.nusc.get('sample_data', token)

        # Check cache, if it doesn't exist, then create label file
        filename = cam_front_data["filename"].split('/')[-1].split('.')[0]
        label_file_path = os.path.join(self.label_dir, f"{filename}.json")
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as json_file:
                object_labels = json.load(json_file)
            
            return object_labels
        else:
            front_camera_sensor = self.nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(front_camera_sensor['camera_intrinsic'])
            ego_pose = self.nusc.get('ego_pose', cam_front_data['ego_pose_token'])

            object_labels = []
            bbox_center_by_track_id = {}
            for bbox_3d in self.nusc.get_boxes(token):
                
                class_name = bbox_3d.name
                if class_name not in NuScenesDataset.CLASS_NAME_TO_ID:
                    continue
                class_id = NuScenesDataset.CLASS_NAME_TO_ID[class_name]
                
                instance_token = self.nusc.get('sample_annotation', bbox_3d.token)['instance_token']
                if instance_token not in self.inst_token_to_track_id:
                    self.inst_token_to_track_id[instance_token] = len(self.inst_token_to_track_id)

                # Project 3D bboxes to 2D 
                # (Code adapted from: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_2d_annotations_as_json.py)
                
                # Move them to the ego-pose frame.
                bbox_3d.translate(-np.array(ego_pose['translation']))
                bbox_3d.rotate(Quaternion(ego_pose['rotation']).inverse)

                # Move them to the calibrated sensor frame.
                bbox_3d.translate(-np.array(front_camera_sensor['translation']))
                bbox_3d.rotate(Quaternion(front_camera_sensor['rotation']).inverse)

                # Filter out the corners that are not in front of the calibrated sensor.
                corners_3d = bbox_3d.corners()
                in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
                corners_3d = corners_3d[:, in_front]

                # Project 3d box to 2d.
                corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()

                # Keep only corners that fall within the image.
                final_coords = self._post_process_coords(corner_coords)

                # Skip if the convex hull of the re-projected corners does not intersect the image canvas.
                if final_coords is None:
                    continue

                min_x, min_y, max_x, max_y = final_coords
                track_id = self.inst_token_to_track_id[instance_token]

                bbox_center_by_track_id[track_id] = bbox_3d.center

                obj_label = {
                    'frame_name': cam_front_data["filename"],
                    'track_id': track_id,
                    'bbox': [min_x, min_y, max_x, max_y],
                    'class_id': class_id,
                    'class_name': class_name,
                    }

                object_labels.append(obj_label)
            
            # Render the furthest bboxes first (closer ones should be on top)
            object_labels.sort(key=lambda label: np.linalg.norm(bbox_center_by_track_id[label["track_id"]]), reverse=True)

            # Cache file
            with open(label_file_path, 'w') as json_file:
                json.dump(object_labels, json_file)
            print("Cached labels:", label_file_path)

            return object_labels   
    

    def _post_process_coords(self, corner_coords: List) -> Union[Tuple[float, float, float, float], None]:
        """
        Get the intersection of the convex hull of the reprojected bbox corners and the image canvas, return None if no intersection.
        :param corner_coords: Corner coordinates of reprojected bounding box.
        :param imsize: Size of the image canvas.
        :return: Intersection of the convex hull of the 2D box corners and the image canvas.
        """
        imsize = (self.orig_width, self.orig_height)

        polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
        img_canvas = box(0, 0, imsize[0], imsize[1])

        if polygon_from_2d_box.intersects(img_canvas):
            img_intersection = polygon_from_2d_box.intersection(img_canvas)
            intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])

            min_x = min(intersection_coords[:, 0])
            min_y = min(intersection_coords[:, 1])
            max_x = max(intersection_coords[:, 0])
            max_y = max(intersection_coords[:, 1])

            return min_x, min_y, max_x, max_y
        else:
            return None


def pre_cache_dataset(dataset_root):
    # Trigger label and bbox image cache generation
    dataset_val = NuScenesDataset(root=dataset_root, train=False, clip_length=25, non_overlapping_clips=True)
    for i in tqdm(range(len(dataset_val))):
        d = dataset_val[i]

    dataset_train = NuScenesDataset(root=dataset_root, train=True, clip_length=25, non_overlapping_clips=True)
    for i in tqdm(range(len(dataset_train))):
        d = dataset_train[i]
    
    print("Done.")
    
if __name__ == "__main__":
    dataset_root = "/path/to/Datasets"
    pre_cache_dataset(dataset_root)