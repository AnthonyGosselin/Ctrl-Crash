import os
import json
from tqdm import tqdm
import csv
import json
import cv2

from src.datasets.base_dataset import BaseDataset

def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    print(filename, "not found")
    return []

def create_video_from_images(images_list, output_video, out_fps, start_frame=None, end_frame=None):

    img0_path = images_list[0]
    img0 = cv2.imread(img0_path)
    height, width, _ = img0.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, out_fps, (width, height))

    for idx, frame_name in enumerate(images_list):

        if start_frame is not None and idx < start_frame:
            continue
        if end_frame is not None and idx >= end_frame:
            continue

        img = cv2.imread(frame_name)
        out.write(img)

    out.release()
    print("Saved video:", output_video)


class MMAUDataset(BaseDataset):

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
                 orig_height=640, orig_width=1024,
                 resize_height=320, resize_width=512,
                 non_overlapping_clips=False,
                 bbox_masking_prob=0.0,
                 sample_clip_from_end=True,
                 ego_only=False,
                 specific_samples=None,
                 specific_categories=None,
                 dada_only=False,
                 cleanup_dataset=False,
                 force_clip_type=None
                 ):
        
        self.ignore_labels = False
        if self.ignore_labels:
            print("IGNORING LABELS in MMAU dataset")
    
        super(MMAUDataset, self).__init__(root=root, 
                                            train=train, 
                                            clip_length=clip_length,
                                            resize_height=resize_height, 
                                            resize_width=resize_width,
                                            non_overlapping_clips=non_overlapping_clips,
                                            bbox_masking_prob=bbox_masking_prob,
                                            sample_clip_from_end=sample_clip_from_end,
                                            ego_only=ego_only,
                                            ignore_labels=self.ignore_labels) # NOTE: Ignoring labels currently
        
        self.dada_only = dada_only
        self.cleanup_dataset = cleanup_dataset
        self.dataset_name = "mmau_images_12fps" if not dada_only else "dada2000_images_12fps"

        self.orig_width = orig_width
        self.orig_height = orig_height
        self.split = "train" if train else "val"

        self.image_dir = os.path.join(self.root, self.dataset_name, "images")
        self.label_dir = os.path.join(self.root, self.dataset_name, "labels")
        self.bbox_image_dir = os.path.join(self.root, self.dataset_name, "bbox_images")

        self.downsample_6fps = True
        if self.downsample_6fps:
            print("Downsampling MMAU clips to 6 fps")
        
        self.ego_only = ego_only
        if self.ego_only:
            print("Ego collisions only filter set for MMAU dataset")

        self.strict_collision_filter = False
        if self.strict_collision_filter:
            print("Strict collision filter set for MMAU dataset")

        self.specific_samples = specific_samples
        if self.specific_samples is not None:
            print("Only loading specific samples:", self.specific_samples)

        self.specific_categories = specific_categories
        if self.specific_categories is not None:
            print("Only loading specific categories:", self.specific_categories)

        self.force_clip_type = force_clip_type
        if self.force_clip_type is not None:
            print("Only loading samples with type:", force_clip_type)

        self._collect_clips()
    
    def _collect_metadata_csv(self, metadata_csv_path):
        accident_frame_metadata = {}
        with open(metadata_csv_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            for i, row in enumerate(csv_reader):
                if i == 0:
                    continue

                video_num = str(int(row[0]))
                video_type = str(int(row[5]))
                abnormal_start_frame_idx = int(row[7])
                accident_frame_idx = int(row[9])
                abnormal_end_frame_idx = int(row[8])

                video_name = f"{video_type}_{video_num.rjust(5, '0')}"

                if accident_frame_idx == "-1":
                    # print("Skipping video:", video_name)
                    continue
                
                downsample_factor = 30/12 if video_num.startswith("90") else 1  # Downsample for DADA (30fps) and CAP (12 fps) are not the same
                if self.downsample_6fps:
                    downsample_factor *= 2
                accident_frame_metadata[video_name] = (int(abnormal_start_frame_idx / downsample_factor), 
                                                       int(accident_frame_idx / downsample_factor + 0.5), 
                                                       int(abnormal_end_frame_idx / downsample_factor + 0.5))
        
        return accident_frame_metadata

    def _collect_clips(self):
        print("Collecting dataset clips...")

        mmau_dataset = os.path.join(self.root, self.dataset_name)

        # Load data split
        datasplit_data = load_json(os.path.join(mmau_dataset, "mmau_datasplit.json"))

        # Compile reject videos
        auto_filtered_vids = load_json(os.path.join(mmau_dataset, "auto_low_quality.json"))
        rejected_vids = load_json(os.path.join(mmau_dataset, "rejected.json"))
        all_rejected_vids = auto_filtered_vids + rejected_vids

        # Collect the accident moment information
        accident_frame_metadata = self._collect_metadata_csv(os.path.join(mmau_dataset, "mmau_metadata.csv"))

        self.clip_type_list = [] # crash or normal or abnormal (abnormal is a scene that has abnormal driving but doesn't contain the actual crash moment)
        self.action_type_list = [] # 0-4, 0 is normal, 1-4 are different types of crashes
        image_indices_by_clip = {}

        null_labels = []
        # Iterate datasplit file
        count_vid = 0
        for category, split in datasplit_data.items():
            for split_name, vid_names in split.items():
                if split_name != self.split:
                    continue

                for vid_name in vid_names:
                    if vid_name in all_rejected_vids:
                        continue

                    if self.dada_only and not vid_name.split("_")[-1].startswith("90"): # NOTE: REMOVE THIS
                        continue
                    
                    # Read image files
                    image_dir = os.path.join(mmau_dataset, "images")
                    clip_file = os.path.join(image_dir, category, vid_name)
                    clip_frames = sorted(os.listdir(clip_file))

                    if self.cleanup_dataset:
                        # NOTE: For renaming frames (can remove this later)
                        fix_label = False
                        for frame_name in clip_frames:
                            if vid_name not in frame_name:
                                fix_label = True
                                new_frame_name = f"{vid_name}_{frame_name}"
                                root_path = os.path.join(mmau_dataset, "images", category, vid_name)
                                os.rename(os.path.join(root_path, frame_name), os.path.join(root_path, new_frame_name))
                        
                        image_dir = os.path.join(mmau_dataset, "images")
                        clip_file = os.path.join(image_dir, category, vid_name)
                        clip_frames = sorted(os.listdir(clip_file))

                        # Also rename in label file
                        label_file_path = os.path.join(self.label_dir, f"{vid_name}.json")
                        if os.path.exists(label_file_path) and fix_label:
                            with open(label_file_path, "r") as f:
                                data = json.load(f)
                                data_field = data["data"]
                                if data_field is None:
                                    print(f"{vid_name}.json CLIP DATA IS NULL 2")
                                    null_labels.append(vid_name)
                                else:
                                    for i, frame_data in enumerate(data_field):
                                        current_frame_name = frame_data["image_source"]
                                        if vid_name not in current_frame_name:
                                            new_frame_name = f"{vid_name}_{current_frame_name}"
                                            data["data"][i]["image_source"] = new_frame_name
                                            
                            with open(label_file_path, "w") as f:
                                json.dump(data, f, indent=1)

                    num_frames = len(clip_frames) if not self.downsample_6fps else len(clip_frames) // 2
                    if num_frames < self.clip_length:
                        print(f"{vid_name} does not have enough frames: has {num_frames}, expected at least {self.clip_length}")
                        continue

                    accident_metadata = accident_frame_metadata.get(vid_name)
                    if accident_metadata is None:
                        print(vid_name, "no accident metadata found")
                        continue

                    step = 2 if self.downsample_6fps else 1
                    clip_frame_names = []
                    for image_idx in range(0, len(clip_frames), step):
                        image_file = clip_frames[image_idx]
                        clip_frame_names.append(image_file)

                    count_vid += 1
                    # Read label file
                    if not self.ignore_labels:
                        label_file_path = os.path.join(self.label_dir, f"{vid_name}.json")
                        if not os.path.exists(label_file_path):
                            if num_frames <= 300:
                                # Because a lot of the long videos were rejected because they were too long to process
                                # print(f"{label_file_path} does not exist")
                                pass
                            continue

                        with open(label_file_path) as json_file:
                            all_data = json.load(json_file)
                            metadata = all_data['metadata']

                            if self.ego_only:
                                if metadata['ego_involved'] == False:
                                    continue
                            
                            if self.strict_collision_filter and metadata["accident_type"] in [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 28, 29, 31, 32, 34, 35, 36]:
                                # Reject video where the collision is with "static" agents
                                continue

                            # Some post-hoc rejected clips
                            if all_data["video_source"] in ["10_90001.mp4"]:
                                continue

                            if self.specific_samples is not None and all_data["video_source"].split(".")[0] not in self.specific_samples:
                                continue

                            if self.specific_categories is not None and metadata["accident_type"] not in self.specific_categories:
                                continue
                                
                            clip_data = all_data["data"]
                            if clip_data is None:
                                print(f"{vid_name}.json CLIP DATA IS NULL")
                                null_labels.append(vid_name)
                                continue

                        clip_label_data = self._parse_clip_labels(clip_data, clip_frame_names) 
                        self.frame_labels.extend(clip_label_data) # In this case, labels are already sorted so they will match up to the image indices

                    image_indices_by_clip[vid_name] = []
                    for image_file in clip_frame_names:
                        self.image_files.append(os.path.join(clip_file, image_file))
                        image_indices_by_clip[vid_name].append(len(self.image_files)-1)

                    if not self.ignore_labels:
                        assert len(self.frame_labels) == len(self.image_files), f"{len(self.frame_labels)} frame labels != {len(self.image_files)} image files" # We assume a one-to-one association between images and labels

                    ab_start_idx, acc_idx, ab_end_idx = accident_metadata
                    def get_clip_type(image_idx, end_image_idx):
                        clip_type = "normal"
                        if image_idx <= acc_idx and end_image_idx >= acc_idx: 
                            # Contains accident frame
                            clip_type = "crash"
                        elif (image_idx >= ab_start_idx and image_idx <= ab_end_idx) \
                            or (end_image_idx >= ab_start_idx and end_image_idx <= ab_end_idx) \
                            or image_idx > acc_idx: # Let's also consider "normal" driving clip that happen after the accident to be "abnormal" as they might show the aftermath (e.g. car damage)
                            # Does not contain accident frame, but contains abnormal driving (moment before and after accident)
                            clip_type = "abnormal"
                            
                        return clip_type

                    # Cut the videos in clips of the correct length 
                    # NOTE: Only implementing strategy of selecting two clips per video: 1 with normal driving and 1 with crash
                    #   Select normal driving from beginning preferably and crash clip try to center it on the accident instant
        
                    # Find crash clip
                    crash_found = False
                    if self.force_clip_type is None or self.force_clip_type == "crash":
                        start_image_idx, end_image_idx = None, None
                        total_frames = len(image_indices_by_clip[vid_name])
                        if acc_idx is not None and self.clip_length is not None:
                            # Keep frame_count frames around accident frame
                            start_image_idx = acc_idx - int(self.clip_length/2 + 0.5)
                            end_image_idx = acc_idx + int(self.clip_length/2)

                            if total_frames < self.clip_length:
                                print(f"Not enough frames in '{vid_name}': {total_frames}, skipping")
                            else:
                                if start_image_idx < 0:
                                    end_image_idx += -(start_image_idx)
                                    start_image_idx = 0
                                
                                if end_image_idx > total_frames:
                                    start_image_idx -= (end_image_idx - total_frames)
                                    end_image_idx = total_frames
                                
                                self.clip_list.append(image_indices_by_clip[vid_name][start_image_idx:end_image_idx])
                                self.clip_type_list.append("crash")
                                action_type = self._get_action_type(metadata["accident_type"])
                                self.action_type_list.append(action_type)
                                crash_found = True

                                # Debug: #############
                                # frame_path_list = [self.image_files[i] for i in image_indices_by_clip[vid_name][start_image_idx:end_image_idx]]
                                # create_video_from_images(frame_path_list, f"outputs/sample_clip_{vid_name}_crash.mp4", out_fps=6 if self.downsample_6fps else 12)

                                # # Debug plot bboxes:
                                # out_bbox_path = os.path.join("outputs", f"{vid_name}_bboxes_crash")
                                # os.makedirs(out_bbox_path, exist_ok=True)
                                # for frame_path, label_data in zip(frame_path_list, clip_label_data[start_image_idx:end_image_idx]):
                                #     plt.figure(figsize=(9, 6))
                                #     plt.axis("off")
                                #     img = Image.open(frame_path)
                                #     plt.imshow(img)
                                #     for obj in label_data:
                                #         color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[obj["class_id"]])) / 255.0
                                #         show_box(obj["bbox"], plt.gca(), label=str(obj["track_id"]), color=color)
                                    
                                #     frame_id_name = frame_path.split("_")[-1].split(".")[0]
                                #     plt.savefig(os.path.join(out_bbox_path, f"bboxes_frame_{frame_id_name}.jpg"))
                                #######################3
                        
                        if not crash_found:
                            print("Crash not found for", vid_name)

                        assert end_image_idx > start_image_idx

                    if self.force_clip_type is None or self.force_clip_type == "normal":
                        normal_found = False
                        for start_image_idx in range(len(image_indices_by_clip[vid_name]) - self.clip_length + 1):
                            end_image_idx = start_image_idx+self.clip_length
                            
                            clip_type = get_clip_type(start_image_idx, end_image_idx)
                            if clip_type == "abnormal" or clip_type == "crash":
                                # Let's just reject the abnormal clips
                                continue

                            self.clip_list.append(image_indices_by_clip[vid_name][start_image_idx:end_image_idx])
                            self.clip_type_list.append(clip_type)
                            self.action_type_list.append(0)
                            normal_found = True
                            
                            # Debug: ########
                            # frame_path_list = [self.image_files[i] for i in image_indices_by_clip[vid_name][start_image_idx:end_image_idx]]
                            # create_video_from_images(frame_path_list, f"outputs/sample_clip_{vid_name}_normal.mp4", out_fps=6 if self.downsample_6fps else 12)

                            # out_bbox_path = os.path.join("outputs", f"{vid_name}_bboxes_normal")
                            # os.makedirs(out_bbox_path, exist_ok=True)
                            # for frame_path, label_data in zip(frame_path_list, clip_label_data[start_image_idx:end_image_idx]):
                            #     plt.figure(figsize=(9, 6))
                            #     plt.axis("off")
                            #     img = Image.open(frame_path)
                            #     plt.imshow(img)
                            #     for obj in label_data:
                            #         color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[obj["class_id"]])) / 255.0
                            #         show_box(obj["bbox"], plt.gca(), label=str(obj["track_id"]), color=color)
                                
                            #     frame_id_name = frame_path.split("_")[-1].split(".")[0]
                            #     plt.savefig(os.path.join(out_bbox_path, f"bboxes_frame_{frame_id_name}.jpg"))
                            #################

                            break
                        
                        # if not normal_found:
                        # print("Normal not found for", vid_name)

                    assert len(self.clip_list) == len(self.clip_type_list) == len(self.action_type_list)
        
        print("Number of clips MMAU:", len(self.clip_list), f"({self.data_split})", f"(from {count_vid} original videos)")
        crash_clip_count = 0
        normal_clip_count = 0
        for clip_type in self.clip_type_list:
            if clip_type == "crash":
                crash_clip_count += 1
            elif clip_type == "normal":
                normal_clip_count += 1
        print(crash_clip_count, "crash clips", normal_clip_count, "normal clips")

        if self.cleanup_dataset and len(null_labels) > 0:
            print("Null labels:", null_labels)
            for label_name in null_labels:
                label_file_path = os.path.join(self.label_dir, f"{label_name}.json")
            if os.path.exists(label_file_path):
                os.remove(label_file_path)
                print("Removed label file:", label_file_path)
    
    def _parse_clip_labels(self, clip_data, clip_frame_names):
        frame_labels = []
        for frame_data in clip_data:
            obj_data = frame_data['labels']
            image_source = frame_data["image_source"]

            if self.downsample_6fps and image_source not in clip_frame_names:
                # Only preserve even numbered frames
                continue

            object_labels = []
            for label in obj_data:
                # Only keep the classes of interest
                class_id = MMAUDataset.CLASS_NAME_TO_ID.get(label['name'])
                if class_id is None:
                    continue

                # Convert bbox coordinates to pixel space wrt to image size
                bbox = label['box']
                bbox_coords_pixel = [int(bbox[0] * self.orig_width), # x1
                                     int(bbox[1] * self.orig_height), # y1
                                     int(bbox[2] * self.orig_width), # x2
                                     int(bbox[3] * self.orig_height)] # y2

                object_labels.append({
                    'frame_name': image_source,
                    'track_id': int(label['track_id']),
                    'bbox': bbox_coords_pixel,
                    'class_id': class_id,
                    'class_name': label['name'], # Class name of the object
                    })
            
            frame_labels.append(object_labels)
        
        return frame_labels

    def _get_action_type(self, accident_type):
        # [0: normal, 1: ego, 2: ego/veh, 3: veh, 4: veh/veh]
        accident_type = int(accident_type)
        if accident_type in [61, 62, 13, 14, 15, 16, 17, 18]:
            return 1
        elif accident_type in range(1, 12 + 1):
            return 2
        elif accident_type in [37, 39, 41, 42, 44] + list(range(19, 36 + 1)) + list(range(52, 60 + 1)):
            return 3
        elif accident_type in [38, 40, 43, 45, 46, 47, 48, 49, 50, 51]:
            return 4
        else:
            raise ValueError(f"Unknown accident type: {accident_type}")

def pre_cache_dataset(dataset_root):
    # dset = MMAUDataset(dataset_root, train=False, cleanup_dataset=True, specific_categories=["42"])

    dset = MMAUDataset(dataset_root, train=False, cleanup_dataset=True)
    # s = dset.__getitem__(0)

    # dset = MMAUDataset(dataset_root, train=False, cleanup_dataset=True)
    # s = dset.__getitem__(0)
    # Trigger label and bbox image cache generation
    # from time import time
    # dataset_train = DADA2000Dataset(root=dataset_root, train=True, clip_length=25, non_overlapping_clips=False)
    # t = time()
    # for i in tqdm(range(len(dataset_train))):
    #     d = dataset_train[i]
    #     if i >= 100:
    #         print("Time:", time() - t)
    #         print("break")

    # dataset_val = DADA2000Dataset(root=dataset_root, train=False, clip_length=25, non_overlapping_clips=True)
    # for i in tqdm(range(len(dataset_val))):
    #     d = dataset_val[i]
    
    # print("Done.")
    
if __name__ == "__main__":
    dataset_root = "/network/scratch/a/anthony.gosselin/Datasets"
    pre_cache_dataset(dataset_root)

    MMAUDataset(dataset_root, train=True)


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
}
"""



