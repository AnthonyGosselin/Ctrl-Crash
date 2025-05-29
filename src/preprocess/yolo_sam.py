import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import random
from itertools import combinations
from tqdm import tqdm
import json
import cv2

from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor

NUM_LOOK_BACK_FRAMES = 3
FPS = 12

CLASSES_TO_KEEP = { # Using YOLO ids
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    6: 'train',
    7: 'truck',
 }


def create_video_from_images(images_dir, output_video, out_fps, start_frame=None, end_frame=None):

    images = sorted(os.listdir(images_dir))

    img0_path = os.path.join(images_dir, images[0])
    img0 = cv2.imread(img0_path)
    height, width, _ = img0.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, out_fps, (width, height))

    for idx, frame_name in enumerate(images):

        if start_frame is not None and idx < start_frame:
            continue
        if end_frame is not None and idx >= end_frame:
            continue

        img = cv2.imread(os.path.join(images_dir, frame_name))
        out.write(img)

    out.release()
    print("Saved video:", output_video)


def show_mask(mask, ax, obj_id=None, random_color=False, label=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if label is not None:
        text_location = mask.nonzero()
        if len(text_location[0]) > 0:
            rand_point = random.randint(0, len(text_location[0]) - 1)
            ax.text(text_location[2][rand_point], text_location[1][rand_point], label, color=(1, 1, 1))
    ax.imshow(mask_image)

def show_box(box, ax, label=None, color=((1, 0.7, 0.7))):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=1))

    if label is not None:
        ax.text(x0 + w // 2, y0 + h // 2, label, color=color)


class TrackedObject:
    def __init__(self, track_id, class_id, bbox, initial_frame_idx):
        self.track_id = track_id
        self.bbox = bbox
        self.class_pred_counts = {class_id: 1}
        self.initial_frame_idx = initial_frame_idx

        self._top_class = class_id
    
    @property
    def class_id(self):
        """
        The class for the object is whichever class was predicted the most for it
        """
        if self._top_class is not None:
            return self._top_class
        
        top_class = None
        top_count = 0
        for class_id, count in self.class_pred_counts.items():
            if count >= top_count:
                top_count = count
                top_class = class_id
        
        self._top_class = top_class
        return top_class

    def new_pred(self, class_id):
        if class_id not in self.class_pred_counts:
            self.class_pred_counts[class_id] = 0
        self.class_pred_counts[class_id] += 1
        self._top_class = None  # Remove cached top_class
    
    def __repr__(self):
        return f"id:{self.track_id}, class:{self.class_id}, bbox:{self.bbox}, init_frame:{self.initial_frame_idx}"

    def __str__(self):
        return f"id:{self.track_id}, class:{self.class_id}, bbox:{self.bbox}, init_frame:{self.initial_frame_idx}"


class YoloSamProcessor():

    def __init__(self, yolo_ckpt, sam_ckpt, sam_cfg, gpu_id=0):
        # Load models
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = f"cuda:{gpu_id}"

        self.yolo_model = YOLO(yolo_ckpt)
        self.sam_model = build_sam2_video_predictor(sam_cfg, sam_ckpt, device=self.device)

    def __call__(self, video_dir, rel_bbox=True):
        # Renaming the videos to be compatible with SAM2
        prev_frame_paths = {}
        src_width, src_height = None, None
        self.num_frames = len(os.listdir(video_dir))
        for vid in os.listdir(video_dir):
            new_name = vid.split("_")[-1]
            og_path = os.path.join(video_dir, vid)
            new_path = os.path.join(video_dir, new_name)
            os.rename(og_path, new_path)
            prev_frame_paths[new_path] = og_path

            if src_width is None:
                img_sample = Image.open(new_path)
                src_width, src_height = img_sample.size

        self.out_data = None
        try:
            # YOLO model
            self.run_yolo(video_dir)
            self.filter_yolo_preds()

            # SAM2 model
            self.run_sam(video_dir)
            self.filter_sam_preds()
            
            self.extract_final_bboxes()

            # Format the data
            self.out_data = []
            for frame_idx, frame_data in enumerate(self.final_bboxes):
                frame_path = self.frame_path_by_idx[frame_idx]
                frame_name = prev_frame_paths[frame_path].split('/')[-1]
                self.out_data.append({
                    "image_source": frame_name,
                    "labels": []
                })
                for obj_id, bbox in frame_data.items():
                    
                    bbox = bbox.tolist()
                    if rel_bbox:
                        # Save bbox coordinates as a ratio to image size
                        formatted_bbox = [bbox[0]/src_width, bbox[1]/src_height, bbox[2]/src_width, bbox[3]/src_height]
                    else:
                        # Keep in absolute coordinates
                        formatted_bbox = bbox

                    tracked_obj = self.initial_bboxes[obj_id]
                    
                    out_label = {
                        "track_id": obj_id,
                        "name": CLASSES_TO_KEEP[tracked_obj.class_id],
                        "class": tracked_obj.class_id,
                        "box": formatted_bbox,
                    }
                    self.out_data[frame_idx]["labels"].append(out_label)
        except Exception as e:
            print("Yolo_Sam processor failed:", e)
        finally:
            # Revert back the names of the files
            print("Restoring names in video dir after exception")
            for new_path, old_path in prev_frame_paths.items():
                os.rename(new_path, old_path)

        return self.out_data
    
    def run_yolo(self, video_dir):
        self.initial_bboxes = {}  # Store the first bbox for each track id
        self.id_reassigns = {}
        all_preds_by_track_id = []

        self.frame_path_by_idx = {}

        # Reset yolo's tracker for new video
        if self.yolo_model.predictor is not None:
            self.yolo_model.predictor.trackers[0].reset()

        new_id_counter = 1
        sorted_frames = sorted(os.listdir(video_dir))
        for frame_idx, frame_file in enumerate(sorted_frames):
            frame_path = os.path.join(video_dir, frame_file)
            self.frame_path_by_idx[frame_idx] = frame_path
            img = Image.open(frame_path)

            yolo_results = self.yolo_model.track(img, persist=True, conf=0.1, verbose=False, device=self.device)
            yolo_boxes = yolo_results[0].boxes
            all_preds_by_track_id.append({})

            # If the detection is has a new track id, then we record
            if yolo_boxes.is_track:
                for idx in range(len(yolo_boxes)):
                    track_id = int(yolo_boxes.id[idx].item())
                    bbox = yolo_boxes.xyxy[idx].numpy()
                    class_id = int(yolo_boxes.cls[idx].item())

                    tracked_obj = self.initial_bboxes.get(track_id)
                    # Check if YOLO is trying to assign a id that was already predicted but was not present in the previous frame
                    #   This means YOLO lost this obj and is attempting to reassign. We will assign a new id for this as we want to
                    #   trust SAM with the tracking not YOLO.
                    prev_track_id = track_id
                    if tracked_obj is not None and frame_idx > 0 and track_id not in all_preds_by_track_id[frame_idx - 1]:

                        # Check if id has been reassigned
                        if self.id_reassigns.get(track_id) is not None:
                            track_id = self.id_reassigns.get(track_id)
                        
                        if track_id not in all_preds_by_track_id[frame_idx - 1]: # Check again because id might have changed
                            # Assign a new track id
                            track_id = 100 + new_id_counter
                            new_id_counter += 1
                            self.id_reassigns[prev_track_id] = track_id
                            tracked_obj = None
                            # print(f"Frame: {frame_idx} re-assigned id {prev_track_id}->{track_id}")
                    
                    if tracked_obj is None:

                        # Check overlap with existing bboxes to make sure this isn't a double detection
                        reject_detection = False
                        for idx2 in range(len(yolo_boxes)):
                            track_id2 = int(yolo_boxes.id[idx2].item())
                            if track_id2 in [track_id, prev_track_id] or self.initial_bboxes.get(track_id2) is None:
                                continue
                            bbox2 = yolo_boxes.xyxy[idx2].numpy()
                            iou = self._bbox_iou(bbox, bbox2)
                            if iou >= 0.8:
                                reject_detection = True
                                # print("Redetection! Frame:", frame_idx, "Iou:", iou, track_id, track_id2)
                                break
                        
                        if not reject_detection:
                            tracked_obj = TrackedObject(track_id, class_id, bbox, frame_idx)
                            self.initial_bboxes[track_id] = tracked_obj
                    else:
                        tracked_obj.new_pred(class_id)
                    
                    if tracked_obj is not None:
                        all_preds_by_track_id[frame_idx][track_id] = TrackedObject(track_id, class_id, bbox, tracked_obj.initial_frame_idx)


    def filter_yolo_preds(self):
        # Smooth classes detected + filter out unwanted classes
        self.filtered_objects = []
        self.filtered_objects_by_frame = {}
        for _, tracked_obj in self.initial_bboxes.items():
            if tracked_obj.class_id in CLASSES_TO_KEEP:
                self.filtered_objects.append(tracked_obj)

                if self.filtered_objects_by_frame.get(tracked_obj.initial_frame_idx) is None:
                    self.filtered_objects_by_frame[tracked_obj.initial_frame_idx] = []
                self.filtered_objects_by_frame[tracked_obj.initial_frame_idx].append(tracked_obj)

        self.initial_frame_idx_by_track_id = {obj.track_id: obj.initial_frame_idx for obj in self.filtered_objects}

        # print("Filtered objects:", self.filtered_objects)


    def run_sam(self, video_dir):
        self.video_segments = {}  # video_segments contains the per-frame segmentation results
        self.track_ids_to_reject = {} # {track_id: reject_all_before_frame_idx}

        if self.filtered_objects is None or len(self.filtered_objects) == 0:
            # There are no objects to track
            return

        inference_state = self.sam_model.init_state(video_path=video_dir) # NOTE: Kind of annoying that the model requires frames to be named with numbers only...

        self.sam_model.reset_state(inference_state)
        for obj in self.filtered_objects:
            _, out_obj_ids, out_mask_logits = self.sam_model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=obj.initial_frame_idx,
                obj_id=obj.track_id,
                box=obj.bbox,
            )
        
        def get_last_frame_occurrence(sam_track_ids_per_frame, track_id, current_idx):
            for frame_idx in range(current_idx-1, -1, -1):
                if track_id in sam_track_ids_per_frame.get(frame_idx, []):
                    return frame_idx
            return -1
            
        # run propagation throughout the video and collect the results in a dict
        sam_track_ids_per_frame = {}
        long_non_existence_track_ids = []
        with torch.cuda.amp.autocast(): # Need this for some reason to fix some casting issues... (BFloat16 and Float16 mismatches)
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_model.propagate_in_video(inference_state):

                sam_tracked_ids = []
                for pred_idx, mask_logits in enumerate(out_mask_logits):
                    mask = (mask_logits > 0.0).cpu().numpy()
                    track_id = out_obj_ids[pred_idx]
                    if mask.sum() > 0 and out_frame_idx > (self.initial_frame_idx_by_track_id[track_id] - NUM_LOOK_BACK_FRAMES):
                        sam_tracked_ids.append(track_id)
                sam_track_ids_per_frame[out_frame_idx] = sam_tracked_ids

                # Compare *new* YOLO preds and make sure they don't overlap with existing SAM preds
                for new_yolo_obj in self.filtered_objects_by_frame.get(out_frame_idx, []):
                    yolo_track_id = new_yolo_obj.track_id
                    yolo_bbox = new_yolo_obj.bbox

                    for ind, sam_track_id in enumerate(out_obj_ids):
                        if (sam_track_id == yolo_track_id) or (out_frame_idx < (self.initial_frame_idx_by_track_id[sam_track_id] - NUM_LOOK_BACK_FRAMES)):
                            continue

                        sam_mask = (out_mask_logits[ind] > 0.0).cpu().numpy()
                        sam_bbox = self._get_bbox_from_mask(sam_mask)
                        if sam_bbox is None:
                            continue

                        # Flag the SAM prediction only if this prediction has been lost for many frames and SAM is trying to recover it 
                        # (in which case we should keep the YOLO pred)
                        
                        last_occurrence_idx = get_last_frame_occurrence(sam_track_ids_per_frame, sam_track_id, out_frame_idx)
                        if (out_frame_idx - last_occurrence_idx) >= (FPS * 0.8) and last_occurrence_idx >= 0:
                            print(sam_track_id, "long non-existence:", out_frame_idx - last_occurrence_idx, "frames")
                            long_non_existence_track_ids.append(sam_track_id)

                        iou = self._bbox_iou(yolo_bbox, sam_bbox)
                        if iou > 0.8:
                            # Reject the SAM tracked object if it was lost for many frames
                            if sam_track_id in long_non_existence_track_ids:
                                rejected_track_id = sam_track_id
                                self.track_ids_to_reject[rejected_track_id] = self.initial_frame_idx_by_track_id[sam_track_id]
                                print(f"Frame {out_frame_idx}. {yolo_track_id} & {sam_track_id} iou: {iou:.2f}. Reject: {rejected_track_id} (all) for long non-existence")
                            else:
                                # Otherwise, choose the obj with the latest yolo initial frame detection to reject
                                yolo_initial_frame = self.initial_frame_idx_by_track_id[yolo_track_id] # This is just the current frame
                                sam_initial_frame = self.initial_frame_idx_by_track_id[sam_track_id]
                                yolo_error = yolo_initial_frame >= sam_initial_frame
                                rejected_track_id = yolo_track_id if yolo_error else sam_track_id
                                reject_all_before_frame_idx = self.num_frames if yolo_error else sam_initial_frame
                                self.track_ids_to_reject[rejected_track_id] = reject_all_before_frame_idx
                                print(f"Frame {out_frame_idx}. {yolo_track_id} & {sam_track_id} iou: {iou:.2f}. Reject: {rejected_track_id} ({'before frame #' + str(reject_all_before_frame_idx) if not yolo_error else 'all'})")

                self.video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        

    def filter_sam_preds(self):
        self.filtered_sam_preds = []
        self.filtered_yolo_preds = []
        self.num_sam_existence_frames_by_track_id = {}

        def check_reject_pred(obj_id, frame_idx):
            return obj_id in self.track_ids_to_reject and frame_idx < self.track_ids_to_reject[obj_id]

        for frame_idx in range(self.num_frames):
            self.filtered_sam_preds.append({})
            self.filtered_yolo_preds.append({})

            if frame_idx not in self.video_segments.keys():
                continue

            for obj_id, mask in self.video_segments[frame_idx].items():
                # Only keep mask predictions that happen after the initial frame (with a small buffer) 
                #   (SAM will try to predict them before the prompt frame, and will often get them wrong)
                if (frame_idx >= self.initial_frame_idx_by_track_id[obj_id] - NUM_LOOK_BACK_FRAMES) and not check_reject_pred(obj_id, frame_idx):
                    self.filtered_sam_preds[frame_idx][obj_id] = mask

                    if mask.sum() > 0:
                        if self.num_sam_existence_frames_by_track_id.get(obj_id) is None:
                            self.num_sam_existence_frames_by_track_id[obj_id] = 0
                        self.num_sam_existence_frames_by_track_id[obj_id] += 1

            for obj in self.filtered_objects:
                if obj.initial_frame_idx == frame_idx and not check_reject_pred(obj.track_id, frame_idx):
                    self.filtered_yolo_preds[frame_idx][obj.track_id] = obj.bbox


    def extract_final_bboxes(self):
        # Extract the bboxes from the predicted masks 
        #   Also filter any overlapping. At this stage if there is overlapping masks this is likely a fault on SAM's side 
        #   (id switching/collecting) and there is not much we can do for this.
        self.final_bboxes = []
        rejected_ids = []
        for frame_idx in range(self.num_frames):
            self.final_bboxes.append({})
            for obj_id, mask in self.filtered_sam_preds[frame_idx].items():
                mask_box = self._get_bbox_from_mask(mask)
                if mask_box is not None:
                    self.final_bboxes[frame_idx][obj_id] = mask_box
            
            # Compute IOU overlap and eliminate duplicates
            items_to_compare = list(self.final_bboxes[frame_idx].items()) + list(self.final_bboxes[frame_idx].items())
            for (id0, bbox0), (id1, bbox1) in combinations(items_to_compare, 2):
                if id0 == id1:
                    continue

                if id0 not in self.final_bboxes[frame_idx] or id1 not in self.final_bboxes[frame_idx]: # Could've been removed previously
                    continue

                if id0 in rejected_ids:
                    del self.final_bboxes[frame_idx][id0]
                    continue
                if id1 in rejected_ids:
                    del self.final_bboxes[frame_idx][id1]
                    continue

                iou = self._bbox_iou(bbox0, bbox1)
                if iou > 0.8:
                    # Rejecting the prediction that exists for the least amount of frames throughout the video
                    frame_count0 = self.num_sam_existence_frames_by_track_id[id0]
                    frame_count1 = self.num_sam_existence_frames_by_track_id[id1]
                    rejected_id = id0 if frame_count0 < frame_count1 else id1

                    del self.final_bboxes[frame_idx][rejected_id]
                    rejected_ids.append(rejected_id)
                    # print(f"Frame {frame_idx}. {id0} & {id1} iou: {iou}. Rejecting {rejected_id}")
    
    def _bbox_iou(self, box1, box2):
        """
        Compute the Intersection over Union (IoU) between two bounding boxes.
        
        Parameters:
            box1: (x1, y1, x2, y2) coordinates of the first box.
            box2: (x1, y1, x2, y2) coordinates of the second box.
        
        Returns:
            iou: Intersection over Union value (0 to 1).
        """
        # Get the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Compute intersection area
        inter_width = max(0, x2 - x1)
        inter_height = max(0, y2 - y1)
        inter_area = inter_width * inter_height

        # Compute areas of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Compute union area
        union_area = box1_area + box2_area - inter_area

        # Compute IoU
        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def _get_bbox_from_mask(self, mask):
        mask_points = mask.nonzero()
        if len(mask_points[0]) == 0:
            return None
        
        x0 = min(mask_points[2])
        y0 = min(mask_points[1])
        x1 = max(mask_points[2])
        y1 = max(mask_points[1])

        return np.array([x0, y0, x1, y1])

from collections import defaultdict
class CVCOLORS:
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)
    BROWN = (42,42,165)
    LIME=(51,255,153)
    GRAY=(128, 128, 128)
    LIGHTPINK = (222,209,255)
    LIGHTGREEN = (204,255,204)
    LIGHTBLUE = (255,235,207)
    LIGHTPURPLE = (255,153,204)
    LIGHTRED = (204,204,255)
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    
    TRACKID_LOOKUP = defaultdict(lambda: (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)))
    TYPE_LOOKUP = [BROWN, BLUE, PURPLE, RED, ORANGE, YELLOW, GREEN, LIGHTPURPLE, LIGHTPINK, LIGHTRED, GRAY]
    REVERT_CHANNEL_F = lambda x: (x[2], x[1], x[0])


if __name__ == "__main__":

    # Checkpoint paths
    yolo_ckpt = "yolov8x.pt" # Will auto download with utltralytics

    # NOTE: Need to download beforehand from https://github.com/facebookresearch/sam2
    sam2_ckpt = "/network/scratch/x/xuolga/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    sam2_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"

    yolo_sam = YoloSamProcessor(yolo_ckpt, sam2_ckpt, sam2_cfg)
    output_dir = f"/path/to/output_dir"
    bboxes_out_dir = os.path.join(output_dir, "bboxes")
    labels_out_dir = os.path.join(output_dir, "json")
    os.makedirs(bboxes_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    video_dir_root = f"/path/to/dada2000_images_12fps/images"
    videos = ['8_90038', '8_90002', '10_90019', '10_90045', '10_90027', '10_90029', '10_90082', '10_90021', '10_90064', '10_90083', '10_90141', '10_90139', '10_90034', '10_90134', '10_90056', '10_90169', '10_90040', '11_90109', '11_90162', '11_90202', '11_90142', '11_90180', '11_90161', '11_90091', '11_90189', '11_90002', '11_90192', '11_90221', '11_90181', '12_90007', '12_90042', '13_90002', '13_90008', '13_90007', '14_90012', '14_90018', '14_90014', '14_90027', '24_90017', '24_90005', '24_90006', '24_90011', '42_90021', '43_90013', '48_90078', '48_90031', '48_90001', '48_90075', '49_90030', '49_90021', '61_90016', '61_90004']

    for video_name in tqdm(videos):
        cat = video_name.split("_")[0]
        video_dir = os.path.join(video_dir_root, cat, video_name)

        if len(os.listdir(video_dir)) > 300:
            print("Skipping:", video_name)
            continue
        out_data = yolo_sam(video_dir, rel_bbox=False)
        
        # Output format:
        """
        out_data = [ # List of Dicts, each inner dict represents one frame of the video
                        {
                            "image_source": img1.jpg,
                            "labels":
                                [ # List of Dicts, each dict represents one tracked object
                                    {'track_id': 0, 'name': 'car', 'class':2, 'bbox': [x0, y0, x1, y1]}, # Obj 0
                                    {...}, # Obj 1
                                ]
                        }, # Frame 0

                        {...}, # Frame 1
                    ]
        """

        # Plot final bboxes and save to file
        out_json_path = os.path.join(labels_out_dir, f"{video_name}.json")
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, 'w') as json_file:
            json.dump(out_data, json_file, indent=1)

        og_frames = sorted(os.listdir(video_dir))
        out_bbox_path = os.path.join(bboxes_out_dir, video_name)
        os.makedirs(out_bbox_path, exist_ok=True)
        for frame_idx, frame_data in enumerate(out_data):
            plt.figure(figsize=(9, 6))
            plt.axis("off")
            img = Image.open(os.path.join(video_dir, og_frames[frame_idx]))
            plt.imshow(img)
            for obj in frame_data["labels"]:
                color = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[obj["class"]])) / 255.0
                show_box(obj["box"], plt.gca(), label=str(obj["track_id"]), color=color)
            
            frame_id_name = og_frames[frame_idx].split("_")[-1].split(".")[0]
            plt.savefig(os.path.join(out_bbox_path, f"bboxes_frame_{frame_id_name}"))

        # Save videos of bboxes
        videos_out_dir = os.path.join(output_dir, "videos")
        out_video_path = os.path.join(videos_out_dir, f"{video_name}_with_bboxes.mp4")
        os.makedirs(os.path.dirname(out_video_path), exist_ok=True)
        create_video_from_images(out_bbox_path, out_video_path, out_fps=12)
        


