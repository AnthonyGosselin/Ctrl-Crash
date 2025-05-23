import cv2
import numpy as np
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
    TYPE_LOOKUP = [BROWN, BLUE, PURPLE, RED, ORANGE, YELLOW, LIGHTPINK, LIGHTPURPLE, GRAY, LIGHTRED, GREEN]
    REVERT_CHANNEL_F = lambda x: (x[2], x[1], x[0])


# TODO: This could be moved to base dataset class (?)
def plot_2d_bbox(img, labels, show_track_color=False, channel_first=True, rgb2bgr=False, box_color=None, masked_track_ids=None, crash_border=False):

    if channel_first:
        img = img.permute((1, 2, 0)).detach().cpu().numpy().copy()*255
    else:
        img = img.detach().cpu().numpy().copy()*255
    
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    masked_track_ids = masked_track_ids or []
    
    for i, label_info in enumerate(labels):
        track_id = label_info['track_id']
        if track_id in masked_track_ids:
            continue
        
        box_2d = label_info['bbox']

        if not show_track_color:
            type_color_i = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[label_info['class_id']])) / 255 if box_color is None else box_color
            track_color_i = CVCOLORS.REVERT_CHANNEL_F((1, 1, 1))

            cv2.rectangle(img, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), type_color_i, cv2.FILLED)
            cv2.rectangle(img, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), track_color_i, 2)
        else:
            type_color_i = np.array(CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TYPE_LOOKUP[label_info['class_id']])) / 255 if box_color is None else box_color
            track_color_i = CVCOLORS.REVERT_CHANNEL_F(CVCOLORS.TRACKID_LOOKUP[label_info['track_id']])

            dim = min(box_2d[2] - box_2d[0], box_2d[3] - box_2d[1])
            b_thick = min(max(dim * 0.1, 2), 8)
            cv2.rectangle(img, (int(box_2d[0]), int(box_2d[1])), (int(box_2d[2]), int(box_2d[3])), type_color_i, cv2.FILLED)
            cv2.rectangle(img, (int(box_2d[0] + b_thick), int(box_2d[1] + b_thick)), (int(box_2d[2] - b_thick), int(box_2d[3] - b_thick)), track_color_i, cv2.FILLED)
    
    if crash_border:
        thickness = 20
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color=(0, 1, 0), thickness=thickness, lineType=cv2.LINE_8)

    return img