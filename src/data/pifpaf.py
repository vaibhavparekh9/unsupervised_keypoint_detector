"""OpenPifPaf 24-keypoint vehicle pseudo-label loading.

EVALUATION / DIAGNOSTICS ONLY — never a training signal.

Keypoint ordering follows the OpenPifPaf ApolloCar3D plugin's
CAR_KEYPOINTS_24 convention (openpifpaf/plugins/apollocar3d). The labels in
/home/vaibhav/3DRealCars-Labels were produced by that model. The left/right
pair structure below is additionally sanity-checked geometrically in the S1
probe (mirror pairs must be roughly mirror-symmetric in 3D).

Thesis convention: keep only frames with exactly one detection.
"""

import json
import os
import numpy as np

NUM_KEYPOINTS = 24

KEYPOINT_NAMES = [
    "front_up_right",     # 0
    "front_up_left",      # 1
    "front_light_right",  # 2
    "front_light_left",   # 3
    "front_low_right",    # 4
    "front_low_left",     # 5
    "central_up_left",    # 6
    "front_wheel_left",   # 7
    "rear_wheel_left",    # 8
    "rear_corner_left",   # 9
    "rear_up_left",       # 10
    "rear_up_right",      # 11
    "rear_light_left",    # 12
    "rear_light_right",   # 13
    "rear_low_left",      # 14
    "rear_low_right",     # 15
    "central_up_right",   # 16
    "rear_corner_right",  # 17
    "rear_wheel_right",   # 18
    "front_wheel_right",  # 19
    "rear_plate_left",    # 20
    "rear_plate_right",   # 21
    "mirror_edge_left",   # 22
    "mirror_edge_right",  # 23
]

# (left_idx, right_idx) mirror pairs across the car's sagittal plane
LEFT_RIGHT_PAIRS = [
    (1, 0),    # front_up
    (3, 2),    # front_light
    (5, 4),    # front_low
    (6, 16),   # central_up
    (7, 19),   # front_wheel
    (8, 18),   # rear_wheel
    (9, 17),   # rear_corner
    (10, 11),  # rear_up
    (12, 13),  # rear_light
    (14, 15),  # rear_low
    (20, 21),  # rear_plate
    (22, 23),  # mirror_edge
]

# (front_idx, rear_idx) same-side front/rear analogues
FRONT_REAR_PAIRS = [
    (0, 11),   # up_right
    (1, 10),   # up_left
    (2, 13),   # light_right
    (3, 12),   # light_left
    (4, 15),   # low_right
    (5, 14),   # low_left
    (7, 8),    # wheel_left
    (19, 18),  # wheel_right
]


def mirror_partner():
    """Return array m where m[i] = index of i's left/right mirror twin
    (or -1 for unpaired — none in this 24-kp set)."""
    m = np.full(NUM_KEYPOINTS, -1, dtype=np.int64)
    for l, r in LEFT_RIGHT_PAIRS:
        m[l], m[r] = r, l
    return m


def front_rear_partner():
    m = np.full(NUM_KEYPOINTS, -1, dtype=np.int64)
    for f, r in FRONT_REAR_PAIRS:
        m[f], m[r] = r, f
    return m


def label_path(labels_root, car_id, frame_name):
    return os.path.join(labels_root, car_id, "labels", frame_name + "_pifpaf.json")


def load_keypoints(labels_root, car_id, frame_name, min_conf=0.0):
    """Load PifPaf keypoints for one frame (exactly-one-detection filter).

    Returns (kps (24,2) float32 in 1920x1440 space, conf (24,)) or None if
    the frame has zero or multiple detections / no label file.
    """
    path = label_path(labels_root, car_id, frame_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    n = data.get("num_annotations", len(anns))
    if n != 1:
        return None
    kps = np.array(anns[0]["keypoints"], dtype=np.float32).reshape(-1, 3)
    if kps.shape[0] != NUM_KEYPOINTS:
        return None
    conf = kps[:, 2].copy()
    if min_conf > 0:
        conf[conf < min_conf] = 0.0
    return kps[:, :2].copy(), conf
