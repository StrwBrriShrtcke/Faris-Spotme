import typing

import cv2
import numpy as np

# ---------------- Pose drawing ----------------
SKELETON = [
    (5, 6),  # shoulders
    (5, 7),  # left shoulder to left elbow
    (7, 9),  # left elbow to left wrist
    (6, 8),  # right shoulder to right elbow
    (8, 10),  # right elbow to right wrist
]

SKELETON_ab = [
    (5, 6),  # shoulders
    (6, 12),  # r shoulder to r hip
    (5, 11),  # l shoulder to l hip
    (11, 12),  # hips
    (11, 13),  # l hip to l knee
    (12, 14),  # r hip to r knee
]

SKELETON_legs = [
    (11, 12),  # hips
    (11, 13),  # l hip to l knee
    (12, 14),  # r hip to r knee
    (13, 15),  # l knee to l ankle
    (14, 16),  # r knee to r ankle
]

keypoints_exercises = {
    "arm": ((5, 7, 9), (6, 8, 10)),  # left arm kpts, right arm kpts
    "hip": ((5, 11, 13), (6, 12, 14)),  # left hip kpts, right hip kpts
    "leg": ((11, 13, 15), (12, 14, 16)),  # left leg kpts, right leg kpts
}

exercises_what_kpts = {
    "bicep curl": keypoints_exercises["arm"],
    "tricep press": keypoints_exercises["arm"],
    "rows": keypoints_exercises["arm"],
    "ab crunch": keypoints_exercises["hip"],
    "leg curl": keypoints_exercises["leg"],
    "chest press": keypoints_exercises["arm"],
}

exercises_what_skeleton = {
    "bicep curl": SKELETON,
    "tricep press": SKELETON,
    "rows": SKELETON,
    "ab crunch": SKELETON_ab,
    "leg curl": SKELETON_legs,
    "chest press": SKELETON,
}

# data for form correction
form_data = {
    "bicep curl": {"min": 27, "max": 170},  # arms kpts
    "tricep press": {"min": 85, "max": 170},  # arms kpts
    "rows": {"min": 85, "max": 170},  # arms kpts
    "ab crunch": {"min": 40, "max": 150},  # shoulder, hip, knee
    "leg curl": {"min": 30, "max": 130},  # leg kpts
    "chest press": {"min": 90, "max": 170},  # arms kpts
}


def norm_exercise(exercise: str) -> str:
    return (exercise or "").strip().lower()


# checking if all 3 keypoints for an arm exist
def check_arm_kp_exist(kconf, idxs, th=0.8):
    if kconf is None:
        return True  # if model doesn't provide conf, just try
    return all(float(kconf[i]) >= th for i in idxs)


def get_left_kp(kxy, exercise):
    exercise = norm_exercise(exercise)
    left_kp = exercises_what_kpts[exercise][0]
    left_kpt1_x, left_kpt1_y = kxy[left_kp[0]]
    left_kpt2_x, left_kpt2_y = kxy[left_kp[1]]
    left_kpt3_x, left_kpt3_y = kxy[left_kp[2]]
    return (
        left_kpt1_x,
        left_kpt1_y,
        left_kpt2_x,
        left_kpt2_y,
        left_kpt3_x,
        left_kpt3_y,
    )


def get_right_kp(kxy, exercise):
    exercise = norm_exercise(exercise)
    right_kp = exercises_what_kpts[exercise][1]
    right_kpt1_x, right_kpt1_y = kxy[right_kp[0]]
    right_kpt2_x, right_kpt2_y = kxy[right_kp[1]]
    right_kpt3_x, right_kpt3_y = kxy[right_kp[2]]
    return (
        right_kpt1_x,
        right_kpt1_y,
        right_kpt2_x,
        right_kpt2_y,
        right_kpt3_x,
        right_kpt3_y,
    )


# form correction feedbackmaxxer
def form_corrector(form_data: dict, angle: float, exercise: str) -> str:
    exercise = norm_exercise(exercise)
    good_text = "You are doing great! Keep it up!"
    if exercise not in form_data:
        return "Exercise not in system"

    if form_data[exercise]["min"] + 10 < angle < form_data[exercise]["min"] + 20:
        return "Your arm is not contracted enough. Keep going!"

    if form_data[exercise]["max"] - 20 < angle < form_data[exercise]["max"] - 10:
        return "Your arm is not extended enough. Keep going!"

    return good_text


def draw_pose(
    out: np.ndarray,
    kpts_xy: np.ndarray,
    exercise: str,
    kpts_conf: typing.Optional[np.ndarray] = None,
    conf_thres: float = 0.3,
) -> None:
    exercise = norm_exercise(exercise)
    skeleton_draw = exercises_what_skeleton[exercise]

    for a, b in skeleton_draw:
        xa, ya = kpts_xy[a]
        xb, yb = kpts_xy[b]
        if kpts_conf is not None and (
            kpts_conf[a] < conf_thres or kpts_conf[b] < conf_thres
        ):
            continue
        if xa > 0 and ya > 0 and xb > 0 and yb > 0:
            cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)

    for idx, (x, y) in enumerate(kpts_xy):
        if x <= 0 or y <= 0:
            continue
        if kpts_conf is not None and kpts_conf[idx] < conf_thres:
            continue
        cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
