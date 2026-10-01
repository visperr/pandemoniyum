import time
import numpy as np


def compute_iou(box1_ltrb, box2_ltrb):
    x1 = max(box1_ltrb[0], box2_ltrb[0])
    y1 = max(box1_ltrb[1], box2_ltrb[1])
    x2 = min(box1_ltrb[2], box2_ltrb[2])
    y2 = min(box1_ltrb[3], box2_ltrb[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1_ltrb[2] - box1_ltrb[0]) * (box1_ltrb[3] - box1_ltrb[1])
    area2 = (box2_ltrb[2] - box2_ltrb[0]) * (box2_ltrb[3] - box2_ltrb[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b))


def get_gallery_similarity(feature, feature_gallery):
    if not feature_gallery:
        return 0.0
    sims = [cosine_similarity(feature, f) for f in feature_gallery]
    return sum(sims) / len(sims)


def get_milliseconds_since_epoch():
    return time.time_ns() // 1_000_000


def player_color_from_id(player_id) -> tuple[int, int, int]:
    if player_id < 1:
        player_id = 1

    colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
    return colors[(player_id - 1) % len(colors)]
