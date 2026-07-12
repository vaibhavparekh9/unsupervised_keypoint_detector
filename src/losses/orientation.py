"""Relative-orientation consistency loss.

Only RELATIVE camera rotations (ARKit metadata, not human annotation)
supervise the orientation head; the canonical frame itself emerges. If the
model predicts R_x: canonical -> camera_x, then for a view pair (A, B):

    R_pred(A) @ R_pred(B)^T  ==  R_camA^T @ R_camB  (= meta R_rel)
"""

import torch

from ..models.rotation import geodesic_distance, wrap_angle


def relative_orientation_loss(R_a, R_b, R_rel):
    """Geodesic loss between predicted and metadata relative rotation."""
    pred_rel = torch.matmul(R_a, R_b.transpose(-1, -2))
    return geodesic_distance(pred_rel, R_rel.to(R_a.device)).mean()


def relative_azimuth_loss(azim_a, azim_b, rel_azimuth):
    """S1 ablation variant: signed azimuth differences must match."""
    d = wrap_angle(azim_a - azim_b - rel_azimuth.to(azim_a.device))
    return d.abs().mean()
