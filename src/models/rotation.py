"""Rotation parameterizations and geodesic distances."""

import torch


def rotation_6d_to_matrix(x):
    """Zhou et al. continuous 6D -> (B, 3, 3) rotation via Gram-Schmidt."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(
        a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def geodesic_distance(R_a, R_b, eps=1e-7):
    """Angle (radians) between rotation matrices, batched."""
    m = torch.matmul(R_a, R_b.transpose(-1, -2))
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos)


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))
