"""Camera geometry for 3DRealCar walkaround captures.

Ported from the thesis codebase (`DVE_multiview/utils/correspondence.py`) —
that math is known-good (validated by the thesis PCK numbers), so the
conventions here are kept identical:

Convention (matches the DVE loss grid format):
    grid[v, u] = (norm_u_tgt, norm_v_tgt) in [-1, 1]
    valid_mask[v, u] = 1.0 if pixel (u, v) in source has a visible
                       correspondent in target, else 0.0.

Camera transform convention:
    cameraPoseARFrame is a row-major 4x4 camera-to-world transform in the
    ARKit camera frame (X right, Y up, Z toward viewer). Image unprojection
    uses OpenCV convention (X right, Y down, Z into scene). The flip matrix
    F = diag(1, -1, -1) converts between them.
"""

import json
import numpy as np
import torch
from PIL import Image

ORIG_W, ORIG_H = 1920, 1440  # RGB capture resolution (uniform across dataset)


# ---------------------------------------------------------------------------
# Frame JSON parsing
# ---------------------------------------------------------------------------

def load_frame_meta(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def parse_intrinsics(json_data):
    """3x3 K from the flat row-major 9-element `intrinsics` list."""
    return np.array(json_data["intrinsics"], dtype=np.float64).reshape(3, 3)


def parse_pose(json_data):
    """4x4 camera-to-world transform from `cameraPoseARFrame` (row-major)."""
    return np.array(json_data["cameraPoseARFrame"], dtype=np.float64).reshape(4, 4)


def scale_intrinsics(K, scale_x, scale_y):
    K = K.copy()
    K[0, :] *= scale_x
    K[1, :] *= scale_y
    return K


def crop_intrinsics(K, crop_left, crop_top):
    K = K.copy()
    K[0, 2] -= crop_left
    K[1, 2] -= crop_top
    return K


# ---------------------------------------------------------------------------
# Square-crop working space
# ---------------------------------------------------------------------------
# All images are processed as: resize 1920x1440 -> (4/3*S x S), then center
# crop to S x S. This matches the thesis preprocessing so PCK protocols are
# directly comparable.

def crop_geometry(size):
    """Return (resize_w, resize_h, crop_left, crop_top) for square size S."""
    resize_w = int(round(size * 4 / 3))
    resize_h = size
    crop_left = (resize_w - size) // 2
    crop_top = (resize_h - size) // 2
    return resize_w, resize_h, crop_left, crop_top


def intrinsics_for_crop(meta, size):
    """K in the S x S square-crop space."""
    K = parse_intrinsics(meta)
    resize_w, resize_h, crop_left, crop_top = crop_geometry(size)
    K = scale_intrinsics(K, resize_w / ORIG_W, resize_h / ORIG_H)
    return crop_intrinsics(K, crop_left, crop_top)


def points_to_crop(pts_orig, size):
    """Map (N,2) pixel coords from 1920x1440 space into S x S crop space."""
    resize_w, resize_h, crop_left, crop_top = crop_geometry(size)
    out = np.empty_like(pts_orig, dtype=np.float64)
    out[:, 0] = pts_orig[:, 0] * resize_w / ORIG_W - crop_left
    out[:, 1] = pts_orig[:, 1] * resize_h / ORIG_H - crop_top
    return out


def prepare_image(img_pil, size):
    """Resize + center-crop a PIL image into the S x S working space."""
    resize_w, resize_h, crop_left, crop_top = crop_geometry(size)
    img = img_pil.resize((resize_w, resize_h), Image.BILINEAR)
    return img.crop((crop_left, crop_top, crop_left + size, crop_top + size))


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def load_depth(depth_path, size, depth_scale=0.001):
    """Load a depth PNG (int32, millimetres, 256x192) into S x S crop space.

    Nearest-neighbour resize (never interpolate depth across edges), then the
    same center crop as the RGB pipeline. Returns float32 metres; 0 = invalid.
    """
    arr = np.asarray(Image.open(depth_path), dtype=np.float32)
    resize_w, resize_h, crop_left, crop_top = crop_geometry(size)
    depth_pil = Image.fromarray(arr)
    depth_pil = depth_pil.resize((resize_w, resize_h), Image.NEAREST)
    depth = np.asarray(depth_pil, dtype=np.float32) * depth_scale
    return depth[crop_top:crop_top + size, crop_left:crop_left + size]


def depth_at_points(depth_path, pts_orig, depth_scale=0.001, robust=False):
    """Depth (metres) at (N,2) points given in 1920x1440 pixel coords, read
    from the raw 256x192 map. 0 = invalid.

    robust=True: foreground-biased lookup — 25th percentile of valid depths
    in a 5x5 depth-pixel window. Keypoints sit on the object silhouette where
    a nearest-neighbour lookup often lands on the background; biasing toward
    the near surface recovers the object depth.
    """
    arr = np.asarray(Image.open(depth_path), dtype=np.float32)
    dh, dw = arr.shape
    u = np.clip(np.round(pts_orig[:, 0] * dw / ORIG_W).astype(np.int64), 0, dw - 1)
    v = np.clip(np.round(pts_orig[:, 1] * dh / ORIG_H).astype(np.int64), 0, dh - 1)
    if not robust:
        return arr[v, u] * depth_scale
    out = np.zeros(len(u), dtype=np.float32)
    for k in range(len(u)):
        win = arr[max(v[k] - 2, 0):v[k] + 3, max(u[k] - 2, 0):u[k] + 3]
        vals = win[win > 1e-6]
        if len(vals):
            out[k] = np.percentile(vals, 25)
    return out * depth_scale


# ---------------------------------------------------------------------------
# Core: unproject -> transform -> reproject
# ---------------------------------------------------------------------------

def unproject_to_world(pts, depths, K, T):
    """Unproject (N,2) pixel coords with (N,) depths (m) to (N,3) world points.

    pts/K must be in the same pixel space. T is ARKit camera-to-world.
    """
    N = pts.shape[0]
    K_inv = np.linalg.inv(K)
    uvw = np.concatenate([pts.T, np.ones((1, N))], axis=0)      # (3, N)
    rays = K_inv @ uvw                                           # OpenCV rays
    pts_cv = rays * depths[None, :]
    # OpenCV -> ARKit camera frame: flip Y and Z
    pts_ar = np.stack([pts_cv[0], -pts_cv[1], -pts_cv[2]], axis=0)
    R, t = T[:3, :3], T[:3, 3]
    return (R @ pts_ar + t[:, None]).T                           # (N, 3)


def project_from_world(pts_world, K, T):
    """Project (N,3) world points into a camera. Returns (uv (N,2), z (N,))
    where z is OpenCV depth (positive = in front of camera)."""
    R, t = T[:3, :3], T[:3, 3]
    pts_ar = R.T @ (pts_world.T - t[:, None])                    # (3, N)
    pts_cv = np.stack([pts_ar[0], -pts_ar[1], -pts_ar[2]], axis=0)
    z = pts_cv[2]
    proj = K @ pts_cv
    uv = np.stack([proj[0] / (z + 1e-10), proj[1] / (z + 1e-10)], axis=1)
    return uv, z


def view_azimuth_deg(T):
    """Azimuth (degrees) of the camera viewing direction projected onto the
    horizontal plane. ARKit world is gravity-aligned (Y up); the camera looks
    along -Z of its own frame."""
    d = T[:3, :3] @ np.array([0.0, 0.0, -1.0])
    return float(np.degrees(np.arctan2(d[0], d[2])))


def azimuth_diff_deg(az_a, az_b):
    """Absolute azimuth separation in [0, 180]."""
    d = abs(az_a - az_b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def relative_rotation(T_a, T_b):
    """R_rel = R_camA^T @ R_camB. If the model predicts R_x: canonical->cam_x,
    then R_pred(A) @ R_pred(B)^T should equal this."""
    return T_a[:3, :3].T @ T_b[:3, :3]


def rotation_angle_deg(R_a, R_b):
    """Geodesic angle in degrees between two rotation matrices."""
    tr = np.trace(R_a.T @ R_b)
    return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))


def compute_grid_and_mask(depth_src, K_src, T_src,
                          depth_tgt, K_tgt, T_tgt,
                          H, W, occlusion_thresh=0.05):
    """Dense source->target correspondence grid + visibility mask.

    Ported verbatim (modulo refactor) from the thesis codebase. All inputs
    are in the same H x W pixel space (use `intrinsics_for_crop`/`load_depth`
    at that resolution).

    Returns:
        grid       (torch.Tensor): (H, W, 2) float32, normalized target coords.
        valid_mask (torch.Tensor): (H, W) float32.
    """
    u_range = np.arange(W, dtype=np.float64)
    v_range = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u_range, v_range)

    Z_src = depth_src.astype(np.float64)
    valid_src = (Z_src > 1e-6).ravel()

    K_src_inv = np.linalg.inv(K_src)
    ones = np.ones((H, W), dtype=np.float64)
    uvw = np.stack([uu, vv, ones], axis=0).reshape(3, -1)
    pts_norm = K_src_inv @ uvw
    Z_flat = Z_src.ravel()

    # OpenCV -> ARKit camera frame (flip Y, Z)
    pts_arkit_src = np.stack([
        pts_norm[0] * Z_flat,
        -pts_norm[1] * Z_flat,
        -Z_flat,
    ], axis=0)

    R_src, t_src = T_src[:3, :3], T_src[:3, 3]
    pts_world = R_src @ pts_arkit_src + t_src[:, None]

    R_tgt, t_tgt = T_tgt[:3, :3], T_tgt[:3, 3]
    pts_arkit_tgt = R_tgt.T @ (pts_world - t_tgt[:, None])

    pts_opencv_tgt = pts_arkit_tgt.copy()
    pts_opencv_tgt[1] *= -1
    pts_opencv_tgt[2] *= -1

    Z_tgt = pts_opencv_tgt[2, :]
    valid_pos_depth = Z_tgt > 1e-6

    proj = K_tgt @ pts_opencv_tgt
    u_tgt = proj[0, :] / (Z_tgt + 1e-10)
    v_tgt = proj[1, :] / (Z_tgt + 1e-10)

    valid_bounds = (
        (u_tgt >= 0) & (u_tgt <= W - 1) &
        (v_tgt >= 0) & (v_tgt <= H - 1)
    )

    u_int = np.clip(np.round(u_tgt).astype(np.int32), 0, W - 1)
    v_int = np.clip(np.round(v_tgt).astype(np.int32), 0, H - 1)
    depth_at_tgt = depth_tgt[v_int, u_int]

    # Visible if reprojected depth matches observed depth; target pixels with
    # no measurement get the benefit of the doubt.
    valid_occlude = (
        (np.abs(Z_tgt - depth_at_tgt) < occlusion_thresh) |
        (depth_at_tgt < 1e-6)
    )

    valid = valid_src & valid_pos_depth & valid_bounds & valid_occlude
    valid_2d = valid.reshape(H, W)

    norm_u = (2.0 * u_tgt / max(W - 1, 1) - 1.0).reshape(H, W)
    norm_v = (2.0 * v_tgt / max(H - 1, 1) - 1.0).reshape(H, W)
    norm_u[~valid_2d] = 0.0
    norm_v[~valid_2d] = 0.0

    grid = np.stack([norm_u, norm_v], axis=-1).astype(np.float32)
    valid_mask = valid_2d.astype(np.float32)

    return torch.from_numpy(grid), torch.from_numpy(valid_mask)
