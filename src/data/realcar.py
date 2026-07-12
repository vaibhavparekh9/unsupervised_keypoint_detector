"""3DRealCar dataset: frame records, view-pair sampling, dense correspondence.

Pair sampling follows the thesis convention: pairs of views of the same car
separated by 30-80 degrees of camera rotation (geodesic angle between ARKit
camera rotations). Every kept frame in the dataset carries a depth map
(verified across the corpus), so any frame can serve as the source view.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from . import geometry as geo

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FrameRecord:
    __slots__ = ("car_id", "car_dir", "name", "K_raw", "T", "depth_path",
                 "img_path", "_depth16")

    def __init__(self, car_id, car_dir, name, meta):
        self.car_id = car_id
        self.car_dir = car_dir
        self.name = name
        self.K_raw = geo.parse_intrinsics(meta)
        self.T = geo.parse_pose(meta)
        self.img_path = os.path.join(car_dir, name + ".jpg")
        self.depth_path = os.path.join(car_dir, "depth_" + name.split("_")[-1] + ".png")
        self._depth16 = None

    def depth16(self):
        """Cached 16x16 depth for cheap pair-quality estimates."""
        if self._depth16 is None:
            self._depth16 = geo.load_depth(self.depth_path, 16)
        return self._depth16


def scan_car(root, car_id, motion_quality_thresh=0.5, max_frames=None):
    """Return list of FrameRecord for one car (pose present, quality ok,
    image + depth files present)."""
    car_dir = os.path.join(root, car_id)
    frame_jsons = sorted(glob.glob(os.path.join(car_dir, "frame_*.json")))
    frame_jsons = [f for f in frame_jsons
                   if "annotation" not in os.path.basename(f).lower()]
    records = []
    for jf in frame_jsons:
        try:
            meta = geo.load_frame_meta(jf)
        except (OSError, ValueError):
            continue
        if "cameraPoseARFrame" not in meta or "intrinsics" not in meta:
            continue
        if meta.get("motionQuality", 1.0) < motion_quality_thresh:
            continue
        name = os.path.splitext(os.path.basename(jf))[0]
        rec = FrameRecord(car_id, car_dir, name, meta)
        if os.path.exists(rec.img_path) and os.path.exists(rec.depth_path):
            records.append(rec)
    if max_frames is not None and len(records) > max_frames:
        step = len(records) / max_frames
        records = [records[int(i * step)] for i in range(max_frames)]
    return records


def pairwise_angles(records):
    """(N, N) matrix of geodesic rotation angles in degrees."""
    R = np.stack([r.T[:3, :3] for r in records], axis=0)
    traces = np.einsum("nij,mij->nm", R, R)
    cos_t = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def pairwise_azimuth_diffs(records):
    """(N, N) matrix of horizontal viewing-azimuth separations in degrees.

    This is the thesis pairing criterion. The rotation-geodesic angle alone
    is NOT sufficient: phone pitch/roll during capture lets opposite-side
    view pairs (~180 deg apart around the car) land in the 30-80 deg
    rotation band, producing near-zero-visibility pairs.
    """
    az = np.array([geo.view_azimuth_deg(r.T) for r in records])
    d = np.abs(az[:, None] - az[None, :]) % 360.0
    return np.where(d <= 180.0, d, 360.0 - d)


class RealCarPairs(Dataset):
    """Same-car view pairs with dense reprojection correspondence.

    Each item:
        im_a, im_b   : (3, S, S) normalized RGB   (if return_images)
        grid         : (G, G, 2) normalized A->B correspondence
        mask         : (G, G) visibility mask
        R_rel        : (3, 3) relative camera rotation R_a^T R_b
        angle_deg    : scalar rotation separation
        car_idx      : integer id of the car within this dataset
        car, frame_a, frame_b : identifiers
    """

    def __init__(self, root, car_ids, input_res=518, grid_res=64,
                 theta_min=30.0, theta_max=80.0, occlusion_thresh=0.05,
                 depth_scale=0.001, motion_quality_thresh=0.5,
                 max_frames_per_car=None, max_pairs_per_car=200,
                 min_valid_ratio=0.05, return_images=True,
                 feature_cache_dir=None, seed=0):
        self.root = root
        self.input_res = input_res
        self.grid_res = grid_res
        self.occlusion_thresh = occlusion_thresh
        self.depth_scale = depth_scale
        self.return_images = return_images
        self.feature_cache_dir = feature_cache_dir
        self.max_pairs_per_car = max_pairs_per_car
        self.min_valid_ratio = min_valid_ratio
        self.tx = image_transform()

        self.cars = {}        # car_id -> list[FrameRecord]
        self.car_index = {}   # car_id -> int
        self._pair_pool = []  # (car_id, [(i, j, angle), ...])
        for car_id in car_ids:
            recs = scan_car(root, car_id, motion_quality_thresh, max_frames_per_car)
            if feature_cache_dir is not None:
                # only frames present in the cache can be used for training
                recs = [r for r in recs if os.path.exists(
                    os.path.join(feature_cache_dir, car_id, r.name + ".npy"))]
            if len(recs) < 2:
                continue
            azd = pairwise_azimuth_diffs(recs)
            iu, ju = np.triu_indices(len(recs), k=1)
            sel = (azd[iu, ju] >= theta_min) & (azd[iu, ju] <= theta_max)
            pool = [(int(i), int(j), float(azd[i, j]))
                    for i, j in zip(iu[sel], ju[sel])]
            if not pool:
                continue
            self.car_index[car_id] = len(self.cars)
            self.cars[car_id] = recs
            self._pair_pool.append((car_id, pool))

        self.pairs = []
        self.resample_pairs(seed)

    def _visfrac16(self, rec_a, rec_b):
        """Cheap visible-fraction estimate at 16x16."""
        K_a = geo.intrinsics_for_crop({"intrinsics": rec_a.K_raw.ravel().tolist()}, 16)
        K_b = geo.intrinsics_for_crop({"intrinsics": rec_b.K_raw.ravel().tolist()}, 16)
        _, mask = geo.compute_grid_and_mask(
            rec_a.depth16(), K_a, rec_a.T, rec_b.depth16(), K_b, rec_b.T,
            H=16, W=16, occlusion_thresh=self.occlusion_thresh)
        return float(mask.mean())

    def resample_pairs(self, seed=0):
        """Redraw up to max_pairs_per_car pairs per car (call per epoch).

        Candidates are screened with a cheap 16x16 visibility estimate:
        temporally distant frames (different loops of the same walkaround)
        accumulate ARKit odometry drift, and the occlusion check then leaves
        near-zero visible overlap — such pairs are dropped rather than fed to
        the loss (min_valid_ratio, thesis convention).
        """
        rng = np.random.default_rng(seed)
        self.pairs = []
        for car_id, pool in self._pair_pool:
            n_want = min(self.max_pairs_per_car, len(pool))
            n_draw = min(3 * n_want, len(pool))
            idx = rng.choice(len(pool), n_draw, replace=False)
            kept = 0
            for k in idx:
                if kept >= n_want:
                    break
                i, j, ang = pool[k]
                recs = self.cars[car_id]
                if self._visfrac16(recs[i], recs[j]) < self.min_valid_ratio:
                    continue
                self.pairs.append((car_id, i, j, ang))
                kept += 1

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, rec):
        img = Image.open(rec.img_path).convert("RGB")
        return self.tx(geo.prepare_image(img, self.input_res))

    def _load_feat(self, rec):
        path = os.path.join(self.feature_cache_dir, rec.car_id, rec.name + ".npy")
        return torch.from_numpy(np.load(path)).float()

    def compute_pair_grid(self, rec_a, rec_b, grid_res=None):
        G = grid_res or self.grid_res
        K_a = geo.intrinsics_for_crop({"intrinsics": rec_a.K_raw.ravel().tolist()}, G)
        K_b = geo.intrinsics_for_crop({"intrinsics": rec_b.K_raw.ravel().tolist()}, G)
        d_a = geo.load_depth(rec_a.depth_path, G, self.depth_scale)
        d_b = geo.load_depth(rec_b.depth_path, G, self.depth_scale)
        return geo.compute_grid_and_mask(
            d_a, K_a, rec_a.T, d_b, K_b, rec_b.T,
            H=G, W=G, occlusion_thresh=self.occlusion_thresh)

    def __getitem__(self, index):
        car_id, i, j, ang = self.pairs[index]
        rec_a, rec_b = self.cars[car_id][i], self.cars[car_id][j]

        grid, mask = self.compute_pair_grid(rec_a, rec_b)
        out = {
            "grid": grid,
            "mask": mask,
            "R_rel": torch.from_numpy(
                geo.relative_rotation(rec_a.T, rec_b.T).astype(np.float32)),
            "rel_azimuth": torch.tensor(np.deg2rad(
                (geo.view_azimuth_deg(rec_a.T) - geo.view_azimuth_deg(rec_b.T)
                 + 180.0) % 360.0 - 180.0), dtype=torch.float32),
            "angle_deg": torch.tensor(ang, dtype=torch.float32),
            "car_idx": torch.tensor(self.car_index[car_id], dtype=torch.long),
            "car": car_id,
            "frame_a": rec_a.name,
            "frame_b": rec_b.name,
        }
        if self.return_images:
            out["im_a"] = self._load_image(rec_a)
            out["im_b"] = self._load_image(rec_b)
        if self.feature_cache_dir is not None:
            out["feat_a"] = self._load_feat(rec_a)
            out["feat_b"] = self._load_feat(rec_b)
        return out


class RealCarFrames(Dataset):
    """Single frames (for feature caching and probes)."""

    def __init__(self, root, car_ids, input_res=518,
                 motion_quality_thresh=0.5, max_frames_per_car=None):
        self.input_res = input_res
        self.tx = image_transform()
        self.records = []
        for car_id in car_ids:
            self.records.extend(
                scan_car(root, car_id, motion_quality_thresh, max_frames_per_car))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        img = Image.open(rec.img_path).convert("RGB")
        return {
            "image": self.tx(geo.prepare_image(img, self.input_res)),
            "car": rec.car_id,
            "frame": rec.name,
        }
