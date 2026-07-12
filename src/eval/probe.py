"""Shared probe machinery: PifPaf-keypoint NN-matching PCK + symmetry
confusion, over any dense-descriptor provider (frozen backbone or trained
model). Used by gate S1 (baseline/motivation) and gate S3 (after training).
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

from ..data import geometry as geo
from ..data import pifpaf
from ..data.realcar import scan_car

FEATURE_RES = 64
AZ_BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180)]
KP_GROUPS = {
    "wheels": [7, 8, 18, 19],
    "lights": [2, 3, 12, 13],
    "mirrors": [22, 23],
}


def load_labeled_frames(image_root, labels_root, car_ids, input_res, cache_root):
    """Per car: frames with exactly-one PifPaf detection AND cached features."""
    out = {}
    for car_id in car_ids:
        frames = []
        for rec in scan_car(image_root, car_id):
            feat_path = os.path.join(cache_root, car_id, rec.name + ".npy")
            if not os.path.exists(feat_path):
                continue
            lab = pifpaf.load_keypoints(labels_root, car_id, rec.name)
            if lab is None:
                continue
            kps_orig, conf = lab
            kps = geo.points_to_crop(kps_orig, input_res)
            in_b = ((kps[:, 0] >= 0) & (kps[:, 0] < input_res) &
                    (kps[:, 1] >= 0) & (kps[:, 1] < input_res))
            visible = (conf > 0.5) & in_b
            if visible.sum() < 2:
                continue
            frames.append({
                "rec": rec, "kps": kps.astype(np.float32), "visible": visible,
                "feat_path": feat_path,
                "azimuth": geo.view_azimuth_deg(rec.T),
            })
        if len(frames) >= 2:
            out[car_id] = frames
    return out


class BackboneDescriptors:
    """Cached fp16 tokens -> bilinear-upsampled L2-normalized descriptors."""

    def __init__(self, max_cache=400):
        self._cache = {}
        self.max_cache = max_cache

    def __call__(self, frame):
        p = frame["feat_path"]
        if p in self._cache:
            return self._cache[p]
        tok = torch.from_numpy(np.load(p)).float()
        fm = tok.permute(2, 0, 1).unsqueeze(0)
        fm = F.interpolate(fm, size=(FEATURE_RES, FEATURE_RES),
                           mode="bilinear", align_corners=True)
        fm = F.normalize(fm.squeeze(0), p=2, dim=0)
        if len(self._cache) < self.max_cache:
            self._cache[p] = fm
        return fm


class ModelDescriptors:
    """Trained head on cached tokens -> L2-normalized descriptor maps."""

    def __init__(self, model, device="cpu", max_cache=400):
        self.model = model.eval()
        self.device = device
        self._cache = {}
        self.max_cache = max_cache

    @torch.no_grad()
    def __call__(self, frame):
        p = frame["feat_path"]
        if p in self._cache:
            return self._cache[p]
        tok = torch.from_numpy(np.load(p)).float().unsqueeze(0).to(self.device)
        desc = self.model(tok)["desc"].squeeze(0)
        if desc.shape[-1] != FEATURE_RES:
            desc = F.interpolate(desc.unsqueeze(0), size=(FEATURE_RES, FEATURE_RES),
                                 mode="bilinear", align_corners=True).squeeze(0)
        fm = F.normalize(desc.cpu(), p=2, dim=0)
        if len(self._cache) < self.max_cache:
            self._cache[p] = fm
        return fm


def sample_at(fm, pts, input_res):
    g = torch.from_numpy(
        2.0 * pts / (input_res - 1) - 1.0).float().reshape(1, 1, -1, 2)
    s = F.grid_sample(fm.unsqueeze(0), g, mode="bilinear",
                      align_corners=True, padding_mode="border")
    return s.squeeze(0).squeeze(1).t()


def match_pair(provider, fa, fb, input_res):
    common = fa["visible"] & fb["visible"]
    idx = np.where(common)[0]
    if len(idx) == 0:
        return []
    fm_a = provider(fa)
    fm_b = provider(fb)

    src = F.normalize(sample_at(fm_a, fa["kps"][idx], input_res), p=2, dim=1)
    tgt = fm_b.reshape(fm_b.shape[0], -1)
    nn = (src @ tgt).argmax(dim=1)
    pred = torch.stack([(nn % FEATURE_RES).float(),
                        (nn // FEATURE_RES).float()], dim=1).numpy()

    scale = FEATURE_RES / input_res
    gt = fb["kps"] * scale
    mirror = pifpaf.mirror_partner()
    out = []
    for n, k in enumerate(idx):
        e = float(np.linalg.norm(pred[n] - gt[k]))
        entry = {"kp": int(k), "err_feat": e}
        m = mirror[k]
        if m >= 0 and fb["visible"][m]:
            d_twin = float(np.linalg.norm(pred[n] - gt[m]))
            d_pair = float(np.linalg.norm(gt[k] - gt[m]))
            if d_pair > 6.0:  # twins must be separated for confusion to be defined
                entry["confused"] = bool(d_twin < e)
        out.append(entry)
    return out


def sample_pairs(car_frames, num, rng, cross=False):
    entries = [(c, i) for c, fr in car_frames.items() for i in range(len(fr))]
    pairs, seen = [], set()
    for _ in range(num * 60):
        if len(pairs) >= num:
            break
        a, b = rng.choice(len(entries), 2, replace=False)
        (ca, ia), (cb, ib) = entries[a], entries[b]
        if cross and ca == cb:
            continue
        if not cross and (ca != cb or ia == ib):
            continue
        key = ((ca, ia), (cb, ib))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((ca, ia, cb, ib))
    return pairs


def run_probe(provider, car_frames, input_res, num_intra=250, num_cross=120,
              seed=0):
    """Returns (pck_results, confusion_curve, group_rates, confusion_records)."""
    rng = np.random.default_rng(seed)
    intra = sample_pairs(car_frames, num_intra, rng, cross=False)
    cross = sample_pairs(car_frames, num_cross, rng, cross=True)

    results = {}
    confusion_records = []
    for name, pairs in [("intra", intra), ("cross", cross)]:
        errs = []
        for ca, ia, cb, ib in pairs:
            fa, fb = car_frames[ca][ia], car_frames[cb][ib]
            for m in match_pair(provider, fa, fb, input_res):
                errs.append(m["err_feat"])
                if name == "intra" and "confused" in m:
                    az = geo.azimuth_diff_deg(fa["azimuth"], fb["azimuth"])
                    confusion_records.append((az, m["confused"], m["kp"]))
        errs = np.array(errs)
        results[name] = {
            "num_pairs": len(pairs),
            "num_kps": len(errs),
            "pck@10_feat64": float((errs < 10).mean() * 100) if len(errs) else 0.0,
            "pck@5_feat64": float((errs < 5).mean() * 100) if len(errs) else 0.0,
            "median_err_feat64": float(np.median(errs)) if len(errs) else -1,
        }

    conf_curve = {}
    for lo, hi in AZ_BINS:
        sel = [c for az, c, _ in confusion_records if lo <= az < hi]
        conf_curve[f"{lo}-{hi}"] = {
            "n": len(sel),
            "confusion_rate": float(np.mean(sel)) if sel else None,
        }
    group_rates = {}
    for gname, gidx in KP_GROUPS.items():
        sel = [c for _, c, k in confusion_records if k in gidx]
        group_rates[gname] = float(np.mean(sel)) if sel else None
    return results, conf_curve, group_rates, confusion_records
