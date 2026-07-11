# Project: Unsupervised 360° Keypoint Discovery (fast-track)

Master's-research continuation toward a publication. Owner: Vaibhav Parekh (vsparekh@andrew.cmu.edu),
advisor Prof. Kenji Shimada (CMU CERLAB). Goal: **original, publishable** unsupervised keypoint discovery
for multi-viewpoint (360°) rigid objects. This repo follows the **fast track**: shortest defensible line
to a submittable result (~6–8 weeks), betting the core mechanism works without intermediate phases.

Headline claim we are building toward: *"First fully unsupervised landmark discovery for 360° rigid
objects — matches or beats weakly-supervised SphericalMaps (CVPR 2024) on its own benchmark without
viewpoint labels."*

## Read these before coding
1. `research_synthesis-1.md` — 2020–2026 literature landscape + novelty audit. Closest rivals:
   SphericalMaps (CVPR'24, needs viewpoint bins), Common3D ('25, template-mesh correspondence, no landmark
   discovery), KeyDiff3D ('25, diffusion prior, limited azimuth), BKinD family (needs articulating agents).
2. `research_direction.md` — full phased plan and design rationale; this CLAUDE.md implements its
   "Fast track" section.
3. `some_more_methods.md` — mirror-ambiguity formalization (`P(I)=P(mirror(I))`): why single-image
   objectives cannot break left/right symmetry; sequences/geometry can. Cite-worthy in the paper.

## Background (user's prior work)
Thesis "Extending Unsupervised Landmark Discovery to Multi-Viewpoint Objects": DVE (Thewlis et al., ICCV
2019, arXiv:1908.06427) with TPS warps replaced by multi-view geometric correspondence (ARKit depth +
poses) + visibility masking, hourglass CNN, on 3DRealCar. Results: 82.8% intra / 69.5% cross PCK@10 at
64×64 feature resolution; beat supervised YOLO-Pose (65.6%) cross-instance. View pairs sampled at 30–80°
azimuth separation; depth-consistency visibility threshold ε=0.05 m; DVE cross-instance exchange retained.
Key thesis ablation: **unmasked** camera correspondences (17.3 intra PCK) are worse than TPS-only (40.4) —
visibility masking is non-negotiable wherever reprojection supervises training.

Old codebase: `/home/vaibhav/DVE_multiview (working)/` (fork of jamt9000/DVE). Port, don't rewrite blind:
- `data_loader/data_loaders.py` — 3DRealCar pair sampling, unprojection/reprojection warp computation.
- `model/loss.py`, `model/folded_correlation_dve.py` — DVE loss machinery (memory-efficient correlation).
- `model/hourglass.py` — CNN reference. `model/keypoint_prediction.py` — regression-eval head.
- `eval_pifpaf.py`, `eval_correspondence_pifpaf.py`, `plot_pck_curve.py` — evaluation against PifPaf labels.
- Configs: `configs-vaibhav/cars-hourglass-64d-dve.json`, `cars-new.json`.
- Thesis PDF + DVE paper PDF are in that folder.

## Data (verified formats)
- `/home/vaibhav/3DRealCars-English/<car_id>/` — ~2585 car folders (`0000`…), each a 360° walkaround:
  - `frame_XXXXX.jpg` — RGB (≈1920×1440, intrinsics suggest landscape 1920×1440).
  - `frame_XXXXX.json` — per-frame metadata: `intrinsics` (flat 3×3, fx≈fy≈1333, cx≈967, cy≈731),
    `cameraPoseARFrame` (flat 4×4 camera-to-world, ARKit convention — verify axis conventions when
    porting; the old data_loader has the working math), `frame_index`, `motionQuality`,
    `averageAngularVelocity`, `projectionMatrix`.
  - `depth_XXXXX.png` — depth for a subset of frames (~every 6th). Verify encoding on first use (likely
    16-bit, mm→m). Only depth-bearing frames can serve as *source* views for reprojection.
  - `annotations.json`, `<car_id>_annotation.json` — capture metadata.
- `/home/vaibhav/3DRealCars-Labels/<car_id>/labels/frame_XXXXX_pifpaf.json` — OpenPifPaf 24-keypoint
  vehicle pseudo-labels: `num_annotations`, `annotations` (list; empty = no detection), plus multiscale
  diagnostics. **Evaluation only, never train on them.** Thesis convention: keep frames with exactly one
  detection; discard 0 or >1.
- Split: follow thesis — 500-car training pool, held-out test cars. Persist the split as a committed file.

## Method (fast-track, final form — build this directly)

**One line:** frozen DINOv3 dense features + a lightweight head that conditions local descriptors on a
self-learned global canonical orientation and maps pixels to canonical spherical coordinates; trained with
multi-view reprojection correspondence + visibility masking + relative-pose orientation consistency +
cross-instance canonical-frame alignment. Monocular at test time. No keypoint/viewpoint/mask annotations.

### Why each piece exists
- **Frozen DINOv3** (Gram-anchored dense features, arXiv:2508.10104): category-level descriptors for free —
  this replaces DVE's original role (cross-instance generalization). Fallback backbone: DINOv2-B/14
  (SphericalMaps used it; enables clean comparison). Use registers variant if available.
- **Global orientation conditioning:** DINO patch features encode "wheel-ness", not "front-left-wheel-ness"
  (symmetry confusion documented in SphericalMaps + Telling-Left-from-Right). The disambiguating signal is
  global (whole image tells you which side you face), so pool an orientation estimate from all tokens via
  cross-attention and condition local descriptors on it (FiLM or concat). This is the paper's core novelty.
- **Orientation is learnable without labels only from sequences:** mirror ambiguity blocks single-image
  learning; relative camera rotations between views (free ARKit metadata, not human annotation) supervise
  *relative* orientation; the model invents the canonical frame.
- **Cross-instance canonical-frame alignment:** nothing forces car #12 and car #500 to agree which end of
  the canonical frame is "front". Mechanisms to try (ablation): (1) DVE-style exchange through auxiliary
  instances; (2) pseudo-correspondences across instances from frozen backbone features feeding the same
  consistency loss. Start with (2) — simpler, likely sufficient given backbone quality.
- **Visibility masking:** occlusion is geometry, not architecture; port thesis machinery (ε=0.05 m
  depth-consistency test after unprojection→transform→reprojection). Never let an invisible pixel
  produce a gradient.

### Architecture sketch
- Input image → frozen DINOv3 → patch tokens (train at ~518–896 px input; descriptor map ≥ 64×64 for
  comparability with thesis PCK protocol; bilinear-upsample tokens or use overlapping-crop feature
  stitching if needed).
- Head (trainable, small — a few transformer blocks or conv+attention):
  - `orientation token`: learned query cross-attending over all patch tokens → global orientation
    estimate. Represent as a continuous rotation (6D rotation parameterization is a solid default;
    if pitch/roll variation proves negligible, an S¹ azimuth embedding is the simpler ablation) — NOT
    discrete bins (bins are what SphericalMaps needed annotations for).
  - `descriptor branch`: patch tokens FiLM-modulated by orientation token → dense descriptor map (C=64 to
    match thesis/DVE conventions).
  - `canonical-coordinate branch`: per-pixel 3D unit vector (point on canonical sphere S²), the
    Thewlis/Mariotti object parameterization — landmark discovery and symmetry metrics read from here.
- Losses (all computed on view pairs from the same car, plus auxiliary instances for alignment):
  1. **Correspondence loss** (ported thesis Eq.): softmax-matching distance between descriptor maps of
     view A and B under the reprojection warp g_mv, restricted to visible pixels, normalized by |V|.
  2. **Orientation consistency:** R_pred(A) · R_pred(B)ᵀ ≈ R_rel(A,B) from camera poses (geodesic/rotation
     loss). Only *relative* rotation is supervised — canonical frame emerges.
  3. **Canonical-coordinate consistency:** sphere coords of corresponding (visible) pixels in A and B must
     agree; optionally tie sphere parameterization to predicted orientation (rotated camera-ray prior).
  4. **Cross-instance alignment:** backbone-feature pseudo-matches between different cars at similar
     predicted orientations must have consistent canonical coordinates / exchangeable descriptors.
- Keep the head <10M params. Batch of view pairs; cache DINOv3 features to disk for the training pool if
  GPU-time bound (frozen backbone → features are constant; huge speedup, ~disk-space tradeoff).

## Execution plan (milestones — do them in order)

### M0 — Repo scaffold + data plumbing (days)
`git init`; `pyproject.toml`/`requirements.txt` (torch, torchvision, timm or torch.hub DINOv3/DINOv2,
numpy, opencv, matplotlib, einops). Layout:
```
src/{data,models,losses,eval,viz}/  configs/  scripts/  outputs/  tests/
```
Dataset class for 3DRealCar: frame/depth/pose loading, pair sampling (30–80° separation, both frames
depth-bearing for the source), unprojection→reprojection warp + visibility mask computation (port from old
`data_loader/data_loaders.py`), PifPaf label loader with the exactly-one-detection filter. **Sanity
visualization is mandatory before any training:** warp view A onto B, overlay, confirm alignment; render
visibility masks. (The old repo's math is known-good — porting errors here silently poison everything.)

### M1 — Week-1 probe: symmetry-confusion baseline (also the paper's motivation figure + baseline row)
Frozen DINOv2 and DINOv3, no training: zero-shot PCK@10 on 3DRealCar test pairs (intra + cross-instance)
by nearest-neighbor descriptor matching; **symmetry-confusion matrix** using PifPaf keypoints — rate at
which left↔right and front↔rear symmetric keypoints (wheels, lights, mirrors) match their mirror twin
instead of the correct part, as a function of azimuth separation. Expected: high confusion at large
separations. If frozen features are already near-perfect, STOP and rediscuss direction with the user.

### M2 — Core model training (weeks 2–4)
Implement head + all four losses. Train on 3DRealCar (ARKit geometry — clean and ready; do NOT spend
fast-track time on VGGT). Success gate: beat thesis 82.8/69.5 intra/cross PCK@10, and symmetry-confusion
collapses vs M1. Debug order if training stalls: (a) correspondence loss alone should roughly reproduce
thesis-level intra PCK with a better backbone; (b) add orientation loss, check R_pred against ARKit ground
truth (it's *metadata* — legal for diagnostics, never for training the descriptors... it IS the training
signal for relative orientation, so "diagnostics" here means checking absolute-frame stability);
(c) add alignment loss last. Log per-loss curves + periodic correspondence visualizations.

### M3 — Keypoint discovery + regression protocols (week 5)
- Unsupervised discovery: cluster canonical-sphere coordinates over the training pool (spherical k-means,
  K≈10–30) → landmark detectors via canonical-coordinate nearest-neighbor; report detection consistency
  across views/instances.
- DVE-protocol evaluation: 50 virtual 1×1×C filters → intermediate heatmaps → linear regressor to PifPaf
  keypoints from few annotations, no gradient into embeddings (port `model/keypoint_prediction.py`).
  This is the comparability row against DVE/thesis/StableKeypoints-style literature.

### M4 — External benchmarks, only where we can win (weeks 6–7)
- **Freiburg cars vs SphericalMaps** (their 360° benchmark, ~100 views/instance): our unsupervised vs
  their viewpoint-supervised numbers. The kill shot — prioritize.
- **Geometry-aware subset of Telling-Left-from-Right** (built on SPair): the community's symmetry-failure
  benchmark. Skip SPair full-breadth (18 categories) — we are a category-level method for now.
- Both are zero-shot transfer: model trained on 3DRealCar, evaluated in the wild. Expect a domain gap;
  report it honestly.

### M5 — Ablations + writeup (week 8)
Minimum table: visibility masking on/off; orientation conditioning on/off; cross-instance mechanism
(none / exchange / pseudo-match); backbone (DINOv2 vs DINOv3; hourglass-CNN reference from thesis numbers).
Paper skeleton exists early: every milestone's outputs are figures/tables written to `outputs/paper/`.

### Deferred (upgrade path to main-conference/journal — additive, not redesign)
Geometry ladder rung 2 (VGGT/MASt3R-recovered poses instead of ARKit → train on raw YouTube walkarounds);
rung 3 (geometry-free: frame order + temporal smoothness + 360° loop closure); partial-arc stitching
experiment; second rigid category (MVImgNet chairs/motorbikes).

## Constraints & conventions
- Single GPU, assume 24–32 GB (lab RTX-3090-class). Backbone frozen always. Mixed precision. Feature
  caching to disk if dataloading dominates.
- Determinism: seed everything; committed split file; config-driven runs (one JSON/YAML per experiment,
  archived in `configs/`).
- Publication bar drives decisions: prefer the experiment that defends a claim reviewers will attack.
  PifPaf pseudo-labels are dev-grade ground truth; external benchmarks (M4) carry the paper.
- Keep `research_direction.md` updated when design decisions change; append to the status log below every
  significant session — this file is the cross-session memory.

## Status log (append, don't rewrite)
- 2026-07-11: Literature sweep done (`research_synthesis-1.md`); direction fixed (`research_direction.md`);
  fast track chosen as the operating plan; data formats verified (frame JSON: intrinsics 3×3 flat,
  cameraPoseARFrame 4×4 flat camera-to-world; depth PNGs every ~6th frame; PifPaf labels under
  `<car>/labels/`, empty `annotations` = no detection). No code yet. Next: M0.
