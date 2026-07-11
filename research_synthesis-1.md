# Research Synthesis 1 — Unsupervised Multi-Viewpoint Keypoint Discovery (2026-07-11)

Context: extending the MS thesis "Extending Unsupervised Landmark Discovery to Multi-Viewpoint Objects"
(DVE + multi-view geometric correspondence on 3DRealCar, 82.8% intra / 69.5% cross PCK@10, beat supervised
YOLO-Pose baseline cross-instance) into a publishable, modern method. Base paper: Thewlis et al.,
"Unsupervised Learning of Landmarks by Descriptor Vector Exchange" (ICCV 2019, arXiv:1908.06427).

---

## 1. Verdict on the existing literature-review claim

Claim under test: *"no truly unsupervised/self-supervised method exists for multi-viewpoint keypoint
discovery."* **Verdict: mostly holds, but it is narrower than stated, and four papers come close enough
that reviewers will raise them.** Positioning against each is mandatory:

| Closest work | Why it doesn't close the gap |
|---|---|
| **SphericalMaps** (Mariotti, Mac Aodha, Bilen — CVPR 2024, arXiv:2312.13216) | Maps object pixels to a canonical sphere to fix left/right symmetry confusion of DINO features; evaluated on **360° Freiburg cars**. But it **requires human viewpoint-bin annotations** (8 bins; drops sharply below ~4 bins) plus Mask R-CNN masks. Weakly supervised, not unsupervised. Backbones: frozen DINOv1-B/8, DINOv2-B/14. Known limits: deformable categories, high-genus shapes, scale variation, needs masks at inference. |
| **Common3D** (arXiv:2504.21749, 2025) — **biggest threat, was NOT in the thesis lit review** | Fully self-supervised 3D morphable models learned from object-centric videos (CO3D-style): 3D template mesh + neural deformation field, appearance as neural features, contrastive correspondence objective through the template. Evaluates shape, pose, semantic correspondence, segmentation. But: task is shape/pose/correspondence, **not landmark discovery**; needs per-instance reconstruction-quality video; no explicit symmetry mechanism; no monocular landmark detector output. Must cite + compare. |
| **KeyDiff3D** (arXiv:2507.12336, 2025) | Unsupervised 3D keypoint discovery using **multi-view diffusion priors** (generate views from a single image; 3D feature volume from diffusion features). Demonstrated on humans (Human3.6M), birds (CUB), dogs (Stanford Dogs) — limited azimuth ranges; inherits generative model's geometry, weak for full-360° rigid objects. |
| **BKinD / BKinD-3D** (CVPR 2021 / CVPR 2023) — the "video keypoints" family | Keypoint discovery by reconstructing **spatiotemporal differences** in (multi-view) videos of behaving agents. Fundamentally requires a *moving, articulating* subject — a rigid car in a walkaround video has zero intra-object motion, so the training signal vanishes. This family does not transfer to rigid multi-view objects. |

Additional context works:

- **StableKeypoints** (Hedlin et al., CVPR 2024, arXiv:2312.00065): optimizes text embeddings so SD
  cross-attention localizes; strong on roughly-aligned categories (CelebA, CUB, DeepFashion, Human3.6M);
  keypoints are view-based; no 360° story; diffusion-scale compute.
- **Telling Left from Right** (Zhang et al., CVPR 2024, arXiv:2311.17034): quantifies a ~20-point PCK gap
  on geometry-ambiguous keypoints for zero-shot foundation features (~10 pts for supervised methods);
  provides the geometry-aware benchmark subset we should evaluate on. Their fixes use pose/keypoint
  supervision.
- **Neural Congealing** (arXiv:2302.03956) / **ASIC**: joint canonical atlases from DINO features —
  collapse or become unreliable under large viewpoint variation; effectively single-mode/limited-pose.
- **MARCO** (arXiv:2604.18267, 2026): current semantic-correspondence SOTA (DINOv2 + coarse-to-fine +
  self-distillation) — but keypoint-supervised; useful as the supervised reference point.
- **Mallis et al.** self-training landmark discovery (NeurIPS 2020 / TPAMI 2023): view-based landmarks,
  initialization-sensitive, no 3D/viewpoint mechanism. Already in thesis related work.
- **Honari & Fua** (3DV 2024): unsupervised 3D keypoints from calibrated multi-view + foreground masks,
  human bodies; sparse keypoints via mask reconstruction; needs calibrated rigs.
- **CoTracker3 / TAPIR / BootsTAP** (2023–2025): point tracking gives free intra-instance correspondence
  in video, but trackers fail exactly under large viewpoint change / occlusion — the regime that matters
  here. Useful as auxiliary supervision, not as the headline mechanism.
- **DINOv3** (Meta, Aug 2025, arXiv:2508.10104): Gram anchoring fixes dense-feature collapse; SOTA on
  keypoint-correspondence linear probes → strongest frozen backbone candidate.
- **PDiscoFormer / pose-guided self-training part discovery** (arXiv:2403.16194 etc.): unsupervised part
  discovery from DINOv2 ViT inductive biases; parts, not geometry-consistent landmarks; no 360° handling.

**The precise open square:** *fully* unsupervised (no keypoint labels, no viewpoint bins, no manual masks)
category-level **landmark discovery + dense correspondence for rigid objects across full 360° viewpoints**,
from raw capture data. Nobody occupies it. The thesis is the closest occupant; its gaps: CNN backbone,
ARKit depth/pose dependence, dense-correspondence-only evaluation (no sparse keypoint discovery), one
dataset.

## 2. Why foundation features + this problem is a good hand now

1. DINOv3 (Gram anchoring) finally makes frozen dense ViT features clean at high resolution — but DINO-family
   features **confuse symmetric/repeated parts** (left↔right doors, front↔rear wheels). Documented in
   SphericalMaps and Telling-Left-from-Right; unsolved without supervision. This failure *dominates* 360°
   rigid objects.
2. Everyone who fixed symmetry paid with supervision (viewpoint bins, pose labels, keypoints). **Capture
   geometry is a supervision source nobody uses**: relative camera pose + depth from a walkaround capture
   are sensor metadata, not human annotation — a method trained on them remains legitimately unsupervised.

## 3. Candidate directions

### Direction A (recommended): geometry-grounded descriptor exchange on frozen foundation features
- Frozen DINOv3 (or DINOv2) + lightweight trainable head producing (i) dense embedding, (ii) canonical
  spherical/3D coordinate per pixel.
- Training signals: multi-view reprojection correspondence + visibility masking (thesis machinery), plus
  **DVE cross-instance exchange in foundation-feature space** to force category-level embeddings.
- Canonical-sphere head gets viewpoint signal **free from relative camera poses** — removes exactly the
  annotation SphericalMaps needed.
- Sparse landmarks emerge by clustering canonical space → restores "keypoint discovery" to the story.
- Generalization unlock: recover poses/depth with a geometry foundation model (VGGT / DUSt3R / MASt3R)
  instead of ARKit → trains on **any raw walkaround video** (honest version of the "video" direction).
- Test time: monocular — single in-the-wild image → symmetry-disambiguated dense landmarks.
- Publishable claims: first fully unsupervised 360° landmark discovery for rigid objects; DVE revived as
  cross-instance mechanism on foundation features; viewpoint supervision replaced by free capture geometry.
- Evaluation: 3DRealCar (PifPaf pseudo-GT for dev), zero-shot SPair-71k cars, geometry-aware subset of
  Telling-Left-from-Right, Freiburg cars head-to-head vs SphericalMaps, DVE-style limited-annotation
  regression protocol.
- Compute: frozen backbone + light heads → fits 24–32 GB single GPU.

### Direction B: point-track supervision (CoTracker3/TAPIR tracks replace warps)
Tracks on unlabeled video give intra-instance correspondence without depth/poses. But trackers lose points
exactly when viewpoint change is large; "tracks as supervision" reads incremental post-CoTracker3.
→ Auxiliary signal inside Direction A, not the headline.

### Direction C: diffusion-prior keypoints for 360° rigid objects (extend KeyDiff3D/StableKeypoints)
Crowded, compute-heavy, contribution would be a delta on someone else's prior. Not recommended.

## 4. Risks

- **Common3D overlap** is the reviewer-2 risk for A. Differentiation: they need per-instance
  reconstructable video + deformable template mesh; no landmark discovery, no explicit symmetry handling,
  no monocular detector. Reproduce/compare, don't ignore.
- PifPaf pseudo-labels fine for development, not headline numbers → external benchmarks above carry the paper.
- "Cars only" reads narrow → one transfer experiment to another rigid category with walkaround video
  (chairs / motorbikes from CO3D or MVImgNet).

## 5. Open decisions (as of this writing)

1. Confirm Direction A vs deeper digging on B/C.
2. Scope: 3DRealCar-first (clean geometry) with VGGT raw-video pipeline as phase 2 — or geometry-free from day one.
3. Target venue/timeline: CVPR (Nov) / ICLR (Sep) / 3DV/WACV (softer first target).

## Sources

- DVE: arXiv:1908.06427 · DINOv3: arXiv:2508.10104 · SphericalMaps: arXiv:2312.13216 ·
  Common3D: arXiv:2504.21749 · KeyDiff3D: arXiv:2507.12336 · StableKeypoints: arXiv:2312.00065 ·
  Telling Left from Right: arXiv:2311.17034 · MARCO: arXiv:2604.18267 · BKinD: arXiv:2112.05121 ·
  BKinD-3D: arXiv:2212.07401 · CoTracker3: arXiv:2410.11831 · Neural Congealing: arXiv:2302.03956 ·
  DIY pseudo-labels: arXiv:2506.05312 · Pose-guided part discovery: arXiv:2403.16194 ·
  Honari & Fua 3DV 2024 · Mallis NeurIPS 2020 / TPAMI 2023 · Mariotti CVPR 2024
