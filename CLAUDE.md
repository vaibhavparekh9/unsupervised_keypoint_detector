# Project: Unsupervised 360° Keypoint Discovery (fast-track)

Master's-research continuation toward a publication. Owner: Vaibhav Parekh (vsparekh@andrew.cmu.edu),
advisor Prof. Kenji Shimada (CMU CERLAB). Goal: **original, publishable** unsupervised keypoint discovery
for multi-viewpoint (360°) rigid objects. This repo follows the **fast track**: shortest defensible line
to a submittable result, betting the core mechanism works without intermediate phases.

Headline claim being built: *"First fully unsupervised landmark discovery for 360° rigid objects — matches
or beats weakly-supervised SphericalMaps (CVPR 2024) on its own benchmark without viewpoint labels."*

## Operating mode: autopilot

The entire pipeline is written and smoke-verified in one continuous effort, without waiting for human
review between stages. Rules:
- Every stage (S0–S5 below) ends with an **automated smoke gate** — a check the agent runs itself and must
  pass before continuing. Human-facing visualizations are still generated and saved under
  `outputs/diagnostics/` but are non-blocking (reviewed later by the user).
- **Update the Status log at the bottom of this file after every completed component** (not just stages),
  so an interrupted session can resume from the exact last step. Commit to git at each green gate with a
  message naming the stage.
- If a gate fails after 2–3 genuine fix attempts, record the failure + hypotheses in the status log,
  stub the blocking piece if a downstream stage can still be built honestly, and flag it prominently —
  do not silently lower the gate.

## Two-machine workflow

| | Dev PC (this machine) | Lab PC (cerlab27) |
|---|---|---|
| GPU | RTX 3070 Laptop, **8 GB** VRAM | RTX 3090, **24 GB** VRAM |
| Driver/CUDA | 580.159.03 | 580.159.04 / CUDA 13.0 |
| RAM / disk | 30 GB / ~199 GB free | more |
| Python | **3.10.12** (system) | **3.12.3** (system) |
| conda | yes (miniconda3) | **no** |
| sudo | — | **no** |

- All code is developed and **smoke-run here** (enough steps to prove the pipeline end-to-end), then pulled
  on the lab PC for full runs.
- **`#TOBECHANGED` convention:** every parameter that is deliberately mellowed for the 8 GB dev GPU carries
  an inline comment with the lab value, e.g. `batch_size: 2  # TOBECHANGED 16 (3090)`. Before a lab run,
  grep for `TOBECHANGED` and apply. Keep dev defaults committed; never commit lab values as defaults.
  Applies to: batch size, input resolution, feature-cache precision, number of cars in the training pool,
  steps/epochs, num_workers, EMA/eval frequency.

## Environment: venv + pinned requirements.txt (NOT docker, NOT conda-dependent)

Decision and reasoning (settled — do not revisit):
- **Docker is ruled out.** The lab PC has no sudo: root-owned files created by containers could not be
  inspected or deleted after a mess-up, and fixing docker group/rootless setup needs privileges we don't
  have. Exactly the failure mode the user fears.
- **Conda is ruled out as a requirement** (lab PC has none). Dev PC's conda may be used *only* to get a
  Python ≥3.10 base if system python is too old — but the project env itself is a plain stdlib venv so the
  same commands work on both machines:
  ```
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  ```
- **Dual-Python constraint: dev is Python 3.10.12, lab is 3.12.3, and the venv is built from system
  python on each machine.** Requirements must install and run on BOTH without version switching:
  - Write code for 3.10 syntax (no 3.12-only features); it then runs on 3.12 automatically.
  - Choose package versions that ship wheels for cp310 AND cp312: torch ≥2.2 (first release with cp312
    wheels), numpy ≥1.26, and generally "recent stable" versions of everything — old pins are the risk,
    not new ones.
  - Pin exact versions in `requirements.txt` from the working dev (3.10) smoke env, then verify cp312
    wheels exist for every pin (`pip download -r requirements.txt --python-version 312 --only-binary=:all:
    -d /tmp/wheelcheck` is a sufficient offline check; fix any pin that fails it).
  - Do not use conda to "align" Python versions — the whole point is that plain system-python venvs work
    on both machines as-is.
- `requirements.txt` is mandatory, fully pinned as described above, including the torch CUDA-wheel line
  (current stable cu12x wheel — driver 580 is backward-compatible with cu12x runtimes on both machines).
  Add a `README.md` setup section with the three lines above.
- `.gitignore`: `.venv/`, `outputs/`, `__pycache__/`, feature caches, checkpoints (keep configs + split
  files + diagnostics scripts tracked).

## Read these before coding
1. `research_synthesis-1.md` — 2020–2026 literature landscape + novelty audit. Closest rivals:
   SphericalMaps (CVPR'24, needs viewpoint bins), Common3D ('25, template-mesh correspondence, no landmark
   discovery), KeyDiff3D ('25, diffusion prior, limited azimuth), BKinD family (needs articulating agents).
2. `research_direction.md` — full design rationale; this file implements its "Fast track" section.
3. `some_more_methods.md` — mirror-ambiguity formalization (`P(I)=P(mirror(I))`): why single-image
   objectives cannot break left/right symmetry; sequences/geometry can. Cite in the paper.

## Background (user's prior work)
Thesis "Extending Unsupervised Landmark Discovery to Multi-Viewpoint Objects": DVE (Thewlis et al., ICCV
2019, arXiv:1908.06427) with TPS warps replaced by multi-view geometric correspondence (ARKit depth +
poses) + visibility masking, hourglass CNN, on 3DRealCar. Results: 82.8% intra / 69.5% cross PCK@10 at
64×64 feature resolution; beat supervised YOLO-Pose (65.6%) cross-instance. View pairs at 30–80° azimuth
separation; depth-consistency visibility threshold ε=0.05 m; DVE exchange retained. Key thesis ablation:
**unmasked** camera correspondences (17.3 intra PCK) score worse than TPS-only (40.4) — visibility masking
is non-negotiable wherever reprojection supervises training.

Old codebase: `/home/vaibhav/DVE_multiview (working)/` (fork of jamt9000/DVE). Port, don't rewrite blind:
- `data_loader/data_loaders.py` — 3DRealCar pair sampling, unprojection/reprojection warp + ARKit pose
  conventions (known-good math).
- `model/loss.py`, `model/folded_correlation_dve.py` — DVE loss machinery (memory-efficient correlation).
- `model/hourglass.py` — CNN reference. `model/keypoint_prediction.py` — regression-eval head.
- `eval_pifpaf.py`, `eval_correspondence_pifpaf.py`, `plot_pck_curve.py` — PifPaf-based evaluation.
- Configs: `configs-vaibhav/cars-hourglass-64d-dve.json`, `cars-new.json`. Thesis + DVE PDFs live there.

## Data (verified formats)
- `/home/vaibhav/3DRealCars-English/<car_id>/` — ~2585 car folders (`0000`…), each a 360° walkaround:
  - `frame_XXXXX.jpg` — RGB ≈1920×1440.
  - `frame_XXXXX.json` — `intrinsics` (flat 3×3; fx≈fy≈1333, cx≈967, cy≈731), `cameraPoseARFrame`
    (flat 4×4 camera-to-world, ARKit convention — trust the old data_loader's axis math), `frame_index`,
    `motionQuality`, `averageAngularVelocity`, `projectionMatrix`.
  - `depth_XXXXX.png` — depth for ~every 6th frame. Verify encoding on first use (likely 16-bit mm→m).
    Only depth-bearing frames can be *source* views for reprojection.
  - `annotations.json`, `<car_id>_annotation.json` — capture metadata.
- `/home/vaibhav/3DRealCars-Labels/<car_id>/labels/frame_XXXXX_pifpaf.json` — OpenPifPaf 24-keypoint
  vehicle pseudo-labels: `annotations` list (empty = no detection). **Evaluation/diagnostics only, never a
  training signal.** Thesis convention: keep frames with exactly one detection.
- Split: thesis convention — 500-car training pool, held-out test cars. Persist as a committed file
  (`configs/split.json`). Dev smoke subset: ~20 cars `# TOBECHANGED 500 (3090)`.

## Method (fast-track, final form — build this directly)

**One line:** frozen DINOv3 dense features + a lightweight head that conditions local descriptors on a
self-learned global canonical orientation and maps pixels to canonical spherical coordinates; trained with
multi-view reprojection correspondence + visibility masking + relative-pose orientation consistency +
cross-instance canonical-frame alignment. Monocular at test time. No keypoint/viewpoint/mask annotations.

### Why each piece exists
- **Frozen DINOv3** (Gram-anchored dense features, arXiv:2508.10104): category-level descriptors for free —
  replaces DVE's original role (cross-instance generalization). Fallback: DINOv2-B/14 (SphericalMaps'
  backbone; enables clean comparison; also lighter for the 8 GB dev GPU). Prefer registers variants.
- **Global orientation conditioning:** DINO patch features encode "wheel-ness", not
  "front-left-wheel-ness" (symmetry confusion documented in SphericalMaps + Telling-Left-from-Right). The
  disambiguating signal is global, so pool an orientation estimate from all tokens via cross-attention and
  condition local descriptors on it (FiLM or concat). **This is the paper's core novelty.**
- **Orientation is learnable without labels only from sequences:** mirror ambiguity blocks single-image
  learning; relative camera rotations between views (ARKit metadata, not human annotation) supervise
  *relative* orientation; the model invents the canonical frame.
- **Cross-instance canonical-frame alignment:** nothing forces car #12 and car #500 to agree which end is
  "front". Mechanisms (ablation): (1) DVE-style exchange via auxiliary instances; (2) pseudo-matches across
  instances from frozen backbone features feeding the same consistency loss. Start with (2) — simpler.
- **Visibility masking:** occlusion is geometry, not architecture; port thesis machinery (ε=0.05 m
  depth-consistency after unproject→transform→reproject). Never let an invisible pixel produce a gradient.

### Architecture sketch
- Image → frozen DINOv3/DINOv2 → patch tokens. Dev input res ~518 px `# TOBECHANGED 896 (3090)`;
  descriptor map ≥64×64 for thesis-comparable PCK (bilinear-upsample tokens if needed).
- Trainable head (<10M params; a few transformer blocks or conv+attention):
  - `orientation token`: learned query cross-attending over patch tokens → continuous rotation (6D rotation
    parameterization default; S¹ azimuth embedding as simpler ablation) — NOT discrete bins (bins are what
    SphericalMaps needed annotations for).
  - `descriptor branch`: patch tokens FiLM-modulated by orientation token → dense descriptors (C=64,
    thesis/DVE convention).
  - `canonical-coordinate branch`: per-pixel unit vector on canonical sphere S² (Thewlis/Mariotti object
    parameterization) — landmark discovery and symmetry metrics read from here.
- Losses (view pairs from same car + auxiliary instances):
  1. **Correspondence** (ported thesis loss): softmax-matching distance between descriptor maps of views
     A/B under reprojection warp g_mv, visible pixels only, normalized by |V|.
  2. **Orientation consistency:** R_pred(A)·R_pred(B)ᵀ ≈ R_rel(A,B) from camera poses (geodesic rotation
     loss). Relative only — the canonical frame emerges.
  3. **Canonical-coordinate consistency:** sphere coords of corresponding visible pixels agree across A/B;
     optionally tie sphere parameterization to predicted orientation.
  4. **Cross-instance alignment:** backbone-feature pseudo-matches between different cars at similar
     predicted orientations must give consistent canonical coordinates / exchangeable descriptors.
- Mixed precision. **Cache frozen backbone features to disk** (fp16) — the backbone never trains, this is
  the single biggest speed/VRAM lever on the 8 GB dev GPU. Mind the ~199 GB free disk: cache only the
  smoke subset on dev `# TOBECHANGED full training pool (3090, check disk)`.

## Pipeline stages with automated smoke gates (S0–S5)

Recommended layout:
```
src/{data,models,losses,eval,viz}/  configs/  scripts/  outputs/{diagnostics,runs,paper}/  tests/
```

### S0 — Scaffold + data plumbing
Dataset class (frames/depth/poses), pair sampling (30–80° separation, source view must be depth-bearing),
unproject→transform→reproject warp + visibility mask (port from old repo), PifPaf loader with
exactly-one-detection filter, committed split file.
**Gate S0 (automated):** for N≥50 sampled pairs across ≥10 cars: (a) warp round-trip A→B→A error < a few px
on visible pixels; (b) visible fraction within sane bounds (~10–70%) and decreasing with angular
separation; (c) **PifPaf reprojection agreement**: project PifPaf keypoints from A into B via depth+pose,
compare against PifPaf's own detections in B — median error must be small (this validates the geometry
port end-to-end without human eyes; diagnostics-only use of labels). Save warp-overlay images to
`outputs/diagnostics/warps/` for later human review.

### S1 — Frozen-feature probe (baseline row + motivation figure)
Zero-shot NN-matching PCK@10 on test pairs (intra + cross-instance) for frozen DINOv2/DINOv3;
**symmetry-confusion matrix**: rate at which left↔right / front↔rear symmetric PifPaf keypoints (wheels,
lights, mirrors) match their mirror twin, as a function of azimuth separation.
**Gate S1:** probe runs end-to-end and reproduces the expected qualitative pattern (confusion grows with
separation). If frozen features show *no* symmetry confusion, STOP autopilot and flag for the user — the
paper's premise needs rechecking. Outputs: `outputs/paper/symmetry_confusion.*`, baseline PCK table.

### S2 — Model + losses + training loop
All four losses, config-driven, checkpointing + resume, per-loss logging, periodic correspondence
visualizations to `outputs/diagnostics/`.
**Gate S2a (overfit test):** on 2 cars / ~20 pairs, total loss falls markedly and intra-pair matching on
those pairs becomes near-perfect — proves gradients flow everywhere.
**Gate S2b (smoke train):** dev-scale run — ~20 cars, few thousand steps `# TOBECHANGED 500 cars, full
schedule (3090)` — completes without OOM/NaN; correspondence loss curve decreasing; predicted relative
rotations correlate with ARKit relative rotations on held-out pairs (rank correlation is enough at smoke
scale).

### S3 — Evaluation suite
(a) Thesis-protocol PCK@10 intra/cross at 64×64 on held-out cars; (b) DVE-protocol light regressor
(50 virtual 1×1×C filters → linear regressor to PifPaf keypoints, frozen embeddings — port
`keypoint_prediction.py`); (c) unsupervised landmark discovery: spherical k-means (K≈10–30) on canonical
coords → detection consistency across views/instances; (d) symmetry-confusion re-run for the
before/after figure.
**Gate S3:** all four run end-to-end on the smoke checkpoint and emit tables/figures to `outputs/paper/`.
(Absolute numbers are meaningless at smoke scale; the gate is mechanical correctness. Full numbers happen
on the 3090.)

### S4 — External benchmark harnesses
Freiburg cars (vs SphericalMaps' reported numbers) + geometry-aware subset of Telling-Left-from-Right.
Downloaders/loaders + zero-shot eval scripts, runnable with a checkpoint argument.
**Gate S4:** harnesses run on the smoke checkpoint end-to-end (numbers will be weak — mechanical
correctness only). If a dataset can't be auto-downloaded, document the manual step in README and stub with
a clearly named placeholder path.

### S5 — Ablation configs + run orchestration
One config per ablation: masking off; orientation conditioning off; cross-instance mechanism
none/exchange/pseudo-match; DINOv2 vs DINOv3. A `scripts/run_all_lab.sh` that executes the full matrix
sequentially on the 3090 (checkpoint-resume safe).
**Gate S5:** each ablation config launches and trains ≥50 steps on dev without error.

### Deferred (upgrade path — additive, not redesign)
VGGT/MASt3R-recovered geometry (drop ARKit; train on raw walkaround video); geometry-free rung (frame
order + temporal smoothness + 360° loop closure); partial-arc stitching; second category (MVImgNet).

## Constraints & conventions
- Backbone frozen always. Determinism: seeded, committed split, config-driven runs archived in `configs/`.
- PifPaf pseudo-labels are dev-grade GT and diagnostics; external benchmarks (S4) carry the paper.
- Publication bar drives decisions: prefer the experiment that defends a claim reviewers will attack.
- Keep `research_direction.md` updated when design decisions change.

## Status log (append after every completed component; this is cross-session memory)
- 2026-07-11: Literature sweep done (`research_synthesis-1.md`); direction fixed (`research_direction.md`);
  fast track chosen. Data formats verified (frame JSON: flat 3×3 intrinsics, flat 4×4 ARKit camera-to-world
  pose; depth PNGs every ~6th frame; PifPaf labels under `<car>/labels/`, empty `annotations` = none).
  Machines profiled: dev = RTX 3070 Laptop 8 GB / 30 GB RAM / ~199 GB disk / conda available; lab cerlab27 =
  RTX 3090 24 GB / CUDA 13.0 / no conda / no sudo. Env decision: plain venv + pinned requirements.txt;
  docker ruled out (no sudo on lab → unfixable root-owned files). `#TOBECHANGED` convention adopted.
  Repo git-tracked. No code yet. Next: S0.
- 2026-07-12 (autopilot session): Env ready: `kp/` venv (Python 3.10.12) + pinned `requirements.txt`
  (torch 2.13.0 default PyPI wheel = CUDA 13 runtime; all pins verified to ship cp310 AND cp312
  manylinux wheels via PyPI API). **BLOCKER (flagged, not fixed): CUDA is unusable on the dev PC —
  `cuInit` returns 999 system-wide; `/dev/nvidia-uvm` open gives EIO (stale nvidia_uvm state;
  nvidia-smi works). Fix needs root: `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` (module
  refcount 0, safe) or reboot. Agent had no sudo permission; all smoke runs this session are CPU
  (code is device-agnostic, `--device cuda` ready).**
- 2026-07-12: **S0 complete, gate PASS** (`outputs/diagnostics/gate_s0.json`): round-trip A→B→A median
  0.98 px @128; vis-frac mean 22.7%, p10 7.9%, Pearson(angle, vis) = -0.28; PifPaf reprojection median
  50.2 px = 2.09% of image diagonal over 174 kp matches (gate ≤2.5%). Components: `src/data/geometry.py`
  (ported thesis correspondence math verbatim + azimuth/projection helpers), `src/data/pifpaf.py`
  (CAR_KEYPOINTS_24 ordering + L/R + F/R pairs), `src/data/realcar.py` (pair dataset), committed
  `configs/split.json` (500-car pool, 2083 test, dev_smoke=20, dev_test_smoke=10), `scripts/gate_s0.py`,
  warp overlays in `outputs/diagnostics/warps/` (visually verified: hood→hood, headlight→headlight).
  Two data findings baked into the loader: (1) pairing must use **horizontal viewing-azimuth
  separation**, not rotation-geodesic angle — phone pitch/roll lets opposite-side pairs into the 30–80°
  band (near-zero visibility); (2) temporally distant same-car frames (multi-loop captures) carry ARKit
  drift that the ε=0.05 m occlusion check rejects wholesale → `min_valid_ratio=0.05` prefilter at pair
  sampling using cached 16×16 depth (visfrac estimate, ~ms/pair). Every frame in the corpus is
  depth-bearing (verified) — any frame can be a source view. Depth PNGs: int32 mm, 256×192. Next: S1.
- 2026-07-12 (session paused at ~90% usage — RESUME POINT). Code written and unit-smoked but gates S1+
  NOT yet run (blocked on feature cache, see below):
  - **S1 code done:** `src/models/backbone.py` (frozen DINOv2 vitb14_reg via torch.hub; DINOv3 gated —
    needs `backbone.dinov3_weights`), `scripts/cache_features.py` (fp16 token cache), `src/eval/probe.py`
    (shared PCK + symmetry-confusion machinery), `scripts/gate_s1.py` (thin wrapper; STOP-flag exit 2 if
    no confusion found). DINOv2 weights fully downloaded to `~/.cache/torch/hub/checkpoints/`
    (346 MB present; a stray 0-byte `.partial` beside it is harmless junk from a raced 2nd download).
  - **S2 code done, unit-smoked (shapes/grads/rotations verified; 2.39M params, all params receive
    grad):** `src/models/head.py` (orientation query + FiLM + sphere branch, sincos posemb =
    resolution-agnostic), `src/models/rotation.py`, `src/losses/{correspondence,orientation,
    cross_instance}.py` (masked corr + DVE-exchange variant + warped consistency + geodesic rel-rot +
    pseudo-match), `src/eval/validation.py` (desc-PCK vs GT warp + rot Spearman), `scripts/train.py`
    (config-driven, AMP, resume, jsonl logs, match-vis), `scripts/gate_s2a.py` (overfit: loss ≤0.6× +
    train-pair PCK ≥70), `scripts/gate_s2b.py` (corr decreasing + rot_spearman >0.3).
  - **S3 code done (not run):** `src/eval/regressor.py` (ported DVE 50-virtual-kp predictor),
    `src/eval/discovery.py` (spherical k-means + cross-view consistency), `scripts/gate_s3.py` (a–d).
  - **S4 partial:** SPair-71k tarball fully downloaded at `data/downloads/SPair-71k.tar.gz` (227 MB, from
    https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz). Freiburg = **Freiburg Static
    Cars 52** (SphericalMaps benchmark), 4.46 GB at https://lmb.informatik.uni-freiburg.de/resources/
    datasets/FreiburgStaticCars52/freiburg_static_cars_52_v1.1.tar.gz — download was ~15% done at pause
    and will NOT survive session end; resume with `curl -C - -L -o data/downloads/freiburg_static_cars_52_v1.1.tar.gz <url>`.
    Config `_base:` inheritance added to `src/utils/config.py` for S5 ablation configs. Harness scripts
    (`bench_external.py`, converters, fixture mode) NOT yet written. S5 not started.
  - **CPU-smoke settings decided** (GPU still broken, see blocker above): backbone fwd 1.45 s/frame CPU;
    corr loss at G=64 too slow on CPU → smoke gates use overrides `data.grid_res=32
    model.descriptor_res=32 train.batch_size=2` (configs keep GPU defaults 518/64).
  - **RESUME CHECKLIST (in order):** (1) restart feature cache: `python scripts/cache_features.py --cars
    dev_smoke dev_test_smoke --max-frames 80 --device cpu --batch 2` (~50 min CPU, ~2000 frames — a
    stalled-download race killed the first attempt; weights are now local so it will not recur; if GPU
    fixed use --device cuda); (2) run `python scripts/gate_s1.py`; (3) `python scripts/gate_s2a.py` then
    `gate_s2b.py` (CPU: defaults already reduced); (4) `gate_s3.py --ckpt outputs/runs/smoke/ckpt_last.pth`;
    (5) write S4 harnesses (bench_external + SPair converter + Freiburg converter + fixture) + gate_s4;
    (6) S5 ablation configs (use `_base: ../base.yaml`) + run_all_lab.sh + gate_s5; (7) commit at each
    green gate; (8) final summary incl. GPU-fix instructions (`sudo rmmod nvidia_uvm && sudo modprobe
    nvidia_uvm`) and lab-PC launch steps.
  - NOTE: `configs/base.yaml` line 8 contains a stray edit ("TOBECHANGEwhen D test") that breaks the
    exact-string TOBECHANGED grep — appeared mid-session outside agent edits; confirm intent and fix,
    or grep for 'TOBECHANGE' instead.
