# Research Direction — Unsupervised 360° Keypoint Discovery (2026-07-11)

Companion to `research_synthesis-1.md` (literature landscape & novelty audit). This file fixes the method
direction and the execution plan.

---

## One-sentence thesis

**A monocular, orientation-aware dense landmark model — frozen DINOv3 features conditioned on a
self-learned global canonical orientation, trained purely from the continuity and geometry of raw
walkaround captures, with no keypoint, viewpoint, or mask annotations — the first fully unsupervised
keypoint discovery method for 360° rigid objects.**

## Two design questions, settled

### Does a transformer backbone make DVE unnecessary?

Mostly yes. DVE existed to solve *cross-instance generalization*: networks trained on warped pairs of one
instance learn instance-specific descriptors, and the exchange trick forced category-level ones. A frozen
DINOv3 backbone already provides category-level features by construction (wheel patches match across car
brands zero-shot). **DVE is no longer the load-bearing mechanism and the paper must not pretend it is.**

One cross-instance problem survives, relocated: the **canonical orientation head**. Nothing supervises
which end of the canonical frame is "front", so each training instance could invent its own. Something
must force all instances to agree:
- Option 1: DVE-style exchange through auxiliary instances (our lineage, cheap).
- Option 2: pseudo-correspond cross-instance pairs via backbone features, feed the same consistency loss.
This is an empirical question → ablation. Honest framing: "the backbone replaces DVE for descriptors;
aligning the canonical frame still needs a cross-instance mechanism; here is the lightest one that works."

### Is visibility masking still needed?

Yes, wherever multi-view reprojection supervises training — occlusion is a property of geometry, not
architecture. Thesis ablation is the proof (unmasked camera correspondences: 17.3 intra PCK vs 40.4 for
TPS-only). It matters *more* with recovered (noisier) geometry in Phase 3. In geometry-free training the
same principle is implemented as tracker-visibility flags / match-confidence gating: **never let an
invisible pixel produce a gradient.**

## Key insight for the symmetry problem

The global information needed to tell left from right is already in the image and in the ViT token set
(coarse viewpoint is estimable zero-shot from DINO features — Telling Left from Right, CVPR 2024). The
failure is that DINO's objective bakes "wheel-ness", not "front-left-wheel-ness", into each *local*
descriptor. Fix = a head, not a backbone: **pool a global orientation estimate from all tokens and
condition the local descriptors on it** ("global context + local focus").

That global orientation cannot be learned from independent still images (mirror ambiguity — see
`some_more_methods.md`, Fundamental Limitation). It **can** be learned from sequences: viewpoint varies
continuously in a walkaround and closes a 360° loop; smoothness + loop closure force the two sides to
opposite orientations without any labels. Topology disambiguates what appearance cannot.

Note: for keypoint discovery we do not need *human* left/right naming — an arbitrary but consistent
canonical assignment across all instances suffices. Human names attach at eval time for free (the light
regressor sees a handful of annotated examples; a single global flip check aligns the convention).

---

## Full plan (phases)

### Phase 0 — Feasibility probe & motivation figure (~1–2 weeks)
- Extract frozen DINOv2/v3 dense features on 3DRealCar.
- Quantify symmetry confusion with PifPaf pseudo-labels: match rates front-left↔rear-right wheel,
  left↔right door, etc., across view pairs; zero-shot PCK as "backbone alone" baseline.
- Re-run thesis numbers as the CNN reference.
- Deliverable: the paper's opening figure + headroom estimate. If frozen features are already fine, stop
  and rethink — cheap insurance.

### Phase 1 — Core model on clean geometry (3DRealCar + ARKit)
- Architecture: frozen DINOv3 → lightweight head with three outputs:
  (a) dense descriptor map;
  (b) **global orientation estimate** pooled via cross-attention over all tokens, conditioning the local
      descriptors (FiLM/concat) — turns "wheel" into "front-left wheel";
  (c) canonical spherical coordinate per pixel.
- Losses:
  - multi-view reprojection correspondence + visibility masking (ported thesis machinery);
  - orientation consistency: predicted orientations of two views must differ by the known *relative*
    camera rotation (relative only — the model invents the canonical frame);
  - cross-instance canonical-frame alignment (exchange or backbone-pseudo-correspondence).
- Deliverable: beat thesis intra/cross PCK on 3DRealCar; symmetry-confusion metric collapses vs Phase 0.

### Phase 2 — Keypoint discovery + external benchmarks
- Sparse landmarks discovered unsupervised by clustering canonical space; stability metrics.
- DVE-protocol light regressor (few annotations, frozen embeddings, no backprop into them) for
  comparability with the 2018–2024 landmark literature. Test-time is monocular; multi-view is training-only.
- Zero-shot transfer: SPair-71k cars; geometry-aware subset of Telling-Left-from-Right; **Freiburg cars
  head-to-head vs SphericalMaps — their benchmark, without their viewpoint labels** (the kill shot).

### Phase 3 — Remove ARKit: the geometry ladder
- Rung 1 (dev/upper bound): ARKit poses+depth (thesis setting).
- Rung 2: VGGT / MASt3R-recovered poses+depth from raw frames → no ARKit anywhere; demonstrate training on
  in-the-wild walkaround video (YouTube / MVImgNet). Caveat framing: geometry models are 3D-supervised
  upstream → claim is "no manual annotation on target data" (same convention as SphericalMaps' Mask R-CNN).
- Rung 3 (highest risk, most original): geometry-free — frame ordering only; temporal smoothness + 360°
  loop closure on the orientation head; short-baseline correspondence from feature matches/point tracks.
- The ladder itself ("how much geometry does unsupervised landmark discovery need?") is a contribution
  even if rung 3 underperforms.
- Partial-arc experiment: instances each seen from half the azimuth range, stitched to full coverage via
  cross-instance alignment — kills the "needs complete walkarounds" objection.

### Phase 4 — Generality + paper
- One non-car rigid category (MVImgNet chairs or motorbikes).
- Ablation table: masking on/off; cross-instance mechanism; orientation conditioning on/off; geometry
  rungs; backbone (DINOv2 vs v3 vs CNN-hourglass reference).
- Writing + figures.

Compute: backbone never trains → everything fits a single 24–32 GB GPU.

Dependency logic: Phase 0 de-risks the premise; Phase 1 proves the mechanism where geometry is clean;
Phases 2–3 turn "a 3DRealCar model" into a general method — the difference between a workshop paper and a
main-conference one.

---

## Fast track — most optimistic direct path (~6–8 weeks to a submittable result)

For when we want the shortest line to a defensible paper, betting that the core mechanism works first try.
Skips the CNN reference reproduction, defers the geometry ladder and the second category.

1. **Week 1 — Mini-probe.** Symmetry-confusion numbers + zero-shot PCK from frozen DINOv3 on 3DRealCar
   (these double as the paper's baseline row and motivation figure; nothing wasted).
2. **Weeks 2–4 — Full model, straight to the final form.** Frozen DINOv3 + orientation-conditioned head +
   canonical sphere, trained on 3DRealCar with ARKit geometry (it is clean and already prepared — do not
   spend fast-track time on VGGT). All three losses in from the start; visibility masking ported from the
   thesis codebase.
3. **Week 5 — Discovery + regressor protocols.** Clustered landmarks + DVE-protocol regressor →
   the comparison rows against DVE/thesis/StableKeypoints.
4. **Weeks 6–7 — External benchmarks only where we can win.** Freiburg cars vs SphericalMaps (unsupervised
   vs their weak supervision) and the geometry-aware subset of Telling-Left-from-Right. Skip SPair breadth.
5. **Week 8 — Minimal ablations (masking, orientation conditioning, cross-instance mechanism) + writeup.**

Fast-track claim: *"first fully unsupervised landmark discovery for 360° rigid objects — matches or beats
weakly-supervised SphericalMaps on its own benchmark without viewpoint labels."* That claim alone is
publishable (3DV/WACV-grade; CVPR-grade if margins are strong). The geometry ladder (rungs 2–3), partial
arcs, and a second category are the upgrade path to a main-conference/journal version — additive, not a
redesign.

Risk accepted by fast track: if the orientation head fails to train on the first architecture attempt,
there is no intermediate diagnostic phase — debugging happens inside weeks 2–4.
