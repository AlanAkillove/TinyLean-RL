# Paper figure and table inventory (V3 closeout)

Status: **planning inventory only — no publication figure has been generated.**
Creating these figures is a separate, later, explicitly-authorised task. This document
fixes the scientific role, data source and placement of each item so that figure work
cannot silently expand or drift after the controller line closed.

Placement convention: **Main** = main text; **App** = appendix / supplementary;
**Repro** = reproducibility package (not printed).

---

## Figures

Priority order below is the recommended main-text order. Priority 1 is the design
diagram because it carries the methodological contribution; priorities 2–4 carry the
empirical story; 5–7 carry the negative and boundary results that motivate the
closeout.

### F1. Study design and evaluation protocol (priority 1)
- **Figure:** F1
- **Scientific question:** How was every claim in this project gated before it was made?
- **Data source:** none (schematic) — structure mirrors
  `experiments/manifests/v3/registry.yaml` and `docs/v3/V3-R001_preregistration.md`.
- **Main visual:** Four-stage flow: frozen sample → frozen signal → prospective
  evaluation → two-gate decision (pooled gate *and* within-synthetic gate), with the
  four frozen gate outcomes A `GO-SEMANTIC` / B `SOURCE-DRIVEN-ONLY` / C `NO-GO` /
  D `INCONCLUSIVE-BY-DATA` shown as terminal states, and the data guard
  (N_analyzed ≥ 103, positives ≥ 5) drawn *before* the gates so the ordering is visible.
- **Why needed:** The paper's primary contribution is the protocol (contribution A1).
  Without this figure the negative result reads as a failed experiment rather than as
  the protocol working as intended.
- **Placement:** Main

### F2. Three-seed RL dynamics (priority 2)
- **Figure:** F2
- **Scientific question:** RQ1/RQ3 — does GRPO on a 0.6B Lean prover produce real
  trainability, and what is the seed-wise shape?
- **Data source:** `experiments/results/e019_dynamics.json`,
  `e020_seed2_dynamics.json`, `e022_seed3_dynamics.json`; IGR scalars 0.158 / 0.150 /
  0.150 from `docs/experiment_log.md:380,419,452-457,491`.
- **Main visual:** Three-panel (one per seed) training curves, with each seed's peak
  window shaded **read off its own `ranges` block** (21–30, 41–50, 1–10) rather than
  from a summary scalar. IGR annotated per panel.
- **Why needed:** Establishes that trainability is nonzero and bounded, and that the
  dynamics are seed-dependent — the context in which the transfer failure (F3) is
  meaningful rather than simply null.
- **Placement:** Main

### F3. Held-out capability differences across seeds (priority 3)
- **Figure:** F3
- **Scientific question:** RQ2 — does in-distribution training gain transfer to the
  sealed held-out set?
- **Data source:** `experiments/results/e023_multiseed_analysis.json`;
  `experiments/manifests/e023_holdout.yaml:73-91`.
- **Main visual:** Forest plot of the three seed deltas (+0.39 / −1.37 / −1.76 pp) with
  CIs, plus the pooled mean ≈ −0.91 pp marked as crossing zero. Explicit zero line.
- **Why needed:** Carries contribution B2 and ruled-out claim D1. A figure is required
  here because the honest-negative result is easy to misread from a table of numbers
  with mixed signs.
- **Placement:** Main

### F4. Family structure and leakage (priority 4)
- **Figure:** F4
- **Scientific question:** RQ6 — how much of the evaluation pool shares structure
  across roles, and what does that imply for leakage?
- **Data source:** `experiments/manifests/v2/family_component_registry.json:49-54`,
  `docs/v2/family_leakage_audit.md:34,64-74`, `docs/v3/data_audit_stats.json`.
- **Main visual:** Component/statement accounting diagram: 7,620 statements →
  1,706 merged components, with overlap shown across roles; a separate lane showing
  the V3 decomposition (V1 686 valid groups → 612 statements → 443 components;
  R001 128 formal + 93 reserve = 221 `consumed_only`). Counting conventions must be
  labelled on the figure (1,706 merged components **vs** 1,710 L3 name-families are
  different objects and must not be conflated).
- **Why needed:** Family structure is the mechanism that makes `SOURCE-DRIVEN-ONLY`
  the correct reading of R001 — it is not a footnote, it is the explanation.
- **Placement:** Main

### F5. B003 hindsight allocation vs learnable allocation (priority 5)
- **Figure:** F5
- **Scientific question:** RQ5 — is there adaptive-compute headroom, and did the
  cross-fitted empirical-response allocator capture it?
- **Data source:** `experiments/results/v2_b003_analysis.json`,
  `docs/v2/b002_memo.md:69-74`, `docs/v2/b003_memo.md:9-14,61-67`.
- **Main visual:** Hindsight (oracle-ish) allocation curve plotted above the learned
  allocator's realised allocation and the uniform baseline, with the 4096-token
  endpoint's expected-solves CI [−1.00, −0.13] annotated. The visual point is the
  *gap* between the top curve and the realised curve.
- **Why needed:** Carries contributions B5 and ruled-out claim D2. It is the earliest
  independent appearance of the project's recurring pattern — an apparently strong
  in-distribution signal that does not survive the harder test.
- **Placement:** Main

### F6. D001 retrospective signal (priority 6)
- **Figure:** F6
- **Scientific question:** RQ7 — is there a real retrospective representation signal
  for reward-informative groups?
- **Data source:** `experiments/manifests/v3/V3-D001_results.json`,
  `V3-D001_fullproc_bootstrap.json`, `V3-D001_fullproc_500rep_fly122.json`.
- **Main visual:** Two panels. (a) Bootstrap delta distributions: fixed-OOF
  +0.14607 [0.05096, 0.24145] and full-procedure +0.10869 [−0.02504, 0.23679] (1000
  reps) with the 500-rep fly122 replication +0.11118 [−0.02427, 0.23934] overlaid.
  (b) Top-k enrichment 3.46659 [3.02435, 3.92865] with the 1000/1000 ≥ 1.75 count
  annotated. The +0.10869 CI crossing zero must be visible, not cropped.
- **Why needed:** This is the result that made the prospective test worth running
  (contribution A2, with its qualification). Cropping the zero-crossing CI would
  overstate it.
- **Placement:** Main

### F7. R001 pooled vs within-synthetic (priority 7)
- **Figure:** F7
- **Scientific question:** RQ8 — does the retrospective signal survive prospective,
  source-aware evaluation?
- **Data source:** `experiments/manifests/v3/V3-R001_results.json` (gate block,
  `source_diagnostics`, `mixture_diagnostics`).
- **Main visual:** Two side-by-side gate panels sharing a y-axis of the test statistic:
  pooled (ratio 2.9867, CI lower 2.2123, PASS/PASS) versus within-synthetic (ratio
  1.5185, CI lower 0.9762, FAIL/FAIL), with the pass threshold drawn. Below, a small
  source-composition strip (synthetic 41 analyzed/21 positives; autoformalizer 47/2;
  human 24/1) explaining the divergence.
- **Why needed:** This is the single most important figure in the paper — the pooled
  and within-source conclusions visibly diverge, and only the latter survives. Carries
  contributions A3 and D5.
- **Placement:** Main

---

## Tables

### Table 1. Experimental setup (Main)
Models with revisions and `theta0` hashes, seeds, sampling parameters, hardware
(RTX 3080 10 GB on fly122; fly90 coordination), verification stack (Lean server
container `tinylean-rl-lean-server-r001`, `LEAN_SERVER_MAX_REPLS=1`, `batch_size=1`,
loopback 127.0.0.1:8010), label rule (y = 1 iff 0 < Σscore < 8 over 8 samples/theorem),
and the frozen-settings hash `bf069ecc…`. Data source: `docs/environment.md`,
`docs/reproduction.md:11-14`, registry.

### Table 2. RL multi-seed results (Main)
Per-seed IGR (0.158 / 0.150 / 0.150), in-distribution gain, held-out delta with CI
(+0.39 / −1.37 / −1.76 pp), and dynamics peak window (21–30 / 41–50 / 1–10). One row
per seed. Data source: `e023_multiseed_analysis.json`, `e019/e020/e022_dynamics.json`.

### Table 3. Adaptive compute (Main)
Hindsight allocation vs the cross-fitted empirical-response allocator vs uniform
baseline, per token budget, with the 4096 endpoint's expected-solves CI
[−1.00, −0.13]. Carries the "headroom exists, allocator did not capture it" reading.
Data source: `v2_b003_analysis.json`, `b002_memo.md`, `b003_memo.md`.

### Table 4. D001 and R001 controller results (Main)
Two stacked blocks. D001 block: B2 AUPRC 0.57280, B1 0.42673, fixed-OOF delta
+0.14607 [0.05096, 0.24145], full-procedure delta +0.10869 [−0.02504, 0.23679]
(1000 reps) and +0.11118 [−0.02427, 0.23934] (500 reps), top-20 enrichment 3.46659
[3.02435, 3.92865]. R001 block: N nominal 128 / analyzed 112 / positives 24 /
prevalence 0.2143; pooled 2.9867 (CI lower 2.2123, P1/P2 PASS); within-synthetic
1.5185 (CI lower 0.9762, S1/S2 FAIL); AUPRC B2 0.6831 / B1 0.5512. Data source:
`V3-D001_results.json`, `V3-R001_results.json`, and the bootstrap JSONs.

### Appendix-only tables
Verifier incident tables (per-session recovery counts, per-rank recovery reasons,
INFRA_CENSORED roster) → **App** only. **Engineering incident tables must not appear
in the main text** — the verifier engineering chronology is secondary to the
scientific result and belongs in the appendix and the reproducibility package.

`docs/v3/v3_closeout.md` §4 and the registry `V3-R001-attempt2-result` amendment are
the source of record for those incident tables.

---

## Prohibited for any figure

Consistent with `docs/v3/v3_closeout.md` §2 and the registry `rescues_forbidden` list,
no figure may present, as a result: a rescued controller variant, a new-layer or MLP
controller, source-balanced retraining, a larger controller, TinyJev/Kev, an
alternative sampling objective, a new prospective sample, or any use of the 93-component
sealed reserve. The reserve was never opened and must not be plotted.
