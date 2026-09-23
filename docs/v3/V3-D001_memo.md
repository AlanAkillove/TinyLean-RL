# V3-D001 — Result memo (offline informative-group decision probe)

Experiment: **V3-D001** · Status: **COMPLETE — offline probe only, NO RL run**
Preregistration: `docs/v3/V3-D001_preregistration.md` (commit `41c5730`) — frozen before fitting
Machine record: `experiments/manifests/v3/V3-D001_results.json` · reps provenance: `V3-D001_theta0_reps_meta.json`
Formal run: 2026-09-23 on **fly122 / RTX 3080** (§20 step 8–11) · Reported to owner; **§20 step 14 = STOP**

---

## Bottom line

The pre-registered ranking gate **G1 ∧ G2 ∧ G3 = GO**, and the calibration criteria (§15)
are additionally met, so the outcome is a **GO — "calibrated semantic controller"** claim
at the *offline* level. The frozen decision representation (frozen theta0 **block-18
last-token** ⊕ training step → L2 logistic) beats the handcrafted **difficulty+source**
baseline **B1** on family-clean nested CV: **+0.146 AUPRC (95% CI [+0.051, +0.241])**,
**top-20 informative-group enrichment 3.47× (CI [3.02, 3.93])** vs B1's 2.94×, and it is
*better* calibrated (Brier 0.088 vs 0.110; ECE 0.032 vs 0.095). This is evidence that
Lean-theorem semantics carry informativeness signal **beyond** what a difficulty/source
proxy already knows. **No RL was run and none is launched here** — V3-R001 is the owner's
call.

Caveats that temper the claim are in *Limitations*; the two sharpest are the **source
concentration** of the signal and the **seed-3 cross-seed fold** where B2 ≈ B1.

---

## §21 report template

```yaml
host:
  hostname: "ubuntu (fly122 logical role)"
  hostname_I: "10.3.25.122"
  gpu: "NVIDIA GeForce RTX 3080, 10240 MiB, uuid GPU-e073cec7-...248f8"
  git_rev: "41c5730122c7d343fe6fa685a9e2137d2a7cd258 (== prereg commit)"
  branch: v3-jev-rl-controller
  formal_node_verified: true          # owner §0; fly90 smoke retained as NON-FORMAL only

V3-0:
  commit: 0377745
  source_manifest: "experiments/manifests/v3/v1_rollout_sources.json (commit bafdc92)"
  groups_total: 720                   # 240 x 3 seeds, all size-8
  groups_valid: 686                   # strict complete-group: 720 - 34 infra-censored
  components: 433
  informative_rate: 0.1516            # 104 / 686
  anomalies: "34 groups (103 candidates) '# System Error:' -> dropped from primary; 0 unparsable"

V3-D001:
  prereg_commit: 41c5730
  theta0_hash: "34e6e630f564d330c79424c404ab0494558a0a659e6201b47d9bd88ccd640fe2 (verified on fly122)"
  theta0_reps_content_sha256: "d0c6b7b211d779e12ea22ca8a6c33808934760c8c74bb90ecf781efa909ecf4c"
  primary_layer: 18                   # last non-padding token, R^1024 (+ step_norm)
  folds_hash: "e0e0d30cfd81aed21d9bda91ae21be5700f644ddf77a198e7729d3fd93ede0d1 (686 OOF, nested 5/4, seed 20260923)"

stationarity:                          # descriptive ONLY; never used to retune model/representation
  IGR_by_step_pooled: "1-10:.168 11-20:.128 21-30:.175 31-40:.140 41-50:.157 51-60:.138 (flat, no trend)"
  repeated_label_consistency:
    within_seed: "16 theorems: 16 always-non-informative, 0 always-informative, 0 flips"
    cross_seed:  "62 theorems: 57 always-non-inf, 2 always-inf, 3 flips  (95% label-stable across seeds)"
  source_prevalence: "synthetic 0.344 (n=270) | autoformalizer 0.030 (n=230) | human 0.022 (n=186)"

B0_prevalence:
  AUPRC: 0.1553
  Brier: 0.1286            # AUROC 0.495 (chance), enrichment ~0.96x -> correct null

B1_handcrafted:
  AUPRC: 0.4267
  Brier: 0.1102
  ECE:   0.0954
  top10_IGR: 0.4783   top20_IGR: 0.4453   top30_IGR: 0.4078
  top20_enrichment: 2.937x

B2_block18_PRIMARY:
  AUPRC: 0.5728
  delta_vs_B1: +0.1461
  delta_CI_95: "[+0.0510, +0.2415]"        # paired family-component bootstrap, 10k reps, seed 20260923
  AUROC: 0.8706
  Brier: 0.0877                            # dBrier(B2-B1) CI [-0.034, -0.011] -> strictly better
  ECE:   0.0323                            # ECE CI [0.021, 0.058] <= 0.10
  top10_IGR: 0.5797   top20_IGR: 0.5256   top30_IGR: 0.4272
  top20_enrichment: 3.467x
  enrichment_CI_95: "[3.024, 3.929]"

robustness:                                # secondary; primary remains block 18
  block9:  "AUPRC 0.502, top20 enrichment 3.32x"
  block27: "AUPRC 0.564, top20 enrichment 3.51x"   # signal spans depth; not a layer-picking artifact

cross_seed:                                # family-isolated, 2 train seeds -> 1 held-out seed
  holdout_seed1: "B1 AUPRC 0.629 -> B2 0.671 (B2>B1 yes); top20 IGR 0.609 > prev 0.160  -> PASS"
  holdout_seed2: "B1 AUPRC 0.375 -> B2 0.503 (B2>B1 yes); top20 IGR 0.468 > prev 0.145  -> PASS"
  holdout_seed3: "B1 AUPRC 0.487 -> B2 0.471 (B2>B1 NO ); top20 IGR 0.455 > prev 0.149  -> FAIL (rank)"
  # 2/3 PASS meets the frozen G3 bar (>=2/3)

gate:
  G1: PASS   # dAUPRC(B2-B1) bootstrap 95% CI lower bound +0.051 > 0
  G2: PASS   # top-20 B2 enrichment point 3.47x >= 1.75 AND CI lower 3.02 > 1.0
  G3: PASS   # 2/3 held-out-seed folds with B2>B1 AUPRC and top20 IGR > prevalence
  ranking_GO_NO_GO: GO
  calibration_claim: "calibrated semantic controller (ranking GO AND Brier(B2)<=Brier(B1) AND ECE(B2)<=0.10)"

compute:                                   # formal fly122 / RTX 3080 numbers
  extraction_time_s: 28.38                 # 612 unique prompts, batch=1, blocks {9,18,27}
  peak_vram_gb: 1.525
  model_load_time_s: 3.66
  cpu_fit_time_s: 290.44                   # nested CV (B1 + B2 x3 layers) + cross-seed + 10k bootstrap, BLAS threads capped=4

unexpected_findings:
  - "Informativeness is strongly source-conditioned: synthetic IGR 0.344 vs human 0.022 / autoformalizer 0.030 - a ~15x spread. The informative class is overwhelmingly the synthetic subgroup."
  - "Cross-seed label stability is high (only 3/62 repeated theorems flip across seeds), supporting the 'learnable theorem property' premise rather than pure per-run reward noise."
  - "B2 improves calibration, not just ranking (ECE 0.095 -> 0.032; Brier -0.022 with CI excluding 0)."
  - "Best inner-CV C is near the grid floor (0.003) for both B1 and B2 across folds - consistent with 1025 features vs 104 positives needing heavy regularization."

limitations:
  - "GO is B2-vs-B1 enrichment over a *strong* difficulty+source baseline; the absolute signal remains largely difficulty/frontier structure (all-fail 84%). B2's contribution is a real but modest margin over B1 (AUPRC +0.146; top-20 3.47x vs 2.94x)."
  - "Source concentration: because informative groups are ~90% synthetic, a B2-driven sampler would oversample synthetic-family theorems -> potential RL distribution shift; the offline probe does not evaluate whether that helps final capability."
  - "G3 is not uniform: seed-3 holdout shows B2 AUPRC marginally BELOW B1 (0.471 vs 0.487). It clears the frozen 2/3 bar but cross-seed generalization is seed-dependent."
  - "Power: 104 positives; DTAUPRC CI [+0.05,+0.24] is positive but wide. Cross-seed folds have 33-37 positives -> point estimates + CI only (as preregistered)."
  - "This is OFFLINE informativeness prediction only. It does NOT demonstrate improved RLVR sample efficiency; that requires the (unauthorized-here) V3-R001 intervention."
  - "Single backbone (theta0), single pooling (last-token), primary depth fixed at 18; raw logistic probabilities, no post-hoc calibration. All frozen before outcomes."

recommendation: >
  Advance the finding to the owner as a genuine, preregistered offline GO with a
  calibration claim: frozen theta0 semantics predict reward-informative GRPO groups
  beyond a handcrafted difficulty baseline. If the owner elects to test the
  sample-efficiency hypothesis, the next step is ONE tightly-scoped V3-R001
  (uniform vs decision-guided theorem sampling, identical theta0/algorithm/n=8/
  optimizer/budget/reward/step-count, only the sampling policy differing), designed to
  (a) measure informative-group density and useful-groups-per-GPU-hour, and
  (b) explicitly address the source-concentration / distribution-shift risk and the
  seed-3 cross-seed weakness surfaced here. Per §20 step 14 and owner §12/§16, I am
  STOPPING here: no RL is launched, no B3 MLP is trained, and no fold/layer/label was
  changed to obtain this outcome.
```

---

### One-line status
V3-D001 offline probe = **GO (G1∧G2∧G3) with calibration claim**; B2 block-18 beats B1 on
family-clean nested CV (+0.146 AUPRC [0.051,0.241], 3.47× top-20 enrichment) and is
better calibrated; **no RL run — awaiting owner decision on V3-R001.**
