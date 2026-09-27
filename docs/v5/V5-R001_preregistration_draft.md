# V5-R001 — Outcome-only RLVR vs Process-Verified RLVR: preregistration DRAFT (rev. 2)

**Status: `DRAFT_FOR_OWNER_REVIEW` — NOT frozen, NOT approved, and NO run is authorized by this
document.**

```
V5_R001_DRAFT_CREATED:            YES (rev. 1 at 020ec51, rev. 2 by this document)
V5_R001_FORMAL_TRAINING_AUTHORIZED: NO
V5_R001_TRAINING_LAUNCHED:         NO
V5_R001_ENGINEERING_SMOKE_RUN:     NO   (proposed in §10, not yet authorized)
SEALED_RESERVE_TOUCHED:            0
```

Drafting authority: the `PROCESS_SIGNAL_GO` `stop_rule` in V5-P001 ("PROCESS_SIGNAL_GO authorizes
drafting a V5-R001 preregistration only") plus the owner's 2026-09-27 review, §5 ("Immediate task:
read `docs/v5/V5-R001_preregistration_draft.md`, perform a design audit and revise the draft only.
Do NOT launch training") and §12. Every item marked **OPEN** is an owner decision; this document is
the agenda for that decision, not a frozen design. Nothing here authorizes loading a model,
generating a token, or updating a weight.

Rev. 2 is a design audit against the owner's §5–§22. Rev. 1's factual errors are corrected in §13.

---

## 1. Scientific question (owner §5, verbatim)

> Holding model, data, rollout budget, optimization budget and verifier budget as constant as
> practically possible, does Lean process-verified credit improve family-clean held-out proving
> capability over outcome-only RLVR for Kimina-Prover-Distill-0.6B?

## 2. Admissible evidence base, and its limits

V5-P001 (`PROCESS_SIGNAL_GO`, canonical artifact
`experiments/manifests/v5/V5-P001_results.json`, sha256
`ab46c12dbe098505ba8f3885fddb8af18ef2420dc0d6c66cc856a26e4d08ddd8`) established, on the frozen V1
historical surface only:

| quantity | value |
|---|---|
| primary all-fail groups admitting the audited first-error structure (G1) | 415/575 = 0.7217, CI [0.6787, 0.7645] |
| all-fail code candidates with a verified prefix + a first error | 1 133/1 736 = 0.6526 |
| extraction agreement (E1) / cross-seed stability (E2) | 1.0000 / 0.98546 |
| G2 replication | 0.7047 / 0.7056 / 0.7568 |
| G3 quartile rates | 0.1538 / 0.8542 / 0.9375 / 0.9375 |
| frozen credit constants | d1 = −0.05 (tokens strictly before the earliest erroneous tactic), d2 = −0.10 (at it and after) |
| credit-position collisions resolved by the frozen precedence | 56 |
| oracle submissions / infra-censored / recoveries | 2 476 / 36 / 6 |

Per owner §1 the ONLY licensed reading of that GO is: *a large fraction of historical
binary-all-fail RLVR groups contain Lean-verifiable local process structure invisible to
whole-proof binary reward.* No capability, gradient or proof-success claim is admissible.

Per owner §2, the G3 quartiles are **a strong length-associated pattern** (0.1538 = 22/143 in the
first quartile vs 0.8542/0.9375/0.9375 in the other three), and G3's PASS must not be described as
evidence of length-independence. P001's classification is not altered; the consequence is carried
forward here as a first-class design constraint (§9, §11.4).

Counts that were previously quoted inconsistently (`3644 / 3123 / 3129 / 3012`) are reconciled in
[`V5-P001_count_reconciliation.md`](V5-P001_count_reconciliation.md) (documentation-only; no
historical artifact edited).

## 3. Arms (owner §6)

Both arms start from **exactly the same theta0**: `models/weights/kimina_distill_0_6b`
(Qwen3-1024/28-layer/16-head/8-KV/3072-FFN, vocab 151 936, tied embeddings, 751 632 384 parameters,
bf16; `model.safetensors` 1 503 300 328 bytes), hash-pinned in the freeze commit.

| | CONTROL (A) | TREATMENT (B) |
|---|---|---|
| algorithm | GRPO-family, DrGRPO mean-only centering (`norm_adv_by_std_in_grpo=False`), no KL, no entropy bonus | identical |
| reward channel | frozen binary outcome only: verified = 1, everything else 0 (`sorry` = failure) | identical outcome reward, **plus** the tactic-level process advantage |
| process fields | **parsed but discarded** | consumed (§7) |
| verifier request | one info-tree-bearing request per candidate (§8) | identical |
| init | theta0 | theta0 — **no warm start from a control checkpoint** |

The intervention is therefore **reward granularity only**. A third arm (process credit without the
outcome term) is a mechanism ablation and is **OPEN**: it costs ~+50 % of the total GPU budget and
is not required by §6; §12's diagnostics are the cheap substitute.

## 4. What is frozen identical, and the asymmetries that are not (owner §7)

Identical by construction, per paired seed: model, prompt pool and ordering, family composition,
`train_batch_size=4` prompts/step → 32 sequences/step at n=8, temperature 1.0, top_p 1.0,
max_prompt_length 1024, max_response_length 4096, rollout engine and decoding settings,
`ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`, sequence/token budgets, optimizer (AdamW,
lr 2e-6), KL settings (off), number of optimizer updates, total generated candidates, generated-token
budget, verifier-call policy and timeout policy, checkpoint evaluation schedule, all seeds, and
every memory-engineering knob (§10).

Documented unavoidable asymmetries:

1. **CPU-side extraction only.** B parses the info-tree into tactic spans and credit positions; A
   discards the same payload after the request. Both send the same request and pay the same server
   cost (§8). Recorded per arm in the run metadata.
2. **Advantage tensor content differs** — that is the intervention; the shape, mask, dtype and
   reduction path are identical.
3. **No asymmetry is permitted in verifier compute, rollout budget or optimizer budget.** §8 makes
   the comparison "same verifier budget", not "control cheap / treatment expensive".

## 5. Prohibited design features (owner §10)

R001 must NOT include: failed-proof context in the prompt, Lean diagnostic text as a user message,
self-revision turns, or repair prompting of any kind. `data.max_length` / multiturn stay at the
frozen V1 values (`multiturn.enable=off`, `max_assistant_turns=1` equivalent), the prompt is the
frozen V1 template at ≤ 1024 prompt tokens, and **process information is optimizer-side supervision
only** — it enters through the advantage tensor and nowhere else. This is checked by an
assertion in the training entry that the assembled prompt equals the frozen template hash.

## 6. Data for training

**OPEN (owner §3.3 / §13).** Constraints already frozen by policy: the V3-FINAL-HOLDOUT 93-component
reserve stays sealed and unused; V1/V2/V3/V4/V5 artifacts are never edited; nothing consumed by
V1 training, the V2 allocator, the V3 controller or the V4 repair probe may enter the held-out
evaluation; the training pool itself may reuse already-consumed families (it is not the capability
evidence) and the V1 pool precedent is the cheapest option.

## 7. Process credit: implementation audit (owner §11)

### 7.1 Where the credit comes from

The frozen V5 process oracle (`scripts/v5_process_oracle.py`, `docs/v5/process_oracle_design.md`)
maps a generated proof to `(verified prefix, first erroneous tactic)` and the tactic → token
mapping `TokenMapper` (`scripts/v5_process_oracle.py:400-457`) assigns d1/d2/success credit to
token positions. Per owner §3 **no new d1/d2 search, surface, process definition or parser
heuristic** is introduced: R001 consumes the frozen oracle and nothing else.

### 7.2 Where the token advantage tensor is built (code-level)

The vendored stack is `third_party/kimina-prover-rl` @ `e16b605e8186614c685875c9b57eb19e841b521a`
(editable, `verl 0.5.0.dev0`). The GRPO outcome path is provably scalar-only:

* `verl/trainer/ppo/core_algos.py:259` — "this implementation only consider outcome supervision,
  where the reward is a scalar";
* `:297` `scores = token_level_rewards.sum(dim=-1)` — the token structure is destroyed here;
* `:300-313` per-group `id2mean`, `:321` `scores[i] = scores[i] - id2mean[index[i]]`
  (mean-only centering because `norm_adv_by_std_in_grpo=False`);
* `:322` `scores = scores.unsqueeze(-1) * response_mask`, `return scores, scores`.

Reward placement is `verl/workers/reward_manager/batch.py:108`
`reward_tensor[i, length - 1] = reward`, and `batch.py:100-103` copies **every key** of a dict
returned by the custom reward function into `reward_extra_info`, which
`verl/trainer/ppo/ray_trainer.py:1289` writes into `batch.non_tensor_batch`. V1 already returns a
dict (`+custom_reward_function.reward_kwargs.return_dict=True`), so the side channel that carries
per-candidate credit into the trainer already exists and is already exercised.

**Design (minimal diff, science-preserving).**
* CONTROL keeps `algorithm.adv_estimator=grpo` byte-identical to V1 — nothing about the historical
  result is re-implemented.
* TREATMENT registers a new estimator via `core_algos.register_adv_est`
  (`core_algos.py:109-127`, dispatched at `ray_trainer.py:274-288`) that (i) recomputes the
  identical mean-centered outcome advantage and (ii) adds `λ·φ_t` at mapped token positions:
  `advantages[i, t] = center_i(Σ_j r_ij) + λ·φ(i, t)`, masked by `response_mask`.
* The actor needs no change: `dp_actor.py:368-380` selects exactly
  `[responses, response_mask, input_ids, attention_mask, position_ids, old_log_probs, advantages]`,
  so the intervention lives entirely inside a tensor that already survives `.select`.

### 7.3 Open plumbing gap (must be closed before freeze — implementation, not science)

`ray_trainer.py:277-284` passes only `token_level_rewards`, `response_mask`, `config`, optionally
`index` (uid) and `reward_baselines` to a custom estimator — **not** arbitrary
`non_tensor_batch` keys. Recommended route (owner to approve): a wrapper in this repo
(`src/tinylean_rl/rl/`) that overrides `compute_advantage` to forward
`data.non_tensor_batch["process_credit"]` to the registered estimator, launched through our own
entry module instead of `python -m verl.trainer.main_ppo`. The pinned submodule itself is never
edited. Alternatives (per-step side file keyed by uid; encoding two channels in one tensor) are
rejected: the first has a write/read ordering race, the second is ambiguous when d2 credit lands on
the terminal token, which is exactly where the outcome reward sits (`batch.py:108`).

### 7.4 Frozen frame, masking, padding and alignment

* **Frame.** Credit positions are re-derived against the *actual generated ids* in the batch's
  `responses` tensor (offset 0 = first generated token), consistent with
  `dp_actor.py:247` (`full_log_probs[:, -response_length-1:-1]`). The offline `TokenMapper` path
  re-tokenizes decoded text and is **not** assumed to agree with the online tokenization.
* **Censoring on mismatch.** If re-tokenization of a decoded response does not reproduce the
  `responses` ids, or a mapped position falls outside `[0, response_len)`, or the tactic → token
  map is `ambiguous` / `outside_response` / `response_only` / `unmapped`, the candidate's process
  term is set to zero **and counted** in a frozen diagnostic counter. Nothing is silently shifted.
* **Padding.** All arithmetic is under `response_mask`; padded positions receive 0 credit and 0
  advantage, and `loss_agg_mode=seq-mean-token-sum-norm` (`core_algos.py:726-728`) divides by the
  padded width identically in both arms, so no arm gains a per-token normalization advantage.
* **Conflict resolution — frozen.** Multiple tactics can map to the same first token; V5-P001
  observed **56** such collisions. The frozen precedence
  `_CREDIT_PRECEDENCE = {"d2": 3, "d1": 2, "success": 1}`
  (`scripts/v5_process_oracle.py:755`, applied at `:779`) is inherited verbatim: **highest
  precedence wins, one writer per position, deterministic, not "last assignment wins"**. R001 adds
  no new rule.

### 7.5 Numerical parity rehearsal before any GPU step (owner §11, §9)

Zero-GPU, on the dependency-free port `src/tinylean_rl/rl/grpo.py` (a line-by-line copy of
`core_algos.py:261-324` whose docstring already requires exactness):

1. `TREATMENT(λ=0) ≡ CONTROL ≡ compute_grpo_outcome_advantage` bit-for-bit on fixture batches,
   including a batch with a `PREFIX_BEARING_FAILURE`-shaped candidate.
2. A collision fixture reproducing the precedence rule on a duplicated position.
3. A padding fixture (mixed response lengths, right padding) proving no credit escapes the mask.
4. A censoring fixture proving a re-tokenization mismatch yields zero credit plus a counter, never
   a shifted credit.

**OPEN:** λ; whether φ is group-centered (all φ ≤ 0 by construction, so centering shifts the
zero-credit baseline); per-token vs per-candidate normalization; whether the credit applies only to
`PREFIX_BEARING_FAILURE`-class candidates; floors; and whether the outcome term and process term
share one clip range.

## 8. Verifier policy: matched verification compute (owner §8)

Both arms issue **the same Lean verification request with the info tree enabled** for every
candidate; CONTROL ignores the process fields, TREATMENT consumes them. Same endpoint, same
container, same image digest, same timeout policy, same server configuration; actual verifier
wall-clock is measured and reported per arm, so the comparison cannot become "control = cheap
verification / treatment = much more verifier compute without accounting".

Required code change, identical for both arms. Today's *training* reward path
(`recipe/kimina_prover_rl/kimina_prover_rl/reward/reward.py:80-105` → `kimina_client`
`/api/check`, `infotree` unset, `timeout=60`, 8 × 40 workers) has two properties that are
unacceptable for R001:

1. it discards the info tree, and
2. `analysis.status != valid` — including server error, timeout and "no result" — becomes
   **reward 0.0** (`reward.py:92-105`), with `sync_client`'s `safe=True` path turning a transport
   exception into `error=str(e)` for the whole batch.

Item 2 contradicts the frozen infrastructure discipline (an infrastructure outcome is censored
data, never a failed proof). So R001 must move both arms onto a verification path that (a) requests
the info tree, (b) distinguishes `verified / proof-fail / infrastructure` and censors the third
class out of the group denominator while keeping the group, and (c) keeps the V5-P001 oracle
semantics (server-confirmed timeout final, bounded isolated retries for infra only, canary gate,
two-strike wedge censoring per Amendment A). The evaluation path
(`src/tinylean_rl/verifier/kimina.py:98, :145`, `/verify` with
`infotree_type="original"` plus `VerificationSession`) already does most of this and is the
natural parent. **OPEN:** reuse the offline session inside training vs a new training-path client
with the same semantics — either way it is one shared implementation for both arms, and the change
is documented as an asymmetry versus V1, not versus CONTROL.

Measured cost of the info tree (V5-P001 Phase B, 2 476 serialized submissions, 1 attempt for
2 475 of them, dedicated container, `LEAN_SERVER_MAX_REPLS=1`):
mean **9.85 s**/candidate, median 2.14, p90 23.7, p95 54.7, p99 134.9, max 180.1 (client cap),
total 6.77 CPU-hours. V1's no-tree outcome path measured ~1.0 s/candidate
(`experiments/manifests/p3_0_complete.yaml` `verification_s_per_candidate`) and a 25.9 s/step
mean window in the frozen run (§13.2). **The info tree is roughly an order of magnitude more
expensive per candidate, and it is mandatory for both arms.** Budget consequences are in §13.

## 9. Length policy and diagnostics (owner §12)

Pre-registered length diagnostics, in training logs **and** in every evaluation: mean, median,
P90 and P95 response tokens; mean tactic count per candidate; length of successful vs failed proofs;
EOS rate; max-token truncation rate. Success conditional on length bins is **descriptive only**.
The primary capability endpoint is **not** length-normalized, and no ad hoc length penalty is
introduced: the literature-compatible objective is tested cleanly first.

Why this is a major secondary mechanism/safety endpoint (owner §2): P001's recoverability is
strongly length-associated (first-quartile 0.1538 = 22/143 vs 0.8542/0.9375/0.9375). The concrete
failure mode to test is that an apparent treatment gain is a **length-avoidance artifact** — d2 =
−0.10 penalizes every token from the first error onward, so a shorter wrong proof mechanically
receives less penalty than a longer one. The preregistered safety read is therefore:
B's held-out gain must be reported alongside Δ(mean tokens), Δ(P95 tokens), Δ(truncation rate) and
Δ(tactic count); if B's gain is confined to the short bins, that is evidence for `B_PROCESS_ONLY`
with a length mechanism, not for a capability GO.

## 10. Training feasibility on the frozen formal node (owner §9)

Formal node: **fly122, RTX 3080 10 GB** (48 SM, 62 GiB RAM, 16 cores). It remains the formal node;
nothing moves to fly90. Measured today: GPU idle (16 MiB used), oracle container up, `.venv` =
torch 2.7.0 / vllm 0.9.1 / transformers 4.53.3 / ray 2.48.0 / flash_attn 2.8.0.post2 /
kimina-client 0.2.1 / verl 0.5.0.dev0 with the submodule at the pinned `e16b605e` — i.e. the
training stack **does** match `experiments/manifests/p3_0_environment.yaml` (an earlier probe that
reported a mismatch is superseded).

The 10 GB envelope is a real problem, and the arithmetic is not close:

| term | bytes/param | size at 751 632 384 params |
|---|---|---|
| bf16 parameters | 2 | 1.40 GiB |
| fp32 master + Adam m + v | 12 | 8.40 GiB |
| bf16 gradients | 2 | 1.40 GiB |
| **static subtotal** | | **11.2 GiB** |
| lm_head logits, per 1 024 tokens of micro-batch, fp32 | | 0.58 GiB (3.1 GiB at 5 120) |

A 10 GB card exposes ≈ 9.5 GiB minus the CUDA/vLLM context, so **the static optimizer state alone
does not fit**; and the V1 measurement confirms the direction: at the frozen shape
(`tb4 → 32 seq`, micro 2, vLLM util 0.40, `max_num_batched_tokens=5120`) the run peaked at
**24.41 GB allocated / 25.98 GB reserved** (`experiments/manifests/p3_0_complete.yaml` full-FT
probe: 19.09/20.52/18 095 MiB at `tb4`, and `.cache/e019_train.log`: 23.44–24.41 GB at the
n=8 shape).

Two further §9 audit items, both in the favorable direction: **there is no reference model** (V1
froze `kl: off`, "no reference worker" — so no second 1.40 GiB weight copy and no ref-logprob pass
beyond old_log_prob), and the **process-credit tensor overhead is negligible**: one extra float32
`(32, 4096)` tensor is 0.5 MiB per step, plus a per-candidate integer position list; it never
replicates across the token dimension of the model, because it is consumed into `advantages` before
`update_actor` (§7.2).

Knobs available at `n_gpus_per_node=1` (all identical for both arms):
`actor_rollout_ref.model.param_offload`, `.optimizer_offload`, `.enable_gradient_checkpointing`,
`.enable_activation_offload`, `.use_remove_padding`, `.use_fused_kernels`,
`.entropy_from_logits_with_chunking`; `rollout.gpu_memory_utilization`,
`.max_num_batched_tokens`, `.enforce_eager`/`cudagraph_capture_sizes`;
`actor.ppo_micro_batch_size_per_gpu`; `use_dynamic_bsz` + `ppo_max_token_len_per_gpu`.
**Unavailable:** Ulysses sequence parallelism (needs > 1 GPU); `offload_policy` is FSDP2-only;
FSDP1 `FULL_SHARD` at world size 1 provides no sharding benefit.

Candidate configuration to be measured (not assumed): param_offload + optimizer_offload +
micro_batch 1 + a per-micro-batch token cap + gradient checkpointing + fused/chunked lm_head +
vLLM util ≈ 0.25–0.30 + activation offload. These are numerically-neutral-by-design
(gradient-accumulation reassociation only); they are **allowed only if they preserve the same
scientific optimization semantics across both arms**, which is exactly what §7.5's fixtures test.

**Non-scientific engineering smokes, authorized for design purposes only (owner §9).** They must
touch no holdout, produce no scientific metric, and be discarded:

1. **S0 (no GPU).** Static config dump: resolved offload flags, derived gradient-accumulation
   count (`ppo_mini_batch_size // ppo_micro_batch_size_per_gpu`, `dp_actor.py:396-399`), token
   budgets, verifier plan; CPU-only.
2. **S1 (GPU, ≤ 3 optimizer steps, training prompts only).** Full colocate rollout+update at the
   frozen shape on **training** data. Records `nvidia-smi` peak, torch peak allocated/reserved,
   per-phase wall clock (gen / old_log_prob / adv / update_actor / save), CPU RSS, vLLM KV probe.
   No evaluation, no metric interpretation, checkpoints deleted after the memory log is archived.
   One bounded retry per failure with a single knob changed, logged in a diff table.
3. **S2 (GPU, 1 step, both arms).** Arm-parity smoke: with λ = 0, TREATMENT's per-token advantage
   and `update_actor` loss must equal CONTROL's to within a preregistered float tolerance on the
   same batch, and the process-credit counters (mapped / censored / conflicts) must be non-zero.
   This is an engineering assertion, not a result.
4. **S3 (CPU only).** Info-tree training-path dry run on **stored** proofs (no generation) to fix
   the endpoint, timeout and censoring semantics and measure realized verifier throughput on
   fly122's 16 cores.

**STOP_FOR_OWNER** if exact comparable training cannot fit 10 GB. LoRA is a *semantically
different* method (different parameterization, different V1 precedent) and would itself need an
owner decision; it is not an automatic fallback.

## 11. Seeds, horizon, checkpoints (owner §15, §16)

* **Seeds.** ≥ 2 independent training seeds per arm if affordable; seeds are **paired**
  (same seed → same data order, same sampling). If only one seed per arm fits the 10 GB budget and
  the wall-clock window, the study is explicitly labeled **exploratory / pilot** and cannot support
  a strong method claim; this limitation is stated in the title of the artifact, not buried.
  **OPEN:** 2 vs 1 (cost table §13).
* **Horizon.** 60 steps is *not* auto-inherited. The horizon is the fixed affordable number chosen
  from the S1/S2 measurements before any outcome, with the ceiling imposed by the info-tree
  verifier (§8, §13). **OPEN:** the number.
* **Checkpoints.** Chosen by rule in advance (final + a fixed periodic grid), evaluated on the same
  theorems with the same generation policy. **No early stopping based on holdout. No checkpoint
  picking from final capability outcomes.**
* **Models evaluated.** theta0, CONTROL final, TREATMENT final (+ every pre-registered
  intermediate). Same theorems, per-model seeds, n, temperature, top_p, max_response and verifier;
  all checkpoints evaluated on fly122 (the E023 host-harmonization precedent: cross-device vLLM
  sampling is not trajectory-identical).

## 12. Endpoints

**Primary (owner §13, §18).** Family-clean held-out theorem success, TREATMENT vs CONTROL, paired
by theorem. The holdout is a set not consumed by V1 training / the V2 allocator / the V3 controller
/ the V4 repair / V5-P001, one theorem per family component, drawn from a prospective pool. See
§14 for why this is currently the binding blocker.

**Secondary / mechanism (explanatory, never substitutes; owner §17).** Informative-group rate
(IGR), process-active-group rate (fraction of groups where the credit tensor is non-zero),
all-fail-groups-reactivated rate, per-class credit coverage (d1/d2/success token positions,
censored-mapping counts, conflict counts), training reward, entropy, KL, gradient norm, response
length, tactic count, verifier infrastructure rate. Process metrics are **diagnostics, never
capability evidence**: the evaluation stays blind to the process channel.

**Safety.** The length-behavior battery of §9, plus truncation rate and verifier infra rate per arm.

## 13. Compute accounting (owner §21)

### 13.1 Measured anchors

| anchor | value | source |
|---|---|---|
| V1 step, frozen shape, fly90 3090 | mean 143.6 s, median 139.1, p90 171.7, max 261.4 (30 steps, steps 31–60) | `.cache/e019_train.log` |
| — of which gen | mean 90.3 s | same |
| — of which update_actor | mean 18.6 s | same |
| — of which old_log_prob | ≈ 4.6 s | same |
| — of which the reward/advantage window | mean 25.9 s, max 165.8 s | same |
| peak memory at that shape | 23.44–24.41 GB allocated | same |
| rollout throughput | 1 285 tokens/s | `p3_0_complete.yaml` |
| info-tree verification, serialized | 9.85 s/candidate mean, p90 23.7, p99 134.9 | 2 476 Phase-B raw records |
| eval generation+verify, 512 candidates | 2 453–4 506 s total, 1 721–3 766 s of it verification | `p3c_fixed_eval.yaml` |

### 13.2 Per-arm-step estimate on fly122 (to be replaced by S1 measurements)

Generation: 3080 (48 SM) vs 3090 (82 SM) ⇒ ~1.3–1.9× slower ⇒ 115–170 s. Update with full
offloading and micro 1: 18.6 s ⇒ 45–90 s (CPU-Adam bound). Old log-prob: 6–10 s. Verifier at
32 candidates/step with the info tree: **315 s serialized** (`LEAN_SERVER_MAX_REPLS=1`, the frozen
V5 oracle setting) or ~105–160 s if the owner approves 3–4 concurrent REPLs (16 cores, and each
Mathlib REPL is multi-GiB RSS, so this is a real contention risk for timing-sensitive tactics).

| scenario | per step | 60 steps | note |
|---|---|---|---|
| both arms, serialized verifier | ≈ 480–640 s | ≈ 8–10.5 h/arm | verifier is 50–65 % of the step |
| both arms, 4 REPLs | ≈ 270–390 s | ≈ 4.5–6.5 h/arm | needs an owner decision on §8 fidelity |

Evaluation per checkpoint (N theorems × n=8): N=112 ⇒ 896 candidates ≈ 0.7 h generation +
2.5 h serialized info-tree verification (0.6–0.9 h at 4 REPLs).

### 13.3 Design totals (order of magnitude, one 10 GB card, arms sequential)

| configuration | GPU-hours | wall clock |
|---|---|---|
| 2 seeds/arm × 2 arms, 60 steps, serialized verifier, 5 checkpoints | ≈ 70 h | ≈ 3 days |
| same, 4 REPLs, 40 steps | ≈ 30–35 h | ≈ 1.5 days |
| 1 seed/arm, 60 steps, serialized verifier, 3 checkpoints (pilot) | ≈ 30–33 h | ≈ 1.5 days |
| 1 seed/arm, 40 steps, 4 REPLs, 3 checkpoints (pilot, cheaper) | ≈ 10–14 h | ≈ 0.5–0.6 day |

These are the numbers behind §15's tension: **the affordable horizon is small and the verifier is
the dominant new cost**, while the historical evidence (§14.3) says a 60-step run of this recipe
changes held-out success by ≈ 0. The owner reviews cost before any launch.

## 14. Holdout and power (owner §13, §20) — the binding blocker

### 14.1 The promptset family-clean pool is exhausted

`experiments/manifests/v3/V3-R001_family_clean_pool.json` (pool hash `d2d3b58e…`) enumerates the
family components over the Kimina promptset universe and reports capacity by reading, one theorem
per component:

| reading | capacity | status now |
|---|---|---|
| `owner_literal` | 86 | spent |
| `owner_literal_wider_v1` (V3's primary) | 56 | spent |
| `owner_literal_all_reservations_honored` | 0 | — |
| `registry_labels_maximal` | 0 | — |
| `consumed_only` | 221 | **128 + 93 = 221, fully allocated** |

The arithmetic is exact and verified from the artifacts: the V3-R001 formal sample
(`v3_r001_formal_sample.json`, `reading: consumed_only`, 128 components) and the sealed final
reserve (`v3_final_holdout_reserve.json`, 93 components) are disjoint (measured overlap 0) and
sum to the whole 221-component `consumed_only` pool. V4 consumed only already-consumed families.
**Therefore: 0 never-consumed promptset components remain.** R001 cannot get a fresh family-clean
holdout from the promptset without an owner decision.

### 14.2 Options (OPEN, owner §13 "prefer a new R001 holdout from untouched components")

| # | option | cost | scientific cost |
|---|---|---|---|
| H1 | Build a new family-clean pool **outside** the promptset — NuminaMath-LEAN (104 119 unique formal statements, 31 634 with proofs) — with its own family-component registry build (the `scripts/v2_build_family_components.py` union-find over name-family / skeleton / NL), a ≤ 1024-token prompt filter, and a θ0 screening rollout | pool build + a screening rollout (the only option with new θ0 GPU cost) | none, if frozen prospectively; this is the §13-preferred path |
| H2 | Reuse the V3-R001 128-component formal sample | ~0 | it *was* consumed by the V3 controller evaluation (no gradient, but its `q` values are known); §13's exclusion list literally names "V3 controller", so this needs an explicit owner waiver |
| H3 | Unseal part of the 93-component V3 final reserve | 0 GPU | destroys the last promptset family-clean reserve; **not recommended**, and requires an explicit unsealing directive |
| H4 | minif2f (partially contacted by the paused E024) | build + screening | partial prior contact must be audited component-by-component |

H1 is the recommended shape; H2 is the only fallback that keeps N ≈ 128 at zero cost and is itself
a waiver. Nothing here touches the reserve.

### 14.3 Baselines and the historical effect band

| anchor | value | source |
|---|---|---|
| θ0 pass@8, family-clean (24/112 analyzed) | **0.2143** | `V3-R001_results.json` |
| θ0 per-candidate verified rate, family-clean | 0.1071 | same |
| θ0 truncation rate (candidates at 4096) | 508/896 = 0.567; mean generated tokens/candidate 3 614 | same |
| θ0 pass@4 on the older M1 sealed 128 | 0.3594 | `e023_holdout.yaml` |
| V1 step-60 arms vs θ0, pass@4 | 0.3438 / 0.3516 / 0.3672 | same |
| V1 theorem-level paired deltas | +0.0039 / −0.0137 / −0.0176, all CIs straddling 0; cross-seed mean **−0.0091**, 1/3 positive signs | same |
| V1 30-step pilot (64 theorems, n=8, not family-clean) | +0.0098, CI [−0.0254, +0.0430], McNemar p = 1.0 | `p3c_fixed_eval.yaml` |
| **model-vs-model binary discordance ρ** | E023 all 6 pairs: 0.086–0.125; E018: 0.094–0.109 | recomputed here from the per-record artifacts |

**This resolves a discrepancy carried by rev. 1:** rev. 1's motivation cited an interim "θ0 50 % →
55.5 % at +30 steps" figure. No canonical artifact supports it. The canonical V1 record
(`e023_holdout.yaml`) shows **no consistent family-clean capability gain** at 60 steps, and
`p3c_fixed_eval.yaml` classifies the 30-step pilot POSITIVE-INCONCLUSIVE. Any R001 statement of the
form "V1 worked, so a differential should be small" is inadmissible; the honest prior is
"outcome-only RLVR at this budget is ≈ 0 on family-clean holdout".

### 14.4 What the available holdout can resolve (owner §20)

Exact McNemar (conditional binomial on the discordant pairs, no independence assumption), 80 %
power, one-sided α = 0.05 (two-sided 0.05 in brackets), detectable difference in **pp of theorems
solved by ≥ 1 of n candidates**, at the measured discordance band:

| N | ρ = 0.09 | ρ = 0.12 | ρ = 0.13 |
|---|---|---|---|
| 56 | not resolvable | 11.9 | 12.4 |
| 86 | 8.3 [n/r] | 9.7 [10.5] | 10.1 [10.9] |
| 96 | 7.9 [8.5] | 9.2 [10.0] | 9.5 [10.4] |
| 112 | 7.4 [7.9] | 8.5 [9.3] | 8.8 [9.7] |
| 128 | 6.9 [7.5] | 7.9 [8.7] | 8.2 [9.1] |
| 192 | 5.6 [6.2] | 6.5 [7.2] | 6.7 [7.5] |
| 221 | 5.2 [5.8] | 6.0 [6.8] | 6.3 [7.0] |

Sample sizes needed for 80 % power: 8 pp ⇒ N = 94–135; 10 pp ⇒ N = 81–88; 6 pp ⇒ N = 167–241;
5 pp ⇒ N = 240–347 (ρ 0.09–0.13). Cross-checked against the frozen V4 power rule
(`scripts/v4_p001_power.py::exact_rule_power`) at N = 128: ρ 0.10 → 7.25 pp vs 7.5, ρ 0.15 → 8.82
vs 9.0, ρ 0.20 → 10.23 vs 10.5, ρ 0.30 → 12.45 vs 12.5 (agreement to within that artifact's
0.005 grid; this scratch computation is re-run and frozen as a real artifact during the freeze).

**Consequences that must be stated rather than hidden.**
1. An N ≈ 56–86 holdout cannot carry the primary test at all (or only ≥ 10 pp effects), and
   `V3-R001_power.json` already learned this lesson for a different endpoint: a gate that is
   unpowered by construction manufactures a NO-GO.
2. At N = 128 the design resolves ≈ 8 pp. The historical whole-recipe effect is ≈ 0–1 pp. So a
   **C_NO_METHOD_GAIN result is the modal expectation even if the process channel works**, and a
   GO requires the process channel to produce an order-of-magnitude-larger effect than outcome-only
   RLVR has ever produced here.
3. The legitimate levers are: more theorems (H1, §14.2), more candidates per theorem at evaluation
   (variance of the per-theorem rate falls like 1/n, at linear verifier cost — see §13), or an
   honest pilot label. Choosing thresholds to make a pass likely is not a lever.
4. **OPEN (owner §19):** exact thresholds are deliberately **not** frozen yet — the owner
   conditions them on the power audit and the compute audit, both of which are above but neither of
   which is final until §14.2 and §10 are settled.

## 15. Statistics (owner §18) and taxonomy (owner §19)

* **Primary test.** Exact McNemar on per-theorem paired indicators (unit = theorem = family
  component, never candidate), paired by theorem and by evaluation seed; helpers already in-repo
  and reused verbatim: `mcnemar_exact`, `paired_bootstrap`, `cluster_bootstrap`, `win_tie_loss`
  (`src/tinylean_rl/evaluation/p3c_stats.py`, `src/tinylean_rl/evaluation/v4_stats.py`; "The
  pairing unit is the theorem (never the candidate)").
* **Effect size.** B − A in percentage points of theorems solved, with a 95 % paired bootstrap CI
  resampled **by family component**, plus the per-theorem success-rate mean delta and its CI, plus
  the win/tie/loss table. Threshold on the CI lower bound deferred to the freeze (§14.4).
* **Pre-registration.** Test, threshold, seed schedule and censoring rules are fixed before any
  outcome. The test is not chosen after seeing outcomes; training reward is never primary evidence.
* **Taxonomy (names frozen by owner §19, thresholds OPEN).**
  * `A_CAPABILITY_GO` — B beats A on the primary endpoint by ≥ the preregistered practical
    threshold with the CI rule satisfied, and the §9 length-safety battery does not explain it away.
  * `B_PROCESS_ONLY` — the process channel demonstrably changes the training signal and mechanism
    endpoints (process-active rate, all-fail reactivation, IGR) without a capability gain.
  * `C_NO_METHOD_GAIN` — neither capability nor a mechanism change attributable to the credit.
  * `D_INCONCLUSIVE` — the sample/budget could not carry the test (censoring rate, verifier
    infrastructure, power). **Not** a licence to redraw the holdout or re-cut the denominators.
  No post-outcome rescue path exists.

## 16. Ordering: what has to happen before this draft can be frozen

1. Owner decisions §14.2 (holdout source), §10 (smoke authorization), §6/§13 (budget, horizon,
   verifier concurrency), §7.5 (λ and normalization), §3 (ablation arm y/n).
2. New pool build + prospective family-clean freeze (H1), or the H2 waiver — with its own commit.
3. §7.5 CPU parity fixtures + tests, §7.3 plumbing decision implemented in this repo only, `ruff`.
4. §8 training-path verifier change, shared by both arms, validated on stored proofs (S3).
5. S0 → S1 → S2 smokes on fly122 (non-scientific, discarded), producing the real per-step cost,
   peak VRAM and the affordability horizon.
6. Freeze: this document rewritten as `V5-R001_preregistration.md` with every OPEN resolved, the
   power artifact committed, seeds/horizon/checkpoints fixed, registry entry `PREREGISTERED`, and
   the owner's freeze signature.
7. Owner launch authorization — a separate act from the freeze.

## 17. Corrections relative to rev. 1

1. Rev. 1 claimed the primary claim must be tested "at the same … verifier budget" without noting
   that today's training path sends **no** info tree and that its infra outcomes silently become
   reward 0.0. Now §8, with the required shared-path change.
2. Rev. 1 called the conflict rule **OPEN** ("no silent last-wins"). It is in fact already frozen —
   precedence `d2 > d1 > success`, 56 historical collisions — and R001 inherits it (§7.4).
3. Rev. 1 asserted a data option space as if a fresh family-clean promptset holdout were available.
   It is not: 0 never-consumed components remain (§14.1).
4. Rev. 1's fly122 flag said the recipe "requires a footprint study"; it now states the binding
   arithmetic (11.2 GiB of static optimizer state > 9.5 GiB usable; 24.41 GB measured peak at the
   frozen shape) and the concrete knob set, plus the S0–S3 protocol and `STOP_FOR_OWNER` (§10).
5. Rev. 1's power skeleton said "a preregistered practical threshold and bootstrap CI" with no
   numbers. It now carries the measured baselines, the discordance band, the exact detectable-effect
   table and the explicit statement that a null is the modal expectation (§14).
6. Model identity corrected: 751 632 384 parameters, Qwen3 architecture 1024/28/16/8/3072, vocab
   151 936, tied embeddings — not 610 M / 896 / 24.
7. Rev. 1 treated ≥ 2 seeds and a 60-step horizon as defaults; both are now costed owner decisions
   with a pilot label attached to the 1-seed fallback (§11, §13).
8. Added: the length-safety framing of owner §2 (P001's G3 is a length-associated pattern, and the
   d2 term creates a length-avoidance failure mode that R001 must test, not describe away).

## 18. Prohibitions in force (unchanged until an owner directive says otherwise)

* No training, no weight update, no model generation, no R001 launch — by this draft or by
  `PROCESS_SIGNAL_GO`.
* No new d1/d2 search, no new historical surface, no new process definition, no new parser
  heuristic, no new process oracle before R001 (owner §3).
* No use of the sealed V3 93-component reserve or any family-clean capability holdout without an
  explicit unsealing directive.
* No modification of any frozen V5-P001 artifact, and no retrospective alteration of the P001
  classification or its `if_GO` field; the canonical analyzer is frozen after its metric-emitting
  run, so any analyzer change needs a new owner-approved amendment.
* No process-reward rescue path: no gate, threshold or denominator changes after an outcome.
* No moving formal training to fly90.

## 19. Provenance

* V5-P001 canonical result `experiments/manifests/v5/V5-P001_results.json` (sha256 `ab46c12d…`),
  classification `PROCESS_SIGNAL_GO`, analysis head `7d0b139` over FreezeA (execution head
  `5b1c5d2`); memo `docs/v5/V5-P001_result.md`; count reconciliation
  `docs/v5/V5-P001_count_reconciliation.md`; oracle and conventions
  `docs/v5/process_oracle_design.md`; amendments A and B; owner review recorded in
  `experiments/manifests/v5/registry.yaml`.
* Phase-B per-submission records `runs/v5_p001_process/raw/*.json` (2 476 files) — the source of
  the §8 verifier latency statistics.
* Pool/power/holdout artifacts: `experiments/manifests/v3/V3-R001_family_clean_pool.json`,
  `v3_r001_formal_sample.json`, `v3_final_holdout_reserve.json`, `V3-R001_power.json`,
  `V3-R001_predictions.json`, `V3-R001_results.json`.
* V1 evidence: `experiments/manifests/e023_holdout.yaml`, `p3c_fixed_eval.yaml`,
  `p3_0_complete.yaml`, `p3_0_environment.yaml`, `.cache/e019_train.log`.
* Stack: `third_party/kimina-prover-rl` @ `e16b605e…`, `src/tinylean_rl/rl/grpo.py`,
  `src/tinylean_rl/evaluation/p3c_stats.py`, `src/tinylean_rl/evaluation/v4_stats.py`,
  `src/tinylean_rl/verifier/kimina.py`, `scripts/v5_process_oracle.py`,
  `scripts/v4_p001_power.py`.
* This draft is a coordination artifact: it records the authorized design audit and freezes nothing.
