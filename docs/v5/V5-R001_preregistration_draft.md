# V5-R001 — Outcome-only RLVR vs Process-Verified RLVR: preregistration DRAFT (rev. 3)

**Status: `DRAFT_FOR_OWNER_REVIEW` — NOT frozen, NOT approved, and NO scientific run is
authorized by this document.** Rev. 3 implements the owner's 2026-09-27 pre-launch directive
(the ~20-section review that (a) accepted the two-arm design in principle, (b) resolved three
owner decisions, (c) froze the intervention, and (d) authorized engineering feasibility work
S0–S3 only).

```
V5_R001_DRAFT_CREATED:               YES (rev. 1 at 020ec51, rev. 2 at 4520413, rev. 3 = this document)
V5_R001_FORMAL_TRAINING_AUTHORIZED:  NO
V5_R001_TRAINING_LAUNCHED:           NO
V5_R001_ENGINEERING_SMOKE_RUN:       AUTHORIZED_S0_S3_NOT_YET_RUN
SEALED_RESERVE_TOUCHED:              0
```

Drafting authority: the `PROCESS_SIGNAL_GO` `stop_rule` in V5-P001 ("PROCESS_SIGNAL_GO authorizes
drafting a V5-R001 preregistration only") plus the owner's 2026-09-27 pre-launch directive (§1–§20)
and the earlier design-audit review. **Every item marked OPEN is an owner decision**, and the
freeze (§17) is a separate act from the launch. Nothing here authorizes loading a model,
generating a token for capability evidence, or updating a weight.

Rev. 3 supersedes rev. 2: the intervention that rev. 2 left OPEN is now frozen by owner decision
(§3, §4, §7), the holdout decision is resolved to H1 (§14), and the engineering-smoke protocol is
specified per stage (§10). Corrections are listed in §18.

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

Per owner §1 of the P001 review the ONLY licensed reading of that GO is: *a large fraction of
historical binary-all-fail RLVR groups contain Lean-verifiable local process structure invisible
to whole-proof binary reward.* No capability, gradient or proof-success claim is admissible.

Per owner §2, the G3 quartiles are **a strong length-associated pattern** (0.1538 = 22/143 in the
first quartile vs 0.8542/0.9375/0.9375 in the other three), and G3's PASS must not be described as
evidence of length-independence. P001's classification is not altered; the consequence is carried
forward here as a first-class design constraint (§9, §12).

Counts that were previously quoted inconsistently (`3644 / 3123 / 3129 / 3012`) are reconciled in
[`V5-P001_count_reconciliation.md`](V5-P001_count_reconciliation.md) (documentation-only; no
historical artifact edited).

## 3. Arms and the frozen intervention (owner §1, §2)

Both arms start from **exactly the same theta0**: `models/weights/kimina_distill_0_6b`
(Qwen3-1024/28-layer/16-head/8-KV/3072-FFN, vocab 151 936, tied embeddings, 751 632 384 parameters,
bf16; `model.safetensors` 1 503 300 328 bytes), hash-pinned in the freeze commit.

| | CONTROL (A) | TREATMENT (B) |
|---|---|---|
| algorithm | GRPO-family, DrGRPO mean-only centering (`norm_adv_by_std_in_grpo=False`), no KL, no entropy bonus | identical |
| outcome channel | frozen binary outcome only: verified = 1, everything else 0 (`sorry` = failure) | identical outcome channel |
| process channel | **parsed but discarded** | consumed exactly as frozen below |
| verifier request | one info-tree-bearing request per candidate (§8) | identical (§8) |
| init | theta0 | theta0 — **no warm start from a control checkpoint** |

**Frozen intervention (owner §1 semantics).**

* CONTROL keeps the DrGRPO mean-only outcome advantage `A_outcome_i = g_i − ḡ`, with `ḡ` the
  **mean global outcome over VALID candidates** of the group (validity = not
  infrastructure-censored, §4.3).
* TREATMENT adds a per-tactic process term at exactly one token position:
  `A_treatment_i,t = A_outcome_i + I[t == first_token(tactic_i,j)] · A_process_i,j`,
  with `A_process_i,j = φ_i,j − ḡ` — the same `ḡ` as CONTROL, no separate baseline.
* `φ = +1` successful tactic; `φ = −0.05` failed, tactic strictly before the first error;
  `φ = −0.10` the first-error tactic and every later tactic.
* **Primary λ = 1. No λ tuning, no φ centering by `mean(φ)`, no summation into a new
  response-level reward, no normalization search.** These are prohibitions, not defaults to be
  optimized later (§19).
* CONTROL must therefore not be described as "byte-identical to V1". The required wording is:
  **"V1-equivalent outcome-only objective under the shared R001 execution stack"**, with the
  scientific reward/advantage semantics reproducing V1 exactly on non-infra fixtures (§7.5).

The intervention is therefore **reward granularity only** (plus the shared verifier-policy change
applied to both arms, §8). A third arm (process credit without the outcome term) is a mechanism
ablation and is **OPEN**: it costs ~+50 % of the total GPU budget and is not required by the
directive; §12's diagnostics are the cheap substitute.

## 4. Frozen semantics of credit, censoring and validity (owner §2, §3, §4)

### 4.1 Where credit applies (owner §2)

Process credit applies **wherever the oracle defines it** — it is not restricted to
`PROCESS_STRUCTURED_RECOVERABLE` candidates. `PROCESS_STRUCTURED_RECOVERABLE` was a **P001
analysis predicate** and is explicitly **not** an R001 training eligibility filter.

| candidate class | process-token contribution |
|---|---|
| verified + mapped | `+1` on the first token of each successfully mapped tactic |
| failed, valid verified prefix | `−0.05` on the first token of each tactic strictly before the first error |
| failed | `−0.10` on the first token of the first-error tactic and of every later tactic (to the end of the response when the blame index extends there) |
| failed on the first tactic | `−0.10` on that tactic (no d1 positions exist) |
| no code / parser-unusable / no mapped tactic | **no process-token contribution at all** (not "reward 0"; the outcome channel alone applies) |
| infrastructure-censored | process term **missing** — excluded from `ḡ` and its response policy loss masked to zero (§4.3); never reward-zeroed |

### 4.2 Collision semantics (owner §3) — frozen, no new rule

`_CREDIT_PRECEDENCE = {"d2": 3, "d1": 2, "success": 1}` (`scripts/v5_process_oracle.py:755`,
applied at `:779`): when several tactics map to the same first token position, the **highest
precedence wins**, **exactly one write per token position**, deterministic — never "last
assignment wins". V5-P001 observed 56 such collisions on the historical surface. R001 inherits the
rule verbatim and adds **direct online-batch fixtures** (§7.5) rather than rederiving it.

### 4.3 Infrastructure censoring (owner §4) — identical in both arms

1. An infrastructure outcome (server error, timeout-final, unreachable endpoint, canary failure,
   two-strike wedge per Amendment A) **excludes the candidate from `ḡ`**;
2. its **response policy loss is masked to zero** (masked-out tokens contribute neither gradient
   nor denominator);
3. an **infra counter** is recorded per step and per arm;
4. `ḡ` = mean outcome over the **valid** candidates only;
5. if `n_valid < 2`, the **whole group update is skipped** and recorded as `GROUP_SKIPPED_INFRA`
   (no resampling, no re-rollout, no surrogate denominator);
6. identical semantics in both arms — this is a shared-stack change, not an arm asymmetry.

An infrastructure outcome is **censored data, never a failed proof**. This is exactly the property
today's training path violates (§8).

## 5. What is frozen identical, and the asymmetries that are not

Identical by construction, per paired seed: model, prompt pool and ordering, family composition,
`train_batch_size=4` prompts/step → 32 sequences/step at n=8, temperature 1.0, top_p 1.0,
max_prompt_length 1024, max_response_length 4096, rollout engine and decoding settings,
`ppo_mini_batch_size`, `ppo_micro_batch_size_per_gpu`, sequence/token budgets, optimizer (AdamW,
lr 2e-6), KL settings (off), number of optimizer updates, total generated candidates,
generated-token budget, verifier-call policy and timeout policy, checkpoint evaluation schedule,
all seeds, and every memory-engineering knob (§10).

Documented unavoidable asymmetries:

1. **CPU-side extraction only.** B parses the info-tree into tactic spans and credit positions; A
   discards the same payload after the request (and may run an equivalent no-op CPU path for
   matched accounting). Both send the same request and pay the same server cost (§8). Recorded per
   arm in the run metadata.
2. **Advantage tensor content differs** — that is the intervention; the shape, mask, dtype and
   reduction path are identical.
3. **No asymmetry is permitted in verifier compute, rollout budget or optimizer budget.** §8 makes
   the comparison "same verifier budget", not "control cheap / treatment expensive", and the
   treatment verifier path is not optimized differently from control's.

## 6. Prohibited design features

R001 must NOT include: failed-proof context in the prompt, Lean diagnostic text as a user message,
self-revision turns, or repair prompting of any kind. `data.max_length` / multiturn stay at the
frozen V1 values (`multiturn.enable=off`, `max_assistant_turns=1` equivalent), the prompt is the
frozen V1 template at ≤ 1024 prompt tokens, and **process information is optimizer-side supervision
only** — it enters through the advantage tensor and nowhere else. This is checked by an assertion
in the training entry that the assembled prompt equals the frozen template hash.

## 7. Process credit and advantage: implementation audit and frozen design

### 7.1 Where the credit comes from

The frozen V5 process oracle (`scripts/v5_process_oracle.py`, `docs/v5/process_oracle_design.md`)
maps a generated proof to `(verified prefix, first erroneous tactic)` and the tactic → token mapping
`TokenMapper` (`scripts/v5_process_oracle.py:400-457`) assigns d1/d2/success credit to token
positions. **No new d1/d2 search, surface, process definition or parser heuristic** is introduced:
R001 consumes the frozen oracle and nothing else.

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

**Design (minimal diff, science-preserving).** Both arms run the *same* registered estimator
through the same wrapper; the arm difference is one frozen constant (λ), so no arm can acquire
bookkeeping the other lacks:

* The estimator (registered via `core_algos.register_adv_est`, `core_algos.py:109-127`, dispatched
  at `ray_trainer.py:274-288`) computes the outcome advantage `A_outcome_i = g_i − ḡ` with `ḡ`
  over the group's **valid** candidates exactly as in §3/§4.3, and — treatment only, λ = 1 — adds
  `A_process_i,j = φ_i,j − ḡ` at the first token of tactic `j` only:
  `advantages[i, t] = A_outcome_i + λ · Σ_j I[t == first_token(tactic_i,j)] · (φ_i,j − ḡ)`,
  masked by `response_mask` (§4.2 guarantees at most one write per position).
* CONTROL is that same estimator with λ = 0: the frozen DrGRPO mean-only outcome advantage
  (`A_outcome_i = g_i − ḡ`) plus the shared infra censoring. `tests/test_v5_r001_process_credit.py`
  pins λ = 0 to `compute_grpo_outcome_advantage` **bit-for-bit** on non-infra batches (§7.5.1).
* The wrapper additionally applies `row_keep` from §4.3.2 (zeroing the censored row's
  `response_mask`), so an infrastructure-censored candidate contributes neither gradient nor
  denominator in either arm.
* **Rev. 3 implementation note.** The earlier sketch in this section (CONTROL = the stock `grpo`
  branch at `ray_trainer.py:262-274`, "unchanged") is superseded: the stock branch takes the mean
  over *all* rows of a group and cannot exclude infrastructure-censored candidates from `ḡ`, which
  §4.3.4 requires in both arms. Control therefore runs the shared estimator with λ = 0; the
  *scientific* content of "V1-equivalent outcome-only objective" is preserved and tested, and §3's
  wording requirement applies unchanged.
* The actor needs no change: `dp_actor.py:368-380` selects exactly
  `[responses, response_mask, input_ids, attention_mask, position_ids, old_log_probs, advantages]`,
  so the intervention lives entirely inside a tensor that already survives `.select`.

### 7.3 Plumbing gap (must be closed before freeze — implementation, not science)

`ray_trainer.py:277-284` passes only `token_level_rewards`, `response_mask`, `config`, optionally
`index` (uid) and `reward_baselines` to a custom estimator — **not** arbitrary `non_tensor_batch`
keys. The route implemented in this repo: a wrapper in `src/tinylean_rl/rl/` that overrides
`compute_advantage` to forward `data.non_tensor_batch["process_credit"]` to the registered
estimator, launched through our own entry module instead of `python -m verl.trainer.main_ppo`.
The pinned submodule itself is never edited. Alternatives (per-step side file keyed by uid;
encoding two channels in one tensor) are rejected: the first has a write/read ordering race, the
second is ambiguous when d2 credit lands on the terminal token, which is exactly where the outcome
reward sits (`batch.py:108`).

### 7.4 Frame, masking, padding and alignment (online batch, not offline re-tokenization)

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
* **Collision.** Frozen precedence (§4.2), one write per position; the online implementation
  asserts the precedence order (d2 > d1 > success) rather than relying on write order.

### 7.5 Parity and fixture battery before any GPU step

Zero-GPU, on the dependency-free port `src/tinylean_rl/rl/grpo.py` (a line-by-line copy of
`core_algos.py:261-324` whose docstring already requires exactness). The frozen credit semantics
live in `src/tinylean_rl/rl/process_credit.py` (φ constants, precedence, positioning, infra
censoring, counters) and the battery in `tests/test_v5_r001_process_credit.py`, whose last fixture
pins `assign_positions` against the offline oracle's `_token_credit` so the two collision
implementations cannot drift:

1. `TREATMENT(λ=0) ≡ CONTROL ≡ compute_grpo_outcome_advantage` bit-for-bit on fixture batches,
   including a batch with a `PREFIX_BEARING_FAILURE`-shaped candidate — this is the
   "V1-equivalent outcome-only objective" evidence (§3).
2. A collision fixture reproducing the precedence rule on a duplicated token position
   (`d2 > d1 > success`, one write, order-independent).
3. A padding fixture (mixed response lengths, right padding) proving no credit escapes the mask.
4. A censoring fixture proving a re-tokenization mismatch yields zero credit plus a counter, never
   a shifted credit.
5. An infra-censoring fixture proving: excluded from `ḡ`, loss masked to zero, counter recorded,
   `n_valid < 2` ⇒ `GROUP_SKIPPED_INFRA` with no group update — identical in both arms.

S2 (GPU) then repeats the λ=0 parity comparison on the *same frozen generated batch* with the
numeric comparison list of §10.3. The OPEN items rev. 2 carried here (λ, φ centering,
normalization, RECOVERABLE-only restriction, floors, shared clip range) are **resolved by the
owner's directive** and are no longer open.

## 8. Verifier policy: matched verification compute, shared by both arms (owner §8, §15)

Both arms issue **the same Lean verification request with the info tree enabled** for every
candidate; CONTROL ignores the process fields, TREATMENT consumes them. Same endpoint, same
container, same image digest, same timeout policy, same server configuration, same
`MAX_REPLS`/recovery/infra-censoring policy; actual verifier wall-clock is measured and reported
per arm, so the comparison cannot become "control = cheap verification / treatment = much more
verifier compute without accounting". The treatment verifier path is **not** optimized differently
from control's.

Required code change, identical for both arms. Today's *training* reward path
(`recipe/kimina_prover_rl/kimina_prover_rl/reward/reward.py:80-105` → `kimina_client`
`/api/check`, `infotree` unset, `timeout=60`, 8 × 40 workers) has two properties that are
unacceptable for R001:

1. it discards the info tree, and
2. `analysis.status != valid` — including server error, timeout and "no result" — becomes
   **reward 0.0** (`reward.py:92-105`), with `sync_client`'s `safe=True` path turning a transport
   exception into `error=str(e)` for the whole batch.

Item 2 contradicts §4.3 (an infrastructure outcome is censored data, never a failed proof). So
R001 must move both arms onto a verification path that (a) requests the info tree, (b)
distinguishes `verified / proof-fail / infrastructure` and censors the third class out of the group
denominator while keeping the group, and (c) keeps the V5-P001 oracle semantics (server-confirmed
timeout final, bounded isolated retries for infra only, canary gate, two-strike wedge censoring per
Amendment A). The evaluation path (`src/tinylean_rl/verifier/kimina.py:98, :145`, `/verify` with
`infotree_type="original"` plus `VerificationSession`) already does most of this and is the natural
parent. **OPEN (engineering, not science):** reuse the offline session inside training vs a new
training-path client with the same semantics — either way it is **one shared implementation for
both arms**, and the change is documented as an asymmetry versus V1, not versus CONTROL.

Measured cost of the info tree (V5-P001 Phase B, 2 476 serialized submissions, 1 attempt for
2 475 of them, dedicated container, `LEAN_SERVER_MAX_REPLS=1`):
mean **9.85 s**/candidate, median 2.14, p90 23.7, p95 54.7, p99 134.9, max 180.1 (client cap),
total 6.77 CPU-hours. V1's no-tree outcome path measured ~1.0 s/candidate
(`experiments/manifests/p3_0_complete.yaml` `verification_s_per_candidate`) and a 25.9 s/step mean
window in the frozen run (§13.1). **The info tree is roughly an order of magnitude more expensive
per candidate, and it is mandatory for both arms.** Budget consequences are in §13.

## 9. Length policy and diagnostics (owner §18) — mandatory in both arms

Pre-registered length diagnostics, in training logs **and** in every evaluation, for **both arms**:
response-token mean, median, P90 and P95; mean tactic count per candidate; EOS rate; max-token
truncation rate; length of successful proofs; length of failed proofs.

**No length reward, no length penalty, no length normalization** is introduced anywhere. Success
conditional on length bins is **descriptive only**; the primary capability endpoint is **not**
length-normalized.

Why this is a major secondary mechanism/safety endpoint: P001's recoverability is strongly
length-associated (first-quartile 0.1538 = 22/143 vs 0.8542/0.9375/0.9375). The concrete failure
mode to test is that an apparent treatment gain is a **length-avoidance artifact** — d2 = −0.10
penalizes every token from the first error onward, so a shorter wrong proof mechanically receives
less penalty than a longer one. The preregistered safety read: B's held-out gain must be reported
alongside Δ(mean tokens), Δ(P95 tokens), Δ(truncation rate), Δ(tactic count) and Δ(EOS rate); if
B's gain is confined to the short bins, that is evidence for `B_PROCESS_ONLY` with a length
mechanism, not for a capability GO.

## 10. Engineering feasibility: S0–S3 on fly122 (owner §10–§14, §19)

Formal node: **fly122, RTX 3080 10 GB** (48 SM, 62 GiB RAM, 16 cores). It remains the formal node;
nothing moves to fly90. The training stack matches `experiments/manifests/p3_0_environment.yaml`
(torch 2.7.0 / vllm 0.9.1 / transformers 4.53.3 / ray 2.48.0 / flash_attn 2.8.0.post2 /
kimina-client 0.2.1 / verl 0.5.0.dev0, submodule `e16b605e`).

**The owner has issued blanket authorization for S0–S3 (no further approval between stages).** All
four stages: consumed training/development prompts only; no capability holdout; no final reserve;
no formal scientific training result. S0–S3 completion does **not** authorize R001 (§19), and the
checkpoints they produce are **deleted after verification** and are never a starting point for a
formal run — the formal R001 restarts both arms from theta0.

### 10.1 S0 — configuration/static audit (no GPU, both arms)

Resolved configuration dump for **both arms**: trainable parameter count; parameter dtype; gradient
bytes; master-weight bytes; optimizer-state bytes; activation policy; gradient checkpointing flag;
CPU parameter offload; optimizer offload; rollout engine memory settings; vLLM
`gpu_memory_utilization`; microbatch sizes; token caps; `max_prompt_length`; `max_response_length`;
`n` (rollout samples per prompt). No model outcome is produced or recorded.

### 10.2 S1 — ≤ 3 optimizer steps on consumed training prompts (GPU)

Purpose only: memory / OOM / wall-clock / verifier-throughput / optimizer feasibility. **No
scientific metric may be interpreted** (no capability, no learning rate of change, no reward
curve). Records `nvidia-smi` peak, torch peak allocated/reserved, per-phase wall clock
(gen / old_log_prob / adv / update_actor / save, seconds/step), CPU RSS, vLLM KV probe.
Checkpoints deleted after the verification log is archived. On failure: at most one knob changed
per retry, each retry recorded in an explicit **diff table**.

* **Allowed knobs (memory engineering only):** CPU parameter offload; CPU optimizer offload;
  gradient checkpointing; activation offload; microbatch size; token-based microbatch cap;
  chunked/fused `lm_head`; vLLM memory utilization; sequential rollout/train phases.
* **Not allowed:** a different `n`; a different theorem batch; a different response cap; a
  different optimizer; a different LR; a different model; any change to the scientific reward.
* Final settings must be **identical for both arms**.
* **STOP_FOR_OWNER** if comparable semantics cannot fit 10 GB.

### 10.3 S2 — arm parity, ≤ 1 optimizer step per condition (GPU)

Mandatory first condition: **`TREATMENT(λ=0) == CONTROL`** on the same frozen generated batch
where possible, with numerical comparison of: the valid-candidate mask, `ḡ`, the outcome
advantages, the token advantage tensor, the policy loss, and the gradient checksum/norm — each
within a preregistered floating tolerance. Only then, as a second condition: the **λ=1
process-credit tensor construction** for engineering validation (counters non-zero, precedence
respected, mask respected). Neither condition may be interpreted as a reward or capability result.

### 10.4 S3 — online info-tree/mapping dry run (CPU/verifier, stored or smoke-generated responses)

On stored or smoke-generated **consumed-training** responses, run the full online chain: response
ids → response text → Lean info tree → tactic AST spans → generated-token offsets → first-token
mapping → collision resolution → process advantage tensor. Required counters: `mapped`,
`ambiguous`, `outside_response`, `retokenization_mismatch`, `conflict`, `infra_censored`,
`no_code`. No holdout contact.

### 10.5 The 10 GB arithmetic behind the stops

| term | bytes/param | size at 751 632 384 params |
|---|---|---|
| bf16 parameters | 2 | 1.40 GiB |
| fp32 master + Adam m + v | 12 | 8.40 GiB |
| bf16 gradients | 2 | 1.40 GiB |
| **static subtotal** | | **11.2 GiB** |
| lm_head logits, per 1 024 tokens of micro-batch, fp32 | | 0.58 GiB (3.1 GiB at 5 120) |

A 10 GB card exposes ≈ 9.5 GiB minus the CUDA/vLLM context, so **the static optimizer state alone
does not fit** without offload; and the V1 measurement confirms the direction: at the frozen shape
(`tb4 → 32 seq`, micro 2, vLLM util 0.40, `max_num_batched_tokens=5120`) the run peaked at
**24.41 GB allocated / 25.98 GB reserved** (`experiments/manifests/p3_0_complete.yaml`; also
`.cache/e019_train.log` 23.44–24.41 GB at the n=8 shape). Favorable audit items: **there is no
reference model** (V1 froze `kl: off` — no second 1.40 GiB weight copy, no ref-logprob pass beyond
old_log_prob), and the **process-credit tensor overhead is negligible**: one extra float32
`(32, 4096)` tensor ≈ 0.5 MiB per step plus a per-candidate integer position list; it never
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
scientific optimization semantics across both arms**, which is exactly what §7.5's fixtures and
S2's parity comparison test.

LoRA is a *semantically different* method (different parameterization, different V1 precedent) and
would itself need an owner decision; it is not an automatic fallback.

## 11. Seeds, horizon, checkpoints (owner §16, §17) — provisional until S1

* **Seeds.** **2 paired training seeds per arm preferred** (one seed per arm is insufficient for a
  strong claim). Seeds are paired (same seed → same data order, same sampling). **Not frozen**
  until the S1 GPU-hour estimates are in.
* **Horizon.** Provisional candidate band **40–60 optimizer steps**; not frozen. After S1 the
  pre-launch report must give measured seconds/step, the generation / verification / optimization
  fractions, and projected GPU-hours for 40 and 60 steps × 2 arms × 2 seeds; the owner freezes.
* **Checkpoints.** Chosen by rule in advance (final + a fixed periodic grid), evaluated on the same
  theorems with the same generation policy. **No early stopping based on holdout. No checkpoint
  picking from final capability outcomes.**
* **Models evaluated.** theta0, CONTROL final, TREATMENT final (+ every pre-registered
  intermediate). Same theorems, per-model seeds, n, temperature, top_p, max_response and verifier;
  all checkpoints evaluated on fly122 (the E023 host-harmonization precedent: cross-device vLLM
  sampling is not trajectory-identical).

## 12. Endpoints (owner §9, §13, §17)

**Primary.** Family-clean held-out theorem success, TREATMENT vs CONTROL, paired by theorem
(unit = theorem = family component, never candidate). Two endpoints are carried into the power
audit, and the choice is an owner decision at freeze:

1. **pass@n binary** (n = 8): theorem solved by ≥ 1 of the n candidates, exact McNemar.
2. **theorem-level continuous rate** `s_i(M) = verified_candidates_i(M) / evaluation_n` (n = 8),
   effect `Δ_rate = mean_i [s_i(T) − s_i(C)]`, paired over theorems.

**Secondary / mechanism** (explanatory, never substitutes): informative-group rate (IGR),
process-active-group rate (fraction of groups where the credit tensor is non-zero),
all-fail-groups-reactivated rate, per-class credit coverage (d1/d2/success token positions,
censored-mapping counts, conflict counts), training reward, entropy, KL, gradient norm, response
length, tactic count, verifier infrastructure rate. Process metrics are **diagnostics, never
capability evidence**: the evaluation stays blind to the process channel.

**Safety.** The length-behavior battery of §9, plus truncation rate and verifier infra rate per arm.

## 13. Compute accounting (owner §17)

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

### 13.2 Per-step estimate on fly122 (to be replaced by S1 measurements)

Generation: 3080 (48 SM) vs 3090 (82 SM) ⇒ ~1.3–1.9× slower ⇒ 115–170 s. Update with full
offloading and micro 1: 18.6 s ⇒ 45–90 s (CPU-Adam bound). Old log-prob: 6–10 s. Verifier at
32 candidates/step with the info tree: **315 s serialized** (`LEAN_SERVER_MAX_REPLS=1`, the frozen
V5 oracle setting) or ~105–160 s at 3–4 concurrent REPLs (16 cores; each Mathlib REPL is multi-GiB
RSS, so contention is a real timing risk — an owner decision on §8 fidelity, to be informed by
S1/S3 measurements).

| scenario | per step | 60 steps | note |
|---|---|---|---|
| both arms, serialized verifier | ≈ 480–640 s | ≈ 8–10.5 h/arm | verifier is 50–65 % of the step |
| both arms, 4 REPLs | ≈ 270–390 s | ≈ 4.5–6.5 h/arm | needs the §8 concurrency decision |

Evaluation per checkpoint (N theorems × n=8): N=112 ⇒ 896 candidates ≈ 0.7 h generation +
2.5 h serialized info-tree verification (0.6–0.9 h at 4 REPLs).

### 13.3 Projection to be produced after S1 (owner §17)

| configuration | GPU-hours | wall clock |
|---|---|---|
| 2 seeds × 2 arms × 40 steps, 5 checkpoints | from S1: `seconds_per_step` × 40 × 4 + evaluation | — |
| 2 seeds × 2 arms × 60 steps, 5 checkpoints | from S1: `seconds_per_step` × 60 × 4 + evaluation | — |
| 1 seed × 2 arms × 60 steps (pilot fallback) | ≈ half of the above | — |

These rows are deliberately formulas, not numbers: the directive forbids freezing seeds/horizon
before S1 reports the measured seconds/step and phase fractions. Rev. 2's order-of-magnitude table
(≈ 30–70 GPU-hours across configurations, verifier-dominated) stands only as a sanity band.

The tension remains: **the affordable horizon is small and the verifier is the dominant new cost**,
while the historical evidence (§14.4) says a 60-step run of this recipe changes family-clean
held-out success by ≈ 0.

## 14. Holdout and power (owner §6–§9) — H1 decided

### 14.1 Decision: H1 new external family-clean holdout (owner §6)

The owner has decided: **H1 — a new external family-clean holdout, NuminaMath-LEAN preferred.**
Explicitly forbidden: reusing the V3-R001 128-component formal sample; unsealing the 93-component
V3 reserve; using E024/MiniF2F as the primary holdout. **The 93 remain SEALED.**

Why this is necessary — the promptset family-clean pool is exhausted
(`experiments/manifests/v3/V3-R001_family_clean_pool.json`, pool hash `d2d3b58e…`):

| reading | capacity | status now |
|---|---|---|
| `owner_literal` | 86 | spent |
| `owner_literal_wider_v1` (V3's primary) | 56 | spent |
| `owner_literal_all_reservations_honored` | 0 | — |
| `registry_labels_maximal` | 0 | — |
| `consumed_only` | 221 | **128 + 93 = 221, fully allocated** |

The V3-R001 formal sample (128 components) and the sealed final reserve (93 components) are
disjoint (measured overlap 0) and sum to the whole 221-component `consumed_only` pool.
**Therefore: 0 never-consumed promptset components remain.**

### 14.2 H1 construction must be outcome-free (owner §7)

Allowed pipeline, in this order, with **no model generation anywhere in selection**:

1. source eligibility;
2. canonical statement normalization;
3. compile/elaboration eligibility via `by sorry` or equivalent **statement-only** validation;
4. context/token-length eligibility;
5. family-component construction;
6. one theorem per component;
7. deterministic hash order.

**Explicitly forbidden during construction:** any theta0 (or other model) proof generation to
select theorems; "no theorem may be selected because theta0 fails/succeeds". The registry must
record `theta0_outcomes_used: 0`.

### 14.3 H1 registry and capacity audit (owner §8)

Deliverable: an **external holdout registry + capacity audit** artifact reporting capacity at least
for **N = 128, 192, 256** (planning target N = 192; **not frozen**), with the number of eligible
family components and their **source / category / context-length distributions**. No proof
generation, no sealed-reserve contact.

Implementation: `scripts/v5_h1_build_pool.py` (`build`/`scan`/`merge`) with fixtures
(`tests/test_v5_h1_pool.py`); registry `experiments/manifests/v5/h1_holdout_pool.json`. Corpus:
the local AI-MO/NuminaMath-LEAN mirror (104 155 rows / 104 119 unique formal statements;
`data/raw/numinamath_lean/data/train-00000-of-00001.parquet`). Contamination reference: the frozen
Kimina Promptset universe (7 620 statements / 1 710 name families) — every V1–V3 consumed/reserved
artifact is a subset of it — closed at family-component level (strong text OR skeleton OR name
family). The scan stage performs statement-only validation (`statement + "sorry"`, **no info
tree**) against the pinned oracle and records per-candidate verdicts; it is not the R001
verification path.

**Eligibility semantics (implementation decision, 2026-09-27).** Lean's default
`autoImplicit = true` silently rebinds a free identifier in a statement as an implicit universal
variable: a broken fragment such as `theorem t (v : ℝ) (h : ... = ... (v + w))` with an unbound
`w` "compiles" as a statement that is false in general. Measured on the first 256 hash-ordered
candidates, 15/256 = 5.9 % of statements are degenerated in exactly this way. The canonical
statement therefore pins `set_option autoImplicit false` immediately after the imports (Mathlib's
own convention; source rows that mention `autoImplicit` at all are excluded as
`explicit_autoimplicit`), so the compile scan, any prompt built from the registry, and any later
R001 evaluation all share one semantics, and an unbound identifier is a rejection reason rather
than a silent universal. This is an eligibility-quality guard only: it consumes no model output,
touches no theorem of the corpus, and changes no capacity-relevant count beyond 6 rejected rows.

**Measured and frozen (2026-09-27).** The pipeline has now been executed through the merge stage:
104 155 parquet rows → 98 353 eligible statements → 93 068 pre-scan family components. The
statement-only scan consumed the first 4 280 hash-ordered candidates (early stop at the
target of 4 096 compiled) with 4 103 compiled / 177 not compiled at 7.56 statements/s over 9.4 min
on fly122 (`tinylean-rl-lean-server-r001`, the pinned digest), forming **4 094 family components**
(4 086 singletons, 7 pairs, 1 triple). One theorem per component gives capacity **≥ 4 094** and all
three planning sizes are supported — 128 ✓ / 192 ✓ / 256 ✓. This is a lower bound from the scanned
window, not the corpus capacity: the pre-scan pool held 93 068 components and the merge is
monotone in the window, so a deeper scan can only add. Distributions of the verified pool: sources
`olympiads` 2 041, `unknown` 549, `aops_forum` 502, `secondary_math` 416, `inequalities` 172,
`olympiads_ref` 108, `amc_aime` 80, `math_train` 80, `number_theory` 57, `math_test` 49,
`synthetic` 18, `cn_k12` 16, `cn_contest` 15; problem types Algebra 1 569 / Number Theory 1 118 /
Inequalities 617 / Calculus 190 / other 60; ground-truth type complete 348 / statement 339 /
with_sorry 63 / empty 3 353; prompt length (chat template incl. the `sorry` placeholder) mean 246.2
tokens, max 956 ≤ `MAX_PROMPT_LENGTH` 1 024. Artifact
`experiments/manifests/v5/h1_holdout_pool.json`, `pool_hash 1405d8e4…`, scan sha256 `8c54d9dd…`,
stage `H1_EXTERNAL_POOL_VERIFIED`; compliance fields in the artifact: `theta0_outcomes_used: 0`,
`sealed_reserve_touched: 0`, `v3_r001_128_reused: false`. No proof generation took place at any
stage of construction.

### 14.4 Baselines and the historical effect band

| anchor | value | source |
|---|---|---|
| θ0 pass@8, family-clean (24/112 analyzed) | **0.2143** | `V3-R001_results.json` |
| θ0 per-candidate verified rate, family-clean | 0.1071 | same |
| θ0 truncation rate (candidates at 4096) | 508/896 = 0.567; mean generated tokens/candidate 3 614 | same |
| θ0 pass@4 on the older M1 sealed 128 | 0.3594 | `e023_holdout.yaml` |
| V1 step-60 arms vs θ0, pass@4 | 0.3438 / 0.3516 / 0.3672 | same |
| V1 theorem-level paired deltas | +0.0039 / −0.0137 / −0.0176, all CIs straddling 0; cross-seed mean **−0.0091**, 1/3 positive signs | same |
| V1 30-step pilot (64 theorems, n=8, not family-clean) | +0.0098, CI [−0.0254, +0.0430], McNemar p = 1.0 | `p3c_fixed_eval.yaml` |
| **model-vs-model binary discordance ρ** | E023 all 6 pairs: 0.086–0.125; E018: 0.094–0.109 | recomputed from the per-record artifacts |

**This resolves a discrepancy carried by rev. 1:** rev. 1's motivation cited an interim "θ0 50 % →
55.5 % at +30 steps" figure. No canonical artifact supports it. The canonical V1 record shows **no
consistent family-clean capability gain** at 60 steps (`e023_holdout.yaml`) and the 30-step pilot is
POSITIVE-INCONCLUSIVE (`p3c_fixed_eval.yaml`). Any R001 statement of the form "V1 worked, so a
differential should be small" is inadmissible; the honest prior is "outcome-only RLVR at this
budget is ≈ 0 on family-clean holdout".

### 14.5 Power audit v2 (owner §9)

The audit must cover **both endpoints** of §12, at **N = 128, 192, 256**, and must include:

* paired family-bootstrap CIs (resampling by family component);
* a paired permutation / sign-flip test where valid;
* a **historical empirical simulation** using E018/E023-style outcome vectors (the measured ρ and
  per-theorem rate distributions) rather than only the analytic McNemar approximation;
* the exact-McNemar detectable-difference table (§14.6) as the binary endpoint's companion.

The audit **does not freeze a practical threshold** — thresholds remain an owner freeze decision.
It reports `recommended_N` and a `proposed_practical_gate` as recommendations only.

**Executed (2026-09-27), artifact `experiments/manifests/v5/v5_r001_power_v2.json`.** Four
historical outcome pools (E018 θ0 n=8, E018 step-30 n=8, E023 θ0 n=4, V3-R001 θ0 n=8; 64–128
theorems, per-theorem rates Jeffreys-smoothed), 2 000 Monte-Carlo replications per cell, treatment
models additive/multiplicative calibrated to the target mean Δ, α = 0.05 one-sided, 80 % power.
The sign-flip test is exact (DP convolution over the Rademacher magnitudes; self-test equals brute
force), the binary test is the exact double-binomial McNemar of
`scripts/v4_p001_power.py::exact_rule_power` (known value reproduced), and the bootstrap resamples
by theorem — which under H1's one-theorem-per-component construction *is* the family bootstrap.
Detectable deltas at 80 % power (multiplicative model), rate endpoint / pass@n endpoint in pp:

| pool | N = 128 | N = 192 | N = 256 |
|---|---|---|---|
| E018 θ0 (n=8) | 3.70 / 5.93 | 3.20 / 4.81 | 2.92 / 3.92 |
| E018 step-30 (n=8) | 3.82 / 6.43 | 3.29 / 5.20 | 2.92 / 4.37 |
| E023 θ0 (n=4) | 4.19 / 7.65 | 3.47 / 5.97 | 3.21 / 5.34 |
| V3-R001 θ0 (n=8) | 3.62 / 5.42 | 3.13 / 4.24 | 2.72 / 3.65 |

**Consequences.** (i) The **continuous endpoint is the sensitive one**: 3.1–4.2 pp at N = 128–192
where the binary endpoint needs 4.2–7.7 pp; §14.6's ρ-band table (5.2–8.3 pp) is reproduced by the
analytic module and remains the binary companion. (ii) At N = 192 the bootstrap-CI decision rule is
markedly more conservative than the sign-flip (0.34–0.48 vs 0.45–0.59 power at Δ = 2 pp; 0.86–0.96
vs 0.92–0.98 at Δ = 4 pp) — this is itself information for the pre-registered decision rule.
(iii) Simulated null rejection at Δ = 0 stayed ≤ 0.046 (binary) / ≤ 0.049 (sign-flip) across all
pools and N, so the exact tests show no anti-conservative drift. (iv) `recommended_N`: an owner
decision; the audit records N = 192 as the planning target and prices N = 256 as a reduction of the
detectable rate delta by about 1 − √(192/256) ≈ 13 %. (v) `proposed_practical_gate`: NOT frozen —
the audit supports reporting the rate endpoint with its paired bootstrap CI and the exact sign-flip
p-value and treating the binary endpoint as the confirmatory view; the threshold choice needs S1's
measured between-arm noise first (owner §17). `practical_threshold_frozen: NO` is recorded in the
artifact.

### 14.6 Rev. 2's exact-McNemar table (retained; binary endpoint)

Exact McNemar (conditional binomial on the discordant pairs), 80 % power, one-sided α = 0.05
(two-sided 0.05 in brackets), detectable difference in **pp of theorems solved by ≥ 1 of n
candidates**, at the measured discordance band:

| N | ρ = 0.09 | ρ = 0.12 | ρ = 0.13 |
|---|---|---|---|
| 56 | not resolvable | 11.9 | 12.4 |
| 86 | 8.3 [n/r] | 9.7 [10.5] | 10.1 [10.9] |
| 96 | 7.9 [8.5] | 9.2 [10.0] | 9.5 [10.4] |
| 112 | 7.4 [7.9] | 8.5 [9.3] | 8.8 [9.7] |
| 128 | 6.9 [7.5] | 7.9 [8.7] | 8.2 [9.1] |
| 192 | 5.6 [6.2] | 6.5 [7.2] | 6.7 [7.5] |
| 221 | 5.2 [5.8] | 6.0 [6.8] | 6.3 [7.0] |

Sample sizes for 80 % power: 8 pp ⇒ N = 94–135; 10 pp ⇒ N = 81–88; 6 pp ⇒ N = 167–241;
5 pp ⇒ N = 240–347 (ρ 0.09–0.13). Cross-checked against the frozen V4 power rule
(`scripts/v4_p001_power.py::exact_rule_power`) at N = 128: ρ 0.10 → 7.25 pp vs 7.5, ρ 0.15 → 8.82
vs 9.0, ρ 0.20 → 10.23 vs 10.5, ρ 0.30 → 12.45 vs 12.5.

**Consequences that must be stated rather than hidden.**
1. An N ≈ 56–86 holdout cannot carry the primary test at all (or only ≥ 10 pp effects), and
   `V3-R001_power.json` already learned this lesson for a different endpoint: a gate that is
   unpowered by construction manufactures a NO-GO.
2. At N = 128 the binary design resolves ≈ 8 pp. The historical whole-recipe effect is ≈ 0–1 pp.
   So a **C_NO_METHOD_GAIN result is the modal expectation even if the process channel works**,
   and a GO requires the process channel to produce an order-of-magnitude-larger effect than
   outcome-only RLVR has ever produced here.
3. The legitimate levers are: more theorems (H1 capacity, §14.3), more candidates per theorem at
   evaluation (variance of the per-theorem rate falls like 1/n, at linear verifier cost), or an
   honest pilot label. Choosing thresholds to make a pass likely is not a lever.
4. **OPEN (owner):** exact thresholds are deliberately **not** frozen yet — conditioned on power
   audit v2 and the compute audit, neither of which is final until §14.3 and §13.2 are settled by
   execution.

## 15. Statistics and taxonomy

* **Primary tests.** Exact McNemar on per-theorem paired indicators for the pass@n endpoint; the
  paired mean of the theorem-level rate `Δ_rate` with a family-component bootstrap CI for the
  continuous endpoint. Unit = theorem = family component, never candidate. Helpers already in-repo
  and reused verbatim: `mcnemar_exact`, `paired_bootstrap`, `cluster_bootstrap`, `win_tie_loss`
  (`src/tinylean_rl/evaluation/p3c_stats.py`, `src/tinylean_rl/evaluation/v4_stats.py`; "The
  pairing unit is the theorem (never the candidate)").
* **Effect size.** B − A in percentage points of theorems solved (binary) and in mean rate points
  (continuous), with 95 % paired bootstrap CIs resampled **by family component**, plus the
  win/tie/loss table. Threshold on the CI lower bound deferred to the freeze.
* **Pre-registration.** Test, threshold, seed schedule and censoring rules are fixed before any
  outcome; the test is not chosen after seeing outcomes; training reward is never primary evidence.
* **Taxonomy (names frozen, thresholds OPEN).**
  * `A_CAPABILITY_GO` — B beats A on the primary endpoint by ≥ the preregistered practical
    threshold with the CI rule satisfied, and the §9 length-safety battery does not explain it away.
  * `B_PROCESS_ONLY` — the process channel demonstrably changes the training signal and mechanism
    endpoints (process-active rate, all-fail reactivation, IGR) without a capability gain.
  * `C_NO_METHOD_GAIN` — neither capability nor a mechanism change attributable to the credit.
  * `D_INCONCLUSIVE` — the sample/budget could not carry the test (censoring rate, verifier
    infrastructure, power). **Not** a licence to redraw the holdout or re-cut the denominators.
  No post-outcome rescue path exists.

## 16. The pre-launch report: `V5_R001_PRELAUNCH_AUDIT_V2` (owner §20)

Required field list, exactly:

| field | content |
|---|---|
| `reward_definition` | the frozen §3/§4/§7 formulas and constants (λ = 1, φ, precedence, censoring) |
| `holdout_H1` | registry path + hash; capacities at 128/192/256; `theta0_outcomes_used: 0`; `sealed_reserve_touched: 0` |
| `power_v2` | both endpoints at N = 128/192/256; `recommended_N`; `proposed_practical_gate` |
| `S0` | configuration audit tables (both arms) |
| `S1` | attempts + diff table; `final_memory_settings`; `peak_VRAM`; `seconds_per_step`; `outcome_control_fit_10GB` |
| `S2` | `lambda0_parity` (numeric comparison), `loss_delta`, `gradient_delta`, `lambda1_tensor_valid` |
| `S3` | the seven counters + status |
| `compute_projection` | 40 / 60 steps × 2 arms × 2 seeds |
| `recommended_seeds_per_arm` | from the S1 cost |
| `recommended_horizon` | from the S1 cost |
| `recommended_holdout_N` | from power v2 + H1 capacity |
| `remaining_blockers` | explicit list |
| `V5_R001_FORMAL_TRAINING_AUTHORIZED` | `NO` |

## 17. Ordering: what has to happen before this draft can be frozen

1. Owner decisions already resolved by the directive: reward definition (§3/§4/§7), holdout
   (§14.1 H1), smoke authorization (§10), control wording (§3).
2. Execute, then STOP: H1 registry + capacity audit (§14.3 — **done**, stage
   `H1_EXTERNAL_POOL_VERIFIED`, capacities 128/192/256 supported) → power audit v2 (§14.5 —
   **done**, artifact `v5_r001_power_v2.json`, no threshold frozen) → intervention implementation
   + fixtures (§7 — module `process_credit.py` and the §7.5 battery landed; the online adapter and
   the training entry remain) → S0 → S1 → S2 → S3 (§10) → `V5_R001_PRELAUNCH_AUDIT_V2` (§16), with
   commit/push/sync of the revised preregistration and the engineering artifacts.
3. Still-OPEN owner decisions at freeze: horizon (40 vs 60 from S1), seeds (2 vs 1 per arm),
   holdout N (128/192/256), exact thresholds/gates, verifier REPL concurrency.
4. Freeze: this document rewritten as `V5-R001_preregistration.md` with every OPEN resolved, the
   power artifact committed, seeds/horizon/checkpoints fixed, registry entry `PREREGISTERED`, and
   the owner's freeze signature.
5. Owner launch authorization — a separate act from the freeze.

## 18. Corrections relative to rev. 2

1. **Intervention frozen by owner decision**: λ = 1 primary, φ constants (+1 / −0.05 / −0.10),
   `A_outcome_i = g_i − ḡ`, `A_process_i,j = φ_i,j − ḡ` at the first tactic token only; no λ
   tuning, no φ centering, no summation into a response-level reward, no normalization search
   (rev. 2 §7.5 carried these as OPEN).
2. **Credit applies wherever defined** — `PROCESS_STRUCTURED_RECOVERABLE` is a P001 analysis
   predicate, not an R001 eligibility filter (rev. 2's question about limiting credit to
   `PREFIX_BEARING_FAILURE`-class candidates is void).
3. **Collision semantics confirmed frozen** with direct online-batch fixtures required, not only
   the offline rule.
4. **Infra-censoring semantics fixed**: excluded from `ḡ`, loss masked to zero, counter,
   `n_valid < 2` ⇒ `GROUP_SKIPPED_INFRA`, no resampling, identical in both arms (rev. 2 left the
   group-denominator treatment open).
5. **Control wording corrected**: never "byte-identical to V1"; "V1-equivalent outcome-only
   objective under the shared R001 execution stack", with V1 reproduction tested on non-infra
   fixtures.
6. **Holdout decided: H1** external family-clean holdout (NuminaMath-LEAN), outcome-free
   construction pipeline fixed, capacities to be reported at 128/192/256; H2/H3/H4 removed as
   options; the 93-reserve stays sealed.
7. **Power audit v2** adds the continuous theorem-level rate endpoint, family-bootstrap CIs,
   permutation/sign-flip, and a historical empirical simulation; no threshold frozen.
8. **S0–S3 authorized with per-stage specs**, including the blanket authorization, the
   allowed/not-allowed knob lists, one-knob-per-retry with a diff table, identical final settings
   for both arms, checkpoint deletion, and `STOP_FOR_OWNER` on 10 GB infeasibility.
9. **Seeds/horizon provisional** (2 paired seeds preferred; 40–60 step band) with a mandated S1
   GPU-hour projection before freezing.
10. **Length diagnostics mandatory** for both arms (mean/median/P90/P95, tactic count, EOS,
    truncation, successful/failed proof lengths), with no length reward/penalty/normalization.
11. **No scientific-training language in the status block**: S0–S3 completion does not authorize
    R001; smoke checkpoints are deleted and unusable; the formal R001 restarts both arms from
    theta0.

## 19. Prohibitions in force

* No scientific training launch, no weight update for capability evidence, no capability
  generation, no R001 launch — by this draft, by `PROCESS_SIGNAL_GO`, or by S0–S3 completion.
* No λ tuning, no φ centering, no credit summation into a response-level reward, no normalization
  search, no new d1/d2 search, no new historical surface, no new process definition, no new parser
  heuristic, no new process oracle before R001.
* No outcome-dependent holdout construction: no model generation in H1 selection; no theorem
  selected because theta0 fails/succeeds; no sealed-reserve contact.
* No use of the sealed V3 93-component reserve or any family-clean capability holdout without an
  explicit unsealing directive.
* No modification of any frozen V5-P001 artifact, and no retrospective alteration of the P001
  classification or its `if_GO` field; the canonical analyzer is frozen after its metric-emitting
  run, so any analyzer change needs a new owner-approved amendment.
* No process-reward rescue path: no gate, threshold or denominator changes after an outcome.
* No moving formal training to fly90.
* No continuation from smoke checkpoints; engineering checkpoints are deleted after verification.

## 20. Provenance

* V5-P001 canonical result `experiments/manifests/v5/V5-P001_results.json` (sha256 `ab46c12d…`),
  classification `PROCESS_SIGNAL_GO`, analysis head `7d0b139` over FreezeA (execution head
  `5b1c5d2`); memo `docs/v5/V5-P001_result.md`; count reconciliation
  `docs/v5/V5-P001_count_reconciliation.md`; oracle and conventions
  `docs/v5/process_oracle_design.md`; amendments A and B; owner review and pre-launch directive
  recorded in `experiments/manifests/v5/registry.yaml`.
* Phase-B per-submission records `runs/v5_p001_process/raw/*.json` (2 476 files) — the source of
  the §8 verifier latency statistics.
* H1 artifacts: `scripts/v5_h1_build_pool.py`, `tests/test_v5_h1_pool.py`,
  `experiments/manifests/v5/h1_holdout_pool.json` (stage `H1_EXTERNAL_POOL_VERIFIED`, `pool_hash
  1405d8e4…`); raw scan `experiments/results/v5_h1/h1_compile_scan.jsonl` (sha256 `8c54d9dd…`) and
  `h1_compile_scan.meta.json`; candidates `experiments/results/v5_h1/h1_candidates.jsonl`; corpus
  `data/raw/numinamath_lean/data/train-00000-of-00001.parquet` (AI-MO/NuminaMath-LEAN mirror,
  sha256 `cddeac3b…`); Promptset reference
  `data/raw/kimina_promptset/data/train-00000-of-00001.parquet`.
* Pool/power/holdout artifacts (V3): `experiments/manifests/v3/V3-R001_family_clean_pool.json`,
  `v3_r001_formal_sample.json`, `v3_final_holdout_reserve.json`, `V3-R001_power.json`,
  `V3-R001_predictions.json`, `V3-R001_results.json`.
* V1 evidence: `experiments/manifests/e023_holdout.yaml`, `p3c_fixed_eval.yaml`,
  `p3_0_complete.yaml`, `p3_0_environment.yaml`, `.cache/e019_train.log`.
* Stack: `third_party/kimina-prover-rl` @ `e16b605e…`, `src/tinylean_rl/rl/grpo.py`,
  `src/tinylean_rl/rl/process_credit.py` (frozen R001 credit/advantage semantics, owner §1-§4),
  `tests/test_v5_r001_process_credit.py` (the §7.5 battery, incl. the offline-oracle parity pin),
  `src/tinylean_rl/evaluation/p3c_stats.py`, `src/tinylean_rl/evaluation/v4_stats.py`,
  `src/tinylean_rl/verifier/kimina.py`, `scripts/v5_process_oracle.py`,
  `scripts/v4_p001_power.py`, `scripts/v5_r001_power_v2.py`.
* This draft is a coordination artifact: it records the authorized design audit and freezes
  nothing. Rev. 2 was committed at `4520413`; rev. 3 implements the owner's pre-launch directive.
