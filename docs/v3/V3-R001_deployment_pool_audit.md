# V3-R001 — Deployment Pool Audit (§4, before preregistration)

> **Status: complete, read-only.** No rollout, no RL step, no optimizer update, no model weights
> loaded (tokenizer only). Machine-checked by `scripts/v3_r001_pool_audit.py`
> → `experiments/manifests/v3/V3-R001_deployment_pool_audit.json`.
> Authorized by owner directive 2026-09-24 §4 and §17 step 3.

## Why this file exists

The archived R001 training draft carried an unverified premise: that V1 sampled
*"uniformly over the 24,418 parquet rows"*. The owner forbade inheriting that unless the trainer
source proves it (`禁止直接沿用旧 draft: uniform over 24,418 rows 除非 trainer 源码明确证明如此`).
It does not. Both halves of the sentence are wrong — the population is not 24,418 and the
distribution over theorems is not uniform. R001's prospective unit had to be settled from code and
logs, not from inheritance.

## 1. The sampling unit is the parquet **row**, of the **filtered** dataframe

18 citations re-read from the pinned verl tree at run time (`e16b605e…521a`, tree clean, and
`import verl` resolves to that same third-party tree); all 18 verified verbatim
(`code_evidence.all_citations_verified_verbatim: true`).

| Step | Code |
|---|---|
| load | `verl/utils/dataset/rl_dataset.py:135` `load_dataset("parquet", data_files=parquet_file)["train"]` |
| concat | `:137` `concatenate_datasets(dataframes)` — no dedup, no groupby, no statement-level index |
| filter | `:141` → `:143-180` `maybe_filter_out_long_prompts`, keeps `doc2len(doc) <= max_prompt_length` |
| length | `:193` `return len(self.dataframe)` |
| index | `:220` `row_dict: dict = self.dataframe[item]` — **positional row access** |
| sampler | `verl/trainer/main_ppo.py:366-369` `elif data_config.shuffle:` → `RandomSampler(generator=…manual_seed(data.get("seed", 1)))` |
| loader | `verl/trainer/ppo/ray_trainer.py:528-534` `StatefulDataLoader(batch_size=train_batch_size, drop_last=True, sampler=sampler)` |
| n | `:1185` `gen_batch.repeat(repeat_times=rollout.n, interleave=True)` |

`statement_id` is only an *payload field* (`prepare_data.py:52` `"index": sample["statement_id"]`);
nothing in the data path groups or deduplicates by it. **Unit = row.**

## 2. The sampled population is the filtered one: 24,246 rows / 7,613 statements

Recomputed here with the θ0 tokenizer and the trainer's own rule
(`len(apply_chat_template(prompt, add_generation_prompt=True)) <= 1024`), and matching the numbers
V1 printed (`dataset len: 24418` → `filter dataset len: 24246`,
`Size of train dataloader: 6061` = 24,246 // 4):

| Quantity | Value |
|---|---|
| rows on disk | 24,418 |
| unique `statement_id` on disk | 7,620 |
| rows the sampler could actually draw | **24,246** |
| unique statements that could appear | **7,613** |
| rows dropped by the 1,024-token filter | 172 |
| statements with exposure probability **exactly 0** | **7** |
| prompt token length min / max | 102 / 2,365 |

The 7 zero-exposure ids are listed in the JSON. They can never acquire a historical label, so a
controller trained on V1 rollouts has never been asked about them; they are legitimate R001 test
theorems **only if** R001 generates them under the same 1,024-token rule (see §5).

## 3. `unique statement_id` ≡ `canonical prompt` — the owner's preferred unit is unambiguous

`distinct_content_rows_in_parquet = 7,620` and `statements_with_more_than_one_distinct_prompt_text = 0`:
the 16,798 "extra" rows (24,418 − 7,620) are **byte-identical copies** (same `prompt`, same
`extra_info`, same `data_source`, same `name`). So "one unique, stable, interpretable theorem unit"
and "one deduplicated prompt" select exactly the same set here; no choice has to be made between
them. R001 uses `statement_id` as that unit.

Per-statement row multiplicity: min 1, mean 3.204, max 54
(histogram: 5,099 statements ×1, 1,001 ×4, 823 ×6, 470 ×9, 99 ×8, 71 ×**54**, …).

## 4. But the induced distribution over theorems is **multiplicity-weighted**, not uniform

`P(statement) = surviving_rows(statement) / 24,246`. Row-unit sampling therefore over-samples
repeated theorems by up to 54×. Two independent confirmations:

* **Realized exposure.** Weighting each pool statement by its number of rows gives a pool mean of
  **3.204** rows/statement; the 720 groups V1 actually rolled out sit on statements whose
  group-weighted mean multiplicity is **11.315** — high-multiplicity theorems are exactly the ones
  that got drawn, which uniform-over-statements would not produce.
* **Bit-exact replay.** Re-drawing V1's own first 240 rows as
  `torch.randperm(24246, generator=manual_seed(data.get("seed",1)))[:240]` mapped through the
  filtered row order reproduces, **for all three seeds**, precisely the realized prompt set —
  and the realized per-step groups (180/180 steps contain the same 4 theorems). The `data.seed`
  values were read from the logs, not assumed: `seed1` has no `data.seed` key → default **1**;
  `seed2` → **20260918**; `seed3` → **20260919**. Landing on the same 240 of 24,246 rows three times
  over by any other mechanism is not plausible, so the row-unit RandomSampler is now an *empirical*
  account of V1, not an interpretation of its source.
  The only residual difference is the order of theorems *inside* a step, which is an artifact of
  `build_records()` sorting each group, not of the trainer
  (`realized_sequence_equals_sorted_replay: true`).

Consequence for R001: the controller's training marginal is multiplicity-weighted over the filtered
rows, while owner §7 requires a **uniform** draw over a clean statement pool. That is a change of
measure which R001 must *state as a covariate-shift caveat*, not hide — and it is one reason a
prospective test is needed at all.

Per-step geometry, confirmed against the reconstruction: 4 prompt rows/step × n=8 = 32 sequences/step,
4 rows consumed per step × 60 steps = 240 rows per seed, `steps_with_a_repeated_statement_inside_the_same_step = 0`
(without-replacement sampling within an epoch makes a within-step repeat essentially impossible).

## 5. Historical label coverage of the pool (input to §5 pool construction)

612 of 7,613 draw-able statements ever received a rollout ⇒ **8.0 %** labelled coverage; the three
seeds are near-disjoint (per-seed unique 227 / 229 / 227; pairwise intersections 27 / 17 / 32,
Jaccard 0.039–0.076). The 244-statement slice V1 used as `data.val_files` is a **disjoint id space**
from the train pool (`overlaps_train_statements: 0`), so it can never supply R001 candidates
directly — but its family components can still overlap the train pool, which is why the owner's §5
exclusion list names it.

## 6. Corrections to the archived draft (frozen history, left unedited)

* `docs/v3/archive/V3-R001-training-draft_SUPERSEDED.md` and
  `experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json` use the **on-disk**
  pool (24,418 / 7,620) as the deployment population. The sampled population is the token-filtered
  one (24,246 / 7,613). Every *row-level* diagnostic in that artifact — pool multiplicities,
  `scored_statement_share_of_pool_rows`, the row-level ESS/entropy block, `TRIAL_DRAWS` weights — is
  computed on an unsamplable population and does **not** transfer to a future sampler. The
  theorem-level quantities are unaffected. The archive is not edited; this audit is the record.
* `scripts/v3_r001_offline_freeze.py` compares the *raw* parquet against `prepare_data.py`'s output
  and concludes nothing was dropped. That comparison structurally cannot see
  `RLHFDataset.maybe_filter_out_long_prompts`, which is where the 172 rows / 7 statements are lost.
* The `epsilon = 0` sampler in that draft also has zero support on every un-scored statement —
  independently abandoned by the owner (§13), and now additionally wrong about its own population.

## 7. What R001 takes from this audit

1. **Unit**: one unique `statement_id` = one canonical prompt (provably identical here).
2. **Eligibility**: a candidate must pass the trainer's own 1,024-token prompt filter, or it is not
   a legitimate test theorem for a sampler-derived claim (7 pool statements are excluded on this
   ground alone).
3. **Covariate shift**: register the multiplicity-weighted training marginal (pool mean 3.204
   rows/statement, 11.315 group-weighted among realized) vs the uniform evaluation measure as a
   preregistered limitation.
4. **No inherited denominators.** Every population number in the R001 preregistration is recomputed
   from this audit or from the pool build, never copied from the archived draft.

## Prohibitions respected

No RL optimizer update · no rollout generated · no 10 GB training smoke · no offload experiment ·
no MLP / new layer / pooling change · no new controller task · no Track B / candidate ranker /
repair policy · no frozen V2 or D001 artifact modified · nothing pushed to `origin`.
