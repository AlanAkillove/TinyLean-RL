# V2 Track B — B1 Budget-Semantics Audit (V2-E001): Results and Gate Decision

**Status**: COMPLETE — canonical single-candidate path **PASS**.
**Worker**: fly122 (`worker/v2-allocator`); base `main2` @ `ca804f1b145ae5b89a1da39ae83a160bcb10cbc3`.
**Run date**: 2026-09-19 (stage 1 14:49:25 UTC -> 18:19:09 UTC; stage 2 the same day).
**Manifest**: `experiments/manifests/v2/V2-E001.yaml` (status COMPLETE; raw run ID
`V2-E001` is permanent — canonical indexing may later register the Track alias
`V2-B001`; no running manifest was renamed).

## 1. Question

Under fixed decoding settings, does a direct generation at budget b equal the
b-token prefix of a 4096-token generation from the same seed — **compared at
token-id level** — so that B2 may derive cheap budget labels by truncating 4096
trajectories instead of generating every budget directly?

Pre-registered conditional probes asked the same question for two faster batch
engines (HF batch=4, vLLM 0.9.1) intended for B2 throughput.

## 2. Method (frozen before launch; summary — manifest is authoritative)

- **Theorems**: 16 development theorems from Promptset train (parquet rev
  `3009c548d90160d0f5e963d72238610c6732f812`; selection seed 20260919; all
  V1-used sets excluded). Full list with sha256:
  `experiments/manifests/v2/v2_e001_theorems.json`.
- **Seeds**: 2 replicates per theorem; `gen_seed = 20260919 + rank * 8 + replicate`.
- **Stage 1 (canonical path)**: transformers 4.53.3 + torch 2.7.0 / fp16,
  `batch_size=1`, `num_return_sequences=1`, `do_sample=True`, `T=1.0`,
  `top_p=1.0`, `set_seed(gen_seed)`, `max_new_tokens = b`; runs per candidate:
  `{512-det1, 512-det2, 1024, 2048, 3072, 4096}` (192 generations).
- **Checks (token ids only)**:
  - *determinism*: `512-det1` vs `512-det2` byte-identical token lists;
  - *prefix*: `direct_b == full4096[:len(direct_b)]` for b in {512, 1024, 2048, 3072}.
- **Stage 2 (conditional)**: HF batch=4 and vLLM 0.9.1 probes on ranks 0-3,
  replicate 0: internal determinism, internal prefix, and token identity vs the
  stage-1 single-candidate streams.
- **Verdict rule (frozen)**: PASS requires 32/32 determinism and 160/160 prefix;
  CONDITIONAL allows ragged determinism with >=95% prefix; otherwise FAIL.

## 3. Stage-1 results — PASS

| Check | Result |
| --- | --- |
| determinism (32 pairs) | **32/32 PASS** |
| prefix (160 comparisons) | **160/160 PASS** |

- EOS early-stops are included and do not break prefix equality (a direct-b
  run stops at the same token as the 4096 run): 9/32 of the 4096 runs ended at
  EOS; none stopped before 1508 tokens.
- Cost on fly122 (RTX 3080 10 GB): 343,895 generated tokens in 12,571 s of
  generation time; average per-budget runtime 18.8 s (512) / 37.4 s (1024) /
  73.6 s (2048) / 107.2 s (3072) / 137.1 s (4096).

**Gate consequence**: prefix-derived budget labels are **valid** for B2 on the
canonical single-candidate path.

## 4. Stage-2 probes (batch engines)

| Engine | internal determinism | internal prefix | identical to single path |
| --- | --- | --- | --- |
| HF batch=4 | 4/4 PASS | 4/4 PASS | 0/4 (expected) |
| vLLM 0.9.1 | **0/4 FAIL** | **0/4 FAIL** | 0/4 |

- **HF batch=4 is internally self-consistent** (same-config rerun identical;
  1024-run == 4096-prefix) but its token streams differ from the single-candidate
  path — expected: the batch shares one RNG stream across the 4 sequences.
- **vLLM 0.9.1 is not reproducible**: a same-configuration rerun diverges at
  tokens 52 / 195 / 153 / 412 (ranks 0-3) and never returns to prefix agreement.
  vLLM must **not** be used for budget-response construction.
- Consequence for B2: a batch engine may only be used with an explicit
  engine-semantics statement *and* re-verification at the exact batch
  composition chosen; the canonical single path remains the reference.

## 5. Protocol deviation record (honest, not backdated)

The B1 manifest was written and frozen **on disk** before stage 1 launched, but
was **not committed to git before the run started**:

- manifest final pre-launch write (mtime): `2026-09-19 14:43:39 UTC`;
- stage-1 service start (`ExecMainStartTimestamp`): `2026-09-19 14:49:25 UTC`;
- first git record of the manifest: commit `f03ebba` (**created after the run
  started**; this results commit is the second record).

No experiment parameter was changed after launch; the deviation is limited to
the missing pre-launch commit, i.e. the preregistration lacked an external
timestamp. It is recorded here and in the manifest, and **not** backdated.
Corrective rule for future Track B experiments: the manifest commit **must**
precede the launch, and the launch command must reference the committed
manifest hash.

## 6. B1 gate decision

**Verdict: PASS (canonical single-candidate path).** B2 is not started by this
document; the B1 gate closes here and B2 planning proceeds separately. Evidence
relevant to B2 design:

1. Prefix-truncation labels are sound on the canonical path (PASS above);
2. If B2 wants HF-batch throughput (~2x faster per 4096 trajectory), it must
   (a) state the batch engine semantics in the B2 manifest, and (b) re-verify
   internal prefix-consistency at the exact batch composition used;
3. vLLM is excluded from B2 label construction;
4. Throughput sizing input: one 4096 trajectory ~137 s on fly122 single path,
   ~34 s/trajectory at batch=4 (stage-2 measurement; see v2_b1_budget_semantics.json).

## 7. Artifacts

Raw artifacts live in gitignored `experiments/results/`; hashes are recorded in
the manifest:

| Artifact | Bytes | sha256 (12) |
| --- | --- | --- |
| `v2_b1_rollouts.jsonl` (198 records, token ids) | 2,125,409 | `ef5219cdacda` |
| `v2_b1_budget_semantics.json` (analysis summary) | 31,024 | `b023391801ce` |
| `v2_e001_stage1_run.log` (run log) | 42,422 | `343da092a6b1` |

Commit chain on `worker/v2-allocator`: B0 guard `1623d6d`; B1 preregistration
`f03ebba`; this results commit as recorded in the Track B handoff.
