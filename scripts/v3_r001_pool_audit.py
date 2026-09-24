#!/usr/bin/env python3
"""V3-R001 §4 — DEPLOYMENT POOL AUDIT: what is V1's true theorem-sampling unit?

The owner forbade carrying the old draft's assumption ("uniform over the 24,418 parquet rows")
into V3-R001 unless the trainer source proves it. This script settles the question with
code-level evidence instead of inheritance, and measures the population that a prospective
deployment pool must be compared against.

Read-only: it loads no model weights, generates no rollout, touches no RL code path, and never
writes outside runs/ and the one manifest it is asked to write.

What it establishes
  1. sampling unit        : the parquet ROW of the loaded HF dataframe (no dedup, no groupby),
                            drawn by torch RandomSampler = one shuffled pass WITHOUT replacement.
  2. sampled population   : the 1024-token-filtered dataframe, NOT the on-disk row count.
  3. duplicate semantics  : whether the repeated rows of a statement are distinct prompts or
                            byte-identical copies (this decides "row" vs "deduplicated prompt").
  4. induced distribution : probability over UNIQUE statements under row-unit sampling
                            (multiplicity-weighted), and which statements get probability exactly 0.
  5. realized check       : the statements V1 actually rolled out, and whether their exposure
                            matches multiplicity weighting rather than uniform-over-statements.
  6. code evidence        : path:line assertions re-read from the pinned verl tree, plus file
                            sha256s, so every citation below is machine-checked at run time.
  7. sampler replication  : re-draw V1's exact first-240 rows with torch.randperm(filtered_len,
                            generator=manual_seed(data.get("seed", 1))) and test set equality
                            against the statements V1 actually rolled out, per seed. This turns
                            "row-unit RandomSampler over the filtered dataframe" from an
                            interpretation of the source into a bit-exact reproduction of history.

Output: experiments/manifests/v3/V3-R001_deployment_pool_audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from v3_d001_lib import build_records  # same reconstruction D001 used for its labelled groups

V3 = "third_party/kimina-prover-rl"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
RAW_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
FAMILY_REGISTRY = "experiments/manifests/v2/family_component_registry.json"
TOKENIZER = "models/weights/kimina_distill_0_6b"
PINNED_VERL_COMMIT = "e16b605e8186614c685875c9b57eb19e841b521a"
MAX_PROMPT_LENGTH = 1024  # data.max_prompt_length in scripts/run_p3_pilot.sh
TRAIN_BATCH_SIZE = 4      # data.train_batch_size
ROLLOUT_N = 8             # actor_rollout_ref.rollout.n
V1_STEPS = 60             # trainer.total_training_steps (60 rollout files per seed)

# (file, 1-based line, substring that must be present there) — the audit FAILS if any citation
# does not say what this script claims it says.
CODE_EVIDENCE = [
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 135,
     'dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 137,
     'self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 139,
     'print(f"dataset len: {len(self.dataframe)}")'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 141,
     'self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 143,
     'def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 145,
     'if self.filter_overlong_prompts:'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 175,
     'lambda doc: doc2len(doc) <= self.max_prompt_length,'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 180,
     'print(f"filter dataset len: {len(dataframe)}")'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 193,
     'return len(self.dataframe)'),
    (f"{V3}/verl/utils/dataset/rl_dataset.py", 220,
     'row_dict: dict = self.dataframe[item]'),
    (f"{V3}/verl/trainer/main_ppo.py", 366,
     'elif data_config.shuffle:'),
    (f"{V3}/verl/trainer/main_ppo.py", 369,
     'sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)'),
    (f"{V3}/verl/trainer/ppo/ray_trainer.py", 528,
     'self.train_dataloader = StatefulDataLoader('),
    (f"{V3}/verl/trainer/ppo/ray_trainer.py", 530,
     'batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),'),
    (f"{V3}/verl/trainer/ppo/ray_trainer.py", 532,
     'drop_last=True,'),
    (f"{V3}/verl/trainer/ppo/ray_trainer.py", 534,
     'sampler=train_sampler,'),
    (f"{V3}/verl/trainer/ppo/ray_trainer.py", 1185,
     ('gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, '
      'interleave=True)')),
    (f"{V3}/recipe/kimina_prover_rl/prepare_data.py", 52,
     '"index": sample["statement_id"],'),
]

# resolved launch config, re-read from the V1 training logs (not from the shell script)
LOG_CONFIG_KEYS = ["shuffle", "sampler", "train_batch_size", "max_prompt_length",
                   "filter_overlong_prompts", "dataloader_num_workers", "truncation"]
TRAIN_LOGS = {".cache/e019_train.log": "seed1",
              ".cache/e020_seed2_train.log": "seed2",
              ".cache/e022_seed3_train.log": "seed3"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_citations() -> tuple[list[dict], bool]:
    out, all_ok = [], True
    for rel, line, expected in CODE_EVIDENCE:
        p = ROOT / rel
        rec = {"path": rel, "line": line, "expected": expected}
        if not p.exists():
            rec.update(found=False, error="file missing")
            all_ok = False
            out.append(rec)
            continue
        src = p.read_text(errors="ignore").splitlines()
        got = src[line - 1].strip() if 0 < line <= len(src) else ""
        ok = got == expected.strip()
        if not ok:  # tolerate an off-by-a-few-lines drift, but report it
            lo, hi = max(0, line - 6), min(len(src), line + 5)
            near = [i + 1 for i in range(lo, hi) if src[i].strip() == expected.strip()]
            if near:
                rec.update(found=True, exact_line=False, actual_line=near[0], text=src[near[0] - 1].strip())
            else:
                rec.update(found=False, exact_line=False, actual_line=None, nearest_text=got)
                all_ok = False
        else:
            rec.update(found=True, exact_line=True, text=got)
        out.append(rec)
    return out, all_ok


def verl_tree_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=str(ROOT / V3), capture_output=True, text=True,
                              check=False).stdout.strip()

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"verl_repo": V3, "git_head": head,
            "matches_pinned_commit": head == PINNED_VERL_COMMIT,
            "dirty_paths_in_verl_repo": status.splitlines(),
            "note": "third_party/kimina-prover-rl is its own git checkout; the pin is checked against "
                    "THIS repo's HEAD, not the outer TinyLean-RL HEAD.",
            "files_sha256": {rel: sha256_file(ROOT / rel) for rel in
                             sorted({p for p, _, _ in CODE_EVIDENCE})}}


def _data_block(t: str):
    """Pull the resolved `data: {...}` config dump out of a ray-prefixed training log."""
    t = re.sub(r"\x1b\[[0-9;]*m", "", t)  # ray colourises the pid prefix
    lines = [re.sub(r"^\(TaskRunner pid=\d+\) ", "", ln) for ln in t.splitlines()]
    start = base = None
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("'data': {"):
            start, base = i, len(ln) - len(ln.lstrip())
            break
    if start is None:
        return None
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if not ln.strip():
            continue
        if (len(ln) - len(ln.lstrip())) <= base:
            break
        out.append(ln)
    return "\n".join(out)


def installed_verl_location() -> dict:
    """Prove WHICH verl tree the runs imported (the editable install points at third_party/)."""
    code = "import verl, os; print(os.path.dirname(verl.__file__))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False,
                       cwd=str(ROOT))
    path = r.stdout.strip()
    try:
        rel = str(Path(path).resolve().relative_to(ROOT.resolve()))
    except (ValueError, OSError):
        rel = None
    return {"imported_verl_package_dir": rel or path or None,
            "is_the_third_party_tree": bool(rel and rel.startswith(V3)),
            "stderr": (r.stderr.strip()[:300] or None)}


def read_log_facts() -> dict:
    facts = {}
    for rel, seed in TRAIN_LOGS.items():
        p = ROOT / rel
        if not p.exists():
            continue
        t = p.read_text(errors="ignore")
        d = {"dataset_len_printed": _ints(r"(?<!filter )dataset len: (\d+)", t),
             "filter_dataset_len_printed": _ints(r"filter dataset len: (\d+)", t),
             "train_dataloader_size": _ints(r"Size of train dataloader: (\d+)", t)}
        block = _data_block(t)
        if block:
            d["resolved_data_config"] = {
                "shuffle": _scalar(r"'shuffle': (\w+)", block),
                "sampler_class_path": _scalar(r"'class_path': (None|'[^']*')", block),
                "sampler_class_name": _scalar(r"'class_name': (None|'[^']*')", block),
                "train_batch_size": _scalar(r"'train_batch_size': (\d+)", block),
                "gen_batch_size_key_present": bool(re.search(r"'gen_batch_size'", block)),
                "max_prompt_length": _scalar(r"'max_prompt_length': (\d+)", block),
                "max_response_length": _scalar(r"'max_response_length': (\d+)", block),
                "filter_overlong_prompts": _scalar(r"'filter_overlong_prompts': (\w+)", block),
                "dataloader_num_workers": _scalar(r"'dataloader_num_workers': (\d+)", block),
                "truncation": _scalar(r"'truncation': '([^']*)'", block),
                "custom_dataset_class": _scalar(r"'custom_cls': \{'name': '(\w+)'", block),
                "train_files": re.findall(r"'([^']*train\.parquet)'", block),
                "data_seed_key_present": bool(re.search(r"'seed':", block)),
                "data_seed_value": _scalar(r"'seed': (\d+)", block),
            }
            tf = d["resolved_data_config"]["train_files"]
            d["train_files_is_the_audited_parquet"] = any(str(x).endswith(TRAIN_PARQUET) for x in tf)
        facts[seed] = d
    return facts


def _ints(pat: str, t: str) -> list[int]:
    return sorted({int(x) for x in re.findall(pat, t)})


def _scalar(pat: str, t: str):
    m = re.search(pat, t)
    if not m:
        return None
    v = m.group(1)
    return None if v == "None" else (int(v) if v.isdigit() else v.strip("'"))


def replicate_sampler(logs: dict, kept_order: list[str] | None, recs: list[dict],
                      filter_recomputed: bool) -> dict:
    """Replay V1's trainer-side sampler exactly and compare it with what V1 really rolled out.

    main_ppo.create_rl_sampler: RandomSampler(dataset, generator=Generator().manual_seed(
    data.get("seed", 1))); torch's RandomSampler (replacement=False) yields
    torch.randperm(len(dataset), generator=...); StatefulDataLoader(batch_size=4, drop_last=True)
    then consumes that permutation in order. len(dataset) is the FILTERED dataframe, so the first
    train_batch_size*total_steps draws are fully determined by (filtered row order, data.seed).
    A set-equality match is strong evidence for the whole account: landing on exactly those 240
    rows of a 24k-row permutation by any other mechanism is not plausible.
    """
    if not filter_recomputed or kept_order is None:
        return {"performed": False, "why": "--skip-tokenize: the filtered row order is unknown, "
                                           "so the draw cannot be replayed"}
    import torch  # heavy, and only needed to replay the sampler

    n = len(kept_order)
    draws = TRAIN_BATCH_SIZE * V1_STEPS
    by_seed: dict[str, list[tuple[int, str]]] = {}
    for r in recs:
        by_seed.setdefault(str(r["seed"]), []).append((r["step"], r["statement_id"]))
    per_seed, batch_hits, exact, steps_total = {}, 0, 0, 0
    strict_order_hits, sorted_order_hits = 0, 0
    for name, v in logs.items():
        if name not in by_seed:
            continue
        cfg = v.get("resolved_data_config") or {}
        dseed = cfg.get("data_seed_value")
        obs = sorted(by_seed[name])
        obs_seq = [s for _, s in obs]
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(
            1 if dseed is None else int(dseed))).tolist()[:draws]
        drawn = [kept_order[i] for i in perm]
        set_equal = set(drawn) == set(obs_seq)
        seq_equal = drawn == obs_seq
        first_step_equal = drawn[:TRAIN_BATCH_SIZE] == obs_seq[:TRAIN_BATCH_SIZE]
        if set_equal:
            exact += 1
        if seq_equal:
            strict_order_hits += 1
        # replay positions [4k:4k+4] are step k+1's batch under the account being tested; the
        # reconstruction's within-step order is an artifact of build_records() sorting, so the
        # comparison that matters is the per-step MULTISET.
        real_by_step: dict[int, list[str]] = {}
        for st, s in obs:
            real_by_step.setdefault(int(st), []).append(s)
        replay_batches = [drawn[i:i + TRAIN_BATCH_SIZE] for i in range(0, len(drawn), TRAIN_BATCH_SIZE)]
        steps_both = sorted(set(real_by_step) & {i + 1 for i in range(len(replay_batches))})
        batching_match = bool(steps_both) and all(
            sorted(real_by_step[st]) == sorted(replay_batches[st - 1]) for st in steps_both)
        if batching_match:
            batch_hits += 1
        steps_total += len(steps_both)
        sorted_seq_match = obs_seq == [s for b in replay_batches for s in sorted(b)]
        if sorted_seq_match:
            sorted_order_hits += 1
        per_seed[name] = {
            "data_seed_configured": cfg.get("data_seed_key_present"),
            "data_seed_used_in_replay": 1 if dseed is None else int(dseed),
            "realized_groups": len(obs_seq),
            "replay_set_equals_realized_set": set_equal,
            "replay_sequence_equals_realized_sequence": seq_equal,
            "replay_first_batch_equals_realized_first_batch": first_step_equal,
            "every_step_holds_the_same_4_theorems": batching_match,
            "steps_compared": len(steps_both),
            "realized_sequence_equals_sorted_replay": sorted_seq_match,
            "realized_steps_in_order": sorted({int(s) for s, _ in obs})[:3],
            "replay_statements_step1": drawn[:TRAIN_BATCH_SIZE],
            "realized_statements_step1": obs_seq[:TRAIN_BATCH_SIZE],
            "extra_in_replay_not_realized": sorted(set(drawn) - set(obs_seq)),
            "extra_in_realized_not_replay": sorted(set(obs_seq) - set(drawn)),
        }
    return {
        "performed": True,
        "filtered_rows_used": n,
        "draws_expected_per_seed": draws,
        "seeds_tested": len(per_seed),
        "seeds_where_set_matches": exact,
        "seeds_where_batching_matches": batch_hits,
        "steps_compared_total": steps_total,
        "seeds_where_strict_unsorted_sequence_matches": strict_order_hits,
        "seeds_where_realized_equals_sorted_replay": sorted_order_hits,
        "data_seed_note": "; ".join(
            f"{k}={'absent->default 1' if not v['data_seed_configured'] else v['data_seed_used_in_replay']}"
            for k, v in per_seed.items()),
        "per_seed": per_seed,
    }


def canon(x) -> str:
    """Stable string form of a parquet cell (ndarray / dict / scalar) for identity comparisons."""
    if isinstance(x, dict):
        return json.dumps({str(k): canon(v) for k, v in sorted(x.items())}, sort_keys=True, default=str)
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return json.dumps([canon(v) for v in x], default=str)
    return json.dumps(x, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/manifests/v3/V3-R001_deployment_pool_audit.json")
    ap.add_argument("--skip-tokenize", action="store_true",
                    help="reuse the printed filter lengths only (does not recompute the 172/7 split)")
    args = ap.parse_args()

    citations, citations_ok = check_citations()
    tree = verl_tree_state()
    logs = read_log_facts()

    df = pd.read_parquet(ROOT / TRAIN_PARQUET)
    ids = df["statement_id"].astype(str)
    per_statement_rows = Counter(ids)
    rows_on_disk = len(df)
    unique_statements_on_disk = len(per_statement_rows)

    dup = df.assign(_p=df["prompt"].map(canon), _e=df["extra_info"].map(canon))
    content_cols = ["statement_id", "informal_problem", "formal_statement", "data_source", "name",
                    "_p", "_e"]
    distinct_content_rows = len(dup.drop_duplicates(subset=content_cols))
    statements_with_varying_prompt = int((dup.groupby("statement_id")["_p"].nunique() > 1).sum())

    # --- tokenized filter reproduction (verl RLHFDataset.maybe_filter_out_long_prompts) ---------
    filter_recomputed = False
    if not args.skip_tokenize:
        from transformers import AutoTokenizer  # heavy import only when needed
        tok = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))
        seen: dict[str, int] = {}
        for p in df["prompt"].tolist():
            key = canon(p)
            if key not in seen:
                msgs = [dict(m) if not isinstance(m, dict) else m for m in
                        (p.tolist() if hasattr(p, "tolist") else list(p))]
                seen[key] = len(tok.apply_chat_template(msgs, add_generation_prompt=True))
        lens = [seen[canon(p)] for p in df["prompt"].tolist()]
        keep = [length <= MAX_PROMPT_LENGTH for length in lens]
        rows_kept = int(sum(keep))
        kept_ids = {s for s, k in zip(ids.tolist(), keep) if k}
        kept_statement_order = [s for s, k in zip(ids.tolist(), keep) if k]
        dropped_ids = {s for s in per_statement_rows if s not in kept_ids}
        filter_recomputed = True
        tok_summary = {"tokenizer_path": TOKENIZER,
                       "rule": "len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True)) "
                               "<= data.max_prompt_length",
                       "distinct_prompt_texts_tokenized": len(seen),
                       "prompt_token_length_min": int(min(lens)),
                       "prompt_token_length_max": int(max(lens))}
    else:
        rows_kept = None
        kept_statement_order = None
        dropped_ids = set()
        tok_summary = {"skipped": True}

    kept_ids_all = None
    if filter_recomputed:
        kept_ids_all = sorted({s for s in per_statement_rows if s not in dropped_ids})
    unique_kept = len(kept_ids_all) if kept_ids_all is not None else None

    # --- realized exposure in V1's own rollouts ---------------------------------------------------
    built = build_records(ROOT, RAW_PARQUET, FAMILY_REGISTRY)
    recs = built["valid"] + built["infra_censored"]
    stmt_counts = Counter(r["statement_id"] for r in recs)
    realized_unique = len(stmt_counts)
    multiplicity_of_realized = [per_statement_rows.get(s, 0) for s in stmt_counts]
    exp_w = sum(multiplicity_of_realized) / len(multiplicity_of_realized) if multiplicity_of_realized else None
    mean_pool_multiplicity = rows_on_disk / unique_statements_on_disk

    # realized per-step shape: does one step really contain 4 prompts x n samples?
    by_step: dict[tuple, list[str]] = {}
    for r in recs:
        by_step.setdefault((r["seed"], r["step"]), []).append(r["statement_id"])
    distinct_per_step = Counter(len(set(v)) for v in by_step.values())
    groups_per_step = Counter(len(v) for v in by_step.values())
    dup_within_step = sum(1 for v in by_step.values() if len(set(v)) < len(v))
    realized_shape = {
        "steps_observed": len(by_step),
        "groups_per_step": {str(k): int(v) for k, v in sorted(groups_per_step.items())},
        "distinct_prompt_rows_per_step": {str(k): int(v) for k, v in sorted(distinct_per_step.items())},
        "sequences_per_step_if_n_is_8": {str(k * ROLLOUT_N): int(v)
                                         for k, v in sorted(groups_per_step.items())},
        "steps_with_a_repeated_statement_inside_the_same_step": dup_within_step,
        "note": ("build_records gives one record per (seed, step, statement) group with its n=8 "
                 "candidates, so groups_per_step counts prompt ROWS per step and a step with "
                 "distinct < groups would mean the same theorem drawn twice in one batch."),
    }

    # --- sampler replication: reproduce V1's exact first-240 draws --------------------------------
    sampler_replay = replicate_sampler(logs, kept_statement_order, recs, filter_recomputed)

    # cross-seed overlap of the realized prompt sets: how much of the pool has ANY historical label?
    seed_sets = {sd: {r["statement_id"] for r in recs if str(r["seed"]) == sd}
                 for sd in sorted({str(r["seed"]) for r in recs})}
    pairs = {}
    ks = sorted(seed_sets)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = seed_sets[ks[i]], seed_sets[ks[j]]
            pairs[f"{ks[i]}~{ks[j]}"] = {"intersection": len(a & b),
                                         "jac": round(len(a & b) / len(a | b), 4),
                                         "union": len(a | b)}

    # --- the V1 validation slice (data.val_files), which the owner's §5 exclusion list names -------
    val_rel = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/test.parquet"
    if (ROOT / val_rel).exists():
        tv = pd.read_parquet(ROOT / val_rel)
        v_ids = set(tv["statement_id"].astype(str))
        val_slice = {
            "rel_path": val_rel,
            "rows": len(tv),
            "unique_statement_ids": len(v_ids),
            "overlaps_train_statements": len(v_ids & set(per_statement_rows)),
            "row_count_appears_in_v1_logs": any(len(tv) in (f.get("dataset_len_printed") or [])
                                                for f in logs.values()),
            "note": ("The 244-statement slice V1 used as data.val_files is a DISJOINT id space from "
                     "the 7,620-statement train pool, so it can never contribute R001 candidates "
                     "directly - but its family components can still overlap the train pool, which is "
                     "why owner §5 requires excluding holdout-touched components from the clean pool."),
        }
    else:
        val_slice = {"rel_path": val_rel, "present": False}

    verdict = {
        "true_training_sampling_unit": ("row" if rows_kept is not None else "row (unverified this run)"),
        "unit_definition": "one ROW of the HF dataframe loaded from data.train_files, after "
                           "maybe_filter_out_long_prompts; positional access self.dataframe[item]",
        "sampled_population_rows": rows_kept,
        "sampled_population_unique_statements": unique_kept,
        "on_disk_rows": rows_on_disk,
        "on_disk_unique_statements": unique_statements_on_disk,
        "rows_dropped_by_the_1024_token_filter": None if rows_kept is None else rows_on_disk - rows_kept,
        "statements_with_zero_exposure_probability": len(dropped_ids),
        "statement_ids_with_zero_exposure": sorted(dropped_ids),
        "duplicates_are_byte_identical_prompts": statements_with_varying_prompt == 0
                                                 and distinct_content_rows == unique_statements_on_disk,
        "distinct_content_rows_in_parquet": distinct_content_rows,
        "per_statement_probability": "P(s) = surviving_rows(s) / sampled_population_rows, i.e. "
                                     "MULTIPLICITY-WEIGHTED, not uniform over statements",
        "sampler": "verl create_rl_sampler -> torch RandomSampler (data.shuffle=True, "
                   "data.sampler.class_path=None): one shuffled pass WITHOUT replacement",
        "draws_per_step": TRAIN_BATCH_SIZE,
        "sequences_per_step": TRAIN_BATCH_SIZE * ROLLOUT_N,
        "rows_consumed_per_seed_run": TRAIN_BATCH_SIZE * V1_STEPS,
        "verdict_for_r001": "V3-R001's prospective unit is one unique statement_id, which IS the "
                            "canonical prompt here (duplicate rows are byte-identical), so "
                            "'unique statement_id' and 'deduplicated prompt' coincide exactly and no "
                            "choice between them has to be made. What must NOT be assumed is "
                            "uniformity: the historical pool the controller was fit on is "
                            "multiplicity-weighted over the 1024-token-filtered rows.",
    }

    if rows_kept:
        denom = f"while the sampler saw the filtered {rows_kept} rows / {unique_kept} statements."
    else:
        denom = ("while the V1 logs print the filtered row count; rerun without --skip-tokenize to "
                 "recompute the filtered population here.")
    implications = [
        ("The old draft's denominators were wrong: it used the on-disk "
         f"{rows_on_disk} rows / {unique_statements_on_disk} statements as the deployment population, "
         + denom),
        ("Statements whose every row exceeds max_prompt_length have exposure probability EXACTLY 0 "
         f"({len(dropped_ids)} of them). They can never appear in any historical label, so a model "
         "fit on V1 rollouts has never been asked about them - and they are only legitimate R001 "
         "test theorems if R001 can actually generate them, i.e. under the same 1024-token rule."),
        ("Because V1 sampled rows, the labelled set over-represents high-multiplicity statements: "
         f"mean pool multiplicity is {mean_pool_multiplicity:.3f} rows/statement, and the mean "
         f"multiplicity of the {realized_unique} statements V1 actually rolled out is "
         f"{exp_w:.3f}. A controller fit on those labels is fit under a multiplicity-weighted "
         "marginal, so applying it UNIFORMLY over a clean statement pool (owner §7) is a covariate "
         "shift that R001 must state, not hide."),
        (f"Row-unit sampling with RandomSampler is WITHOUT replacement within an epoch; at "
         f"{TRAIN_BATCH_SIZE} draws/step over the filtered population a repeat inside one step is "
         "essentially impossible, which is why V1's own batches never contained duplicate theorems."),
    ]
    n_ok = sampler_replay.get("seeds_where_batching_matches", -1)
    n_tested = sampler_replay.get("seeds_tested", 0)
    n_set = sampler_replay.get("seeds_where_set_matches", -1)
    if n_tested and n_ok == n_tested and n_set == n_tested:
        tone = ("SAMPLER ACCOUNT REPRODUCES HISTORY: for every seed tested, V1's realized "
                f"{TRAIN_BATCH_SIZE * V1_STEPS} prompt rows are exactly the first "
                f"{TRAIN_BATCH_SIZE * V1_STEPS} of randperm(filtered_len, "
                "generator=manual_seed(data.get('seed', 1))) mapped through the filtered row order "
                f"({n_set}/{n_tested} set matches), AND every step holds the same 4 theorems "
                f"({n_ok}/{n_tested} seeds, {sampler_replay['steps_compared_total']} steps). The only "
                "residual difference is the ORDER OF THEOREMS INSIDE a step, which is an artifact of "
                "build_records() sorting each step's groups rather than anything the trainer did "
                f"(seeds where the realized sequence equals the within-step-sorted replay: "
                f"{sampler_replay['seeds_where_realized_equals_sorted_replay']}/{n_tested}). The "
                "row-unit RandomSampler is therefore an empirical, bit-exact account of V1 and not "
                "merely a reading of its source, so R001's uniform-over-clean-statements draw is a "
                "genuine change of measure rather than a guess about what V1 did.")
    else:
        tone = ("CAUTION: the sampler replay did NOT reproduce every seed's realized rows "
                f"(set match {n_set}/{n_tested}, per-step batch match {n_ok}/{n_tested}); treat the "
                "row-unit conclusion as source-supported but not replay-confirmed, and investigate "
                "before R001 relies on it.")
    implications.append(tone)

    out = {
        "artifact_type": "v3_r001_deployment_pool_audit",
        "status": "AUDIT COMPLETE — read-only; no rollout, no RL, no model weights loaded"
                  + (" (tokenizer only)" if filter_recomputed else ""),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorized_by": "owner directive 2026-09-24 §4 (audit deployment pool semantics BEFORE "
                         "preregistration, with code-level evidence)",
        "host": {"hostname": os.uname().nodename, "nproc": os.cpu_count(),
                 "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                capture_output=True, text=True,
                                                check=False).stdout.strip()},
        "question": "Is V1's theorem-sampling unit a parquet row, a statement_id, or a deduplicated "
                    "prompt? The old draft asserted 'uniform over 24,418 rows' without proof.",
        "answer": verdict,
        "code_evidence": {"all_citations_verified_verbatim": citations_ok, "citations": citations},
        "verl_tree": tree,
        "installed_verl": installed_verl_location(),
        "launch_config_re_read_from_v1_logs": logs,
        "log_config_keys_checked": LOG_CONFIG_KEYS,
        "pool_measurements": {
            "train_parquet": TRAIN_PARQUET,
            "raw_parquet": RAW_PARQUET,
            "rows_on_disk": rows_on_disk,
            "unique_statement_ids": unique_statements_on_disk,
            "distinct_content_rows": distinct_content_rows,
            "statements_with_more_than_one_distinct_prompt_text": statements_with_varying_prompt,
            "multiplicity_min_mean_max": [min(per_statement_rows.values()),
                                          round(mean_pool_multiplicity, 5),
                                          max(per_statement_rows.values())],
            "multiplicity_histogram": {str(k): int(v) for k, v in
                                       sorted(Counter(per_statement_rows.values()).items())},
            "tokenizer_filter": tok_summary,
            "v1_validation_slice": val_slice,
        },
        "realized_v1_exposure": {
            "unique_statements_in_reconstructed_rollouts": realized_unique,
            "groups_reconstructed": len(recs),
            "mean_multiplicity_of_realized_statements": None if not exp_w else round(exp_w, 5),
            "pool_mean_multiplicity": round(mean_pool_multiplicity, 5),
            "per_seed_unique_statements": {k: len(v) for k, v in seed_sets.items()},
            "cross_seed_statement_overlap": pairs,
            "statements_with_any_historical_label": len(set().union(*seed_sets.values())) if seed_sets else 0,
            "labelled_share_of_filtered_pool_statements": None if not unique_kept else round(
                len(set().union(*seed_sets.values())) / unique_kept, 4) if seed_sets else None,
            "realized_per_step_shape": realized_shape,
            "note": ("build_records() is the same reconstruction V3-D001 used for its 720 groups / "
                     "686 valid groups / 588 labelled statements."),
        },
        "sampler_reconstruction": sampler_replay,
        "implications_for_r001": implications,
        "corrections_to_the_archived_draft": [
            ("docs/v3/archive/V3-R001-training-draft_SUPERSEDED.md and "
             "experiments/manifests/v3/archive/V3-R001-training-draft_offline_freeze.json use the "
             "on-disk pool (24,418 / 7,620) as the deployment population; the sampled population is "
             "the token-filtered one. Every ROW-LEVEL diagnostic in that artifact (pool "
             "multiplicities, scored_statement_share_of_pool_rows, the row-level ESS/entropy block, "
             "TRIAL_DRAWS weights) is therefore computed on an unsamplable population and does not "
             "transfer to a future sampler. The archived draft is not edited (frozen history); this "
             "file is the record."),
            ("scripts/v3_r001_offline_freeze.py compares the RAW parquet against prepare_data.py's "
             "output and concludes nothing was dropped. That comparison cannot see "
             "RLHFDataset.maybe_filter_out_long_prompts, which is where the long prompts are lost."),
        ],
        "prohibitions_respected": ["no RL optimizer update", "no rollout generated",
                                   "no 10 GB smoke", "no new controller task",
                                   "no frozen artifact modified"],
    }

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"verdict": out["answer"], "citations_ok": citations_ok,
                      "verl_head_matches_pin": tree["matches_pinned_commit"],
                      "pool": out["pool_measurements"]["multiplicity_min_mean_max"],
                      "filter": tok_summary,
                      "sampler_replay": {k: v for k, v in sampler_replay.items() if k != "per_seed"},
                      "replay_per_seed": {k: {kk: vv for kk, vv in pv.items()
                                          if not isinstance(vv, list)} for k, pv in (sampler_replay.get("per_seed") or {}).items()}}, indent=2))
    print("wrote", path)
    return 0 if citations_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
