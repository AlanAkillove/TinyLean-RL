#!/usr/bin/env python3
"""V5-R001 H1 external family-clean pool: outcome-free construction + capacity audit.

Owner directive (V5-R001 pre-launch, §7/§8): the R001 holdout must be a NEW external
family-clean set (NuminaMath-LEAN preferred), built outcome-free:

    source eligibility -> canonical statement normalization -> compile/elaboration
    eligibility (``by sorry`` statement-only validation) -> context/token-length
    eligibility -> family-component construction -> one theorem per component ->
    deterministic hash order

No theta0 (or any model) generation may take part in selection: ``theta0_outcomes_used: 0``.
The V3 sealed reserve is untouched: ``sealed_reserve_touched: 0``.

Stages (three subcommands, all deterministic and resumable):

  build   (CPU, fly90)  normalize + filter + dedupe + Promptset-contamination closure +
                        token-length filter + family components; freeze the registry and
                        emit the hash-ordered scan input (``--out-candidates``).
  scan    (fly122)      submit ``statement + "sorry"`` to the pinned Kimina oracle for every
                        candidate in hash order; record per-candidate verdicts. The scan is
                        statement-only validation: it does NOT request an info tree and is
                        NOT the R001 verification path.
  merge   (CPU, fly90)  fold the scan verdicts into the registry: compile-eligible set,
                        family components over it, one theorem per component, capacities at
                        128/192/256, distributions, and the frozen pool hash.

The canonical statement form kept for the pool is the file text through the declaration's
first ``:= by`` (a trailing ``:=`` is completed to ``:= by``); any placeholder proof body
(``sorry``) is dropped. A candidate is compile-eligible when that file compiles with a bare
``sorry`` body and no error-severity message and no REPL-level error (``sorry`` itself is a
warning and is expected).

Contamination closure: a NuminaMath-LEAN statement is "Promptset-touching" when, after
canonicalization, its strong text, or its skeleton (first declaration name abstracted), or
its name family (``name`` minus a trailing ``_v<digits>``) matches the frozen Kimina
Promptset universe (7,620 statements; the source of every consumed/reserved V1-V3 artifact).
Components containing a touching member are excluded in full.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
from pyarrow import parquet

ROOT = Path(__file__).resolve().parents[1]

NM_PARQUET = "data/raw/numinamath_lean/data/train-00000-of-00001.parquet"
PS_PARQUET = "data/raw/kimina_promptset/data/train-00000-of-00001.parquet"
TOKENIZER_DIR = "models/weights/kimina_distill_0_6b"
MAX_PROMPT_LENGTH = 1024

SYSTEM_PROMPT = "You are an expert in mathematics and proving theorems in Lean 4."
USER_TEMPLATE = (
    "Think about and solve the following problems step by step in Lean 4.\n\n"
    "# Problem:\n{problem}\n\n# Formal Statement:\n```lean4\n{formal_statement}\n```"
)

DEFAULT_REGISTRY = "experiments/manifests/v5/h1_holdout_pool.json"
DEFAULT_CANDIDATES = "experiments/results/v5_h1/h1_candidates.jsonl"
DEFAULT_SCAN = "experiments/results/v5_h1/h1_compile_scan.jsonl"
DEFAULT_SERVER = "http://127.0.0.1:8020"
# Eligibility semantics: with Lean's default `autoImplicit = true`, a free identifier in a
# statement (for example an unbound ``w`` in ``... (v + w) ...``) is silently rebound as an
# implicit universal variable, so a broken fragment "compiles" as a statement that is false in
# general. The canonical statement therefore pins ``set_option autoImplicit false`` right after
# the imports (Mathlib's own convention). The same text is what the compile scan validates and
# what any later R001 evaluation will check, so eligibility and evaluation share one semantics.
STRICT_OPTION = "set_option autoImplicit false"
SCAN_WINDOW = 12288
SCAN_TARGET_COMPILED = 4096
SCAN_BATCH = 8
SCAN_WORKERS = 6

WHITESPACE_RE = re.compile(r"\s+")
BLOCK_COMMENT_RE = re.compile(r"/-.*?-/", re.DOTALL)
DECL_KW_RE = re.compile(r"^[ \t]*(?:theorem|lemma|example)\b", re.MULTILINE)
DECL_NAME_RE = re.compile(r"\b(theorem|lemma|example)\s+\S+")
DECL_NAME_INLINE_RE = re.compile(r"^[ \t]*(?:theorem|lemma|example)\s+([^\s{(\[]+)", re.MULTILINE)
OTHER_DECL_RE = re.compile(
    r"^[ \t]*(?:(?:noncomputable|private|protected|partial|unsafe|scoped)\s+)*"
    r"(?:def|instance|abbrev|structure|class|inductive|namespace|section|end|macro|syntax|"
    r"attribute|notation|deriving)\b",
    re.MULTILINE,
)
HASH_CMD_RE = re.compile(r"^[ \t]*#", re.MULTILINE)
PRELUDE_ALLOW_RE = re.compile(
    r"^[ \t]*(?:import\b|open\b|set_option\b|universe\b|variable\b|include\b|noncomputable\b|"
    r"local\b|scoped\b|attribute\b|@\[|--)"
)
VARIANT_SUFFIX_RE = re.compile(r"_v\d+$")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# canonicalization layers (pure functions; mirror scripts/v2_family_leakage_audit.py)
# --------------------------------------------------------------------------


def normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).splitlines()).strip()


def strip_block_comments(text: str) -> str:
    return BLOCK_COMMENT_RE.sub(" ", text)


def strong_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def skeleton_text(text: str) -> str:
    return DECL_NAME_RE.sub(r"\1 _", strong_text(text), count=1)


def family_key(name: str | None) -> str | None:
    if not name:
        return None
    return VARIANT_SUFFIX_RE.sub("", name) or name


def declaration_name(text: str) -> str | None:
    match = DECL_NAME_INLINE_RE.search(strip_block_comments(text))
    return match.group(1) if match else None


def with_strict_option(head: str) -> str:
    """Insert ``set_option autoImplicit false`` after the last import line of the prelude."""

    lines = head.split("\n") if head else []
    last_import = None
    for index, line in enumerate(lines):
        if line.lstrip().startswith("import "):
            last_import = index
    if last_import is None:
        return f"{STRICT_OPTION}\n{head}" if head else STRICT_OPTION
    return "\n".join(lines[: last_import + 1] + [STRICT_OPTION] + lines[last_import + 1 :])


def canonicalize(raw: str) -> tuple[str | None, str | None, str | None]:
    """Return ``(canonical_statement, reject_reason, name)``.

    The canonical form is the whole file text through the declaration's first ``:= by``
    (a trailing ``:=`` is completed to ``:= by``); any proof body is dropped. Rejections
    are structural only - they never depend on any model output.
    """

    text = normalize(raw)
    if not text:
        return None, "empty_statement", None
    if "import Mathlib" not in text:
        return None, "no_import", None
    decls = DECL_KW_RE.findall(text)
    if len(decls) != 1:
        return None, "num_decl_not_1", None
    stripped = strip_block_comments(text)
    if OTHER_DECL_RE.search(stripped):
        return None, "other_decl", None
    if HASH_CMD_RE.search(stripped):
        return None, "hash_cmd", None
    if "autoImplicit" in stripped:
        return None, "explicit_autoimplicit", None
    match = DECL_KW_RE.search(stripped)
    for line in stripped[: match.start()].split("\n"):
        if line.strip() and not PRELUDE_ALLOW_RE.match(line):
            return None, "prelude_junk", None
    head = stripped[: match.start()].rstrip()
    tail = stripped[match.start() :]
    by_match = re.search(r":=\s*by\b", tail)
    if by_match:
        statement = tail[: by_match.end()]
    elif tail.rstrip().endswith(":="):
        statement = tail.rstrip() + " by"
    else:
        return None, "bad_ending", None
    statement = statement.strip()
    if "sorry" in statement:
        return None, "sorry_inside", None
    head = with_strict_option(head)
    canonical = f"{head}\n{statement}" if head else statement
    return canonical, None, declaration_name(canonical)


def prompt_messages(problem: str, statement: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(problem=problem, formal_statement=statement)},
    ]


def check_text(statement: str) -> str:
    return f"{statement}\n  sorry"


# --------------------------------------------------------------------------
# build stage
# --------------------------------------------------------------------------


def load_promptset_reference() -> dict:
    table = parquet.read_table(ROOT / PS_PARQUET, columns=["natural_language", "formal_statement", "source", "name"])
    statements = [normalize(t) for t in table.column("formal_statement").to_pylist()]
    strong = set()
    skeleton = set()
    for text in statements:
        canonical, _, _ = canonicalize(text)
        body = canonical or text
        strong.add(strong_text(body))
        skeleton.add(skeleton_text(body))
    names = {str(n) for n in table.column("name").to_pylist() if n}
    families = {family_key(n) for n in names}
    return {
        "statements": len(set(statements)),
        "strong": strong,
        "skeleton": skeleton,
        "families": families,
        "source_counts": dict(sorted(collections.Counter(table.column("source").to_pylist()).items())),
    }


def build_components(rows: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Union-find over [L3 name family, L2 skeleton, L4 natural language] edges."""

    parent: dict[str, str] = {row["statement_id"]: row["statement_id"] for row in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: list[dict] = [
        collections.defaultdict(list),
        collections.defaultdict(list),
        collections.defaultdict(list),
    ]
    for row in rows:
        sid = row["statement_id"]
        family = family_key(row["name"])
        if family is not None:
            groups[0][family].append(sid)
        groups[1][skeleton_text(row["canonical"])].append(sid)
        if row["problem"].strip():
            groups[2][strong_text(row["problem"])].append(sid)
    for layer in groups:
        for ids in layer.values():
            ids.sort()
            for other in ids[1:]:
                union(ids[0], other)
    component_of = {sid: find(sid) for sid in parent}
    members: dict[str, list[str]] = collections.defaultdict(list)
    for sid, comp in component_of.items():
        members[comp].append(sid)
    for ids in members.values():
        ids.sort()
    return component_of, dict(members)


def build(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    nm_path = ROOT / NM_PARQUET
    frame = parquet.read_table(nm_path).to_pandas()
    problems = ["" if pd.isna(p) else str(p) for p in frame["problem"]]
    formals = [normalize(t) for t in frame["formal_statement"].fillna("")]
    uuids = [str(u) for u in frame["uuid"]]
    sources = ["" if pd.isna(s) else str(s) for s in frame["source"]]
    problem_types = ["" if pd.isna(s) else str(s) for s in frame["problem_type"]]
    gt_types = ["" if pd.isna(s) else str(s) for s in frame["ground_truth_type"]]

    rejections: collections.Counter[str] = collections.Counter()
    index_rows: list[dict] = []
    for i, raw in enumerate(formals):
        text, reason, name = canonicalize(raw)
        if reason:
            rejections[f"structural_{reason}"] += 1
            continue
        index_rows.append(
            {
                "row": i,
                "statement_id": sha256_hex(text),
                "canonical": text,
                "name": name,
                "uuid": uuids[i],
                "source": sources[i],
                "problem_type": problem_types[i],
                "ground_truth_type": gt_types[i],
                "problem": problems[i],
            }
        )

    ps = load_promptset_reference()
    deduped: list[dict] = []
    seen_ids: set[str] = set()
    dup_rows = 0
    for row in index_rows:
        if row["statement_id"] in seen_ids:
            dup_rows += 1
            continue
        seen_ids.add(row["statement_id"])
        row["ps_touch_text"] = strong_text(row["canonical"]) in ps["strong"]
        row["ps_touch_skeleton"] = skeleton_text(row["canonical"]) in ps["skeleton"]
        row["ps_touch_family"] = family_key(row["name"]) in ps["families"] if row["name"] else False
        deduped.append(row)

    touched = [r for r in deduped if r["ps_touch_text"] or r["ps_touch_skeleton"] or r["ps_touch_family"]]
    kept = [r for r in deduped if not (r["ps_touch_text"] or r["ps_touch_skeleton"] or r["ps_touch_family"])]

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER_DIR))
    for row in kept:
        plain = prompt_messages(row["problem"], row["canonical"])
        with_placeholder = prompt_messages(
            row["problem"], re.sub(r":=\s*by$", ":= by sorry", row["canonical"])
        )
        row["prompt_len"] = len(tokenizer.apply_chat_template(plain, add_generation_prompt=True))
        row["prompt_len_placeholder"] = len(
            tokenizer.apply_chat_template(with_placeholder, add_generation_prompt=True)
        )
    length_stats = sorted(r["prompt_len_placeholder"] for r in kept)
    long_rows = [r for r in kept if r["prompt_len_placeholder"] > MAX_PROMPT_LENGTH]
    kept = [r for r in kept if r["prompt_len_placeholder"] <= MAX_PROMPT_LENGTH]

    component_of, members = build_components(kept)
    ordered = sorted(kept, key=lambda r: r["statement_id"])

    candidates_path = ROOT / args.out_candidates
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with candidates_path.open("w", encoding="utf-8") as handle:
        for rank, row in enumerate(ordered, start=1):
            handle.write(
                json.dumps(
                    {
                        "rank": rank,
                        "statement_id": row["statement_id"],
                        "component_id": component_of[row["statement_id"]],
                        "check_text": check_text(row["canonical"]),
                        "source": row["source"],
                        "problem_type": row["problem_type"],
                        "ground_truth_type": row["ground_truth_type"],
                        "prompt_len_placeholder": row["prompt_len_placeholder"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def quantile(values: list[int], q: float):
        if not values:
            return None
        return values[min(len(values) - 1, int(q * (len(values) - 1)))]

    registry = {
        "stage": "H1_EXTERNAL_POOL_BUILT",
        "generated_utc": utc_now(),
        "git_revision": git_revision(),
        "owner_directive": {
            "holdout_decision": "H1 new external family-clean holdout (NuminaMath-LEAN)",
            "construction": "outcome-free: source eligibility -> normalization -> compile eligibility "
            "(`by sorry`) -> token eligibility -> family components -> one theorem per component -> "
            "deterministic hash order",
            "theta0_outcomes_used": 0,
            "sealed_reserve_touched": 0,
            "v3_r001_128_reused": False,
        },
        "source": {
            "corpus": "AI-MO/NuminaMath-LEAN",
            "parquet": NM_PARQUET,
            "sha256": sha256_file(nm_path),
            "rows": len(frame),
        },
        "promptset_reference": {
            "parquet": PS_PARQUET,
            "sha256": sha256_file(ROOT / PS_PARQUET),
            "unique_statements": ps["statements"],
            "families": len(ps["families"]),
            "source_counts": ps["source_counts"],
        },
        "counts": {
            "rows": len(frame),
            "structural_rejections": dict(sorted(rejections.items())),
            "duplicate_statement_rows": dup_rows,
            "ps_touch_excluded_rows": len(touched),
            "token_too_long_rows": len(long_rows),
            "eligible_candidates": len(ordered),
            "components": len(members),
        },
        "ps_touch": {
            "excluded_rows": len(touched),
            "by_text": sum(1 for r in touched if r["ps_touch_text"]),
            "by_skeleton": sum(1 for r in touched if r["ps_touch_skeleton"]),
            "by_name_family": sum(1 for r in touched if r["ps_touch_family"]),
            "rule": "canonical strong text OR skeleton OR name family matches the frozen Promptset universe",
        },
        "token_length": {
            "tokenizer": TOKENIZER_DIR,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "convention": "chat template (system+user) with `add_generation_prompt=True`",
            "with_placeholder": {
                "evaluated": len(length_stats),
                "le_1024": len(length_stats) - len(long_rows),
                "mean": round(sum(length_stats) / max(1, len(length_stats)), 1),
                "median": quantile(length_stats, 0.5),
                "p95": quantile(length_stats, 0.95),
                "p99": quantile(length_stats, 0.99),
                "max": length_stats[-1] if length_stats else None,
                "over_1024": len(long_rows),
            },
            "no_placeholder_le_1024": sum(1 for r in kept if r["prompt_len"] <= MAX_PROMPT_LENGTH),
        },
        "components_pre_scan": {
            "count": len(members),
            "size_distribution": dict(sorted(collections.Counter(len(v) for v in members.values()).items())),
        },
        "distributions_pre_scan": {
            "source": dict(sorted(collections.Counter(r["source"] for r in ordered).items())),
            "problem_type": dict(sorted(collections.Counter(r["problem_type"] for r in ordered).items())),
            "ground_truth_type": dict(sorted(collections.Counter(r["ground_truth_type"] for r in ordered).items())),
        },
        "candidates_file": {
            "path": args.out_candidates,
            "sha256": sha256_file(candidates_path),
            "ordered_by": "statement_id (sha256 hex of the canonical statement text) ascending",
        },
        "scan": None,
        "verified": None,
        "pool_hash": None,
    }
    out = ROOT / args.out_registry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(registry, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(
        f"[build] rows={len(frame)} eligible={len(ordered)} components={len(members)} "
        f"ps_touch={len(touched)} token_long={len(long_rows)} dup={dup_rows}"
    )
    print(f"[build] registry -> {out}")
    print(f"[build] candidates -> {candidates_path} ({len(ordered)} rows)")
    return 0


# --------------------------------------------------------------------------
# scan stage (fly122; statement-only validation, no info tree)
# --------------------------------------------------------------------------


def scan_one_batch(client: httpx.Client, url: str, batch: list[dict], server_timeout: int) -> list[dict]:
    payload = {
        "codes": [{"custom_id": str(item["rank"]), "proof": item["check_text"]} for item in batch],
        "timeout": server_timeout,
    }
    response = client.post(url, json=payload)
    response.raise_for_status()
    decoded = response.json()
    results = decoded["results"] if isinstance(decoded, dict) and "results" in decoded else decoded
    by_id = {}
    for item in results if isinstance(results, list) else []:
        if isinstance(item, dict):
            by_id[str(item.get("custom_id"))] = item
    return [by_id.get(str(item["rank"]), {"custom_id": str(item["rank"]), "error": "missing result"}) for item in batch]


def classify_item(item: dict) -> str:
    if item.get("error"):
        return "infra_error"
    response = item.get("response")
    if not isinstance(response, dict):
        return "infra_error"
    if "message" in response:
        return "not_compiled"
    for message in response.get("messages") or []:
        if isinstance(message, dict) and message.get("severity") == "error":
            return "not_compiled"
    return "compiled"


def scan(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    candidates: list[dict] = []
    with (ROOT / args.candidates).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["rank"] <= args.window:
                candidates.append(record)
    out_path = ROOT / args.out_scan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = args.server.rstrip("/") + "/verify"

    compiled = 0
    scanned = 0
    started = utc_now()
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as sink, httpx.Client(timeout=args.http_timeout, trust_env=False) as client:
        pending = [candidates[i : i + args.batch] for i in range(0, len(candidates), args.batch)]

        def run(batch: list[dict]) -> list[dict]:
            last_error = None
            for attempt in range(1, args.attempts + 1):
                try:
                    results = scan_one_batch(client, url, batch, args.server_timeout)
                    verdicts = []
                    retry = False
                    for item, candidate in zip(results, batch):
                        verdict = classify_item(item)
                        if verdict == "infra_error" and attempt < args.attempts:
                            last_error = item.get("error")
                            retry = True
                            break
                        verdicts.append(
                            {
                                "rank": candidate["rank"],
                                "statement_id": candidate["statement_id"],
                                "verdict": verdict,
                                "attempts": attempt,
                                "error": None if verdict != "infra_error" else str(item.get("error"))[:200],
                            }
                        )
                    if not retry:
                        return verdicts
                except (httpx.HTTPError, OSError) as exc:  # transport-level failure: retry the batch
                    last_error = repr(exc)[:200]
                    time.sleep(min(5.0, 1.0 * attempt))
            return [
                {
                    "rank": candidate["rank"],
                    "statement_id": candidate["statement_id"],
                    "verdict": "infra_unresolved",
                    "attempts": args.attempts,
                    "error": str(last_error)[:200],
                }
                for candidate in batch
            ]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for verdicts in pool.map(run, pending):
                for verdict in verdicts:
                    sink.write(json.dumps(verdict) + "\n")
                    scanned += 1
                    if verdict["verdict"] == "compiled":
                        compiled += 1
                elapsed = time.time() - t0
                print(
                    f"[scan] scanned={scanned}/{len(candidates)} compiled={compiled} "
                    f"rate={compiled / max(1, scanned):.3f} elapsed={elapsed / 60:.1f}min "
                    f"({scanned / max(1e-9, elapsed):.2f}/s)",
                    flush=True,
                )
                if compiled >= args.target_compiled:
                    print(f"[scan] target compiled={args.target_compiled} reached; stopping early")
                    break

    meta = {
        "stage": "H1_COMPILE_SCAN_DONE",
        "started_utc": started,
        "finished_utc": utc_now(),
        "window": args.window,
        "scanned": scanned,
        "compiled": compiled,
        "server": args.server,
        "batch": args.batch,
        "workers": args.workers,
        "attempts": args.attempts,
        "server_timeout": args.server_timeout,
        "scan_file": args.out_scan,
        "scan_sha256": sha256_file(out_path),
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[scan] wrote {out_path} ({scanned} verdicts, {compiled} compiled)")
    return 0


# --------------------------------------------------------------------------
# merge stage (fly90)
# --------------------------------------------------------------------------


def merge(args: argparse.Namespace) -> int:
    registry = json.loads((ROOT / args.registry).read_text(encoding="utf-8"))
    candidates: dict[int, dict] = {}
    with (ROOT / args.candidates).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            candidates[record["rank"]] = record
    verdicts: list[dict] = []
    with (ROOT / args.scan).open(encoding="utf-8") as handle:
        for line in handle:
            verdicts.append(json.loads(line))

    compiled_ids = sorted(v["statement_id"] for v in verdicts if v["verdict"] == "compiled")
    id_to_component = {record["statement_id"]: record["component_id"] for record in candidates.values()}
    id_to_row = {record["statement_id"]: record for record in candidates.values()}
    verdict_counts = dict(sorted(collections.Counter(v["verdict"] for v in verdicts).items()))

    parent: dict[str, str] = {sid: sid for sid in compiled_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_component: dict[str, list[str]] = collections.defaultdict(list)
    for sid in compiled_ids:
        by_component[id_to_component[sid]].append(sid)
    for ids in by_component.values():
        for other in ids[1:]:
            union(ids[0], other)
    components: dict[str, list[str]] = collections.defaultdict(list)
    for sid in compiled_ids:
        components[find(sid)].append(sid)
    representatives = sorted(min(ids) for ids in components.values())
    capacity = len(representatives)

    def bucket(n: int) -> dict:
        return {"required": n, "available": capacity, "supported": capacity >= n}

    verified_set = set(compiled_ids)
    registry["scan"] = {
        "scan_file": args.scan,
        "verdict_counts": verdict_counts,
        "scanned": len(verdicts),
        "window": max((v["rank"] for v in verdicts), default=0),
        "throughput_note": "statement-only validation; no info tree; not the R001 verification path",
    }
    registry["verified"] = {
        "compiled_candidates": len(compiled_ids),
        "family_components": capacity,
        "capacity": {"128": bucket(128), "192": bucket(192), "256": bucket(256)},
        "one_theorem_per_component": "representative = lexicographically smallest statement_id in the component",
        "representatives": representatives,
        "distributions": {
            "source": dict(sorted(collections.Counter(id_to_row[s]["source"] for s in compiled_ids).items())),
            "problem_type": dict(
                sorted(collections.Counter(id_to_row[s]["problem_type"] for s in compiled_ids).items())
            ),
            "ground_truth_type": dict(
                sorted(collections.Counter(id_to_row[s]["ground_truth_type"] for s in compiled_ids).items())
            ),
            "prompt_len_placeholder": {
                "mean": round(
                    sum(id_to_row[s]["prompt_len_placeholder"] for s in compiled_ids) / max(1, len(compiled_ids)), 1
                ),
                "max": max((id_to_row[s]["prompt_len_placeholder"] for s in compiled_ids), default=None),
            },
        },
        "component_size_distribution": dict(
            sorted(collections.Counter(len(ids) for ids in components.values()).items())
        ),
    }
    registry["stage"] = "H1_EXTERNAL_POOL_VERIFIED"
    registry["completed_utc"] = utc_now()
    registry["pool_hash"] = sha256_hex(
        json.dumps(
            {
                "representatives": representatives,
                "max_prompt_length": MAX_PROMPT_LENGTH,
                "rule": "one theorem per verified family component, hash order",
            },
            sort_keys=True,
        )
    )
    out = ROOT / args.out_registry
    out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(f"[merge] verdicts={verdict_counts} compiled={len(verified_set)} components={capacity}")
    print(f"[merge] registry -> {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build")
    builder.add_argument("--out-registry", default=DEFAULT_REGISTRY)
    builder.add_argument("--out-candidates", default=DEFAULT_CANDIDATES)
    builder.set_defaults(func=build)

    scanner = sub.add_parser("scan")
    scanner.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    scanner.add_argument("--out-scan", default=DEFAULT_SCAN)
    scanner.add_argument("--server", default=DEFAULT_SERVER)
    scanner.add_argument("--window", type=int, default=SCAN_WINDOW)
    scanner.add_argument("--target-compiled", type=int, default=SCAN_TARGET_COMPILED)
    scanner.add_argument("--batch", type=int, default=SCAN_BATCH)
    scanner.add_argument("--workers", type=int, default=SCAN_WORKERS)
    scanner.add_argument("--attempts", type=int, default=2)
    scanner.add_argument("--server-timeout", type=int, default=120)
    scanner.add_argument("--http-timeout", type=float, default=300.0)
    scanner.set_defaults(func=scan)

    merger = sub.add_parser("merge")
    merger.add_argument("--registry", default=DEFAULT_REGISTRY)
    merger.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    merger.add_argument("--scan", default=DEFAULT_SCAN)
    merger.add_argument("--out-registry", default=DEFAULT_REGISTRY)
    merger.set_defaults(func=merge)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
