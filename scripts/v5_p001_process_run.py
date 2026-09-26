#!/usr/bin/env python3
"""V5-P001 Phase-B runner: process the frozen primary historical surface once.

Phase B (owner §26, §28-§30) submits every candidate of the primary historical
surface -- all candidates of the 686 non-contaminated V1 groups -- exactly once
to the dedicated process oracle, then freezes and structurally re-validates the
result.  Stages:

  preflight  provenance re-verification of every planned candidate against the
             frozen surface, the V1 rollout files and the dataset; no oracle
             call, no label; writes ``preflight.json`` as evidence
  process    resumable, append-only, exactly-once processing; stdout is
             infrastructure-only by construction
  freeze     structural/infrastructure freeze: label hash, coverage, raw-item
             manifest, infrastructure accounting
  validate   independent re-derivation of every stored record from the rollout
             sources and the archived raw items (no oracle calls)
  status     infrastructure-only progress summary

Scientific metrics (RecoveryRate and friends) are computed exactly once, later,
by ``scripts/v5_p001_analyze.py`` -- never by this runner (owner §28, §30).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
_UPSTREAM = ROOT / "third_party" / "kimina-prover-rl" / "recipe" / "kimina_prover_rl"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

import pyarrow.parquet as pq
import v5_p001_reconstruct as R
import v5_p001_spec as S
import v5_process_oracle as O
from kimina_prover_rl.reward.proof_utils import extract_proof_from_text

from tinylean_rl.verifier.policy import (
    Classified,
    VerifierUnhealthyError,
    VerifyOutcome,
)

#: Scripts whose bytes are part of the run stamp; any edit invalidates a resume.
STAMP_SCRIPTS = (
    "v5_p001_spec.py",
    "v5_process_oracle.py",
    "v5_p001_reconstruct.py",
    "v5_p001_process_run.py",
    "v5_p001_analyze.py",
)

PROGRESS_EVERY = 25
MAX_VIOLATIONS_SHOWN = 20


# --------------------------------------------------------------------------------------
# provenance: stamp, plan, source access
# --------------------------------------------------------------------------------------


def git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def code_stamp() -> dict[str, Any]:
    return {
        "git_head": git_head(),
        "scripts": {name: S.sha256_file(ROOT / "scripts" / name) for name in STAMP_SCRIPTS},
        "historical_surface_sha256": S.sha256_file(S.HISTORICAL_SURFACE),
        "oracle_validation_sha256": S.sha256_file(S.ORACLE_VALIDATION),
    }


def load_surface() -> dict[str, Any]:
    surface = json.loads(S.HISTORICAL_SURFACE.read_text(encoding="utf-8"))
    verification = surface.get("verification", {})
    if verification.get("passed") is not True:
        raise RuntimeError(f"{S.HISTORICAL_SURFACE} has not passed binary reconstruction")
    if not all(verification.get("checks", {}).values()):
        raise RuntimeError(f"{S.HISTORICAL_SURFACE} carries failed reconstruction checks")
    if surface["inputs"]["dataset_sha256"] != S.sha256_file(S.DATASET):
        raise RuntimeError("dataset parquet bytes changed since the frozen surface was built")
    return surface


def load_formal_by_statement_id() -> dict[str, str]:
    table = pq.read_table(S.DATASET, columns=["statement_id", "formal_statement"])
    out: dict[str, str] = {}
    for statement_id, formal in zip(
        table.column("statement_id").to_pylist(), table.column("formal_statement").to_pylist()
    ):
        out.setdefault(statement_id, formal)
    return out


def build_plan(surface: dict[str, Any]) -> list[dict[str, Any]]:
    """The frozen processing order: surface order, primary groups only, slot order."""

    planned: list[dict[str, Any]] = []
    for group in surface["groups"]:
        if group["contaminated"]:
            continue
        for cand in group["candidates"]:
            planned.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "group_key": group["group_key"],
                    "seed": group["seed"],
                    "group_label": group["label"],
                    "component_id": group["component_id"],
                    "statement_id": group["statement_id"],
                    "slot": cand["slot"],
                    "file": cand["file"],
                    "line": cand["line"],
                    "score": cand["score"],
                    "acc": cand["acc"],
                    "infra": bool(cand["infra"]),
                    "pred_kind": cand["pred_kind"],
                    "pred_sha256": cand["pred_sha256"],
                    "response_sha256": cand["response_sha256"],
                    "extractable": cand["pred_kind"] == "code",
                }
            )
    if len(planned) != sum(len(g["candidates"]) for g in surface["groups"] if not g["contaminated"]):
        raise RuntimeError("plan construction lost candidates")
    return planned


class SourceReader:
    """Line-accurate access to the frozen V1 rollout files (one file cached)."""

    def __init__(self) -> None:
        self._key: tuple[str, str] | None = None
        self._rows: dict[int, dict[str, Any]] = {}

    def row(self, seed: str, file: str, line: int) -> dict[str, Any]:
        key = (seed, file)
        if key != self._key:
            records, lines, unparsable = R.read_records(S.SEED_DIRS[seed] / file)
            if unparsable:
                raise RuntimeError(f"{seed}/{file}: {unparsable} unparsable source lines")
            self._rows = dict(zip(lines, records))
            self._key = key
        if line not in self._rows:
            raise RuntimeError(f"{seed}/{file}: source line {line} not found")
        return self._rows[line]


def candidate_inputs(
    entry: dict[str, Any],
    reader: SourceReader,
    formal_by_statement_id: dict[str, str],
) -> dict[str, Any]:
    """Re-read one planned candidate and re-verify every frozen provenance fact."""

    row = reader.row(entry["seed"], entry["file"], entry["line"])
    response = row.get("response")
    pred = row.get("pred")
    pred = pred if isinstance(pred, str) else ""
    problems: list[str] = []
    if not isinstance(response, str):
        problems.append("response is not a string")
        response = ""
    if S.sha256_text(pred) != entry["pred_sha256"]:
        problems.append("pred sha256 mismatch")
    if S.sha256_text(response) != entry["response_sha256"]:
        problems.append("response sha256 mismatch")
    if R.sentinel_kind(pred) != entry["pred_kind"]:
        problems.append(f"pred kind {R.sentinel_kind(pred)} != {entry['pred_kind']}")
    if R.is_system_error(row.get("tool_feedback", "")) != entry["infra"]:
        problems.append("infra flag mismatch")
    formal = formal_by_statement_id.get(entry["statement_id"]) if entry["statement_id"] else None
    locate = O.locate_generated_tail(response, formal or "")
    if entry["extractable"]:
        if formal is None:
            problems.append("no dataset formal statement for statement_id")
        elif not locate["found"]:
            problems.append("generated tail not located in the response")
        elif locate["pred_recomputed"] != pred:
            problems.append("recomputed pred differs from the historical pred")
        elif pred in S.PRED_SENTINELS:
            problems.append("extractable candidate carries a sentinel pred")
    if problems:
        raise RuntimeError(f"{entry['candidate_id']}: " + "; ".join(problems))
    return {"pred": pred, "response": response, "formal": formal, "locate": locate}


# --------------------------------------------------------------------------------------
# run directory: paths, meta, logs
# --------------------------------------------------------------------------------------


@dataclass
class RunPaths:
    root: Path

    labels: Path = field(init=False)
    raw: Path = field(init=False)
    raw_manifest: Path = field(init=False)
    freeze: Path = field(init=False)
    validation: Path = field(init=False)
    preflight: Path = field(init=False)
    run_meta: Path = field(init=False)
    infra_log: Path = field(init=False)

    def __post_init__(self) -> None:
        self.labels = self.root / S.PROCESS_LABELS.name
        self.raw = self.root / S.PROCESS_RAW_DIR.name
        self.raw_manifest = self.root / S.PROCESS_RAW_MANIFEST.name
        self.freeze = self.root / S.PROCESS_FREEZE.name
        self.validation = self.root / S.PROCESS_VALIDATION.name
        self.preflight = self.root / "v5_p001_process_preflight.json"
        self.run_meta = self.root / "v5_p001_process_run_meta.json"
        self.infra_log = self.root / "v5_p001_process_infra_events.jsonl"
        if self.root == S.PROCESS_RUN_DIR:
            assert self.labels == S.PROCESS_LABELS and self.freeze == S.PROCESS_FREEZE
            assert self.validation == S.PROCESS_VALIDATION


def display_path(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise (scratch run dirs)."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def raw_name(candidate_id: str) -> str:
    return candidate_id.replace(":", "_") + ".json"


def stable_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Record projection that must be reproducible without touching the oracle."""

    projection = {key: value for key, value in record.items() if key != "oracle"}
    oracle = record.get("oracle")
    if isinstance(oracle, dict):
        projection["oracle"] = {k: v for k, v in oracle.items() if k != "seconds"}
    return projection


def record_hash(record: dict[str, Any]) -> str:
    return S.sha256_text(S.canonical_json(stable_projection(record)))


class InfraLog:
    """Append-only infrastructure event log (no scientific content, by policy)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.n_events = 0

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = S.canonical_json(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kind": kind,
                **payload,
            }
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.n_events += 1


class LabelsLog:
    """Append-only label store with partial-tail recovery and exactly-once ids."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.existing: dict[str, str] = {}
        self.n_truncated_suffixes = 0

    def load(self, planned_ids: set[str]) -> dict[str, str]:
        if not self.path.exists():
            return {}
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            # A crash during an append leaves a partial final line; drop only that.
            cut = raw.rfind(b"\n") + 1
            self.n_truncated_suffixes += 1
            with open(self.path, "r+b") as handle:
                handle.truncate(cut)
            raw = raw[:cut]
        for chunk in raw.split(b"\n"):
            if not chunk.strip():
                continue
            record = json.loads(chunk.decode("utf-8"))
            candidate_id = record["candidate_id"]
            if candidate_id not in planned_ids:
                raise RuntimeError(f"label file carries unplanned candidate {candidate_id}")
            if candidate_id in self.existing:
                raise RuntimeError(f"label file carries duplicate candidate {candidate_id}")
            self.existing[candidate_id] = record_hash(record)
        return self.existing

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(S.canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class RawArchive:
    """One archived oracle item per code-extractable candidate."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def write(self, candidate_id: str, pred_sha256: str, result: dict[str, Any]) -> Path:
        classified = result["classified"]
        item = result.get("item")
        payload = {
            "candidate_id": candidate_id,
            "pred_sha256": pred_sha256,
            "classified": {
                "outcome": classified.outcome.value,
                "message": classified.message,
                "redeclaration": bool(classified.redeclaration),
            },
            "attempts": result.get("attempts") or [],
            "seconds": result.get("seconds"),
            "item": item,
        }
        path = self.directory / raw_name(candidate_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(S.canonical_json(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def read(self, candidate_id: str) -> dict[str, Any]:
        return json.loads((self.directory / raw_name(candidate_id)).read_text(encoding="utf-8"))

    def files(self) -> list[Path]:
        return sorted(self.directory.glob("*.json"))


# --------------------------------------------------------------------------------------
# oracle submission with bounded, fail-close container recovery
# --------------------------------------------------------------------------------------


def container_restart() -> None:
    proc = subprocess.run(
        ["docker", "restart", "-t", "10", S.ORACLE_INFRA["container"]],
        capture_output=True,
        text=True,
        check=False,
        timeout=float(S.ORACLE_INFRA["restart_timeout_s"]),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker restart {S.ORACLE_INFRA['container']} failed: {proc.stderr.strip()[:300]}"
        )


def wait_healthy(client: O.OracleClient) -> None:
    deadline = time.monotonic() + float(S.ORACLE_INFRA["health_poll_timeout_s"])
    while time.monotonic() < deadline:
        health = client.health()
        if health["ok"]:
            return
        time.sleep(float(S.ORACLE_INFRA["health_poll_interval_s"]))
    raise RuntimeError("oracle /health did not recover within the frozen poll budget")


@dataclass
class Counters:
    submitted: int = 0
    sentinel: int = 0
    infra_censored: int = 0
    infra_events: int = 0
    recoveries: int = 0
    outcome_histogram: dict[str, int] = field(default_factory=dict)
    attempts_total: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def bump_outcome(self, outcome: str) -> None:
        self.outcome_histogram[outcome] = self.outcome_histogram.get(outcome, 0) + 1


class ProcessStage:
    def __init__(
        self,
        *,
        paths: RunPaths,
        client: O.OracleClient | None,
        mapper: O.TokenMapper,
        infra_log: InfraLog,
        manage_container: bool,
    ) -> None:
        self.paths = paths
        self.client = client
        self.mapper = mapper
        self.infra_log = infra_log
        self.manage_container = manage_container
        self.counters = Counters()

    # -- oracle ---------------------------------------------------------------------------

    def submit(self, candidate_id: str, pred: str) -> dict[str, Any]:
        custom_id = f"v5p001-{candidate_id}"
        result = self.client.verify_raw(pred, custom_id=custom_id)
        self.counters.attempts_total += len(result.get("attempts") or [])
        if not result["classified"].outcome.is_infrastructure:
            return result
        return self.handle_infrastructure(candidate_id, pred, custom_id, result)

    def handle_infrastructure(
        self, candidate_id: str, pred: str, custom_id: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        self.counters.infra_events += 1
        self.infra_log.event(
            "infra_event",
            {
                "candidate_id": candidate_id,
                "outcome": result["classified"].outcome.value,
                "message": result["classified"].message[:300],
                "attempts": len(result.get("attempts") or []),
            },
        )
        try:
            self.client.require_healthy()
            return result  # candidate-specific infrastructure: record as censored
        except VerifierUnhealthyError as exc:
            if not self.manage_container:
                raise RuntimeError(
                    "oracle canary failed after an infrastructure event; "
                    "container recovery is disabled"
                ) from exc
        self.recover("canary failed after infrastructure event")
        return self.repeat_after_recovery(candidate_id, pred, custom_id)

    def repeat_after_recovery(
        self, candidate_id: str, pred: str, custom_id: str
    ) -> dict[str, Any]:
        result = self.client.verify_raw(pred, custom_id=custom_id)
        self.counters.attempts_total += len(result.get("attempts") or [])
        if not result["classified"].outcome.is_infrastructure:
            return result
        self.counters.infra_events += 1
        self.infra_log.event(
            "infra_event_after_recovery",
            {
                "candidate_id": candidate_id,
                "outcome": result["classified"].outcome.value,
                "message": result["classified"].message[:300],
            },
        )
        try:
            self.client.require_healthy()
        except VerifierUnhealthyError:
            # Amendment A: a proof whose elaboration outlives both the bounded
            # client window and the bounded canary window wedges the serialized
            # instance (observed: 737 s elaboration vs 180 s + 2 x 180 s budgets).
            # Owner section 26: an infrastructure outcome is censored, never a
            # failed proof and never a reason to abandon the remaining surface.
            self.infra_log.event(
                "candidate_censored_two_strikes",
                {
                    "candidate_id": candidate_id,
                    "outcome": result["classified"].outcome.value,
                    "message": result["classified"].message[:300],
                },
            )
            self.recover(f"instance wedged again by {candidate_id}; censoring candidate")
        return result

    def recover(self, reason: str) -> None:
        if self.counters.recoveries >= int(S.ORACLE_INFRA["max_recoveries_per_run"]):
            raise RuntimeError("frozen container-recovery budget exhausted; stop and inspect")
        self.counters.recoveries += 1
        self.infra_log.event("recovery_start", {"n": self.counters.recoveries, "reason": reason})
        container_restart()
        wait_healthy(self.client)
        time.sleep(float(S.ORACLE_INFRA["restart_grace_s"]))
        canary = self.client.warmup()  # cold canary: raises on failure
        self.infra_log.event(
            "recovery_done",
            {"n": self.counters.recoveries, "canary_seconds": canary["seconds"]},
        )

    # -- one candidate --------------------------------------------------------------------

    def derive(
        self,
        entry: dict[str, Any],
        inputs: dict[str, Any],
        oracle_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        lattice = self.mapper.lattice(inputs["response"]) if entry["extractable"] else None
        meta = {
            "group_key": entry["group_key"],
            "seed": entry["seed"],
            "group_label": entry["group_label"],
            "component_id": entry["component_id"],
            "statement_id": entry["statement_id"],
            "slot": entry["slot"],
            "file": entry["file"],
            "line": entry["line"],
            "score": entry["score"],
            "acc": entry["acc"],
            "pred_kind": entry["pred_kind"],
        }
        return O.derive_facts(
            candidate_id=entry["candidate_id"],
            pred=inputs["pred"],
            response_text=inputs["response"],
            formal_statement=inputs["formal"] or "",
            oracle_result=oracle_result,
            token_lattice=lattice,
            meta=meta,
        )

    def process_one(
        self, entry: dict[str, Any], inputs: dict[str, Any], archive: RawArchive
    ) -> dict[str, Any]:
        if not entry["extractable"]:
            self.counters.sentinel += 1
            return self.derive(entry, inputs, None)
        result = self.submit(entry["candidate_id"], inputs["pred"])
        self.counters.submitted += 1
        classified = result["classified"]
        self.counters.bump_outcome(classified.outcome.value)
        if classified.outcome.is_infrastructure:
            self.counters.infra_censored += 1
        archive.write(entry["candidate_id"], S.sha256_text(inputs["pred"]), result)
        return self.derive(entry, inputs, result)


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------


def prepare(
    paths: RunPaths, planned: list[dict[str, Any]], formal_by_statement_id: dict[str, str]
) -> dict[str, Any]:
    """Re-verify every planned candidate's provenance; no oracle, no label write."""

    reader = SourceReader()
    counters = {"checked": 0, "code": 0, "sentinel": 0}
    started = time.monotonic()
    for entry in planned:
        inputs = candidate_inputs(entry, reader, formal_by_statement_id)
        if entry["extractable"] and extract_proof_from_text(inputs["response"], inputs["formal"]) != inputs["pred"]:
            raise RuntimeError(f"{entry['candidate_id']}: upstream extractor disagrees with pred")
        counters["checked"] += 1
        counters["code" if entry["extractable"] else "sentinel"] += 1
    counters["seconds"] = round(time.monotonic() - started, 2)
    counters["order_sha256"] = S.sha256_text(S.canonical_json([e["candidate_id"] for e in planned]))
    counters["planned"] = len(planned)
    return counters


def stage_preflight(
    paths: RunPaths,
    *,
    planned: list[dict[str, Any]],
    stamp: dict[str, Any],
    mapper: O.TokenMapper,
) -> int:
    formal_by_statement_id = load_formal_by_statement_id()
    checks = prepare(paths, planned, formal_by_statement_id)
    by_seed: dict[str, int] = {}
    for entry in planned:
        by_seed[entry["seed"]] = by_seed.get(entry["seed"], 0) + 1
    payload = {
        "artifact": "v5_p001_process_preflight",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_stamp": stamp,
        "tokenizer_identity": mapper.identity,
        "oracle": {
            "endpoint": S.ORACLE_INFRA["endpoint"],
            "container": S.ORACLE_INFRA["container"],
            "image_digest": S.ORACLE_INFRA["image_digest"],
        },
        "plan": {
            "n_planned": checks["planned"],
            "n_code_extractable": checks["code"],
            "n_sentinel": checks["sentinel"],
            "by_seed": by_seed,
            "order_sha256": checks["order_sha256"],
        },
        "provenance": {
            "candidates_reverified": checks["checked"],
            "seconds": checks["seconds"],
            "pred_sha256_matches": True,
            "response_sha256_matches": True,
            "upstream_extractor_agrees": True,
            "dataset_sha256_matches_surface": True,
        },
        "passed": True,
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.preflight.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"[v5p001] preflight PASSED n={checks['planned']} code={checks['code']} "
          f"sentinel={checks['sentinel']} order={checks['order_sha256'][:16]} "
          f"seconds={checks['seconds']}")
    return 0


def assert_stamp_matches(meta: dict[str, Any], stamp: dict[str, Any]) -> None:
    if meta["code_stamp"] != stamp:
        changed = [
            key for key in stamp if meta["code_stamp"].get(key) != stamp[key]
        ]
        raise RuntimeError(
            f"run stamp mismatch ({', '.join(changed)}); regenerate the run directory"
        )


def stage_process(
    paths: RunPaths,
    *,
    planned: list[dict[str, Any]],
    stamp: dict[str, Any],
    mapper: O.TokenMapper,
    client: O.OracleClient,
    limit: int | None,
    manage_container: bool,
) -> int:
    formal_by_statement_id = load_formal_by_statement_id()
    checks = prepare(paths, planned, formal_by_statement_id)
    planned_ids = [entry["candidate_id"] for entry in planned]
    expected = {
        "n_planned": len(planned),
        "n_code_extractable": checks["code"],
        "n_sentinel": checks["sentinel"],
        "order_sha256": checks["order_sha256"],
    }
    paths.root.mkdir(parents=True, exist_ok=True)
    fresh_run = not paths.run_meta.exists()
    if not fresh_run:
        meta = json.loads(paths.run_meta.read_text(encoding="utf-8"))
        assert_stamp_matches(meta, stamp)
        if meta["plan"] != expected:
            raise RuntimeError("run_meta plan differs from the current surface plan")
        if meta.get("debug_limit") != limit:
            raise RuntimeError("run_meta was created with a different --limit")
        meta["n_resumed_runs"] = meta.get("n_resumed_runs", 0) + 1
        paths.run_meta.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    else:
        meta = {
            "artifact": "v5_p001_process_run_meta",
            "experiment": S.EXPERIMENT_ID,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_stamp": stamp,
            "tokenizer_identity": mapper.identity,
            "oracle": {
                "endpoint": client.endpoint,
                "container": S.ORACLE_INFRA["container"],
                "image_digest": S.ORACLE_INFRA["image_digest"],
                "server_timeout_s": S.ORACLE_INFRA["server_timeout_s"],
                "max_single_retries": S.ORACLE_INFRA["max_single_retries"],
            },
            "plan": expected,
            "debug_limit": limit,
            "n_resumed_runs": 0,
        }
        paths.run_meta.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    labels = LabelsLog(paths.labels)
    existing = labels.load(set(planned_ids))
    archive = RawArchive(paths.raw)
    infra_log = InfraLog(paths.infra_log)
    if fresh_run:
        infra_log.event("process_start", {"n_planned": len(planned), "limit": limit})
    else:
        infra_log.event(
            "process_resume",
            {"n_planned": len(planned), "n_done": len(existing), "limit": limit},
        )

    stage = ProcessStage(
        paths=paths,
        client=client,
        mapper=mapper,
        infra_log=infra_log,
        manage_container=manage_container,
    )
    if not existing:
        canary = client.warmup()
        infra_log.event("cold_canary", {"seconds": canary["seconds"]})

    todo = [entry for entry in planned if entry["candidate_id"] not in existing]
    if limit is not None:
        todo = todo[:limit]
    reader = SourceReader()
    print(
        f"[v5p001] process start planned={len(planned)} done={len(existing)} "
        f"todo={len(todo)} code={checks['code']} endpoint={client.endpoint}"
    )
    try:
        for index, entry in enumerate(todo):
            inputs = candidate_inputs(entry, reader, formal_by_statement_id)
            record = stage.process_one(entry, inputs, archive)
            labels.append(record)
            counters = stage.counters
            if (index + 1) % PROGRESS_EVERY == 0 or index + 1 == len(todo):
                elapsed = time.monotonic() - counters.started_at
                rate = elapsed / (index + 1)
                eta = rate * (len(todo) - index - 1)
                print(
                    f"[v5p001] {len(existing) + index + 1}/{len(planned)} "
                    f"submitted={counters.submitted} sentinel={counters.sentinel} "
                    f"infra_censored={counters.infra_censored} recoveries={counters.recoveries} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s"
                )
    except KeyboardInterrupt:
        print("[v5p001] interrupted; the labels file is consistent and resumable")
        return 130
    infra_log.event(
        "process_end",
        {
            "n_done": len(existing) + len(todo),
            "n_planned": len(planned),
            "submitted": stage.counters.submitted,
            "sentinel": stage.counters.sentinel,
            "infra_censored": stage.counters.infra_censored,
            "recoveries": stage.counters.recoveries,
            "complete": len(existing) + len(todo) == len(planned),
        },
    )
    print(
        f"[v5p001] processing run finished: total={len(existing) + len(todo)}/{len(planned)} "
        f"submitted={stage.counters.submitted} sentinel={stage.counters.sentinel} "
        f"infra_censored={stage.counters.infra_censored} recoveries={stage.counters.recoveries}"
    )
    return 0


def stage_status(paths: RunPaths, planned: list[dict[str, Any]]) -> int:
    planned_ids = {entry["candidate_id"] for entry in planned}
    existing: dict[str, str] = {}
    if paths.labels.exists():
        labels = LabelsLog(paths.labels)
        existing = labels.load(set(planned_ids))
    print(
        json.dumps(
            {
                "n_planned": len(planned),
                "n_done": len(existing),
                "n_missing": len(planned) - len(existing),
                "run_meta_present": paths.run_meta.exists(),
                "freeze_present": paths.freeze.exists(),
                "validation_present": paths.validation.exists(),
            },
            indent=1,
        )
    )
    return 0


def raw_manifest(paths: RunPaths, archive: RawArchive) -> dict[str, Any]:
    entries = []
    for path in archive.files():
        entries.append(
            {"file": path.name, "sha256": S.sha256_file(path), "bytes": path.stat().st_size}
        )
    return {
        "artifact": "v5_p001_process_raw_manifest",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "directory": display_path(paths.raw),
        "n_files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "files": entries,
    }


def stage_freeze(
    paths: RunPaths,
    *,
    planned: list[dict[str, Any]],
    stamp: dict[str, Any],
) -> int:
    if not paths.run_meta.exists():
        raise RuntimeError("no run_meta.json: run --stage process first")
    meta = json.loads(paths.run_meta.read_text(encoding="utf-8"))
    assert_stamp_matches(meta, stamp)
    debug_limit = meta.get("debug_limit")
    if debug_limit is not None and paths.root == S.PROCESS_RUN_DIR:
        raise RuntimeError("the canonical run directory cannot carry a debug-limited freeze")
    labels = LabelsLog(paths.labels)
    existing = labels.load({entry["candidate_id"] for entry in planned})
    processed = [entry for entry in planned if entry["candidate_id"] in existing]
    missing = [entry["candidate_id"] for entry in planned if entry["candidate_id"] not in existing]
    archive = RawArchive(paths.raw)
    expected_names = {(paths.raw / raw_name(entry["candidate_id"])) for entry in processed
                      if entry["extractable"]}
    actual_names = set(archive.files())
    extra_raw = sorted(path.name for path in actual_names - expected_names)
    missing_raw = sorted(path.name for path in expected_names - actual_names)
    if debug_limit is None and (missing or extra_raw or missing_raw):
        raise RuntimeError(
            f"freeze refused: missing_labels={len(missing)} missing_raw={len(missing_raw)} "
            f"extra_raw={len(extra_raw)}"
        )
    if extra_raw:
        raise RuntimeError(f"freeze refused: {len(extra_raw)} raw items outside the plan")

    manifest = raw_manifest(paths, archive)
    paths.raw_manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    infra_events = (
        [json.loads(line) for line in paths.infra_log.read_text(encoding="utf-8").split("\n") if line.strip()]
        if paths.infra_log.exists()
        else []
    )
    outcomes: dict[str, int] = {}
    n_infra_censored = 0
    for line in paths.labels.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        oracle = record.get("oracle")
        if oracle:
            outcomes[oracle["outcome"]] = outcomes.get(oracle["outcome"], 0) + 1
            n_infra_censored += int(bool(oracle["infra"]))
    per_seed: dict[str, int] = {}
    n_processed_code = 0
    for entry in processed:
        per_seed[entry["seed"]] = per_seed.get(entry["seed"], 0) + 1
        n_processed_code += int(entry["extractable"])
    freeze = {
        "artifact": "v5_p001_process_freeze",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_stamp": stamp,
        "run_meta_sha256": S.sha256_file(paths.run_meta),
        "debug_limit": debug_limit,
        "labels": {
            "path": display_path(paths.labels),
            "sha256": S.sha256_file(paths.labels),
            "bytes": paths.labels.stat().st_size,
            "n_records": len(existing),
            "n_unique_candidate_ids": len(existing),
            "n_code_extractable": n_processed_code,
            "n_sentinel": len(processed) - n_processed_code,
            "by_seed": per_seed,
        },
        "raw": {
            "manifest": display_path(paths.raw_manifest),
            "n_files": manifest["n_files"],
            "total_bytes": manifest["total_bytes"],
            "manifest_sha256": S.sha256_file(paths.raw_manifest),
        },
        "infrastructure": {
            "n_submissions": n_processed_code,
            "outcome_histogram": outcomes,
            "n_infra_censored": n_infra_censored,
            "n_isolated_retries": sum(
                max(0, int(event.get("attempts", 1)) - 1)
                for event in infra_events
                if event.get("kind") == "infra_event"
            ),
            "n_infra_events": sum(
                1 for event in infra_events if str(event.get("kind", "")).startswith("infra_event")
            ),
            "n_recoveries": sum(
                1 for event in infra_events if event.get("kind") == "recovery_start"
            ),
            "n_truncated_partial_records": labels.n_truncated_suffixes,
        },
        "coverage": {
            "n_planned": len(planned),
            "n_processed": len(existing),
            "n_missing": len(missing),
            "n_missing_raw_files": len(missing_raw),
            "n_extra_raw_files": len(extra_raw),
        },
        "checks": {
            "labels_complete": not missing,
            "one_record_per_candidate": len(existing) == len(planned),
            "one_raw_item_per_processed_code_candidate": manifest["n_files"]
            == n_processed_code,
            "run_stamp_matches": True,
        },
        "passed": not extra_raw and not missing_raw and (not missing or debug_limit is not None),
    }
    paths.freeze.write_text(json.dumps(freeze, indent=1) + "\n", encoding="utf-8")
    print(
        f"[v5p001] freeze PASSED records={len(existing)}/{len(planned)} "
        f"raw_items={manifest['n_files']}/{n_processed_code} "
        f"labels_sha256={freeze['labels']['sha256'][:16]}"
    )
    return 0


def rebuild_oracle_result(archived: dict[str, Any]) -> dict[str, Any]:
    classified = Classified(
        outcome=VerifyOutcome(archived["classified"]["outcome"]),
        message=archived["classified"]["message"],
        redeclaration=bool(archived["classified"]["redeclaration"]),
    )
    return {
        "item": archived.get("item"),
        "classified": classified,
        "attempts": archived.get("attempts") or [],
        "seconds": archived.get("seconds"),
    }


def stage_validate(
    paths: RunPaths,
    *,
    planned: list[dict[str, Any]],
    stamp: dict[str, Any],
    mapper: O.TokenMapper,
) -> int:
    freeze = json.loads(paths.freeze.read_text(encoding="utf-8"))
    assert_stamp_matches(freeze, stamp)
    debug_limit = freeze.get("debug_limit")
    if debug_limit is None and freeze["coverage"]["n_missing"]:
        raise RuntimeError("canonical validation requires every planned candidate to be stored")
    if S.sha256_file(paths.labels) != freeze["labels"]["sha256"]:
        raise RuntimeError("labels file changed after the freeze")
    manifest = json.loads(paths.raw_manifest.read_text(encoding="utf-8"))
    if S.sha256_file(paths.raw_manifest) != freeze["raw"]["manifest_sha256"]:
        raise RuntimeError("raw manifest changed after the freeze")
    formal_by_statement_id = load_formal_by_statement_id()
    labels = LabelsLog(paths.labels)
    existing = labels.load({entry["candidate_id"] for entry in planned})
    archive = RawArchive(paths.raw)
    reader = SourceReader()
    stage = ProcessStage(
        paths=paths,
        client=None,
        mapper=mapper,
        infra_log=InfraLog(paths.infra_log),
        manage_container=False,
    )
    mismatches: list[str] = []
    n_rebuilt = 0
    for entry in planned:
        candidate_id = entry["candidate_id"]
        if candidate_id not in existing:
            continue
        inputs = candidate_inputs(entry, reader, formal_by_statement_id)
        oracle_result = None
        if entry["extractable"]:
            archived = archive.read(candidate_id)
            if archived["pred_sha256"] != S.sha256_text(inputs["pred"]):
                mismatches.append(f"{candidate_id}: archived pred sha differs")
                continue
            oracle_result = rebuild_oracle_result(archived)
        record = stage.derive(entry, inputs, oracle_result)
        n_rebuilt += 1
        if record_hash(record) != existing[candidate_id]:
            mismatches.append(f"{candidate_id}: re-derivation hash differs")
    manifest_ok = all(
        S.sha256_file(paths.raw / item["file"]) == item["sha256"] for item in manifest["files"]
    )
    payload = {
        "artifact": "v5_p001_process_validation",
        "experiment": S.EXPERIMENT_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_stamp": stamp,
        "debug_limit": debug_limit,
        "freeze_sha256": S.sha256_file(paths.freeze),
        "labels_sha256": S.sha256_file(paths.labels),
        "checks": {
            "n_planned": len(planned),
            "n_records": len(existing),
            "n_rebuilt_from_sources": n_rebuilt,
            "all_records_reproduced": not mismatches,
            "n_mismatches": len(mismatches),
            "raw_manifest_bytes_match": manifest_ok,
            "raw_file_count_matches_processed_code_candidates": len(manifest["files"])
            == sum(1 for entry in planned if entry["extractable"] and entry["candidate_id"] in existing),
            "coverage_complete": len(existing) == len(planned),
            "plan_order_sha256": S.sha256_text(
                S.canonical_json([entry["candidate_id"] for entry in planned])
            ),
        },
        "mismatches": mismatches[:MAX_VIOLATIONS_SHOWN],
        "passed": not mismatches and manifest_ok,
    }
    paths.validation.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"[v5p001] validate {'PASSED' if payload['passed'] else 'FAILED'} "
        f"rebuilt={n_rebuilt}/{len(existing)} of {len(planned)} planned "
        f"mismatches={len(mismatches)}"
    )
    return 0 if payload["passed"] else 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "process", "freeze", "validate", "status"),
        default="preflight",
    )
    parser.add_argument("--run-dir", default=str(S.PROCESS_RUN_DIR))
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug only; blocks freeze")
    parser.add_argument("--no-container-recovery", action="store_true")
    args = parser.parse_args(argv)

    paths = RunPaths(Path(args.run_dir))
    surface = load_surface()
    planned = build_plan(surface)
    stamp = code_stamp()

    if args.stage == "status":
        return stage_status(paths, planned)
    if args.stage == "preflight":
        return stage_preflight(
            paths, planned=planned, stamp=stamp, mapper=O.TokenMapper()
        )
    if args.stage == "process":
        client = O.OracleClient(endpoint=args.endpoint)
        return stage_process(
            paths,
            planned=planned,
            stamp=stamp,
            mapper=O.TokenMapper(),
            client=client,
            limit=args.limit,
            manage_container=not args.no_container_recovery,
        )
    if args.stage == "freeze":
        return stage_freeze(paths, planned=planned, stamp=stamp)
    return stage_validate(paths, planned=planned, stamp=stamp, mapper=O.TokenMapper())


if __name__ == "__main__":
    raise SystemExit(main())
