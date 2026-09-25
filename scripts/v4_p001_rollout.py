#!/usr/bin/env python3
"""V4-P001 formal runner -- frozen screening, freeze boundary, paired four-arm second stage.

Four subcommands, and the split is a scientific requirement rather than an implementation detail
(owner §11-§12). No stage may do the next stage's work, and in particular **the process that
discovers the primary cohort must not generate a single second-stage candidate**:

    --stage screening   one theta0 first attempt per frozen screening rank (1..640), canonical
                        verification, taxonomy, status; stops as soon as 128 primary semantic
                        failures exist in frozen order. Generation may not start unless the owner
                        launch authorization is passed.
    --stage freeze      reads the screening artifact and writes the three boundary artifacts --
                        `V4-P001_primary_cohort.json`, `V4-P001_diagnostic_derangement.json`,
                        `V4-P001_second_stage_plan.json` -- each recording
                        `second_stage_candidates_generated = 0`. CPU only: tokenizer, hashes,
                        the frozen derangement, the frozen arm schedule, the frozen seeds.
    --stage validate    the deterministic boundary check the owner requires before the second
                        stage: cohort membership, error taxonomy, prompt fit, derangement, seed set,
                        arm schedule, sealed reserve. Writes `V4-P001_boundary_validation.json`.
    --stage second      re-validates the boundary (fail-closed), then executes the 512 paired
                        A/B/C/D candidates in the frozen balanced arm order with one shared seed per
                        formal rank. Requires the owner launch authorization.

Nothing here draws theorems, re-orders the screen, re-fits anything, re-selects a family, or touches
n / temperature / top_p / max_response / the verifier semantics: every one of those is a frozen
object it reads and verifies by hash (`v4_p001_spec.load_frozen`), and every check is fail-closed.
The runner never converts an infrastructure outcome into a proof failure: an unresolved candidate is
written with `score = None` and an infrastructure status, and the sensitivity reading that completes
it as a failure belongs to the analyzer (Amendment A §5).

C′ verifier lifecycle (Amendment A §16, inherited from V3-R001): verification runs one candidate at a
time against the dedicated Kimina instance with ``MAX_REPLS = 1``, and any non-conclusive candidate
verdict triggers the mandatory recovery sequence -- restart, /health, nonformal canary -- before the
next candidate. The recovery adds no candidate attempt and moves no label. The ceilings are the
frozen ones: 192 per launch and 3 per theorem; a theorem that exceeds its exposure has its remaining
arm candidates marked INFRA_CENSORED (missing, never a failure) and the run continues.

Raw artifacts live under ``runs/v4_p001/rollout/`` (gitignored): only hashes, schema and provenance
are ever committed.

Reproduce:
    .venv/bin/python scripts/v4_p001_rollout.py --stage screening --dry-run
    .venv/bin/python scripts/v4_p001_rollout.py --stage validate   # after a formal screening+freeze
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # `typing.Self` is 3.11+, the package still supports 3.10
    from typing_extensions import Self

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import v4_p001_spec as S
from promptset_rollout_probe import complete_verifier_code
from tqdm import tqdm

from tinylean_rl.evaluation import v4_diagnostics as D
from tinylean_rl.evaluation import v4_prompts as P
from tinylean_rl.evaluation.v4_derange import BUCKET_EDGES, DERANGEMENT_VERSION, derange
from tinylean_rl.evaluation.v4_schedule import balance_report, schedule_hash
from tinylean_rl.evaluation.v4_seeds import first_stage_seed, second_stage_seed
from tinylean_rl.evaluation.v4_taxonomy import (
    PRIMARY_CATEGORIES,
    category_counts,
    classify_first_attempt,
    matched_primary_rules,
)
from tinylean_rl.inference.extract import _FENCED_LEAN, extract_proof
from tinylean_rl.verifier.policy import (
    CANARY_PROOF,
    Classified,
    VerificationSession,
    VerifierUnhealthyError,
    VerifyOutcome,
    classify_result_item,
    classify_transport_error,
)

#: Screening generation is chunked for engine efficiency; the *screen* is still the frozen order and
#: stops at the first chunk boundary after the 128th primary failure, so at most `SCREENING_CHUNK - 1`
#: theorems beyond the cohort are screened and reported.
SCREENING_CHUNK = 16
#: The second stage generates one position at a time, in the frozen balanced arm order, for a chunk
#: of theorems: positions stay ordered (position 1 completes before position 2 starts) while the
#: engine sees enough prompts per call to batch them.
SECOND_STAGE_CHUNK = 8

STAGES = ("screening", "freeze", "validate", "second")
FORMAL_STAGES = ("screening", "second")

#: Files whose bytes define this run's behaviour; their hashes go into every summary.
SCRIPT_VERSION_FILES = (
    "scripts/v4_p001_rollout.py", "scripts/v4_p001_spec.py",
    "scripts/promptset_rollout_probe.py",
    "src/tinylean_rl/evaluation/v4_prompts.py", "src/tinylean_rl/evaluation/v4_diagnostics.py",
    "src/tinylean_rl/evaluation/v4_derange.py", "src/tinylean_rl/evaluation/v4_schedule.py",
    "src/tinylean_rl/evaluation/v4_seeds.py", "src/tinylean_rl/evaluation/v4_taxonomy.py",
    "src/tinylean_rl/evaluation/v4_stats.py",
)


def abort(reason: str, checks: list | None = None) -> int:
    """The only legal way out of a failed preflight: say why, name the check, write nothing."""
    print(f"[ABORT BEFORE GENERATION] {reason}", file=sys.stderr)
    for c in checks or []:
        if not c["pass"]:
            print(f"  failed check: {c['check']} -- {c['detail']}", file=sys.stderr)
    return 3


def out_dir_of(args: argparse.Namespace) -> Path:
    path = Path(args.out_dir)
    return path if path.is_absolute() else ROOT / path


# --- the frozen policy's raw response, observed and never altered --------------------------------
#
# Every verdict in a V4 raw artifact is produced by `tinylean_rl.verifier.policy.VerificationSession`
# with batch_size = 1. The runner additionally needs one thing the frozen policy does not expose: the
# raw `/verify` result item, because the Arm-C/D diagnostic is defined (preregistration §8.1) as the
# first error-severity message of that item, with its `line L, column C:` prefix -- i.e.
# `v4_diagnostics.diagnostic_from_verify_item` applied to the response. The recorder below forwards
# its arguments and its return value unchanged and only keeps a reference to the decoded response, so
# the request, the retries, the canary gates, the classification and the verdict remain the policy's.
# It adds no verification behaviour and can change no outcome; `tests/test_v4_p001_fixtures.py`
# asserts that a recorded run and an unrecorded run classify identically.

class ResultItemRecorder:
    """Observe the frozen policy's transport without altering it."""

    def __init__(self) -> None:
        self.responses: list[dict] = []
        self._policy = None
        self._originals: tuple | None = None

    def __enter__(self) -> Self:
        policy = importlib.import_module("tinylean_rl.verifier.policy")
        self._policy = policy

        def wrap(original):
            def wrapper(*args, **kwargs):
                decoded = original(*args, **kwargs)
                self.responses.append(decoded)
                return decoded
            return wrapper

        self._originals = (policy.verify_code, policy.verify_codes)
        policy.verify_code = wrap(self._originals[0])
        policy.verify_codes = wrap(self._originals[1])
        return self

    def __exit__(self, *_exc) -> bool:
        if self._policy is not None and self._originals is not None:
            self._policy.verify_code, self._policy.verify_codes = self._originals
        return False

    def begin(self) -> None:
        self.responses.clear()

    def items(self) -> list[dict]:
        """The items of the last response, read with the policy's own item extractor."""
        if not self.responses or self._policy is None:
            return []
        return self._policy._response_items(self.responses[-1])

    def item_for(self, custom_id: str) -> dict | None:
        items = self.items()
        for item in items:
            if str(item.get("custom_id") or "") == custom_id:
                return item
        return items[0] if len(items) == 1 else None


def candidate_diagnostic(recorder: ResultItemRecorder, custom_id: str, classified: Classified,
                         tokenizer) -> dict:
    """The bounded, normalized Arm-C/D diagnostic of one verdict, with its provenance.

    Order is frozen: the raw item's first error-severity message (preregistration §8.1), and only if
    that is empty -- reachable for a ``sorry`` verdict, which Lean reports as a warning -- the
    policy's own verdict message. A verified candidate carries no diagnostic, and a candidate with
    nothing attributable gets `none` and can never select the cohort (`v4_p001_spec` asserts that).
    """
    out = {"diagnostic_text": "", "diagnostic_sha256": "", "diagnostic_tokens": 0,
           "diagnostic_source": "none", "truncated_diagnostic": False}
    if classified.outcome is VerifyOutcome.VERIFIED:
        return out
    text, _status = D.diagnostic_from_verify_item(recorder.item_for(custom_id))
    source = "verify_item"
    if not text.strip():
        if classified.outcome.is_conclusive and classified.message.strip():
            text, source = D.normalize_diagnostic(classified.message), "verdict_message"
        else:
            return out
    bounded, truncated = D.bound_diagnostic(text, tokenizer)
    out.update({"diagnostic_text": bounded, "diagnostic_sha256": S.sha256_text(bounded),
                "diagnostic_tokens": len(tokenizer.encode(bounded, add_special_tokens=False)),
                "diagnostic_source": source, "truncated_diagnostic": bool(truncated)})
    return out


# --- C′ verifier lifecycle: dedicated instance, mandatory recovery (Amendment A §16) --------------
#
# Copied from `scripts/v3_r001_rollout.py` (the C′ supervisor, closed provenance: V3's script hash is
# recorded in the V3-R001 run summary and that executable is never modified) with one addition: the
# per-theorem exposure of the frozen verifier plan. The invariant being enforced is unchanged --
#
#   no candidate verification attempt may leave a Lean computation occupying a reusable REPL after
#   that attempt is classified timed out/failed; capacity is restored before the next candidate
#
# -- and a restart still never adds a candidate attempt.

class RecoveryError(RuntimeError):
    """Capacity restoration could not be completed; the run must stop (fail-close)."""


def _published_endpoints(info: dict) -> set:
    """The (host_ip, host_port) pairs docker publishes for a container, as 'ip:port' strings."""
    out = set()
    for bindings in (info.get("ports") or {}).values():
        for binding in bindings or []:
            host_port = (binding or {}).get("HostPort")
            if host_port:
                out.add(f"{(binding.get('HostIp') or '0.0.0.0')}:{host_port}")
    return out


class VerifierRecovery:
    """The C′ supervisor around the dedicated verifier instance (`S.VERIFIER_INFRA`)."""

    def __init__(self, out_dir: Path, *, container: str | None = None,
                 endpoint: str | None = None, image: str | None = None) -> None:
        infra = S.VERIFIER_INFRA
        self.container = container or infra["container"]
        self.endpoint = (endpoint or infra["endpoint"]).rstrip("/")
        self.image = image or infra["image"]
        self.max_recoveries = int(infra["max_recoveries_per_run"])
        self.max_per_theorem = int(infra["max_recoveries_per_theorem"])
        self.restart_timeout = float(infra["restart_timeout_s"])
        self.health_timeout = float(infra["health_poll_timeout_s"])
        self.health_interval = float(infra["health_poll_interval_s"])
        self.restart_grace = int(infra["restart_grace_s"])
        self.out_dir = Path(out_dir)
        self.log_path = self.out_dir / S.RECOVERY_LOG_BASENAME
        self.n_attempted = 0
        self.n_succeeded = 0
        self.recoveries_by_theorem: dict[int, int] = {}
        self.events: list[dict] = []

    # -- docker -----------------------------------------------------------------------------------

    def _docker(self, *args: str, timeout: float) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["docker", *args], capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise RecoveryError(f"docker CLI is not available on this host: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RecoveryError(f"docker {' '.join(args)} exceeded {timeout}s") from exc

    def describe_instance(self) -> dict:
        """What docker sees for the dedicated container: state, image, env, published ports."""
        fmt = ("{{.State.Running}}\t{{.Config.Image}}\t{{json .Config.Env}}\t"
               "{{json .NetworkSettings.Ports}}\t{{.Id}}")
        proc = self._docker("inspect", "--type", "container", "--format", fmt, self.container,
                            timeout=30.0)
        if proc.returncode != 0:
            return {"present": False, "error": (proc.stderr or proc.stdout).strip()[:400]}
        parts = proc.stdout.strip().split("\t")
        if len(parts) < 5:
            return {"present": False,
                    "error": f"unexpected docker inspect output: {proc.stdout[:200]!r}"}
        running, image, env_json, ports_json, container_id = parts[:5]
        try:
            env = dict(item.split("=", 1) for item in json.loads(env_json) if "=" in item)
        except json.JSONDecodeError:
            env = {}
        try:
            ports = json.loads(ports_json)
        except json.JSONDecodeError:
            ports = {}
        return {"present": True, "running": running == "true", "image": image,
                "container_id": container_id[:12],
                "env": {key: env.get(key) for key in (
                    "LEAN_SERVER_MAX_REPLS", "LEAN_SERVER_MAX_REPL_USES",
                    "LEAN_SERVER_MAX_REPL_MEM", "LEAN_SERVER_MAX_WAIT", "LEAN_SERVER_PORT")},
                "published": sorted(_published_endpoints({"ports": ports})),
                "ports": ports}

    def verify_identity(self) -> dict:
        """Fail-close preflight: raise unless the endpoint is the dedicated frozen instance."""
        info = self.describe_instance()
        problems = []
        if not info.get("present"):
            problems.append(f"container {self.container!r} not found: {info.get('error', '')}")
        else:
            if not info["running"]:
                problems.append(f"container {self.container!r} is not running")
            if info["image"] != self.image:
                problems.append(f"image is {info['image']!r}, expected {self.image!r}")
            expected_repls = str(S.VERIFIER_INFRA["max_repls"])
            if info["env"].get("LEAN_SERVER_MAX_REPLS") != expected_repls:
                problems.append(
                    f"LEAN_SERVER_MAX_REPLS={info['env'].get('LEAN_SERVER_MAX_REPLS')!r}, expected "
                    f"{expected_repls!r}: the formal endpoint must not share the reusable pool")
            from urllib.parse import urlsplit
            host, port = urlsplit(self.endpoint).hostname, urlsplit(self.endpoint).port
            if port is None or f"{host}:{port}" not in info["published"]:
                problems.append(
                    f"endpoint {self.endpoint!r} is not a published port of {self.container!r} "
                    f"(published: {info['published']})")
        if problems:
            raise RecoveryError(
                "the dedicated verifier instance is not the frozen one -- " + "; ".join(problems) +
                f". Start it with: docker compose -f {S.VERIFIER_INFRA['compose_file']} up -d")
        return info

    # -- health, canary ---------------------------------------------------------------------------

    def health(self) -> dict:
        import httpx

        url = f"{self.endpoint}/health"
        try:
            response = httpx.get(url, timeout=10.0, trust_env=False)
            return {"url": url, "status": response.status_code, "body": str(response.text)[:200]}
        except httpx.HTTPError as exc:
            return {"url": url, "status": f"transport error: {type(exc).__name__}: {exc}"}

    def wait_healthy(self) -> dict:
        """Poll ``/health`` until it answers 200 or the frozen budget expires."""
        started = time.perf_counter()
        deadline = started + self.health_timeout
        last: dict = {}
        while True:
            last = self.health()
            if last.get("status") == 200:
                return {**last, "ok": True, "seconds": round(time.perf_counter() - started, 2)}
            if time.perf_counter() >= deadline:
                return {**last, "ok": False, "seconds": round(time.perf_counter() - started, 2)}
            time.sleep(self.health_interval)

    def cold_canary(self, tag: str) -> dict:
        """The post-restart canary: nonformal, generous first-use budget, re-warms the singleton."""
        import httpx

        from tinylean_rl.verifier.kimina import verify_code

        v = S.VERIFIER
        started = time.perf_counter()
        try:
            decoded = verify_code(CANARY_PROOF, custom_id=f"v4-p001-canary-{tag}",
                                  base_url=self.endpoint,
                                  timeout=v["first_timeout_s"] + v["client_slack_s"],
                                  server_timeout=int(v["first_timeout_s"]))
            items = decoded.get("results") or decoded.get("codes") or []
            classified = classify_result_item(items[0] if items else None)
        except httpx.HTTPError as exc:
            classified = classify_transport_error(exc)
        except Exception as exc:  # noqa: BLE001 - a health probe must report, never raise
            classified = Classified(VerifyOutcome.VERIFIER_UNHEALTHY, f"{type(exc).__name__}: {exc}")
        return {"verified": classified.outcome is VerifyOutcome.VERIFIED,
                "status": classified.outcome.value, "detail": classified.message[:300],
                "seconds": round(time.perf_counter() - started, 2),
                "proof_sha256": S.sha256_text(CANARY_PROOF),
                "uses_formal_theorem": False, "budget_s": v["first_timeout_s"]}

    # -- the mandated recovery sequence -----------------------------------------------------------

    def theorem_exhausted(self, rank: int) -> bool:
        """True when this theorem has already used its whole frozen recovery exposure."""
        return self.recoveries_by_theorem.get(rank, 0) >= self.max_per_theorem

    def recover(self, trigger: dict, rank: int) -> dict:
        """restart -> /health -> nonformal canary, before the next candidate. Fail-closed."""
        self.n_attempted += 1
        self.recoveries_by_theorem[rank] = self.recoveries_by_theorem.get(rank, 0) + 1
        if self.n_attempted > self.max_recoveries:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger, "rank": rank,
                    "reason": (f"recovery budget exhausted: {self.n_attempted - 1} of "
                               f"{self.max_recoveries} restarts already used in this launch")}
        try:
            self.verify_identity()
        except RecoveryError as exc:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger, "rank": rank,
                    "reason": str(exc)}

        print(f"[{S.EXPERIMENT_ID}] recovery {self.n_attempted}/{self.max_recoveries} "
              f"(theorem {rank}: {self.recoveries_by_theorem[rank]}/{self.max_per_theorem}): "
              f"docker restart {self.container} (trigger: outcome={trigger.get('outcome')} "
              f"{trigger.get('custom_id')})")
        started = time.perf_counter()
        proc = self._docker("restart", "-t", str(self.restart_grace), self.container,
                            timeout=self.restart_timeout)
        restart_seconds = round(time.perf_counter() - started, 2)
        if proc.returncode != 0:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger, "rank": rank,
                    "restart_seconds": restart_seconds,
                    "reason": (f"docker restart failed (rc={proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[:300]}")}

        health = self.wait_healthy()
        canary = (self.cold_canary(f"rec{self.n_attempted}") if health["ok"]
                  else {"verified": False, "status": "skipped",
                        "detail": "health poll failed; the canary was not attempted"})
        ok = bool(health["ok"] and canary["verified"])
        event = {"n": self.n_attempted, "at": datetime.now(timezone.utc).isoformat(),
                 "theorem_rank": rank, "theorem_recovery_index": self.recoveries_by_theorem[rank],
                 "trigger": trigger, "restart_seconds": restart_seconds, "health": health,
                 "canary": canary, "recovery_seconds": round(time.perf_counter() - started, 2),
                 "ok": ok}
        self.events.append(event)
        S.append_rows_durable(self.log_path, [event])
        if ok:
            self.n_succeeded += 1
        print(f"[{S.EXPERIMENT_ID}] recovery {self.n_attempted}: restart {restart_seconds}s, "
              f"health {health.get('seconds')}s, canary {canary.get('status')} "
              f"{canary.get('seconds')}s -> {'ok' if ok else 'FAILED'}")
        return {"ok": ok, **event}


def verifier_endpoint() -> str:
    return S.VERIFIER_INFRA["endpoint"]


def make_session() -> VerificationSession:
    v = S.VERIFIER
    return VerificationSession(base_url=verifier_endpoint(), server_timeout=v["server_timeout_s"],
                               client_slack=v["client_slack_s"], batch_size=v["batch_size"],
                               max_single_retries=v["max_single_retries"],
                               canary_timeout=v["canary_timeout_s"],
                               canary_retries=v["canary_retries"])


def check_verifier(session: VerificationSession) -> dict:
    """`/health`, then the cold-start canary. Neither touches a formal theorem."""
    import httpx

    url = f"{verifier_endpoint()}/health"
    out = {"health_url": url, "endpoint": verifier_endpoint()}
    try:
        r = httpx.get(url, timeout=10.0, trust_env=False)
        out["health_status"] = r.status_code
        out["health_body"] = str(r.text)[:200]
    except httpx.HTTPError as exc:
        out["health_status"] = f"transport error: {type(exc).__name__}: {exc}"
        return out
    if r.status_code != 200:
        return out

    t0 = time.perf_counter()
    try:
        from tinylean_rl.verifier.kimina import verify_code

        decoded = verify_code(CANARY_PROOF, custom_id="v4-p001-canary",
                              base_url=verifier_endpoint(),
                              timeout=S.VERIFIER["first_timeout_s"] + S.VERIFIER["client_slack_s"],
                              server_timeout=int(S.VERIFIER["first_timeout_s"]))
        items = decoded.get("results") or decoded.get("codes") or []
        c = classify_result_item(items[0] if items else None)
    except httpx.HTTPError as exc:
        c = classify_transport_error(exc)
    except Exception as exc:  # noqa: BLE001 - a health probe must report, never raise
        c = Classified(VerifyOutcome.VERIFIER_UNHEALTHY, f"{type(exc).__name__}: {exc}")
    session._record("canary", f"cold-start outcome={c.outcome.value} {c.message}")
    out["canary"] = {"verified": c.outcome is VerifyOutcome.VERIFIED, "status": c.outcome.value,
                     "detail": c.message, "seconds": round(time.perf_counter() - t0, 2),
                     "proof_sha256": S.sha256_text(CANARY_PROOF),
                     "uses_formal_theorem": False,
                     "budget_s": S.VERIFIER["first_timeout_s"]}
    return out


def verify_with_recovery(session: VerificationSession, recovery: VerifierRecovery,
                         proof: str, custom_id: str, *, rank: int, recorder: ResultItemRecorder,
                         tokenizer, want_diagnostic: bool,
                         ) -> tuple[Classified, dict, str | None, bool]:
    """One candidate's frozen policy run; capacity is restored before the next candidate.

    Returns ``(classified, diagnostic, abort_reason, theorem_exhausted)``. ``abort_reason`` non-None
    means capacity could not be restored and the caller must stop (fail-close, resumable);
    ``theorem_exhausted`` means this theorem has no recovery exposure left, so its remaining
    candidates are censored (missing data, never failures) and the run moves on.
    """
    if want_diagnostic:
        recorder.begin()
    try:
        classified = session.verify([proof], [custom_id])[0]
    except VerifierUnhealthyError as exc:
        classified = Classified(
            VerifyOutcome.UNRESOLVED_INFRA_ERROR,
            f"candidate verification interrupted by the canary fail-close "
            f"(rank={rank} id={custom_id}): {exc}"[:500])
    diagnostic = (candidate_diagnostic(recorder, custom_id, classified, tokenizer)
                  if want_diagnostic else
                  {"diagnostic_text": "", "diagnostic_sha256": "", "diagnostic_tokens": 0,
                   "diagnostic_source": "none", "truncated_diagnostic": False})
    if classified.outcome.is_conclusive:
        return classified, diagnostic, None, False
    if recovery.theorem_exhausted(rank):
        return classified, diagnostic, None, True
    restored = recovery.recover({"outcome": classified.outcome.value, "custom_id": custom_id,
                                 "message": classified.message[:300]}, rank)
    if not restored["ok"]:
        return classified, diagnostic, (
            f"capacity restoration failed after {classified.outcome.value} on rank={rank} "
            f"id={custom_id}: {restored.get('reason') or 'the recovery canary did not verify'}"), False
    return classified, diagnostic, None, False


def flush_session_events(session: VerificationSession, path: Path, cursor: int) -> int:
    """Append the policy's own event trail durably and return the new cursor."""
    events = session.events[cursor:]
    if events:
        S.append_rows_durable(path, [{"at": e.at, "kind": e.kind, "detail": e.detail}
                                     for e in events])
    return len(session.events)


def verifier_infrastructure_stamp(session: VerificationSession, recovery: VerifierRecovery,
                                 instance: dict | None, events_path: Path,
                                 recorded_items: bool) -> dict:
    """The C′ infrastructure block of a run summary: what ran, where, and what it cost."""
    return {
        "endpoint": recovery.endpoint,
        "container": recovery.container,
        "image": recovery.image,
        "max_repls": S.VERIFIER_INFRA["max_repls"],
        "verification_concurrency": S.VERIFIER_INFRA["verification_concurrency"],
        "policy_batch_size": S.VERIFIER["batch_size"],
        "compose_file": S.VERIFIER_INFRA["compose_file"],
        "dedicated_instance": instance,
        "recoveries_allowed": recovery.max_recoveries,
        "recoveries_allowed_per_theorem": recovery.max_per_theorem,
        "recoveries_attempted": recovery.n_attempted,
        "recoveries_succeeded": recovery.n_succeeded,
        "recoveries_by_theorem": {str(k): v for k, v in sorted(recovery.recoveries_by_theorem.items())
                                  if v},
        "recovery_events": recovery.events,
        "recovery_log_path": str(recovery.log_path),
        "raw_result_items_recorded": recorded_items,
        "raw_item_recording_changes_no_verdict": True,
        "verifier_events_recorded": len(session.events),
        "verifier_events_path": str(events_path),
    }


# --- the frozen input surface --------------------------------------------------------------------

def load_surface(tokenizer) -> dict:
    """statement_id -> canonical messages, first-attempt prompt and formal statement.

    The parquet's bytes are pinned by the frozen pool (`load_frozen`), and the first-attempt prompt is
    rendered by the same `v4_prompts.render_prompt` the Arm-A provenance audit used, so screening and
    Arm A are the same string by construction. The token count is checked against the pool's frozen
    `prompt_tokens` per theorem in `build_screening_plan`.
    """
    df = pd.read_parquet(ROOT / S.TRAIN_PARQUET,
                         columns=["statement_id", "formal_statement", "prompt"])
    out: dict = {}
    for row in df.itertuples(index=False):
        sid = str(row.statement_id)
        raw = row.prompt.tolist() if hasattr(row.prompt, "tolist") else list(row.prompt)
        messages = P.canonical_messages(raw)
        prompt = P.render_prompt(tokenizer, messages)
        entry = {"statement_id": sid,
                 "formal_statement": str(row.formal_statement),
                 "messages": messages, "prompt": prompt,
                 "prompt_sha256": S.sha256_text(prompt),
                 "prompt_token_count": len(tokenizer.encode(prompt, add_special_tokens=False))}
        if sid in out:
            previous = out[sid]
            if (previous["prompt_sha256"], previous["formal_statement"]) != (
                    entry["prompt_sha256"], entry["formal_statement"]):
                raise S.FrozenViolation(
                    f"statement {sid} appears twice in the pinned parquet with different content; "
                    "the screening prompt of this theorem would not be well defined")
            continue
        out[sid] = entry
    if not out:
        raise S.FrozenViolation(f"{S.TRAIN_PARQUET} yielded no theorems")
    return out


def build_screening_plan(frozen: S.Frozen, surface: dict) -> list[dict]:
    """The 640 theorems a formal screen may touch, in the frozen order, with their frozen seed."""
    schedule = frozen.seeds["screening_schedule"]
    plan = []
    for member in frozen.screening_theorems:
        rank = member["screening_rank"]
        sid = member["statement_id"]
        if sid not in surface:
            raise S.FrozenViolation(f"screening rank {rank}: statement {sid} is not in the pinned "
                                    "promptset parquet")
        row = surface[sid]
        if row["prompt_token_count"] != member["prompt_tokens"]:
            raise S.FrozenViolation(
                f"screening rank {rank} ({sid}): the rendered first-attempt prompt tokenizes to "
                f"{row['prompt_token_count']} but the frozen pool records {member['prompt_tokens']}"
                " -- the string this run would send is not the string the pool was built from")
        entry_schedule = schedule[rank - 1]
        if entry_schedule["statement_id"] != sid or entry_schedule["seed"] != first_stage_seed(rank):
            raise S.FrozenViolation(
                f"screening rank {rank}: the frozen seed schedule does not agree with "
                "first_stage_seed(rank) and the pool order")
        plan.append({
            "screening_rank": rank, "statement_id": sid, "component_id": member["component_id"],
            "name": member["name"], "source": member["source"], "tier": member["tier"],
            "classes": list(member["classes"]), "prompt_tokens": member["prompt_tokens"],
            "seed": first_stage_seed(rank),
        })
    if len({e["statement_id"] for e in plan}) != len(plan):
        raise S.FrozenViolation("the screening plan repeats a statement")
    return plan


def context_check(tokenizer, messages, *, failed_proof: str, diagnostic: str) -> dict:
    """Owner §I, measured on the rendered Arm-C prompt of this theorem and nothing else."""
    prompt = P.render_arm(tokenizer, messages, arm="C_VERIFIER_REPAIR", failed_proof=failed_proof,
                          diagnostic=diagnostic)
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    total = tokens + S.MAX_RESPONSE_TOKENS
    return {"context_fits": total <= S.MAX_MODEL_LEN, "context_tokens_with_response": total,
            "arm_c_prompt_tokens": tokens}


# --- stage: screening ----------------------------------------------------------------------------

def screen_one(tokenizer, session: VerificationSession, recovery: VerifierRecovery,
               recorder: ResultItemRecorder, entry: dict, surface_row: dict, completion,
               generation_seconds: float, env: dict, run_id: str,
               n_primary_so_far: int) -> tuple[dict | None, str | None]:
    """Generate-free part of the screening loop: extract, verify, diagnose, classify, row.

    Returns ``(row, abort_reason)``; ``row=None`` means the candidate may not be written and the
    caller must stop. The row is validated here, so a malformed row cannot reach disk.
    """
    text = completion.text
    try:
        extracted = extract_proof(text)
    except ValueError:
        extracted = ""
    lean_source = complete_verifier_code(surface_row["formal_statement"], extracted)
    has_lean_block = bool(_FENCED_LEAN.search(text))
    custom_id = f"screening-{entry['screening_rank']}"
    if lean_source is None:
        # Nothing Lean was produced: the frozen policy is not called (it would have nothing to
        # verify), the candidate is a conclusive model failure under the frozen `format_error`
        # status, and it is never infrastructure.
        verify_status = S.FORMAT_ERROR
        message = ("no Lean proof body could be extracted from the completion; the candidate was "
                   "never sent to the verifier")
        diagnostic = {"diagnostic_text": "", "diagnostic_sha256": "", "diagnostic_tokens": 0,
                      "diagnostic_source": "none", "truncated_diagnostic": False}
        abort_reason = None
    else:
        recorder.begin()
        classified, diagnostic, abort_reason, _theorem_exhausted = verify_with_recovery(
            session, recovery, lean_source, custom_id, rank=entry["screening_rank"],
            recorder=recorder, tokenizer=tokenizer, want_diagnostic=True)
        verify_status = classified.outcome.value
        message = classified.message

    category = classify_first_attempt(
        verify_status=verify_status,
        lean_message=message if lean_source is not None else "",
        truncated=bool(completion.finish_reason == "length"), format_ok=lean_source is not None,
        has_lean_block=has_lean_block, extracted=extracted)
    if category in PRIMARY_CATEGORIES and not diagnostic["diagnostic_text"].strip():
        # A Lean-level semantic failure whose diagnostic could not be attributed means Arms C and D
        # would have nothing to show. The frozen schema may not carry such a row into the cohort, so
        # this stops rather than silently mislabelling it.
        reason = (f"screening rank {entry['screening_rank']}: the classifier assigned the primary "
                  f"category {category!r} but no diagnostic text could be attributed "
                  f"(verify_status={verify_status!r}); Arms C and D cannot be rendered for this "
                  "theorem. Stopping before anything is written.")
        return None, reason
    context = (context_check(tokenizer, surface_row["messages"], failed_proof=extracted,
                             diagnostic=diagnostic["diagnostic_text"])
               if lean_source is not None and diagnostic["diagnostic_text"].strip() else
               {"context_fits": True, "context_tokens_with_response": None,
                "arm_c_prompt_tokens": None})
    in_cohort = (category in PRIMARY_CATEGORIES and bool(diagnostic["diagnostic_text"].strip())
                 and context["context_fits"])
    status = S.screening_status(verify_status=verify_status, category=category,
                                primary_eligible=in_cohort, context_fits=context["context_fits"])
    if status == S.CONTEXT_INELIGIBLE:
        # Owner §I: excluded from the cohort and reported; the measured attempt is still recorded.
        pass
    score = S.candidate_score(verify_status)
    row = {
        "schema_version": S.SCREENING_SCHEMA_VERSION, "run_id": run_id,
        "experiment_id": S.EXPERIMENT_ID,
        "screening_rank": entry["screening_rank"],
        "formal_rank": (n_primary_so_far + 1) if in_cohort and n_primary_so_far < S.N_PRIMARY else None,
        "statement_id": entry["statement_id"], "component_id": entry["component_id"],
        "name": entry["name"], "source": entry["source"], "tier": entry["tier"],
        "classes": entry["classes"], "prompt_tokens": entry["prompt_tokens"],
        "prompt_sha256": surface_row["prompt_sha256"],
        "prompt_token_count": surface_row["prompt_token_count"], "seed": entry["seed"],
        "model_sha256": env["model_sha256"], "model_revision": env["model_revision"],
        "completion_text": text, "completion_sha256": S.sha256_text(text),
        "generated_tokens": len(list(completion.token_ids)),
        "truncated": bool(completion.finish_reason == "length"),
        "format_ok": lean_source is not None, "has_lean_block": has_lean_block,
        "extracted_proof": extracted, "extracted_proof_sha256": S.sha256_text(extracted),
        "extracted_proof_present": bool(extracted.strip()),
        "extracted_proof_tokens": (len(tokenizer.encode(extracted, add_special_tokens=False))
                                   if extracted.strip() else 0),
        **diagnostic,
        "context_fits": context["context_fits"],
        "context_tokens_with_response": context["context_tokens_with_response"],
        "verified": score == 1,
        "score": score,
        "acc": score,
        "verify_status": verify_status, "lean_message": message[:500],
        "error_category": category, "screening_status": status,
        "primary_eligible": in_cohort,
        "generation_time": round(generation_seconds, 4),
        "verification_time": None,
        "primary_rule_matches": list(matched_primary_rules(
            verify_status=verify_status, lean_message=message if lean_source is not None else None)),
        "host": env["hostname"], "gpu": env["gpu_names"].splitlines()[0] if env["gpu_names"] else "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    S.validate_screening_row(row)
    return row, abort_reason


def stage_screening(args: argparse.Namespace) -> int:
    out_dir = out_dir_of(args)
    raw_path = out_dir / S.SCREENING_RAW_BASENAME
    summary_path = out_dir / S.SCREENING_SUMMARY_BASENAME

    print(f"[{S.EXPERIMENT_ID}] stage=screening dry_run={args.dry_run} resume={args.resume} "
          f"out={raw_path}")
    env = S.collect_env()
    try:
        S.check_environment(env, require_formal_host=not args.dry_run)
        frozen = S.load_frozen(verify_files=True)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] host={env['hostname']} git={env['git_revision'][:9]} "
          f"branch={env['branch']} | {len(frozen.checks)} frozen-artifact checks passed")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    try:
        surface = load_surface(tokenizer)
        plan = build_screening_plan(frozen, surface)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] screening plan: {len(plan)} theorems in frozen order, "
          f"{len(surface)} theorems loaded from the pinned parquet; every prompt re-tokenized to its "
          "frozen length")
    for tag, entry in (("first", plan[0]), ("last", plan[-1])):
        print(f"[{S.EXPERIMENT_ID}] {tag}: rank={entry['screening_rank']} name={entry['name']} "
              f"source={entry['source']} tier={entry['tier']} prompt_tokens={entry['prompt_tokens']} "
              f"seed={entry['seed']}")

    # resume state: one row per screened theorem, keyed by the frozen screening rank
    existing: dict[int, dict] = {}
    if raw_path.exists():
        if not args.resume:
            return abort(f"{raw_path} already exists; refusing to append to a raw artifact without "
                         "--resume")
        try:
            for row in S.rows_jsonl(raw_path):
                S.validate_screening_row(row)
                rank = row["screening_rank"]
                if rank in existing:
                    return abort(f"{raw_path} holds two rows for screening_rank {rank}; the "
                                 "artifact is not consumable")
                existing[rank] = row
        except S.FrozenViolation as exc:
            return abort(f"existing screening artifact is not consumable: {exc}")
        off_sample = sorted(set(existing) - {e["screening_rank"] for e in plan})
        if off_sample:
            return abort(f"the screening artifact holds rank(s) outside the frozen plan: "
                         f"{off_sample[:5]}")
        if any(row["statement_id"] != surface[row["statement_id"]]["statement_id"]
               for row in existing.values()):
            return abort("a stored screening row does not match the pinned parquet")
    counted = [row for _, row in sorted(existing.items())]
    n_primary = sum(1 for r in counted if r["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE)
    resume_plan = {"rows_on_disk": len(existing), "primary_failures_on_disk": n_primary,
                   "cohort_complete": n_primary >= S.N_PRIMARY,
                   "to_screen": [e["screening_rank"] for e in plan if e["screening_rank"] not in existing]}
    print(f"[{S.EXPERIMENT_ID}] resume: {len(existing)} screened, {n_primary}/{S.N_PRIMARY} primary "
          f"failures, {len(resume_plan['to_screen'])} theorem(s) to screen")

    # verifier instance, identity, health and canary -- all before anything is generated
    recovery = VerifierRecovery(out_dir)
    try:
        instance = recovery.verify_identity()
        identity_error = None
    except RecoveryError as exc:
        instance, identity_error = recovery.describe_instance(), str(exc)
    if identity_error is not None:
        if not args.dry_run:
            return abort(identity_error)
        print(f"[note] dry-run: the dedicated-instance check is report-only. {identity_error}")
    else:
        print(f"[{S.EXPERIMENT_ID}] dedicated verifier instance: {instance.get('container_id')} "
              f"{instance.get('image')} published={instance.get('published')} "
              f"env={instance.get('env')}")

    stamp = {"experiment_id": S.EXPERIMENT_ID, "stage": "screening",
             "schema_version": S.SCREENING_SCHEMA_VERSION,
             "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
             "frozen_settings": S.FROZEN_SETTINGS,
             "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
             "script_versions": S.script_version_hash(*SCRIPT_VERSION_FILES),
             "prereg_commit": S.PREREG_COMMIT, "git": env,
             "screening_budget": frozen.pool["screening_budget"],
             "screening_order_hash": frozen.pool["order_hash"],
             "screening_seed_hash": frozen.seeds["screening_seed_hash"],
             "pool_hash": frozen.pool["pool_hash"],
             "n_planned": len(plan), "resume_plan": resume_plan,
             "checks": frozen.checks, "mode": "FORMAL" if not args.dry_run else "DRY_RUN"}

    if args.dry_run:
        stamp["formal_candidates_generated"] = 0
        stamp["raw_artifact_written"] = False
        stamp["verifier_identity_error"] = identity_error
        print(json.dumps({k: stamp[k] for k in
                          ("mode", "run_id", "n_planned", "resume_plan", "screening_budget",
                           "screening_order_hash", "screening_seed_hash", "pool_hash",
                           "frozen_settings_sha256")}, indent=2))
        print(f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE: llm.load_called=false generate_called=false "
              "model.generate on formal theorems=0 candidates_generated=0 verifier_formal_calls=0 "
              "formal_raw_result_files_created=0")
        return 0

    session = make_session()
    health = check_verifier(session)
    print(f"[{S.EXPERIMENT_ID}] verifier health: {health.get('health_status')} "
          f"canary={health.get('canary', {}).get('status')} "
          f"({health.get('canary', {}).get('seconds')}s)")
    if not health.get("canary", {}).get("verified"):
        return abort(f"verifier is not healthy (canary did not verify): {health}")
    verifier_events_path = out_dir / S.VERIFIER_EVENTS_BASENAME
    events_cursor = flush_session_events(session, verifier_events_path, 0)

    if n_primary >= S.N_PRIMARY:
        print(f"[{S.EXPERIMENT_ID}] the cohort is already complete on disk ({n_primary} primary "
              "failures); nothing to generate. Run --stage freeze.")
        stamp.update({"candidates_generated": 0, "n_primary_failures": n_primary,
                      "cohort_complete": True, "verifier_health": health,
                      "verifier_infrastructure": verifier_infrastructure_stamp(
                          session, recovery, instance, verifier_events_path, False)})
        S.write_json_atomic(summary_path, stamp)
        return 0

    todo = [e for e in plan if e["screening_rank"] not in existing]
    generated = 0
    screens = []
    abort_reason: str | None = None

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"[{S.EXPERIMENT_ID}] loading theta0 into vLLM ({S.MODEL['dir']})")
    llm = LLM(model=str(ROOT / S.MODEL["dir"]), tokenizer=str(ROOT / S.MODEL["dir"]),
              trust_remote_code=True, max_model_len=S.MAX_MODEL_LEN,
              gpu_memory_utilization=S.GPU_MEMORY_UTILIZATION, max_num_seqs=S.MAX_NUM_SEQS,
              max_num_batched_tokens=S.MAX_NUM_BATCHED_TOKENS, disable_log_stats=True)
    env["model_sha256"] = S.MODEL["weights_sha256"]
    env["model_revision"] = S.MODEL["revision"]
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with ResultItemRecorder() as recorder:
        chunks = [todo[i:i + SCREENING_CHUNK] for i in range(0, len(todo), SCREENING_CHUNK)]
        for chunk in tqdm(chunks, desc="V4-P001 screening", unit="chunk", dynamic_ncols=True):
            prompts, params = [], []
            for entry in chunk:
                prompts.append(surface[entry["statement_id"]]["prompt"])
                params.append(SamplingParams(temperature=S.TEMPERATURE, top_p=S.TOP_P,
                                             max_tokens=S.MAX_RESPONSE_TOKENS, n=1,
                                             seed=entry["seed"]))
            gen_t = time.perf_counter()
            outputs = llm.generate(prompts, params)
            gen_seconds = (time.perf_counter() - gen_t) / max(1, len(prompts))
            for entry, output in zip(chunk, outputs):
                row, abort_reason = screen_one(
                    tokenizer, session, recovery, recorder, entry,
                    surface[entry["statement_id"]], output.outputs[0], gen_seconds, env,
                    stamp["run_id"], n_primary)
                if row is None:
                    break
                S.append_rows_durable(raw_path, [row])
                generated += 1
                screens.append({"screening_rank": row["screening_rank"],
                                "screening_status": row["screening_status"],
                                "error_category": row["error_category"]})
                if row["screening_status"] == S.PRIMARY_SEMANTIC_FAILURE:
                    n_primary += 1
                events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
                if abort_reason:
                    break
            if abort_reason or n_primary >= S.N_PRIMARY:
                break

    events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
    status_counts = {name: 0 for name in S.SCREENING_STATUSES}
    for row in screens:
        status_counts[row["screening_status"]] += 1
    screened_ranks = {row["screening_rank"] for row in screens}
    off_sample = sorted(screened_ranks - {e["screening_rank"] for e in plan})
    if off_sample:
        return abort(f"a written screening row is outside the frozen plan: {off_sample[:5]}")
    stamp.update({
        "candidates_generated": generated,
        "n_screened_this_launch": generated,
        "n_primary_failures": n_primary,
        "cohort_complete": n_primary >= S.N_PRIMARY,
        "screens_used": len(existing) + generated,
        "screens_remaining": S.MAX_SCREENING - (len(existing) + generated),
        "status_counts_this_launch": status_counts,
        "theorems_this_launch": screens,
        "generation_seconds": round(time.perf_counter() - t0, 1),
        "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vllm_version": vllm.__version__,
        "raw_artifact_sha256": S.sha256_file(raw_path) if raw_path.exists() else None,
        "verifier_health": health,
        "verifier_infrastructure": verifier_infrastructure_stamp(session, recovery, instance,
                                                                verifier_events_path, True),
    })
    if abort_reason:
        stamp["aborted"] = abort_reason
        S.write_json_atomic(summary_path.with_name(
            S.SCREENING_SUMMARY_BASENAME.replace(".json", ".aborted.json")), stamp)
        return abort(f"{abort_reason}. {generated} screened theorem(s) were kept; resume with "
                     "--resume once the dedicated verifier instance is healthy.")
    S.write_json_atomic(summary_path, stamp)
    print(json.dumps({"n_screened_this_launch": generated, "n_primary_failures": n_primary,
                      "cohort_complete": stamp["cohort_complete"],
                      "screens_used": stamp["screens_used"],
                      "status_counts": status_counts, "recoveries": recovery.n_attempted,
                      "raw": str(raw_path), "summary": str(summary_path)}, indent=2))
    if stamp["cohort_complete"]:
        print(f"[{S.EXPERIMENT_ID}] SCREENING COMPLETE: {n_primary} primary semantic failures in "
              "frozen order. STOPPING here by design -- the process that discovers the cohort must "
              "not generate a second-stage candidate. Next: --stage freeze, then --stage validate.")
        return 0
    print(f"[{S.EXPERIMENT_ID}] STOP: only {n_primary} of {S.N_PRIMARY} primary semantic failures "
          f"were collected within the frozen {S.MAX_SCREENING}-screen budget. The run stops and "
          "reports before any second-stage generation (owner §C).")
    return 2


# --- stage: freeze (owner §12 boundary; CPU only, zero generation) --------------------------------

def seal(payload: dict) -> dict:
    """Add the artifact's own canonical content hash, computed over everything else."""
    return {**payload, "content_sha256": S.sha({k: v for k, v in payload.items()
                                                if k != "content_sha256"})}


def unseal(payload: dict) -> str:
    return S.sha({k: v for k, v in payload.items() if k != "content_sha256"})


def _distribution(values: list[int]) -> dict:
    ordered = sorted(values)

    def q(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {"n": len(ordered), "min": ordered[0], "p50": q(0.5), "p95": q(0.95), "p99": q(0.99),
            "max": ordered[-1], "mean": round(sum(ordered) / len(ordered), 1)}


def read_screening(raw_path: Path) -> list[dict]:
    """Every screening row, validated, in the frozen screening order."""
    if not raw_path.exists():
        raise S.FrozenViolation(f"{raw_path} does not exist; this stage may only run after a "
                                "completed --stage screening")
    rows = []
    for row in S.rows_jsonl(raw_path):
        S.validate_screening_row(row)
        rows.append(row)
    if not rows:
        raise S.FrozenViolation(f"{raw_path} holds no screening rows")
    ranks = [row["screening_rank"] for row in rows]
    if len(set(ranks)) != len(ranks):
        raise S.FrozenViolation(f"{raw_path} holds duplicate screening_rank rows")
    return sorted(rows, key=lambda row: row["screening_rank"])


def cohort_rows(rows: list[dict]) -> list[dict]:
    """The 128 primary semantic failures, in frozen order, exactly once each."""
    by_rank: dict[int, dict] = {}
    for row in rows:
        rank = row.get("formal_rank")
        if rank is None:
            continue
        if rank in by_rank:
            raise S.FrozenViolation(f"two screening rows carry formal_rank {rank}")
        by_rank[rank] = row
    missing = [rank for rank in range(1, S.N_PRIMARY + 1) if rank not in by_rank]
    if missing:
        raise S.FrozenViolation(
            f"the screening artifact holds {len(by_rank)} of {S.N_PRIMARY} cohort members; missing "
            f"formal_rank(s) {missing[:8]}{'...' if len(missing) > 8 else ''}. Stage 2 must not begin "
            "unless exactly 128 primary semantic failures were obtained within the frozen budget")
    cohort = []
    for rank in range(1, S.N_PRIMARY + 1):
        row = by_rank[rank]
        if row["screening_status"] != S.PRIMARY_SEMANTIC_FAILURE:
            raise S.FrozenViolation(f"formal_rank {rank} carries screening_status="
                                    f"{row['screening_status']!r}")
        if row["error_category"] not in PRIMARY_CATEGORIES:
            raise S.FrozenViolation(f"formal_rank {rank} carries the non-primary category "
                                    f"{row['error_category']!r}")
        if not str(row["extracted_proof"]).strip():
            raise S.FrozenViolation(f"formal_rank {rank} has no extracted failed proof; Arms B, C "
                                    "and D would be unrenderable")
        if not str(row["diagnostic_text"]).strip():
            raise S.FrozenViolation(f"formal_rank {rank} has no attributable diagnostic; Arms C and "
                                    "D would be unrenderable")
        cohort.append(row)
    return cohort


def cohort_entries(cohort: list[dict]) -> list[dict]:
    """The per-theorem cohort record: everything the boundary validation re-checks."""
    return [
        {
            "formal_rank": row["formal_rank"], "screening_rank": row["screening_rank"],
            "statement_id": row["statement_id"], "component_id": row["component_id"],
            "name": row["name"], "source": row["source"], "tier": row["tier"],
            "classes": row["classes"], "error_category": row["error_category"],
            "screening_status": row["screening_status"], "screening_seed": row["seed"],
            "prompt_sha256": row["prompt_sha256"], "prompt_tokens": row["prompt_token_count"],
            "completion_sha256": row["completion_sha256"],
            "failed_proof_sha256": row["extracted_proof_sha256"],
            "failed_proof_tokens": row["extracted_proof_tokens"],
            "diagnostic_sha256": row["diagnostic_sha256"],
            "diagnostic_tokens": row["diagnostic_tokens"],
            "diagnostic_source": row["diagnostic_source"],
            "truncated_diagnostic": row["truncated_diagnostic"],
            "screening_verify_status": row["verify_status"],
        }
        for row in cohort
    ]


def build_second_stage_plan(*, cohort: list[dict], derangement: dict, surface: dict,
                            tokenizer) -> dict:
    """Render the four arms of all 128 theorems and prove the frozen invariants hold for each.

    Shared by `--stage freeze` and `--stage validate`: the validation re-runs this exact function, so
    a plan artifact that does not reproduce cannot pass. Everything here is a pure function of the
    frozen cohort, the frozen derangement, the pinned parquet and the frozen renderer.
    """
    by_statement = {row["statement_id"]: row for row in cohort}
    donors = {row["recipient"]: row["donor"] for row in derangement["mapping"]}
    if set(donors) != set(by_statement):
        raise S.FrozenViolation("the derangement does not cover exactly the cohort")
    theorems = []
    for row in cohort:
        rank = row["formal_rank"]
        sid = row["statement_id"]
        if sid not in surface:
            raise S.FrozenViolation(f"formal_rank {rank}: statement {sid} is not in the pinned "
                                    "promptset parquet")
        donor_sid = donors[sid]
        if donor_sid not in by_statement:
            raise S.FrozenViolation(f"formal_rank {rank}: donor {donor_sid} is not a cohort member")
        donor_row = by_statement[donor_sid]
        if donor_sid == sid:
            raise S.FrozenViolation(f"formal_rank {rank}: the donor diagnostic is the theorem's own")
        failed, own, donor = (str(row["extracted_proof"]), str(row["diagnostic_text"]),
                              str(donor_row["diagnostic_text"]))
        base = surface[sid]["messages"]
        invariants = P.diagnostic_arm_invariants(base, failed_proof=failed, own_diagnostic=own,
                                                 donor_diagnostic=donor)
        if not all(invariants.values()):
            raise S.FrozenViolation(f"formal_rank {rank}: the Arm C/D invariants do not hold: "
                                    f"{invariants}")
        arms: dict[str, dict] = {}
        for arm in S.ARM_ORDER:
            if arm == "A_FRESH_RETRY":
                diagnostic = ""
                text = P.render_arm(tokenizer, base, arm=arm)
            else:
                diagnostic = (own if arm == "C_VERIFIER_REPAIR" else
                              (donor if arm == "D_MISMATCHED_DIAGNOSTIC" else ""))
                text = P.render_arm(tokenizer, base, arm=arm, failed_proof=failed,
                                    diagnostic=diagnostic)
            tokens = len(tokenizer.encode(text, add_special_tokens=False))
            total = tokens + S.MAX_RESPONSE_TOKENS
            if total > S.MAX_MODEL_LEN:
                raise S.FrozenViolation(
                    f"formal_rank {rank} {arm}: {tokens} prompt tokens + {S.MAX_RESPONSE_TOKENS} "
                    f"response tokens exceeds the frozen window {S.MAX_MODEL_LEN}")
            arms[arm] = {
                "prompt_sha256": S.sha256_text(text), "prompt_tokens": tokens,
                "context_tokens_with_response": total,
                "diagnostic_sha256": (S.sha256_text(diagnostic) if diagnostic else ""),
                "diagnostic_source": (row["diagnostic_source"] if arm == "C_VERIFIER_REPAIR" else
                                      ("derangement_donor" if arm == "D_MISMATCHED_DIAGNOSTIC"
                                       else "none")),
            }
        if (arms["A_FRESH_RETRY"]["prompt_sha256"] != row["prompt_sha256"]
                or arms["A_FRESH_RETRY"]["prompt_tokens"] != row["prompt_token_count"]):
            raise S.FrozenViolation(
                f"formal_rank {rank}: the rendered Arm-A prompt is not the screening prompt of this "
                "theorem, so a fresh retry would not be the canonical first-attempt prompt")
        seed = second_stage_seed(rank)
        if seed == row["seed"]:
            raise S.FrozenViolation(f"formal_rank {rank}: the second-stage seed repeats the "
                                    "screening seed of the same theorem")
        theorems.append({
            "formal_rank": rank, "screening_rank": row["screening_rank"], "statement_id": sid,
            "error_category": row["error_category"], "seed": seed,
            "arm_order": list(S.arm_order(rank)),
            "failed_proof_sha256": row["extracted_proof_sha256"],
            "failed_proof_tokens": row["extracted_proof_tokens"],
            "own_diagnostic_sha256": row["diagnostic_sha256"],
            "donor_statement_id": donor_sid,
            "donor_diagnostic_sha256": donor_row["diagnostic_sha256"],
            "arm_d_invariants": {k: bool(v) for k, v in invariants.items()},
            "arms": arms,
        })
    if [t["formal_rank"] for t in theorems] != list(range(1, S.N_PRIMARY + 1)):
        raise S.FrozenViolation("the plan's formal ranks are not 1..128 in order")
    prompt_hashes = [{"formal_rank": t["formal_rank"], "arm_order": t["arm_order"],
                      "arms": {arm: t["arms"][arm]["prompt_sha256"] for arm in S.ARM_ORDER}}
                     for t in theorems]
    return {
        "theorems": theorems,
        "n_theorems": len(theorems),
        "n_candidates": len(theorems) * S.N_ARMS,
        "prompt_tokens_by_arm": {arm: _distribution([t["arms"][arm]["prompt_tokens"]
                                                     for t in theorems])
                                 for arm in S.ARM_ORDER},
        "context_tokens_with_response_by_arm": {
            arm: {"max": max(t["arms"][arm]["context_tokens_with_response"] for t in theorems),
                  "min": min(t["arms"][arm]["context_tokens_with_response"] for t in theorems)}
            for arm in S.ARM_ORDER},
        "max_model_len": S.MAX_MODEL_LEN,
        "max_response_tokens": S.MAX_RESPONSE_TOKENS,
        "arm_prompt_plan_sha256": S.sha(prompt_hashes),
        "plan_sha256": S.sha(theorems),
        # The frozen recipe of scripts/v4_p001_freeze_seeds.py, recomputed rather than trusted: a
        # bare list here would hash differently and the check below would compare a different object
        # against PAIRED_SEED_HASH while still "passing" its shape.
        "second_stage_seed_hash": S.sha({"n": S.N_PRIMARY, "seeds": [t["seed"] for t in theorems]}),
        "arm_schedule_hash": schedule_hash(),
        "balance": balance_report(),
        "derangement_mapping_sha256": derangement["mapping_sha256"],
        "donor_is_a_different_theorem_for_every_rank": all(
            t["donor_statement_id"] != t["statement_id"] for t in theorems),
        "n_byte_identical_donor_diagnostics": sum(
            1 for t in theorems if t["donor_diagnostic_sha256"] == t["own_diagnostic_sha256"]),
    }


def stage_freeze(args: argparse.Namespace) -> int:
    out_dir = out_dir_of(args)
    raw_path = out_dir / S.SCREENING_RAW_BASENAME
    cohort_path = out_dir / S.PRIMARY_COHORT_BASENAME
    derange_path = out_dir / S.DERANGEMENT_BASENAME
    plan_path = out_dir / S.SECOND_STAGE_PLAN_BASENAME

    print(f"[{S.EXPERIMENT_ID}] stage=freeze dry_run={args.dry_run} out={out_dir}")
    env = S.collect_env()
    try:
        S.check_environment(env, require_formal_host=False)
        frozen = S.load_frozen(verify_files=True)
        rows = read_screening(raw_path)
        cohort = cohort_rows(rows)
    except S.FrozenViolation as exc:
        return abort(str(exc))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    try:
        surface = load_surface(tokenizer)
        entries = cohort_entries(cohort)
        derangement = derange([
            {"statement_id": row["statement_id"], "category": row["error_category"],
             "diagnostic_sha256": row["diagnostic_sha256"],
             "diagnostic_tokens": row["diagnostic_tokens"]}
            for row in cohort
        ])
        plan = build_second_stage_plan(cohort=cohort, derangement=derangement, surface=surface,
                                       tokenizer=tokenizer)
    except S.FrozenViolation as exc:
        return abort(str(exc))

    screening_raw_sha256 = S.sha256_file(raw_path)
    last_cohort_screen = max(row["screening_rank"] for row in cohort)
    beyond = [{"screening_rank": row["screening_rank"], "statement_id": row["statement_id"],
               "screening_status": row["screening_status"], "error_category": row["error_category"]}
              for row in rows if row["screening_rank"] > last_cohort_screen]
    status_counts = {name: 0 for name in S.SCREENING_STATUSES}
    for row in rows:
        status_counts[row["screening_status"]] += 1
    frozen_references = {
        "prereg_commit": S.PREREG_COMMIT,
        "pool_sha256": frozen.pool["pool_hash"], "order_hash": frozen.pool["order_hash"],
        "screening_seed_hash": frozen.seeds["screening_seed_hash"],
        "paired_seed_hash": frozen.seeds["paired_seed_hash"],
        "arm_schedule_hash": frozen.schedule["schedule_hash"],
        "file_hashes": frozen.file_hashes, "registry_pins": frozen.registry_pins,
        "model": S.MODEL, "train_parquet": S.TRAIN_PARQUET,
    }
    created = datetime.now(timezone.utc).isoformat()
    common = {"experiment_id": S.EXPERIMENT_ID, "created_at_utc": created,
              "host": env["hostname"], "git_revision": env["git_revision"],
              "script_versions": S.script_version_hash(*SCRIPT_VERSION_FILES),
              "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
              "second_stage_candidates_generated": 0}

    cohort_payload = seal({
        **common,
        "artifact_type": "v4_p001_primary_cohort",
        "boundary": "stage1_screening -> stage2_freeze",
        "selection_rule": ("the first 128 theorems, in the frozen screening order, whose first "
                           "attempt was a PRIMARY_SEMANTIC_FAILURE with an attributable Lean "
                           "diagnostic and a fitting Arm-C context"),
        "screening_raw_path": str(raw_path), "screening_raw_sha256": screening_raw_sha256,
        "screening_rows": len(rows), "screening_max_screens": S.MAX_SCREENING,
        "screening_status_counts": status_counts,
        "n_primary_required": S.N_PRIMARY, "cohort_size": len(entries),
        "formal_ranks": [entry["formal_rank"] for entry in entries],
        "error_category_counts": category_counts([row["error_category"] for row in cohort]),
        "screened_after_the_cohort_closed": beyond,
        "members": entries,
        "members_canonical_sha256": S.sha(entries),
        "frozen_references": frozen_references,
    })
    der_payload = seal({
        **common,
        "artifact_type": "v4_p001_diagnostic_derangement",
        "cohort_content_sha256": cohort_payload["content_sha256"],
        "derangement": derangement,
        "mapping_sha256": derangement["mapping_sha256"],
        "version": DERANGEMENT_VERSION,
        "version_matches_the_frozen_derangement": derangement["version"] == DERANGEMENT_VERSION,
        "bucket_edges": list(BUCKET_EDGES),
        "owner_report": {
            "same_error_category_match_rate": derangement["same_error_category_match_rate"],
            "diagnostic_token_length_difference": derangement["diagnostic_token_length_difference"],
            "fallback_count": derangement["fallback_count"],
            "fallback_recipients": derangement["fallback_recipients"],
            "donor_reuse_count": derangement["donor_reuse_count"],
            "no_self_diagnostic": derangement["checks"]["no_self_diagnostic"],
            "every_recipient_assigned_exactly_once": derangement["checks"][
                "every_recipient_assigned_exactly_once"],
        },
        "deterministic": True,
        "frozen_before_any_second_stage_generation": True,
    })
    plan_payload = seal({
        **common,
        "artifact_type": "v4_p001_second_stage_plan",
        "cohort_content_sha256": cohort_payload["content_sha256"],
        "derangement_mapping_sha256": derangement["mapping_sha256"],
        "renderer_version": P.RENDERER_VERSION,
        "normalization_version": D.NORMALIZATION_VERSION,
        "model": S.MODEL,
        "common_random_numbers": ("all four arms of a theorem share second_stage_seed(formal_rank); "
                                  "the arms differ only in their prompt"),
        "theorems": plan["theorems"],
        "n_theorems": plan["n_theorems"], "n_candidates": plan["n_candidates"],
        "prompt_tokens_by_arm": plan["prompt_tokens_by_arm"],
        "context_tokens_with_response_by_arm": plan["context_tokens_with_response_by_arm"],
        "max_model_len": plan["max_model_len"], "max_response_tokens": plan["max_response_tokens"],
        "arm_prompt_plan_sha256": plan["arm_prompt_plan_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "second_stage_seed_hash": plan["second_stage_seed_hash"],
        "arm_schedule_hash": plan["arm_schedule_hash"],
        "balance": plan["balance"],
        "donor_is_a_different_theorem_for_every_rank": plan[
            "donor_is_a_different_theorem_for_every_rank"],
        "n_byte_identical_donor_diagnostics": plan["n_byte_identical_donor_diagnostics"],
    })

    print(f"[{S.EXPERIMENT_ID}] cohort: {len(entries)} primary semantic failures from "
          f"{len(rows)} screens ({len(beyond)} screened after the cohort closed)")
    print(f"[{S.EXPERIMENT_ID}] error categories: {category_counts([r['error_category'] for r in cohort])}")
    print(f"[{S.EXPERIMENT_ID}] derangement: version={derangement['version']} "
          f"mapping_sha256={derangement['mapping_sha256'][:16]}... "
          f"same_category_rate={derangement['same_error_category_match_rate']} "
          f"fallback_count={derangement['fallback_count']} "
          f"donor_reuse={derangement['donor_reuse_count']} "
          f"token_difference={derangement['diagnostic_token_length_difference']}")
    print(f"[{S.EXPERIMENT_ID}] plan: {plan['n_candidates']} candidates, prompt tokens by arm "
          f"{ {arm: plan['prompt_tokens_by_arm'][arm]['max'] for arm in S.ARM_ORDER} } (max), "
          f"arm_schedule_hash={plan['arm_schedule_hash'][:16]}... "
          f"balance={ {arm: plan['balance']['positions_per_arm'][arm] for arm in S.ARM_ORDER} }")
    if plan["n_byte_identical_donor_diagnostics"]:
        print(f"[note] {plan['n_byte_identical_donor_diagnostics']} theorem(s) receive a donor "
              "diagnostic whose bytes equal their own (different theorem, identical message); this is "
              "reported, not repaired: the frozen rule requires a different theorem, not different "
              "text.")
    if args.dry_run:
        print(f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE: nothing written. Would write "
              f"{cohort_path.name} ({cohort_payload['content_sha256'][:16]}...), "
              f"{derange_path.name} ({der_payload['content_sha256'][:16]}...), "
              f"{plan_path.name} ({plan_payload['content_sha256'][:16]}...) with "
              "second_stage_candidates_generated=0")
        return 0
    existing = [p.name for p in (cohort_path, derange_path, plan_path) if p.exists()]
    if existing and not args.resume:
        return abort(f"boundary artifact(s) {existing} already exist; pass --resume to overwrite "
                     "them with the recomputed boundary, or remove them deliberately")
    S.write_json_atomic(cohort_path, cohort_payload)
    S.write_json_atomic(derange_path, der_payload)
    S.write_json_atomic(plan_path, plan_payload)
    print(f"[{S.EXPERIMENT_ID}] wrote {cohort_path}, {derange_path}, {plan_path} "
          "(second_stage_candidates_generated = 0)")
    print(f"[{S.EXPERIMENT_ID}] STOP BEFORE SECOND-STAGE GENERATION by design (owner §12). "
          "Next: --stage validate.")
    return 0


# --- stage: validate (owner §12 deterministic boundary check) -------------------------------------

def validate_boundary(*, frozen: S.Frozen, tokenizer, surface: dict, out_dir: Path,
                      write_report: bool) -> tuple[dict, dict]:
    """Recompute the whole Stage-1/Stage-2 boundary from the raw artifact and the frozen design.

    Every check is a recomputation, not a comparison of recorded flags. Nothing here generates,
    verifies or decides anything: it either reproduces the boundary byte-for-byte or it fails.
    """
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    cohort_path = out_dir / S.PRIMARY_COHORT_BASENAME
    der_path = out_dir / S.DERANGEMENT_BASENAME
    plan_path = out_dir / S.SECOND_STAGE_PLAN_BASENAME
    raw_path = out_dir / S.SCREENING_RAW_BASENAME
    raw_second_path = out_dir / S.SECOND_STAGE_RAW_BASENAME
    paths = {"primary_cohort": cohort_path, "diagnostic_derangement": der_path,
             "second_stage_plan": plan_path}
    missing = sorted(name for name, path in paths.items() if not path.exists())
    check("boundary_artifacts_present", not missing, f"missing: {missing}")
    if missing:
        report = {"artifact_type": "v4_p001_boundary_validation", "status": "FAIL",
                  "experiment_id": S.EXPERIMENT_ID,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "host": S.collect_env()["hostname"],
                  "checks": checks, "n_checks": len(checks), "n_failed": len(checks),
                  "second_stage_candidates_generated": 0,
                  "note": ("the boundary artifacts do not exist yet: run --stage screening (formal, "
                           "owner-gated) and then --stage freeze before validating")}
        if write_report:
            S.write_json_atomic(out_dir / S.BOUNDARY_VALIDATION_BASENAME, report)
        return report, {}

    artifacts = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    cohort_artifact = artifacts["primary_cohort"]
    der_artifact = artifacts["diagnostic_derangement"]
    plan_artifact = artifacts["second_stage_plan"]
    for name, payload in artifacts.items():
        check(f"self_hash::{name}", unseal(payload) == payload.get("content_sha256"),
              f"recorded {str(payload.get('content_sha256'))[:16]}...")
    check("second_stage_candidates_generated_is_zero",
          all(int(payload.get("second_stage_candidates_generated", -1)) == 0
              for payload in artifacts.values()),
          "the boundary artifacts may only be written before any second-stage generation")
    check("boundary_artifact_type",
          (cohort_artifact["artifact_type"], der_artifact["artifact_type"],
           plan_artifact["artifact_type"]) ==
          ("v4_p001_primary_cohort", "v4_p001_diagnostic_derangement",
           "v4_p001_second_stage_plan"),
          "an unexpected artifact type at a boundary path")

    try:
        rows = read_screening(raw_path)
        cohort = cohort_rows(rows)
    except S.FrozenViolation as exc:
        check("screening_artifact_consumable", False, str(exc))
        rows, cohort = [], []
    else:
        check("screening_artifact_consumable", True,
              f"{len(rows)} screening rows, {len(cohort)} cohort members")
    check("screening_raw_hash_matches",
          raw_path.exists() and S.sha256_file(raw_path) == cohort_artifact["screening_raw_sha256"],
          f"on disk {S.sha256_file(raw_path)[:16] if raw_path.exists() else 'missing'}... vs recorded "
          f"{str(cohort_artifact['screening_raw_sha256'])[:16]}...")
    check("cohort_is_exactly_128", len(cohort) == S.N_PRIMARY, f"{len(cohort)} members")
    pool_statements = {member["statement_id"] for member in frozen.members}
    check("every_screened_theorem_is_a_pool_member",
          all(row["statement_id"] in pool_statements for row in rows),
          "a screening row names a theorem outside the frozen pool")

    if cohort:
        entries = cohort_entries(cohort)
        check("cohort_membership_matches_the_artifact",
              entries == cohort_artifact["members"]
              and S.sha(entries) == cohort_artifact["members_canonical_sha256"],
              "the artifact's member list is not the cohort the raw artifact selects")
        check("cohort_categories_are_primary",
              all(row["error_category"] in PRIMARY_CATEGORIES for row in cohort)
              and category_counts([row["error_category"] for row in cohort])
              == cohort_artifact["error_category_counts"],
              str(category_counts([row["error_category"] for row in cohort])))
        check("cohort_ranks_are_contiguous",
              [row["formal_rank"] for row in cohort] == list(range(1, S.N_PRIMARY + 1)),
              "formal ranks are not 1..128")
        try:
            derangement = derange([
                {"statement_id": row["statement_id"], "category": row["error_category"],
                 "diagnostic_sha256": row["diagnostic_sha256"],
                 "diagnostic_tokens": row["diagnostic_tokens"]}
                for row in cohort
            ])
        except (ValueError, AssertionError) as exc:
            derangement = {}
            check("derangement_recomputes", False, f"{type(exc).__name__}: {exc}")
        else:
            check("derangement_recomputes",
                  derangement["mapping_sha256"] == der_artifact["mapping_sha256"]
                  and derangement == der_artifact["derangement"],
                  f"recomputed mapping {derangement['mapping_sha256'][:16]}... vs recorded "
                  f"{str(der_artifact['mapping_sha256'])[:16]}...")
            check("derangement_no_self_diagnostic",
                  all(row["recipient"] != row["donor"] for row in derangement["mapping"])
                  and derangement["checks"]["no_self_diagnostic"] is True,
                  f"{sum(1 for row in derangement['mapping'] if row['recipient'] == row['donor'])} "
                  "self-diagnostic(s)")
            check("derangement_every_recipient_assigned_once",
                  len(derangement["mapping"]) == S.N_PRIMARY
                  and len({row["recipient"] for row in derangement["mapping"]}) == S.N_PRIMARY,
                  f"{len(derangement['mapping'])} mapping rows")

        if derangement:
            try:
                plan = build_second_stage_plan(cohort=cohort, derangement=derangement,
                                               surface=surface, tokenizer=tokenizer)
            except S.FrozenViolation as exc:
                plan = {}
                check("second_stage_plan_recomputes", False, str(exc))
            else:
                check("second_stage_plan_recomputes",
                      plan["theorems"] == plan_artifact["theorems"]
                      and plan["plan_sha256"] == plan_artifact["plan_sha256"]
                      and plan["arm_prompt_plan_sha256"] == plan_artifact["arm_prompt_plan_sha256"],
                      f"recomputed plan {plan['plan_sha256'][:16]}... vs recorded "
                      f"{str(plan_artifact['plan_sha256'])[:16]}...")
                screening_prompt_hashes = {row["statement_id"]: row["prompt_sha256"]
                                           for row in cohort}
                check("arm_a_is_the_screening_prompt",
                      all(t["arms"]["A_FRESH_RETRY"]["prompt_sha256"]
                          == screening_prompt_hashes[t["statement_id"]] for t in plan["theorems"]),
                      "an Arm-A prompt is not the theorem's screening prompt")
                check("arm_c_is_b_plus_the_own_diagnostic_and_d_is_c_with_the_donor",
                      all(all(t["arm_d_invariants"].values()) for t in plan["theorems"]),
                      "the Arm C/D invariants do not hold for every theorem")
                check("arm_d_uses_the_derangement_donor",
                      all(t["arms"]["D_MISMATCHED_DIAGNOSTIC"]["diagnostic_sha256"]
                          == t["donor_diagnostic_sha256"] for t in plan["theorems"]),
                      "Arm D does not carry the derangement-assigned donor diagnostic")
                worst_context = max(t["arms"][arm]["context_tokens_with_response"]
                                    for t in plan["theorems"] for arm in S.ARM_ORDER)
                check("context_fits_all_arms",
                      worst_context <= S.MAX_MODEL_LEN
                      and plan_artifact["max_model_len"] == S.MAX_MODEL_LEN
                      and plan_artifact["max_response_tokens"] == S.MAX_RESPONSE_TOKENS,
                      f"worst case {worst_context} tokens vs the frozen window {S.MAX_MODEL_LEN}")
                check("second_stage_seeds_are_the_frozen_stream",
                      [t["seed"] for t in plan["theorems"]]
                      == [second_stage_seed(rank) for rank in range(1, S.N_PRIMARY + 1)]
                      and plan["second_stage_seed_hash"] == S.PAIRED_SEED_HASH,
                      f"recomputed {plan['second_stage_seed_hash'][:16]}... vs frozen "
                      f"{S.PAIRED_SEED_HASH[:16]}...")
                check("second_stage_seeds_are_unique_and_disjoint_from_screening",
                      len({t["seed"] for t in plan["theorems"]}) == S.N_PRIMARY
                      and not ({t["seed"] for t in plan["theorems"]}
                               & {row["seed"] for row in rows}),
                      "a second-stage seed repeats another second-stage or a screening seed")
                expected_orders = [list(S.arm_order(rank)) for rank in range(1, S.N_PRIMARY + 1)]
                check("arm_order_is_the_frozen_schedule",
                      [t["arm_order"] for t in plan["theorems"]] == expected_orders
                      and plan["arm_schedule_hash"] == schedule_hash()
                      == frozen.schedule["schedule_hash"],
                      f"recomputed {schedule_hash()[:16]}... vs recorded "
                      f"{str(plan_artifact['arm_schedule_hash'])[:16]}...")
                check("arm_schedule_is_position_balanced",
                      balance_report() == plan_artifact["balance"]
                      and all(counts == [S.N_PRIMARY // S.N_ARMS] * S.N_ARMS
                              for counts in plan_artifact["balance"]["positions_per_arm"].values()),
                      str(plan_artifact["balance"]["positions_per_arm"]))
                check("plan_candidate_count",
                      plan["n_candidates"] == plan_artifact["n_candidates"]
                      == S.SECOND_STAGE_CANDIDATES
                      and plan["n_theorems"] == S.N_PRIMARY,
                      f"{plan['n_candidates']} candidates")

    check("hash_chain",
          der_artifact.get("cohort_content_sha256") == cohort_artifact.get("content_sha256")
          and plan_artifact.get("cohort_content_sha256") == cohort_artifact.get("content_sha256")
          and plan_artifact.get("derangement_mapping_sha256") == der_artifact.get("mapping_sha256"),
          "the boundary artifacts do not reference each other's content hashes")
    pool_checks = frozen.pool["checks"]
    check("sealed_reserve_untouched",
          all(entry["pass"] for entry in frozen.checks)
          and pool_checks.get("sealed_components_touched") == 0
          and pool_checks.get("sealed_statements_touched") == 0
          and pool_checks.get("v3_formal_sample_components_touched") == 0
          and pool_checks.get("v3_formal_sample_statements_touched") == 0
          and pool_checks.get("a_reserve_components_touched") == 0
          and pool_checks.get("b1_audit_components_touched") == 0
          and pool_checks.get("b1_reserved_ids_components_touched") == 0
          and pool_checks.get("b003_buffer_components_touched") == 0,
          f"pool checks {pool_checks}")
    check("frozen_design_checks_all_pass",
          all(entry["pass"] for entry in frozen.checks),
          str([entry["check"] for entry in frozen.checks if not entry["pass"]]))

    n_failed = sum(1 for entry in checks if not entry["pass"])
    env = S.collect_env()
    report = {
        "artifact_type": "v4_p001_boundary_validation",
        "schema_version": "v4-p001-boundary-1",
        "experiment_id": S.EXPERIMENT_ID,
        "boundary": "stage1_screening -> stage2_freeze -> stage2_generation",
        "purpose": ("the owner-required deterministic check that must pass before any second-stage "
                    "generation: cohort membership, error taxonomy, prompt fit, derangement, seed "
                    "set, arm schedule, sealed reserve"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": env["hostname"], "git_revision": env["git_revision"], "git_branch": env["branch"],
        "script_versions": S.script_version_hash(*SCRIPT_VERSION_FILES),
        "checks": checks, "n_checks": len(checks), "n_failed": n_failed,
        "status": "PASS" if n_failed == 0 else "FAIL",
        "artifacts": {name: {"path": str(path), "sha256": S.sha256_file(path),
                             "content_sha256": payload.get("content_sha256")}
                      for (name, path), payload in zip(paths.items(), artifacts.values())},
        "screening_raw": {"path": str(raw_path),
                          "sha256": S.sha256_file(raw_path) if raw_path.exists() else None,
                          "rows": len(rows)},
        "second_stage_raw": {"path": str(raw_second_path),
                             "present": raw_second_path.exists(),
                             "rows": sum(1 for _ in S.rows_jsonl(raw_second_path))
                             if raw_second_path.exists() else 0},
        "second_stage_candidates_generated": 0,
        "frozen_design_checks": frozen.checks,
    }
    if write_report:
        S.write_json_atomic(out_dir / S.BOUNDARY_VALIDATION_BASENAME, seal(report))
    return report, artifacts


def stage_validate(args: argparse.Namespace) -> int:
    out_dir = out_dir_of(args)
    print(f"[{S.EXPERIMENT_ID}] stage=validate dry_run={args.dry_run} out={out_dir}")
    try:
        S.check_environment(S.collect_env(), require_formal_host=False)
        frozen = S.load_frozen(verify_files=True)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    try:
        surface = load_surface(tokenizer)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    report, _artifacts = validate_boundary(frozen=frozen, tokenizer=tokenizer, surface=surface,
                                           out_dir=out_dir, write_report=not args.dry_run)
    failed = [entry for entry in report["checks"] if not entry["pass"]]
    for entry in report["checks"]:
        print(f"  [{'PASS' if entry['pass'] else 'FAIL'}] {entry['check']}"
              + (f" -- {entry['detail']}" if not entry["pass"] and entry["detail"] else ""))
    print(f"[{S.EXPERIMENT_ID}] boundary validation: {report['n_checks']} checks, "
          f"{report['n_failed']} failed -> {report['status']}")
    if failed:
        return abort("the Stage-1/Stage-2 boundary is not valid; the second stage must not run",
                     failed)
    print(f"[{S.EXPERIMENT_ID}] the boundary is frozen and valid. It does not authorize anything: "
          "the second stage needs the owner's authorization.")
    return 0


# --- stage: second (the paired A/B/C/D generation the owner §11 requires) -------------------------

def censored_repair_row(*, plan_row: dict, screening_row: dict, arm: str, position: int, env: dict,
                        run_id: str, reason: str) -> dict:
    """An arm candidate the verifier plan censored: never generated, never verified, never a failure."""
    arm_plan = plan_row["arms"][arm]
    return {
        "schema_version": S.REPAIR_SCHEMA_VERSION, "run_id": run_id,
        "experiment_id": S.EXPERIMENT_ID,
        "formal_rank": plan_row["formal_rank"], "screening_rank": plan_row["screening_rank"],
        "statement_id": plan_row["statement_id"], "component_id": screening_row["component_id"],
        "name": screening_row["name"], "source": screening_row["source"],
        "error_category": plan_row["error_category"], "arm": arm, "position": position,
        "seed": plan_row["seed"],
        "prompt_sha256": arm_plan["prompt_sha256"], "prompt_token_count": arm_plan["prompt_tokens"],
        "context_tokens_with_response": arm_plan["context_tokens_with_response"],
        "renderer_version": P.RENDERER_VERSION, "normalization_version": D.NORMALIZATION_VERSION,
        "diagnostic_sha256": arm_plan["diagnostic_sha256"],
        "diagnostic_source": arm_plan["diagnostic_source"],
        "failed_proof_sha256": plan_row["failed_proof_sha256"],
        "model_sha256": env["model_sha256"], "model_revision": env["model_revision"],
        "completion_text": "", "completion_sha256": S.sha256_text(""), "generated_tokens": 0,
        "truncated": False, "format_ok": False, "has_lean_block": False,
        "extracted_proof_present": False,
        "verified": False, "score": None, "acc": None,
        "verify_status": VerifyOutcome.UNRESOLVED_INFRA_ERROR.value, "lean_message": "",
        "error_category_second": None, "success": False,
        "censored": True, "censored_reason": reason,
        "generation_time": None, "verification_time": None,
        "host": env["hostname"],
        "gpu": env["gpu_names"].splitlines()[0] if env["gpu_names"] else "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def repair_one(tokenizer, session: VerificationSession, recovery: VerifierRecovery,
               recorder: ResultItemRecorder, *, plan_row: dict, screening_row: dict, arm: str,
               position: int, completion, generation_seconds: float, env: dict, run_id: str,
               ) -> tuple[dict, str | None, bool]:
    """One arm candidate: extract, verify (frozen policy + C′ recovery), classify, row."""
    text = completion.text
    try:
        extracted = extract_proof(text)
    except ValueError:
        extracted = ""
    lean_source = complete_verifier_code(screening_row["formal_statement"], extracted)
    has_lean_block = bool(_FENCED_LEAN.search(text))
    rank = plan_row["formal_rank"]
    custom_id = f"rank{rank}-{arm}"
    if lean_source is None:
        verify_status = S.FORMAT_ERROR
        message = ("no Lean proof body could be extracted from the completion; the candidate was "
                   "never sent to the verifier")
        abort_reason, exhausted = None, False
    else:
        classified, _diagnostic, abort_reason, exhausted = verify_with_recovery(
            session, recovery, lean_source, custom_id, rank=rank, recorder=recorder,
            tokenizer=tokenizer, want_diagnostic=False)
        verify_status, message = classified.outcome.value, classified.message
    score = S.candidate_score(verify_status)
    second_category = classify_first_attempt(
        verify_status=verify_status, lean_message=message if lean_source is not None else "",
        truncated=bool(completion.finish_reason == "length"), format_ok=lean_source is not None,
        has_lean_block=has_lean_block, extracted=extracted)
    arm_plan = plan_row["arms"][arm]
    row = {
        "schema_version": S.REPAIR_SCHEMA_VERSION, "run_id": run_id,
        "experiment_id": S.EXPERIMENT_ID,
        "formal_rank": rank, "screening_rank": plan_row["screening_rank"],
        "statement_id": plan_row["statement_id"], "component_id": screening_row["component_id"],
        "name": screening_row["name"], "source": screening_row["source"],
        "error_category": plan_row["error_category"], "arm": arm, "position": position,
        "seed": plan_row["seed"],
        "prompt_sha256": arm_plan["prompt_sha256"], "prompt_token_count": arm_plan["prompt_tokens"],
        "context_tokens_with_response": arm_plan["context_tokens_with_response"],
        "renderer_version": P.RENDERER_VERSION, "normalization_version": D.NORMALIZATION_VERSION,
        "diagnostic_sha256": arm_plan["diagnostic_sha256"],
        "diagnostic_source": arm_plan["diagnostic_source"],
        "failed_proof_sha256": plan_row["failed_proof_sha256"],
        "model_sha256": env["model_sha256"], "model_revision": env["model_revision"],
        "completion_text": text, "completion_sha256": S.sha256_text(text),
        "generated_tokens": len(list(completion.token_ids)),
        "truncated": bool(completion.finish_reason == "length"),
        "format_ok": lean_source is not None, "has_lean_block": has_lean_block,
        "extracted_proof_present": bool(extracted.strip()),
        "verified": score == 1, "score": score, "acc": score,
        "verify_status": verify_status, "lean_message": message[:500],
        "error_category_second": second_category, "success": score == 1,
        "censored": False, "censored_reason": "",
        "generation_time": round(generation_seconds, 4), "verification_time": None,
        "host": env["hostname"],
        "gpu": env["gpu_names"].splitlines()[0] if env["gpu_names"] else "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    S.validate_repair_row(row)
    return row, abort_reason, exhausted


def compact_resume_tail(raw_path: Path) -> tuple[dict, dict[int, list[dict]]]:
    """Drop a torn tail from an interrupted launch and return the complete groups.

    A group is written as one atomic append of four rows, so only the *final* run of rows can be
    short, and only after a crash mid-write. Nothing is rewritten: the file is truncated at the byte
    offset where the last complete group ends, and the dropped theorem is re-run from its frozen
    seed. Any other anomaly (a duplicate group, a short non-final group) makes the artifact
    non-consumable and the caller must stop.
    """
    if not raw_path.exists():
        return {"present": False}, {}
    data = raw_path.read_bytes()
    runs: list[tuple[int, list[dict]]] = []
    end_offsets: list[int] = []
    offset = 0
    for line in data.split(b"\n"):
        offset += len(line) + 1
        if not line.strip():
            continue
        row = json.loads(line.decode("utf-8"))
        S.validate_repair_row(row)
        rank = row["formal_rank"]
        if runs and runs[-1][0] == rank:
            runs[-1][1].append(row)
        else:
            runs.append((rank, [row]))
        end_offsets.append(offset)
    counts: dict[int, int] = {}
    for rank, _rows in runs:
        counts[rank] = counts.get(rank, 0) + 1
    duplicated = sorted(rank for rank, count in counts.items() if count > 1)
    if duplicated:
        raise S.FrozenViolation(f"{raw_path} holds more than one group for formal_rank "
                                f"{duplicated[:5]}; the artifact is not consumable")
    short = [(rank, len(rows)) for rank, rows in runs[:-1] if len(rows) != S.N_ARMS]
    if short:
        raise S.FrozenViolation(f"{raw_path} holds a non-final group of {short[0][1]} rows at "
                                f"formal_rank {short[0][0]}; the artifact is not consumable")
    dropped: list[dict] = []
    if runs and len(runs[-1][1]) != S.N_ARMS:
        dropped = [{"formal_rank": row["formal_rank"], "arm": row["arm"]} for row in runs[-1][1]]
        runs = runs[:-1]
    n_kept_rows = sum(len(rows) for _rank, rows in runs)
    kept_bytes = end_offsets[n_kept_rows - 1] if n_kept_rows else 0
    if kept_bytes < len(data):
        with raw_path.open("r+b") as fh:
            fh.truncate(kept_bytes)
            fh.flush()
            os.fsync(fh.fileno())
    return ({"present": True, "rows_in_file": n_kept_rows + len(dropped), "rows_kept": n_kept_rows,
             "rows_dropped_torn_tail": len(dropped), "dropped_rows": dropped,
             "bytes_truncated": len(data) - kept_bytes, "groups": len(runs)},
            {rank: rows for rank, rows in runs})


def stage_second(args: argparse.Namespace) -> int:
    out_dir = out_dir_of(args)
    raw_path = out_dir / S.SECOND_STAGE_RAW_BASENAME
    summary_path = out_dir / S.SUMMARY_BASENAME

    print(f"[{S.EXPERIMENT_ID}] stage=second dry_run={args.dry_run} resume={args.resume} "
          f"out={raw_path}")
    env = S.collect_env()
    try:
        S.check_environment(env, require_formal_host=not args.dry_run)
        frozen = S.load_frozen(verify_files=True)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] host={env['hostname']} git={env['git_revision'][:9]} "
          f"branch={env['branch']} | {len(frozen.checks)} frozen-artifact checks passed")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / S.MODEL["dir"], trust_remote_code=True,
                                             local_files_only=True)
    try:
        surface = load_surface(tokenizer)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    report, artifacts = validate_boundary(frozen=frozen, tokenizer=tokenizer, surface=surface,
                                          out_dir=out_dir, write_report=False)
    failed = [entry for entry in report["checks"] if not entry["pass"]]
    if failed:
        return abort("the Stage-1/Stage-2 boundary did not validate; no second-stage candidate may "
                     "be generated (owner §12)", failed)
    print(f"[{S.EXPERIMENT_ID}] boundary validation: {report['n_checks']} checks PASS "
          f"(cohort, derangement, prompts, seeds, arm schedule, sealed reserve)")
    plan_artifact = artifacts["second_stage_plan"]
    screening_rows = read_screening(out_dir / S.SCREENING_RAW_BASENAME)
    screening_by_statement = {row["statement_id"]: row for row in screening_rows}
    plan_by_rank = {t["formal_rank"]: t for t in plan_artifact["theorems"]}
    for rank, theorem in plan_by_rank.items():
        row = screening_by_statement[theorem["statement_id"]]
        if row["extracted_proof_sha256"] != theorem["failed_proof_sha256"]:
            return abort(f"formal_rank {rank}: the screening artifact's failed proof no longer "
                         "matches the frozen plan")
    texts: dict[int, dict[str, str]] = {}
    for rank, theorem in plan_by_rank.items():
        row = screening_by_statement[theorem["statement_id"]]
        base = surface[theorem["statement_id"]]["messages"]
        failed = str(row["extracted_proof"])
        own = str(row["diagnostic_text"])
        donor = str(screening_by_statement[theorem["donor_statement_id"]]["diagnostic_text"])
        rendered = {}
        for arm in S.ARM_ORDER:
            if arm == "A_FRESH_RETRY":
                text = P.render_arm(tokenizer, base, arm=arm)
            else:
                diagnostic = (own if arm == "C_VERIFIER_REPAIR" else
                              (donor if arm == "D_MISMATCHED_DIAGNOSTIC" else ""))
                text = P.render_arm(tokenizer, base, arm=arm, failed_proof=failed,
                                    diagnostic=diagnostic)
            if S.sha256_text(text) != theorem["arms"][arm]["prompt_sha256"]:
                return abort(f"formal_rank {rank} {arm}: the rendered prompt no longer matches the "
                             "frozen plan hash; the prompt surface moved after the boundary was "
                             "frozen")
            rendered[arm] = text
        texts[rank] = rendered
    print(f"[{S.EXPERIMENT_ID}] all {len(plan_by_rank) * S.N_ARMS} prompts re-rendered and matched "
          "against the frozen plan hashes")

    # resume: complete groups are kept, a torn tail is dropped (never rewritten), and the dropped
    # theorem is re-run from its own frozen seed. A dry run reads the artifact and writes nothing.
    complete: dict[int, list[dict]] = {}
    compaction: dict = {"present": raw_path.exists()}
    if raw_path.exists():
        if not args.resume:
            return abort(f"{raw_path} already exists; refusing to append to a raw artifact without "
                         "--resume")
        if args.dry_run:
            compaction.update({"rows_in_file": sum(1 for _ in S.rows_jsonl(raw_path)),
                               "note": "dry run: the raw artifact is read only, no tail is dropped"})
        else:
            try:
                compaction, complete = compact_resume_tail(raw_path)
            except S.FrozenViolation as exc:
                return abort(f"existing second-stage artifact is not consumable: {exc}")
    todo = [rank for rank in range(1, S.N_PRIMARY + 1) if rank not in complete]
    print(f"[{S.EXPERIMENT_ID}] resume: {len(complete)} complete group(s) on disk, "
          f"{len(todo)} theorem(s) to run; compaction={json.dumps(compaction)}")

    stamp: dict = {"experiment_id": S.EXPERIMENT_ID, "stage": "second",
                   "schema_version": S.REPAIR_SCHEMA_VERSION,
                   "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                   "frozen_settings": S.FROZEN_SETTINGS,
                   "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
                   "script_versions": S.script_version_hash(*SCRIPT_VERSION_FILES),
                   "prereg_commit": S.PREREG_COMMIT, "git": env,
                   "boundary_validation": {"status": report["status"], "n_checks": report["n_checks"],
                                           "n_failed": report["n_failed"],
                                           "artifacts": report["artifacts"],
                                           "screening_raw": report["screening_raw"]},
                   "plan_sha256": plan_artifact["plan_sha256"],
                   "arm_prompt_plan_sha256": plan_artifact["arm_prompt_plan_sha256"],
                   "arm_schedule_hash": plan_artifact["arm_schedule_hash"],
                   "second_stage_seed_hash": plan_artifact["second_stage_seed_hash"],
                   "derangement_mapping_sha256": plan_artifact["derangement_mapping_sha256"],
                   "n_theorems": S.N_PRIMARY, "n_candidates": S.SECOND_STAGE_CANDIDATES,
                   "resume_compaction": compaction,
                   "mode": "FORMAL" if not args.dry_run else "DRY_RUN"}

    if args.dry_run:
        stamp.update({"candidates_generated": 0, "raw_artifact_written": False,
                      "theorems_to_run": todo})
        print(json.dumps({k: stamp[k] for k in ("mode", "run_id", "n_candidates", "plan_sha256",
                                                "arm_schedule_hash", "second_stage_seed_hash",
                                                "theorems_to_run")}, indent=2))
        print(f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE: llm.load_called=false generate_called=false "
              "model.generate on formal theorems=0 candidates_generated=0 verifier_formal_calls=0 "
              "formal_raw_result_files_created=false")
        return 0

    recovery = VerifierRecovery(out_dir)
    try:
        instance = recovery.verify_identity()
    except RecoveryError as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] dedicated verifier instance: {instance.get('container_id')} "
          f"{instance.get('image')} published={instance.get('published')} env={instance.get('env')}")
    session = make_session()
    health = check_verifier(session)
    print(f"[{S.EXPERIMENT_ID}] verifier health: {health.get('health_status')} "
          f"canary={health.get('canary', {}).get('status')} "
          f"({health.get('canary', {}).get('seconds')}s)")
    if not health.get("canary", {}).get("verified"):
        return abort(f"verifier is not healthy (canary did not verify): {health}")
    verifier_events_path = out_dir / S.VERIFIER_EVENTS_BASENAME
    events_cursor = flush_session_events(session, verifier_events_path, 0)

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"[{S.EXPERIMENT_ID}] loading theta0 into vLLM ({S.MODEL['dir']})")
    llm = LLM(model=str(ROOT / S.MODEL["dir"]), tokenizer=str(ROOT / S.MODEL["dir"]),
              trust_remote_code=True, max_model_len=S.MAX_MODEL_LEN,
              gpu_memory_utilization=S.GPU_MEMORY_UTILIZATION, max_num_seqs=S.MAX_NUM_SEQS,
              max_num_batched_tokens=S.MAX_NUM_BATCHED_TOKENS, disable_log_stats=True)
    env["model_sha256"] = S.MODEL["weights_sha256"]
    env["model_revision"] = S.MODEL["revision"]
    torch.cuda.reset_peak_memory_stats()

    generated = 0
    censored_rows = 0
    theorems_this_launch: list[dict] = []
    abort_reason: str | None = None
    t0 = time.perf_counter()
    with ResultItemRecorder() as recorder:
        chunks = [todo[i:i + SECOND_STAGE_CHUNK] for i in range(0, len(todo), SECOND_STAGE_CHUNK)]
        for chunk in tqdm(chunks, desc="V4-P001 second stage", unit="chunk", dynamic_ncols=True):
            buffer: dict[int, dict[int, dict]] = {rank: {} for rank in chunk}
            exhausted: set[int] = set()
            for position in range(1, S.N_ARMS + 1):
                batch = [(rank, plan_by_rank[rank]["arm_order"][position - 1])
                         for rank in chunk
                         if rank not in exhausted and position not in buffer[rank]]
                if not batch:
                    continue
                prompts = [texts[rank][arm] for rank, arm in batch]
                params = [SamplingParams(temperature=S.TEMPERATURE, top_p=S.TOP_P,
                                         max_tokens=S.MAX_RESPONSE_TOKENS, n=1,
                                         seed=plan_by_rank[rank]["seed"])
                          for rank, _arm in batch]
                gen_t = time.perf_counter()
                outputs = llm.generate(prompts, params)
                gen_seconds = (time.perf_counter() - gen_t) / max(1, len(prompts))
                for (rank, arm), output in zip(batch, outputs):
                    row, abort_reason, theorem_exhausted = repair_one(
                        tokenizer, session, recovery, recorder,
                        plan_row=plan_by_rank[rank],
                        screening_row=screening_by_statement[plan_by_rank[rank]["statement_id"]],
                        arm=arm, position=position, completion=output.outputs[0],
                        generation_seconds=gen_seconds, env=env, run_id=stamp["run_id"])
                    buffer[rank][position] = row
                    generated += 1
                    if abort_reason:
                        break
                    if theorem_exhausted:
                        exhausted.add(rank)
                        reason = (f"theorem recovery exposure exhausted: {recovery.max_per_theorem} "
                                  f"of {recovery.max_per_theorem} restarts used by formal_rank {rank}; "
                                  "the candidate was neither generated nor verified "
                                  "(missing data, never a failure)")
                        for later in range(position + 1, S.N_ARMS + 1):
                            buffer[rank][later] = censored_repair_row(
                                plan_row=plan_by_rank[rank],
                                screening_row=screening_by_statement[
                                    plan_by_rank[rank]["statement_id"]],
                                arm=plan_by_rank[rank]["arm_order"][later - 1], position=later,
                                env=env, run_id=stamp["run_id"], reason=reason)
                            censored_rows += 1
                    events_cursor = flush_session_events(session, verifier_events_path,
                                                         events_cursor)
                if abort_reason:
                    break
            if abort_reason:
                break
            for rank in chunk:
                rows = [buffer[rank][position] for position in sorted(buffer[rank])]
                if len(rows) != S.N_ARMS:
                    abort_reason = (f"internal error: formal_rank {rank} produced {len(rows)} rows; "
                                    "the group was not written and will be re-run on resume")
                    break
                S.append_rows_durable(raw_path, rows)
                theorems_this_launch.append({
                    "formal_rank": rank, "statement_id": plan_by_rank[rank]["statement_id"],
                    "seed": plan_by_rank[rank]["seed"], "arm_order": plan_by_rank[rank]["arm_order"],
                    "success_by_arm": {plan_by_rank[rank]["arm_order"][p - 1]: rows[p - 1]["success"]
                                       for p in range(1, S.N_ARMS + 1)},
                    "censored_by_arm": {plan_by_rank[rank]["arm_order"][p - 1]: rows[p - 1]["censored"]
                                        for p in range(1, S.N_ARMS + 1)},
                })
            events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
            if abort_reason:
                break

    events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
    with raw_path.open(encoding="utf-8") as fh:
        rows_on_disk = sum(1 for line in fh if line.strip())
    status_counts: dict[str, int] = {}
    arm_counts: dict[str, int] = {arm: 0 for arm in S.ARM_ORDER}
    for row in S.rows_jsonl(raw_path):
        status_counts[row["verify_status"]] = status_counts.get(row["verify_status"], 0) + 1
        arm_counts[row["arm"]] += 1
    stamp.update({
        "candidates_generated": generated,
        "candidates_censored": censored_rows,
        "rows_on_disk": rows_on_disk,
        "theorems_complete_on_disk": rows_on_disk // S.N_ARMS,
        "status_counts_on_disk": status_counts, "arm_counts_on_disk": arm_counts,
        "theorems_this_launch": theorems_this_launch,
        "generation_seconds": round(time.perf_counter() - t0, 1),
        "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "vllm_version": vllm.__version__,
        "raw_artifact_sha256": S.sha256_file(raw_path) if raw_path.exists() else None,
        "verifier_health": health,
        "verifier_infrastructure": verifier_infrastructure_stamp(session, recovery, instance,
                                                                verifier_events_path, True),
        "per_theorem_recovery_exposure_is_per_launch": True,
        "no_training_no_controller_update": True,
    })
    if abort_reason:
        stamp["aborted"] = abort_reason
        stamp["resume_command"] = (".venv/bin/python scripts/v4_p001_rollout.py --stage second "
                                   "--resume --i-have-owner-launch-authorization")
        S.write_json_atomic(summary_path.with_name(
            S.SUMMARY_BASENAME.replace(".json", ".aborted.json")), stamp)
        return abort(f"{abort_reason}. {generated} second-stage candidate(s) and "
                     f"{rows_on_disk} row(s) were kept; resume with --stage second --resume once the "
                     "dedicated verifier instance is healthy.")
    S.write_json_atomic(summary_path, stamp)
    print(json.dumps({"candidates_generated": generated, "candidates_censored": censored_rows,
                      "rows_on_disk": rows_on_disk,
                      "theorems_complete": stamp["theorems_complete_on_disk"],
                      "status_counts": status_counts, "arm_counts": arm_counts,
                      "recoveries": recovery.n_attempted,
                      "raw": str(raw_path), "summary": str(summary_path)}, indent=2))
    print(f"[{S.EXPERIMENT_ID}] SECOND STAGE COMPLETE: {rows_on_disk // S.N_ARMS} of "
          f"{S.N_PRIMARY} theorem quadruplets on disk. The analysis is a separate, deterministic "
          "step: --stage second only produces raw candidates.")
    return 0


# --- command line ---------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4-P001 formal runner: frozen screening, the Stage-1/Stage-2 boundary, and the "
                    "paired four-arm second stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=STAGES, required=True,
                        help="screening (formal generation) | freeze (CPU boundary artifacts) | "
                             "validate (deterministic boundary check) | second (formal paired "
                             "generation)")
    parser.add_argument("--out-dir", default="runs/v4_p001/rollout",
                        help="artifact directory (gitignored; only hashes are committed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="preflight only: no generation, no raw artifact, no writes for freeze / "
                             "validate")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted stage from its own artifacts")
    parser.add_argument("--i-have-owner-launch-authorization", dest="owner_launch_authorization",
                        action="store_true",
                        help="required for a formal stage: the owner has explicitly authorized this "
                             "exact formal launch")
    return parser.parse_args(argv)


def launch_interlock(args: argparse.Namespace) -> str | None:
    """A formal stage may not start without an explicit owner launch authorization on the command."""
    if args.dry_run or args.stage not in FORMAL_STAGES:
        return None
    if args.owner_launch_authorization:
        return None
    return (f"--stage {args.stage} generates formal candidates and is gated: pass "
            "--i-have-owner-launch-authorization only if the owner has authorized this exact V4-P001 "
            "formal launch. The execution package ends with V4_FORMAL_SCREENING_AUTHORIZED: NO and "
            "V4_SECOND_STAGE_AUTHORIZED: NO, so no authorization exists yet.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    deny = launch_interlock(args)
    if deny:
        return abort(deny)
    if args.stage == "screening":
        return stage_screening(args)
    if args.stage == "freeze":
        return stage_freeze(args)
    if args.stage == "validate":
        return stage_validate(args)
    return stage_second(args)


if __name__ == "__main__":
    raise SystemExit(main())
