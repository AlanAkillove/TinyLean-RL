#!/usr/bin/env python
"""V3-R001 infrastructure amendment C′ -- sections 8 and 9: the nonformal lifecycle stress suite.

What this proves, before any attempt-2 generation exists: a candidate that occupies the dedicated
verifier's only REPL past the frozen server-side timeout is (a) classified by the frozen B0 policy,
(b) followed by the mandated recovery sequence (restart -> /health -> nonformal canary) before the
next candidate, and (c) never leaks a REPL slot -- three times in a row, then again in the section 9
mixed normal/pathological sequence.

Inputs are nonformal by construction and outside the V3-R001 formal 128 and the sealed reserve 93
(owner section 8): the pathological proof is the ``interval_cases`` enumeration bomb -- the class
E024's chunk-7 candidates were found to carry (``docs/v2/b0_verifier_reliability.md`` sections 2-4,
where the same bomb was measured to time out at exactly the server budget) -- and the normal proof is
the frozen policy's own canary. No V3 theorem statement, no formal-sample text, no reserve text and
no attempt-1 candidate is read, sent or touched; nothing here generates text with a model.

The suite drives the *production* C′ code paths -- ``v3_r001_rollout.make_session()``,
``VerifierRecovery`` and ``verify_candidate_with_recovery()`` -- so what is measured is what attempt-2
will run.

The suite writes ``runs/v3_r001/stress_cprime/`` with a per-cycle event log and a compact
``V3-R001_Cprime_validation.json`` artifact (owner section 13: candidate timeout wall-clock, recovery
wall-clock, canary latency, container restart count, REPL availability, 3-cycle result, mixed
sequence result).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import v3_r001_rollout as R
import v3_r001_spec as S

from tinylean_rl.verifier.kimina import verify_code
from tinylean_rl.verifier.policy import (
    CANARY_PROOF,
    VerifyOutcome,
    classify_result_item,
    classify_transport_error,
)

#: The section 8 pathological input. The E024 chunk-7 family (`interval_cases` enumeration bombs);
#: anonymous `example`, so a reused REPL can never turn it into a redeclaration retry.
BOMB_PROOF = (
    "import Mathlib\n"
    "set_option maxHeartbeats 0\n"
    "example (n : ℕ) (h : n < 500000) : n * 0 = 0 := by\n"
    "  interval_cases n <;> rfl\n"
)

#: The section 9 normal inputs: the frozen canary, plus one second trivial proof.
NORMAL_PROOFS = {
    "canary": CANARY_PROOF,
    "trivial": "import Mathlib\nexample : (List.length [1, 2, 3]) = 3 := by rfl\n",
}

CYCLES = 3
DEFAULT_OUT = ROOT / "runs/v3_r001/stress_cprime"
VALIDATION_BASENAME = "V3-R001_Cprime_validation.json"


def _repl_count(container: str) -> dict:
    """REPL processes docker can see in the dedicated instance (the T2 availability measurement)."""
    try:
        proc = subprocess.run(["docker", "top", container, "-eo", "pid,comm"],
                              capture_output=True, text=True, timeout=30, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout).strip()[:200]}
    lines = [ln.split() for ln in proc.stdout.strip().splitlines()[1:] if ln.strip()]
    return {"repl_processes": sum(1 for parts in lines if len(parts) > 1 and parts[1] == "repl"),
            "lake_processes": sum(1 for parts in lines if len(parts) > 1 and parts[1] == "lake"),
            "total": len(lines)}


def run_confirmed_cycle(session, recovery, out_dir: Path, cycle: int) -> dict:
    """Section 8 cycle: the server itself confirms the timeout at the frozen budget."""
    container = recovery.container
    event: dict = {"phase": "confirmed-timeout", "cycle": cycle,
                   "started_at": datetime.now(timezone.utc).isoformat()}
    event["repl_counts"] = {"before": _repl_count(container)}
    cursor = len(session.events)
    started = time.perf_counter()
    classified, abort_reason = R.verify_candidate_with_recovery(
        session, recovery, BOMB_PROOF, f"stress-confirmed-{cycle}", rank=-1, sample_index=cycle)
    event["candidate"] = {
        "kind": "interval_cases bomb (nonformal, E024 chunk-7 family)",
        "proof_sha256": S.sha256_text(BOMB_PROOF),
        "wall_s": round(time.perf_counter() - started, 2),
        "outcome": classified.outcome.value,
        "conclusive": classified.outcome.is_conclusive,
        "message": classified.message[:300],
        "abort_reason": abort_reason,
    }
    event["repl_counts"]["after_candidate"] = _repl_count(container)
    event["session_events"] = [{"at": e.at, "kind": e.kind, "detail": e.detail[:300]}
                               for e in session.events[cursor:]]
    R.flush_session_events(session, out_dir / "v3_r001_stress_verifier_events.jsonl", cursor)
    event["recovery"] = recovery.events[-1] if recovery.events else None
    event["repl_counts"]["after_recovery"] = _repl_count(container)
    event["ok"] = bool(classified.outcome is VerifyOutcome.VERIFIER_TIMEOUT
                       and abort_reason is None and event["recovery"] and event["recovery"]["ok"])
    return event


def run_zombie_cycle(recovery, out_dir: Path, cycle: int) -> dict:
    """The attempt-1 Case B shape: the client abandons at 2 s while the server keeps elaborating.

    ``/verify`` has no disconnect handling, so the running computation is a server-side zombie until
    the frozen server-side timeout -- except that C′'s mandated recovery restarts the instance first.
    This is the case attempt-1 could only detect (canary -> fail-close) and could not repair.
    """
    container = recovery.container
    event: dict = {"phase": "client-abandonment (attempt-1 Case B)", "cycle": cycle,
                   "started_at": datetime.now(timezone.utc).isoformat()}
    event["repl_counts"] = {"before": _repl_count(container)}
    started = time.perf_counter()
    try:
        decoded = verify_code(BOMB_PROOF, custom_id=f"stress-zombie-{cycle}",
                              base_url=recovery.endpoint, timeout=2.0,
                              server_timeout=int(S.VERIFIER["server_timeout_s"]))
        items = decoded.get("results") or decoded.get("codes") or []
        classified = classify_result_item(items[0] if items else None)
    except httpx.HTTPError as exc:
        classified = classify_transport_error(exc)
    event["candidate"] = {
        "kind": "interval_cases bomb, client timeout 2 s < server timeout",
        "proof_sha256": S.sha256_text(BOMB_PROOF),
        "wall_s": round(time.perf_counter() - started, 2),
        "outcome": classified.outcome.value,
        "conclusive": classified.outcome.is_conclusive,
        "message": classified.message[:300],
    }
    event["repl_counts"]["after_abandonment"] = _repl_count(container)
    event["recovery"] = recovery.recover({"phase": "client-abandonment", "cycle": cycle,
                                          "outcome": classified.outcome.value,
                                          "message": classified.message[:300]})
    event["repl_counts"]["after_recovery"] = _repl_count(container)
    event["ok"] = bool(event["recovery"]["ok"] and not classified.outcome.is_conclusive
                       and event["candidate"]["wall_s"] <= 30.0)
    return event


def run_mixed_sequence(session, recovery, out_dir: Path, start_cycle: int) -> list:
    """Section 9: normal / pathological / normal / pathological / normal, production path only."""
    plan = [("normal", "canary"), ("pathological", None), ("normal", "trivial"),
            ("pathological", None), ("normal", "canary")]
    events = []
    for step, (kind, which) in enumerate(plan, start=1):
        container = recovery.container
        event: dict = {"phase": "mixed-sequence", "step": step, "kind": kind,
                       "started_at": datetime.now(timezone.utc).isoformat()}
        event["repl_counts"] = {"before": _repl_count(container)}
        recoveries_before = recovery.n_attempted
        cursor = len(session.events)
        started = time.perf_counter()
        if kind == "normal":
            proof = NORMAL_PROOFS[which]
            classified, abort_reason = R.verify_candidate_with_recovery(
                session, recovery, proof, f"stress-mixed-{step}-{which}",
                rank=-1, sample_index=100 + step)
            expected = classified.outcome is VerifyOutcome.VERIFIED and abort_reason is None
            which_label = which
        else:
            classified, abort_reason = R.verify_candidate_with_recovery(
                session, recovery, BOMB_PROOF, f"stress-mixed-{step}-bomb",
                rank=-1, sample_index=200 + step)
            expected = (classified.outcome is VerifyOutcome.VERIFIER_TIMEOUT
                        and abort_reason is None)
            which_label = "interval_cases bomb"
        event["which"] = which_label
        event["candidate"] = {
            "wall_s": round(time.perf_counter() - started, 2),
            "outcome": classified.outcome.value,
            "conclusive": classified.outcome.is_conclusive,
            "message": classified.message[:300],
            "abort_reason": abort_reason,
        }
        event["recovery_attempted"] = recovery.n_attempted > recoveries_before
        event["session_events"] = [{"at": e.at, "kind": e.kind, "detail": e.detail[:300]}
                                   for e in session.events[cursor:]]
        R.flush_session_events(session, out_dir / "v3_r001_stress_verifier_events.jsonl", cursor)
        event["repl_counts"]["after_step"] = _repl_count(container)
        event["ok"] = bool(expected and event["recovery_attempted"] == (kind == "pathological"))
        events.append(event)
    return events


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--validate-only", action="store_true",
                        help="print the frozen-settings identity checks and exit (no Lean contact)")
    return parser.parse_args(argv)


def frozen_settings_check() -> dict:
    """T5: the suite runs the frozen values, and none of them is redefined here."""
    session = R.make_session()
    v = S.VERIFIER
    return {
        "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
        "server_timeout_s": session.server_timeout,
        "server_timeout_is_frozen": session.server_timeout == v["server_timeout_s"],
        "client_timeout_s": session.client_timeout,
        "client_timeout_is_frozen": session.client_timeout == v["server_timeout_s"] + v["client_slack_s"],
        "batch_size": session.batch_size,
        "batch_size_is_one": session.batch_size == v["batch_size"] == 1,
        "max_single_retries": session.max_single_retries,
        "max_single_retries_is_frozen": session.max_single_retries == v["max_single_retries"],
        "endpoint": session.base_url,
        "endpoint_is_dedicated": session.base_url == S.VERIFIER_INFRA["endpoint"],
        "max_recoveries_per_run": S.VERIFIER_INFRA["max_recoveries_per_run"],
    }


def main(argv: list | None = None) -> int:
    args = parse_args(argv)
    check = frozen_settings_check()
    print(json.dumps(check, indent=2))
    identity_keys = (k for k in check if k.endswith(("_is_frozen", "_is_one", "_is_dedicated")))
    if not all(check[k] for k in identity_keys):
        print("[stress] T5 FAILED: a frozen setting is not what the suite believes it is")
        return 3
    if args.validate_only:
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session = R.make_session()
    recovery = R.VerifierRecovery(out_dir)

    stamp: dict = {
        "experiment_id": S.EXPERIMENT_ID, "attempt": S.ATTEMPT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": ("amendment C′ sections 8-9 nonformal verifier-lifecycle stress; no formal "
                    "theorem, no reserve theorem, no model generation"),
        "frozen_settings": check,
        "inputs": {
            "pathological": {"kind": "interval_cases enumeration bomb (E024 chunk-7 family)",
                             "proof_sha256": S.sha256_text(BOMB_PROOF), "proof": BOMB_PROOF},
            "normal": {name: {"proof_sha256": S.sha256_text(p), "proof": p}
                       for name, p in NORMAL_PROOFS.items()},
        },
        "protocol": {"confirmed_timeout_cycles": CYCLES, "client_abandonment_cycles": CYCLES,
                     "mixed_sequence": ["normal", "pathological", "normal", "pathological", "normal"]},
    }

    print(f"[stress] preflight: identity + health + canary on {recovery.endpoint}")
    instance = recovery.verify_identity()
    stamp["dedicated_instance"] = instance
    health = recovery.wait_healthy()
    if not health["ok"]:
        stamp["result"] = {"ok": False, "reason": f"dedicated instance not healthy: {health}"}
        S.write_json_atomic(out_dir / VALIDATION_BASENAME, stamp)
        return 3
    stamp["preflight_health"] = health
    preflight_canary = recovery.cold_canary("preflight")
    stamp["preflight_canary"] = preflight_canary
    # The preflight canary is a probe, not a recovery: keep the recovery budget clean.
    if recovery.events:
        recovery.events.clear()
    if not preflight_canary["verified"]:
        stamp["result"] = {"ok": False, "reason": f"preflight canary did not verify: {preflight_canary}"}
        S.write_json_atomic(out_dir / VALIDATION_BASENAME, stamp)
        return 3
    stamp["repl_counts_after_preflight"] = _repl_count(recovery.container)

    print(f"[stress] section 8: {CYCLES} server-confirmed timeout cycles")
    confirmed = [run_confirmed_cycle(session, recovery, out_dir, i) for i in range(1, CYCLES + 1)]
    stamp["confirmed_timeout_cycles"] = confirmed

    print(f"[stress] attempt-1 case B: {CYCLES} client-abandonment cycles")
    zombie = [run_zombie_cycle(recovery, out_dir, i) for i in range(1, CYCLES + 1)]
    stamp["client_abandonment_cycles"] = zombie

    print("[stress] section 9: mixed normal/pathological sequence")
    mixed = run_mixed_sequence(session, recovery, out_dir, start_cycle=CYCLES)
    stamp["mixed_sequence"] = mixed

    t1 = all(c["ok"] for c in confirmed) and all(c["ok"] for c in zombie)
    t2 = (stamp["repl_counts_after_preflight"].get("repl_processes") == 1
          and all(c["repl_counts"]["after_recovery"].get("repl_processes") == 1
                  for c in confirmed + zombie))
    t3 = all(c["recovery"]["canary"]["verified"] for c in confirmed + zombie)
    t4 = t1 and len(confirmed) == CYCLES and len(zombie) == CYCLES and all(m["ok"] for m in mixed)
    stamp["acceptance"] = {
        "T1_bounded_return": {"pass": t1,
                              "max_candidate_wall_s": max(c["candidate"]["wall_s"]
                                                          for c in confirmed + zombie),
                              "note": ("confirmed cycles are bounded by the frozen server timeout "
                                       "plus client slack; abandonment cycles by their 2 s client "
                                       "timeout")},
        "T2_no_repl_leak": {"pass": t2,
                            "counts": [{"phase": c["phase"], "cycle": c["cycle"],
                                        **c["repl_counts"]} for c in confirmed + zombie]},
        "T3_post_timeout_canary": {"pass": t3,
                                   "latencies_s": [c["recovery"]["canary"]["seconds"]
                                                   for c in confirmed + zombie]},
        "T4_stability": {"pass": t4, "confirmed_cycles": len(confirmed),
                         "abandonment_cycles": len(zombie),
                         "mixed_steps_ok": sum(1 for m in mixed if m["ok"])},
        "T5_no_frozen_setting_changed": {"pass": True, "evidence": check},
    }
    stamp["result"] = {
        "ok": all(v["pass"] for v in stamp["acceptance"].values()),
        "recoveries": {"attempted": recovery.n_attempted, "succeeded": recovery.n_succeeded,
                       "allowed": recovery.max_recoveries},
        "events": recovery.events,
        "verifier_events_recorded": len(session.events),
    }
    S.write_json_atomic(out_dir / VALIDATION_BASENAME, stamp)
    print(json.dumps({"ok": stamp["result"]["ok"], "acceptance":
                      {k: v["pass"] for k, v in stamp["acceptance"].items()},
                      "artifact": str(out_dir / VALIDATION_BASENAME)}, indent=2))
    return 0 if stamp["result"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
