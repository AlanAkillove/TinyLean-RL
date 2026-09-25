#!/usr/bin/env python3
"""V3-R001 formal rollout runner -- frozen sample -> theta0 n=8 -> verifier -> raw records. NOTHING ELSE.

Owner 2026-09-24 review sections 2-9: this program is a pipe with a safety interlock. Its whole
contract is

    frozen input -> theta0 generate n=8 -> canonical verify -> raw record

It does not draw theorems, re-order them, recompute or re-fit the controller head, re-select a layer
or an arm, recompute power, or touch n / temperature / top_p / max_response_length / verifier
semantics. Every one of those is a frozen object it reads and verifies by hash (see
`v3_r001_spec.load_frozen`), and every check is fail-closed: a mismatch aborts BEFORE the model is
loaded, not after some candidates were generated.

The launch interlock is deliberate. `READY_FOR_OWNER_LAUNCH` is NO until the owner says otherwise, so
a run without `--dry-run` additionally requires `--i-have-owner-launch-authorization`; the flag is
recorded in the run summary so an accidental invocation cannot quietly become a formal result.

Raw artifacts live under `runs/v3_r001/rollout_attempt2/` (gitignored): only hashes, schema and
provenance are ever committed. Attempt-1's directory (`runs/v3_r001/rollout/`) is refused by the
runner: that artifact is immutable provenance for an infrastructure abort and may never be resumed
or combined with this attempt (owner sections 2-4, 11-12).

C′ (owner 2026-09-25 sections 3-6, infrastructure-only): verification runs one candidate at a time
against a *dedicated* Kimina instance with ``MAX_REPLS = 1``, and any non-conclusive candidate
verdict triggers the mandatory recovery sequence -- restart the dedicated instance, poll /health,
re-run a nonformal canary -- before the next candidate is attempted. The recovery is bounded
(``max_recoveries_per_run``), fail-closed, and recorded in the run summary and in
``v3_r001_recovery_log.jsonl`` next to the raw artifact. It never adds a candidate attempt: the
attempt budget stays exactly the frozen policy's.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import v3_r001_spec as S
from promptset_rollout_probe import build_prompt_text, complete_verifier_code
from tqdm import tqdm

from tinylean_rl.evaluation.p3c_stats import classify_candidate
from tinylean_rl.inference.extract import extract_proof
from tinylean_rl.verifier.policy import (
    CANARY_PROOF,
    Classified,
    VerificationSession,
    VerifierUnhealthyError,
    VerifyOutcome,
    classify_result_item,
    classify_transport_error,
)

RAW_BASENAME = "v3_r001_raw_rollout.jsonl"
SUMMARY_BASENAME = "v3_r001_run_summary.json"
RECOVERY_LOG_BASENAME = "v3_r001_recovery_log.jsonl"
VERIFIER_EVENTS_BASENAME = "v3_r001_verifier_events.jsonl"


def abort(reason: str, checks: list | None = None) -> int:
    """The only legal way out of a failed preflight: say why, name the check, write nothing."""
    print(f"[ABORT BEFORE GENERATION] {reason}", file=sys.stderr)
    for c in checks or []:
        if not c["pass"]:
            print(f"  failed check: {c['check']} -- {c['detail']}", file=sys.stderr)
    return 3


# --- C′ verifier lifecycle: dedicated instance, mandatory recovery (owner sections 3, 5, 6) -------
#
# Attempt-1 (2026-09-25) aborted three times because a candidate request that the client had already
# abandoned kept a server-side Lean computation alive on a reusable REPL: the frozen B0 policy could
# only *detect* that state (canary -> fail-close) and stop. C′ keeps the frozen detection exactly as
# it is and adds the missing half -- restoring capacity -- as a client-side, bounded, fail-closed
# step: `docker restart` of the dedicated instance, then /health, then a nonformal canary, then the
# next candidate. Evidence and the case analysis are in docs/v3/V3-R001_infrastructure_amendment_
# Cprime.md; the invariant being enforced is owner section 3:
#
#   no candidate verification attempt may leave a Lean computation occupying a reusable REPL after
#   that attempt is classified timed out/failed; capacity is restored before the next candidate.
#
# This is infrastructure recovery, not a candidate retry (owner section 6): the candidate's attempt
# budget is decided by the frozen policy alone, and a restart never adds an attempt to it.

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
    """The C′ supervisor around the dedicated verifier instance (`S.VERIFIER_INFRA`).

    ``verify_identity()`` is the fail-closed preflight check: the endpoint this run verifies against
    must be *this* container (pinned image, ``LEAN_SERVER_MAX_REPLS = 1``, its published port), not
    the shared 16-slot server. ``recover(trigger)`` is the owner section 6 sequence -- restart,
    /health, nonformal canary -- recorded as one event per call in ``events`` and appended durably to
    the run directory, and bounded by ``max_recoveries_per_run`` so a pathological run stops instead
    of looping.
    """

    def __init__(self, out_dir: Path, *, container: str | None = None,
                 endpoint: str | None = None, image: str | None = None) -> None:
        infra = S.VERIFIER_INFRA
        self.container = container or infra["container"]
        self.endpoint = (endpoint or infra["endpoint"]).rstrip("/")
        self.image = image or infra["image"]
        self.max_recoveries = int(infra["max_recoveries_per_run"])
        self.restart_timeout = float(infra["restart_timeout_s"])
        self.health_timeout = float(infra["health_poll_timeout_s"])
        self.health_interval = float(infra["health_poll_interval_s"])
        self.restart_grace = int(infra["restart_grace_s"])
        self.out_dir = Path(out_dir)
        self.log_path = self.out_dir / RECOVERY_LOG_BASENAME
        self.n_attempted = 0
        self.n_succeeded = 0
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
        """The post-restart canary: nonformal, generous first-use budget, re-warms the singleton.

        The first request after a restart creates the only REPL and pays its Mathlib import inside
        that request, so it must run with the cold-start budget (``first_timeout_s``), not the warm
        canary budget the frozen policy uses for its in-run gates.
        """
        import httpx

        from tinylean_rl.verifier.kimina import verify_code

        v = S.VERIFIER
        started = time.perf_counter()
        try:
            decoded = verify_code(CANARY_PROOF, custom_id=f"v3-r001-canary-{tag}",
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

    def recover(self, trigger: dict) -> dict:
        """Owner section 6: restart -> /health -> nonformal canary, before the next candidate.

        Fail-closed: any step that cannot be completed returns ``ok=False`` with a reason, and the
        caller must abort (the run stays resumable). The trigger is recorded verbatim so the summary
        shows why capacity had to be restored.
        """
        self.n_attempted += 1
        if self.n_attempted > self.max_recoveries:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger,
                    "reason": (f"recovery budget exhausted: {self.n_attempted - 1} of "
                               f"{self.max_recoveries} restarts already used in this launch")}
        try:
            self.verify_identity()
        except RecoveryError as exc:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger, "reason": str(exc)}

        print(f"[{S.EXPERIMENT_ID}] recovery {self.n_attempted}/{self.max_recoveries}: docker "
              f"restart {self.container} (trigger: outcome={trigger.get('outcome')} "
              f"rank={trigger.get('theorem_rank')} sample={trigger.get('sample_index')})")
        started = time.perf_counter()
        proc = self._docker("restart", "-t", str(self.restart_grace), self.container,
                            timeout=self.restart_timeout)
        restart_seconds = round(time.perf_counter() - started, 2)
        if proc.returncode != 0:
            return {"ok": False, "n": self.n_attempted, "trigger": trigger,
                    "restart_seconds": restart_seconds,
                    "reason": (f"docker restart failed (rc={proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[:300]}")}

        health = self.wait_healthy()
        canary = (self.cold_canary(f"rec{self.n_attempted}") if health["ok"]
                  else {"verified": False, "status": "skipped",
                        "detail": "health poll failed; the canary was not attempted"})
        ok = bool(health["ok"] and canary["verified"])
        event = {"n": self.n_attempted, "at": datetime.now(timezone.utc).isoformat(),
                 "trigger": trigger, "restart_seconds": restart_seconds, "health": health,
                 "canary": canary, "recovery_seconds": round(time.perf_counter() - started, 2),
                 "ok": ok}
        self.events.append(event)
        self._append_event(event)
        if ok:
            self.n_succeeded += 1
        print(f"[{S.EXPERIMENT_ID}] recovery {self.n_attempted}: restart {restart_seconds}s, "
              f"health {health.get('seconds')}s, canary {canary.get('status')} "
              f"{canary.get('seconds')}s -> {'ok' if ok else 'FAILED'}")
        return {"ok": ok, **event}

    def _append_event(self, event: dict) -> None:
        S.append_rows_durable(self.log_path, [event])


# --- the frozen input surface -------------------------------------------------------------------

def load_statement_texts() -> dict:
    """statement_id -> (name, natural_language, formal_statement, source) from the pinned parquet.

    The parquet's bytes are already checked against the frozen pool's `inputs` pin inside
    `load_frozen`, so reading it here cannot silently disagree with the design.
    """
    rows = pd.read_parquet(ROOT / S.RAW_PARQUET,
                           columns=["statement_id", "name", "natural_language",
                                     "formal_statement", "source"])
    out: dict = {}
    for r in rows.itertuples(index=False):
        sid = str(r.statement_id)
        if sid in out:
            continue                                  # first occurrence: same rule as the probe path
        out[sid] = {"name": str(r.name), "natural_language": str(r.natural_language or ""),
                    "formal_statement": str(r.formal_statement), "source": str(r.source)}
    return out


def seed_schedule(theorems: list[dict]) -> dict:
    """statement_id -> its 8 generation seeds, a function of the frozen draw-order rank ONLY.

    Kept separate from `build_plan` so the property owner amendment B requires -- permuting q_B2,
    q_B1 or the source label must not move a single seed -- is testable without a tokenizer or a GPU.
    """
    return {t["statement_id"]: [S.candidate_seed(t["formal_sample_rank"], s)
                                for s in range(S.N_SAMPLES)] for t in theorems}


def seed_set_sha256(plan: list[dict]) -> str:
    """One hash over the 1024 (formal_sample_rank, sample_index, seed) triples, in plan order."""
    return S.sha([[e["formal_sample_rank"], s, e["seeds"][s]]
                  for e in plan for s in range(S.N_SAMPLES)])


def build_plan(frozen: S.Frozen, tokenizer) -> list[dict]:
    """The 128 theorems in FROZEN rank order, with the prompt each candidate will be given.

    Order is `rank_by_q_B2` ascending -- the order the block was cut in -- and is never re-sorted by
    anything the runner knows about. The generation seeds are a function of the OTHER frozen rank,
    `formal_sample_rank` (the outcome-free draw order), so no RNG stream is decided by the
    controller's ranking (owner amendment B).

    The prompt is checked three ways before it may be sent, because the q that selected this
    theorem was computed on a rendered string, not on a statement id:
      1. the frozen rendering (`prompt_of`, the same function the pre-scoring used) re-tokenizes to
         exactly the frozen `prompt_token_count`;
      2. the canonical generation prompt (E023/E024's `build_prompt_text`) is the SAME string once
         the chat template's special tokens are skipped -- so the rollout proves the theorem whose
         representation produced the q, not a look-alike;
      3. the `# Formal Statement:` block of the canonical prompt is the pinned parquet's statement
         under `_normalize` (the same normalization the B1 features were computed on), and its raw
         length is the frozen `formal_char_count`.
    """
    from v3_d001_lib import _normalize, formal_text
    from v3_r001_extract import prompt_of  # the rendering the frozen q was computed on

    texts = load_statement_texts()
    pm = pd.read_parquet(ROOT / S.TRAIN_PARQUET, columns=["statement_id", "prompt"])
    msgs: dict = {}
    for sid, p in zip(pm["statement_id"].astype(str), pm["prompt"].tolist()):
        msgs.setdefault(sid, p.tolist() if hasattr(p, "tolist") else list(p))

    schedule = seed_schedule(frozen.theorems)
    plan = []
    for t in sorted(frozen.theorems, key=lambda r: r["rank_by_q_B2"]):
        sid = t["statement_id"]
        if sid not in texts:
            raise S.FrozenViolation(f"statement {sid} is not in the pinned promptset parquet")
        if sid not in msgs:
            raise S.FrozenViolation(f"statement {sid} has no pinned chat-template messages")
        row = texts[sid]
        if row["source"] != t["source"]:
            raise S.FrozenViolation(
                f"{sid}: source is {row['source']!r} on disk but {t['source']!r} in the frozen sample")
        if len(row["formal_statement"]) != t["formal_char_count"]:
            raise S.FrozenViolation(
                f"{sid}: formal statement is {len(row['formal_statement'])} chars but the frozen "
                f"sample records {t['formal_char_count']}")
        frozen_prompt = prompt_of(tokenizer, msgs[sid])
        n_frozen = len(tokenizer(frozen_prompt, add_special_tokens=False)["input_ids"])
        if n_frozen != t["prompt_token_count"]:
            raise S.FrozenViolation(
                f"{sid}: the frozen rendering tokenizes to {n_frozen} but the sample records "
                f"{t['prompt_token_count']} -- the q of this theorem was computed on a different "
                "string, so it must not be rolled out")
        prompt = build_prompt_text(tokenizer, {"natural_language": row["natural_language"],
                                               "formal_statement": row["formal_statement"]})
        n_tok = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        normalized = tokenizer.decode(tokenizer(prompt, add_special_tokens=False)["input_ids"],
                                      skip_special_tokens=True)
        if normalized != frozen_prompt:
            raise S.FrozenViolation(
                f"{sid}: the canonical generation prompt is not the frozen prompt modulo template "
                "special tokens; generation would be scoring a different string than the q claims")
        if formal_text(prompt) != _normalize(row["formal_statement"]):
            raise S.FrozenViolation(
                f"{sid}: the '# Formal Statement:' block is not the pinned statement text under "
                "the same normalization the B1 features use")
        if n_tok + S.MAX_RESPONSE_TOKENS > S.MAX_MODEL_LEN:
            raise S.FrozenViolation(
                f"{sid}: prompt {n_tok} + max response {S.MAX_RESPONSE_TOKENS} exceeds "
                f"max_model_len {S.MAX_MODEL_LEN} (the freeze recorded a context check that this "
                "run must reproduce, not work around)")
        plan.append({
            "theorem_rank": t["rank_by_q_B2"],
            "formal_sample_rank": t["formal_sample_rank"],
            "statement_id": sid,
            "component_id": t["component_id"],
            "name": row["name"],
            "source": row["source"],
            "formal_statement": row["formal_statement"],
            "prompt": prompt,
            "prompt_sha256": S.sha256_text(prompt),
            "formal_statement_sha256": S.sha256_text(row["formal_statement"]),
            "prompt_token_count": n_tok,
            "frozen_prompt_sha256": S.sha256_text(frozen_prompt),
            "in_top20pct_block": bool(t["in_top20pct_block"]),
            "q_B2": t["q_B2_controller"],
            "seeds": schedule[sid],
        })
    if [p["theorem_rank"] for p in plan] != list(range(1, S.N_NOMINAL + 1)):
        raise S.FrozenViolation("the plan is not the frozen 1..128 rank sequence")
    if sorted(p["formal_sample_rank"] for p in plan) != list(range(1, S.N_NOMINAL + 1)):
        raise S.FrozenViolation("the plan does not carry the frozen 1..128 draw-order ranks")
    if len({p["statement_id"] for p in plan}) != S.N_NOMINAL:
        raise S.FrozenViolation("the plan does not have 128 unique statements")
    if len({p["component_id"] for p in plan}) != S.N_NOMINAL:
        raise S.FrozenViolation("the plan does not have one theorem per family component")
    all_seeds = [s for p in plan for s in p["seeds"]]
    if len(set(all_seeds)) != S.EXPECTED_TOTAL_CANDIDATES:
        raise S.FrozenViolation(
            f"the plan's {len(all_seeds)} seeds are not all distinct; two candidates sharing a seed "
            "would not be the two independent draws the design describes (owner amendment B)")
    if {e["statement_id"]: e["seeds"] for e in plan} != schedule:
        raise S.FrozenViolation("a planned theorem does not carry its frozen seed schedule")
    return plan


# --- verifier health (B0 policy: canary, fail-close) --------------------------------------------
#
# C′: every verifier call in this runner -- the preflight canary, the frozen policy's own probes and
# the recovery canary -- is addressed to the dedicated instance endpoint explicitly. Nothing in the
# run may fall back to the environment's default URL (the shared 16-slot server).

def verifier_endpoint() -> str:
    return S.VERIFIER_INFRA["endpoint"]


def make_session() -> VerificationSession:
    v = S.VERIFIER
    return VerificationSession(base_url=verifier_endpoint(), server_timeout=v["server_timeout_s"],
                               client_slack=v["client_slack_s"], batch_size=v["batch_size"],
                               max_single_retries=v["max_single_retries"],
                               canary_timeout=v["canary_timeout_s"], canary_retries=v["canary_retries"])


def check_verifier(session: VerificationSession) -> dict:
    """`/health`, then the canary with E024's cold-start budget. Neither touches a formal theorem.

    The generous first budget exists because the first Mathlib import on a fresh server is
    legitimately slow; without it a warm-up delay would be recorded as an infrastructure failure of
    the design's first group (owner 21 allows a non-formal canary in a dry run).
    """
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

        decoded = verify_code(CANARY_PROOF, custom_id="v3-r001-canary",
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


# --- generation and verification -----------------------------------------------------------------

def error_category(truncated: bool, format_ok: bool, status: str, lean_message: str) -> str:
    """A machine-readable cause for every non-verified candidate; infra never borrows a proof label."""
    if status == VerifyOutcome.VERIFIED.value:
        return "none"
    if status in S.INFRA_STATUSES:
        return status                    # e.g. verifier_timeout: the run must not hide behind "lean_error"
    return classify_candidate(truncated=truncated, format_ok=format_ok, verify_status=status,
                              lean_message=lean_message)


def candidate_from_output(entry: dict, sample_index: int, completion, per_candidate_s: float,
                          env: dict, run_id: str) -> tuple[dict, str | None]:
    """One raw candidate row plus the Lean source to verify (None when nothing was extractable).

    The row is returned unvalidated: its verifier fields are decided by `verify_group`, and
    `validate_row` runs once per row immediately before the group is appended.
    """
    text = completion.text
    try:
        extracted = extract_proof(text)
    except ValueError:
        extracted = ""
    proof = complete_verifier_code(entry["formal_statement"], extracted)
    row = {
        "schema_version": S.SCHEMA_VERSION,
        "run_id": run_id,
        "experiment_id": S.EXPERIMENT_ID,
        "theorem_rank": entry["theorem_rank"],
        "formal_sample_rank": entry["formal_sample_rank"],
        "statement_id": entry["statement_id"],
        "component_id": entry["component_id"],
        "name": entry["name"],
        "source": entry["source"],
        "sample_index": sample_index,
        "seed": entry["seeds"][sample_index],
        "prompt_sha256": entry["prompt_sha256"],
        "frozen_prompt_sha256": entry["frozen_prompt_sha256"],
        "formal_statement_sha256": entry["formal_statement_sha256"],
        "model_sha256": env["model_sha256"],
        "model_revision": env["model_revision"],
        "completion_text": text,
        "completion_sha256": S.sha256_text(text),
        "generated_tokens": len(list(completion.token_ids)),
        "truncated": bool(completion.finish_reason == "length"),
        "format_ok": bool(proof),
        "verified": False,
        "score": None,
        "acc": 0,
        "verify_status": "",
        "verifier_error_category": "",
        "lean_message": "",
        "generation_time": round(per_candidate_s, 4),
        "verification_time": None,
        "host": env["hostname"],
        "gpu": env["gpu_names"].splitlines()[0] if env["gpu_names"] else "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_token_count": entry["prompt_token_count"],
        "extracted_proof_present": bool(extracted),
    }
    return row, proof


def finalize_rows(verified_rows: list[dict], classified: list, format_failed: list[dict]) -> None:
    """Write the verdict into each row; a candidate that never reached the verifier gets FORMAT_ERROR.

    `format_failed` rows are conclusive model failures (score 0), NOT infrastructure: nothing was sent
    to the verifier because there was no proof body to send. `verified_rows` is `classified` in order.
    Every row leaves this function category-tagged and validated, so a malformed row cannot reach disk.
    """
    if len(verified_rows) != len(classified):
        raise S.FrozenViolation(
            f"{len(verified_rows)} candidates but {len(classified)} verifier outcomes; a group "
            "cannot be finalized against a mismatched verdict list")
    for row, c in zip(verified_rows, classified):
        row["verify_status"] = c.outcome.value
        row["lean_message"] = c.message[:500]
        row["verified"] = c.outcome is VerifyOutcome.VERIFIED
        row["score"] = S.candidate_score(c.outcome.value)
        row["acc"] = 1 if row["verified"] else 0
    for row in format_failed:
        row["verify_status"] = S.FORMAT_ERROR
        row["lean_message"] = ("no Lean proof body could be extracted from the completion; the "
                               "candidate was never sent to the verifier")
        row["verified"] = False
        row["score"] = 0
        row["acc"] = 0
    for row in list(verified_rows) + list(format_failed):
        row["verifier_error_category"] = error_category(
            truncated=row["truncated"], format_ok=row["format_ok"], status=row["verify_status"],
            lean_message=row["lean_message"])
        S.validate_row(row)


def pending_groups(plan: list[dict], existing: dict) -> tuple[list[dict], dict]:
    """Whole-group resume: finished groups are skipped, unfinished ones are dropped and re-run.

    A group counts as finished when all 8 candidate rows exist -- including INFRA_CENSORED ones. Its
    label then stays missing by rule, and re-running it would merge candidates from two runs into one
    group, which owner 8 forbids. Anything with fewer than 8 rows is discarded as a unit and re-run
    from the same deterministic seeds, so it is provably identical to an uninterrupted group.
    """
    resume_plan = {"n_rows_on_disk": sum(len(v) for v in existing.values()),
                   "groups_on_disk": len(existing), "rerun": [], "skipped": []}
    todo = []
    for entry in plan:
        rows = existing.get(entry["statement_id"])
        if not rows:
            todo.append(entry)
            continue
        fin = S.finalize_group(rows)
        if fin["group_status"] in (S.GROUP_COMPLETE, S.GROUP_INFRA_CENSORED):
            resume_plan["skipped"].append({"statement_id": entry["statement_id"],
                                           "rank": entry["theorem_rank"],
                                           "group_status": fin["group_status"]})
        else:
            resume_plan["rerun"].append({"statement_id": entry["statement_id"],
                                         "rank": entry["theorem_rank"],
                                         "n_rows": len(rows), "reason": fin["reason"]})
            todo.append(entry)
    resume_plan["groups_to_run"] = len(todo)
    resume_plan["expected_new_candidates"] = len(todo) * S.N_SAMPLES
    return todo, resume_plan


def compact_existing(path: Path, existing: dict, plan: list[dict]) -> None:
    """Rewrite the raw artifact with only the retained (finished, not re-run) groups.

    Append-only cannot un-write a partial group, so resume compacts first: the file is replaced
    atomically by a temp-and-rename containing exactly the groups that will not be re-run. After this
    point every append is a whole new group and the never-merge invariant holds by construction.
    """
    keep_ids = {e["statement_id"] for e in plan}
    rows = []
    for sid, group in existing.items():
        if sid not in keep_ids:
            continue
        if S.finalize_group(group)["group_status"] in (S.GROUP_COMPLETE, S.GROUP_INFRA_CENSORED):
            rows.extend(sorted(group, key=lambda r: r["sample_index"]))
    tmp = path.with_suffix(".jsonl.compact.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                   encoding="utf-8")
    with tmp.open("a", encoding="utf-8") as fh:
        os.fsync(fh.fileno())
    tmp.replace(path)


# --- C′ per-candidate verification and run provenance -------------------------------------------

def verify_candidate_with_recovery(session: VerificationSession, recovery: VerifierRecovery,
                                   proof: str, custom_id: str, *, rank: int, sample_index: int,
                                   ) -> tuple[Classified, str | None]:
    """One candidate's frozen policy run; capacity is restored before the next candidate.

    The policy is called with exactly this candidate -- ``batch_size = 1``, i.e. the owner's section
    5 architecture of one verification in flight -- and everything inside it (the batch attempt, the
    isolation retries, the canary gates, the taxonomy, the fail-close) is the frozen B0 policy. Two
    things are C′, and only two:

    - a ``VerifierUnhealthyError`` no longer ends the run. The candidate is classified
      ``unresolved_infra_error`` -- never a score, never a candidate failure -- and the recovery
      sequence runs; the group's label becomes INFRA_CENSORED by the frozen rule;
    - after ANY non-conclusive classification, the recovery sequence runs before the next candidate.
      That is the owner's section 3 invariant made mechanical: whatever the server-side lifecycle
      did to the reusable REPL, the client restores it by construction.

    Returns ``(classified, abort_reason)``. A non-None abort reason means capacity could not be
    restored and the caller must stop (fail-close); the candidate's verdict is still returned so the
    caller can record it.
    """
    try:
        classified = session.verify([proof], [custom_id])[0]
    except VerifierUnhealthyError as exc:
        classified = Classified(
            VerifyOutcome.UNRESOLVED_INFRA_ERROR,
            f"candidate verification interrupted by the canary fail-close "
            f"(rank={rank} sample={sample_index}): {exc}"[:500])
    if classified.outcome.is_conclusive:
        return classified, None
    restored = recovery.recover({"theorem_rank": rank, "sample_index": sample_index,
                                 "custom_id": custom_id, "outcome": classified.outcome.value,
                                 "message": classified.message[:300]})
    if not restored["ok"]:
        return classified, (f"capacity restoration failed after {classified.outcome.value} on "
                            f"rank={rank} sample={sample_index}: "
                            f"{restored.get('reason') or 'the recovery canary did not verify'}")
    return classified, None


def flush_session_events(session: VerificationSession, path: Path, cursor: int) -> int:
    """Append the policy's own event trail durably and return the new cursor.

    ``session.events`` is the frozen policy's audit trail (every sub-batch, isolation and canary
    event with its timestamp). It is written incrementally so a run that aborts still leaves the
    verifier-side history on disk next to the raw artifact.
    """
    events = session.events[cursor:]
    if events:
        S.append_rows_durable(path, [{"at": e.at, "kind": e.kind, "detail": e.detail}
                                     for e in events])
    return len(session.events)


def verifier_infrastructure_stamp(session: VerificationSession, recovery: VerifierRecovery,
                                  instance: dict | None, events_path: Path) -> dict:
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
        "recoveries_attempted": recovery.n_attempted,
        "recoveries_succeeded": recovery.n_succeeded,
        "recovery_events": recovery.events,
        "recovery_log_path": str(recovery.log_path),
        "verifier_events_recorded": len(session.events),
        "verifier_events_path": str(events_path),
    }


def write_aborted_summary(summary_path: Path, stamp: dict, generated: int, groups: list,
                          session: VerificationSession, recovery: VerifierRecovery,
                          instance: dict | None, events_path: Path, reason: str) -> None:
    """The fail-close record: what was generated, what was kept, and why the run stopped."""
    S.write_json_atomic(
        summary_path.with_name(SUMMARY_BASENAME.replace(".json", ".aborted.json")),
        {**stamp, "candidates_generated": generated, "groups": groups, "aborted": reason,
         "verifier_infrastructure": verifier_infrastructure_stamp(session, recovery, instance,
                                                                  events_path)})


# --- main ---------------------------------------------------------------------------------------

def parse_args(argv: list | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="run every check, load nothing onto the GPU, generate nothing")
    ap.add_argument("--resume", action="store_true", help="keep finished groups, re-run partial ones")
    ap.add_argument("--out-dir", default="runs/v3_r001/rollout_attempt2")
    ap.add_argument("--i-have-owner-launch-authorization", dest="authorized", action="store_true",
                    help="required for a non-dry run; the owner launch decision is recorded here")
    return ap.parse_args(argv)


def launch_interlock(args: argparse.Namespace) -> str | None:
    """The owner's launch decision, as a rule that can be tested without a tokenizer or a GPU."""
    if args.dry_run or args.authorized:
        return None
    return ("READY_FOR_OWNER_LAUNCH = NO. A formal rollout requires the owner's launch "
            "authorization; pass --i-have-owner-launch-authorization only when the owner has given "
            "it in writing.")


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    raw_path = out_dir / RAW_BASENAME
    summary_path = out_dir / SUMMARY_BASENAME

    # The attempt this code may write (owner sections 2-4, 11-12): attempt-1's artifacts are
    # immutable provenance, so their directory is refused outright. A `--resume` there would skip
    # attempt-1's finished groups and produce a single artifact mixing two executions, which the
    # owner forbids.
    superseded = (ROOT / S.SUPERSEDED_ATTEMPT1_DIR).resolve()
    if superseded == out_dir.resolve() or superseded in out_dir.resolve().parents:
        return abort(f"{S.SUPERSEDED_ATTEMPT1_DIR} holds V3-R001 attempt-1's raw artifact "
                     "(INFRASTRUCTURE_ABORT / NO_SCIENTIFIC_OUTCOME / NOT_ANALYZED), which is "
                     "immutable provenance; formal attempt-2 must run in a fresh directory "
                     "(default: runs/v3_r001/rollout_attempt2)")

    print(f"[{S.EXPERIMENT_ID}] attempt {S.ATTEMPT} settings hash {S.sha(S.FROZEN_SETTINGS)}")
    print(f"[{S.EXPERIMENT_ID}] dry_run={args.dry_run} resume={args.resume} out={raw_path}")

    # 1. host / repo interlock (owner 2, 16)
    env = S.collect_env()
    try:
        S.check_environment(env, require_formal_host=not args.dry_run)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    if args.dry_run and S.FORMAL_HOST_IP not in env["ips"]:
        print("[note] dry-run on a non-formal host: host checks were run in report-only mode. "
              "A formal run still requires fly122.")
    print(f"[{S.EXPERIMENT_ID}] host={env['hostname']} ip={env['ips']} gpu={env['gpu_names']!r} "
          f"git={env['git_revision'][:9]} branch={env['branch']}")

    # 2. frozen objects and their hashes
    try:
        frozen = S.load_frozen(verify_files=True)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] {len(frozen.checks)} frozen-artifact checks passed "
          f"(sample {frozen.formal['sample_sha256'][:12]}, block m={len(frozen.block_ids)}, "
          f"theta0 {frozen.theta0['weights_sha256'][:12]})")

    # 3. prompt surface (tokenizer only; no model weights loaded yet)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / frozen.theta0["dir"], trust_remote_code=True,
                                              local_files_only=True)
    try:
        plan = build_plan(frozen, tokenizer)
    except S.FrozenViolation as exc:
        return abort(str(exc))
    print(f"[{S.EXPERIMENT_ID}] plan: {len(plan)} theorems x {S.N_SAMPLES} samples = "
          f"{len(plan) * S.N_SAMPLES} candidates; every prompt re-tokenized to its frozen length")
    for tag, entry in (("first", plan[0]), ("last", plan[-1])):
        print(f"[{S.EXPERIMENT_ID}] {tag}: rank={entry['theorem_rank']} name={entry['name']} "
              f"source={entry['source']} block={entry['in_top20pct_block']} "
              f"q_B2={entry['q_B2']:.4f} prompt_sha={entry['prompt_sha256'][:12]} "
              f"seeds {entry['seeds'][0]}..{entry['seeds'][-1]}")

    # 4. resume state
    existing: dict = {}
    if raw_path.exists():
        if not args.resume:
            return abort(f"{raw_path} already exists; refusing to append to a raw artifact without "
                         "--resume (a merged group of candidates from two runs is forbidden)")
        try:
            existing = S.read_groups(raw_path)
        except S.FrozenViolation as exc:
            return abort(f"existing raw artifact is not consumable: {exc}")
        off_sample = sorted(set(existing) - {e["statement_id"] for e in plan})
        if off_sample:
            return abort(f"the raw artifact holds {len(off_sample)} statement(s) outside the frozen "
                         f"sample ({off_sample[:3]}); this file was not produced by this design")
        for group in existing.values():
            for r in group:
                if r["model_sha256"] != frozen.theta0["weights_sha256"]:
                    return abort("an existing candidate row carries a different model hash; the "
                                 "stored group was produced by another checkpoint and cannot be "
                                 "merged with this run")
    todo, resume_plan = pending_groups(plan, existing)

    # 5. verifier instance, health and canary -- all before anything is generated (owner 9; C′ 5-6)
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
        print(f"[{S.EXPERIMENT_ID}] dedicated verifier instance: "
              f"{instance.get('container_id')} {instance.get('image')} "
              f"published={instance.get('published')} env={instance.get('env')}")

    session = make_session()
    health = check_verifier(session)
    print(f"[{S.EXPERIMENT_ID}] verifier health: {health.get('health_status')} "
          f"canary={health.get('canary', {}).get('status')} "
          f"({health.get('canary', {}).get('seconds')}s)")
    if not health.get("canary", {}).get("verified"):
        return abort(f"verifier is not healthy (canary did not verify): {health}")
    verifier_events_path = out_dir / VERIFIER_EVENTS_BASENAME
    events_cursor = flush_session_events(session, verifier_events_path, 0)

    stamp = {"experiment_id": S.EXPERIMENT_ID, "schema_version": S.SCHEMA_VERSION,
             "attempt": S.ATTEMPT,
             "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
             "frozen_settings": S.FROZEN_SETTINGS,
             "frozen_settings_sha256": S.sha(S.FROZEN_SETTINGS),
             "script_versions": S.script_version_hash("scripts/v3_r001_rollout.py",
                                                      "scripts/v3_r001_spec.py",
                                                      "scripts/v3_r001_gate.py"),
             "prereg_commit": S.PREREG_COMMIT, "git": env,
             "sample_sha256": frozen.formal["sample_sha256"],
             "block_sha256": frozen.formal["top20pct_block"]["sha256"],
             "gate_sha256": frozen.file_hashes[S.GATE],
             "formal_sample_sha256_on_disk": frozen.file_hashes[S.FORMAL_SAMPLE],
             "theta0": {k: frozen.theta0[k] for k in ("dir", "revision", "weights_sha256",
                                                      "weights_bytes", "repo_id")},
             "prompt_source": {S.RAW_PARQUET: frozen.pool["inputs"][S.RAW_PARQUET]},
             "verifier_health": health, "resume_plan": resume_plan,
             "verifier_infrastructure": verifier_infrastructure_stamp(session, recovery, instance,
                                                                      verifier_events_path),
             "n_theorems_planned": len(plan), "n_theorems_to_run": len(todo),
             "expected_total_candidates": len(plan) * S.N_SAMPLES,
             "checks": frozen.checks}

    if args.dry_run:
        stamp["mode"] = "DRY_RUN"
        stamp["formal_candidates_generated"] = 0
        stamp["expected_total_candidates"] = S.EXPECTED_TOTAL_CANDIDATES
        stamp["formal_theorems_enumerated"] = len(plan)
        stamp["n_unique_seeds"] = len({s for e in plan for s in e["seeds"]})
        stamp["seed_set_sha256"] = seed_set_sha256(plan)
        stamp["canary_uses_no_formal_theorem"] = not any(
            "tinylean_v2_canary" in e["prompt"] for e in plan)
        stamp["raw_artifact_written"] = False
        stamp["formal_result_files_created"] = []
        print(json.dumps({k: stamp[k] for k in
                          ("mode", "attempt", "formal_theorems_enumerated",
                           "expected_total_candidates", "formal_candidates_generated",
                           "n_unique_seeds", "seed_set_sha256", "canary_uses_no_formal_theorem",
                           "raw_artifact_written", "formal_result_files_created", "run_id",
                           "frozen_settings_sha256")},
                         indent=2))
        print(f"[{S.EXPERIMENT_ID}] DRY RUN COMPLETE: llm.load_called=false "
              f"generate_called=false model.generate on formal theorems=0 "
              f"candidates_generated=0 verifier_formal_calls=0 "
              f"formal_raw_result_files_created=0")
        _print_dry_run_summary(session, stamp)
        return 0

    deny = launch_interlock(args)
    if deny:
        return abort(deny)

    # 6. the real path: engine, chunks, whole-group append
    stamp["mode"] = "FORMAL"
    stamp["owner_launch_authorized"] = True
    if resume_plan["rerun"]:
        compact_existing(raw_path, existing, plan)
        print(f"[{S.EXPERIMENT_ID}] compacted {len(resume_plan['rerun'])} partial group(s) before "
              "appending")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"[{S.EXPERIMENT_ID}] loading theta0 into vLLM ({frozen.theta0['dir']})")
    llm = LLM(model=str(ROOT / frozen.theta0["dir"]), tokenizer=str(ROOT / frozen.theta0["dir"]),
              trust_remote_code=True, max_model_len=S.MAX_MODEL_LEN,
              gpu_memory_utilization=S.GPU_MEMORY_UTILIZATION, max_num_seqs=256,
              max_num_batched_tokens=8192, disable_log_stats=True)
    env["model_sha256"] = frozen.theta0["weights_sha256"]
    env["model_revision"] = frozen.theta0["revision"]
    torch.cuda.reset_peak_memory_stats()
    entry_by_rank = {e["theorem_rank"]: e for e in todo}

    t0 = time.perf_counter()
    generated = 0
    groups = []
    chunks = [todo[i:i + S.CHUNK_THEOREMS] for i in range(0, len(todo), S.CHUNK_THEOREMS)]
    try:
        for chunk in tqdm(chunks, desc="V3-R001", unit="chunk", dynamic_ncols=True):
            prompts, params, keys = [], [], []
            for entry in chunk:
                for s in range(S.N_SAMPLES):
                    prompts.append(entry["prompt"])
                    params.append(SamplingParams(temperature=S.TEMPERATURE, top_p=S.TOP_P,
                                                 max_tokens=S.MAX_RESPONSE_TOKENS, n=1,
                                                 seed=entry["seeds"][s]))
                    keys.append((entry["theorem_rank"], s))
            gen_t = time.perf_counter()
            outputs = llm.generate(prompts, params)               # theorem-major, frozen seeds, n=1 x 8
            gen_seconds = time.perf_counter() - gen_t
            per_candidate = gen_seconds / len(prompts)

            per_slot: dict[int, dict] = {entry["theorem_rank"]: {} for entry in chunk}
            for (rank, s), output in zip(keys, outputs):
                row, proof = candidate_from_output(entry_by_rank[rank], s, output.outputs[0],
                                                   per_candidate, env, stamp["run_id"])
                per_slot[rank][s] = (row, proof)

            ver_t = time.perf_counter()
            all_checked: list[dict] = []
            for rank in sorted(per_slot):
                pairs = [per_slot[rank][s] for s in range(S.N_SAMPLES)]
                checked = [(row, proof) for row, proof in pairs if proof]
                # C′: one candidate at a time, capacity restored before the next one. With
                # batch_size = 1 this is the same sequence of requests the frozen policy would issue
                # for this group (see the amendment for the ordering argument); what it adds is that
                # a verdict is kept even when the policy has to fail closed mid-group, and that no
                # candidate is attempted on a server that has not just been proven healthy.
                classified: list[Classified] = []
                for row, proof in checked:
                    verdict, abort_reason = verify_candidate_with_recovery(
                        session, recovery, proof, f"{rank}-{row['sample_index']}",
                        rank=rank, sample_index=row["sample_index"])
                    events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
                    if abort_reason:
                        write_aborted_summary(summary_path, stamp, generated, groups, session,
                                              recovery, instance, verifier_events_path, abort_reason)
                        return abort(f"{abort_reason}. {generated} candidates in {len(groups)} "
                                     "finished groups were kept; resume with --resume once the "
                                     "dedicated verifier instance is healthy.")
                    classified.append(verdict)
                all_checked.extend(row for row, _proof in checked)
                finalize_rows([row for row, _proof in checked], classified,
                              [row for row, proof in pairs if not proof])
            per_verify = (time.perf_counter() - ver_t) / max(1, len(all_checked))
            for row in all_checked:
                row["verification_time"] = round(per_verify, 4)

            for rank in sorted(per_slot):
                rows = sorted((row for row, _proof in per_slot[rank].values()),
                              key=lambda r: r["sample_index"])
                entry = entry_by_rank[rank]
                for row in rows:
                    row["verifier_error_category"] = error_category(
                        truncated=row["truncated"], format_ok=row["format_ok"],
                        status=row["verify_status"], lean_message=row["lean_message"])
                    S.validate_row(row)
                fin = S.finalize_group(rows)
                if fin["group_status"] == S.GROUP_INCOMPLETE:
                    # cannot happen after a full generate of 8, but fail closed rather than assume
                    return abort(f"group rank={rank} is {fin['group_status']}: {fin['reason']}")
                S.append_rows_durable(raw_path, rows)             # one whole group per append
                generated += len(rows)
                groups.append({"statement_id": entry["statement_id"], "theorem_rank": rank,
                               "component_id": entry["component_id"], "source": entry["source"],
                               "in_top20pct_block": entry["in_top20pct_block"], **fin})
                print(f"  rank {rank:>3} {entry['source'][:13]:<13} {fin['group_status']:<15} "
                      f"y={fin['y']} n_pos={fin['n_pos_score']}")
            events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
    except VerifierUnhealthyError as exc:
        # Defensive: `verify_candidate_with_recovery` converts a policy fail-close into an
        # unresolved_infra_error plus a recovery attempt, so this clause is only reachable if a
        # future policy raises from somewhere else. Stop cleanly, keep the finished groups, and
        # leave the same fail-close record on disk.
        write_aborted_summary(summary_path, stamp, generated, groups, session, recovery, instance,
                              verifier_events_path, f"VerifierUnhealthyError: {exc}")
        return abort(f"verifier became unhealthy mid-run: {exc}. {generated} candidates in "
                     f"{len(groups)} finished groups were kept; resume with --resume.")

    events_cursor = flush_session_events(session, verifier_events_path, events_cursor)
    stamp.update({"candidates_generated": generated, "groups": groups,
                  "generation_seconds": round(time.perf_counter() - t0, 1),
                  "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                  "vllm_version": vllm.__version__,
                  "formal_candidates_generated": generated,
                  "raw_artifact_sha256": S.sha256_file(raw_path) if raw_path.exists() else None,
                  "verifier_infrastructure": verifier_infrastructure_stamp(
                      session, recovery, instance, verifier_events_path)})
    n_infra = sum(1 for g in groups if g["group_status"] == S.GROUP_INFRA_CENSORED)
    stamp["infra_censored_groups_this_run"] = n_infra
    stamp["censoring_rate_this_run"] = round(n_infra / max(1, len(groups)), 4)
    S.write_json_atomic(summary_path, stamp)
    print(json.dumps({"candidates_generated": generated,
                      "expected_total_candidates": S.EXPECTED_TOTAL_CANDIDATES,
                      "groups": len(groups), "infra_censored": n_infra,
                      "recoveries": recovery.n_attempted,
                      "raw": str(raw_path), "summary": str(summary_path)}, indent=2))
    return 0


def _print_dry_run_summary(session: VerificationSession, stamp: dict) -> None:
    """What a formal run WOULD do, and the evidence that nothing was generated."""
    infra = stamp["verifier_infrastructure"]
    instance = infra.get("dedicated_instance") or {}
    print(json.dumps({
        "run_id": stamp["run_id"],
        "attempt": stamp["attempt"],
        "frozen_settings_sha256": stamp["frozen_settings_sha256"],
        "script_versions": stamp["script_versions"],
        "n_theorems_planned": stamp["n_theorems_planned"],
        "n_theorems_to_run": stamp["n_theorems_to_run"],
        "expected_total_candidates": stamp["expected_total_candidates"],
        "formal_theorems_enumerated": stamp["formal_theorems_enumerated"],
        "n_unique_seeds": stamp["n_unique_seeds"],
        "seed_set_sha256": stamp["seed_set_sha256"],
        "canary_uses_no_formal_theorem": stamp["canary_uses_no_formal_theorem"],
        "formal_result_files_created": stamp["formal_result_files_created"],
        "candidates_generated": 0,
        "verifier_events_recorded": len(session.events),
        "verifier_infrastructure": {
            "endpoint": infra["endpoint"], "container": infra["container"],
            "image": infra["image"], "max_repls": infra["max_repls"],
            "verification_concurrency": infra["verification_concurrency"],
            "policy_batch_size": infra["policy_batch_size"],
            "recoveries_allowed": infra["recoveries_allowed"],
            "dedicated_instance_running": instance.get("running"),
            "dedicated_instance_id": instance.get("container_id"),
            "dedicated_instance_env": instance.get("env"),
        },
        "resume_plan": stamp["resume_plan"],
        "planned_settings": {k: stamp["frozen_settings"][k] for k in
                             ("n_samples", "temperature", "top_p", "max_response_tokens",
                              "max_model_len", "gpu_memory_utilization", "chunk_theorems",
                              "seed_base", "seed_group_size", "seed_formula")},
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
