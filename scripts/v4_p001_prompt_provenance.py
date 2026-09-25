#!/usr/bin/env python3
"""V4-P001 Amendment A §9 — prompt provenance: Arm-A canonical reproduction + renderer audit.

Two questions, both answered from artifacts that already exist:

1. **Arm A must be the historical theta0 theorem prompt.** The rendered Arm-A prompt is compared
   against every historical record that carries prompt evidence: the V3-R001 archive (recorded
   ``prompt_sha256``, ``prompt_token_count``, ``frozen_prompt_sha256`` per theorem), the V1 rule-RL
   roll-out dumps (recorded ``input``, the chat rendering with special tokens dropped) and E023
   (recorded ``prompt_tokens``). No historical artifact stores token **ids**, so the strongest
   reconstructible evidence is used and stated as such: exact ids of the canonical
   ``apply_chat_template`` call, exact text against the canonical builder, exact hash against the
   recorded hash, and exact special-token-dropped text against the recorded dump string.

2. **Is there an existing supported multi-turn / error-fixing renderer?** The vendored upstream
   recipe is searched, and the project's own RL configuration is searched, so the answer separates
   "a pathway exists upstream" from "the project ever enabled it". The outcome decides whether
   ``v4-prompt-1`` reuses a supported pathway or is declared a new inference-only convention.

Read-only; no generation, no verifier, no GPU.

Output: experiments/manifests/v4/v4_p001_prompt_provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from promptset_rollout_probe import build_prompt_text
from transformers import AutoTokenizer

from tinylean_rl.evaluation.v4_prompts import (
    CORRECTION_REQUEST,
    RENDERER_VERSION,
    canonical_messages,
    render_arm,
    sha256_text,
)

OUT = "experiments/manifests/v4/v4_p001_prompt_provenance.json"
POOL = "experiments/manifests/v4/v4_p001_pool.json"
TRAIN_PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
TOKENIZER = "models/weights/kimina_distill_0_6b"
R001_RAW = "runs/v3_r001_rollout_archive_from_fly122/attempt2_complete_20260925T0839Z/v3_r001_raw_rollout.jsonl"
E023_BASE = "experiments/results/e023_holdout_base.json"
V1_DUMP_GLOBS = (
    "runs/p3b_pilot/rollout_data/*.jsonl",
    "runs/m1_seed2/rollout_data/*.jsonl",
    "runs/m1_seed3/rollout_data/*.jsonl",
)
SUBMODULE = "third_party/kimina-prover-rl"
UPSTREAM_FILES = {
    "dataset": f"{SUBMODULE}/recipe/kimina_prover_rl/kimina_prover_rl/dataset.py",
    "error_fixing": f"{SUBMODULE}/recipe/kimina_prover_rl/kimina_prover_rl/reward/error_fixing.py",
    "launcher": f"{SUBMODULE}/recipe/kimina_prover_rl/kimina_prover_0.6B.sh",
}
PROJECT_CONFIGS = (
    "configs/rl/kimina_0.6b_pilot.yaml",
    "scripts/run_p3_pilot.sh",
    "scripts/run_p3_smoke.sh",
)
DIAGNOSTIC_MARKERS = ("Error message:", "Goals state", "Lean 4 code snippet with error:")


def sha256_text_local(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grep(path: Path, pattern: str) -> list[dict]:
    hits = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if pattern in line:
            hits.append({"line": number, "text": line.strip()[:200]})
    return hits


def prompt_of(tokenizer, messages) -> str:
    """verl's roll-out dump convention: chat template rendered, then special tokens dropped."""

    ids = tokenizer.apply_chat_template(
        [dict(message) for message in messages], add_generation_prompt=True, tokenize=True
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    ids = list(ids)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tokenizer.decode(ids, skip_special_tokens=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / TOKENIZER))
    pool = json.loads((ROOT / POOL).read_text())
    frame = pd.read_parquet(ROOT / TRAIN_PARQUET)
    messages_by_sid: dict[str, list] = {}
    rows_by_sid: dict[str, dict] = {}
    for sid, prompt, informal, formal in zip(
        frame["statement_id"].astype(str),
        frame["prompt"].tolist(),
        frame["informal_problem"].tolist(),
        frame["formal_statement"].tolist(),
        strict=True,
    ):
        messages_by_sid.setdefault(sid, prompt.tolist() if hasattr(prompt, "tolist") else list(prompt))
        rows_by_sid.setdefault(sid, {"natural_language": informal, "formal_statement": formal})

    def canonical(sid: str) -> str:
        return build_prompt_text(tokenizer, rows_by_sid[sid])

    def arm_a(sid: str) -> str:
        return render_arm(tokenizer, canonical_messages(messages_by_sid[sid]), arm="A_FRESH_RETRY")

    def ids_of(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    def template_ids(sid: str) -> list[int]:
        ids = tokenizer.apply_chat_template(
            [dict(message) for message in messages_by_sid[sid]], add_generation_prompt=True, tokenize=True
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        ids = list(ids)
        return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)

    def render_then_decode(sid: str) -> str:
        """End-to-end: render Arm A, re-tokenize the rendered text, drop special tokens (verl style)."""

        return tokenizer.decode(ids_of(arm_a(sid)), skip_special_tokens=True)

    # --- 1. Arm A against the V3-R001 archive ----------------------------------------------------
    r001_rows = [json.loads(line) for line in (ROOT / R001_RAW).read_text().splitlines() if line.strip()]
    r001 = {row["statement_id"]: row for row in r001_rows}
    if not r001:
        raise SystemExit("the V3-R001 archive is empty")
    r001_checks = {
        "prompt_sha256_matches": 0,
        "prompt_token_count_matches": 0,
        "frozen_prompt_sha256_matches": 0,
        "arm_a_text_equals_canonical_builder": 0,
        "arm_a_token_ids_equal_canonical_ids": 0,
        "arm_a_render_then_decode_equals_frozen_prompt": 0,
    }
    for sid, row in r001.items():
        c = canonical(sid)
        a = arm_a(sid)
        r001_checks["prompt_sha256_matches"] += int(sha256_text_local(c) == row["prompt_sha256"])
        r001_checks["prompt_token_count_matches"] += int(len(template_ids(sid)) == row["prompt_token_count"])
        r001_checks["frozen_prompt_sha256_matches"] += int(
            sha256_text_local(prompt_of(tokenizer, messages_by_sid[sid])) == row["frozen_prompt_sha256"]
        )
        r001_checks["arm_a_text_equals_canonical_builder"] += int(a == c)
        r001_checks["arm_a_token_ids_equal_canonical_ids"] += int(ids_of(a) == template_ids(sid))
        r001_checks["arm_a_render_then_decode_equals_frozen_prompt"] += int(
            render_then_decode(sid) == prompt_of(tokenizer, messages_by_sid[sid])
        )
    r001_total = len(r001)

    # --- 2. Arm A against the V1 roll-out dumps --------------------------------------------------
    v1_inputs: dict[str, int] = {}
    for pattern in V1_DUMP_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                text = str(json.loads(line).get("input") or "")
                if text:
                    v1_inputs[text] = v1_inputs.get(text, 0) + 1
    index: dict[str, str] = {
        prompt_of(tokenizer, messages): sid for sid, messages in messages_by_sid.items()
    }
    v1_matched = [text for text in v1_inputs if text in index]
    v1_byte_identical = sum(1 for text in v1_matched if render_then_decode(index[text]) == text)

    # --- 3. Arm A against E023 -------------------------------------------------------------------
    e023 = json.loads((ROOT / E023_BASE).read_text())
    e023_records = e023["records"]
    e023_match = 0
    e023_missing = 0
    for record in e023_records:
        sid = str(record["statement_id"])
        if sid not in messages_by_sid:
            e023_missing += 1
            continue
        e023_match += int(len(template_ids(sid)) == int(record["prompt_tokens"]))

    # --- 4. every pool member: renderer ids == canonical chat-template ids -----------------------
    pool_ids_equal = 0
    for member in pool["members"]:
        sid = member["statement_id"]
        pool_ids_equal += int(ids_of(arm_a(sid)) == template_ids(sid))

    # --- 5. multi-turn / error-fixing renderer audit ---------------------------------------------
    upstream_head = subprocess.run(
        ["git", "-C", str(ROOT / SUBMODULE), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    upstream = {}
    for name, relative in UPSTREAM_FILES.items():
        path = ROOT / relative
        upstream[name] = {
            "path": relative,
            "sha256": sha256_file(path) if path.exists() else None,
            "multiturn_hits": grep(path, "multiturn")[:6] if path.exists() else [],
        }
    upstream["dataset"]["builder"] = grep(ROOT / UPSTREAM_FILES["dataset"], "def create_one_multiturn_prompt")
    upstream["dataset"]["appends_assistant_then_user"] = grep(ROOT / UPSTREAM_FILES["dataset"], '"role": "assistant",')
    upstream["error_fixing"]["builder"] = grep(ROOT / UPSTREAM_FILES["error_fixing"], "def create_tool_message")
    project = {}
    for relative in PROJECT_CONFIGS:
        path = ROOT / relative
        project[relative] = {
            "sha256": sha256_file(path) if path.exists() else None,
            "multiturn_hits": grep(path, "multiturn") if path.exists() else [],
        }
    template = tokenizer.chat_template or ""
    probe_multi = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "U2"},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    template_facts = {
        "chat_template_sha256": sha256_text(template),
        "multi_turn_assistant_turn_renders_plain": "<|im_start|>assistant\nA1<|im_end|>" in probe_multi,
        "generation_prompt_suffix_present": probe_multi.endswith("<|im_start|>assistant\n"),
        "tool_role_supported_by_template": "tool" in template,
        "tool_role_used_by_v4_renderer": False,
        "upstream_uses_tool_role": bool(
            grep(ROOT / UPSTREAM_FILES["dataset"], '"role": "tool"')
            + grep(ROOT / UPSTREAM_FILES["error_fixing"], '"role": "tool"')
        ),
    }
    diagnostics = []
    for path in sorted(ROOT.glob(V1_DUMP_GLOBS[0])):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            feedback = str(json.loads(line).get("tool_feedback") or "")
            if feedback.startswith("# Error"):
                diagnostics.append(feedback)
    marker_counts = {marker: sum(1 for text in diagnostics if marker in text) for marker in DIAGNOSTIC_MARKERS}

    checks = {
        "r001_prompt_sha256_reproduced": r001_checks["prompt_sha256_matches"] == r001_total,
        "r001_prompt_token_count_reproduced": r001_checks["prompt_token_count_matches"] == r001_total,
        "r001_frozen_prompt_sha256_reproduced": r001_checks["frozen_prompt_sha256_matches"] == r001_total,
        "r001_arm_a_equals_canonical": r001_checks["arm_a_text_equals_canonical_builder"] == r001_total,
        "r001_arm_a_token_ids_equal_canonical": r001_checks["arm_a_token_ids_equal_canonical_ids"] == r001_total,
        "r001_arm_a_render_then_decode_equals_frozen_prompt": r001_checks[
            "arm_a_render_then_decode_equals_frozen_prompt"
        ]
        == r001_total,
        "v1_inputs_all_matched": len(v1_matched) == len(v1_inputs),
        "v1_inputs_byte_identical": v1_byte_identical == len(v1_inputs),
        "e023_prompt_tokens_reproduced": e023_match == len(e023_records),
        "pool_arm_a_ids_equal_canonical": pool_ids_equal == len(pool["members"]),
        "upstream_multiturn_pathway_exists": bool(
            upstream["dataset"]["builder"] and upstream["dataset"]["multiturn_hits"]
        ),
        "upstream_pathway_disabled_in_every_project_run": all(
            entry["multiturn_hits"] and any("multiturn=False" in h["text"] or "multiturn: false" in h["text"] for h in entry["multiturn_hits"])
            for entry in project.values()
        ),
    }
    failing = [name for name, ok in checks.items() if not ok]
    if failing:
        raise SystemExit(f"prompt provenance failed: {failing}")

    artifact = {
        "artifact_type": "v4_p001_prompt_provenance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.run(["hostname"], capture_output=True, text=True, check=False).stdout.strip(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),
        "note": "read-only provenance audit over already-consumed artifacts; no generation, no verifier, no GPU",
        "renderer_version": RENDERER_VERSION,
        "correction_request": CORRECTION_REQUEST,
        "arm_a_canonical_reproduction": {
            "method": (
                "no historical artifact stores token ids, so the check is four-fold: (i) the renderer's ids "
                "equal the canonical apply_chat_template ids, (ii) the rendered Arm-A text equals the "
                "canonical builder's text, (iii) sha256 of the canonical text equals the recorded hash, "
                "(iv) the special-token-dropped rendering equals the recorded dump string"
            ),
            "target": "100% on every audited historical example",
            "v3_r001_archive": {
                "n_theorems": r001_total,
                "n_rows": len(r001_rows),
                "checks": r001_checks,
            },
            "v1_rollout_dumps": {
                "n_distinct_inputs": len(v1_inputs),
                "n_rows": sum(v1_inputs.values()),
                "n_matched_to_a_pinned_theorem": len(v1_matched),
                "n_byte_identical_to_arm_a": v1_byte_identical,
            },
            "e023_holdout": {
                "n_records": len(e023_records),
                "n_prompt_tokens_matched": e023_match,
                "n_statement_not_in_parquet": e023_missing,
                "counting_convention": "E023 records prompt_tokens with special tokens (verified here: 512/512 under the with-special-tokens count)",
            },
            "pool_members_with_arm_a_ids_equal_to_canonical": pool_ids_equal,
            "pool_size": len(pool["members"]),
        },
        "multiturn_renderer_audit": {
            "question": "does an existing supported Kimina/Qwen multi-turn or error-fixing renderer exist, and is it enabled here?",
            "upstream_submodule": {"path": SUBMODULE, "head": upstream_head, "files": upstream},
            "project_rl_configuration": project,
            "model_chat_template": template_facts,
            "diagnostic_format_provenance": {
                "definition": "how often the consumed V1 verifier feedback carries the upstream error-fixing markers; the V4 normalizer removes server-wrapper lines only and never invents a format",
                "n_diagnostics": len(diagnostics),
                "marker_counts": marker_counts,
            },
            "finding": (
                "an upstream multi-turn error-fixing pathway exists (the recipe appends an assistant turn "
                "with the previous response and a plain user turn carrying the formatted Lean feedback, "
                "then re-renders with the same chat template), but every project RL run disables it: the "
                "pilot configuration sets multiturn: false and both launchers pass +data.multiturn=False. "
                "It is also a training-data construction, not an inference convention for one verifier "
                "diagnostic turn (no diagnostic normalization, no donoring, no per-theorem control)."
            ),
            "decision": (
                "v4-prompt-1 is a NEW inference-only message convention of the same shape as the upstream "
                "pathway -- assistant turn with the extracted failed proof, then a user turn with the "
                "correction request (and, in C/D, the bounded normalized diagnostic). It is not described "
                "anywhere as the model's native chat format."
            ),
            "justification_from_model_behaviour": (
                "the frozen chat template renders arbitrary assistant/user turns plainly, keeps the "
                "generation prompt suffix, and supports a tool role that this renderer deliberately does "
                "not use; the historical theta0 first attempts were rendered with the same template and the "
                "same canonical messages (reproduced above at 100%), so the first attempt and the revision "
                "turn cannot disagree about the template."
            ),
        },
        "checks": checks,
    }
    destination = ROOT / args.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2) + "\n")
    print("R001 reproduction:", json.dumps(r001_checks))
    print(f"V1 inputs: {len(v1_inputs)} distinct, {len(v1_matched)} matched, {v1_byte_identical} byte-identical")
    print(f"E023 prompt_tokens: {e023_match}/{len(e023_records)} (missing {e023_missing})")
    print(f"pool Arm-A ids == canonical ids: {pool_ids_equal}/{len(pool['members'])}")
    print("multiturn template facts:", json.dumps(template_facts))
    print("checks:", json.dumps(checks, indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
