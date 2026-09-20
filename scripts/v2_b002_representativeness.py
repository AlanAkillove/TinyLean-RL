#!/usr/bin/env python
"""V2-B002 representativeness audit: is the 512 pilot a biased slice?

Compares the B002 pilot against the full Promptset (7620 unique statements),
the E023 sealed holdout (128) and the IGR mechanism set (64) on: source, domain
(name family), statement/NL length (chars + tokens), component size structure
(singleton vs multi-variant), and approximate Lean syntax complexity.

Also quantifies whether the component-level exclusions (V1-used / E023 / A001 /
A-reserve / B1 / C-joint) push the B002 pool toward singletons / the hard tail.

Data: registry component metadata + Promptset parquet + sealed-set manifests.
Output: experiments/results/v2_b002_representativeness.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REGISTRY = "experiments/manifests/v2/family_component_registry.json"
PARQUET = "data/processed/p3_promptset/prompt_sets/AI-MO/Kimina-Prover-Promptset/train.parquet"
PILOT = "experiments/manifests/v2/v2_b002_pilot_set.json"
E023 = "experiments/manifests/m1_final_holdout.json"
MECH = "experiments/manifests/igr_mechanism_set.json"
OUTPUT = "experiments/results/v2_b002_representativeness.json"

HYP_RE = re.compile(r"\([^()]{1,80}:")
FORALL = "\u2200"
ARROW = "\u2192"


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_texts() -> dict[str, dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(resolve(PARQUET), columns=["statement_id", "formal_statement", "informal_problem", "data_source", "name"])
    texts: dict[str, dict] = {}
    for row in table.to_pylist():
        sid = row["statement_id"]
        if sid not in texts:  # dedup order: keep first occurrence
            texts[sid] = row
    return texts


def load_registry() -> tuple[dict[str, dict], dict[str, int], set[str], list[dict]]:
    reg = json.loads(resolve(REGISTRY).read_text(encoding="utf-8"))
    statement_comp: dict[str, dict] = {}
    eligible_ids: set[str] = set()
    for comp in reg["components"]:
        for sid in comp["member_statement_ids"]:
            statement_comp[sid] = comp
            if comp.get("eligible_for_b2"):
                eligible_ids.add(sid)
    return statement_comp, reg["counts"], eligible_ids, reg["components"]


def syntax_of(text: str) -> dict:
    head = text
    marker = "theorem "
    idx = head.find(marker)
    if idx >= 0:
        head = head[idx:]
        for stop in (" := ", ":=", "\n"):
            p = head.find(stop)
            if p > 0:
                head = head[:p]
                break
    return {
        "hyp_approx": len(HYP_RE.findall(head)),
        "forall_count": text.count(FORALL),
        "arrow_count": text.count(ARROW),
        "lines": text.count("\n") + 1,
    }


def stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    return {
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 1),
    }


def summarize(name: str, sids: list[str], texts: dict, statement_comp: dict, tokenizer) -> dict:
    rows = [texts[s] for s in sids if s in texts]
    missing = len(sids) - len(rows)
    formal_chars = [len(r["formal_statement"]) for r in rows]
    nl_chars = [len(r["informal_problem"] or "") for r in rows]
    tokens = [len(tokenizer(r["formal_statement"])["input_ids"]) for r in rows]
    hyp, forall, arrow, lines = [], [], [], []
    for r in rows:
        sx = syntax_of(r["formal_statement"])
        hyp.append(sx["hyp_approx"])
        forall.append(sx["forall_count"])
        arrow.append(sx["arrow_count"])
        lines.append(sx["lines"])
    sources = collections.Counter(r["data_source"] for r in rows)
    domains = collections.Counter((r["name"].split("_")[0] if r["name"] else "?") for r in rows)
    sizes = []
    multi_variant = 0
    comp_multi_comp = collections.Counter()
    for s in sids:
        comp = statement_comp.get(s)
        if comp is None:
            continue
        sizes.append(comp["size"])
        if comp["size"] > 1:
            multi_variant += 1
        comp_multi_comp[comp["component_id"]] = comp["size"]
    size_hist = collections.Counter(
        "1" if s == 1 else ("2-4" if s <= 4 else ("5-9" if s <= 9 else "10+")) for s in sizes
    )
    result = {
        "name": name,
        "n_requested": len(sids),
        "n_with_text": len(rows),
        "missing_text": missing,
        "data_source": dict(sources),
        "domain_top": dict(domains.most_common(10)),
        "formal_chars": stats(formal_chars),
        "formal_tokens": stats(tokens),
        "nl_chars": stats(nl_chars),
        "hyp_approx": stats(hyp),
        "forall": stats(forall),
        "arrow": stats(arrow),
        "lines": stats(lines),
        "component": {
            "singleton_pct": round(100 * (1 - multi_variant / len(sizes)), 1) if sizes else None,
            "multi_variant_pct": round(100 * multi_variant / len(sizes), 1) if sizes else None,
            "size_mean": round(statistics.mean(sizes), 2) if sizes else None,
            "size_max": max(sizes) if sizes else None,
            "size_hist": dict(size_hist),
            "unique_components": len(comp_multi_comp),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="V2-B002 representativeness audit.")
    parser.add_argument("--output", default=OUTPUT)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "models/weights/kimina_distill_0_6b"), trust_remote_code=True, local_files_only=True
    )
    texts = load_texts()
    statement_comp, reg_counts, eligible_ids, components = load_registry()

    pilot = json.loads(resolve(PILOT).read_text(encoding="utf-8"))
    pilot_ids = [t["statement_id"] for t in pilot["theorems"]]
    e023 = json.loads(resolve(E023).read_text(encoding="utf-8"))
    e023_ids = [t["statement_id"] for t in e023.get("theorems", [])]
    mech = json.loads(resolve(MECH).read_text(encoding="utf-8"))
    mech_ids = [t["statement_id"] for t in mech.get("theorems", [])]

    full_ids = sorted(statement_comp.keys())
    print(f"full={len(full_ids)} eligible_ids={len(eligible_ids)} pilot={len(pilot_ids)} e023={len(e023_ids)} mech={len(mech_ids)}")

    report = {
        "artifact_type": "v2_b002_representativeness",
        "registry_counts": reg_counts,
        "sets": {
            "full_promptset": summarize("full_promptset", full_ids, texts, statement_comp, tokenizer),
            "b002_pilot_512": summarize("b002_pilot_512", pilot_ids, texts, statement_comp, tokenizer),
            "e023_holdout_128": summarize("e023_holdout_128", e023_ids, texts, statement_comp, tokenizer),
            "igr_mechanism_64": summarize("igr_mechanism_64", mech_ids, texts, statement_comp, tokenizer),
        },
        "eligible_pool": {
            "statements": len(eligible_ids),
            "pct_of_full": round(100 * len(eligible_ids) / len(full_ids), 1),
            "multi_variant_pct": round(
                100 * sum(1 for s in eligible_ids if statement_comp[s]["size"] > 1) / len(eligible_ids), 1
            ),
            "component_level": {
                "components": sum(1 for c in components if c.get("eligible_for_b2")),
                "multi_variant_components": sum(
                    1 for c in components if c.get("eligible_for_b2") and c["size"] > 1
                ),
            },
            "full_component_level": {
                "components": len(components),
                "multi_variant_components": sum(1 for c in components if c["size"] > 1),
            },
        },
    }

    # exclusion-shift diagnosis: which statements do the exclusions consume?
    excluded_ids = set(full_ids) - eligible_ids
    ex_sizes = [statement_comp[s]["size"] for s in excluded_ids]
    ex_hist = collections.Counter("1" if s == 1 else ("2-4" if s <= 4 else ("5-9" if s <= 9 else "10+")) for s in ex_sizes)
    report["excluded_pool"] = {
        "statements": len(excluded_ids),
        "pct_of_full": round(100 * len(excluded_ids) / len(full_ids), 1),
        "multi_variant_pct": round(
            100 * sum(1 for s in excluded_ids if statement_comp[s]["size"] > 1) / max(1, len(excluded_ids)), 1
        ),
        "size_hist": dict(ex_hist),
    }

    output_path = resolve(args.output)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    keys = ["formal_tokens", "nl_chars", "hyp_approx", "arrow", "lines"]
    names = list(report["sets"].keys())
    print(f"{'metric':16s} " + " ".join(f"{n:>20s}" for n in names))
    for k in keys:
        row = []
        for n in names:
            s = report["sets"][n][k]
            row.append(f"med={s['median']} mean={s['mean']}")
        print(f"{k:16s} " + " ".join(f"{v:>20s}" for v in row))
    print()
    for n in names:
        s = report["sets"][n]
        print(
            f"{n:22s} n={s['n_requested']:5d} singleton%={s['component']['singleton_pct']} "
            f"multivar%={s['component']['multi_variant_pct']} size_mean={s['component']['size_mean']} "
            f"size_hist={s['component']['size_hist']}"
        )
    for n in names:
        print(f"{n:22s} data_source={report['sets'][n]['data_source']} domain_top={list(report['sets'][n]['domain_top'].items())[:5]}")
    print("eligible pool:", report["eligible_pool"])
    print("excluded pool:", report["excluded_pool"])
    print(f"artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
