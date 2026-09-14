#!/usr/bin/env python3
"""Check one known-good and one known-bad Lean request through Kimina."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinylean_rl.verifier.kimina import verify_code

CASES = {
    "positive": "import Mathlib\n\nexample : 1 + 1 = 2 := by\n  norm_num",
    "negative": "import Mathlib\n\nexample : 1 + 1 = 3 := by\n  norm_num",
}


def main() -> int:
    print("Kimina Lean Server verifier smoke test")
    for name, proof in CASES.items():
        try:
            result = verify_code(proof, custom_id=f"p0-{name}")
        except Exception as exc:  # noqa: BLE001 - report endpoint failures
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            return 2
        print(f"[{name}] {json.dumps(result, ensure_ascii=False)}")
    print("Review the returned statuses: positive must pass and negative must fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

