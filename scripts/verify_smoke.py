#!/usr/bin/env python3
"""Check one known-good and one known-bad Lean request through Kimina."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Windows consoles default to GBK, which cannot print Lean goal symbols.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tinylean_rl.verifier.kimina import result_lean_status, verify_code

CASES = {
    "positive": "import Mathlib\n\nexample : 1 + 1 = 2 := by\n  norm_num",
    "negative": "import Mathlib\n\nexample : 1 + 1 = 3 := by\n  norm_num",
}
EXPECTED = {"positive": "valid", "negative": "lean_error"}


def main() -> int:
    print("Kimina Lean Server verifier smoke test")
    ok = True
    for name, proof in CASES.items():
        try:
            result = verify_code(proof, custom_id=f"p0-{name}")
        except Exception as exc:  # noqa: BLE001 - report endpoint failures
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            return 2
        items = result.get("results", result if isinstance(result, list) else [])
        status = result_lean_status(items[0]) if items else "unknown"
        expected = EXPECTED[name]
        verdict = "OK" if status == expected else "UNEXPECTED"
        ok = ok and status == expected
        print(f"[{name}] status={status} expected={expected} ({verdict})")
        print(f"  raw: {json.dumps(result, ensure_ascii=False)[:400]}")
    print("Verifier smoke test " + ("passed." if ok else "did NOT match expectations."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

