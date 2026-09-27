"""Driver-level regressions for the R001 smoke harness (owner §12; draft §10).

The driver only observes the trainer, so two of its own contracts are safety rails worth pinning
without a GPU:

* ``VramSampler`` must stop cleanly - a failed attempt's peak VRAM is part of the S1 record, and
  ``threading.Thread`` already owns the ``_stop`` attribute (shadows raise a ``TypeError`` in
  ``join``, which silently dropped the whole attempt record once);
* ``append_attempt`` is the §12 gate: the first attempt is free, and every later attempt must
  change exactly one ``ALLOWED_KNOBS`` entry - never the science.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "scripts", ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import v5_r001_smoke as smoke

STAMP = {"git_head": "deadbeef", "files": {}}
KNOB = "actor_rollout_ref.rollout.gpu_memory_utilization"


def _attempt(knobs: dict[str, str], outcome: str = "completed") -> dict[str, object]:
    return {"knobs": dict(knobs), "outcome": outcome, "seconds": 1.0}


def test_vram_sampler_stops_cleanly() -> None:
    sampler = smoke.VramSampler(interval=0.05)
    sampler.start()
    sampler.stop()
    assert sampler.is_alive() is False, "stop() must join, not raise"
    assert sampler.peak_mib is None or isinstance(sampler.peak_mib, int)


def test_first_attempt_is_recorded_without_a_diff(tmp_path: Path) -> None:
    path = tmp_path / "attempts.json"
    attempts = smoke.append_attempt(path, STAMP, _attempt(smoke.BASELINE_KNOBS))
    assert len(attempts) == 1
    assert attempts[0]["attempt"] == 1
    # No previous attempt: the "diff" is the whole baseline knob set, which is the record of what
    # the first attempt ran with.
    assert attempts[0]["diff_from_previous"] == {
        key: {"previous": None, "current": value} for key, value in smoke.BASELINE_KNOBS.items()
    }


def test_retry_must_change_exactly_one_allowed_knob(tmp_path: Path) -> None:
    path = tmp_path / "attempts.json"
    smoke.append_attempt(path, STAMP, _attempt(smoke.BASELINE_KNOBS, outcome="oom"))

    with pytest.raises(SystemExit, match="changes no knob"):
        smoke.append_attempt(path, STAMP, _attempt(smoke.BASELINE_KNOBS, outcome="oom"))

    with pytest.raises(SystemExit, match="must change exactly one knob"):
        smoke.append_attempt(
            path,
            STAMP,
            _attempt(
                {
                    **smoke.BASELINE_KNOBS,
                    KNOB: "0.25",
                    "actor_rollout_ref.rollout.enforce_eager": "True",
                }
            ),
        )

    with pytest.raises(SystemExit, match="non-allowed knobs"):
        smoke.append_attempt(
            path, STAMP, _attempt({**smoke.BASELINE_KNOBS, "trainer.total_training_steps": "1"})
        )

    attempts = smoke.append_attempt(path, STAMP, _attempt({**smoke.BASELINE_KNOBS, KNOB: "0.25"}))
    assert attempts[-1]["attempt"] == 2
    assert attempts[-1]["diff_from_previous"] == {KNOB: {"previous": "0.30", "current": "0.25"}}


def test_knob_override_replaces_the_baseline_value() -> None:
    overrides = smoke.base_overrides(
        arm="treatment",
        steps=1,
        rollout_dir=None,
        knobs={KNOB: "0.25"},
    )
    values = [item for item in overrides if item.startswith(KNOB + "=")]
    assert values == [f"{KNOB}=0.25"], "a knob override must replace, never duplicate"


def test_ray_prefixed_marker_lines_are_parsed() -> None:
    # The driver process prints R001_ENTRY itself; the rest come from Ray workers.
    log = (
        'R001_ENTRY {"argv": ["a=b"]}\n'
        '(R001TaskRunner pid=293285) R001_STACK {"adv_estimator": "r001_process"}\n'
        '(compute_reward_async pid=293571) R001_REWARD {"rows": 16, "submitted": 7}\n'
        '(WorkerDict pid=7) R001_MEM {"max_rss_mib": 1054.9}\n'
        '(R001TaskRunner pid=1) R001_STATS {"credit_tokens": 53}\n'
        # Payload text is never re-interpreted as a marker, and unknown markers are ignored.
        '(R001TaskRunner pid=1) R001_REWARD {"note": "see R001_STATS line"}\n'
        'worker R001_MEM {"max_rss_mib": 1}\n'
    )
    found = smoke.parse_r001_lines(log)
    assert found["R001_ENTRY"] == [{"argv": ["a=b"]}]
    assert found["R001_STACK"] == [{"adv_estimator": "r001_process"}]
    assert found["R001_REWARD"] == [{"rows": 16, "submitted": 7}, {"note": "see R001_STATS line"}]
    assert found["R001_MEM"] == [{"max_rss_mib": 1054.9}]
    assert found["R001_STATS"] == [{"credit_tokens": 53}]


def test_ray_prefixed_step_lines_are_parsed() -> None:
    log = "(R001TaskRunner pid=9) step:1 - actor/pg_loss:0.5 - perf/time_per_step:12.25\n"
    rows = smoke.parse_step_metrics(log)
    assert rows == [{"step": 1.0, "actor/pg_loss": 0.5, "perf/time_per_step": 12.25}]
