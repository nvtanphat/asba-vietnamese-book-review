from __future__ import annotations

import math
from typing import Any


def _num(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def evaluate_quality_gate(metrics: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    val = _num(metrics, "val_f1_combined")
    test = _num(metrics, "test_f1_combined")
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "pass": bool(ok), "actual": actual, "expected": expected})

    for key in gate.get("require_metrics", []):
        value = _num(metrics, str(key))
        add(f"require_metric:{key}", value is not None, value, "present")

    if "min_val_f1_combined" in gate:
        threshold = float(gate["min_val_f1_combined"])
        add("min_val_f1_combined", val is not None and val >= threshold, val, f">={threshold}")
    if gate.get("require_test"):
        add("require_test", test is not None, test, "present")
    if "min_test_f1_combined" in gate:
        threshold = float(gate["min_test_f1_combined"])
        add("min_test_f1_combined", test is not None and test >= threshold, test, f">={threshold}")
    if "max_generalization_gap" in gate:
        threshold = float(gate["max_generalization_gap"])
        gap = abs(val - test) if val is not None and test is not None else None
        # Candidate/staging can legitimately be validation-only; only enforce gap when test exists.
        ok = True if gap is None and not gate.get("require_test") else (gap is not None and gap <= threshold)
        add("max_generalization_gap", ok, gap, f"<={threshold}")

    passed = all(item["pass"] for item in checks)
    return {"status": "PASS" if passed else "FAIL", "passed": passed, "checks": checks}
