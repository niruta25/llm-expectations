"""Calibration storage."""

from __future__ import annotations

from llmex import FileCalibrationStore, LabelledScore, calibrate


def _cal(cal_id: str = "invoices_v1"):
    return calibrate(
        cal_id,
        [LabelledScore(f"d{i}", "vendor", 0.9 if i % 2 else 0.1, bool(i % 2)) for i in range(12)],
        provider_id="mock",
        model_version="mock-1.0",
        strategy_id="diverse_ensemble",
    )


def test_put_then_get_round_trips(tmp_path):
    store = FileCalibrationStore(tmp_path / "calibrations")
    cal = _cal()
    store.put(cal)
    back = store.get("invoices_v1")
    assert back is not None
    assert back.fingerprint() == cal.fingerprint()
    assert back.thresholds == cal.thresholds
    assert back.metrics["auroc"] == cal.metrics["auroc"]


def test_missing_id_returns_none_rather_than_raising(tmp_path):
    assert FileCalibrationStore(tmp_path).get("nope") is None


def test_list_and_load_all(tmp_path):
    store = FileCalibrationStore(tmp_path / "calibrations")
    assert store.list() == []
    store.put(_cal("a"))
    store.put(_cal("b"))
    assert store.list() == ["a", "b"]
    assert set(store.load_all()) == {"a", "b"}


def test_a_stored_calibration_still_gates_the_planner(tmp_path):
    """The store is the delivery mechanism; the guard is unchanged by it."""
    import pytest
    from conftest import cfg, make_tiny_batch

    from llmex import PlanError, Planner, Suite

    store = FileCalibrationStore(tmp_path / "calibrations")
    store.put(_cal())

    c = cfg(
        expectations=[
            {
                "type": "expect_field_trustworthy",
                "provider": "v",
                "calibration": "invoices_v1",
                "severity": "error",
            }
        ]
    )
    suite = Suite.from_dict(c, calibrations=store.load_all())
    Planner().plan(suite, make_tiny_batch())  # does not raise

    stale = _cal()
    stale.model_version = "mock-9.9"
    store.put(stale)
    suite = Suite.from_dict(c, calibrations=store.load_all())
    with pytest.raises(PlanError, match="Recalibrate"):
        Planner().plan(suite, make_tiny_batch())
