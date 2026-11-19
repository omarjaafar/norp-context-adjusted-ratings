"""scripts/verify_outputs.py: the committed outputs pass, and a changed number is caught."""

import importlib.util

import pytest

from theorylab.catalog import REPO_ROOT

spec = importlib.util.spec_from_file_location("verify_outputs", REPO_ROOT / "scripts" / "verify_outputs.py")
vo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vo)


@pytest.mark.parametrize("name", sorted(vo.CHECKS))
def test_committed_outputs_pass(name):
    assert vo.CHECKS[name]()


def test_a_changed_sweep_number_is_caught(monkeypatch):
    real = vo._load

    def tampered(rel):
        d = real(rel)
        d["main_cells"][0]["h1_r"] = 0.2  # above the floor, yet the cell says unsupported
        return d

    monkeypatch.setattr(vo, "_load", tampered)
    with pytest.raises(AssertionError):
        vo.check_sweep_cells()


def test_a_missing_input_is_skipped_not_passed(monkeypatch):
    monkeypatch.setattr(vo, "REPO_ROOT", REPO_ROOT / "does-not-exist")
    with pytest.raises(vo.Skip):
        vo.check_outcome_tails()
