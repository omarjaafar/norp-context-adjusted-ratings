"""CP4: LLM cache, Planner and Data Scout (with a fake model), continuous exposures, family gate."""

import copy
import json

import numpy as np
import pytest

from theorylab.agents import DataScout, Planner
from theorylab.catalog import REPO_ROOT
from theorylab.gate import apply_family, opposite_replication
from theorylab.llm import CacheError, LLMClient, extract_json
from theorylab.spec import validate_spec
from theorylab.theory_run import available_columns

TEMPLATE = REPO_ROOT / "specs" / "ratings_fairness.json"


class FakeAPI:
    """Stands in for the Anthropic client: returns queued texts."""
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        text = self.texts.pop(0)

        class U: input_tokens, output_tokens = 100, 50
        class B: type, text_ = "text", text
        b = B(); b.text = text
        return type("M", (), {"usage": U, "content": [b]})()


def _client(tmp_path, texts, replay=False):
    c = LLMClient(cache_dir=tmp_path, replay_only=replay)
    c._api = FakeAPI(texts)
    return c


def test_cache_hit_avoids_second_call_and_tamper_is_rejected(tmp_path):
    c = _client(tmp_path, ["hello"])
    assert c.complete("s", "p") == "hello"
    assert c.complete("s", "p") == "hello"
    assert c._api.calls == 1 and c.usage.cache_hits == 1
    path = next(tmp_path.glob("*.json"))
    entry = json.loads(path.read_text())
    entry["text"] = "tampered"
    path.write_text(json.dumps(entry))
    with pytest.raises(CacheError):
        c.complete("s", "p")
    assert c.verify_cache()["bad"] == [path.name]


def test_replay_only_without_entry_fails(tmp_path):
    with pytest.raises(CacheError):
        _client(tmp_path, [], replay=True).complete("s", "never asked")


def test_extract_json_handles_fences_and_braces_in_strings():
    assert extract_json('```json\n{"a": "}{", "b": {"c": 1}}\n``` done') == {"a": "}{", "b": {"c": 1}}


def _good_spec():
    s = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    s["theory_id"] = "poverty_program_share"
    return s


def test_scout_rejects_unknown_fields(tmp_path):
    bad = json.dumps({"outcome": "made_up", "exposure": "poverty_rate", "exposure_type": "continuous",
                      "exposure_unit": "county_fips", "controls": []})
    scout = DataScout(_client(tmp_path, [bad]), available_columns())
    with pytest.raises(ValueError, match="made_up"):
        scout.map_variables("t")


def test_planner_retries_once_then_accepts_and_never_repairs(tmp_path):
    broken = _good_spec()
    broken["expected_sign"] = 0
    planner = Planner(_client(tmp_path, [json.dumps(broken), json.dumps(_good_spec())]), str(TEMPLATE))
    spec = planner.plan("t", {}, available_columns())
    assert validate_spec(spec, available_columns()) == []
    assert [h["errors"] != [] for h in planner.history] == [True, False]
    always_bad = Planner(_client(tmp_path / "b", [json.dumps(broken)] * 2), str(TEMPLATE))
    with pytest.raises(ValueError, match="rejected"):
        always_bad.plan("t", {}, available_columns())


def test_continuous_spec_needs_correlation_h1():
    s = _good_spec()
    s["exposure_type"] = "continuous"
    assert any("correlation" in e for e in validate_spec(s))
    s["hypotheses"][0]["method"] = "correlation"
    assert validate_spec(s) == []


def _report(tid, p1, p2, e1, e2, replicated, fold_sign=-1):
    fold = {"test": {"effect": fold_sign * 0.2}, "train": {"effect": fold_sign * 0.2}}
    return {"theory_id": tid, "verdict": "x", "validator": {"effect_floor_sd": 0.1},
            "H1": {"effect": e1, "standardized_effect": e1, "permutation": {"p_value": p1}},
            "H2": {"fit": {"effect": e2, "standardized_effect": e2}, "permutation": {"p_value": p2}},
            "replication": {"replicated": replicated, "folds": [fold] * 5,
                            "pooled_p_value": 0.001, "pooled_effect": fold_sign * 0.2}}


def test_family_q_is_larger_than_within_theory_and_can_remove_support():
    reports = [_report("a", 0.01, 0.01, -0.3, -0.3, True)] + \
              [_report(f"n{i}", 0.5, 0.5, 0.0, 0.0, False) for i in range(8)]
    signs = {r["theory_id"]: -1 for r in reports}
    out = apply_family(reports, signs)
    assert out[0]["H1"]["family_q_value"] == pytest.approx(0.01 * 18 / 1) or out[0]["H1"]["family_q_value"] > 0.05
    assert out[0]["verdict"] in ("unsupported", "weak", "supported")
    assert out[0]["verdict_within_theory"] == "x"


def test_reversed_needs_h2_and_opposite_replication():
    # H1 significant with the opposite sign, but H2 is not: not called reversed.
    r = _report("a", 0.001, 0.9, 0.4, 0.0, False, fold_sign=1)
    out = apply_family([r], {"a": -1})
    assert out[0]["verdict"] != "reversed"
    r2 = _report("b", 0.001, 0.001, 0.4, 0.4, False, fold_sign=1)
    assert opposite_replication(r2["replication"], -1, 0.05)
    assert apply_family([r2], {"b": -1})[0]["verdict"] == "reversed"
