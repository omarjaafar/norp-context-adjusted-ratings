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
        return type("M", (), {"usage": U, "content": [b], "stop_reason": "end_turn"})()


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


def _report(tid, p1, p2, e1, e2, replicated, fold_sign=-1, alpha=0.05):
    fold = {"test": {"effect": fold_sign * 0.2}, "train": {"effect": fold_sign * 0.2}}
    return {"theory_id": tid, "verdict": "x", "validator": {"effect_floor_sd": 0.1, "alpha": alpha},
            "H1": {"effect": e1, "standardized_effect": e1, "permutation": {"p_value": p1}},
            "H2": {"fit": {"effect": e2, "standardized_effect": e2}, "permutation": {"p_value": p2}},
            "replication": {"replicated": replicated, "folds": [fold] * 5,
                            "pooled_p_value": 0.001, "pooled_effect": fold_sign * 0.2}}


def _family(true_p, n_null=9):
    """1 true theory (H1 and H2 at true_p, large effect, replicated) + n_null null theories."""
    return ([_report("true", true_p, true_p, -0.3, -0.3, True)]
            + [_report(f"null{i}", 0.5 + i / 100, 0.6 + i / 100, 0.0, 0.0, False)
               for i in range(n_null)])


def test_true_theory_survives_only_if_its_p_clears_bh_over_20_hypotheses():
    # 10 theories -> 20 hypotheses. The true theory's two p-values rank 1 and 2, so
    # BH gives q = p * 20 / 2 for both: it survives at alpha 0.05 only if p < 0.005.
    signs = {f"null{i}": -1 for i in range(9)} | {"true": -1}
    passes = apply_family(_family(0.004), signs)
    assert passes[0]["H1"]["family_q_value"] == pytest.approx(0.004 * 20 / 2)
    assert passes[0]["verdict"] == "supported"
    fails = apply_family(_family(0.006), signs)
    assert fails[0]["H1"]["family_q_value"] == pytest.approx(0.006 * 20 / 2)
    assert fails[0]["verdict"] == "unsupported"
    assert all(r["verdict"] == "unsupported" for r in passes[1:] + fails[1:])
    assert passes[0]["validator"]["fdr_family"].endswith("(20)")


def test_theory_supported_alone_is_demoted_inside_a_big_family():
    alone = _report("t", 0.01, 0.01, -0.3, -0.3, True)
    signs = {"t": -1} | {f"null{i}": -1 for i in range(19)}
    single = apply_family([alone], signs)[0]
    assert single["H1"]["family_q_value"] == pytest.approx(0.01)   # family of 2
    assert single["verdict"] == "supported"
    big = apply_family([alone] + [_report(f"null{i}", 0.5, 0.5, 0.0, 0.0, False)
                                  for i in range(19)], signs)[0]
    assert big["H1"]["family_q_value"] == pytest.approx(0.01 * 40 / 2)  # 40 hypotheses
    assert big["verdict"] == "unsupported"
    assert big["verdict_within_theory"] == "x"


def test_family_alpha_comes_from_the_specs_not_a_constant():
    signs = {"a": -1}
    # q = 0.02 * 2 / 2 = 0.02: supported at alpha 0.05, not at alpha 0.01
    assert apply_family([_report("a", 0.02, 0.02, -0.3, -0.3, True)], signs)[0]["verdict"] == "supported"
    strict = apply_family([_report("a", 0.02, 0.02, -0.3, -0.3, True, alpha=0.01)], signs)[0]
    assert strict["verdict"] == "unsupported"
    assert strict["validator"]["family_alpha"] == 0.01
    with pytest.raises(ValueError, match="different alphas"):
        apply_family([_report("a", 0.02, 0.02, -0.3, -0.3, True),
                      _report("b", 0.02, 0.02, -0.3, -0.3, True, alpha=0.01)], {"a": -1, "b": -1})


def test_reversed_needs_h2_and_opposite_replication():
    # H1 significant with the opposite sign, but H2 is not: not called reversed.
    r = _report("a", 0.001, 0.9, 0.4, 0.0, False, fold_sign=1)
    out = apply_family([r], {"a": -1})
    assert out[0]["verdict"] != "reversed"
    r2 = _report("b", 0.001, 0.001, 0.4, 0.4, False, fold_sign=1)
    assert opposite_replication(r2["replication"], -1, 0.05)
    assert apply_family([r2], {"b": -1})[0]["verdict"] == "reversed"
