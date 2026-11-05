"""CP4 audit fixes for the agent path (W2, W4, W7, W8, W16), with a fake LLM.

- the Planner cannot loosen the gate; every attempt is logged as an override
- the Planner must use the Scout's outcome, exposure and unit
- both agents get a description for every panel column
- the cache stores each call's own tokens, sends no temperature, never caches a refusal,
  and a replay recomputes the cost of the original run
- the orchestrator loads every column all theories need, end to end on a synthetic panel
"""

import json

import numpy as np
import pytest
import yaml

import theorylab.orchestrator as orch_mod
from theorylab.agents import COLUMNS_PATH, DataScout, Planner, scout_disagreements
from theorylab.catalog import REPO_ROOT
from theorylab.llm import LLMClient, LLMError
from theorylab.orchestrator import Orchestrator
from theorylab.spec import GATE_POLICY, lock_gate
from theorylab.theory_run import DERIVED, available_columns

from test_theory_run import _panel

TEMPLATE = REPO_ROOT / "specs" / "ratings_fairness.json"


class FakeAPI:
    """Stands in for the Anthropic client: returns queued (text, tokens, stop_reason)."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.messages = self

    def create(self, **kw):
        self.requests.append(kw)
        text, (tin, tout), stop = self.replies.pop(0)

        class U:
            input_tokens, output_tokens = tin, tout

        class B:
            type = "text"

        b = B()
        b.text = text
        return type("M", (), {"usage": U, "content": [b], "stop_reason": stop})()


def _client(tmp_path, texts, tokens=(100, 50), stop="end_turn", **kw):
    c = LLMClient(cache_dir=tmp_path, **kw)
    c._api = FakeAPI([(t, tokens, stop) for t in texts])
    return c


def _ratings_spec():
    return json.loads(TEMPLATE.read_text(encoding="utf-8"))


# ---- gate lock (W2) -----------------------------------------------------------

def test_gate_policy_is_the_tested_ratings_gate():
    assert _ratings_spec()["validator"] == GATE_POLICY


def test_lock_gate_replaces_a_loosened_gate_and_logs_it():
    loose = _ratings_spec()
    loose["validator"].update({"alpha": 0.5, "effect_floor_sd": 0.0, "n_permutations": 19,
                               "permutation": {"keep_paths": False}, "extra": 1})
    loose["hypotheses"][1]["cluster"] = "ein"
    locked, overrides = lock_gate(loose)
    assert locked["validator"] == GATE_POLICY
    assert locked["hypotheses"][1]["cluster"] == "county_fips"
    fields = {o["field"] for o in overrides}
    assert {"validator.alpha", "validator.effect_floor_sd", "validator.n_permutations",
            "validator.permutation", "validator.extra", "H2.cluster"} == fields
    assert loose["validator"]["alpha"] == 0.5  # the input is not mutated


def test_planner_spec_runs_under_the_policy_even_when_it_asks_for_less(tmp_path):
    loose = _ratings_spec()
    loose["validator"]["effect_floor_sd"] = 0.0
    planner = Planner(_client(tmp_path, [json.dumps(loose)]), str(TEMPLATE))
    spec = planner.plan("t", {}, available_columns())
    assert spec["validator"] == GATE_POLICY
    assert spec["gate_overrides"] == [{"field": "validator.effect_floor_sd", "planner": 0.0,
                                       "policy": 0.1}]


# ---- Scout and Planner agree (W4) ---------------------------------------------

def test_planner_must_use_the_scouts_variables(tmp_path):
    mapping = {"outcome": "operating_margin", "exposure": "high_need", "exposure_type": "binary",
               "exposure_unit": "county_fips", "controls": []}
    spec = _ratings_spec()  # outcome standard_rating, not operating_margin
    assert scout_disagreements(spec, mapping) == [
        "outcome is 'standard_rating' but the Scout mapped 'operating_margin'"]
    planner = Planner(_client(tmp_path, [json.dumps(spec)] * 2), str(TEMPLATE))
    with pytest.raises(ValueError, match="Scout mapped"):
        planner.plan("t", mapping, available_columns())
    assert scout_disagreements(spec, {"fields_available": []}) == []  # --no-scout


def test_every_panel_column_has_a_description():
    described = yaml.safe_load(COLUMNS_PATH.read_text(encoding="utf-8"))["columns"]
    missing = sorted(set(available_columns()) - set(described))
    assert missing == [], f"add these to catalog/panel_columns.yaml: {missing}"
    assert set(DERIVED) <= set(described)


def test_scout_prompt_explains_columns(tmp_path):
    scout = DataScout(_client(tmp_path, []), available_columns())
    text = scout.prompt("t")
    assert "- liabilities_to_assets: Total liabilities / total assets" in text
    assert "(no description)" not in text


# ---- LLM client and cache (W7, W16) ------------------------------------------

def test_cache_keeps_each_calls_tokens_and_replay_recomputes_cost(tmp_path):
    live = LLMClient(cache_dir=tmp_path)
    live._api = FakeAPI([("a", (1000, 200), "end_turn"), ("b", (3000, 400), "end_turn")])
    live.complete("s", "p1")
    live.complete("s", "p2")
    entries = sorted((json.loads(p.read_text()) for p in tmp_path.glob("*.json")),
                     key=lambda e: e["input_tokens"])
    assert [(e["input_tokens"], e["output_tokens"]) for e in entries] == [(1000, 200), (3000, 400)]
    assert all("temperature" not in kw for kw in live._api.requests)
    # $2 / $10 per million tokens for claude-sonnet-5-5
    assert live.usage.cost_usd("claude-sonnet-5-5") == pytest.approx(4000 * 2e-6 + 600 * 10e-6)
    replay = LLMClient(cache_dir=tmp_path, replay_only=True)
    assert replay.complete("s", "p2") == "b" and replay.complete("s", "p1") == "a"
    assert replay.usage.calls == 0 and replay.usage.cache_hits == 2
    assert replay.usage.cost_usd("claude-sonnet-5-5") == pytest.approx(live.usage.cost_usd("claude-sonnet-5-5"))
    assert replay.usage.cost_usd("claude-sonnet-5-5", include_cached=False) == 0


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refused_or_truncated_answers_are_not_cached(tmp_path, stop):
    c = _client(tmp_path, ["{"], stop=stop)
    with pytest.raises(LLMError, match=stop):
        c.complete("s", "p")
    assert list(tmp_path.glob("*.json")) == []
    assert c.usage.calls == 1  # it was still paid for


# ---- orchestrator end to end (W8) ---------------------------------------------

def _continuous_spec():
    s = _ratings_spec()
    s.update({"theory_id": "poverty_margin", "outcome": "operating_margin", "exposure": "poverty_rate",
              "exposure_type": "continuous", "expected_sign": -1})
    s["hypotheses"][0]["method"] = "correlation"
    s["population"]["require_present"] = ["operating_margin", "poverty_rate", "log_revenue",
                                          "category", "county_fips"]
    s["robustness"] = []
    return s


def test_orchestrator_end_to_end_loads_every_theorys_columns(tmp_path, monkeypatch):
    panel = _panel(penalty=-1.0)
    rng = np.random.default_rng(1)
    pov = {c: rng.uniform(0.05, 0.3) for c in panel["county_fips"].unique()}
    panel["poverty_rate"] = panel["county_fips"].map(pov)
    panel["operating_margin"] = rng.normal(0, 0.1, len(panel))
    asked = {}

    def fake_load_panel(columns=None, **_):
        asked["columns"] = set(columns)
        return panel

    monkeypatch.setattr(orch_mod, "load_panel", fake_load_panel)
    ratings = _ratings_spec()
    ratings["robustness"] = []
    cont = _continuous_spec()
    replies = []
    for spec in (ratings, cont):
        mapping = {k: spec[k] for k in ("outcome", "exposure", "exposure_unit")}
        mapping.update(exposure_type=spec.get("exposure_type", "binary"), controls=["log_revenue"])
        replies += [json.dumps(mapping), json.dumps(spec)]
    llm = _client(tmp_path / "cache", replies)
    orch = Orchestrator(llm, panel=None, n_permutations=19)
    summary = orch.run_theories(["ratings text", "poverty text"], out_dir=tmp_path / "runs")
    assert summary["plan_failures"] == [] and summary["run_errors"] == []
    assert {"standard_rating", "high_need", "operating_margin", "poverty_rate", "state"} <= asked["columns"]
    verdicts = {t["theory_id"]: t["verdict"] for t in summary["theories"]}
    assert set(verdicts) == {"ratings_fairness", "poverty_margin"}
    assert verdicts["poverty_margin"] == "unsupported"  # pure noise
    assert (tmp_path / "runs" / "poverty_margin.spec.json").exists()
    assert llm.usage.calls == 4
