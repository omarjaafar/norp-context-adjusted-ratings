"""Gate rules for many theories (CP4).

1. One Benjamini-Hochberg family over every primary hypothesis (H1 and H2) of every
   theory in a run, so testing 20 theories costs the same multiplicity as testing one
   family of 40 hypotheses. Within-theory q-values from run_spec are replaced.
2. "reversed" is held to the standard of "supported": the opposite-signed effect must
   hold for H1 and H2 (family q, effect floor) and replicate in the held-out states
   with the opposite sign in every fold and a significant pooled estimate.
Verdicts are recomputed from numbers already in each report; nothing is re-estimated.
"""

from __future__ import annotations

import copy

import numpy as np

from theorylab.validator import GateInput, benjamini_hochberg, decide_verdict


def opposite_replication(replication: dict, expected_sign: int, alpha: float) -> bool:
    """True when every held-out fold has the opposite sign and the pooled estimate is significant."""
    folds = replication["folds"]
    signs_ok = all(int(np.sign(f["test"]["effect"])) == -expected_sign
                   and int(np.sign(f["train"]["effect"])) == -expected_sign for f in folds)
    p = replication.get("pooled_p_value")
    pooled = replication.get("pooled_effect")
    return bool(signs_ok and p is not None and p < alpha and pooled is not None
                and int(np.sign(pooled)) == -expected_sign)


def apply_family(reports: list, expected_signs: dict, alpha: float = 0.05, floor: float | None = None) -> list:
    """Returns copies of the reports with family q-values and re-decided verdicts.

    expected_signs: {theory_id: -1 or 1}. floor: effect floor in SD; when None each
    report's own validator.effect_floor_sd is used.
    """
    p_values, index = [], []
    for i, r in enumerate(reports):
        for h in ("H1", "H2"):
            p_values.append(r[h]["permutation"]["p_value"])
            index.append((i, h))
    q, _ = benjamini_hochberg(p_values, alpha)
    out = [copy.deepcopy(r) for r in reports]
    for (i, h), qv in zip(index, q):
        out[i][h]["family_q_value"] = float(qv)
    for r in out:
        sign = int(expected_signs[r["theory_id"]])
        fl = float(floor if floor is not None else r["validator"]["effect_floor_sd"])
        g1 = GateInput(r["H1"]["effect"], r["H1"]["standardized_effect"], r["H1"]["family_q_value"])
        g2 = GateInput(r["H2"]["fit"]["effect"], r["H2"]["fit"]["standardized_effect"],
                       r["H2"]["family_q_value"])
        replicated = bool(r["replication"]["replicated"])
        d = decide_verdict(g1, g2, expected_sign=sign, alpha=alpha, floor=fl, replicated=replicated)
        if d["verdict"] == "reversed":
            solid = (g2.holds(-sign, alpha, fl) and opposite_replication(r["replication"], sign, alpha))
            if not solid:
                d = {"verdict": "weak",
                     "reasons": ["H1 is significant in the opposite direction, but H2 or the held-out "
                                 "replication does not hold in that direction, so it is not called reversed"]}
        r["verdict_within_theory"] = r["verdict"]
        r["verdict"], r["reasons"] = d["verdict"], d["reasons"]
        r["validator"]["fdr_family"] = f"all primary hypotheses in the run ({len(p_values)})"
    return out
