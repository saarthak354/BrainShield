"""
Unit tests for the Python risk model API.

Complements src/risk/validate.py (which checks the reconstruction against the source
paper's published numbers) and tests/test_risk_parity.js (which checks the JS port
matches Python). This file covers the API surface: input validation, error paths, and
the invariants a caller is entitled to rely on.

    python3 -m pytest tests/test_risk_model.py -q
"""
import os, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "risk"))
import fsrp
from fsrp import Profile
from uncertainty import PartialProfile, estimate, counterfactuals


# ---- input validation ----
@pytest.mark.parametrize("kwargs", [
    dict(age=70, sex="M", sbp=130),            # wrong sex encoding
    dict(age=70, sex="male", sbp=130),
    dict(age=5, sex="men", sbp=130),           # implausible age
    dict(age=150, sex="men", sbp=130),
    dict(age=70, sex="men", sbp=20),           # implausible BP
    dict(age=70, sex="men", sbp=400),
])
def test_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        Profile(**kwargs)


def test_unknown_coefficient_set_raises():
    with pytest.raises(KeyError):
        fsrp.risk(Profile(age=70, sex="men", sbp=130), coefset="does_not_exist")


def test_unsupported_horizon_raises_informatively():
    # REGARDS observed only 5 years, so a 10-year number cannot be produced from it
    with pytest.raises(KeyError, match="10-year"):
        fsrp.solve_baseline_survival("REGARDS_all", "men", horizon=10)


# ---- invariants ----
def test_risk_is_a_probability():
    for age in (55, 70, 84):
        for sex in ("men", "women"):
            for sbp in (90, 130, 200):
                r = fsrp.risk(Profile(age=age, sex=sex, sbp=sbp))
                assert 0.0 < r < 1.0


def test_every_risk_factor_increases_risk():
    base = Profile(age=68, sex="men", sbp=130)
    r0 = fsrp.risk(base)
    for field in ("current_smoker", "diabetes", "prevalent_cvd", "atrial_fib"):
        p = Profile(**{**base.__dict__, field: True})
        assert fsrp.risk(p) > r0, f"{field} did not increase risk"


def test_risk_monotonic_in_age_and_bp():
    assert fsrp.risk(Profile(age=60, sex="women", sbp=130)) < \
           fsrp.risk(Profile(age=75, sex="women", sbp=130))
    assert fsrp.risk(Profile(age=70, sex="men", sbp=120)) < \
           fsrp.risk(Profile(age=70, sex="men", sbp=160))


def test_reproduces_published_cohort_average():
    """The whole calibration rests on this; if it drifts, S0 is wrong."""
    import numpy as np
    for cs in ("REGARDS_all", "NewFHS"):
        for sex in ("men", "women"):
            target = fsrp.COEF["coefficient_sets"][cs]["observed_5yr_risk"][sex]
            lp = fsrp.synthetic_cohort(cs, sex, n=100_000, seed=1234)
            s0 = fsrp.baseline_survival(cs, sex, 5)
            got = float(np.mean(1 - s0 ** np.exp(lp - fsrp.centering_M(cs, sex))))
            assert abs(got - target) < 0.002, f"{cs}/{sex}: {got:.4f} vs {target:.4f}"


# ---- uncertainty ----
def test_complete_profile_has_no_interval():
    pp = PartialProfile(age=68, sex="men", sbp=140, on_htn_meds=False, diabetes=False,
                        current_smoker=False, prevalent_cvd=False, atrial_fib=False,
                        smokeless_tobacco=False)
    e = estimate(pp, n=5000)
    assert e["is_certain"]
    assert e["interval_width_pct"] < 1e-9
    assert "value_of_information" not in e


def test_unknowns_widen_the_interval():
    e = estimate(PartialProfile(age=68, sex="men"), n=5000)
    assert not e["is_certain"]
    assert e["interval_width_pct"] > 0.1
    assert len(e["unknown_fields"]) == 7


def test_value_of_information_is_a_valid_variance_decomposition():
    e = estimate(PartialProfile(age=68, sex="men"), n=8000)
    voi = e["value_of_information"]
    assert all(0.0 <= r["variance_share"] <= 1.0 for r in voi), \
        "Sobol indices must lie in [0, 1]"
    shares = [r["variance_share"] for r in voi]
    assert shares == sorted(shares, reverse=True), "must be ranked most-informative first"


def test_knowing_an_input_never_makes_it_unknown():
    pp = PartialProfile(age=68, sex="men", sbp=150)
    assert "sbp" not in estimate(pp, n=3000)["unknown_fields"]


# ---- counterfactuals ----
def test_counterfactuals_never_recommend_starting_medication():
    """
    At a fixed measured SBP the fitted model scores treated people BELOW untreated
    people, an artifact of the flatter SBP slope in treated cohorts. Surfacing that as
    advice would tell users a pill beats lowering their blood pressure.
    """
    p = Profile(age=70, sex="men", sbp=170, diabetes=True, current_smoker=True)
    for opt in counterfactuals(p)["options"]:
        assert "medication" not in opt["change"].lower()


def test_counterfactuals_only_ever_reduce_risk():
    p = Profile(age=72, sex="women", sbp=165, current_smoker=True, diabetes=True)
    for opt in counterfactuals(p)["options"]:
        if opt["relative_reduction"] is not None:
            assert opt["relative_reduction"] > 0


def test_atrial_fibrillation_produces_a_referral_not_a_toggle():
    p = Profile(age=72, sex="women", sbp=140, atrial_fib=True)
    opts = counterfactuals(p)["options"]
    af = [o for o in opts if "fibrillation" in o["change"].lower()]
    assert af, "AF should produce guidance"
    assert af[0]["new_risk_pct"] is None, "AF must not be presented as switchable"
    assert "anticoagulation" in af[0]["caveat"].lower()


# ---- applicability ----
def test_under_55_is_flagged_as_extrapolation():
    a = fsrp.applicability(Profile(age=40, sex="men", sbp=120))
    assert a["status"] == "extrapolation"
    assert a["notes"]


def test_in_range_age_is_not_flagged():
    assert fsrp.applicability(Profile(age=68, sex="men", sbp=120))["status"] == "in_range"


def test_hypertensive_crisis_gets_its_own_note():
    a = fsrp.applicability(Profile(age=68, sex="men", sbp=195))
    assert any("180" in n for n in a["notes"])


# ---- adjunct term: smokeless tobacco ----
def test_smokeless_tobacco_raises_risk_by_the_published_hazard_ratio():
    base = Profile(age=62, sex="men", sbp=140)
    slt = Profile(age=62, sex="men", sbp=140, smokeless_tobacco=True)
    hr = fsrp.COEF["adjunct_terms"]["smokeless_tobacco"]["hazard_ratio"]
    # for small absolute risks the risk ratio approaches the hazard ratio
    assert abs(fsrp.risk(slt) / fsrp.risk(base) - hr) < 0.02


def test_smokeless_tobacco_is_weaker_than_smoking():
    """Folding chewers into the smoking question would overstate their risk by ~50%."""
    slt = Profile(age=62, sex="men", sbp=140, smokeless_tobacco=True)
    smk = Profile(age=62, sex="men", sbp=140, current_smoker=True)
    assert fsrp.risk(slt) < fsrp.risk(smk)


def test_adjunct_term_does_not_disturb_cohort_calibration():
    """
    The adjunct is centred at zero, so the cohort-mean person (a non-user) is
    unaffected and the solved S0 stays valid.
    """
    import numpy as np
    lp = fsrp.synthetic_cohort("REGARDS_all", "men", n=80_000, seed=5)
    s0 = fsrp.baseline_survival("REGARDS_all", "men", 5)
    got = float(np.mean(1 - s0 ** np.exp(lp - fsrp.centering_M("REGARDS_all", "men"))))
    assert abs(got - 0.031) < 0.002


def test_non_user_is_identical_to_before_the_term_existed():
    p_default = Profile(age=70, sex="women", sbp=150)
    p_explicit = Profile(age=70, sex="women", sbp=150, smokeless_tobacco=False)
    assert fsrp.risk(p_default) == fsrp.risk(p_explicit)


def test_chewing_tobacco_offered_as_a_counterfactual():
    p = Profile(age=60, sex="men", sbp=130, smokeless_tobacco=True)
    assert any("chewing tobacco" in o["change"].lower() for o in counterfactuals(p)["options"])
