"""
Uncertainty propagation and value-of-information.

The problem this solves: systolic blood pressure is the input most people cannot
report, and it carries a large share of the model's weight. The tempting fix is to
substitute a population average and print a single confident number. That is a
fabrication -- it reports a made-up input as though it were measured.

Instead, an unknown input is carried as a distribution and propagated through the
model by Monte Carlo, so the output is an interval. We then compute, for each unknown,
how much the interval would narrow if the user went and measured it. That turns "I
don't know my blood pressure" from a silent guess into an actionable instruction.

Priors are the REGARDS cohort marginals already cited in coefficients.json -- the same
reference population the coefficients came from, so nothing new is introduced.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrp
from fsrp import Profile, COEF

UNKNOWN = None

# Fields a user may legitimately not know, in the order we'd ask them to go find out.
UNCERTAIN_FIELDS = ["sbp", "on_htn_meds", "diabetes", "atrial_fib", "prevalent_cvd",
                    "current_smoker", "smokeless_tobacco"]

HUMAN = {
    "sbp": "your systolic blood pressure",
    "on_htn_meds": "whether you take blood-pressure medication",
    "diabetes": "whether you have diabetes",
    "atrial_fib": "whether you have atrial fibrillation",
    "prevalent_cvd": "whether you have existing heart or vascular disease",
    "current_smoker": "whether you currently smoke",
    "smokeless_tobacco": "whether you chew or hold tobacco (gutkha, khaini, zarda, paan masala)",
}

HOW_TO_FIND_OUT = {
    "sbp": "A pharmacy BP machine, a clinic, or a home cuff. Take it twice, seated, "
           "and use the average. This is the single most useful number you can bring.",
    "on_htn_meds": "Check your prescriptions for amlodipine, lisinopril, losartan, "
                   "ramipril, telmisartan, a thiazide, or similar.",
    "diabetes": "A fasting glucose or HbA1c test. If you have never been tested and are "
                "over 55, this is worth asking for regardless of this tool.",
    "atrial_fib": "Requires an ECG. If you have had palpitations or an irregular pulse, "
                  "ask a doctor - untreated AF is one of the most treatable stroke risks.",
    "prevalent_cvd": "A prior heart attack, angina, bypass, stent, heart failure, or "
                     "peripheral artery disease all count.",
    "current_smoker": "Any smoking in the past year counts.",
    "smokeless_tobacco": "Gutkha, khaini, zarda, paan masala, mishri, or betel quid (paan) with tobacco. Chewed or held in the mouth rather than smoked.",
}


@dataclass
class PartialProfile:
    """A profile where any uncertain field may be None, meaning 'not known'."""
    age: float
    sex: str
    sbp: Optional[float] = UNKNOWN
    on_htn_meds: Optional[bool] = UNKNOWN
    diabetes: Optional[bool] = UNKNOWN
    current_smoker: Optional[bool] = UNKNOWN
    prevalent_cvd: Optional[bool] = UNKNOWN
    atrial_fib: Optional[bool] = UNKNOWN
    smokeless_tobacco: Optional[bool] = UNKNOWN

    def unknowns(self) -> list[str]:
        return [f for f in UNCERTAIN_FIELDS if getattr(self, f) is UNKNOWN]


def _priors(sex: str, coefset: str) -> dict:
    """Population priors, taken from the cohort marginals in coefficients.json."""
    m = COEF["coefficient_sets"][coefset]["means"][sex]
    adj = COEF.get("adjunct_terms", {}).get("smokeless_tobacco", {})
    return {
        # No US cohort prevalence exists for this; the Indian survey figure is used as
        # the prior because it is the population the question was added for.
        "smokeless_tobacco": adj.get("means", {}).get(sex, 0.0),
        "on_htn_meds": m["htn_treatment"],
        "diabetes": m["dm_overall"],
        "current_smoker": m["current_smoking"],
        "prevalent_cvd": m["prevalent_cvd"],
        "atrial_fib": m["atrial_fib"],
        "sbp_untreated_mean": m["sbp_untreated_mmHg"],
        "sbp_treated_mean": m["sbp_treated_mmHg"],
        "sbp_sd": fsrp.SBP_SD_MMHG,
    }


def _sample(pp: PartialProfile, n: int, rng, coefset: str) -> np.ndarray:
    """Draw n completed profiles and return the resulting risks."""
    pr = _priors(pp.sex, coefset)
    b = fsrp._betas(coefset, pp.sex)
    M = fsrp.centering_M(coefset, pp.sex)
    s0 = fsrp.baseline_survival(coefset, pp.sex, 5)

    age = np.full(n, float(pp.age))
    ge65 = age >= 65

    def draw_bool(field):
        v = getattr(pp, field)
        return np.full(n, bool(v)) if v is not UNKNOWN else rng.random(n) < pr[field]

    treated = draw_bool("on_htn_meds")
    dm = draw_bool("diabetes")
    smoke = draw_bool("current_smoker")
    cvd = draw_bool("prevalent_cvd")
    af = draw_bool("atrial_fib")
    slt = draw_bool("smokeless_tobacco")

    if pp.sbp is not UNKNOWN:
        sbp = np.full(n, float(pp.sbp))
    else:
        mean = np.where(treated, pr["sbp_treated_mean"], pr["sbp_untreated_mean"])
        sbp = np.clip(rng.normal(mean, pr["sbp_sd"], n), 80, 240)

    lp = b["age"] * age + b["age_ge65"] * ge65
    lp += b["current_smoking"] * smoke + b["prevalent_cvd"] * cvd + b["atrial_fib"] * af
    lp += np.where(ge65, b["dm_age_ge65"], b["dm_age_lt65"]) * dm
    lp += np.where(treated, b["htn_treatment"] + b["sbp_treated"] * sbp,
                   b["sbp_untreated"] * sbp)
    adj = COEF.get("adjunct_terms", {}).get("smokeless_tobacco")
    if adj is not None:
        lp = lp + np.log(adj["hazard_ratio"]) * slt
    return 1.0 - s0 ** np.exp(lp - M)


def estimate(pp: PartialProfile, n: int = 60_000, seed: int = 11,
             coefset: str = "REGARDS_all") -> dict:
    """
    Risk as a point estimate plus an interval reflecting what the user doesn't know.

    With nothing unknown the interval collapses to a point, and the number is the
    plain model output. With unknowns it widens honestly.
    """
    rng = np.random.default_rng(seed)
    risks = _sample(pp, n, rng, coefset)
    lo, med, hi = np.percentile(risks, [10, 50, 90])
    unknowns = pp.unknowns()

    out = {
        "point_pct": 100 * float(med),
        "low_pct": 100 * float(lo),
        "high_pct": 100 * float(hi),
        "interval_width_pct": 100 * float(hi - lo),
        "is_certain": not unknowns,
        "unknown_fields": unknowns,
        "horizon_years": 5,
        "coefficient_set": coefset,
    }

    if unknowns:
        out["value_of_information"] = value_of_information(pp, n=n // 3, seed=seed + 1,
                                                           coefset=coefset)
    return out


def value_of_information(pp: PartialProfile, n: int = 40_000, seed: int = 12,
                         coefset: str = "REGARDS_all") -> list[dict]:
    """
    For each unknown input: what fraction of the uncertainty would measuring it remove?

    This is the first-order Sobol sensitivity index -- the share of the output variance
    attributable to that one input:

        S_j = [ Var(R) - E_{x_j}[ Var(R | X_j = x_j) ] ] / Var(R)

    Variance-based, so it is provably non-negative and bounded by 1. An earlier version
    of this ranked by 10-90 interval width averaged over a hand-picked grid of candidate
    values; that is not a proper expectation over the prior and could return a negative
    "narrowing", which is meaningless. Widths are still reported for display, but the
    ranking uses the variance index.
    """
    rng = np.random.default_rng(seed)
    pr = _priors(pp.sex, coefset)
    base = _sample(pp, n, rng, coefset)
    base_var = float(np.var(base))
    base_w = float(np.percentile(base, 90) - np.percentile(base, 10))

    rows = []
    for field in pp.unknowns():
        if field == "sbp":
            # Monte-Carlo the conditional variance over SBP drawn from its actual prior.
            treated_known = pp.on_htn_meds is not UNKNOWN
            g = np.random.default_rng(seed + 17)
            if treated_known:
                centre = pr["sbp_treated_mean"] if pp.on_htn_meds else pr["sbp_untreated_mean"]
            else:
                centre = (pr["on_htn_meds"] * pr["sbp_treated_mean"]
                          + (1 - pr["on_htn_meds"]) * pr["sbp_untreated_mean"])
            draws = np.clip(g.normal(centre, pr["sbp_sd"], 24), 80, 240)
            cond_vars, cond_ws = [], []
            for c in draws:
                r = _sample(replace(pp, sbp=float(c)), max(4000, n // 10),
                            np.random.default_rng(seed + 23), coefset)
                cond_vars.append(np.var(r))
                cond_ws.append(np.percentile(r, 90) - np.percentile(r, 10))
            exp_var, exp_w = float(np.mean(cond_vars)), float(np.mean(cond_ws))
        else:
            p_true = pr[field]
            r_t = _sample(replace(pp, **{field: True}), n, np.random.default_rng(seed + 4), coefset)
            r_f = _sample(replace(pp, **{field: False}), n, np.random.default_rng(seed + 5), coefset)
            exp_var = float(p_true * np.var(r_t) + (1 - p_true) * np.var(r_f))
            exp_w = float(p_true * (np.percentile(r_t, 90) - np.percentile(r_t, 10))
                          + (1 - p_true) * (np.percentile(r_f, 90) - np.percentile(r_f, 10)))

        sobol = max(0.0, (base_var - exp_var) / base_var) if base_var > 0 else 0.0
        rows.append({
            "field": field,
            "label": HUMAN[field],
            "how_to_find_out": HOW_TO_FIND_OUT[field],
            "variance_share": sobol,            # ranking quantity, in [0, 1]
            "current_width_pct": 100 * base_w,
            "expected_width_pct": 100 * exp_w,
            "narrowing_pct_points": 100 * max(0.0, base_w - exp_w),
        })

    rows.sort(key=lambda r: -r["variance_share"])
    return rows


# --------------------------------------------------------------------------------
# Counterfactuals: what would actually change this number
# --------------------------------------------------------------------------------
# Deliberately excluded: "start blood-pressure medication". At a FIXED measured SBP the
# fitted model assigns treated people LOWER risk than untreated people (an artifact of
# the flatter SBP-risk slope observed in treated cohorts, not a causal treatment effect).
# Presenting that as a lever would tell users a pill beats lowering their BP, which the
# data does not support. The honest lever is the SBP value itself.
def counterfactuals(p: Profile, coefset: str = "REGARDS_all") -> list[dict]:
    """Modifiable changes, with the risk each would remove. Ordered by benefit."""
    current = fsrp.risk(p, coefset=coefset)
    out = []

    def add(label, newp, caveat=None):
        r = fsrp.risk(newp, coefset=coefset)
        if r < current - 1e-12:
            out.append({
                "change": label,
                "new_risk_pct": 100 * r,
                "absolute_reduction_pct_points": 100 * (current - r),
                "relative_reduction": (current - r) / current,
                "caveat": caveat,
            })

    if p.sbp > 120:
        for target in (140, 130, 120):
            if p.sbp > target:
                add(f"Lower systolic BP from {p.sbp:.0f} to {target} mmHg",
                    replace(p, sbp=float(target)),
                    "Achieved through treatment, salt reduction, weight loss and exercise "
                    "- decided with a doctor, not by this tool.")
    if p.smokeless_tobacco:
        add("Stop chewing tobacco", replace(p, smokeless_tobacco=False),
            "Modelled as the contrast with a non-user, using a pooled hazard ratio from "
            "a meta-analysis of chewing-tobacco users.")
    if p.current_smoker:
        add("Stop smoking", replace(p, current_smoker=False),
            "The model treats smoking as current/not-current, so this shows the "
            "contrast with a non-smoker rather than a predicted trajectory after quitting.")
    if p.atrial_fib:
        out.append({
            "change": "Atrial fibrillation: ask about anticoagulation",
            "new_risk_pct": None,
            "absolute_reduction_pct_points": None,
            "relative_reduction": None,
            "caveat": "AF is in this model as a risk factor, so it cannot be 'switched off'. "
                      "But AF-related stroke is substantially preventable with appropriate "
                      "anticoagulation. This is a conversation to have with a doctor now.",
        })

    out.sort(key=lambda r: -(r["relative_reduction"] or 0))
    return {"current_risk_pct": 100 * current, "options": out}
