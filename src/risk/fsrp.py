"""
Absolute 5-year stroke risk from a Cox proportional-hazards model.

    risk(t | x) = 1 - S0(t) ** exp(LP(x) - M)

Coefficients come from published cohort tables (see coefficients.json, which carries
the full citation and a per-parameter source string). Nothing in this module invents
a number: every constant is either read from that file or *solved* from it by a
documented procedure.

The one genuinely derived quantity is S0(t), the baseline survival. The source papers
publish Kaplan-Meier cohort risk but not S0 directly. We recover it by building a
synthetic cohort that matches the published covariate marginals and solving for the
S0 that reproduces the published mean risk. Reading S0 off the cohort mean directly
(the naive S0 = 1 - KM) is biased, because risk is convex in the linear predictor, so
mean-of-risk exceeds risk-at-mean (Jensen). See solve_baseline_survival().
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import truncnorm, norm

HERE = os.path.dirname(os.path.abspath(__file__))
COEF_PATH = os.path.join(HERE, "coefficients.json")

with open(COEF_PATH) as _f:
    COEF = json.load(_f)

Sex = Literal["men", "women"]

# Assumptions that are NOT published and are needed to build the synthetic cohort.
# They affect only the Jensen correction inside solve_baseline_survival(), which is a
# second-order effect; sensitivity to them is quantified in validate.py.
SBP_SD_MMHG = 17.0        # within-treatment-group SD of systolic BP in adults 55+
AGE_MAX = 90.0
AGE_MIN_COHORT = 55.0


# --------------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------------
@dataclass
class Profile:
    """One person's inputs. Every field is self-reportable except sbp."""
    age: float
    sex: Sex
    sbp: float                    # systolic blood pressure, mmHg
    on_htn_meds: bool = False     # currently taking blood-pressure medication
    diabetes: bool = False
    current_smoker: bool = False  # smoked in the past year
    prevalent_cvd: bool = False   # prior MI / CHD / heart failure / peripheral vascular disease
    atrial_fib: bool = False
    # Adjunct factor, not in the source cohort model. See coefficients.json
    # -> adjunct_terms.smokeless_tobacco for the hazard ratio, source and caveats.
    smokeless_tobacco: bool = False

    def __post_init__(self):
        if self.sex not in ("men", "women"):
            raise ValueError(f"sex must be 'men' or 'women', got {self.sex!r}")
        if not (18 <= self.age <= 110):
            raise ValueError(f"age out of range: {self.age}")
        if not (70 <= self.sbp <= 260):
            raise ValueError(f"systolic BP out of plausible range: {self.sbp}")


# --------------------------------------------------------------------------------
# Coefficients
# --------------------------------------------------------------------------------
def _betas(coefset: str, sex: Sex) -> dict[str, float]:
    """beta = ln(HR), rescaled to per-unit for the continuous terms."""
    cs = COEF["coefficient_sets"][coefset]
    hrs = cs["hazard_ratios"][sex]
    age_per = COEF["units"]["age"]["per"]
    sbp_per = COEF["units"]["sbp"]["per"]

    b = {k: math.log(v["hr"]) for k, v in hrs.items()}
    b["age"] /= age_per              # per year
    b["sbp_untreated"] /= sbp_per    # per mmHg
    b["sbp_treated"] /= sbp_per
    return b


def linear_predictor(p: Profile, coefset: str = "REGARDS_all") -> float:
    """LP(x) = sum_j beta_j x_j, using the interaction structure the paper specifies."""
    b = _betas(coefset, p.sex)
    lp = b["age"] * p.age
    if p.age >= 65:
        lp += b["age_ge65"]
    if p.current_smoker:
        lp += b["current_smoking"]
    if p.prevalent_cvd:
        lp += b["prevalent_cvd"]
    if p.atrial_fib:
        lp += b["atrial_fib"]
    if p.diabetes:
        # exactly one of the two, per the age x diabetes interaction
        lp += b["dm_age_ge65"] if p.age >= 65 else b["dm_age_lt65"]
    if p.on_htn_meds:
        lp += b["htn_treatment"] + b["sbp_treated"] * p.sbp
    else:
        lp += b["sbp_untreated"] * p.sbp
    lp += adjunct_lp(p)
    return lp


def adjunct_lp(p: Profile) -> float:
    """
    Extra terms for risk factors absent from the source cohort model.

    Centred at zero rather than at population prevalence: the base cohort is the
    reference S0 was calibrated against, and smokeless tobacco use was negligible in
    it, so non-use already IS the reference. A user gets +ln(HR); a non-user gets
    nothing. See coefficients.json -> adjunct_terms for the full rationale.
    """
    lp = 0.0
    adj = COEF.get("adjunct_terms", {})
    if p.smokeless_tobacco and "smokeless_tobacco" in adj:
        lp += math.log(adj["smokeless_tobacco"]["hazard_ratio"])
    return lp


def centering_M(coefset: str = "REGARDS_all", sex: Sex = "men") -> float:
    """
    M = sum_j beta_j * xbar_j, the linear predictor at the cohort covariate mean.

    Published tables give only marginal means, so the conditional terms are expanded
    using the published conditional prevalences where available:
        E[1(dm & age>=65)] = P(age>=65) * P(dm | age>=65)
        E[sbp * 1(treated)] = P(treated) * E[sbp | treated]
    """
    b = _betas(coefset, sex)
    m = COEF["coefficient_sets"][coefset]["means"][sex]

    p65 = m["age_ge65"]
    M = b["age"] * m["age_years"]
    M += b["age_ge65"] * p65
    M += b["current_smoking"] * m["current_smoking"]
    M += b["prevalent_cvd"] * m["prevalent_cvd"]
    M += b["atrial_fib"] * m["atrial_fib"]
    M += b["dm_age_lt65"] * (1 - p65) * m["dm_if_lt65"]
    M += b["dm_age_ge65"] * p65 * m["dm_if_ge65"]

    ptx = m["htn_treatment"]
    M += b["htn_treatment"] * ptx
    M += b["sbp_treated"] * ptx * m["sbp_treated_mmHg"]
    M += b["sbp_untreated"] * (1 - ptx) * m["sbp_untreated_mmHg"]
    # Adjunct terms are deliberately NOT added here: they are centred at zero, so the
    # cohort-mean person (a non-user) contributes nothing and S0 stays valid as solved.
    return M


# --------------------------------------------------------------------------------
# Synthetic cohort, used only to calibrate S0
# --------------------------------------------------------------------------------
def _age_distribution(mean_age: float, p_ge65: float):
    """
    Solve a truncated normal on [55, 90] whose mean and P(age >= 65) match the
    published marginals. Two equations, two unknowns (mu, sigma).
    """
    def moments(params):
        mu, sigma = params
        a, b = (AGE_MIN_COHORT - mu) / sigma, (AGE_MAX - mu) / sigma
        d = truncnorm(a, b, loc=mu, scale=sigma)
        return d.mean() - mean_age, d.sf(65.0) - p_ge65

    # coarse-to-fine grid search then polish; robust and dependency-light
    best, best_err = None, float("inf")
    for mu in np.arange(55.0, 80.0, 0.5):
        for sigma in np.arange(3.0, 18.0, 0.5):
            e1, e2 = moments((mu, sigma))
            err = e1 * e1 + (e2 * 100) ** 2
            if err < best_err:
                best, best_err = (mu, sigma), err
    mu, sigma = best
    for _ in range(60):  # local refinement
        improved = False
        for dmu in (-0.05, 0, 0.05):
            for dsg in (-0.05, 0, 0.05):
                cand = (mu + dmu, max(1.0, sigma + dsg))
                e1, e2 = moments(cand)
                err = e1 * e1 + (e2 * 100) ** 2
                if err < best_err - 1e-12:
                    mu, sigma, best_err = cand[0], cand[1], err
                    improved = True
        if not improved:
            break
    a, b = (AGE_MIN_COHORT - mu) / sigma, (AGE_MAX - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


def synthetic_cohort(coefset: str, sex: Sex, n: int = 200_000, seed: int = 7):
    """
    Draw a cohort matching the published marginals.

    ASSUMPTION: risk factors are drawn independently given age group. The papers
    publish marginals only, so the true covariance is unavailable. This affects the
    spread of LP and therefore the size of the Jensen correction in S0; validate.py
    quantifies how much S0 moves under plausible alternatives.
    """
    rng = np.random.default_rng(seed)
    m = COEF["coefficient_sets"][coefset]["means"][sex]
    b = _betas(coefset, sex)

    age = _age_distribution(m["age_years"], m["age_ge65"]).rvs(size=n, random_state=rng)
    ge65 = age >= 65

    smoke = rng.random(n) < m["current_smoking"]
    cvd = rng.random(n) < m["prevalent_cvd"]
    af = rng.random(n) < m["atrial_fib"]
    treated = rng.random(n) < m["htn_treatment"]

    # diabetes prevalence depends on age group (the model's only interaction)
    u = rng.random(n)
    dm = np.where(ge65, u < m["dm_if_ge65"], u < m["dm_if_lt65"])

    sbp_mean = np.where(treated, m["sbp_treated_mmHg"], m["sbp_untreated_mmHg"])
    sbp = np.clip(rng.normal(sbp_mean, SBP_SD_MMHG, n), 80, 240)

    lp = b["age"] * age + b["age_ge65"] * ge65
    lp += b["current_smoking"] * smoke + b["prevalent_cvd"] * cvd + b["atrial_fib"] * af
    lp += np.where(ge65, b["dm_age_ge65"], b["dm_age_lt65"]) * dm
    lp += np.where(treated,
                   b["htn_treatment"] + b["sbp_treated"] * sbp,
                   b["sbp_untreated"] * sbp)
    return lp


def solve_baseline_survival(coefset: str = "REGARDS_all", sex: Sex = "men",
                            horizon: int = 5, n: int = 200_000, seed: int = 7) -> float:
    """
    Find S0(horizon) such that the mean predicted risk over the synthetic cohort
    equals the published Kaplan-Meier risk for that cohort.
    """
    key = f"observed_{horizon}yr_risk"
    cs = COEF["coefficient_sets"][coefset]
    if key not in cs:
        raise KeyError(
            f"{coefset} publishes no {horizon}-year observed risk; "
            f"available: {[k for k in cs if k.startswith('observed_')]}"
        )
    target = cs[key][sex]
    lp = synthetic_cohort(coefset, sex, n=n, seed=seed)
    M = centering_M(coefset, sex)
    rel = np.exp(lp - M)

    def gap(s0):
        return float(np.mean(1.0 - s0 ** rel)) - target

    return brentq(gap, 1e-9, 1 - 1e-12, xtol=1e-14)


_S0_CACHE: dict[tuple, float] = {}


def baseline_survival(coefset: str = "REGARDS_all", sex: Sex = "men", horizon: int = 5) -> float:
    k = (coefset, sex, horizon)
    if k not in _S0_CACHE:
        _S0_CACHE[k] = solve_baseline_survival(coefset, sex, horizon)
    return _S0_CACHE[k]


# --------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------
def applicability(p: Profile, coefset: str = "REGARDS_all") -> dict:
    """Whether this person falls inside the range the model was actually derived on."""
    lo, hi = COEF["applicability"]["derived_age_range"]
    notes = []
    status = "in_range"
    if p.age < lo:
        status = "extrapolation"
        notes.append(
            f"Age {p.age:.0f} is below {lo}, the youngest age this model was derived on. "
            f"The estimate is an extrapolation and should be read as a rough indication "
            f"of direction, not a calibrated probability."
        )
    elif p.age > hi:
        status = "extrapolation"
        notes.append(f"Age {p.age:.0f} is above the derivation range (max {hi}).")
    if p.sbp > 180:
        notes.append("Systolic BP above 180 mmHg is a medical concern in its own right, "
                     "independent of any 5-year estimate.")
    return {"status": status, "notes": notes,
            "derived_age_range": [lo, hi],
            "requires_stroke_free": COEF["applicability"]["requires_stroke_free_at_baseline"]}


def risk(p: Profile, horizon: int = 5, coefset: str = "REGARDS_all") -> float:
    """Absolute probability of a first stroke within `horizon` years."""
    s0 = baseline_survival(coefset, p.sex, horizon)
    lp = linear_predictor(p, coefset)
    M = centering_M(coefset, p.sex)
    return float(1.0 - s0 ** math.exp(lp - M))


def risk_detail(p: Profile, horizon: int = 5, coefset: str = "REGARDS_all") -> dict:
    """Risk plus the context needed to make the number mean something."""
    r = risk(p, horizon, coefset)
    # a same-age, same-sex person with no risk factors and optimal BP
    ideal = Profile(age=p.age, sex=p.sex, sbp=110.0)
    r_ideal = risk(ideal, horizon, coefset)
    # the cohort-average person of this sex
    r_avg = COEF["coefficient_sets"][coefset][f"observed_{horizon}yr_risk"][p.sex]
    return {
        "risk": r,
        "risk_pct": 100 * r,
        "horizon_years": horizon,
        "coefficient_set": coefset,
        "comparison": {
            "same_age_sex_no_risk_factors": r_ideal,
            "ratio_vs_optimal": r / r_ideal if r_ideal > 0 else float("nan"),
            "cohort_average": r_avg,
        },
        "applicability": applicability(p, coefset),
        "profile": asdict(p),
    }
