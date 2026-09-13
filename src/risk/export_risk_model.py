"""
Export everything the browser needs: betas, centering constants M, baseline survival
S0(5), and the population priors used for uncertainty propagation.

The heavy work (solving S0 against a synthetic cohort) happens here, in Python, once.
The browser only evaluates closed-form arithmetic, so the client stays small and fast.

Also emits a fixture of Python-computed risks that tests/test_risk_parity.js replays,
so the JS port cannot drift from the Python without a test failing.
"""
import json, os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsrp
from fsrp import Profile
from uncertainty import _priors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "risk_model_export.json")
FIXTURE = os.path.join(HERE, "..", "..", "tests", "risk_parity_fixture.json")

SETS = ["REGARDS_all", "NewFHS"]

export = {
    "_generated_by": "src/risk/export_risk_model.py",
    "citation": fsrp.COEF["citation"],
    "applicability": fsrp.COEF["applicability"],
    "known_limitations": fsrp.COEF["known_limitations"],
    "horizon_years": 5,
    "default_set": "REGARDS_all",
    "adjunct_terms": fsrp.COEF.get("adjunct_terms", {}),
    "population_priors": fsrp.COEF.get("population_priors", {}),
    "sets": {},
}

for cs in SETS:
    entry = {"label": fsrp.COEF["coefficient_sets"][cs]["label"], "sexes": {}}
    if "warning" in fsrp.COEF["coefficient_sets"][cs]:
        entry["warning"] = fsrp.COEF["coefficient_sets"][cs]["warning"]
    for sex in ("men", "women"):
        entry["sexes"][sex] = {
            "betas": fsrp._betas(cs, sex),
            "M": fsrp.centering_M(cs, sex),
            "S0_5yr": fsrp.baseline_survival(cs, sex, 5),
            "cohort_avg_5yr_risk": fsrp.COEF["coefficient_sets"][cs]["observed_5yr_risk"][sex],
            "priors": _priors(sex, cs),
        }
    export["sets"][cs] = entry

with open(OUT, "w") as f:
    json.dump(export, f, indent=2)
print(f"wrote {OUT}  ({os.path.getsize(OUT)} bytes)")

# ---- parity fixture: exhaustive-ish grid of profiles with exact Python risks ----
cases = []
for sex in ("men", "women"):
    for age in (45, 55, 62, 68, 75, 84, 90):
        for sbp in (105, 120, 135, 150, 175, 200):
            for meds, dm, smk, cvd, af, slt in itertools.product((False, True), repeat=6):
                if len(cases) % 7 != 0:      # subsample the 2^5 grid for file size
                    cases.append(None)
                    cases.pop()
                p = Profile(age=age, sex=sex, sbp=sbp, on_htn_meds=meds, diabetes=dm,
                            current_smoker=smk, prevalent_cvd=cvd, atrial_fib=af,
                            smokeless_tobacco=slt)
                for cs in SETS:
                    cases.append({
                        "set": cs, "age": age, "sex": sex, "sbp": sbp,
                        "on_htn_meds": meds, "diabetes": dm, "current_smoker": smk,
                        "prevalent_cvd": cvd, "atrial_fib": af, "smokeless_tobacco": slt,
                        "expected_risk": fsrp.risk(p, coefset=cs),
                    })

os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
with open(FIXTURE, "w") as f:
    json.dump({"_note": "Python-computed risks. tests/test_risk_parity.js must reproduce "
                        "every one of these to 1e-12.", "cases": cases}, f)
print(f"wrote {FIXTURE}  ({len(cases)} cases, {os.path.getsize(FIXTURE)} bytes)")
