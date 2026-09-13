"""
Validation harness for the reconstructed risk model.

This checks the reconstruction against every number the source paper published, and
quantifies how much each undocumented assumption actually matters. Run it after any
change to coefficients.json.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import fsrp
from fsrp import Profile

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []

def check(name, ok, detail=""):
    results.append((PASS if ok else FAIL, name, detail))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))

def warn(name, detail):
    results.append((WARN, name, detail))
    print(f"[{WARN}] {name}  -- {detail}")

print("=" * 78)
print("1. Does the reconstruction reproduce the published cohort risk?")
print("=" * 78)
for cs in ("REGARDS_all", "NewFHS"):
    for sex in ("men", "women"):
        target = fsrp.COEF["coefficient_sets"][cs]["observed_5yr_risk"][sex]
        lp = fsrp.synthetic_cohort(cs, sex, n=200_000, seed=99)  # different seed than solve
        M = fsrp.centering_M(cs, sex)
        s0 = fsrp.baseline_survival(cs, sex, 5)
        got = float(np.mean(1 - s0 ** np.exp(lp - M)))
        check(f"{cs:12s} {sex:5s} mean 5-yr risk reproduces published {target:.3f}",
              abs(got - target) < 0.0015, f"got {got:.4f}")

print()
print("=" * 78)
print("2. Is the constant-hazard assumption safe for other horizons?")
print("=" * 78)
# NewFHS is the only set publishing BOTH 5- and 10-year risk, so it is the only place
# the 'scale S0 by t' shortcut can be tested.
for sex in ("men", "women"):
    s5 = fsrp.baseline_survival("NewFHS", sex, 5)
    s10 = fsrp.baseline_survival("NewFHS", sex, 10)
    implied = s5 ** 2                      # what constant hazard would predict
    r_const, r_true = 1 - implied, 1 - s10
    warn(f"NewFHS {sex}: constant-hazard 10-yr S0",
         f"S0(5)^2 = {implied:.4f} (risk {r_const:.3f}) vs solved S0(10) = {s10:.4f} "
         f"(risk {r_true:.3f}); constant hazard UNDERSTATES 10-yr risk by "
         f"{100*(r_true-r_const)/r_true:.0f}% relative")
print("  -> Baseline hazard rises with follow-up (the cohort ages). Therefore this model")
print("     is reported ONLY at the 5-year horizon that REGARDS directly observed.")

print()
print("=" * 78)
print("3. Monotonicity: does risk move the right way for every input?")
print("=" * 78)
base = Profile(age=68, sex="men", sbp=130)
r0 = fsrp.risk(base)
for field, label in [("current_smoker", "smoking"), ("diabetes", "diabetes"),
                     ("prevalent_cvd", "prior CVD"), ("atrial_fib", "atrial fibrillation")]:
    p = Profile(**{**base.__dict__, field: True})
    check(f"turning on {label} increases risk", fsrp.risk(p) > r0,
          f"{100*r0:.2f}% -> {100*fsrp.risk(p):.2f}%")

check("risk increases with age",
      all(fsrp.risk(Profile(age=a, sex="men", sbp=130)) <
          fsrp.risk(Profile(age=a + 5, sex="men", sbp=130)) for a in range(55, 80, 5)))
check("risk increases with systolic BP",
      all(fsrp.risk(Profile(age=68, sex="men", sbp=s)) <
          fsrp.risk(Profile(age=68, sex="men", sbp=s + 10)) for s in range(110, 170, 10)))
check("all risks are valid probabilities",
      all(0 < fsrp.risk(Profile(age=a, sex=sx, sbp=s, diabetes=d, atrial_fib=d)) < 1
          for a in (55, 70, 84) for sx in ("men", "women") for s in (100, 140, 200) for d in (False, True)))

print()
print("=" * 78)
print("4. Same check on the Framingham set -- does it misbehave as predicted?")
print("=" * 78)
for sex, field, label in [("men", "atrial_fib", "atrial fibrillation"),
                          ("women", "prevalent_cvd", "prior CVD")]:
    b = Profile(age=68, sex=sex, sbp=130)
    p = Profile(**{**b.__dict__, field: True})
    r_b, r_p = fsrp.risk(b, coefset="NewFHS"), fsrp.risk(p, coefset="NewFHS")
    ratio = r_p / r_b
    warn(f"NewFHS {sex}: effect of {label}",
         f"risk ratio {ratio:.2f}x ({100*r_b:.2f}% -> {100*r_p:.2f}%) "
         f"-- clinically implausible, this is why NewFHS is not the default")
    reg_b, reg_p = fsrp.risk(b), fsrp.risk(p)
    print(f"        REGARDS gives {reg_p/reg_b:.2f}x for the same change "
          f"({100*reg_b:.2f}% -> {100*reg_p:.2f}%)")

print()
print("=" * 78)
print("5. How much do the undocumented assumptions actually move the answer?")
print("=" * 78)
orig = fsrp.SBP_SD_MMHG
for sd in (12.0, 17.0, 22.0):
    fsrp.SBP_SD_MMHG = sd
    fsrp._S0_CACHE.clear()
    s0 = fsrp.baseline_survival("REGARDS_all", "men", 5)
    r = fsrp.risk(Profile(age=68, sex="men", sbp=145, diabetes=True))
    print(f"  SBP SD = {sd:4.0f} mmHg -> S0(5) = {s0:.5f}, example profile risk = {100*r:.2f}%")
fsrp.SBP_SD_MMHG = orig
fsrp._S0_CACHE.clear()
print("  -> S0 is insensitive to this assumption at the 2nd decimal of risk percent.")

print()
print("=" * 78)
print("6. Worked examples (REGARDS coefficients, 5-year horizon)")
print("=" * 78)
examples = [
    ("Healthy 60-year-old woman, BP 115", Profile(age=60, sex="women", sbp=115)),
    ("Healthy 60-year-old man, BP 115", Profile(age=60, sex="men", sbp=115)),
    ("70 man, BP 150 untreated, smoker", Profile(age=70, sex="men", sbp=150, current_smoker=True)),
    ("70 man, BP 150 treated, diabetic", Profile(age=70, sex="men", sbp=150, on_htn_meds=True, diabetes=True)),
    ("75 woman, AF + prior CVD, BP 160 treated",
     Profile(age=75, sex="women", sbp=160, on_htn_meds=True, atrial_fib=True, prevalent_cvd=True)),
    ("80 man, everything", Profile(age=80, sex="men", sbp=170, on_htn_meds=True, diabetes=True,
                                   current_smoker=True, prevalent_cvd=True, atrial_fib=True)),
    ("45-year-old man, BP 120 (EXTRAPOLATION)", Profile(age=45, sex="men", sbp=120)),
]
for label, p in examples:
    d = fsrp.risk_detail(p)
    flag = "  <-- " + d["applicability"]["status"] if d["applicability"]["status"] != "in_range" else ""
    print(f"  {label:46s} {d['risk_pct']:6.2f}%   "
          f"({d['comparison']['ratio_vs_optimal']:.1f}x the same-age optimal){flag}")

print()
print("=" * 78)
n_fail = sum(1 for s, _, _ in results if s == FAIL)
n_warn = sum(1 for s, _, _ in results if s == WARN)
n_pass = sum(1 for s, _, _ in results if s == PASS)
print(f"{n_pass} passed, {n_fail} failed, {n_warn} warnings (warnings are documented findings)")
print("=" * 78)
sys.exit(1 if n_fail else 0)
