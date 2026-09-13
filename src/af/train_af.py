"""
Phase 0, Stage A + the feasibility curve.

1. Train an RR-interval AF classifier on MIT-BIH AFDB, held out BY RECORD, so no
   patient contributes to both training and test.
2. Then degrade the RR intervals with camera-grade measurement noise and watch
   performance fall. This is the number that decides whether a phone camera can
   carry this at all -- BUT PPG later tells us where on this curve we actually sit.
"""
import os, sys, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_afdb import features, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "afdb_windows.npz"), allow_pickle=True)
X, y, groups = d["X"], d["y"], d["groups"]
names = list(d["feature_names"])
print(f"{X.shape[0]} windows / {len(set(groups))} records / {100*y.mean():.1f}% AF\n")

RULE_OUT_SENS = 0.95    # the rule-out threshold must not miss more than 5% of AF


def make_model():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0))])


# ---------------------------------------------------------- record-held-out ----
print("=" * 76)
print("Leave-one-record-out cross-validation")
print("=" * 76)
logo = LeaveOneGroupOut()
oof = np.zeros(len(y), dtype=float)
for tr, te in logo.split(X, y, groups):
    if len(set(y[tr])) < 2:
        oof[te] = y[tr].mean(); continue
    m = make_model().fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]

auc = roc_auc_score(y, oof)
fpr, tpr, thr = roc_curve(y, oof)
i = int(np.argmin(np.abs(tpr - RULE_OUT_SENS)))
t_ruleout = float(thr[i])
sens, spec = float(tpr[i]), float(1 - fpr[i])
print(f"  AUROC (out-of-fold, record-held-out): {auc:.4f}")
print(f"  At rule-out threshold {t_ruleout:.3f}: sensitivity {100*sens:.1f}%, "
      f"specificity {100*spec:.1f}%")

# per-record spread: does it work for everyone, or on average?
print("\n  Per-record AUROC (the spread matters more than the mean):")
per = []
for g in sorted(set(groups)):
    m_ = groups == g
    if len(set(y[m_])) == 2:
        a = roc_auc_score(y[m_], oof[m_]); per.append((g, a, int(m_.sum())))
per.sort(key=lambda r: r[1])
for g, a, n in per[:3]:
    print(f"    worst: {g}  AUROC {a:.3f}  (n={n})")
for g, a, n in per[-2:]:
    print(f"    best : {g}  AUROC {a:.3f}  (n={n})")
print(f"    records with both classes: {len(per)} of {len(set(groups))}")


def ppv_npv(sens, spec, prev):
    tp, fn = sens * prev, (1 - sens) * prev
    fp, tn = (1 - spec) * (1 - prev), spec * (1 - prev)
    return (tp / (tp + fp) if tp + fp else 0.0), (tn / (tn + fn) if tn + fn else 0.0)


print("\n  What that means when screening a real population:")
for prev, lab in [(0.02, "general adults 55+"), (0.05, "adults 65+"), (0.10, "high-risk 75+")]:
    p, n = ppv_npv(sens, spec, prev)
    print(f"    AF prevalence {100*prev:4.1f}% ({lab:18s}) -> PPV {100*p:4.1f}%  NPV {100*n:5.2f}%")

# ------------------------------------------------- feasibility under noise ----
print()
print("=" * 76)
print("Feasibility curve: what camera-grade RR error does to this")
print("=" * 76)
print("Rebuilding features from AFDB beat times with added measurement noise.")

import wfdb
from build_afdb import RECORDS, WINDOW_SEC, STEP_SEC, MIN_BEATS, PURITY

CACHE = os.path.join(HERE, "afdb_beats.npz")
if os.path.exists(CACHE):
    c = np.load(CACHE, allow_pickle=True)
    beats_by_rec = {k: c[k] for k in c.files if not k.endswith("__af")}
    af_by_rec = {k[:-4]: c[k] for k in c.files if k.endswith("__af")}
else:
    from build_afdb import rhythm_intervals
    beats_by_rec, af_by_rec = {}, {}
    for rec in RECORDS:
        try:
            qrs = wfdb.rdann(rec, "qrs", pn_dir="afdb")
            atr = wfdb.rdann(rec, "atr", pn_dir="afdb")
        except Exception:
            continue
        b = np.asarray(qrs.sample, float)
        flag = np.zeros(len(b), bool)
        for a_, b_, is_af in rhythm_intervals(atr, b[-1] + 1):
            if is_af:
                flag |= (b >= a_) & (b < b_)
        beats_by_rec[rec] = b / qrs.fs      # seconds
        af_by_rec[rec] = flag
    np.savez(CACHE, **beats_by_rec, **{k + "__af": v for k, v in af_by_rec.items()})
print(f"  cached beat times for {len(beats_by_rec)} records\n")


def rebuild(jitter_ms, miss_rate, fps=None, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels, recs = [], [], []
    for rec, bt in beats_by_rec.items():
        flag = af_by_rec[rec]
        t = bt.copy()
        if miss_rate > 0:                       # dropped beats: a real PPG failure mode
            keep = rng.random(len(t)) > miss_rate
            keep[0] = keep[-1] = True
            t, flag = t[keep], flag[keep]
        if fps:                                 # quantise onto the video frame grid
            t = np.round(t * fps) / fps
        if jitter_ms > 0:
            t = t + rng.normal(0, jitter_ms / 1000.0, len(t))
        t = np.sort(t)
        s, tend = t[0], t[-1]
        while s + WINDOW_SEC <= tend:
            sel = (t >= s) & (t < s + WINDOW_SEC)
            bb = t[sel]
            if len(bb) >= MIN_BEATS:
                rr = np.diff(bb) * 1000.0
                rr = rr[(rr > 250) & (rr < 2500)]
                if len(rr) >= MIN_BEATS - 1:
                    fa = float(np.mean(flag[sel]))
                    if fa >= PURITY or fa <= 1 - PURITY:
                        f = features(rr)
                        v = [f[k] for k in FEATURE_NAMES]
                        if not any(np.isnan(v)):
                            rows.append(v); labels.append(1 if fa >= PURITY else 0); recs.append(rec)
            s += STEP_SEC
    return np.array(rows), np.array(labels), np.array(recs)


def evaluate(Xn, yn, gn):
    oof_ = np.zeros(len(yn))
    for tr, te in LeaveOneGroupOut().split(Xn, yn, gn):
        if len(set(yn[tr])) < 2:
            oof_[te] = yn[tr].mean(); continue
        oof_[te] = make_model().fit(Xn[tr], yn[tr]).predict_proba(Xn[te])[:, 1]
    a = roc_auc_score(yn, oof_)
    f_, t_, th = roc_curve(yn, oof_)
    j = int(np.argmin(np.abs(t_ - RULE_OUT_SENS)))
    return a, float(t_[j]), float(1 - f_[j])


print(f"  {'condition':46s} {'AUROC':>7s} {'sens':>7s} {'spec':>7s}")
print("  " + "-" * 70)
scenarios = [
    ("ECG ground truth (no degradation)",            0,  0.00, None),
    ("30 fps grid only (no interpolation)",          0,  0.00, 30),
    ("30 fps + 8 ms jitter (parabolic interp)",      8,  0.00, None),
    ("60 fps + 5 ms jitter",                         5,  0.00, None),
    ("15 ms jitter",                                15,  0.00, None),
    ("25 ms jitter",                                25,  0.00, None),
    ("40 ms jitter",                                40,  0.00, None),
    ("8 ms jitter + 2% missed beats",                8,  0.02, None),
    ("8 ms jitter + 5% missed beats",                8,  0.05, None),
    ("25 ms jitter + 5% missed beats",              25,  0.05, None),
]
curve = []
for label, jit, miss, fps in scenarios:
    Xn, yn, gn = rebuild(jit, miss, fps)
    a, s_, sp = evaluate(Xn, yn, gn)
    curve.append({"condition": label, "jitter_ms": jit, "miss_rate": miss,
                  "fps": fps, "auroc": a, "sens": s_, "spec": sp})
    print(f"  {label:46s} {a:7.4f} {100*s_:6.1f}% {100*sp:6.1f}%")

# ------------------------------------------------------------------ export ----
final = make_model().fit(X, y)
sc, lr = final.named_steps["sc"], final.named_steps["lr"]
export = {
    "_source": "MIT-BIH Atrial Fibrillation Database (PhysioNet afdb), 25 records",
    "_validation": "leave-one-record-out cross-validation",
    "feature_names": FEATURE_NAMES,
    "coef": lr.coef_[0].tolist(),
    "intercept": float(lr.intercept_[0]),
    "mean": sc.mean_.tolist(),
    "scale": sc.scale_.tolist(),
    "threshold_rule_out": t_ruleout,
    "window_seconds": WINDOW_SEC,
    "min_beats": MIN_BEATS,
    "performance": {
        "auroc_record_held_out": auc,
        "sensitivity_at_threshold": sens,
        "specificity_at_threshold": spec,
        "ppv_npv_by_prevalence": {
            f"{p}": dict(zip(("ppv", "npv"), ppv_npv(sens, spec, p)))
            for p in (0.02, 0.05, 0.10)
        },
        "per_record_auroc_min": min(a for _, a, _ in per),
        "per_record_auroc_max": max(a for _, a, _ in per),
        "noise_curve": curve,
    },
}
out = os.path.join(HERE, "af_model_export.json")
json.dump(export, open(out, "w"), indent=2)
print(f"\nwrote {out}")
