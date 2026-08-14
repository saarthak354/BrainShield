"""
BrainShield v1 — stroke incident-risk modeling pipeline
Dataset: BRFSS 2015 (CDC), cleaned/binarized public release (Teboul, 2021),
originally sourced from CDC's Behavioral Risk Factor Surveillance System.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              roc_curve, confusion_matrix, classification_report)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
OUT = "/home/claude/work/brainshield/results"
import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("/home/claude/work/brainshield/data/brfss2015_heart.csv")
print("Full shape:", df.shape)

TARGET = "Stroke"
y = df[TARGET].astype(int)

# ---- Classical-score-style baseline feature set (mirrors Framingham/QRISK-style inputs) ----
BASELINE_FEATS = ["HighBP", "HighChol", "Smoker", "Diabetes", "BMI", "Age", "Sex",
                   "HeartDiseaseorAttack", "PhysActivity"]

# ---- Full self-reportable feature set for the ML model ----
ML_FEATS = [c for c in df.columns if c != TARGET]

X_base = df[BASELINE_FEATS]
X_ml = df[ML_FEATS]

# single consistent split (stratified) shared across models for fair comparison
idx_train, idx_test = train_test_split(
    df.index, test_size=0.2, random_state=RNG, stratify=y
)
idx_train, idx_val = train_test_split(
    idx_train, test_size=0.15, random_state=RNG, stratify=y.loc[idx_train]
)

def split(X):
    return X.loc[idx_train], X.loc[idx_val], X.loc[idx_test]

Xb_tr, Xb_va, Xb_te = split(X_base)
Xm_tr, Xm_va, Xm_te = split(X_ml)
y_tr, y_va, y_te = y.loc[idx_train], y.loc[idx_val], y.loc[idx_test]

print(f"Train {len(y_tr)} | Val {len(y_va)} | Test {len(y_te)} | "
      f"Stroke prevalence train/test: {y_tr.mean():.4f}/{y_te.mean():.4f}")

results = {}

def bootstrap_auc_ci(y_true, y_score, n=1000, seed=RNG):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n_s = len(y_true)
    aucs = []
    for _ in range(n):
        idx = rng.randint(0, n_s, n_s)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(np.mean(aucs)), float(lo), float(hi)

def evaluate(name, y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    auc_mean, lo, hi = bootstrap_auc_ci(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    results[name] = {"AUROC": auc, "AUROC_CI_lo": lo, "AUROC_CI_hi": hi,
                      "AUPRC": ap, "Brier": brier}
    print(f"{name:35s} AUROC={auc:.4f} (95% CI {lo:.4f}-{hi:.4f})  AUPRC={ap:.4f}  Brier={brier:.4f}")
    return auc

# ---------------- 1. Classical-score-style baseline (plain logistic regression, few features) ----------------
base_model = LogisticRegression(max_iter=2000)
base_model.fit(Xb_tr, y_tr)
base_prob = base_model.predict_proba(Xb_te)[:, 1]
evaluate("Classical-score baseline (LR, 9 feats)", y_te, base_prob)

# ---------------- 2. Full-feature logistic regression ----------------
lr_full = LogisticRegression(max_iter=2000, class_weight="balanced")
lr_full.fit(Xm_tr, y_tr)
lr_full_prob = lr_full.predict_proba(Xm_te)[:, 1]
evaluate("Logistic Regression (all 21 feats, balanced)", y_te, lr_full_prob)

# ---------------- 3. Random Forest ----------------
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced_subsample",
                             random_state=RNG, n_jobs=-1)
rf.fit(Xm_tr, y_tr)
rf_prob = rf.predict_proba(Xm_te)[:, 1]
evaluate("Random Forest (all 21 feats)", y_te, rf_prob)

# ---------------- 4. XGBoost (primary ML model) ----------------
pos = y_tr.sum(); neg = len(y_tr) - pos
xgb = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=neg / pos, eval_metric="auc",
    random_state=RNG, n_jobs=-1
)
xgb.fit(Xm_tr, y_tr, eval_set=[(Xm_va, y_va)], verbose=False)
xgb_prob = xgb.predict_proba(Xm_te)[:, 1]
evaluate("XGBoost (all 21 feats)", y_te, xgb_prob)

# ---------------- Calibrated version of XGBoost for honest probability output ----------------
xgb_uncal = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=RNG, n_jobs=-1
)  # no scale_pos_weight -> natural probabilities, then calibrate
xgb_cal = CalibratedClassifierCV(xgb_uncal, method="isotonic", cv=3)
xgb_cal.fit(Xm_tr, y_tr)
xgb_cal_prob = xgb_cal.predict_proba(Xm_te)[:, 1]
evaluate("XGBoost, calibrated (isotonic)", y_te, xgb_cal_prob)

# ---------------- Statistical comparison: XGBoost vs classical baseline (DeLong-style via bootstrap paired diff) ----------------
rng = np.random.RandomState(RNG)
y_te_arr = y_te.values
n_s = len(y_te_arr)
diffs = []
for _ in range(2000):
    idx = rng.randint(0, n_s, n_s)
    if len(np.unique(y_te_arr[idx])) < 2:
        continue
    a = roc_auc_score(y_te_arr[idx], xgb_prob[idx])
    b = roc_auc_score(y_te_arr[idx], base_prob[idx])
    diffs.append(a - b)
diffs = np.array(diffs)
p_value_like = float((diffs <= 0).mean())  # fraction of bootstrap resamples where ML does NOT beat baseline
delta_mean = float(diffs.mean())
delta_lo, delta_hi = np.percentile(diffs, [2.5, 97.5])
print(f"\nAUROC delta (XGBoost - baseline): {delta_mean:.4f} (95% CI {delta_lo:.4f}-{delta_hi:.4f}), "
      f"fraction of bootstraps where ML<=baseline: {p_value_like:.4f}")
results["_comparison"] = {
    "delta_auc_mean": delta_mean, "delta_auc_ci_lo": float(delta_lo), "delta_auc_ci_hi": float(delta_hi),
    "bootstrap_frac_ml_not_better": p_value_like
}

# ---------------- Subgroup analysis (Sex, Age) on best model (XGBoost) ----------------
subgroup_rows = []
te_df = df.loc[idx_test].copy()
te_df["y_true"] = y_te.values
te_df["y_prob"] = xgb_prob

for sex_val, sex_name in [(0.0, "Female"), (1.0, "Male")]:
    sub = te_df[te_df["Sex"] == sex_val]
    if sub["y_true"].nunique() == 2:
        subgroup_rows.append(("Sex", sex_name, len(sub), roc_auc_score(sub["y_true"], sub["y_prob"])))

# Age is coded 1-13 (BRFSS 5-yr bands); bucket into 3 groups for readability
def age_bucket(a):
    if a <= 6: return "18-44"
    if a <= 9: return "45-59"
    return "60+"
te_df["AgeGroup"] = te_df["Age"].apply(age_bucket)
for grp, sub in te_df.groupby("AgeGroup"):
    if sub["y_true"].nunique() == 2:
        subgroup_rows.append(("AgeGroup", grp, len(sub), roc_auc_score(sub["y_true"], sub["y_prob"])))

subgroup_df = pd.DataFrame(subgroup_rows, columns=["Dimension", "Group", "N", "AUROC"])
subgroup_df.to_csv(f"{OUT}/subgroup_auc.csv", index=False)
print("\nSubgroup AUROC:\n", subgroup_df)

# ---------------- Save metrics ----------------
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

pd.DataFrame(results).T.to_csv(f"{OUT}/metrics_table.csv")

# ---------------- Plots ----------------
plt.figure(figsize=(6, 6))
for name, prob, style in [
    ("Classical-score baseline", base_prob, "--"),
    ("Logistic Regression (full)", lr_full_prob, ":"),
    ("Random Forest", rf_prob, "-."),
    ("XGBoost", xgb_prob, "-"),
]:
    fpr, tpr, _ = roc_curve(y_te, prob)
    auc_val = roc_auc_score(y_te, prob)
    plt.plot(fpr, tpr, style, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linewidth=0.8)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves — stroke risk models (BRFSS 2015 held-out test set)")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 6))
for name, prob in [("XGBoost (raw)", xgb_prob), ("XGBoost (calibrated)", xgb_cal_prob)]:
    frac_pos, mean_pred = calibration_curve(y_te, prob, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=name)
plt.plot([0, 1], [0, 1], color="gray", linewidth=0.8, label="Perfect calibration")
plt.xlabel("Mean predicted risk"); plt.ylabel("Observed stroke rate")
plt.title("Calibration — XGBoost, raw vs. isotonic-calibrated")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/calibration.png", dpi=150)
plt.close()

print("\nDone. Outputs in", OUT)
