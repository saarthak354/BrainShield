"""
Train a clean XGBoost + single-fold isotonic calibration pipeline and export
everything needed for a pure client-side (JS) re-implementation:
  - tree ensemble as JSON (recursive split/leaf structure)
  - base_score (log-odds intercept XGBoost adds before summing trees)
  - isotonic calibration curve as sorted (x, y) breakpoints for linear interpolation
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

RNG = 42
df = pd.read_csv("/home/claude/work/brainshield/data/brfss2015_heart.csv")
y = df["Stroke"].astype(int)
FEATS = [c for c in df.columns if c != "Stroke"]
X = df[FEATS]

idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=RNG, stratify=y)
idx_train, idx_val = train_test_split(idx_train, test_size=0.15, random_state=RNG, stratify=y.loc[idx_train])

Xtr, ytr = X.loc[idx_train], y.loc[idx_train]
Xva, yva = X.loc[idx_val], y.loc[idx_val]
Xte, yte = X.loc[idx_test], y.loc[idx_test]

xgb = XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="auc", random_state=RNG, n_jobs=-1,
)  # no scale_pos_weight -> natural probabilities suitable for isotonic calibration
xgb.fit(Xtr, ytr)

val_raw = xgb.predict_proba(Xva)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(val_raw, yva)

test_raw = xgb.predict_proba(Xte)[:, 1]
test_cal = iso.predict(test_raw)

auc_raw = roc_auc_score(yte, test_raw)
auc_cal = roc_auc_score(yte, test_cal)
brier_raw = brier_score_loss(yte, test_raw)
brier_cal = brier_score_loss(yte, test_cal)
print(f"Uncalibrated: AUROC={auc_raw:.4f} Brier={brier_raw:.4f}")
print(f"Calibrated:   AUROC={auc_cal:.4f} Brier={brier_cal:.4f}")

# ---------------- Export trees ----------------
booster = xgb.get_booster()
dump = booster.get_dump(dump_format="json")
trees = [json.loads(t) for t in dump]

# XGBoost's sklearn wrapper default base_score for binary:logistic is 0.5 (probability space);
# get the raw base_score used internally (log-odds) from the booster config.
config = json.loads(booster.save_config())
base_score_str = config["learner"]["learner_model_param"]["base_score"]
base_score_raw = float(str(base_score_str).strip("[]"))
# base_score_raw is in probability space (e.g. 0.5); convert to log-odds margin XGBoost actually adds
base_margin = float(np.log(base_score_raw / (1 - base_score_raw))) if 0 < base_score_raw < 1 else 0.0

# ---------------- Export isotonic calibration curve ----------------
iso_x = iso.X_thresholds_.tolist()
iso_y = iso.y_thresholds_.tolist()

export = {
    "feature_order": FEATS,
    "n_trees": len(trees),
    "base_margin": base_margin,
    "trees": trees,
    "isotonic_x": iso_x,
    "isotonic_y": iso_y,
    "metrics": {"auroc_calibrated_test": auc_cal, "brier_calibrated_test": brier_cal},
}
with open("/home/claude/work/brainshield/site/model_export.json", "w") as f:
    json.dump(export, f)

print("Exported", len(trees), "trees. base_margin=", base_margin)
print("Isotonic breakpoints:", len(iso_x))

# Save a handful of test rows + true model outputs for JS cross-validation
sample_idx = Xte.sample(12, random_state=1).index
sample = X.loc[sample_idx].copy()
sample["xgb_raw"] = xgb.predict_proba(X.loc[sample_idx])[:, 1]
sample["xgb_calibrated"] = iso.predict(sample["xgb_raw"])
sample.to_json("/home/claude/work/brainshield/site/validation_samples.json", orient="records")
print("Saved validation samples.")
