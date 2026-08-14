# BrainShield

A stroke risk self-assessment tool that runs entirely in the browser, plus a working
camera-based pulse measurement.

**Live demo:** https://saarthak354.github.io/BrainShield/

Two independent things live here:

1. **Stroke risk prediction** — a calibrated gradient-boosted model trained on 253,680
   CDC survey respondents, using only inputs a person can report about themselves. No
   clinic visit, lab test, or imaging required.
2. **Camera pulse measurement (rPPG)** — a real physical measurement. The camera detects
   the sub-1% brightness flicker caused by blood flow under the skin and recovers a pulse
   waveform from it.

Everything runs client-side. No data is transmitted anywhere; there is no backend.

---

## Results

| Model | AUROC | 95% CI | Brier |
|---|---|---|---|
| Classical-score-style baseline (LR, 9 features) | 0.789 | 0.780–0.799 | 0.037 |
| Logistic regression (21 features, balanced) | 0.826 | 0.818–0.834 | 0.172 |
| Random forest | 0.824 | 0.816–0.833 | 0.147 |
| **XGBoost, isotonic-calibrated** | **0.828** | **0.819–0.836** | **0.036** |

The ML model beats the classical points-based scoring paradigm by **ΔAUROC = +0.034
(95% CI 0.028–0.040)**; in a 2,000-resample paired bootstrap, 0 resamples favoured the
baseline.

For context from the literature: CHA2DS2-VASc ≈ 0.64, CHADS2 ≈ 0.66, QRISK3 ≈ 0.72,
best statistical calculator (AECRS 2.0) ≈ 0.83. DeepRETStroke reaches ≈ 0.90 but requires
retinal imaging — a different and richer input than anything used here.

### A limitation worth reading

Discrimination drops for older respondents — AUROC 0.736 in the 60+ group versus
0.816–0.830 in younger groups (n = 17,823 in that test subgroup). This is treated as a
real weakness, not noise, and is the main target for future work.

---

## Camera pulse measurement

Fingertip-with-flash is the **default** mode, deliberately: active illumination raises
signal strength for everyone and disproportionately helps where skin reflectance is lower.
Face mode (POS / CHROM algorithms) is available but is far more sensitive to motion.

### Skin tone accuracy — stated plainly

Published work (not our measurements) finds rPPG mean absolute error rising from about
**5.2 bpm on Fitzpatrick I–III to 14.1 bpm on Fitzpatrick V–VI**, with existing datasets
under-representing darker skin tones (<25% of participants). Melanin absorbs light, which
weakens the signal this technique depends on.

**This module has not been validated on real human subjects of any skin tone.** All
testing to date uses synthetic signals. See `docs/rppg-validation-note.md`.

### The quality gating problem

An early version gated reliability on signal-to-noise ratio alone. Adversarial testing
showed this fails dangerously: when the estimator locks onto a motion artifact, that
artifact is *itself* a clean spectral peak, so SNR reports an excellent signal while the
reported heart rate is wrong. Under simulated darker-skin, in-motion conditions this
produced **13 confidently-wrong readings out of 60**.

Two further independent gates were added — temporal consistency across sliding windows,
and cross-algorithm agreement between POS and CHROM. Silent failures fell to **0 out of
60**. Readings that cannot be trusted are labelled "Poor signal — not reliable" rather
than shown as a number.

A known irreducible limit remains: a rhythmic disturbance at a steady, heart-rate-plausible
frequency is physically ambiguous to a single camera in face mode. Documented, not solved.

### Deliberately kept separate

The heart rate is **not** fed into the stroke risk score. The risk model was trained
exclusively on questionnaire variables; mixing a physiological measurement into its output
would misrepresent what the reported AUROC actually validated.

SDNN/RMSSD are shown as raw variability numbers labelled experimental. They are **not** an
atrial-fibrillation test.

---

## Repository layout

```
index.html                  the complete site — model and code inlined, no dependencies
src/
  model_pipeline.py         trains and evaluates all models, produces results/
  export_model.py           exports trees + calibration curve for browser inference
  rppg_core.js              rPPG signal processing (POS, CHROM, FFT, quality gating)
  rppg_ui.js                camera capture UI
  build.py                  inlines everything into index.html
tests/
  test_rppg.js              DSP validation against known-frequency signals
  e2e_camera_test.py        browser capture path, synthetic frames via MediaStream
results/                    metrics, ROC/calibration/SHAP figures
docs/
  rppg-validation-note.md   what was tested, what wasn't, and the numbers
  how_rppg_works.png        the signal-processing pipeline, end to end
```

## Running it

Camera access requires a secure context — `https://` or `localhost`. Opening
`index.html` directly from the filesystem will load the page but the camera will be
refused by the browser.

```bash
python3 -m http.server 8000     # then open http://localhost:8000
```

### Rebuilding

```bash
pip install pandas scikit-learn xgboost shap matplotlib
python3 src/model_pipeline.py     # train + evaluate
python3 src/export_model.py       # export model for the browser
python3 src/build.py              # regenerate index.html
```

### Tests

```bash
node tests/test_rppg.js           # 20 DSP tests
python3 tests/e2e_camera_test.py  # browser capture (needs a local server running)
```

---

## Data

CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, via a cleaned and binarized
public release. 253,680 respondents, 21 self-reportable predictors, 4.06% positive rate.

Dataset provenance was a deliberate choice. In May 2026 several published stroke and
diabetes ML papers were found to rest on an undocumented Kaggle dataset containing
duplicated records and mislabeled celebrity photographs. Every source used here is
documented and traceable to a government survey.

## Limitations

- The outcome label is self-reported and cross-sectional ("ever told you had a stroke") —
  not a clinically confirmed, prospectively ascertained incident event.
- 2015 survey data; single country.
- No independent external cohort validation.
- The rPPG module has no human validation of any kind.

## Status

Research prototype. **Not a diagnostic device.** Not for clinical use.
