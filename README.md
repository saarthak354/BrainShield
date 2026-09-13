# BrainShield

**A multimodal stroke risk assessment that runs entirely on your phone.**

Answer a short questionnaire, let the camera check your pulse rhythm for 60 seconds,
optionally photograph your medicines — and get a calibrated five-year absolute stroke
probability. No account, no backend, no upload. Every computation happens in the browser.

---

## Three inputs, one number

**1 · Questionnaire** — age, sex, blood pressure, medication, diabetes, smoking,
smokeless tobacco, prior cardiovascular disease, atrial fibrillation.

**2 · Pulse rhythm check** — fingertip over the rear camera with the torch on for 60
seconds. Contact photoplethysmography detects the sub-1% brightness flicker of each
heartbeat and analyses the rhythm.

**3 · Medical documents** *(optional)* — photograph a medicine box or lab report. Text
is read on-device and turned into questions you confirm.

---

## Results

### Risk model

A Cox proportional-hazards survival model anchored to **23,691 adults followed
prospectively with physician-adjudicated stroke events**, with five years of follow-up
matching the five-year prediction horizon exactly.

| | |
|---|---|
| Reproduces published cohort risk | within **0.002** |
| Coefficient provenance | every value traceable to a published table |
| Exported model size | **6 KB** |

Baseline survival is recovered by solving against a synthetic cohort matched to the
published marginals, which removes the convexity bias in the usual shortcut and changes
a typical person's estimate by ~15% relative.

### Rhythm detection

Trained on the MIT-BIH Atrial Fibrillation Database — **29,018 windows, 25 patients** —
and validated **leave-one-patient-out**, so no patient appears in both training and test.

| | |
|---|---|
| AUROC | **0.9903** |
| Sensitivity / specificity | **95% / 97.4%** |
| Per-patient AUROC range | 0.931 – 1.000 |
| Exported classifier | **4 KB**, no runtime dependency |

### Independent validation on real smartphone recordings

Two external corpora, neither used in development.

**BUT PPG** (PhysioNet) — 300 fingertip recordings on consumer handsets with
synchronous 1 kHz ECG:

| | |
|---|---|
| Beat-detection sensitivity | **90%** |
| Beat-detection precision | **90%** |
| Inter-beat interval error | **28.4 ms** |
| Heart-rate error | **1.0 bpm** |

**Gdańsk University of Technology corpus** — 60-second recordings at the deployed
window length:

| | |
|---|---|
| Specificity, healthy volunteers | **91.7%** |
| Independently projected from BUT PPG | 92.4% |

Two unrelated corpora, different countries and handsets, agreeing to within **0.7
percentage points**.

---

## What makes it work

**Unknown answers produce a range, not a guess.** Most people don't know their blood
pressure. Rather than substituting a population average and printing a confident
number, unknown inputs are carried as distributions and propagated by Monte Carlo. You
get an honest interval — and a ranked list of which single measurement would narrow it
most, computed as a variance decomposition and personalised to you.

**The rhythm check knows what it can and cannot claim.** A regular pulse rules atrial
fibrillation out of your estimate and sharpens the number. An irregular pulse says *get
an ECG* — never *you have AF* — because at population prevalence the negative predictive
value is 99.9% while a positive is right roughly one time in five. The decision rule
follows the evidence rather than overstating it.

**Quality gating built for arrhythmia.** The metrics normally used to gate
photoplethysmography — spectral peak sharpness and heart-rate stability across windows —
measure regularity, and so reject irregular rhythms as noise. This system gates on
band-aggregate and time-domain criteria instead, which catch motion without discarding
the signal being looked for.

**Sub-sample beat timing.** A 30 fps camera quantises beats to 33 ms. Parabolic peak
interpolation recovers timing to a fraction of a frame — 15.9 ms to 0.04 ms on a test
signal — so frame rate stops being the limiting factor.

**Counterfactuals you can act on.** "Stop smoking → 3.8%." "Lower systolic BP from 150 to
130 → 6.2%." Ranked by benefit, with the reasoning shown.

**Adapted for India.** Smokeless tobacco — gutkha, khaini, zarda, paan masala — affects
29.6% of Indian men and 12.8% of women and is absent from every Western risk equation.
It's a separate question with its own published hazard ratio of 1.35, weighted
independently of smoking. Document scanning recognises 56 generic drugs and 191 brand
names weighted toward Indian packaging.

**Nothing leaves your device.** The risk model, rhythm analysis and text recognition all
run in the browser. No questionnaire response, video frame or photograph is transmitted.

---

## Quality

**≈11,245 test assertions**, all passing.

The deployed JavaScript is verified against the Python reference implementation by
replaying **10,764 risk values and 358 rhythm features** through it, requiring agreement
to 1e-12. Observed worst-case divergence: **1.1e-16**.

Model parameters are cryptographically hashed before validation and verified unchanged
afterwards, so every published figure provably describes the shipped code.

```bash
python3 src/risk/validate.py               # model vs published paper
python3 -m pytest tests/test_risk_model.py # unit tests
node tests/test_risk_parity.js             # Python/JS parity
node tests/test_af_parity.js
node tests/test_beat_matching.js
node tests/test_rppg.js
python3 -m http.server 8000 &              # then the browser suites:
python3 tests/e2e_assess_test.py
python3 tests/e2e_risk5_test.py
```

---

## Running it

```bash
python3 -m http.server 8000
```

Then open **http://localhost:8000/assess.html**. The camera needs a secure context —
`https://` or `localhost`.

| Page | |
|---|---|
| `assess.html` | the full three-step assessment (120 KB) |
| `risk5.html` | questionnaire only (48 KB) |
| `index.html` | association model and rPPG explainer |

### Rebuilding

```bash
pip install numpy scipy scikit-learn pandas xgboost wfdb
python3 src/risk/export_risk_model.py    # solve S0, export constants
python3 src/build_assess.py              # build assess.html
python3 src/build_risk5.py               # build risk5.html
python3 src/build.py                     # build index.html
```

---

## Paper

`paper/brainshield_ieee.tex` — IEEE conference format, compiles with pdfLaTeX or
Overleaf. A built PDF is included.

---

## Status

Research prototype. Not a diagnostic device, and not for clinical use — the interface
says so too, and directs anyone with active symptoms to emergency services.
