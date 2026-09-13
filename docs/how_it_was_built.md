# BrainShield — Build and Validation Record

A complete account of what was built, what external data it rests on, what was
measured, and how each figure can be reproduced. This is the engineering record; the
condensed scientific write-up is `paper/brainshield_ieee.tex`.

---

## 1. What exists

Three pages, all self-contained and executing entirely in the browser:

| Page | Size | Purpose |
|---|---|---|
| `assess.html` | 120 KB | Three-step assessment: questionnaire → camera rhythm check → optional documents → 5-year absolute stroke risk |
| `risk5.html` | 48 KB | Questionnaire-only risk calculator |
| `index.html` | 1.8 MB | Pre-existing BRFSS association model and rPPG explainer |

Behind them are three engines: a Cox survival risk model (`src/risk/`), an atrial
fibrillation classifier with camera capture (`src/af/`), and a medical-document reader
(`src/docscan/`).

---

## 2. External data used

Everything the system rests on, with provenance. **None of it is redistributed in this
repository** — all are excluded via `.gitignore` and re-fetchable by the scripts noted.

| Source | What it is | Used for | Access |
|---|---|---|---|
| **Dufouil et al. 2017**, *Circulation* 135(12):1145–1159 | Revised Framingham Stroke Risk Profile. Published hazard ratios, covariate means and Kaplan–Meier risks for two cohorts | Every risk-model coefficient | Open access, PMC5504355 |
| **REGARDS** (via the above) | 23,691 adults aged 55+, 5-year follow-up, adjudicated stroke, 42% Black participants | Default coefficient set | Published tables only |
| **Contemporary Framingham** (via the above) | 5,072 adults, 10-year follow-up | Alternative coefficient set, retained for reproducibility | Published tables only |
| **MIT-BIH AFDB** (PhysioNet `afdb`) | 25 long-term ECG recordings with verified beat annotations and AF episode labels | Training and validating the AF classifier | Open; `src/af/build_afdb.py` |
| **BUT PPG v2.0.0** (PhysioNet `butppg`) | 3,888 smartphone-camera fingertip recordings at 30 fps with synchronous 1 kHz ECG and verified QRS positions, 50 subjects | Independent validation of the camera acquisition path | Open; `src/af/cache_butppg.py` |
| **Gdańsk Univ. of Technology corpus** (Bancerewicz et al., MIT licence) | Smartphone PPG: one 61 s recording with aligned ECG, plus three ~10 min recordings | Independent validation at the deployed 60-second window length | Open on GitHub |
| **Gupta et al. 2020**, *J Public Health* 42(2):e150–e157 | Meta-analysis: stroke RR 1.35 (1.20–1.50) for chewing tobacco | Smokeless tobacco adjunct term | Open access |
| **GATS-2 India 2016–17** | Smokeless tobacco prevalence: 29.6% men, 12.8% women | Population priors for unknown inputs | Public report |
| **NFHS-5 (2019–21)** | Indian hypertension prevalence, awareness, diabetes prevalence | Population priors | Public report |
| **MIMIC PERform AF** (Charlton et al., Zenodo) | Fingertip PPG + ECG, 35 ICU adults (19 AF / 16 not), 20 min each at 125 Hz | Classifier sensitivity at the deployed 60 s window on fingertip PPG | Open, ODC-BY, no login; `src/af/prep_mimic_af.py` |
| **Stanford DeepBeat** (Synapse syn21985690) | Wrist PPG, 25 s segments at 32 Hz with rhythm and signal-quality labels. Held-out test split: 17,617 segments, 22 subjects (6 AF) | Classifier sensitivity on PPG-derived rather than ECG-derived intervals | Free Synapse account; `src/af/prep_deepbeat.py` |

---

## 3. Starting point

The repository already contained an XGBoost model trained on BRFSS 2015 (253,680
survey respondents) and an rPPG pulse-measurement module.

Three findings shaped everything that followed.

**The existing model could not produce a 5-year risk.** Its target was the BRFSS item
*"(Ever told) you had a stroke"* — cross-sectional lifetime prevalence with no time
axis. The interface nevertheless labelled its output *"estimated 5-year self-reported
stroke risk."* No calibration converts a prevalence model into a forward risk, so a
separate survival model was required. The mislabel was corrected and the page now
states what it actually measures.

**The model was partly detecting rather than predicting.** In
`results/shap_importance.csv`, `GenHlth` (#3), `DiffWalk` (#5) and `PhysHlth` (#7)
together carry ~0.76 of mean |SHAP| — all three are *consequences* of stroke, not
antecedents.

**That explains its known weakness in older respondents.** AUROC is 0.736 in the 60+
group against 0.816–0.830 below it: in older people poor general health and difficulty
walking are common for many reasons, so the reverse-causal shortcut stops separating
cases.

---

## 4. The risk model

### Model form

```
risk(5 yr | x) = 1 − S₀(5) ^ exp( LP(x) − M )
```

with `LP(x) = Σ βⱼxⱼ`, `βⱼ = ln(HRⱼ)`, and `M = Σ βⱼx̄ⱼ` centring on the cohort mean.
Diabetes enters through one of two age-conditioned coefficients (the only interaction
the source model retained); systolic pressure enters through a treated or untreated
coefficient by medication status.

### Choosing coefficients

The source paper publishes two sets. REGARDS was adopted because its derivation sample
is 23,691 against 5,072, its follow-up is exactly five years (no extrapolation to the
prediction horizon), events were adjudicated under six-monthly surveillance, and its
hazard ratios are all biologically coherent.

The Framingham set is not: atrial fibrillation in men is HR 1.08 (95% CI 0.61–1.94),
statistically null for one of the strongest known stroke risk factors, and prevalent
CVD in women is HR 0.97, i.e. protective. Both come from roughly 60 events.
`src/risk/validate.py` asserts that this set misbehaves, so the reason it is not the
default lives in the test output rather than in folklore.

### Recovering baseline survival

The paper reports Kaplan–Meier cohort risk but not S₀. Since risk is convex in the
linear predictor, mean-of-risk exceeds risk-at-mean, so `S₀ = 1 − KM` is biased.
Instead a synthetic cohort matching the published marginals is constructed and S₀
solved numerically so that mean predicted risk reproduces the observed value.

Result: **S₀(5) = 0.97310** (men), **0.97803** (women). The naive substitution gives
0.969 and 0.973 — overstating a typical person's risk by ~15% relative.

### Uncertainty as an output

Unknown inputs are drawn from population priors and propagated by Monte Carlo, giving a
10th–90th percentile interval. A fully specified profile collapses it to a point.

Each unknown is then ranked by its **first-order Sobol index** — the share of output
variance it explains. An earlier version ranked by interval width averaged over a
hand-picked grid; that is not a proper expectation and could return a negative
"narrowing." Variance-based decomposition is bounded in [0,1] and well defined here.

The ranking is subject-specific: for a 68-year-old man with nothing known, smoking
explains 52% of the variance and blood pressure 10%; for a 58-year-old woman the order
inverts, with blood pressure at 36%, reflecting the steeper pressure gradient in the
female coefficients.

### The India adjunct term

Smokeless tobacco is absent from every Western equation but affects 29.6% of Indian men.
It was added as an extra linear-predictor term with the published adjusted RR of 1.35,
**centred at zero rather than at Indian prevalence** — the derivation cohort is the
reference S₀ was calibrated against, and smokeless tobacco was negligible in it, so
non-use already is the reference. Centring on target prevalence would give every Indian
non-user an unearned discount. A test asserts cohort calibration is undisturbed.

It is asked separately from smoking because its effect is much smaller (HR 2.00 for
smoking); combining them would overstate risk for exclusive smokeless users by ~50%.
The same meta-analysis reports the association essentially unchanged under strict
adjustment for smoking (1.17 vs 1.18), supporting the independence assumption.

### One thing the tool refuses to say

At fixed measured systolic pressure the fitted model scores medicated people *below*
unmedicated ones, and the gap widens with pressure (3.29% vs 5.26% at SBP 190). That is
an artifact of the flatter pressure–risk gradient observed in treated cohorts, not a
causal treatment effect. "Start blood-pressure medication" is therefore excluded from
the counterfactual engine, with a test asserting it can never appear.

---

## 5. Atrial fibrillation detection

### Classifier

Eight scale-free interval descriptors: normalised RMSSD, coefficient of variation,
pRR50, pRR70, normalised Shannon entropy, sample entropy, Poincaré SD1/SD2, and mean
rate. A standardised logistic regression over these — deliberately linear, so the
decision is auditable and the export is 4 KB with no runtime dependency.

Trained on MIT-BIH AFDB: 60-second windows at 30-second hops, labelled fibrillatory
when ≥90% of beats fall in an annotated episode. **29,018 windows, 37.2% AF**, validated
**leave-one-record-out** so no patient appears in both training and test.

### Sample entropy at the extremes

Sample entropy is undefined when no template pairs match within tolerance. This occurs
at *both* extremes of regularity: highly irregular sequences (no matches exist) and
near-constant ones (the matching tolerance approaches zero). The original code returned
NaN, and the training script dropped NaN rows — so both tails were silently excluded
from training, and real AF patients would have received "couldn't read your pulse."

Replaced with the conventional upper bound `−ln(2/[(N−m−1)(N−m)])`. The dataset
recovered **1,167 windows (27,851 → 29,018)**, and honest performance moved *down*
(AUROC 0.9929 → 0.9903) because the previously excluded hard cases were now included.

### Sub-sample beat timing

At 30 fps, beats quantise to 33 ms — comparable to the beat-to-beat variability that
separates sinus rhythm from AF. Parabolic interpolation through each peak and its
neighbours recovers the extremum to a fraction of a frame: on a test signal whose period
is deliberately incommensurate with the frame interval, interval error falls from
**15.9 ms to 0.04 ms**. Frame rate is therefore not the limiting factor.

### Quality gating built for arrhythmia

The two metric families conventionally used to gate photoplethysmography both fail here,
for the same underlying reason — each measures regularity, which is precisely the signal:

- **Inter-window rate consistency** penalises a rate that varies between windows. In AF
  the rate genuinely varies, so the criterion rejects the target condition. Confirmed
  empirically: a clean synthetic AF waveform was discarded as "poor."
- **Spectral signal-to-noise ratio** measures dominant-peak sharpness. Irregular
  intervals spread pulse energy across the band, so AF has no sharp peak by
  construction. A clean synthetic fibrillatory waveform measured **SNR −2.6**.

Replaced with criteria that are band-aggregate or purely temporal, responding to motion
without responding to rhythm: in-band power fraction, photometric stability, beat
amplitude consistency, and interval plausibility rate.

### The asymmetric decision rule

Three outcomes, never two:

- **Regular** → atrial fibrillation resolved to negative in the risk model; the interval narrows
- **Irregular** → estimate left unchanged, user advised to obtain an ECG; no classification asserted
- **Inconclusive** → quality criteria unmet; no inference drawn

This follows quantitatively from the predictive values below, not from caution.

### Tolerance to acquisition error

Features were recomputed from annotated beat times after injecting timing perturbation
and beat omission, repeating the full leave-one-record-out procedure at each level.

| Condition | AUROC | Sens | Spec |
|---|---|---|---|
| Annotated beat times | 0.9903 | 95% | 97.4% |
| 30 fps, no interpolation | 0.9918 | 95% | 96.9% |
| 30 fps with interpolation | 0.9900 | 95% | 96.9% |
| 60 fps, 5 ms jitter | 0.9906 | 95% | 97.3% |
| 15 ms jitter | 0.9880 | 95% | 95.7% |
| 25 ms jitter | 0.9826 | 95% | 92.5% |
| 40 ms jitter | 0.9699 | 95% | 87.4% |
| 8 ms jitter, 5% beats omitted | 0.9900 | 95% | 96.7% |

Discrimination holds to roughly 25 ms of per-beat error. Random beat omission at 5% is
negligible, showing the interval statistics are robust to sparse detection failure.

---

## 6. Medical document reading

Optical character recognition runs on-device (Tesseract.js), so no image leaves the
handset. Recognised text is matched against a curated lexicon of **8 drug classes, 56
generic agents and 191 proprietary names**, weighted toward Indian packaging where brand
names dominate.

Matching tolerates the substitution errors typical of curved, specular surfaces —
normalised tokens compared under a length-dependent bounded edit distance, so
`"TEIMA 4O"` resolves to telmisartan. Laboratory values are extracted by proximity of a
numeric literal to a recognised assay name, with range validation rejecting spurious
matches.

Classes with a single dominant indication propose the corresponding variable directly.
Classes with several — anticoagulants, prescribed for fibrillation, thrombosis or
prosthetic valves alike — raise a clarifying question rather than a value. Corroborating
evidence accumulates into one proposal, so metformin plus a raised HbA1c reads as
*"Found metformin; also hba1c of 7.8%."*

**No extracted value modifies the computation without explicit confirmation.** This makes
recognition error an interaction concern rather than a correctness one.

---

## 7. Independent validation

### Freezing first

Before any external evaluation, every file that can change a prediction was
cryptographically hashed into `src/af/FROZEN.json`. `python3 src/af/freeze.py --verify`
fails loudly if any drifts. This makes "validated on an independent dataset" a checkable
claim; it has passed at every stage since.

### BUT PPG — the acquisition path

300 fingertip recordings on consumer handsets with synchronous ECG. Two dataset defects
were found and handled: records before 112001 carry a transposed header (nsig/nsamp
swapped) *and* all-zero data, and `wfdb.rdrecord` returns silent garbage on them rather
than erroring. Records from 112001 onward are correct 3-channel RGB at 30 Hz.

Beat accuracy was assessed by **temporal correspondence**, not sequence subtraction: the
pulse transit delay is estimated per recording, each peak matched to an ECG beat within
100 ms, and interval error computed only across pairs whose endpoints both matched and
were consecutive in both streams. Measured offsets span −24 to −243 ms, so the lag search
covers a full beat interval in both directions.

The matching procedure has **its own test** (`tests/test_beat_matching.js`, 9 cases)
against synthetic sequences with known transit delay, known jitter, and deliberately
omitted and inserted beats — it recovers a known 250 ms lag exactly and a known 12 ms
jitter as 11.3 ms.

| Subset | Beat sens. | Beat PPV | RR error | HR error |
|---|---|---|---|---|
| Expert-graded good (n=100) | 90% | 90% | 28.4 ms | 1.0 bpm |
| Accepted by our gate | 85% | 92% | 28.9 ms | 2.1 bpm |
| Expert-graded poor (n=200) | 40% | 57% | 63.2 ms | 19.7 bpm |
| Rejected by our gate | 44% | 63% | 47.6 ms | 15.4 bpm |

The gate agrees with the database's own expert annotators, accepting recordings with
roughly twice the beat sensitivity of those it rejects.

### Gdańsk corpus — the deployed window length

BUT PPG clips are 10 s; the classifier operates on 60 s. The Gdańsk recordings are 61 s
and ~10 min, allowing evaluation at the deployed configuration.

**Specificity on healthy volunteers, continuous recordings: 91.7%** (22 regular / 2
irregular across 24 windows). The BUT PPG error-injection projected 92.4% — two
unrelated corpora, different countries and handsets, agreeing to within 0.7 points.

A third recording returned 7/7 irregular and was investigated rather than reported: it
proved to be several sessions concatenated (818 s of gaps, one of 332 s, 43% coverage).
It also exposed a genuine failure mode — **missed and spurious beats occurring together
and cancelling in the count** while destroying the intervals, producing rmssd_norm higher
than typical AF on a healthy subject. Beat count is therefore not a validity check. This
is recorded for the next iteration; the algorithm was not altered.

### MIMIC PERform AF — the deployed window on fingertip PPG

DeepBeat answered whether the classifier survives pulse-wave intervals, but only on 25 s
wrist reflectance. MIMIC PERform AF answers the sharper question: does it work at the
**deployed 60 s window** on **fingertip** PPG — transmission through a digit, the same
optical geometry as a torch-lit camera. 35 ICU adults, 19 in AF, 20 minutes each, cut
into 659 non-overlapping 60 s windows.

| Sampling rate | Sensitivity | Specificity | AUROC |
|---|---|---|---|
| **30 Hz** (camera frame rate) | **97.0%** [352/363] | **95.6%** [283/296] | **0.9918** |
| 125 Hz (native) | 96.7% | 94.9% | 0.9902 |
| *MIT-BIH, ECG intervals* | *95%* | *97.4%* | *0.9903* |

95% CIs at 30 Hz: sensitivity [94.7%, 98.3%], specificity [92.6%, 97.4%]. Per subject,
**19/19 AF and 15/16 sinus correct**. Every window scored; none rejected.

Two things worth drawing out.

**Frame rate is not the bottleneck.** Decimating 125 Hz → 30 Hz moved sensitivity 0.3
points and specificity 0.7 — inside the confidence intervals. The sub-sample parabolic
interpolation was built precisely so that 33 ms frame quantisation would not dominate
beat timing, and this measures that rather than assuming it.

**The predictive values improve.** At this operating point, 2% prevalence gives PPV
31.0% and NPV 99.94%, against the 20.4% / 99.89% projected from BUT PPG. The projection
was conservative. The asymmetry that forces the decision rule is unchanged — a positive
is still wrong more often than not.

Caveats that belong with the numbers: these are contact pulse oximeters on perfused ICU
patients, not a camera held by a person, so this isolates the classifier and says nothing
about acquisition. Labels are per recording, so an AF patient who converted to sinus
mid-recording contributes mislabelled windows — 35 subjects is the effective sample size,
not 659.

### DeepBeat — sensitivity on PPG-derived intervals

Every sensitivity figure up to this point came from MIT-BIH, i.e. from intervals timed
off the QRS complex. DeepBeat is the one accessible corpus pairing PPG-derived intervals
with rhythm labels, so it answers the question the others cannot: does the classifier
still separate rhythms when the intervals come from a pulse wave?

Only the held-out `test.npz` split was downloaded; `train.npz` and `validate.npz` were
never accessed.

**On excellent-quality segments: sensitivity 91.5%, specificity 92.4%** — against 95%
and 97.4% on ECG intervals. Softer peaks and respiratory amplitude modulation cost about
three points of sensitivity and five of specificity, and no more.

| Signal quality | Segments | Sensitivity | Specificity |
|---|---|---|---|
| Excellent | 3,246 | 91.5% | 92.4% |
| Acceptable | 2,032 | 91.7% | 81.0% |
| Poor | 12,336 | 76.1% | 40.6% |

Performance falls monotonically with signal quality. 70% of this corpus is graded poor,
which is what continuous wrist wear looks like — not deliberate fingertip capture — so
the unstratified figures (78.9% / 56.6%, AUROC 0.718) describe the corpus more than the
classifier.

Applying the excellent-quality operating point at 2% prevalence gives **PPV 19.7%, NPV
99.81%** — within one point of the 20.4% / 99.89% projected independently from BUT PPG.
A third derivation, from a different corpus, a different anatomical site and a different
sensing modality, landing on the same numbers.

Two structural facts govern how these should be read. The rhythm label is **constant
within a subject**, so 17,617 segments are 22 observations, not 17,617; per subject on
excellent-quality segments the result is 4/4 fibrillatory and 6/6 sinus correct, which
is worth reporting only as counts. And the segments are 25 s against the deployed 60 s,
so entropy terms rest on ~29 intervals instead of ~70.

Two bugs were fixed in the harness before any number was trusted. The prep script's
shape heuristic picked the wrong axis on the `(n, 800, 1)` signal array; more seriously,
it took column 0 of the one-hot `rhythm` array, which would have **inverted every
label** without failing. It now argmaxes, asserts the arrays are genuinely one-hot, and
asserts AF is the minority class. Separately, the quality-label ordering was assumed
backwards; it was established empirically instead, from mean in-band power of 0.629 /
0.838 / 0.917 across classes 0/1/2, making class 0 *poor* and class 2 *excellent*. Taken
at face value the table above would have shown the classifier performing worse on
cleaner signal — a result that reads as a mystery rather than as the bug it was.

### Projected screening performance

The measured 28.4 ms interval error corresponds to ~25 ms per-beat dispersion, which the
tolerance table places at 92.4% specificity at 95% sensitivity.

| Population (AF prevalence) | PPV | NPV |
|---|---|---|
| General adults 55+ (2%) | 20.4% | 99.89% |
| Adults 65+ (5%) | 39.8% | 99.72% |
| Higher-risk 75+ (10%) | 58.3% | 99.40% |

These fall inside the 20–40% positive and near-100% negative predictive values reported
for validated smartphone screening applications, despite being derived independently.
They are what make the asymmetric decision rule mandatory rather than cautious.

---

## 8. Verification infrastructure

**Cross-implementation parity.** The deployed code is JavaScript; the reference is
Python. Agreement is enforced by replaying **10,764 reference-computed risk values** and
**358 reference-computed rhythm features** through the deployed code at a tolerance of
1e-12. Observed worst-case divergence: **1.1 × 10⁻¹⁶**.

**Reconstruction check.** `src/risk/validate.py` confirms the rebuilt model reproduces
the source paper's published five-year cohort risks to within 0.002 for both sexes and
both coefficient sets, using an independently seeded synthetic cohort.

**Parameter freeze.** SHA-256 over the six prediction-affecting files, verified before
and after every external evaluation.

### Full suite

| Suite | Assertions |
|---|---|
| `src/risk/validate.py` — model vs published paper | 11 |
| `tests/test_risk_model.py` — unit tests | 27 |
| `tests/test_risk_parity.js` — Python/JS parity | 10,764 |
| `tests/test_af_parity.js` — Python/JS parity | 358 |
| `tests/test_beat_matching.js` — validates the validator | 9 |
| `tests/test_rppg.js` — DSP | 20 |
| `tests/e2e_assess_test.py` — browser, three-step flow | 38 |
| `tests/e2e_risk5_test.py` — browser | 18 |
| `tests/e2e_camera_test.py` — browser capture path | pass |
| **Total** | **≈11,245** |

---

## 9. Reproducing everything

```bash
pip install numpy scipy scikit-learn pandas wfdb playwright
npm --version   # Node 18+ for the JS suites

# --- risk model ---
python3 src/risk/export_risk_model.py    # solves S0, exports constants + parity fixture
python3 src/risk/validate.py             # checks against the published paper

# --- AF classifier (downloads MIT-BIH annotations, ~2 min) ---
python3 src/af/build_afdb.py             # build feature windows
python3 src/af/train_af.py               # train + tolerance curve + export
python3 src/af/freeze.py                 # hash the parameters

# --- independent validation ---
python3 src/af/cache_butppg.py           # download BUT PPG (~30 min)
node tests/validate_butppg.js
node tests/validate_gdansk.js            # needs the Gdansk CSVs in src/af/gdansk/

# --- MIMIC PERform AF (open, no login) ---
curl -LO https://zenodo.org/records/15906524/files/mimic_perform_af_wfdb.zip
curl -LO https://zenodo.org/records/15906524/files/mimic_perform_non_af_wfdb.zip
unzip -q mimic_perform_af_wfdb.zip -d af && unzip -q mimic_perform_non_af_wfdb.zip -d nonaf
python3 src/af/prep_mimic_af.py af/mimic_perform_af_wfdb nonaf/mimic_perform_non_af_wfdb --fs 30
node tests/validate_mimic_af.js src/af/mimic_af_prepared_30hz.json

# --- DeepBeat (needs a free Synapse account; test split only) ---
pip install synapseclient
# create ~/.synapseConfig with [authentication] / authtoken = <personal access token>
synapse get syn22006407                      # test.npz, 108 MB
python3 src/af/prep_deepbeat.py test.npz
node tests/validate_deepbeat.js

# --- build the pages ---
python3 src/build_assess.py
python3 src/build_risk5.py
python3 src/build.py

# --- the paper ---
cd paper && pdflatex brainshield_ieee.tex && pdflatex brainshield_ieee.tex
# needs IEEEtran, newtx, cite, algorithms, carlisle, oberdiek, xpatch, xstring.
# a minimal toolchain: curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh
```

Third-party data is excluded from version control; each script re-fetches what it needs.
The Gdańsk CSVs are currently fetched manually from the upstream repository.

---

## 10. Prepared but not yet run

**Quality-gate behaviour across acquisition modalities.** The DeepBeat run scored
17,614 of 17,617 segments — it rejected three. It therefore accepted all 12,336
poor-quality segments and returned 40.6% specificity on them. Signal quality is the
dominant term in whether this system is right, so this matters more than any remaining
accuracy work.

One qualification before treating it as a verdict on the deployed gate: the validator
calls `RPPG.analyse` directly and never invokes `afQuality()`, which needs raw camera
frames for photometric stability and amplitude CV. So what this measures is that *the
path exercised here* filters almost nothing. Whether the real gate would have caught
these is untested — and if it would not, that is the highest-value fix outstanding.

**Recalibration of S₀ to Indian incidence.** The relative effects transport between
populations; the absolute baseline does not. This needs age- and sex-specific incidence
from the ICMR National Stroke Registry or GBD India.
`fsrp.solve_baseline_survival()` already accepts observed risk as its target, so it is a
data task rather than a modelling one. Note that age-specific Indian rates are reported
as broadly comparable to Western populations — the headline difference is largely an
age-structure effect, so a single scaling factor would be wrong.

**Camera exposure locking.** The capture requests resolution, frame rate and torch, but
does not lock auto-exposure or auto-white-balance, so the sensor hunts against the ~1%
pulsatile signal.

**Prospective pilot.** The remaining gap is end-to-end performance on this capture path
in the target population, with Fitzpatrick skin phototype recorded. An elective
cardioversion clinic is the most efficient setting: participants are in AF by
definition, before-and-after recordings give matched AF and sinus-rhythm pairs from the
same finger and handset, and a 12-lead ECG is performed anyway.
