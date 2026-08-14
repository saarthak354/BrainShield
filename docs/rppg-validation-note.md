# BrainShield — Camera Pulse Measurement (rPPG): Implementation & Validation Note

*Generated 2026-08-14. Companion to the BrainShield v1 paper.*

## What was built

A working remote-photoplethysmography (rPPG) module integrated into the BrainShield
website. It recovers a real pulse waveform from camera frames — a physical measurement,
as distinct from the questionnaire inputs that drive the risk model.

**Two capture modes:**

- **Fingertip + flash (default).** Finger covers the rear lens with the torch on.
  Contact PPG under active illumination. Chosen as the default specifically because it
  floods tissue with light rather than depending on ambient reflection, which raises
  signal amplitude for everyone and disproportionately helps where skin reflectance is
  lower. Algorithm: adaptive colour-channel selection by in-band SNR.
- **Face (contactless).** Algorithm: POS (Wang et al., IEEE TBME 2017) and CHROM
  (de Haan & Jeanne, 2013), with whichever yields the better SNR selected per capture.

**Processing chain:** per-frame ROI mean RGB -> uniform resampling (browser frame
delivery jitters; FFT estimation assumes uniform spacing) -> POS/CHROM projection ->
zero-phase FFT bandpass 0.7-4.0 Hz (42-240 bpm) -> Hann-windowed periodogram with
parabolic peak interpolation -> HR, SNR, HRV.

Everything runs client-side. No video or image data leaves the browser.

## The quality-gating problem, and why SNR alone was insufficient

The first implementation gated reliability on spectral SNR alone. Adversarial testing
showed this **fails dangerously**: when the estimator locks onto a motion artifact, that
artifact is itself a clean spectral peak, so SNR reports an excellent signal while the
reported heart rate is simply wrong. Under simulated darker-skin, in-motion conditions
this produced **13 confidently-wrong readings out of 60** (error > 5 bpm while displaying
a confident result).

Two additional independent gates were added:

1. **Temporal consistency** — HR is estimated independently in overlapping 8-second
   windows; the median absolute deviation across windows must be small. A true pulse is
   stable; motion artifacts wander in frequency.
2. **Cross-algorithm agreement** — POS and CHROM rest on different assumptions, so a
   genuine pulse appears in both. Disagreement beyond 8 bpm downgrades confidence.

With all three gates, silent failures fell to **0 out of 60** under the same conditions.
Readings that cannot be trusted are labelled "Poor signal — not reliable" rather than
shown as a number.

## Known irreducible limitation

A rhythmic disturbance at a steady, heart-rate-plausible frequency carrying a pulse-like
chromatic signature (a moving vehicle, rocking, some flickering lights) is physically
ambiguous to a single camera. Testing confirms face mode can lock onto it. This is
documented, not solved, and is a further reason fingertip mode is the default.

## Skin-tone accuracy — stated plainly

Published findings (not our measurements): a 2025 review of 100 rPPG studies reports mean
absolute error rising from ~5.2 bpm on Fitzpatrick I-III to ~14.1 bpm on Fitzpatrick V-VI,
with a cited meta-analysis reporting more than twofold degradation; those datasets
under-represent darker skin tones (<25% of participants). A UCLA group concluded a proper
fix requires additional hardware (camera + radar fusion), not software alone.

Given deployment in India, this is the central engineering risk for this feature. Our
mitigations (fingertip+flash default, POS/CHROM, three-gate quality control) are
partial. **This module has not been validated against a medical-grade reference on real
human subjects of any skin tone.** The website states this to the user directly.

## Validation performed

Two independent suites. Both use synthetic signals with known ground-truth heart rates.
Neither is a human validation study.

### 1. DSP unit suite (Node)

```
=== 1. Heart-rate recovery across the physiological range (face mode, POS/CHROM) ===
   45 bpm ->  est 44.9  err 0.13  SNR 21.1 dB  [good]  via POS
  PASS  45 bpm recovered within 2 bpm
   55 bpm ->  est 55.0  err 0.01  SNR 21.4 dB  [good]  via POS
  PASS  55 bpm recovered within 2 bpm
   62 bpm ->  est 62.0  err 0.01  SNR 21.5 dB  [good]  via POS
  PASS  62 bpm recovered within 2 bpm
   72 bpm ->  est 72.0  err 0.01  SNR 20.7 dB  [good]  via POS
  PASS  72 bpm recovered within 2 bpm
   85 bpm ->  est 85.0  err 0.01  SNR 20.8 dB  [good]  via POS
  PASS  85 bpm recovered within 2 bpm
  100 bpm ->  est 100.0  err 0.02  SNR 21.8 dB  [good]  via POS
  PASS  100 bpm recovered within 2 bpm
  120 bpm ->  est 120.0  err 0.02  SNR 20.4 dB  [good]  via POS
  PASS  120 bpm recovered within 2 bpm
  150 bpm ->  est 150.0  err 0.00  SNR 19.6 dB  [good]  via POS
  PASS  150 bpm recovered within 2 bpm

=== 2. Skin-tone / melanin robustness (simulated, face mode) ===
  melanin 0 = lightest, 1 = darkest. Attenuates AC pulse amplitude,
  which is the documented cause of real-world rPPG accuracy loss.

  melanin 0.00  ->  MAE 0.02 bpm   mean SNR 21.4 dB   usable 4/4
  melanin 0.25  ->  MAE 0.02 bpm   mean SNR 19.0 dB   usable 4/4
  melanin 0.50  ->  MAE 0.03 bpm   mean SNR 15.9 dB   usable 4/4
  melanin 0.70  ->  MAE 0.05 bpm   mean SNR 12.9 dB   usable 4/4
  melanin 0.85  ->  MAE 0.08 bpm   mean SNR 9.9 dB   usable 4/4
  melanin 0.95  ->  MAE 0.11 bpm   mean SNR 7.4 dB   usable 4/4
  PASS  lightest simulated skin: MAE < 1.5 bpm
  PASS  degradation with melanin is reproduced (expected — matches published findings)

=== 2b. Same sweep under REALISTIC conditions (motion artifact present) ===
  Motion is the dominant real-world noise source and is NOT reduced by melanin,
  so the artifact-to-pulse ratio worsens as the pulse signal shrinks.

  melanin 0.00  ->  MAE 0.14 bpm   SNR 8.8 dB   within-5bpm 10/10   flagged-poor 0/10   SILENT-FAIL 0
  melanin 0.25  ->  MAE 0.20 bpm   SNR 6.0 dB   within-5bpm 10/10   flagged-poor 0/10   SILENT-FAIL 0
  melanin 0.50  ->  MAE 0.29 bpm   SNR 2.8 dB   within-5bpm 10/10   flagged-poor 0/10   SILENT-FAIL 0
  melanin 0.70  ->  MAE 0.51 bpm   SNR -0.2 dB   within-5bpm 10/10   flagged-poor 5/10   SILENT-FAIL 0
  melanin 0.85  ->  MAE 11.68 bpm   SNR -1.8 dB   within-5bpm 5/10   flagged-poor 9/10   SILENT-FAIL 0
  melanin 0.95  ->  MAE 15.40 bpm   SNR -1.8 dB   within-5bpm 4/10   flagged-poor 9/10   SILENT-FAIL 0

  Silent failures overall (confidently wrong by >5 bpm): 0/60
  PASS  realistic conditions: error grows with melanin (matches published direction)
  PASS  darkest tone triggers more POOR-quality flags (tool warns rather than lies)
  PASS  silent-failure rate <= 5% (wrong readings are flagged, not shown confidently)

=== 2c. Worst case: perfectly METRONOMIC in-band interferer (irreducible) ===
  A rhythmic disturbance at a fixed, heart-rate-plausible frequency with a
  pulse-like colour signature is physically ambiguous — no single-camera
  algorithm can separate it from a real pulse. Documented, not 'solved'.

  melanin 0.85  58 bpm -> est 90.2  err 32.2  [poor]
  melanin 0.85  72 bpm -> est 90.3  err 18.3  [fair]
  melanin 0.85  90 bpm -> est 90.1  err 0.1  [good]
  melanin 0.95  58 bpm -> est 90.1  err 32.1  [fair]
  melanin 0.95  72 bpm -> est 90.3  err 18.3  [fair]
  melanin 0.95  90 bpm -> est 90.2  err 0.2  [fair]

  Flagged poor: 1/6   Silent failures: 3/6
  -> This is why finger+torch is the recommended default (see section 3):
     contact PPG raises pulse amplitude far above any motion artifact.

=== 3. Finger-on-lens contact mode (strong signal, torch) ===
   60 bpm ->  est 60.0  err 0.00  SNR 24.0 dB  [good]  via contact PPG (blue channel)
  PASS  finger mode 60 bpm recovered within 2 bpm
   72 bpm ->  est 72.0  err 0.00  SNR 21.0 dB  [good]  via contact PPG (blue channel)
  PASS  finger mode 72 bpm recovered within 2 bpm
   95 bpm ->  est 95.0  err 0.00  SNR 21.1 dB  [good]  via contact PPG (green channel)
  PASS  finger mode 95 bpm recovered within 2 bpm
  130 bpm ->  est 130.0  err 0.00  SNR 20.6 dB  [good]  via contact PPG (green channel)
  PASS  finger mode 130 bpm recovered within 2 bpm

=== 4. Rejects garbage instead of inventing a number ===
  pure noise -> quality "poor"  SNR -6.2 dB
  PASS  pure noise is flagged POOR (not reported as a valid reading)
  2s capture -> ok=false  Not enough data — need at least a few seconds of steady capture.
  PASS  too-short capture is refused

=== 5. Irregular frame timing (browser jitter) ===
  jittered/dropped frames (528 samples) -> est 78.0 bpm (truth 78)  err 0.04
  PASS  survives irregular frame timing (resampling works)

================  20 passed, 0 failed  ================
```

### 2. End-to-end browser suite

Synthetic frames with per-pixel dithering noise are pushed through a real MediaStream
(`canvas.captureStream()`), exercising the genuine path: video element -> drawImage ->
getImageData -> per-frame RGB averaging -> browser frame-timing jitter -> resampling ->
POS/CHROM -> quality gating -> rendered UI.

```
===== END-TO-END BROWSER CAPTURE TEST =====
(synthetic frames through a real MediaStream — not human validation)

  face mode, lighter skin, still                   truth  72 -> est 72   err 0.0
       [Good signal]  SNR 31.8 dB | stability ±0.0 bpm | POS | frames 750 | beats 31
       -> PASS

  finger mode (flash-lit contact)                  truth  96 -> est 96   err 0.0
       [Good signal]  SNR 33.7 dB | stability ±0.0 bpm | contact PPG (blue channel) | frames 748 | beats 40
       -> PASS

  face mode, SIMULATED dark skin (melanin 0.85)    truth  64 -> est 64   err 0.0
       [Good signal]  SNR 21.2 dB | stability ±0.0 bpm | POS | frames 749 | beats 27
       -> PASS

  finger mode, SIMULATED dark skin (melanin 0.85)  truth  78 -> est 78   err 0.0
       [Good signal]  SNR 37.8 dB | stability ±0.0 bpm | contact PPG (green channel) | frames 751 | beats 33
       -> PASS

  face, dark skin + MOTION (should warn, not lie)  truth  68 -> est 68   err 0.0
       [Good signal]  SNR 16.0 dB | stability ±0.0 bpm | POS | frames 750 | beats 29
       -> PASS (accurate)

===== ALL OK =====
```

**What these tests do not cover:** real human skin, real camera hardware, JPEG/H.264
compression artifacts, auto-exposure and auto-white-balance (both significant real-world
confounders that a canvas-sourced stream does not reproduce), torch control, and
getUserMedia device negotiation. Chromium's fake-camera device was unavailable in the
build environment, which is why frame injection was used instead.

## Relationship to the risk model — deliberately kept separate

The heart-rate reading is **not** fed into the stroke risk score. The XGBoost model was
trained and validated exclusively on BRFSS questionnaire variables; silently mixing a
physiological measurement into its output would misrepresent what that model's reported
AUROC of 0.828 actually validated. The two are presented side by side as distinct things.

Likewise, SDNN and RMSSD are displayed as raw variability numbers explicitly labelled
experimental. They are **not** an atrial-fibrillation test. The ~97%-accuracy AF results
in the literature come from classifiers trained against ECG-confirmed diagnoses in
hundreds of patients; no such classifier has been built or validated here.

## Honest next step

The single highest-value action is a small real-world validation: 15-25 volunteers across
a range of skin tones, each measured simultaneously with the tool and a reference pulse
oximeter, reporting Bland-Altman agreement and error stratified by skin tone. That is
cheap, requires no gated dataset, and would convert every claim above from "synthetically
tested" to "measured" — and would itself be a publishable contribution given how
under-represented darker skin tones are in existing rPPG datasets.
