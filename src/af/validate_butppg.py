"""
Independent validation of the FROZEN AF/PPG pipeline on real phone-camera recordings.

BUT PPG v2.0.0 (PhysioNet): subjects held a finger over the rear camera with the LED
lit, on a Xiaomi Mi9 or Huawei P20 Pro, while a Bittium Faros recorded a 1000 Hz ECG
whose QRS positions were detected and manually verified. That ECG is the ground truth.

WHAT THIS CAN AND CANNOT MEASURE
  CAN: camera -> PPG -> quality gate -> beat detection -> RR intervals.
       Also the FALSE POSITIVE rate, because this cohort is young volunteers with
       essentially no atrial fibrillation, so every "irregular" call is an error.
  CANNOT: sensitivity. There are no AF labels and no AF cases to find.

TWO DATA QUIRKS, both handled below and both reported in the output:
  1. Records before 112001 have a transposed header (nsig/nsamp swapped) AND an
     all-zero .dat. They are unusable and are counted separately, not silently dropped.
  2. Recordings are 10 seconds. The shipped classifier requires 60-second windows,
     so running it here is OFF-LABEL and pessimistic: ~12 beats instead of ~70 makes
     the interval statistics much noisier, which inflates irregular calls. It is
     reported as a stress-test upper bound on the false-positive rate, not as the
     shipped system's specificity.

The algorithm is frozen (see FROZEN.json). This script only measures it.
"""
import os, re, sys, json, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "butppg_results.json")
BASE = "https://physionet.org/files/butppg/2.0.0"
MAX_RECORDS = int(os.environ.get("BUTPPG_N", "400"))

FS_ECG = 1000.0
MOTION = {0: "rest", 1: "finger pressure", 2: "moving on lens", 3: "walking",
          4: "coughing", 5: "laughing", 6: "changing light", 7: "talking"}


def fetch(url, timeout=45):
    return urllib.request.urlopen(url, timeout=timeout).read()


# ---------------------------------------------------------------- signal ----
def bandpass(x, fs, lo=0.7, hi=3.5):
    n = len(x)
    X = np.fft.rfft(x - np.mean(x))
    f = np.fft.rfftfreq(n, 1 / fs)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n)


def find_peaks_refined(sig, fs, max_bpm=180):
    """Peak picking + parabolic sub-sample refinement. Mirrors af_core.js exactly."""
    min_dist = max(2, int(0.6 * fs * 60.0 / max_bpm))
    peaks = []
    for i in range(1, len(sig) - 1):
        if sig[i] > sig[i - 1] and sig[i] >= sig[i + 1] and sig[i] > 0:
            if peaks and i - peaks[-1] < min_dist:
                if sig[i] > sig[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    out = []
    for p in peaks:
        if 0 < p < len(sig) - 1:
            y0, y1, y2 = sig[p - 1], sig[p], sig[p + 1]
            d = y0 - 2 * y1 + y2
            delta = 0.5 * (y0 - y2) / d if d < 0 else 0.0
            if not np.isfinite(delta) or abs(delta) > 0.5:
                delta = 0.0
            out.append(p + delta)
        else:
            out.append(float(p))
    return np.array(out)


def rr_from(peaks, fs):
    if len(peaks) < 2:
        return np.array([])
    rr = np.diff(peaks) / fs * 1000.0
    return rr[(rr > 250) & (rr < 2500)]


def detrend_var(a):
    n = len(a)
    if n < 3:
        return 0.0
    m = np.mean(a)
    idx = np.arange(n) - (n - 1) / 2
    sx = np.sum(idx ** 2)
    slope = np.sum(idx * (a - m)) / sx if sx > 0 else 0.0
    d = a - m - slope * idx
    return float(np.var(d))


# ------------------------------------------------------- frozen gate/model ----
AF_MODEL = json.load(open(os.path.join(HERE, "af_model_export.json")))


def af_quality(raw_r, wave, fs):
    """Mirrors af_capture.js afQuality(), minus the 45s duration floor."""
    vt = detrend_var(raw_r)
    frac = min(1.0, detrend_var(wave) / vt) if vt > 0 else 0.0
    if frac < 0.02:
        return False, "no pulse detectable (in-band fraction %.3f)" % frac, frac
    lum = raw_r
    stability = (np.std(lum - np.mean(lum)) / np.mean(lum) * 100) if np.mean(lum) > 0 else 99
    if stability > 8:
        return False, "brightness unstable (%.1f%%)" % stability, frac
    peaks = find_peaks_refined(wave, fs)
    if len(peaks) < 12:
        return False, "too few beats (%d)" % len(peaks), frac
    amps = np.array([abs(wave[int(round(p))]) for p in peaks
                     if 0 <= int(round(p)) < len(wave)])
    amp_cv = float(np.std(amps) / np.mean(amps)) if len(amps) and np.mean(amps) > 0 else 99
    if amp_cv > 0.75:
        return False, "beat amplitude inconsistent (CV %.2f)" % amp_cv, frac
    allrr = np.diff(peaks) / fs * 1000.0
    bad = np.mean((allrr <= 250) | (allrr >= 2500)) if len(allrr) else 1.0
    if bad > 0.2:
        return False, "%.0f%% of beats untimeable" % (100 * bad), frac
    return True, "ok", frac


def af_features(rr):
    sys.path.insert(0, HERE)
    from build_afdb import features
    return features(rr)


def classify(feats):
    z = AF_MODEL["intercept"]
    for i, name in enumerate(AF_MODEL["feature_names"]):
        v = feats.get(name)
        if v is None or not np.isfinite(v):
            return None
        z += AF_MODEL["coef"][i] * ((v - AF_MODEL["mean"][i]) / AF_MODEL["scale"][i])
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ main ----
def parse_header(txt):
    lines = [l for l in txt.splitlines() if l.strip()]
    parts = lines[0].split()
    nsig, fs, nsamp = int(parts[1]), float(parts[2]), int(parts[3])
    gains, baselines, names = [], [], []
    for l in lines[1:1 + nsig]:
        m = re.search(r"\s(-?[\d.]+)\((-?\d+)\)", l)
        if m:
            gains.append(float(m.group(1))); baselines.append(float(m.group(2)))
        else:
            gains.append(1.0); baselines.append(0.0)
        names.append(l.split()[-1])
    return nsig, fs, nsamp, gains, baselines, names


def main():
    print("Loading metadata...")
    qual, spot, motion = {}, {}, {}
    for name in ("quality-hr-ann.csv", "subject-info.csv"):
        txt = fetch(f"{BASE}/{name}", 120).decode("utf8", "ignore")
        lines = [l for l in txt.splitlines() if l.strip()]
        hdr = [h.strip().lower().lstrip("﻿") for h in lines[0].split(",")]
        for l in lines[1:]:
            p = [c.strip() for c in l.split(",")]
            if not p or not p[0]:
                continue
            if name.startswith("quality"):
                try: qual[p[0]] = int(float(p[1]))
                except Exception: pass
            else:
                if "ear/finger" in hdr:
                    k = hdr.index("ear/finger")
                    if k < len(p) and p[k]:
                        try: spot[p[0]] = int(float(p[k]))
                        except Exception: pass
                if "motion" in hdr:
                    k = hdr.index("motion")
                    if k < len(p) and p[k]:
                        try: motion[p[0]] = int(float(p[k]))
                        except Exception: pass
    print(f"  quality={len(qual)} spot={len(spot)} motion={len(motion)}")

    idx = fetch(f"{BASE}/", 120).decode("utf8", "ignore")
    recs = sorted(set(re.findall(r'href="(\d{6})/"', idx)))
    print(f"  {len(recs)} records listed\n")

    rows = []
    counts = {"listed": len(recs), "ear_skipped": 0, "malformed_or_empty": 0,
              "fetch_error": 0, "attempted": 0}

    for rid in recs:
        if counts["attempted"] >= MAX_RECORDS:
            break
        if spot.get(rid) == 0:
            counts["ear_skipped"] += 1
            continue
        try:
            hea = fetch(f"{BASE}/{rid}/{rid}_PPG.hea", 30).decode("utf8", "ignore")
            nsig, fs, nsamp, gains, bases, names = parse_header(hea)
        except Exception:
            counts["fetch_error"] += 1
            continue
        # the pre-112001 records declare nsig=300/nsamp=1 and hold only zeros
        if nsig != 3 or nsamp < 150:
            counts["malformed_or_empty"] += 1
            continue
        try:
            raw = fetch(f"{BASE}/{rid}/{rid}_PPG.dat", 30)
            qrs_txt = fetch(f"{BASE}/{rid}/{rid}.qrs", 30)
        except Exception:
            counts["fetch_error"] += 1
            continue

        d = np.frombuffer(raw, dtype="<i2").astype(float)
        if len(d) < nsig * nsamp or not np.count_nonzero(d):
            counts["malformed_or_empty"] += 1
            continue
        d = d[:nsig * nsamp].reshape(nsamp, nsig)
        chans = [(d[:, i] - bases[i]) / gains[i] for i in range(nsig)]
        R, G, B = chans[0], chans[1], chans[2]

        # ECG ground truth: parse the WFDB annotation file for QRS sample positions
        try:
            import wfdb
            ann = wfdb.rdann(rid, "qrs", pn_dir=f"butppg/{rid}")
            qsamp = np.asarray(ann.sample, dtype=float)
        except Exception:
            counts["fetch_error"] += 1
            continue
        ref_rr = rr_from(qsamp, FS_ECG)
        if len(ref_rr) < 3:
            counts["malformed_or_empty"] += 1
            continue

        counts["attempted"] += 1

        # ---- frozen pipeline ----
        wave = bandpass(R, fs)
        ok, reason, frac = af_quality(R, wave, fs)
        row = {"record": rid, "quality_label": qual.get(rid),
               "motion": motion.get(rid), "gate_ok": bool(ok), "gate_reason": reason,
               "in_band_fraction": float(frac), "n_ref_rr": int(len(ref_rr)),
               "hr_ecg": float(60000 / np.mean(ref_rr))}

        if ok:
            peaks = find_peaks_refined(wave, fs)
            ppg_rr = rr_from(peaks, fs)
            raw_rr = rr_from(np.round(peaks), fs)
            row["n_ppg_rr"] = int(len(ppg_rr))
            if len(ppg_rr) >= 3:
                k = min(len(ppg_rr), len(ref_rr))
                err = ppg_rr[:k] - ref_rr[:k]
                row["rr_mae"] = float(np.mean(np.abs(err)))
                row["rr_rmse"] = float(np.sqrt(np.mean(err ** 2)))
                kr = min(len(raw_rr), len(ref_rr))
                if kr >= 3:
                    row["rr_mae_no_interp"] = float(np.mean(np.abs(raw_rr[:kr] - ref_rr[:kr])))
                row["hr_ppg"] = float(60000 / np.mean(ppg_rr))
                row["hr_err"] = abs(row["hr_ppg"] - row["hr_ecg"])
                feats = af_features(ppg_rr)
                p = classify(feats)
                row["af_prob"] = None if p is None else float(p)
                row["af_call"] = ("irregular" if (p is not None and p >= AF_MODEL["threshold_rule_out"])
                                  else ("regular" if p is not None else "inconclusive"))
            else:
                row["af_call"] = "inconclusive"
        else:
            row["af_call"] = "rejected"
        rows.append(row)
        if counts["attempted"] % 25 == 0:
            print(f"  processed {counts['attempted']}...")

    json.dump({"counts": counts, "rows": rows,
               "model_threshold": AF_MODEL["threshold_rule_out"],
               "_caveat_10s": "Recordings are 10s; the shipped classifier needs 60s. "
                              "Irregular-call rates here are a pessimistic upper bound."},
              open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}  ({len(rows)} usable records)")


if __name__ == "__main__":
    main()
