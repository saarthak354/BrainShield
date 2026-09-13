"""
Phase 0, Stage A: can a classical RR-interval classifier separate AF from non-AF?

Uses MIT-BIH Atrial Fibrillation Database (PhysioNet 'afdb'): 25 records, ~10h each,
with beat annotations (.qrs) and rhythm annotations (.atr) marking AF episodes.

Only annotations are downloaded -- the raw ECG is never needed, since the classifier
operates on RR intervals. This keeps the download tiny.

Output: afdb_windows.npz, one row per analysis window with features + label + record id.
"""
import os, sys, json
import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "afdb_windows.npz")

RECORDS = "00735 03665 04015 04043 04048 04126 04746 04908 04936 05091 05121 05261 " \
          "06426 06453 06995 07162 07859 07879 07910 08215 08219 08378 08405 08434 08455".split()

WINDOW_SEC = 60          # analysis window; commercial apps use 60s-5min
STEP_SEC = 30            # hop between windows
MIN_BEATS = 20           # a window needs enough beats for the statistics to mean anything
PURITY = 0.9             # a window is labelled AF only if >=90% of its beats are in AF


# ---------------------------------------------------------------- features ----
def shannon_entropy(rr, bins=16):
    """Shannon entropy of the RR histogram. AF spreads mass across many bins."""
    if len(rr) < 4:
        return np.nan
    lo, hi = np.percentile(rr, [1, 99])
    if hi <= lo:
        return 0.0
    h, _ = np.histogram(rr, bins=bins, range=(lo, hi))
    p = h / max(1, h.sum())
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(bins))   # normalised to [0,1]


def sample_entropy(rr, m=2, r_frac=0.2):
    """Sample entropy: regularity of the sequence. Higher = less predictable."""
    n = len(rr)
    if n < m + 2:
        return np.nan
    x = np.asarray(rr, dtype=float)
    r = r_frac * np.std(x)
    if r <= 0:
        return 0.0

    def count(mm):
        # embed into mm-dim vectors, count pairs within Chebyshev distance r
        k = n - mm + 1
        if k < 2:
            return 0
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:k]
        c = 0
        for i in range(k - 1):
            d = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
            c += int(np.sum(d <= r))
        return c

    a, b = count(m + 1), count(m)
    # Richman & Moorman leave SampEn undefined when no template pairs match, which happens
    # precisely when the series is MOST irregular -- i.e. in the clearest atrial
    # fibrillation. Returning NaN there made the classifier reject its easiest cases and,
    # worse, silently dropped those windows from training. Use the conventional upper
    # bound instead, so "too irregular to find any matches" maps to "maximum entropy".
    upper = float(-np.log(2.0 / ((n - m - 1) * (n - m)))) if n > m + 1 else 0.0
    if a == 0 or b == 0:
        return upper
    return float(min(-np.log(a / b), upper))


def poincare(rr):
    """SD1 (short-term, beat-to-beat) and SD2 (long-term) of the Poincare plot."""
    if len(rr) < 3:
        return np.nan, np.nan
    d = np.diff(rr)
    sd1 = float(np.sqrt(np.var(d, ddof=1) / 2)) if len(d) > 1 else np.nan
    sd = float(np.var(rr, ddof=1)) if len(rr) > 1 else np.nan
    sd2sq = 2 * sd - (sd1 ** 2) if not np.isnan(sd1) else np.nan
    sd2 = float(np.sqrt(sd2sq)) if (sd2sq is not None and sd2sq > 0) else np.nan
    return sd1, sd2


def features(rr):
    """rr in milliseconds."""
    rr = np.asarray(rr, dtype=float)
    mean_rr = float(np.mean(rr))
    d = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan
    sd1, sd2 = poincare(rr)
    return {
        "mean_rr": mean_rr,
        "hr": 60000.0 / mean_rr,
        "rmssd": rmssd,
        "rmssd_norm": rmssd / mean_rr,                 # scale-free: the key AF feature
        "cv": float(np.std(rr, ddof=1)) / mean_rr,
        "prr50": float(np.mean(np.abs(d) > 50)) if len(d) else np.nan,
        "prr70": float(np.mean(np.abs(d) > 70)) if len(d) else np.nan,
        "shannon": shannon_entropy(rr),
        "sampen": sample_entropy(rr),
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2": (sd1 / sd2) if (sd2 and not np.isnan(sd2) and sd2 > 0) else np.nan,
    }


FEATURE_NAMES = ["rmssd_norm", "cv", "prr50", "prr70", "shannon", "sampen", "sd1_sd2", "hr"]


# ------------------------------------------------------------------ build ----
def rhythm_intervals(ann, n_samples_hint):
    """Turn .atr rhythm annotations into (start_sample, end_sample, is_af) spans."""
    spans, cur, start = [], None, None
    for s, note in zip(ann.sample, ann.aux_note):
        note = (note or "").strip()
        if not note.startswith("("):
            continue
        if cur is not None:
            spans.append((start, s, cur == "AFIB"))
        cur, start = note[1:], s
    if cur is not None:
        spans.append((start, n_samples_hint, cur == "AFIB"))
    return spans


def main():
    rows, labels, recs = [], [], []
    for rec in RECORDS:
        try:
            qrs = wfdb.rdann(rec, "qrs", pn_dir="afdb")
            atr = wfdb.rdann(rec, "atr", pn_dir="afdb")
        except Exception as e:
            print(f"  {rec}: skipped ({e})")
            continue

        fs = qrs.fs
        beats = np.asarray(qrs.sample, dtype=float)
        if len(beats) < 100:
            print(f"  {rec}: too few beats"); continue

        spans = rhythm_intervals(atr, beats[-1] + 1)
        # per-beat AF flag
        af_flag = np.zeros(len(beats), dtype=bool)
        for a, b, is_af in spans:
            if is_af:
                af_flag |= (beats >= a) & (beats < b)

        win, step = WINDOW_SEC * fs, STEP_SEC * fs
        t0, tend = beats[0], beats[-1]
        n_win = 0
        s = t0
        while s + win <= tend:
            sel = (beats >= s) & (beats < s + win)
            bt = beats[sel]
            if len(bt) >= MIN_BEATS:
                rr = np.diff(bt) / fs * 1000.0
                rr = rr[(rr > 250) & (rr < 2500)]       # physiologic plausibility
                if len(rr) >= MIN_BEATS - 1:
                    frac_af = float(np.mean(af_flag[sel]))
                    if frac_af >= PURITY or frac_af <= (1 - PURITY):
                        f = features(rr)
                        v = [f[k] for k in FEATURE_NAMES]
                        if not any(np.isnan(v)):
                            rows.append(v)
                            labels.append(1 if frac_af >= PURITY else 0)
                            recs.append(rec)
                            n_win += 1
            s += step
        print(f"  {rec}: {n_win} windows")

    X = np.array(rows, dtype=float)
    y = np.array(labels, dtype=int)
    g = np.array(recs)
    np.savez(OUT, X=X, y=y, groups=g, feature_names=np.array(FEATURE_NAMES))
    print(f"\nwrote {OUT}")
    print(f"  {X.shape[0]} windows, {X.shape[1]} features, "
          f"{y.sum()} AF ({100*y.mean():.1f}%), {len(set(g))} records")


if __name__ == "__main__":
    main()
