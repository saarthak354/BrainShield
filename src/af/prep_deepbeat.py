"""
Convert the Stanford DeepBeat PPG corpus into the JSON the JS validator consumes.

DeepBeat (Torres-Soto & Ashley, 2020; Synapse syn21985690) holds wrist-worn PPG from
108 atrial fibrillation and 67 non-AF subjects, as 25-second segments sampled at 32 Hz
with a rhythm label and a three-level signal-quality label.

Why it matters here: it is the only accessible corpus that pairs PPG-derived intervals
with AF labels, so it supplies the one figure the rest of our validation cannot --
SENSITIVITY on photoplethysmography rather than on electrocardiographic intervals.

Structure of test.npz, as observed (the arrays are not documented upstream):
    signal      (17617, 800, 1) float64   25 s at 32 Hz, bandpassed, scaled to [0,1]
    rhythm      (17617, 2)      float32   one-hot; column 1 is the minority class
    qa_label    (17617, 3)      float32   one-hot; 0 POOR, 1 acceptable, 2 excellent
                                          (ordering established empirically: mean in-band
                                           power 0.629 / 0.838 / 0.917 across classes 0/1/2)
    parameters  (17617, 3)      object    (timestamp, session letter, subject id)

Three caveats, all reported in the output rather than hidden:
  * Segments are 25 s; the deployed classifier is trained on 60 s. At ~70 bpm that is
    ~28 intervals instead of ~70, so entropy estimates are noisier. Far less severe
    than the 10 s BUT PPG clips, but not the deployed configuration.
  * Wrist reflectance PPG is not fingertip transmission PPG. This validates the
    CLASSIFIER on real pulse-wave intervals; it does not validate our camera capture.
  * The rhythm label is CONSTANT WITHIN A SUBJECT -- every segment from an AF subject
    carries the AF label. The test split holds 22 subjects (16 non-AF, 6 AF), so the
    effective sample size is 22, not 17617. Per-segment intervals would be far too
    narrow; the validator aggregates per subject for this reason.

Usage:  python3 src/af/prep_deepbeat.py path/to/test.npz [--max N]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deepbeat_prepared.json")
FS = 32.0

# Upstream names. We key off these rather than guessing from shape: the earlier
# shape heuristic picked the wrong axis on (n, samples, 1) signal arrays.
SIG, LAB, QUAL, PARAM = "signal", "rhythm", "qa_label", "parameters"


def onehot_to_index(a, name):
    """One-hot -> class index, refusing anything that is not actually one-hot."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise SystemExit(f"{name}: expected 2-D one-hot, got shape {a.shape}")
    rowsum = a.sum(axis=1)
    if not np.allclose(rowsum, 1.0, atol=1e-6):
        bad = int((np.abs(rowsum - 1.0) > 1e-6).sum())
        raise SystemExit(f"{name}: {bad} of {len(a)} rows are not one-hot; refusing to guess")
    return a.argmax(axis=1)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    cap = None
    if "--max" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--max") + 1])

    print(f"Loading {path} ...")
    z = np.load(path, allow_pickle=True)
    missing = [k for k in (SIG, LAB, QUAL, PARAM) if k not in z]
    if missing:
        raise SystemExit(f"missing arrays {missing}; found {list(z.keys())}")

    sig = np.asarray(z[SIG], dtype=float)
    if sig.ndim == 3:
        if sig.shape[2] != 1:
            raise SystemExit(f"signal has {sig.shape[2]} channels; expected 1")
        sig = sig[:, :, 0]
    lab = onehot_to_index(z[LAB], LAB)
    qual = onehot_to_index(z[QUAL], QUAL)
    subj = np.array([str(x).strip() for x in np.asarray(z[PARAM], dtype=object)[:, 2]])

    n, nsamp = sig.shape
    print(f"\n  segments: {n}   samples each: {nsamp} ({nsamp/FS:.1f} s at {FS:.0f} Hz)")
    print(f"  rhythm classes : {dict(zip(*[x.tolist() for x in np.unique(lab, return_counts=True)]))}")
    print(f"  quality classes: {dict(zip(*[x.tolist() for x in np.unique(qual, return_counts=True)]))}")

    # Label polarity is asserted, not assumed: the AF class must be the minority one
    # AND must partition subjects cleanly. If either fails we stop rather than
    # silently report an inverted sensitivity.
    counts = np.bincount(lab, minlength=2)
    af_class = int(np.argmin(counts))
    if af_class != 1:
        raise SystemExit(f"expected the minority rhythm class to be column 1, got {af_class}")

    print(f"\n  subjects: {len(np.unique(subj))}")
    mixed = []
    for s in np.unique(subj):
        m = subj == s
        f = lab[m].mean()
        if 0.0 < f < 1.0:
            mixed.append((s, f, int(m.sum())))
    n_af_subj = sum(1 for s in np.unique(subj) if lab[subj == s].mean() > 0.5)
    print(f"    AF subjects {n_af_subj}   non-AF subjects {len(np.unique(subj)) - n_af_subj}")
    if mixed:
        print(f"    NOTE: {len(mixed)} subjects carry mixed labels: {mixed[:5]}")
    else:
        print("    labels are constant within every subject "
              "(effective n = subjects, not segments)")

    idx = np.arange(n)
    if cap and len(idx) > cap:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, cap, replace=False))
        print(f"\n  subsampled to {cap} segments (seed 0)")

    recs, dropped = [], 0
    for i in idx:
        v = sig[i]
        if not np.all(np.isfinite(v)) or np.std(v) == 0:
            dropped += 1
            continue
        recs.append({
            "id": int(i),
            "subject": subj[i],
            "fs": FS,
            "ppg": [round(float(x), 5) for x in v],
            "af": int(lab[i] == af_class),
            "quality": int(qual[i]),
        })
    if dropped:
        print(f"  dropped {dropped} flat or non-finite segments")

    json.dump({"_source": "Stanford DeepBeat (Synapse syn21985690), test split",
               "_fs": FS, "_segment_seconds": nsamp / FS,
               "_af_class_column": af_class,
               "records": recs}, open(OUT, "w"))
    print(f"\nwrote {OUT}  ({len(recs)} segments, {os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
