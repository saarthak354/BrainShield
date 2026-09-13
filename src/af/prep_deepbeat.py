"""
Convert the Stanford DeepBeat PPG corpus into the JSON the JS validator consumes.

DeepBeat (Torres-Soto & Ashley, 2020; Synapse syn21985690) holds wrist-worn PPG from
108 atrial fibrillation and 67 non-AF subjects, as 25-second segments sampled at 32 Hz
with binary rhythm labels and a signal-quality label.

Why it matters here: it is the only accessible corpus that pairs PPG-derived intervals
with AF labels, so it supplies the one figure the rest of our validation cannot --
SENSITIVITY on photoplethysmography rather than on electrocardiographic intervals.

Two caveats, both reported in the output rather than hidden:
  * Segments are 25 s; the deployed classifier is trained on 60 s. At ~70 bpm that is
    ~28 intervals instead of ~70, so entropy estimates are noisier. Far less severe
    than the 10 s BUT PPG clips, but not the deployed configuration.
  * Wrist reflectance PPG is not fingertip transmission PPG. This validates the
    CLASSIFIER on real pulse-wave intervals; it does not validate our camera capture.

Usage:  python3 src/af/prep_deepbeat.py path/to/test.npz [--max N]
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deepbeat_prepared.json")
FS = 32.0


def find_keys(z):
    """DeepBeat's array names are not documented; identify them by shape and content."""
    keys = list(z.keys())
    print("  arrays in file:")
    for k in keys:
        a = z[k]
        print(f"    {k:22s} shape={str(a.shape):22s} dtype={a.dtype}")

    sig_key = None
    for k in keys:
        a = z[k]
        if a.ndim >= 2 and a.shape[-1] >= 256 and np.issubdtype(a.dtype, np.floating):
            if sig_key is None or a.shape[-1] > z[sig_key].shape[-1]:
                sig_key = k
    if sig_key is None:
        for k in keys:
            if z[k].ndim >= 2 and z[k].shape[-1] >= 256:
                sig_key = k; break

    n = z[sig_key].shape[0]
    label_key, qual_key = None, None
    for k in keys:
        if k == sig_key:
            continue
        a = z[k]
        if a.shape[0] != n:
            continue
        flat = a.reshape(n, -1)[:, 0] if a.ndim > 1 else a
        u = np.unique(flat[~np.isnan(flat.astype(float))]) if flat.size else []
        if len(u) <= 4:
            name = k.lower()
            if any(t in name for t in ("qa", "qual")):
                qual_key = k
            elif label_key is None or any(t in name for t in ("label", "rhythm", "af", "y")):
                label_key = k
    return sig_key, label_key, qual_key


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    cap = None
    if "--max" in sys.argv:
        cap = int(sys.argv[sys.argv.index("--max") + 1])

    print(f"Loading {path} ...")
    z = np.load(path, allow_pickle=True)
    sig_key, label_key, qual_key = find_keys(z)
    print(f"\n  -> signal={sig_key}  label={label_key}  quality={qual_key}")

    sig = np.asarray(z[sig_key])
    if sig.ndim == 3:
        sig = sig[:, :, 0] if sig.shape[2] < sig.shape[1] else sig[:, 0, :]
    lab = np.asarray(z[label_key]).reshape(sig.shape[0], -1)[:, 0] if label_key else None
    qual = np.asarray(z[qual_key]).reshape(sig.shape[0], -1)[:, 0] if qual_key else None

    print(f"\n  segments: {sig.shape[0]}  samples each: {sig.shape[1]} "
          f"({sig.shape[1]/FS:.1f} s at {FS:.0f} Hz)")
    if lab is not None:
        vals, counts = np.unique(lab[~np.isnan(lab.astype(float))], return_counts=True)
        print(f"  label values: {dict(zip(vals.tolist(), counts.tolist()))}")
    if qual is not None:
        vals, counts = np.unique(qual[~np.isnan(qual.astype(float))], return_counts=True)
        print(f"  quality values: {dict(zip(vals.tolist(), counts.tolist()))}")

    idx = np.arange(sig.shape[0])
    if cap and len(idx) > cap:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, cap, replace=False))
        print(f"  subsampled to {cap} segments")

    recs = []
    for i in idx:
        v = np.asarray(sig[i], dtype=float)
        if not np.all(np.isfinite(v)) or np.std(v) == 0:
            continue
        recs.append({
            "id": int(i),
            "fs": FS,
            "ppg": [round(float(x), 6) for x in v],
            "af": None if lab is None else int(lab[i]),
            "quality": None if qual is None else float(qual[i]),
        })

    json.dump({"_source": "Stanford DeepBeat (Synapse syn21985690)",
               "_fs": FS, "_segment_seconds": sig.shape[1] / FS,
               "records": recs}, open(OUT, "w"))
    print(f"\nwrote {OUT}  ({len(recs)} segments, {os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
