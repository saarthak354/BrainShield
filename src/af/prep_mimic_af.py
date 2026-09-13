"""
Convert the MIMIC PERform AF corpus into the JSON the JS validator consumes.

MIMIC PERform AF (Charlton et al., derived from the MIMIC-III Waveform Database
Matched Subset) holds 35 critically-ill adults -- 19 in atrial fibrillation, 16 not --
each with a 20-minute simultaneous recording at 125 Hz. The header declares the
photoplethysmogram as "Fingertip PPG recorded using bedside monitor".

Why this corpus and not another: it is the only openly accessible PPG corpus with
rhythm labels that is both FINGERTIP (transmission through the digit, the same optical
geometry as a torch-and-camera capture) and LONG ENOUGH to cut the deployed 60-second
window. DeepBeat is wrist reflectance in 25-second segments; this is neither.

What it does not test: these are contact sensors on perfused ICU patients, not a camera
held by a person. It isolates the classifier at the deployed window length on the
deployed anatomical site. The acquisition path is out of scope here.

Two sampling rates are produced so the frame-rate question is answered rather than
assumed. Native 125 Hz shows the classifier given clean timing; 30 Hz matches the camera
frame rate the system actually deploys at, where sub-sample peak interpolation has to
recover beat timing from coarse samples.

Usage:  python3 src/af/prep_mimic_af.py <af_dir> <non_af_dir> [--fs 30|125]
"""
import glob, json, os, sys
import numpy as np
import wfdb
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE_FS = 125.0
WINDOW_SEC = 60.0


def ppg_channel(rec, name):
    """Locate the PPG signal by name, refusing to guess if it is absent."""
    names = [s.upper() for s in rec.sig_name]
    for i, n in enumerate(names):
        if "PPG" in n or "PLETH" in n:
            return i
    raise SystemExit(f"{name}: no PPG channel among {rec.sig_name}")


def load(path):
    """Read one record, with the sanity checks wfdb will not do for us.

    wfdb.rdrecord returns silent garbage rather than raising on a malformed header --
    this cost us a day on BUT PPG -- so length, finiteness and variance are asserted
    here rather than discovered downstream.
    """
    name = os.path.basename(path)
    rec = wfdb.rdrecord(path)
    if abs(rec.fs - NATIVE_FS) > 1e-6:
        raise SystemExit(f"{name}: fs is {rec.fs}, expected {NATIVE_FS}")
    x = np.asarray(rec.p_signal[:, ppg_channel(rec, name)], dtype=float)
    if x.size < NATIVE_FS * WINDOW_SEC:
        raise SystemExit(f"{name}: only {x.size} samples, shorter than one window")
    return name, x


def windows(x, fs_out):
    """Non-overlapping 60 s windows, resampled to fs_out, flat ones dropped."""
    n = int(NATIVE_FS * WINDOW_SEC)
    out, flat = [], 0
    for s in range(0, len(x) - n + 1, n):
        w = x[s:s + n]
        if not np.all(np.isfinite(w)):
            flat += 1
            continue
        # A disconnected pulse oximeter reads a constant. Such a window carries no
        # rhythm information at all and is not a fair test of anything.
        if np.std(w) < 1e-9 or len(np.unique(w)) < 10:
            flat += 1
            continue
        if abs(fs_out - NATIVE_FS) > 1e-6:
            up, down = int(round(fs_out)), int(round(NATIVE_FS))
            g = np.gcd(up, down)
            w = resample_poly(w, up // g, down // g)
        out.append(w)
    return out, flat


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__); sys.exit(1)
    fs_out = NATIVE_FS
    if "--fs" in sys.argv:
        fs_out = float(sys.argv[sys.argv.index("--fs") + 1])

    recs, flat_total = [], 0
    for d, af in ((args[0], 1), (args[1], 0)):
        paths = sorted(glob.glob(os.path.join(d, "*.hea")))
        if not paths:
            raise SystemExit(f"no .hea files in {d}")
        print(f"{'AF' if af else 'non-AF'}: {len(paths)} records in {d}")
        for p in paths:
            name, x = load(p[:-4])
            ws, flat = windows(x, fs_out)
            flat_total += flat
            for k, w in enumerate(ws):
                recs.append({"id": f"{name}_w{k}", "subject": name, "fs": fs_out,
                             "ppg": [round(float(v), 5) for v in w], "af": af,
                             "quality": None})

    n_af = sum(1 for r in recs if r["af"] == 1)
    print(f"\n  {len(recs)} windows of {WINDOW_SEC:.0f} s at {fs_out:g} Hz "
          f"({n_af} AF, {len(recs)-n_af} non-AF)")
    print(f"  subjects: {len({r['subject'] for r in recs})}")
    if flat_total:
        print(f"  dropped {flat_total} flat or non-finite windows (sensor disconnected)")

    out = os.path.join(HERE, f"mimic_af_prepared_{int(fs_out)}hz.json")
    json.dump({"_source": "MIMIC PERform AF (Charlton et al.), fingertip PPG",
               "_fs": fs_out, "_segment_seconds": WINDOW_SEC,
               "records": recs}, open(out, "w"))
    print(f"\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
