"""
Freeze the AF algorithm before independent validation.

Records a SHA-256 of every file that can change a prediction. Re-running this after
the BUT PPG results are in will prove whether the algorithm that produced those
results is the same one described in the write-up. Without this, "we validated on an
independent dataset" is an unverifiable claim.
"""
import hashlib, json, os, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
OUT = os.path.join(HERE, "FROZEN.json")

# Anything that can alter an AF prediction. Not the UI, not the tests.
FILES = [
    "src/af/build_afdb.py",      # feature definitions
    "src/af/train_af.py",        # fitting + threshold selection
    "src/af/af_model_export.json",  # the fitted coefficients and threshold
    "src/af/af_core.js",         # features, classifier, decision rule
    "src/af/af_capture.js",      # capture, quality gating, beat detection
    "src/rppg_core.js",          # filtering, resampling, peak finding
]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest():
    return {p: sha(os.path.join(ROOT, p)) for p in FILES}

def main():
    verify = "--verify" in sys.argv
    cur = manifest()

    if verify:
        if not os.path.exists(OUT):
            print("No freeze recorded. Run without --verify first."); sys.exit(2)
        old = json.load(open(OUT))
        drift = [p for p in FILES if old["files"].get(p) != cur[p]]
        print(f"Frozen at: {old['frozen_at']}  ({old['label']})")
        if drift:
            print("\nALGORITHM HAS CHANGED since the freeze:")
            for p in drift:
                print(f"  CHANGED  {p}")
            print("\nAny independent-validation result computed before this change no longer\n"
                  "describes the current algorithm. Either revert, or re-freeze and re-run\n"
                  "the validation from scratch.")
            sys.exit(1)
        print(f"\nAll {len(FILES)} files unchanged. Validation results remain valid.")
        sys.exit(0)

    rec = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": sys.argv[1] if len(sys.argv) > 1 else "pre-BUTPPG-validation",
        "why": "Algorithm frozen before independent validation. Changing any listed file "
               "after this point invalidates results computed against it.",
        "development_results": {
            "dataset": "MIT-BIH AFDB, 29,018 windows, 25 records",
            "validation": "leave-one-record-out",
            "note": "This is DEVELOPMENT performance. The algorithm was iterated against "
                    "this data, so it is not an independent estimate.",
        },
        "files": cur,
    }
    json.dump(rec, open(OUT, "w"), indent=2)
    print(f"Frozen {len(FILES)} files -> {OUT}")
    for p in FILES:
        print(f"  {cur[p][:12]}  {p}")

if __name__ == "__main__":
    main()
