"""Turn butppg_results.json into the validation report."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "butppg_results.json")))
rows, counts = d["rows"], d["counts"]
MOTION = {0: "rest", 1: "finger pressure", 2: "moving on lens", 3: "walking",
          4: "coughing", 5: "laughing", 6: "changing light", 7: "talking"}

def med(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")

def pct(n, d):
    return (100.0 * n / d) if d else float("nan")

print("=" * 78)
print("INDEPENDENT VALIDATION — frozen pipeline on BUT PPG (real phone camera)")
print("=" * 78)
print(f"""
Records listed in dataset            {counts['listed']}
  ear recordings skipped             {counts['ear_skipped']}   (different optical path)
  malformed or all-zero PPG          {counts['malformed_or_empty']}   (pre-112001 release defect)
  fetch errors                       {counts['fetch_error']}
  USABLE, fed to the pipeline        {counts['attempted']}
""")

ok = [r for r in rows if r["gate_ok"]]
rej = [r for r in rows if not r["gate_ok"]]
scored = [r for r in ok if r.get("rr_mae") is not None]

print("-" * 78)
print("QUALITY GATE")
print("-" * 78)
print(f"  passed                             {len(ok):4d}  ({pct(len(ok), len(rows)):.1f}%)")
print(f"  rejected                           {len(rej):4d}  ({pct(len(rej), len(rows)):.1f}%)")
print(f"  passed and produced intervals      {len(scored):4d}  ({pct(len(scored), len(rows)):.1f}%)")
if rej:
    print("\n  why recordings were rejected:")
    reasons = {}
    for r in rej:
        key = r["gate_reason"].split("(")[0].strip()
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"    {v:4d}  ({pct(v, len(rows)):4.1f}%)  {k}")

print()
print("-" * 78)
print("ACCURACY vs ECG GROUND TRUTH  (recordings that passed the gate)")
print("-" * 78)
if scored:
    mae = [r["rr_mae"] for r in scored]
    ni = [r.get("rr_mae_no_interp") for r in scored]
    hre = [r["hr_err"] for r in scored]
    print(f"  RR interval error   median {med(mae):6.1f} ms   "
          f"p90 {np.percentile([m for m in mae if np.isfinite(m)], 90):6.1f} ms")
    if any(x is not None for x in ni):
        print(f"    without sub-sample interpolation: median {med(ni):6.1f} ms")
    print(f"  Heart rate error    median {med(hre):6.1f} bpm  "
          f"p90 {np.percentile([h for h in hre if np.isfinite(h)], 90):6.1f} bpm")
    within = sum(1 for m in mae if m < 25)
    print(f"  Within the 25 ms tolerance the noise curve allows: "
          f"{within}/{len(scored)} ({pct(within, len(scored)):.0f}%)")
    hr5 = sum(1 for h in hre if h <= 5)
    print(f"  Heart rate within 5 bpm (IEC 60601-2-27): "
          f"{hr5}/{len(scored)} ({pct(hr5, len(scored)):.0f}%)")
else:
    print("  no scored recordings")

print()
print("-" * 78)
print("FALSE POSITIVES — this cohort has essentially no atrial fibrillation,")
print("so every 'irregular' call is an error.")
print("-" * 78)
print("  NOTE: recordings are 10s; the shipped classifier requires 60s. With ~12")
print("  intervals instead of ~70 the statistics are far noisier, so these rates are a")
print("  PESSIMISTIC UPPER BOUND, not the shipped system's specificity.")
print()
calls = {}
for r in rows:
    calls[r["af_call"]] = calls.get(r["af_call"], 0) + 1
for k in ("regular", "irregular", "inconclusive", "rejected"):
    if k in calls:
        print(f"    {k:14s} {calls[k]:4d}  ({pct(calls[k], len(rows)):5.1f}% of all records)")
analysed = calls.get("regular", 0) + calls.get("irregular", 0)
if analysed:
    fp = calls.get("irregular", 0)
    print(f"\n  Of the {analysed} recordings that produced a verdict: "
          f"{fp} irregular = {pct(fp, analysed):.1f}% false positive rate")
    print(f"  Implied specificity (10s, off-label): {100 - pct(fp, analysed):.1f}%")

print()
print("-" * 78)
print("BY CONDITION — where it breaks")
print("-" * 78)
hdr = f"  {'condition':18s} {'n':>4s} {'usable':>7s} {'rejected':>9s} {'irregular':>10s} {'RR err':>8s} {'HR err':>8s}"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for code in sorted(MOTION):
    sub = [r for r in rows if r.get("motion") == code]
    if not sub:
        continue
    su = [r for r in sub if r["gate_ok"]]
    ssc = [r for r in su if r.get("rr_mae") is not None]
    irr = sum(1 for r in sub if r["af_call"] == "irregular")
    ver = sum(1 for r in sub if r["af_call"] in ("regular", "irregular"))
    print(f"  {MOTION[code]:18s} {len(sub):4d} {pct(len(su), len(sub)):6.0f}% "
          f"{pct(len(sub) - len(su), len(sub)):8.0f}% "
          f"{(pct(irr, ver) if ver else float('nan')):9.0f}% "
          f"{med([r['rr_mae'] for r in ssc]):7.1f}ms "
          f"{med([r['hr_err'] for r in ssc]):7.1f}")

print()
print("-" * 78)
print("BY DATASET QUALITY LABEL — do we agree with the human annotators?")
print("-" * 78)
for lab, name in [(1, "annotated good"), (0, "annotated poor")]:
    sub = [r for r in rows if r.get("quality_label") == lab]
    if not sub:
        continue
    su = [r for r in sub if r["gate_ok"]]
    ssc = [r for r in su if r.get("rr_mae") is not None]
    irr = sum(1 for r in sub if r["af_call"] == "irregular")
    ver = sum(1 for r in sub if r["af_call"] in ("regular", "irregular"))
    print(f"  {name:16s} n={len(sub):4d}  our gate passed {pct(len(su), len(sub)):5.1f}%  "
          f"RR err {med([r['rr_mae'] for r in ssc]):6.1f} ms  "
          f"irregular {(pct(irr, ver) if ver else float('nan')):5.1f}%")
print()
