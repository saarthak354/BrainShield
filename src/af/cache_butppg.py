"""
Download BUT PPG finger recordings once and cache the raw RGB traces + ECG beat times.

Caching matters: the first validation attempt had TWO harness bugs (element-wise RR
comparison without beat matching, and a hardcoded red channel where the shipped
pipeline selects the best one). Re-downloading 400 records to test each fix is 45
minutes per iteration. With a cache the analysis is instant and can be re-run against
the real JS pipeline.
"""
import os, re, json, sys, urllib.request
import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "butppg_cache.json")
BASE = "https://physionet.org/files/butppg/2.0.0"
N = int(os.environ.get("BUTPPG_N", "300"))
FS_ECG = 1000.0


def fetch(u, t=45):
    return urllib.request.urlopen(u, timeout=t).read()


def parse_header(txt):
    lines = [l for l in txt.splitlines() if l.strip()]
    p = lines[0].split()
    nsig, fs, nsamp = int(p[1]), float(p[2]), int(p[3])
    gains, bases, names = [], [], []
    for l in lines[1:1 + nsig]:
        m = re.search(r"\s(-?[\d.]+)\((-?\d+)\)", l)
        gains.append(float(m.group(1)) if m else 1.0)
        bases.append(float(m.group(2)) if m else 0.0)
        names.append(l.split()[-1])
    return nsig, fs, nsamp, gains, bases, names


def main():
    meta = {}
    for name in ("quality-hr-ann.csv", "subject-info.csv"):
        txt = fetch(f"{BASE}/{name}", 120).decode("utf8", "ignore")
        lines = [l for l in txt.splitlines() if l.strip()]
        hdr = [h.strip().lower().lstrip("﻿") for h in lines[0].split(",")]
        for l in lines[1:]:
            c = [x.strip() for x in l.split(",")]
            if not c or not c[0]:
                continue
            m = meta.setdefault(c[0], {})
            if name.startswith("quality"):
                try: m["quality"] = int(float(c[1]))
                except Exception: pass
            else:
                for key, col in (("spot", "ear/finger"), ("motion", "motion"),
                                 ("age", "age [years]"), ("gender", "gender")):
                    if col in hdr:
                        k = hdr.index(col)
                        if k < len(c) and c[k]:
                            try: m[key] = int(float(c[k]))
                            except Exception: m[key] = c[k]
    print(f"metadata for {len(meta)} records")

    idx = fetch(f"{BASE}/", 120).decode("utf8", "ignore")
    recs = sorted(set(re.findall(r'href="(\d{6})/"', idx)))
    print(f"{len(recs)} records listed")

    out, skipped = [], {"ear": 0, "malformed": 0, "error": 0}
    for rid in recs:
        if len(out) >= N:
            break
        if meta.get(rid, {}).get("spot") == 0:
            skipped["ear"] += 1
            continue
        try:
            nsig, fs, nsamp, g, b, names = parse_header(
                fetch(f"{BASE}/{rid}/{rid}_PPG.hea", 30).decode("utf8", "ignore"))
        except Exception:
            skipped["error"] += 1
            continue
        if nsig != 3 or nsamp < 150:
            skipped["malformed"] += 1
            continue
        try:
            raw = fetch(f"{BASE}/{rid}/{rid}_PPG.dat", 30)
            ann = wfdb.rdann(rid, "qrs", pn_dir=f"butppg/{rid}")
        except Exception:
            skipped["error"] += 1
            continue
        d = np.frombuffer(raw, dtype="<i2").astype(float)
        if len(d) < nsig * nsamp or not np.count_nonzero(d):
            skipped["malformed"] += 1
            continue
        d = d[:nsig * nsamp].reshape(nsamp, nsig)
        ch = {names[i]: ((d[:, i] - b[i]) / g[i]).tolist() for i in range(nsig)}
        beats_s = (np.asarray(ann.sample, float) / FS_ECG).tolist()
        if len(beats_s) < 4:
            skipped["malformed"] += 1
            continue
        rec = {"id": rid, "fs": fs, "n": nsamp,
               "r": ch.get("PPG_R"), "g": ch.get("PPG_G"), "b": ch.get("PPG_B"),
               "ecg_beats_s": beats_s}
        rec.update({k: meta.get(rid, {}).get(k) for k in ("quality", "motion", "age", "gender")})
        out.append(rec)
        if len(out) % 25 == 0:
            print(f"  cached {len(out)}...")

    json.dump({"skipped": skipped, "records": out}, open(OUT, "w"))
    print(f"\nwrote {OUT}  ({len(out)} records, {os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
