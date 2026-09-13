/*
 * Validation of the FROZEN pipeline on the Gdansk University of Technology
 * smartphone-PPG corpus (Bancerewicz et al., MIT licence).
 *
 * Complements the BUT PPG evaluation in one specific way: those recordings are 10 s,
 * whereas the deployed classifier operates on 60 s windows. These recordings are 64 s
 * and ~10 min, so the pipeline can be exercised at its actual deployed configuration.
 *
 *   Part A  aligned PPG + ECG (peak-annotated) -> beat detection and RR accuracy at 60 s
 *   Part B  three ~10 min PPG recordings, no ECG -> gate pass rate and, since these are
 *           healthy volunteers, the false-positive rate at the deployed window length
 *
 * The PPG here is a single already-extracted green-channel mean, not RGB. It is supplied
 * on all three channels so the pipeline's channel selector is a no-op rather than being
 * bypassed; this is the one place the input differs from a live capture.
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");
const DATA = path.join(ROOT, "src/af/gdansk");

global.window = global;
const RPPG = require(path.join(ROOT, "src/rppg_core.js"));
const AFC  = require(path.join(ROOT, "src/af/af_core.js"));
global.RPPG = RPPG; global.BrainShieldAF = AFC;
const CAP  = require(path.join(ROOT, "src/af/af_capture.js"));
const AF_MODEL = JSON.parse(fs.readFileSync(path.join(ROOT, "src/af/af_model_export.json"), "utf8"));

function readCsv(p) {
    const lines = fs.readFileSync(p, "utf8").trim().split("\n");
    const hdr = lines[0].split(",").map(s => s.trim());
    return lines.slice(1).map(l => {
        const c = l.split(",");
        const o = {};
        hdr.forEach((h, i) => o[h] = parseFloat(c[i]));
        return o;
    }).filter(r => Object.values(r).every(v => !Number.isNaN(v)));
}

const med = a => { const v = a.filter(x => x != null && isFinite(x)).sort((x, y) => x - y);
                   return v.length ? v[Math.floor(v.length / 2)] : NaN; };
const pctl = (a, q) => { const v = a.filter(x => x != null && isFinite(x)).sort((x, y) => x - y);
                         return v.length ? v[Math.floor(v.length * q)] : NaN; };

/* --- beat matching, identical to tests/validate_butppg.js --- */
const MATCH_TOL = 0.10;
function bestLag(ppgT, ecgT) {
    const rr = []; for (let i = 1; i < ecgT.length; i++) rr.push(ecgT[i] - ecgT[i-1]);
    rr.sort((a,b)=>a-b);
    const medRR = rr.length ? rr[Math.floor(rr.length/2)] : 0.8;
    const span = Math.max(0.8, 1.2 * medRR);
    let best = {lag:0, n:-1, cost:Infinity};
    for (let lag = -span; lag <= span; lag += 0.005) {
        let n = 0, cost = 0;
        for (const p of ppgT) {
            const t = p - lag; let bd = Infinity;
            for (const e of ecgT) { const d = Math.abs(e - t); if (d < bd) bd = d; }
            if (bd <= MATCH_TOL) { n++; cost += bd; }
        }
        if (n > best.n || (n === best.n && cost < best.cost)) best = {lag, n, cost};
    }
    return best;
}
function matchBeats(ppgT, ecgT, lag) {
    const used = new Set(), pairs = [];
    ppgT.forEach((p, pi) => {
        const t = p - lag; let bi = -1, bd = Infinity;
        ecgT.forEach((e, ei) => { if (used.has(ei)) return;
            const d = Math.abs(e - t); if (d < bd) { bd = d; bi = ei; } });
        if (bi >= 0 && bd <= MATCH_TOL) { used.add(bi); pairs.push({pi, ei: bi}); }
    });
    pairs.sort((a,b)=>a.pi-b.pi);
    return pairs;
}

/* Build the sample object the pipeline expects from a single-channel series. */
function toSamples(rows) {
    const t0 = rows[0].time;
    const times = rows.map(r => (r.time - t0) / 1000);
    const v = rows.map(r => r.ppg);
    return {times, r: v, g: v, b: v};
}
function estFs(times) {
    const d = []; for (let i = 1; i < times.length; i++) d.push(times[i] - times[i-1]);
    d.sort((a,b)=>a-b);
    return 1 / d[Math.floor(d.length/2)];
}

console.log("=".repeat(78));
console.log("FROZEN PIPELINE on the Gdansk smartphone-PPG corpus");
console.log("=".repeat(78));

/* ================= PART A: accuracy vs ECG at 60 s ================= */
console.log("\n" + "-".repeat(78));
console.log("PART A — accuracy against ECG, at the deployed 60-second window length");
console.log("-".repeat(78));

const ppgRows = readCsv(path.join(DATA, "ppg_data_aligned.csv"));
const ecgRows = readCsv(path.join(DATA, "ecg_data_aligned.csv"));
const t0 = ppgRows[0].time;
const samples = toSamples(ppgRows);
const fsHz = estFs(samples.times);
const dur = samples.times[samples.times.length - 1];
/* The committed `peak` columns are entirely zero, so ground truth is derived from the
   ECG waveform here rather than taken from author annotations. A Polar chest-strap ECG
   is clean enough for a simple detector: band-pass to the QRS band, square, integrate
   over a short window, then threshold with a refractory period. Beat count is
   sanity-checked against a plausible heart rate below. */
function detectRPeaks(rows, t0) {
    const t = rows.map(r => (r.time - t0) / 1000);
    const x = rows.map(r => r.ecg);
    const n = x.length;
    const fsE = (n - 1) / (t[n-1] - t[0]);

    // Pan-Tompkins style, time domain: 5-point derivative emphasises the QRS slope,
    // square rectifies and amplifies it, then integrate over ~120 ms (roughly one QRS
    // width) to produce a single lobe per beat.
    const d = new Float64Array(n);
    for (let i = 2; i < n - 2; i++) {
        d[i] = (2*x[i+2] + x[i+1] - x[i-1] - 2*x[i-2]) / 8;
    }
    const w = Math.max(3, Math.round(0.12 * fsE));
    const e = new Float64Array(n);
    let acc = 0;
    for (let i = 0; i < n; i++) {
        acc += d[i]*d[i];
        if (i >= w) acc -= d[i-w]*d[i-w];
        e[i] = acc / w;
    }
    const sorted = Array.from(e).sort((a,b)=>a-b);
    const thr = sorted[Math.floor(sorted.length*0.97)] * 0.30;
    const refractory = Math.round(0.25 * fsE);   // 250 ms: no physiological RR is shorter
    const peaks = [];
    for (let i = 1; i < n-1; i++) {
        if (e[i] > thr && e[i] >= e[i-1] && e[i] > e[i+1]) {
            if (peaks.length && i - peaks[peaks.length-1] < refractory) {
                if (e[i] > e[peaks[peaks.length-1]]) peaks[peaks.length-1] = i;
            } else peaks.push(i);
        }
    }
    // the integrator lags the true R-wave by about half its window; correct for it
    const lagCorr = Math.round(w / 2);
    return peaks.map(i => t[Math.max(0, i - lagCorr)]);
}
const ecgBeats = detectRPeaks(ecgRows, t0);
const ppgAnnot = ppgRows.filter(r => r.peak === 1).length;

console.log(`  recording: ${dur.toFixed(1)} s, PPG ${samples.times.length} samples @ ${fsHz.toFixed(1)} Hz`);
console.log(`  ECG R-peaks annotated: ${ecgBeats.length}  (mean HR ${(60*ecgBeats.length/dur).toFixed(1)} bpm)`);
console.log(`  PPG peaks annotated by the authors: ${ppgAnnot}  (committed columns are empty)`);
console.log(`  ECG ground truth: DERIVED from the waveform, not author-annotated`);

const res = RPPG.analyse(samples, {mode: "finger", fs: Math.round(fsHz)});
if (!res.ok) {
    console.log(`  PIPELINE REJECTED: ${res.reason}`);
} else {
    const q = CAP.afQuality(res, samples);
    const peaks = AFC.refinePeaks(res.waveform, RPPG.findPeaks(res.waveform, res.fs, (res.hr||70)/60));
    const ppgT = peaks.map(p => samples.times[0] + p / res.fs);
    const {lag} = bestLag(ppgT, ecgBeats);
    const pairs = matchBeats(ppgT, ecgBeats, lag);

    const errs = [];
    for (let i = 1; i < pairs.length; i++) {
        const a = pairs[i-1], b = pairs[i];
        if (b.ei !== a.ei + 1) continue;
        errs.push((ppgT[b.pi]-ppgT[a.pi])*1000 - (ecgBeats[b.ei]-ecgBeats[a.ei])*1000);
    }
    const mae = errs.reduce((s,e)=>s+Math.abs(e),0)/errs.length;
    const bias = errs.reduce((s,e)=>s+e,0)/errs.length;
    const hrEcg = 60*ecgBeats.length/dur;

    console.log(`\n  quality gate: ${q.ok ? "PASSED" : "REJECTED — " + q.reason}`);
    console.log(`  our peaks detected: ${ppgT.length}   matched to ECG: ${pairs.length}`);
    console.log(`  beat sensitivity : ${(100*pairs.length/ecgBeats.length).toFixed(1)}%`);
    console.log(`  beat PPV         : ${(100*pairs.length/ppgT.length).toFixed(1)}%`);
    console.log(`  pulse transit lag: ${(lag*1000).toFixed(0)} ms`);
    console.log(`  RR error         : ${mae.toFixed(1)} ms  (bias ${bias>=0?"+":""}${bias.toFixed(1)} ms, n=${errs.length} intervals)`);
    console.log(`  heart rate       : ours ${res.hr.toFixed(1)} bpm vs ECG ${hrEcg.toFixed(1)} bpm  (err ${Math.abs(res.hr-hrEcg).toFixed(1)} bpm)`);

    if (q.ok) {
        const rr = AFC.rrFromPeaks(peaks, res.fs);
        const feats = AFC.afFeatures(rr);
        const p = AFC.classify(feats, AF_MODEL);
        const call = p >= AF_MODEL.threshold_rule_out ? "irregular" : "regular";
        console.log(`  rhythm verdict   : ${call}  (p=${p.toFixed(3)}, threshold ${AF_MODEL.threshold_rule_out.toFixed(3)})`);
    }
}

/* ================= PART B: long recordings, 60 s windows ================= */
console.log("\n" + "-".repeat(78));
console.log("PART B — three ~10 min recordings, split into deployed 60-second windows");
console.log("-".repeat(78));
console.log("  No ECG reference. These are healthy volunteers, so an 'irregular' verdict");
console.log("  is a false positive. This is the specificity test at deployed window length.");

const WIN = 60, HOP = 30;
let allW = 0, allGate = 0, allReg = 0, allIrr = 0, allInc = 0, skippedGap = 0;
const hrs = [], reasons = {};

for (const f of ["ppg_data.csv", "ppg_data_1.csv", "ppg_data_johnny_10min.csv"]) {
    const rows = readCsv(path.join(DATA, f));
    const s = toSamples(rows);
    const fz = estFs(s.times);
    const total = s.times[s.times.length - 1];
    let w = 0, gate = 0, reg = 0, irr = 0, inc = 0;
    const gapBefore = skippedGap;

    for (let start = 0; start + WIN <= total; start += HOP) {
        const idx = [];
        for (let i = 0; i < s.times.length; i++) if (s.times[i] >= start && s.times[i] < start + WIN) idx.push(i);
        if (idx.length < fz * WIN * 0.6) continue;
        /* Reject windows straddling a recording gap. ppg_data.csv is several sessions
           concatenated, with 818 s of gaps and a single gap of 332 s; a window spanning
           one contains two disjoint recordings and is irregular by construction, not by
           physiology. A live 60 s capture is continuous by definition, so excluding
           these measures the deployed condition rather than an artefact of the corpus. */
        let maxGap = 0;
        for (let k = 1; k < idx.length; k++) {
            const g = s.times[idx[k]] - s.times[idx[k-1]];
            if (g > maxGap) maxGap = g;
        }
        if (maxGap > 0.5) { skippedGap++; continue; }
        const sub = {times: idx.map(i => s.times[i] - start), r: idx.map(i => s.r[i]),
                     g: idx.map(i => s.g[i]), b: idx.map(i => s.b[i])};
        w++;
        const r2 = RPPG.analyse(sub, {mode: "finger", fs: Math.round(fz)});
        if (!r2.ok) { inc++; reasons[r2.reason.slice(0,40)] = (reasons[r2.reason.slice(0,40)]||0)+1; continue; }
        const q2 = CAP.afQuality(r2, sub);
        if (!q2.ok) { inc++; const k = q2.reason.split("(")[0].trim().slice(0,45);
                      reasons[k] = (reasons[k]||0)+1; continue; }
        gate++;
        hrs.push(r2.hr);
        const pk = AFC.refinePeaks(r2.waveform, RPPG.findPeaks(r2.waveform, r2.fs, (r2.hr||70)/60));
        const rr = AFC.rrFromPeaks(pk, r2.fs);
        const ft = rr.length >= 4 ? AFC.afFeatures(rr) : null;
        const p = ft ? AFC.classify(ft, AF_MODEL) : null;
        if (p === null) { inc++; continue; }
        if (p >= AF_MODEL.threshold_rule_out) irr++; else reg++;
    }
    console.log(`\n  ${f}  (${(total/60).toFixed(1)} min @ ${fz.toFixed(1)} Hz)`);
    console.log(`    windows ${w}   gate passed ${gate} (${w?(100*gate/w).toFixed(0):0}%)   regular ${reg}   irregular ${irr}   unusable ${inc}` +
                (skippedGap > gapBefore ? `   [${skippedGap-gapBefore} windows skipped: straddled a recording gap]` : ""));
    allW += w; allGate += gate; allReg += reg; allIrr += irr; allInc += inc;
}

console.log("\n  " + "-".repeat(74));
console.log(`  TOTAL: ${allW} continuous windows of 60 s  (${skippedGap} skipped for gaps)`);
console.log(`    passed quality gate : ${allGate}  (${(100*allGate/allW).toFixed(1)}%)`);
console.log(`    verdict regular     : ${allReg}`);
console.log(`    verdict irregular   : ${allIrr}`);
console.log(`    unusable            : ${allInc}`);
if (allReg + allIrr) {
    const fp = 100*allIrr/(allReg+allIrr);
    console.log(`\n    FALSE POSITIVE RATE : ${fp.toFixed(1)}%   ->  specificity ${(100-fp).toFixed(1)}%`);
    console.log(`    (projected from BUT PPG error injection was 92.4%)`);
}
console.log(`\n    heart rate across accepted windows: median ${med(hrs).toFixed(1)} bpm ` +
            `(p10 ${pctl(hrs,0.1).toFixed(0)}, p90 ${pctl(hrs,0.9).toFixed(0)})`);
if (Object.keys(reasons).length) {
    console.log("\n  why windows were not usable:");
    Object.entries(reasons).sort((a,b)=>b[1]-a[1]).forEach(([k,v]) => console.log(`    ${String(v).padStart(4)}  ${k}`));
}
console.log();
