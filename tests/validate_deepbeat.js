/*
 * Sensitivity of the FROZEN AF classifier on real photoplethysmographic intervals.
 *
 * Every sensitivity figure we have so far comes from MIT-BIH, i.e. from intervals
 * derived from electrocardiography. DeepBeat supplies AF-labelled PPG, so this is the
 * first measurement of whether the classifier still separates rhythms once the
 * intervals come from a pulse wave -- with its softer peaks, respiratory amplitude
 * modulation and genuine detection failures -- rather than from a QRS complex.
 *
 * The acquisition path is NOT under test here: these are wrist recordings, not our
 * camera. This isolates the classifier.
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");

global.window = global;
const RPPG = require(path.join(ROOT, "src/rppg_core.js"));
const AFC  = require(path.join(ROOT, "src/af/af_core.js"));
global.RPPG = RPPG; global.BrainShieldAF = AFC;
const AF_MODEL = JSON.parse(fs.readFileSync(path.join(ROOT, "src/af/af_model_export.json"), "utf8"));

const DATA = path.join(ROOT, "src/af/deepbeat_prepared.json");
if (!fs.existsSync(DATA)) {
    console.error("Missing " + DATA + "\nRun: python3 src/af/prep_deepbeat.py <test.npz>");
    process.exit(2);
}
const D = JSON.parse(fs.readFileSync(DATA, "utf8"));
const recs = D.records;
console.log("=".repeat(78));
console.log("FROZEN CLASSIFIER on DeepBeat — sensitivity on PPG-derived intervals");
console.log("=".repeat(78));
console.log(`\n  ${recs.length} segments of ${D._segment_seconds.toFixed(1)} s at ${D._fs} Hz`);
console.log(`  NOTE: trained on 60 s windows; these are ${D._segment_seconds.toFixed(0)} s, so interval`);
console.log(`  statistics rest on ~${Math.round(D._segment_seconds*70/60)} intervals instead of ~70. Entropy terms are`);
console.log(`  noisier here than in deployment.\n`);

const out = [];
for (const r of recs) {
    const n = r.ppg.length, fsHz = r.fs;
    const times = []; for (let i = 0; i < n; i++) times.push(i / fsHz);
    const samples = {times, r: r.ppg, g: r.ppg, b: r.ppg};
    const row = {id: r.id, subject: r.subject, af: r.af, quality: r.quality, call: null, p: null};

    const res = RPPG.analyse(samples, {mode: "finger", fs: Math.round(fsHz)});
    if (!res.ok) { row.call = "unusable"; row.why = res.reason; out.push(row); continue; }
    const peaks = AFC.refinePeaks(res.waveform, RPPG.findPeaks(res.waveform, res.fs, (res.hr||70)/60));
    const rr = AFC.rrFromPeaks(peaks, res.fs);
    row.n_rr = rr.length;
    row.hr = res.hr;
    if (rr.length < 4) { row.call = "unusable"; row.why = "too few intervals"; out.push(row); continue; }
    const feats = AFC.afFeatures(rr);
    const p = AFC.classify(feats, AF_MODEL);
    if (p === null) { row.call = "unusable"; row.why = "non-finite feature"; out.push(row); continue; }
    row.p = p;
    row.rmssd_norm = feats.rmssd_norm;
    row.call = p >= AF_MODEL.threshold_rule_out ? "irregular" : "regular";
    out.push(row);
}

fs.writeFileSync(path.join(ROOT, "src/af/deepbeat_results.json"), JSON.stringify({rows: out}, null, 1));

/* ---------------- report ---------------- */
const scored = out.filter(r => r.p !== null && r.af !== null);
const unusable = out.filter(r => r.call === "unusable");
console.log("-".repeat(78));
console.log("COVERAGE");
console.log("-".repeat(78));
console.log(`  scored              ${scored.length}  (${(100*scored.length/out.length).toFixed(1)}%)`);
console.log(`  unusable            ${unusable.length}  (${(100*unusable.length/out.length).toFixed(1)}%)`);
if (unusable.length) {
    const why = {};
    unusable.forEach(r => { const k=(r.why||"?").slice(0,45); why[k]=(why[k]||0)+1; });
    Object.entries(why).sort((a,b)=>b[1]-a[1]).slice(0,5)
        .forEach(([k,v]) => console.log(`    ${String(v).padStart(5)}  ${k}`));
}

const AFLBL = 1;
const tp = scored.filter(r => r.af === AFLBL && r.call === "irregular").length;
const fn = scored.filter(r => r.af === AFLBL && r.call === "regular").length;
const fp = scored.filter(r => r.af !== AFLBL && r.call === "irregular").length;
const tn = scored.filter(r => r.af !== AFLBL && r.call === "regular").length;

console.log("\n" + "-".repeat(78));
console.log("CLASSIFICATION  (label 1 = atrial fibrillation)");
console.log("-".repeat(78));
console.log(`                    called irregular   called regular`);
console.log(`  AF present        ${String(tp).padStart(12)}     ${String(fn).padStart(12)}`);
console.log(`  AF absent         ${String(fp).padStart(12)}     ${String(tn).padStart(12)}`);
const sens = tp+fn ? tp/(tp+fn) : NaN, spec = tn+fp ? tn/(tn+fp) : NaN;
console.log(`\n  SENSITIVITY (on PPG intervals): ${(100*sens).toFixed(1)}%   [${tp}/${tp+fn}]`);
console.log(`  SPECIFICITY                    : ${(100*spec).toFixed(1)}%   [${tn}/${tn+fp}]`);
console.log(`\n  For comparison, on ECG intervals (MIT-BIH, leave-one-patient-out):`);
console.log(`    sensitivity ${(100*AF_MODEL.performance.sensitivity_at_threshold).toFixed(0)}%   specificity ${(100*AF_MODEL.performance.specificity_at_threshold).toFixed(1)}%`);

/* ROC, independent of our chosen threshold */
const ps = scored.map(r => ({p: r.p, y: r.af === AFLBL ? 1 : 0}));
ps.sort((a,b) => b.p - a.p);
const P = ps.filter(x=>x.y===1).length, N = ps.length - P;
let tpc=0, fpc=0, auc=0, prev=0;
for (const x of ps) {
    if (x.y===1) tpc++; else { fpc++; auc += tpc; }
}
const aucv = (P&&N) ? auc/(P*N) : NaN;
console.log(`\n  AUROC on this corpus: ${aucv.toFixed(4)}   (threshold-independent)`);
console.log(`  MIT-BIH AUROC       : ${AF_MODEL.performance.auroc_record_held_out.toFixed(4)}`);

/* stratify by DeepBeat's own quality label if present */
const qs = [...new Set(scored.map(r => r.quality).filter(q => q !== null))].sort();
if (qs.length > 1) {
    console.log("\n" + "-".repeat(78));
    console.log("BY DEEPBEAT'S OWN SIGNAL-QUALITY LABEL");
    console.log("-".repeat(78));
    const QNAME = {0: "poor", 1: "acceptable", 2: "excellent"};
    for (const q of qs) {
        const s = scored.filter(r => r.quality === q);
        const a = s.filter(r => r.af === AFLBL);
        const b = s.filter(r => r.af !== AFLBL);
        const se = a.length ? a.filter(r=>r.call==="irregular").length/a.length : NaN;
        const sp = b.length ? b.filter(r=>r.call==="regular").length/b.length : NaN;
        console.log(`  ${(QNAME[q]||q).padEnd(10)} n=${String(s.length).padStart(5)}   sensitivity ${(100*se).toFixed(1)}%  [${a.filter(r=>r.call==="irregular").length}/${a.length}]   specificity ${(100*sp).toFixed(1)}%  [${b.filter(r=>r.call==="regular").length}/${b.length}]`);
    }
}

/* ----------------------------------------------------------------------------
 * Per-subject aggregation.
 *
 * The rhythm label is constant within a subject, so the 17617 segments are not
 * 17617 independent observations -- they are 22. Segment-level intervals would be
 * roughly sqrt(800) times too narrow. A subject is called AF if the majority of
 * its scored segments are called irregular.
 * -------------------------------------------------------------------------- */
function subjectTable(rows, caption) {
    const m = new Map();
    for (const r of rows) { if (!m.has(r.subject)) m.set(r.subject, []); m.get(r.subject).push(r); }
    let tp=0, fn=0, fp=0, tn=0;
    for (const k of [...m.keys()].sort((a,b)=>Number(a)-Number(b))) {
        const rs = m.get(k);
        const truth = rs[0].af === AFLBL;
        const call = rs.filter(r => r.call === "irregular").length / rs.length > 0.5;
        if (truth && call) tp++; else if (truth && !call) fn++;
        else if (!truth && call) fp++; else tn++;
    }
    const se = (tp+fn) ? 100*tp/(tp+fn) : NaN, sp = (tn+fp) ? 100*tn/(tn+fp) : NaN;
    console.log(`  ${caption}`);
    console.log(`    subjects ${m.size}   sensitivity ${se.toFixed(1)}%  [${tp}/${tp+fn}]   specificity ${sp.toFixed(1)}%  [${tn}/${tn+fp}]`);
}

const bySubj = new Map();
for (const r of scored) {
    if (!bySubj.has(r.subject)) bySubj.set(r.subject, []);
    bySubj.get(r.subject).push(r);
}
console.log("-".repeat(78));
console.log("PER SUBJECT  (label is constant within subject; effective n = " + bySubj.size + ")");
console.log("-".repeat(78));
console.log("  subject   truth    scored   frac called irregular   subject call");
let sTP=0, sFN=0, sFP=0, sTN=0;
const keys = [...bySubj.keys()].sort((a,b) => Number(a) - Number(b));
for (const k of keys) {
    const rs = bySubj.get(k);
    const truth = rs[0].af === AFLBL;
    const frac = rs.filter(r => r.call === "irregular").length / rs.length;
    const call = frac > 0.5;
    if (truth && call) sTP++; else if (truth && !call) sFN++;
    else if (!truth && call) sFP++; else sTN++;
    console.log(`  ${String(k).padStart(7)}   ${(truth?"AF":"non-AF").padEnd(7)} ${String(rs.length).padStart(6)}   ${frac.toFixed(3).padStart(19)}   ${call?"irregular":"regular"}`);
}
const sSens = (sTP+sFN) ? sTP/(sTP+sFN) : NaN, sSpec = (sTN+sFP) ? sTN/(sTN+sFP) : NaN;
console.log(`\n  SUBJECT-LEVEL sensitivity ${(100*sSens).toFixed(1)}%  [${sTP}/${sTP+sFN}]   specificity ${(100*sSpec).toFixed(1)}%  [${sTN}/${sTN+sFP}]`);
console.log(`  With 6 AF and 16 non-AF subjects these rest on very few units; report them`);
console.log(`  with the counts, never as a bare percentage.\n`);
subjectTable(scored.filter(r => r.quality === 2),
             "Restricted to EXCELLENT-quality segments (the stratum deployment targets):");
subjectTable(scored.filter(r => r.quality !== 0),
             "Restricted to excellent + acceptable:");
console.log();
