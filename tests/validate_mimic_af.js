/*
 * The FROZEN AF classifier on MIMIC PERform AF -- fingertip PPG, 60-second windows.
 *
 * This is the first evaluation at BOTH the deployed window length and the deployed
 * anatomical site. MIT-BIH gave sensitivity on electrocardiographic intervals; DeepBeat
 * gave it on 25-second wrist reflectance. Here the intervals come from transmission
 * through a fingertip, in windows of exactly the length the system deploys at.
 *
 * The acquisition path is NOT under test: these are bedside pulse oximeters on perfused
 * ICU patients, not a phone camera. This isolates the classifier.
 *
 * Usage:  node tests/validate_mimic_af.js [src/af/mimic_af_prepared_30hz.json]
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");

global.window = global;
const RPPG = require(path.join(ROOT, "src/rppg_core.js"));
const AFC  = require(path.join(ROOT, "src/af/af_core.js"));
global.RPPG = RPPG; global.BrainShieldAF = AFC;
const AF_MODEL = JSON.parse(fs.readFileSync(path.join(ROOT, "src/af/af_model_export.json"), "utf8"));

const DATA = process.argv[2] || path.join(ROOT, "src/af/mimic_af_prepared_30hz.json");
if (!fs.existsSync(DATA)) {
    console.error("Missing " + DATA + "\nRun: python3 src/af/prep_mimic_af.py <af_dir> <non_af_dir> --fs 30");
    process.exit(2);
}
const D = JSON.parse(fs.readFileSync(DATA, "utf8"));
const recs = D.records;
console.log("=".repeat(78));
console.log("FROZEN CLASSIFIER on MIMIC PERform AF — fingertip PPG at the deployed window");
console.log("=".repeat(78));
console.log(`\n  ${recs.length} windows of ${D._segment_seconds.toFixed(0)} s at ${D._fs} Hz, ${new Set(recs.map(r=>r.subject)).size} subjects`);
console.log(`  source: ${D._source}\n`);

const out = [];
for (const r of recs) {
    const n = r.ppg.length, fsHz = r.fs;
    const times = []; for (let i = 0; i < n; i++) times.push(i / fsHz);
    const samples = {times, r: r.ppg, g: r.ppg, b: r.ppg};
    const row = {id: r.id, subject: r.subject, af: r.af, call: null, p: null};

    const res = RPPG.analyse(samples, {mode: "finger", fs: Math.round(fsHz)});
    if (!res.ok) { row.call = "unusable"; row.why = res.reason; out.push(row); continue; }
    const peaks = AFC.refinePeaks(res.waveform, RPPG.findPeaks(res.waveform, res.fs, (res.hr||70)/60));
    const rr = AFC.rrFromPeaks(peaks, res.fs);
    row.n_rr = rr.length;
    if (rr.length < 4) { row.call = "unusable"; row.why = "too few intervals"; out.push(row); continue; }
    const p = AFC.classify(AFC.afFeatures(rr), AF_MODEL);
    if (p === null) { row.call = "unusable"; row.why = "non-finite feature"; out.push(row); continue; }
    row.p = p;
    row.call = p >= AF_MODEL.threshold_rule_out ? "irregular" : "regular";
    out.push(row);
}

const tag = String(Math.round(D._fs));
fs.writeFileSync(path.join(ROOT, `src/af/mimic_af_results_${tag}hz.json`), JSON.stringify({rows: out}, null, 1));

const scored = out.filter(r => r.p !== null);
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
const sens = tp+fn ? tp/(tp+fn) : NaN, spec = tn+fp ? tn/(tn+fp) : NaN;

console.log("\n" + "-".repeat(78));
console.log("PER WINDOW");
console.log("-".repeat(78));
console.log(`                    called irregular   called regular`);
console.log(`  AF present        ${String(tp).padStart(12)}     ${String(fn).padStart(12)}`);
console.log(`  AF absent         ${String(fp).padStart(12)}     ${String(tn).padStart(12)}`);
console.log(`\n  SENSITIVITY ${(100*sens).toFixed(1)}%  [${tp}/${tp+fn}]      SPECIFICITY ${(100*spec).toFixed(1)}%  [${tn}/${tn+fp}]`);

const ps = scored.map(r => ({p: r.p, y: r.af === AFLBL ? 1 : 0})).sort((a,b) => b.p - a.p);
const P = ps.filter(x=>x.y===1).length, N = ps.length - P;
let tpc=0, auc=0;
for (const x of ps) { if (x.y===1) tpc++; else auc += tpc; }
console.log(`  AUROC ${((P&&N)?auc/(P*N):NaN).toFixed(4)}   (MIT-BIH, ECG intervals: ${AF_MODEL.performance.auroc_record_held_out.toFixed(4)})`);
console.log(`  Reference on ECG intervals: sensitivity ${(100*AF_MODEL.performance.sensitivity_at_threshold).toFixed(0)}%, specificity ${(100*AF_MODEL.performance.specificity_at_threshold).toFixed(1)}%`);

/* Labels are constant within a subject, so 35 subjects is the effective sample size. */
const bySubj = new Map();
for (const r of scored) { if (!bySubj.has(r.subject)) bySubj.set(r.subject, []); bySubj.get(r.subject).push(r); }
let sTP=0,sFN=0,sFP=0,sTN=0; const rows=[];
for (const k of [...bySubj.keys()].sort()) {
    const rs = bySubj.get(k), truth = rs[0].af === AFLBL;
    const frac = rs.filter(r => r.call === "irregular").length / rs.length;
    const call = frac > 0.5;
    if (truth&&call) sTP++; else if (truth&&!call) sFN++; else if (!truth&&call) sFP++; else sTN++;
    if (truth !== call) rows.push(`    ${k}  truth ${truth?"AF":"non-AF"}  windows ${rs.length}  frac irregular ${frac.toFixed(3)}`);
}
console.log("\n" + "-".repeat(78));
console.log(`PER SUBJECT  (label constant within subject; effective n = ${bySubj.size})`);
console.log("-".repeat(78));
const ss=(sTP+sFN)?100*sTP/(sTP+sFN):NaN, sp2=(sTN+sFP)?100*sTN/(sTN+sFP):NaN;
console.log(`  SENSITIVITY ${ss.toFixed(1)}%  [${sTP}/${sTP+sFN}]      SPECIFICITY ${sp2.toFixed(1)}%  [${sTN}/${sTN+sFP}]`);
if (rows.length) { console.log("\n  misclassified subjects:"); rows.forEach(r=>console.log(r)); }
console.log();
