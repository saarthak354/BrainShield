/*
 * Independent validation of the FROZEN pipeline on real phone-camera recordings.
 *
 * Runs the ACTUAL shipped JavaScript (rppg_core.js -> af_core.js -> af_capture.js)
 * rather than a Python re-implementation. The first attempt at this validation used a
 * Python re-implementation and silently diverged from the shipped code (it hardcoded
 * the red channel where the real pipeline selects the best channel), producing numbers
 * that described nothing. Running the real files removes that whole class of error.
 *
 * BEAT MATCHING. PPG peaks lag ECG R-waves by the pulse transit time, and either
 * stream can miss or add a beat. Comparing RR[i] to RR[i] element-wise therefore
 * compares different heartbeats -- that bug produced an apparent 267 ms RR error,
 * which is simply the size of one heartbeat. Here each PPG peak is matched to an ECG
 * beat in TIME after estimating the lag, and RR error is computed only across pairs
 * where both endpoints matched.
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");

global.window = global;
// rppg_core.js sets module.exports OR the global, not both, so capture the return
// value rather than reaching for globalThis.RPPG. af_capture.js reads root.RPPG and
// root.BrainShieldAF at call time, so those globals must also be populated by hand.
const RPPG = require(path.join(ROOT, "src/rppg_core.js"));
const AFC  = require(path.join(ROOT, "src/af/af_core.js"));
global.RPPG = RPPG;
global.BrainShieldAF = AFC;
const CAP  = require(path.join(ROOT, "src/af/af_capture.js"));
const AF_MODEL = JSON.parse(fs.readFileSync(path.join(ROOT, "src/af/af_model_export.json"), "utf8"));

const cache = JSON.parse(fs.readFileSync(path.join(ROOT, "src/af/butppg_cache.json"), "utf8"));
const MOTION = {0:"rest",1:"finger pressure",2:"moving on lens",3:"walking",
                4:"coughing",5:"laughing",6:"changing light",7:"talking"};
const MATCH_TOL = 0.10;   // 100 ms: a matched beat must be this close after lag removal

/**
 * Find the PPG->ECG lag that maximises matched beats.
 *
 * Match count alone is not enough: with ~800 ms between beats and a +/-100 ms
 * tolerance, a whole plateau of lags matches every beat, and taking the first one
 * biases the estimate low (a self-test with a true 250 ms lag reported 155 ms).
 * So among the lags that tie on match count, take the one that minimises the total
 * timing offset -- that is the actual transit delay.
 */
function bestLag(ppgT, ecgT) {
    // Search a FULL beat interval in both directions. Which ECG beat a given PPG peak
    // "belongs" to is ambiguous when the offset approaches one RR interval, and the
    // detected PPG landmark is not necessarily the systolic peak (it may be the
    // dicrotic/diastolic feature, adding a few hundred ms). Observed per-record offsets
    // on this dataset span -24 to -243 ms, so a narrow window silently fails to match
    // whole recordings -- that alone dropped apparent beat sensitivity to 41.7%.
    // A constant per-record offset does not affect RR INTERVALS, which is what we measure.
    const rr = [];
    for (let i = 1; i < ecgT.length; i++) rr.push(ecgT[i] - ecgT[i-1]);
    rr.sort((a,b)=>a-b);
    const medRR = rr.length ? rr[Math.floor(rr.length/2)] : 0.8;
    const span = Math.max(0.8, 1.2 * medRR);
    let best = {lag: 0, n: -1, cost: Infinity};
    for (let lag = -span; lag <= span; lag += 0.005) {
        let n = 0, cost = 0;
        for (const p of ppgT) {
            const t = p - lag;
            let bd = Infinity;
            for (const e of ecgT) { const d = Math.abs(e - t); if (d < bd) bd = d; }
            if (bd <= MATCH_TOL) { n++; cost += bd; }
        }
        if (n > best.n || (n === best.n && cost < best.cost)) best = {lag, n, cost};
    }
    return best;
}

/** Greedy nearest-neighbour matching, one ECG beat per PPG peak. */
function matchBeats(ppgT, ecgT, lag) {
    const used = new Set(), pairs = [];
    ppgT.forEach((p, pi) => {
        const t = p - lag;
        let bi = -1, bd = Infinity;
        ecgT.forEach((e, ei) => {
            if (used.has(ei)) return;
            const d = Math.abs(e - t);
            if (d < bd) { bd = d; bi = ei; }
        });
        if (bi >= 0 && bd <= MATCH_TOL) { used.add(bi); pairs.push({pi, ei: bi, d: bd}); }
    });
    pairs.sort((a, b) => a.pi - b.pi);
    return pairs;
}

function peaksCountFor(res) {
    return AFC.refinePeaks(res.waveform,
        RPPG.findPeaks(res.waveform, res.fs, (res.hr || 70) / 60)).length;
}

const rows = [];
for (const rec of cache.records) {
    const n = rec.n, fsHz = rec.fs;
    const times = []; for (let i = 0; i < n; i++) times.push(i / fsHz);
    const samples = {times, r: rec.r, g: rec.g, b: rec.b};

    // ---- the real shipped analysis ----
    const res = RPPG.analyse(samples, {mode: "finger", fs: Math.round(fsHz)});
    const row = {id: rec.id, quality: rec.quality, motion: rec.motion, age: rec.age,
                 gender: rec.gender, analysed: !!res.ok};
    if (!res.ok) { row.gate = false; row.reason = res.reason; row.af_call = "rejected"; rows.push(row); continue; }

    const q = CAP.afQuality(res, samples);
    row.gate = !!q.ok;
    row.reason = q.ok ? "ok" : q.reason;

    /* The gate's minimum-beat criterion (>=12 beats) is calibrated for the 60-second
       capture the app actually records. A 10-second BUT PPG clip CONTAINS only ~12
       beats at 70 bpm, so that criterion rejects almost everything here for reasons
       that have nothing to do with signal quality -- 243 of 271 rejections.
       `gate_fair` re-applies every OTHER criterion, so the gate's real discriminating
       ability can be seen. Both are reported; neither is hidden. */
    const durS = times[times.length-1] - times[0];
    const expectedBeats = Math.max(4, Math.floor(durS * (res.hr || 70) / 60 * 0.7));
    row.gate_fair = q.ok || (/Too few heartbeats/i.test(q.reason || "")
                             && peaksCountFor(res) >= expectedBeats);
    row.rppg_quality = res.quality;
    row.hr_ppg = res.hr;

    const ecgT = rec.ecg_beats_s;
    const ecgRR = [];
    for (let i = 1; i < ecgT.length; i++) ecgRR.push((ecgT[i] - ecgT[i-1]) * 1000);
    row.hr_ecg = 60000 / (ecgRR.reduce((a,c)=>a+c,0) / ecgRR.length);
    row.n_ecg_beats = ecgT.length;

    const peaksIdx = AFC.refinePeaks(res.waveform, RPPG.findPeaks(res.waveform, res.fs, (res.hr||70)/60));
    const ppgT = peaksIdx.map(p => times[0] + p / res.fs);
    row.n_ppg_peaks = ppgT.length;

    const {lag, n: nMatch} = bestLag(ppgT, ecgT);
    const pairs = matchBeats(ppgT, ecgT, lag);
    row.lag_ms = lag * 1000;
    row.n_matched = pairs.length;
    // beat-detection accuracy
    row.beat_sensitivity = pairs.length / ecgT.length;          // ECG beats we found
    row.beat_ppv = ppgT.length ? pairs.length / ppgT.length : 0; // our peaks that are real

    // RR error only across CONSECUTIVE matched pairs on both sides
    const errs = [];
    for (let i = 1; i < pairs.length; i++) {
        const a = pairs[i-1], b = pairs[i];
        if (b.ei !== a.ei + 1) continue;            // an ECG beat was skipped: not comparable
        const ppgRR = (ppgT[b.pi] - ppgT[a.pi]) * 1000;
        const refRR = (ecgT[b.ei] - ecgT[a.ei]) * 1000;
        errs.push(ppgRR - refRR);
    }
    row.n_rr_compared = errs.length;
    if (errs.length) {
        row.rr_mae = errs.reduce((s,e)=>s+Math.abs(e),0) / errs.length;
        row.rr_bias = errs.reduce((s,e)=>s+e,0) / errs.length;
        row.rr_rmse = Math.sqrt(errs.reduce((s,e)=>s+e*e,0) / errs.length);
    }
    row.hr_err = Math.abs(row.hr_ppg - row.hr_ecg);

    if (q.ok) {
        const rr = AFC.rrFromPeaks(peaksIdx, res.fs);
        const feats = rr.length >= 4 ? AFC.afFeatures(rr) : null;
        const p = feats ? AFC.classify(feats, AF_MODEL) : null;
        row.af_prob = p;
        row.af_call = p === null ? "inconclusive"
                    : (p >= AF_MODEL.threshold_rule_out ? "irregular" : "regular");
    } else row.af_call = "rejected";
    rows.push(row);
}

fs.writeFileSync(path.join(ROOT, "src/af/butppg_js_results.json"),
                 JSON.stringify({skipped: cache.skipped, rows}, null, 1));

// ------------------------------------------------------------------ report ----
const med = a => { const v=a.filter(x=>x!=null&&isFinite(x)).sort((x,y)=>x-y);
                   return v.length ? v[Math.floor(v.length/2)] : NaN; };
const p90 = a => { const v=a.filter(x=>x!=null&&isFinite(x)).sort((x,y)=>x-y);
                   return v.length ? v[Math.floor(v.length*0.9)] : NaN; };
const pc = (n,d) => d ? (100*n/d) : NaN;

const gated = rows.filter(r => r.gate);
const scored = gated.filter(r => r.rr_mae != null);
console.log("=".repeat(78));
console.log("INDEPENDENT VALIDATION — frozen JS pipeline on BUT PPG (real phone camera)");
console.log("=".repeat(78));
const fair = rows.filter(r => r.gate_fair);
console.log(`\nRecords analysed                ${rows.length}`);
console.log(`  quality gate passed           ${gated.length}  (${pc(gated.length,rows.length).toFixed(1)}%)`);
console.log(`  quality gate rejected         ${rows.length-gated.length}  (${pc(rows.length-gated.length,rows.length).toFixed(1)}%)`);
console.log(`\n  NOTE: the gate requires >=12 beats, a floor set for the 60s capture the app`);
console.log(`  records. A 10s clip contains only ~12 beats, so that one criterion causes`);
console.log(`  ${rows.filter(r=>!r.gate && /Too few/i.test(r.reason||"")).length} of the ${rows.length-gated.length} rejections here for reasons unrelated to signal quality.`);
console.log(`  Excluding only that criterion: ${fair.length} pass (${pc(fair.length,rows.length).toFixed(1)}%).`);

console.log("\n" + "-".repeat(78));
console.log("BEAT DETECTION vs ECG  (all analysable records, gate aside)");
console.log("-".repeat(78));
const det = rows.filter(r => r.beat_sensitivity != null);
console.log(`  ECG beats correctly found (sensitivity)  median ${(100*med(det.map(r=>r.beat_sensitivity))).toFixed(1)}%`);
console.log(`  detected peaks that are real (PPV)       median ${(100*med(det.map(r=>r.beat_ppv))).toFixed(1)}%`);
console.log(`  estimated pulse transit lag              median ${med(det.map(r=>r.lag_ms)).toFixed(0)} ms`);

console.log("\n" + "-".repeat(78));
console.log("RR INTERVAL ACCURACY  (matched consecutive beats only)");
console.log("-".repeat(78));
if (scored.length) {
  console.log(`  records with comparable intervals  ${scored.length}`);
  console.log(`  RR error   median ${med(scored.map(r=>r.rr_mae)).toFixed(1)} ms   p90 ${p90(scored.map(r=>r.rr_mae)).toFixed(1)} ms`);
  console.log(`  RR bias    median ${med(scored.map(r=>r.rr_bias)).toFixed(1)} ms`);
  const within = scored.filter(r=>r.rr_mae < 25).length;
  console.log(`  within the 25 ms tolerance:  ${within}/${scored.length} (${pc(within,scored.length).toFixed(0)}%)`);
  const hr5 = gated.filter(r=>r.hr_err <= 5).length;
  console.log(`  heart rate within 5 bpm:     ${hr5}/${gated.length} (${pc(hr5,gated.length).toFixed(0)}%)`);
  console.log(`  heart rate error  median ${med(gated.map(r=>r.hr_err)).toFixed(1)} bpm`);
}

console.log("\n" + "-".repeat(78));
console.log("FALSE POSITIVES  (cohort has essentially no AF, so every 'irregular' is wrong)");
console.log("-".repeat(78));
console.log("  CAVEAT: 10s recordings; the shipped classifier needs 60s. ~12 intervals");
console.log("  instead of ~70 makes the statistics far noisier. Pessimistic upper bound.");
const calls = {};
rows.forEach(r => calls[r.af_call] = (calls[r.af_call]||0)+1);
Object.keys(calls).sort().forEach(k => console.log(`    ${k.padEnd(14)} ${String(calls[k]).padStart(4)}  (${pc(calls[k],rows.length).toFixed(1)}%)`));
const verdicts = (calls.regular||0)+(calls.irregular||0);
if (verdicts) console.log(`\n  Of ${verdicts} verdicts: ${calls.irregular||0} irregular = ${pc(calls.irregular||0,verdicts).toFixed(1)}% false positive`);

console.log("\n" + "-".repeat(78));
console.log("BY CONDITION");
console.log("-".repeat(78));
console.log("  condition            n   gate-pass  beat-sens   RR err   HR err  irregular");
console.log("  " + "-".repeat(72));
Object.keys(MOTION).map(Number).sort((a,b)=>a-b).forEach(code => {
  const sub = rows.filter(r => r.motion === code);
  if (!sub.length) return;
  const g = sub.filter(r=>r.gate), s = g.filter(r=>r.rr_mae!=null);
  const v = sub.filter(r=>r.af_call==="regular"||r.af_call==="irregular").length;
  const irr = sub.filter(r=>r.af_call==="irregular").length;
  console.log(`  ${MOTION[code].padEnd(18)} ${String(sub.length).padStart(3)}   ${pc(g.length,sub.length).toFixed(0).padStart(6)}%  ` +
    `${(100*med(sub.map(r=>r.beat_sensitivity))).toFixed(0).padStart(7)}%  ${med(s.map(r=>r.rr_mae)).toFixed(1).padStart(7)}ms  ` +
    `${med(g.map(r=>r.hr_err)).toFixed(1).padStart(6)}  ${(v?pc(irr,v):NaN).toFixed(0).padStart(7)}%`);
});

console.log("\n" + "-".repeat(78));
console.log("BY SUBSET — does the gate select the recordings it should?");
console.log("-".repeat(78));
console.log("  subset                          n   beat-sens    PPV    RR err   HR err");
console.log("  " + "-".repeat(72));
[["all records", rows],
 ["dataset says GOOD quality", rows.filter(r=>r.quality===1)],
 ["dataset says POOR quality", rows.filter(r=>r.quality===0)],
 ["OUR gate passed", rows.filter(r=>r.gate)],
 ["OUR gate rejected", rows.filter(r=>!r.gate)]].forEach(([lab, sub]) => {
  if (!sub.length) return;
  const sc = sub.filter(r=>r.rr_mae!=null);
  console.log(`  ${lab.padEnd(28)} ${String(sub.length).padStart(4)}   ` +
    `${(100*med(sub.map(r=>r.beat_sensitivity))).toFixed(0).padStart(6)}%  ` +
    `${(100*med(sub.map(r=>r.beat_ppv))).toFixed(0).padStart(5)}%  ` +
    `${med(sc.map(r=>r.rr_mae)).toFixed(1).padStart(7)}ms  ` +
    `${med(sub.map(r=>r.hr_err)).toFixed(1).padStart(5)} bpm`);
});

console.log("\n" + "-".repeat(78));
console.log("AGREEMENT WITH THE DATASET'S HUMAN QUALITY ANNOTATORS");
console.log("-".repeat(78));
[[1,"annotated GOOD"],[0,"annotated POOR"]].forEach(([lab,name]) => {
  const sub = rows.filter(r => r.quality === lab);
  if (!sub.length) return;
  const g = sub.filter(r=>r.gate), s = g.filter(r=>r.rr_mae!=null);
  console.log(`  ${name}  n=${String(sub.length).padStart(4)}   our gate passed ${pc(g.length,sub.length).toFixed(1).padStart(5)}%   ` +
    `RR err ${med(s.map(r=>r.rr_mae)).toFixed(1)} ms   HR err ${med(g.map(r=>r.hr_err)).toFixed(1)} bpm`);
});
console.log();
