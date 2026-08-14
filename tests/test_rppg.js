/* Validation suite for the rPPG DSP core.
   Feeds synthetic signals with a KNOWN heart rate and checks recovery. */
const RPPG = require("./rppg_core.js");

function run(label, opts, mode) {
  const s = RPPG.synth(opts);
  const res = RPPG.analyse(s, { mode: mode || "face", fs: opts.fs || 30 });
  if (!res.ok) return { label, ok: false, reason: res.reason };
  const err = Math.abs(res.hr - opts.bpm);
  return {
    label, ok: true, truth: opts.bpm, est: res.hr, err,
    snr: res.snr, quality: res.quality, method: res.method,
    hrv: res.hrv ? { sdnn: res.hrv.sdnn, beats: res.hrv.nBeats } : null
  };
}

let pass = 0, fail = 0;
function check(cond, msg) { if (cond) { pass++; console.log("  PASS  " + msg); } else { fail++; console.log("  FAIL  " + msg); } }

console.log("\n=== 1. Heart-rate recovery across the physiological range (face mode, POS/CHROM) ===");
[45, 55, 62, 72, 85, 100, 120, 150].forEach(bpm => {
  const r = run(`${bpm} bpm`, { bpm, dur: 20, fs: 30, melanin: 0, noise: 0.15 });
  console.log(`  ${String(bpm).padStart(3)} bpm ->  est ${r.est ? r.est.toFixed(1) : "n/a"}  err ${r.err ? r.err.toFixed(2) : "n/a"}  SNR ${r.snr ? r.snr.toFixed(1) : "n/a"} dB  [${r.quality}]  via ${r.method}`);
  check(r.ok && r.err < 2.0, `${bpm} bpm recovered within 2 bpm`);
});

console.log("\n=== 2. Skin-tone / melanin robustness (simulated, face mode) ===");
console.log("  melanin 0 = lightest, 1 = darkest. Attenuates AC pulse amplitude,");
console.log("  which is the documented cause of real-world rPPG accuracy loss.\n");
const melResults = [];
[0, 0.25, 0.5, 0.7, 0.85, 0.95].forEach(m => {
  const errs = [], snrs = [], quals = [];
  [58, 72, 88, 105].forEach((bpm, i) => {
    const r = run("", { bpm, dur: 20, fs: 30, melanin: m, noise: 0.15, seed: i + 1 });
    if (r.ok) { errs.push(r.err); snrs.push(r.snr); quals.push(r.quality); }
  });
  const mae = errs.reduce((a, b) => a + b, 0) / errs.length;
  const msnr = snrs.reduce((a, b) => a + b, 0) / snrs.length;
  const nGoodFair = quals.filter(q => q !== "poor").length;
  melResults.push({ m, mae, msnr, nGoodFair, n: quals.length });
  console.log(`  melanin ${m.toFixed(2)}  ->  MAE ${mae.toFixed(2)} bpm   mean SNR ${msnr.toFixed(1)} dB   usable ${nGoodFair}/${quals.length}`);
});
check(melResults[0].mae < 1.5, "lightest simulated skin: MAE < 1.5 bpm");
check(melResults[melResults.length - 1].mae > melResults[0].mae,
      "degradation with melanin is reproduced (expected — matches published findings)");

console.log("\n=== 2b. Same sweep under REALISTIC conditions (motion artifact present) ===");
console.log("  Motion is the dominant real-world noise source and is NOT reduced by melanin,");
console.log("  so the artifact-to-pulse ratio worsens as the pulse signal shrinks.\n");
const realResults = [];
let totalSilent = 0, totalRuns = 0;
[0, 0.25, 0.5, 0.7, 0.85, 0.95].forEach(m => {
  const errs = [], quals = [], snrs = [];
  let silent = 0;
  [52, 58, 61, 68, 74, 79, 83, 90, 96, 110].forEach((bpm, i) => {
    const s = RPPG.synth({ bpm, dur: 20, fs: 30, melanin: m, noise: 0.4, motion: 0.9,
                           interf: 1.5, interfFreq: 1.35 + (i % 4) * 0.22,
                           interfWander: 0.22, seed: i * 13 + 5 });
    const res = RPPG.analyse(s, { mode: "face", fs: 30 });
    if (res.ok) {
      const e = Math.abs(res.hr - bpm);
      errs.push(e); quals.push(res.quality); snrs.push(res.snr);
      // SILENT FAILURE: confidently reported (not flagged poor) but materially wrong
      if (res.quality !== "poor" && e > 5) silent++;
    }
  });
  const mae = errs.reduce((a, b) => a + b, 0) / errs.length;
  const msnr = snrs.reduce((a, b) => a + b, 0) / snrs.length;
  const nPoor = quals.filter(q => q === "poor").length;
  const within5 = errs.filter(e => e < 5).length;
  totalSilent += silent; totalRuns += errs.length;
  realResults.push({ m, mae, msnr, nPoor, within5, silent, n: errs.length });
  console.log(`  melanin ${m.toFixed(2)}  ->  MAE ${mae.toFixed(2)} bpm   SNR ${msnr.toFixed(1)} dB   within-5bpm ${within5}/${errs.length}   flagged-poor ${nPoor}/${quals.length}   SILENT-FAIL ${silent}`);
});
const first = realResults[0], last = realResults[realResults.length - 1];
console.log(`\n  Silent failures overall (confidently wrong by >5 bpm): ${totalSilent}/${totalRuns}`);
check(first.mae < last.mae, "realistic conditions: error grows with melanin (matches published direction)");
check(last.nPoor >= first.nPoor, "darkest tone triggers more POOR-quality flags (tool warns rather than lies)");
check(totalSilent / totalRuns <= 0.05, "silent-failure rate <= 5% (wrong readings are flagged, not shown confidently)");

console.log("\n=== 2c. Worst case: perfectly METRONOMIC in-band interferer (irreducible) ===");
console.log("  A rhythmic disturbance at a fixed, heart-rate-plausible frequency with a");
console.log("  pulse-like colour signature is physically ambiguous — no single-camera");
console.log("  algorithm can separate it from a real pulse. Documented, not 'solved'.\n");
let metroSilent = 0, metroN = 0, metroFlag = 0;
[0.85, 0.95].forEach(m => {
  [58, 72, 90].forEach((bpm, i) => {
    const s = RPPG.synth({ bpm, dur: 20, fs: 30, melanin: m, noise: 0.4, motion: 0.9,
                           interf: 1.6, interfFreq: 1.5, interfWander: 0, seed: i * 7 + 2 });
    const res = RPPG.analyse(s, { mode: "face", fs: 30 });
    const e = Math.abs(res.hr - bpm);
    metroN++;
    if (res.quality === "poor") metroFlag++;
    if (res.quality !== "poor" && e > 5) metroSilent++;
    console.log(`  melanin ${m}  ${bpm} bpm -> est ${res.hr.toFixed(1)}  err ${e.toFixed(1)}  [${res.quality}]`);
  });
});
console.log(`\n  Flagged poor: ${metroFlag}/${metroN}   Silent failures: ${metroSilent}/${metroN}`);
console.log("  -> This is why finger+torch is the recommended default (see section 3):");
console.log("     contact PPG raises pulse amplitude far above any motion artifact.");

console.log("\n=== 3. Finger-on-lens contact mode (strong signal, torch) ===");
[60, 72, 95, 130].forEach(bpm => {
  // contact PPG under flash: much larger AC amplitude, melanin matters far less
  const s = RPPG.synth({ bpm, dur: 15, fs: 30, melanin: 0.85, noise: 0.15 });
  for (let i = 0; i < s.r.length; i++) {           // simulate flash-illuminated finger
    const t = i / 30, f = bpm / 60;
    const p = Math.sin(2 * Math.PI * f * t) + 0.35 * Math.sin(4 * Math.PI * f * t + 0.6);
    s.r[i] = 200 + 9 * p; s.g[i] = 60 + 12 * p; s.b[i] = 30 + 5 * p;
  }
  const res = RPPG.analyse(s, { mode: "finger", fs: 30 });
  const err = Math.abs(res.hr - bpm);
  console.log(`  ${String(bpm).padStart(3)} bpm ->  est ${res.hr.toFixed(1)}  err ${err.toFixed(2)}  SNR ${res.snr.toFixed(1)} dB  [${res.quality}]  via ${res.method}`);
  check(err < 2.0, `finger mode ${bpm} bpm recovered within 2 bpm`);
});

console.log("\n=== 4. Rejects garbage instead of inventing a number ===");
const noiseOnly = { times: [], r: [], g: [], b: [] };
let sd = 7;
const rnd = () => { sd = (sd * 1103515245 + 12345) & 0x7fffffff; return (sd / 0x7fffffff) * 2 - 1; };
for (let i = 0; i < 600; i++) {
  noiseOnly.times.push(i / 30);
  noiseOnly.r.push(120 + 25 * rnd()); noiseOnly.g.push(100 + 25 * rnd()); noiseOnly.b.push(90 + 25 * rnd());
}
const nres = RPPG.analyse(noiseOnly, { mode: "face", fs: 30 });
console.log(`  pure noise -> quality "${nres.quality}"  SNR ${nres.snr.toFixed(1)} dB`);
check(nres.quality === "poor", "pure noise is flagged POOR (not reported as a valid reading)");

const tooShort = RPPG.synth({ bpm: 72, dur: 2, fs: 30 });
const sres = RPPG.analyse(tooShort, { mode: "face", fs: 30 });
console.log(`  2s capture -> ok=${sres.ok}  ${sres.reason || ""}`);
check(!sres.ok, "too-short capture is refused");

console.log("\n=== 5. Irregular frame timing (browser jitter) ===");
const jit = RPPG.synth({ bpm: 78, dur: 20, fs: 60 });
const kt = [], kr = [], kg = [], kb = [];
let sj = 3;
const rj = () => { sj = (sj * 1103515245 + 12345) & 0x7fffffff; return sj / 0x7fffffff; };
for (let i = 0; i < jit.times.length; i++) {
  if (rj() < 0.55) continue;                       // randomly drop frames
  kt.push(jit.times[i] + (rj() - 0.5) * 0.012);    // ±6 ms jitter
  kr.push(jit.r[i]); kg.push(jit.g[i]); kb.push(jit.b[i]);
}
const jres = RPPG.analyse({ times: kt, r: kr, g: kg, b: kb }, { mode: "face", fs: 30 });
console.log(`  jittered/dropped frames (${kt.length} samples) -> est ${jres.hr.toFixed(1)} bpm (truth 78)  err ${Math.abs(jres.hr - 78).toFixed(2)}`);
check(Math.abs(jres.hr - 78) < 2.5, "survives irregular frame timing (resampling works)");

console.log(`\n================  ${pass} passed, ${fail} failed  ================\n`);
process.exit(fail === 0 ? 0 : 1);
