/*
 * Enforces that src/af/af_core.js computes the same AF features as src/af/build_afdb.py.
 *
 * The classifier coefficients are fitted in Python against Python-computed features.
 * If the JS computes them even slightly differently, the coefficients are being applied
 * to the wrong numbers and the probability is silently wrong. This test is the thing
 * standing between that and a shipped rhythm screen.
 */
const fs = require("fs");
const path = require("path");
const ROOT = path.join(__dirname, "..");
const AF = require(path.join(ROOT, "src/af/af_core.js"));
const fixture = JSON.parse(fs.readFileSync(path.join(ROOT, "tests/af_parity_fixture.json"), "utf8"));

let pass = 0, fail = 0, worst = 0, worstName = "";
const TOL = 1e-9;

for (const c of fixture.cases) {
  const got = AF.afFeatures(c.rr);
  if (!got) { console.error(`FAIL ${c.label}: afFeatures returned null`); fail++; continue; }
  for (const [k, exp] of Object.entries(c.expected)) {
    const g = got[k];
    if (exp === null) {
      if (g === null || g === undefined || Number.isNaN(g)) pass++;
      else { console.error(`FAIL ${c.label}.${k}: python NaN, js ${g}`); fail++; }
      continue;
    }
    if (g === undefined || Number.isNaN(g)) {
      console.error(`FAIL ${c.label}.${k}: python ${exp}, js NaN`); fail++; continue;
    }
    const rel = Math.abs(g - exp) / Math.max(1e-12, Math.abs(exp));
    if (rel > worst) { worst = rel; worstName = `${c.label}.${k}`; }
    if (rel < TOL) pass++;
    else { console.error(`FAIL ${c.label}.${k}: python ${exp}, js ${g}, rel ${rel.toExponential(2)}`); fail++; }
  }
}
console.log(`Feature parity: ${pass} matched, ${fail} diverged`);
console.log(`Worst relative difference: ${worst.toExponential(3)} (${worstName})`);

function check(name, ok, detail) {
  if (ok) { console.log(`[PASS] ${name}${detail ? "  -- " + detail : ""}`); pass++; }
  else { console.error(`[FAIL] ${name}${detail ? "  -- " + detail : ""}`); fail++; }
}

// ---- sub-sample peak refinement ----
// A clean sinusoid sampled off-grid: refined peaks must land closer to truth.
// 1.17 Hz at 30 fps is 25.64 samples per cycle -- deliberately NOT a whole number of
// frames, so peaks fall between samples and quantisation error is actually present.
// (A commensurate rate like 1.2 Hz = exactly 25 samples/cycle has zero error to fix.)
const fs_ = 30, freqHz = 1.17, N = 600;
const sig = [];
for (let i = 0; i < N; i++) sig.push(Math.sin(2 * Math.PI * freqHz * (i / fs_) + 0.37));
const raw = [];
for (let i = 1; i < N - 1; i++) if (sig[i] > sig[i - 1] && sig[i] >= sig[i + 1]) raw.push(i);
const refined = AF.refinePeaks(sig, raw);
const rawRR = AF.rrFromPeaks(raw, fs_);
const refRR = AF.rrFromPeaks(refined, fs_);
const trueRR = 1000 / freqHz;
const err = a => Math.sqrt(a.reduce((s, v) => s + (v - trueRR) ** 2, 0) / a.length);
check("sub-sample interpolation reduces RR error",
  err(refRR) < err(rawRR),
  `raw ${err(rawRR).toFixed(2)} ms -> refined ${err(refRR).toFixed(2)} ms`);
check("refined RR error is under one frame (33 ms)", err(refRR) < 33.3,
  `${err(refRR).toFixed(2)} ms`);

// ---- AF features must actually separate the two rhythms ----
const sinusF = fixture.cases.filter(c => c.label.startsWith("sinus")).map(c => AF.afFeatures(c.rr));
const afF = fixture.cases.filter(c => c.label.startsWith("af_")).map(c => AF.afFeatures(c.rr));
const meanOf = (arr, k) => arr.reduce((s, f) => s + f[k], 0) / arr.length;
check("RMSSD(normalised) separates AF from sinus",
  meanOf(afF, "rmssd_norm") > 2 * meanOf(sinusF, "rmssd_norm"),
  `sinus ${meanOf(sinusF, "rmssd_norm").toFixed(4)} vs AF ${meanOf(afF, "rmssd_norm").toFixed(4)}`);
check("pRR50 separates AF from sinus",
  meanOf(afF, "prr50") > meanOf(sinusF, "prr50"),
  `sinus ${meanOf(sinusF, "prr50").toFixed(3)} vs AF ${meanOf(afF, "prr50").toFixed(3)}`);

// ---- the safety rule ----
const model = {
  feature_names: ["rmssd_norm"], coef: [5], intercept: 0,
  mean: [0.05], scale: [0.05], threshold_rule_out: 0.5
};
const regular = AF.decide(AF.afFeatures(fixture.cases.find(c => c.label === "sinus_0").rr),
                          model, { ok: true });
const irregular = AF.decide(AF.afFeatures(fixture.cases.find(c => c.label === "af_0").rr),
                            model, { ok: true });
check("a regular reading rules AF out (sets_af === false)", regular.sets_af === false, regular.outcome);
check("an irregular reading does NOT set AF true", irregular.sets_af === null, irregular.outcome);
// The wording must explicitly negate the diagnosis, not merely avoid the phrase.
check("an irregular reading explicitly denies it is a diagnosis",
  /does not mean you have atrial fibrillation/i.test(irregular.detail));
check("an irregular reading leaves the risk estimate unchanged",
  /left unchanged|unchanged rather than/i.test(irregular.detail));
check("an irregular reading tells the user to get an ECG",
  /ecg/i.test(irregular.headline + " " + irregular.detail));
check("poor signal quality produces no conclusion",
  AF.decide(AF.afFeatures(fixture.cases[0].rr), model, { ok: false, reason: "too noisy" }).sets_af === null);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
