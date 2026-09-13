/*
 * Atrial fibrillation screening from pulse intervals.
 *
 * Mirrors src/af/build_afdb.py feature-for-feature; parity is enforced by
 * tests/test_af_parity.js against fixtures computed in Python.
 *
 * Design rule enforced in decide(): this module may conclude "rhythm looks regular"
 * but must never conclude "you have AF". On phone-camera PPG in an unscreened
 * population the positive predictive value is roughly 20-40% while the negative
 * predictive value is near 100% (meta-analysis, 10 studies / 3,852 participants,
 * PMC7125433). So a negative is trustworthy and a positive is a referral, not a finding.
 */
(function (root) {
  "use strict";

  /* ---- sub-sample peak refinement --------------------------------------
   * A 30 fps camera quantises every beat to a 33 ms grid, which is the same order
   * as the beat-to-beat variability that distinguishes normal rhythm from AF.
   * Fitting a parabola through the peak sample and its two neighbours recovers the
   * true maximum to a fraction of a frame, which is what makes the interval
   * statistics usable at all at this frame rate.
   * -------------------------------------------------------------------- */
  function refinePeaks(sig, peaks) {
    var out = [];
    for (var i = 0; i < peaks.length; i++) {
      var p = peaks[i];
      if (p <= 0 || p >= sig.length - 1) { out.push(p); continue; }
      var y0 = sig[p - 1], y1 = sig[p], y2 = sig[p + 1];
      var denom = y0 - 2 * y1 + y2;
      // denom >= 0 means this is not a strict local max; fall back to the integer index
      var delta = (denom < 0) ? 0.5 * (y0 - y2) / denom : 0;
      if (!isFinite(delta) || Math.abs(delta) > 0.5) delta = 0;
      out.push(p + delta);
    }
    return out;
  }

  /** Peak indices (possibly fractional) -> RR intervals in milliseconds. */
  function rrFromPeaks(peaks, fs) {
    var rr = [];
    for (var i = 1; i < peaks.length; i++) rr.push((peaks[i] - peaks[i - 1]) / fs * 1000);
    return rr.filter(function (v) { return v > 250 && v < 2500; });
  }

  /* ---- features (must match build_afdb.py) ------------------------------ */
  function mean(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function varianceSample(a) {
    if (a.length < 2) return NaN;
    var m = mean(a), s = 0;
    for (var i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return s / (a.length - 1);
  }
  function percentile(sorted, q) {
    var i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
    return lo === hi ? sorted[lo] : sorted[lo] + (i - lo) * (sorted[hi] - sorted[lo]);
  }
  function diff(a) { var d = []; for (var i = 1; i < a.length; i++) d.push(a[i] - a[i - 1]); return d; }

  function shannonEntropy(rr, bins) {
    bins = bins || 16;
    if (rr.length < 4) return NaN;
    var sorted = rr.slice().sort(function (x, y) { return x - y; });
    var lo = percentile(sorted, 0.01), hi = percentile(sorted, 0.99);
    if (hi <= lo) return 0;
    var h = new Array(bins).fill(0), i, idx, n = 0;
    for (i = 0; i < rr.length; i++) {
      if (rr[i] < lo || rr[i] > hi) continue;
      idx = Math.min(bins - 1, Math.floor((rr[i] - lo) / (hi - lo) * bins));
      h[idx]++; n++;
    }
    if (!n) return 0;
    var e = 0;
    for (i = 0; i < bins; i++) { var p = h[i] / n; if (p > 0) e -= p * Math.log(p); }
    return e / Math.log(bins);
  }

  function sampleEntropy(rr, m, rFrac) {
    m = m || 2; rFrac = rFrac || 0.2;
    var n = rr.length;
    if (n < m + 2) return NaN;
    var sd = Math.sqrt(varianceSample(rr) * (n - 1) / n);   // population SD, matches numpy
    var r = rFrac * sd;
    if (!(r > 0)) return 0;

    function count(mm) {
      var k = n - mm + 1, c = 0, i, j, d, q;
      if (k < 2) return 0;
      for (i = 0; i < k - 1; i++) {
        for (j = i + 1; j < k; j++) {
          d = 0;
          for (q = 0; q < mm; q++) d = Math.max(d, Math.abs(rr[i + q] - rr[j + q]));
          if (d <= r) c++;
        }
      }
      return c;
    }
    var a = count(m + 1), b = count(m);
    // See build_afdb.py: no matching templates means maximal irregularity, not an
    // undefined value. Returning NaN here rejected the clearest AF cases outright.
    var upper = (n > m + 1) ? -Math.log(2.0 / ((n - m - 1) * (n - m))) : 0;
    if (a === 0 || b === 0) return upper;
    return Math.min(-Math.log(a / b), upper);
  }

  function poincare(rr) {
    if (rr.length < 3) return { sd1: NaN, sd2: NaN };
    var d = diff(rr);
    var sd1 = d.length > 1 ? Math.sqrt(varianceSample(d) / 2) : NaN;
    var sd = varianceSample(rr);
    var sd2sq = 2 * sd - sd1 * sd1;
    return { sd1: sd1, sd2: sd2sq > 0 ? Math.sqrt(sd2sq) : NaN };
  }

  function afFeatures(rr) {
    if (!rr || rr.length < 4) return null;
    var meanRR = mean(rr), d = diff(rr), i, s = 0;
    for (i = 0; i < d.length; i++) s += d[i] * d[i];
    var rmssd = d.length ? Math.sqrt(s / d.length) : NaN;
    var pc = poincare(rr);
    var over50 = 0, over70 = 0;
    for (i = 0; i < d.length; i++) {
      if (Math.abs(d[i]) > 50) over50++;
      if (Math.abs(d[i]) > 70) over70++;
    }
    return {
      mean_rr: meanRR,
      hr: 60000 / meanRR,
      rmssd: rmssd,
      rmssd_norm: rmssd / meanRR,
      cv: Math.sqrt(varianceSample(rr)) / meanRR,
      prr50: d.length ? over50 / d.length : NaN,
      prr70: d.length ? over70 / d.length : NaN,
      shannon: shannonEntropy(rr),
      sampen: sampleEntropy(rr),
      sd1: pc.sd1,
      sd2: pc.sd2,
      sd1_sd2: (pc.sd2 > 0) ? pc.sd1 / pc.sd2 : NaN
    };
  }

  /* ---- classifier ------------------------------------------------------- */
  function classify(feats, model) {
    if (!model) throw new Error("AF_MODEL not loaded");
    var z = model.intercept, i, name, v;
    for (i = 0; i < model.feature_names.length; i++) {
      name = model.feature_names[i];
      v = feats[name];
      if (v === undefined || v === null || !isFinite(v)) return null;
      z += model.coef[i] * ((v - model.mean[i]) / model.scale[i]);
    }
    return 1 / (1 + Math.exp(-z));
  }

  /* ---- the asymmetric decision ------------------------------------------
   * Three outcomes, never two. "Positive" is deliberately not a conclusion.
   * -------------------------------------------------------------------- */
  function decide(feats, model, quality) {
    if (!quality || !quality.ok) {
      return {
        outcome: "unusable",
        sets_af: null,
        headline: "Couldn't read your pulse clearly",
        detail: (quality && quality.reason) || "Signal quality too low to analyse.",
        probability: null
      };
    }
    var p = classify(feats, model);
    if (p === null) {
      return { outcome: "unusable", sets_af: null,
               headline: "Couldn't read your pulse clearly",
               detail: "Not enough clean beats to assess the rhythm.", probability: null };
    }
    if (p < model.threshold_rule_out) {
      return {
        outcome: "regular", sets_af: false, probability: p,
        headline: "Your pulse looked regular",
        detail: "No sign of an irregular rhythm in this recording. This is used to rule "
              + "atrial fibrillation out of your risk estimate. It is not a substitute for "
              + "an ECG, and AF can come and go — a single regular reading does not prove "
              + "you have never had it."
      };
    }
    return {
      outcome: "irregular", sets_af: null, probability: p,
      headline: "Your pulse looked irregular — worth getting an ECG",
      detail: "This does NOT mean you have atrial fibrillation. On a phone camera, most "
            + "irregular readings turn out not to be AF — common causes are movement, "
            + "pressing too hard, or ordinary ectopic beats. But AF is worth ruling out "
            + "properly, because it is treatable and it matters for stroke risk. Your risk "
            + "estimate has been left unchanged rather than assuming the worst."
    };
  }

  var API = {
    refinePeaks: refinePeaks, rrFromPeaks: rrFromPeaks, afFeatures: afFeatures,
    classify: classify, decide: decide,
    shannonEntropy: shannonEntropy, sampleEntropy: sampleEntropy, poincare: poincare
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.BrainShieldAF = API;
})(typeof globalThis !== "undefined" ? globalThis : this);
