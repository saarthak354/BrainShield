/* ============================================================================
   BrainShield — rPPG signal-processing core
   ----------------------------------------------------------------------------
   Recovers a pulse waveform from a sequence of per-frame mean RGB values.

   Face mode  : POS algorithm (Wang et al., "Algorithmic Principles of Remote
                PPG", IEEE TBME 2017) — projects normalized RGB onto a plane
                orthogonal to the skin-tone direction, which suppresses
                intensity/motion artifacts far better than raw green channel.
   Finger mode: contact PPG with torch. POS's skin-reflection model does not
                hold when a finger is pressed to the lens under flash, so we
                instead pick the colour channel with the best in-band SNR.

   All stages are plain JS with no dependencies so the same file runs in Node
   (for the validation tests) and in the browser.
   ========================================================================== */
(function (root) {
  "use strict";

  // ---------------------------------------------------------------- FFT ----
  function nextPow2(n) { var p = 1; while (p < n) p <<= 1; return p; }

  function fft(re, im) {
    var n = re.length, i, j, bit, len, ang, wr, wi, cwr, cwi, ur, ui, vr, vi, nwr, tmp, half;
    for (i = 1, j = 0; i < n; i++) {
      bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        tmp = re[i]; re[i] = re[j]; re[j] = tmp;
        tmp = im[i]; im[i] = im[j]; im[j] = tmp;
      }
    }
    for (len = 2; len <= n; len <<= 1) {
      ang = -2 * Math.PI / len;
      wr = Math.cos(ang); wi = Math.sin(ang);
      half = len >> 1;
      for (i = 0; i < n; i += len) {
        cwr = 1; cwi = 0;
        for (j = 0; j < half; j++) {
          ur = re[i + j]; ui = im[i + j];
          vr = re[i + j + half] * cwr - im[i + j + half] * cwi;
          vi = re[i + j + half] * cwi + im[i + j + half] * cwr;
          re[i + j] = ur + vr; im[i + j] = ui + vi;
          re[i + j + half] = ur - vr; im[i + j + half] = ui - vi;
          nwr = cwr * wr - cwi * wi;
          cwi = cwr * wi + cwi * wr;
          cwr = nwr;
        }
      }
    }
  }

  function ifft(re, im) {
    var n = re.length, i;
    for (i = 0; i < n; i++) im[i] = -im[i];
    fft(re, im);
    for (i = 0; i < n; i++) { re[i] /= n; im[i] = -im[i] / n; }
  }

  // ------------------------------------------------------------ helpers ----
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function std(a) {
    var m = mean(a), s = 0, i;
    for (i = 0; i < a.length; i++) s += (a[i] - m) * (a[i] - m);
    return Math.sqrt(s / Math.max(1, a.length - 1));
  }
  function detrend(a) {
    // remove linear trend (least squares)
    var n = a.length, i, sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (i = 0; i < n; i++) { sx += i; sy += a[i]; sxx += i * i; sxy += i * a[i]; }
    var den = n * sxx - sx * sx;
    var slope = den === 0 ? 0 : (n * sxy - sx * sy) / den;
    var icpt = (sy - slope * sx) / n;
    var out = new Array(n);
    for (i = 0; i < n; i++) out[i] = a[i] - (slope * i + icpt);
    return out;
  }

  /* Resample irregularly-timed samples onto a uniform grid. Browser frame
     delivery jitters, and FFT-based estimation assumes uniform spacing, so
     skipping this quietly biases the heart-rate estimate. */
  function resampleUniform(times, channels, fs) {
    var t0 = times[0], t1 = times[times.length - 1];
    var n = Math.floor((t1 - t0) * fs);
    if (n < 2) return null;
    var out = [], c;
    for (c = 0; c < channels.length; c++) out.push(new Array(n));
    var idx = 0, i, k, tt, frac;
    for (i = 0; i < n; i++) {
      tt = t0 + i / fs;
      while (idx < times.length - 2 && times[idx + 1] < tt) idx++;
      var ta = times[idx], tb = times[idx + 1];
      frac = tb > ta ? (tt - ta) / (tb - ta) : 0;
      if (frac < 0) frac = 0; if (frac > 1) frac = 1;
      for (c = 0; c < channels.length; c++) {
        out[c][i] = channels[c][idx] + frac * (channels[c][idx + 1] - channels[c][idx]);
      }
    }
    return out;
  }

  // ------------------------------------------------------- POS algorithm ----
  function pos(R, G, B, fs) {
    var N = R.length;
    var l = Math.round(1.6 * fs);          // 1.6 s sliding window (Wang et al.)
    if (l < 8 || N < l) l = Math.max(8, Math.min(N, l));
    var H = new Array(N).fill(0);
    var n, i, mr, mg, mb, S1 = new Array(l), S2 = new Array(l), a, h, hm;

    for (n = 0; n + l <= N; n++) {
      mr = 0; mg = 0; mb = 0;
      for (i = 0; i < l; i++) { mr += R[n + i]; mg += G[n + i]; mb += B[n + i]; }
      mr /= l; mg /= l; mb /= l;
      if (mr === 0 || mg === 0 || mb === 0) continue;
      for (i = 0; i < l; i++) {
        var rn = R[n + i] / mr, gn = G[n + i] / mg, bn = B[n + i] / mb;
        S1[i] = gn - bn;                 // P row 1: [ 0, 1, -1]
        S2[i] = gn + bn - 2 * rn;        // P row 2: [-2, 1,  1]
      }
      var s2s = std(S2);
      a = s2s === 0 ? 0 : std(S1) / s2s;
      h = new Array(l);
      for (i = 0; i < l; i++) h[i] = S1[i] + a * S2[i];
      hm = mean(h);
      for (i = 0; i < l; i++) H[n + i] += (h[i] - hm);   // overlap-add
    }
    return H;
  }

  // ----------------------------------------------------- CHROM algorithm ----
  function chrom(R, G, B, fs) {
    var N = R.length;
    var mr = mean(R), mg = mean(G), mb = mean(B);
    if (mr === 0 || mg === 0 || mb === 0) return new Array(N).fill(0);
    var Xs = new Array(N), Ys = new Array(N), i;
    for (i = 0; i < N; i++) {
      var rn = R[i] / mr, gn = G[i] / mg, bn = B[i] / mb;
      Xs[i] = 3 * rn - 2 * gn;
      Ys[i] = 1.5 * rn + gn - 1.5 * bn;
    }
    var Xf = bandpass(Xs, fs, 0.7, 4.0);
    var Yf = bandpass(Ys, fs, 0.7, 4.0);
    var sy = std(Yf);
    var a = sy === 0 ? 0 : std(Xf) / sy;
    var S = new Array(N);
    for (i = 0; i < N; i++) S[i] = Xf[i] - a * Yf[i];
    return S;
  }

  // ------------------------------------------------------- zero-phase BP ----
  function bandpass(x, fs, lo, hi) {
    var N = x.length, M = nextPow2(N), k;
    var re = new Float64Array(M), im = new Float64Array(M);
    var xd = detrend(x);
    for (k = 0; k < N; k++) re[k] = xd[k];
    fft(re, im);
    for (k = 0; k < M; k++) {
      var f = Math.abs((k <= M / 2 ? k : k - M) * fs / M);
      if (f < lo || f > hi) { re[k] = 0; im[k] = 0; }
    }
    ifft(re, im);
    var out = new Array(N);
    for (k = 0; k < N; k++) out[k] = re[k];
    return out;
  }

  // ------------------------------------------------------------ spectrum ----
  function spectrum(x, fs) {
    var N = x.length, M = nextPow2(N * 4), i;   // zero-pad ×4 for finer bins
    var re = new Float64Array(M), im = new Float64Array(M);
    var xd = detrend(x);
    for (i = 0; i < N; i++) {
      var w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / Math.max(1, N - 1)); // Hann
      re[i] = xd[i] * w;
    }
    fft(re, im);
    var half = M >> 1, freqs = new Array(half), power = new Array(half);
    for (i = 0; i < half; i++) {
      freqs[i] = i * fs / M;
      power[i] = re[i] * re[i] + im[i] * im[i];
    }
    return { freqs: freqs, power: power };
  }

  var HR_LO = 0.7, HR_HI = 4.0;   // 42–240 bpm

  function estimateHR(sig, fs) {
    var sp = spectrum(sig, fs);
    var f = sp.freqs, p = sp.power;
    var best = -1, bi = -1, i;
    for (i = 0; i < f.length; i++) {
      if (f[i] >= HR_LO && f[i] <= HR_HI && p[i] > best) { best = p[i]; bi = i; }
    }
    if (bi <= 0 || bi >= f.length - 1) return { hr: null, snr: -99, freq: null, spectrum: sp };

    // parabolic interpolation for sub-bin frequency accuracy
    var y0 = p[bi - 1], y1 = p[bi], y2 = p[bi + 1];
    var den = (y0 - 2 * y1 + y2);
    var delta = den === 0 ? 0 : 0.5 * (y0 - y2) / den;
    if (!isFinite(delta) || Math.abs(delta) > 1) delta = 0;
    var df = f[1] - f[0];
    var fpk = f[bi] + delta * df;

    /* SNR after de Haan & Jeanne: power within ±0.1 Hz of the fundamental and
       ±0.2 Hz of the first harmonic, versus everything else in the HR band. */
    var sigP = 0, totP = 0;
    for (i = 0; i < f.length; i++) {
      if (f[i] < HR_LO || f[i] > HR_HI) continue;
      totP += p[i];
      if (Math.abs(f[i] - fpk) <= 0.1 || Math.abs(f[i] - 2 * fpk) <= 0.2) sigP += p[i];
    }
    var noiseP = Math.max(1e-12, totP - sigP);
    var snr = 10 * Math.log10(Math.max(1e-12, sigP) / noiseP);
    return { hr: fpk * 60, snr: snr, freq: fpk, spectrum: sp };
  }

  // -------------------------------------------------------------- peaks ----
  function findPeaks(sig, fs, hrHz) {
    var minDist = Math.max(3, Math.round(0.6 * (1 / hrHz) * fs));
    var peaks = [], i;
    for (i = 1; i < sig.length - 1; i++) {
      if (sig[i] > sig[i - 1] && sig[i] >= sig[i + 1] && sig[i] > 0) {
        if (peaks.length && i - peaks[peaks.length - 1] < minDist) {
          if (sig[i] > sig[peaks[peaks.length - 1]]) peaks[peaks.length - 1] = i;
        } else peaks.push(i);
      }
    }
    return peaks;
  }

  function hrvMetrics(peaks, fs) {
    if (peaks.length < 4) return null;
    var ibi = [], i;
    for (i = 1; i < peaks.length; i++) ibi.push((peaks[i] - peaks[i - 1]) / fs * 1000);
    // drop physiologically implausible intervals (250–2000 ms)
    ibi = ibi.filter(function (v) { return v > 250 && v < 2000; });
    if (ibi.length < 3) return null;
    var sdnn = std(ibi);
    var d = [], rmssd;
    for (i = 1; i < ibi.length; i++) d.push((ibi[i] - ibi[i - 1]) * (ibi[i] - ibi[i - 1]));
    rmssd = Math.sqrt(mean(d));
    return { sdnn: sdnn, rmssd: rmssd, meanIBI: mean(ibi), nBeats: ibi.length + 1, ibi: ibi };
  }

  // ------------------------------------------------------ channel picker ----
  /* Finger-on-lens contact PPG: POS assumes ambient-light skin reflection and
     misbehaves here, so choose whichever channel carries the cleanest pulse. */
  function bestChannel(R, G, B, fs) {
    var cands = [{ n: "green", x: G }, { n: "red", x: R }, { n: "blue", x: B }];
    var best = null, i;
    for (i = 0; i < cands.length; i++) {
      var filt = bandpass(cands[i].x, fs, HR_LO, HR_HI);
      var est = estimateHR(filt, fs);
      if (!best || est.snr > best.est.snr) best = { name: cands[i].n, sig: filt, est: est };
    }
    return best;
  }

  /* --------------------------------------------------- consistency check ----
     SNR alone is NOT a sufficient quality gate. When the estimator locks onto
     a motion artifact, that artifact is itself a clean spectral peak, so SNR
     looks excellent while the reported rate is simply wrong. A true pulse is
     stable over time; motion artifacts wander. Estimating HR independently in
     overlapping sub-windows and measuring the spread catches exactly the
     failure SNR misses. */
  function temporalConsistency(sig, fs) {
    var winSec = 8, stepSec = 2;
    var w = Math.round(winSec * fs), st = Math.round(stepSec * fs);
    if (sig.length < w + st) return { spread: null, hrs: [], median: null };
    var hrs = [], i;
    for (i = 0; i + w <= sig.length; i += st) {
      var seg = sig.slice(i, i + w);
      var e = estimateHR(seg, fs);
      if (e.hr !== null) hrs.push(e.hr);
    }
    if (hrs.length < 2) return { spread: null, hrs: hrs, median: null };
    var sorted = hrs.slice().sort(function (a, b) { return a - b; });
    var median = sorted[Math.floor(sorted.length / 2)];
    // median absolute deviation — robust to a single bad window
    var devs = hrs.map(function (h) { return Math.abs(h - median); }).sort(function (a, b) { return a - b; });
    var mad = devs[Math.floor(devs.length / 2)];
    return { spread: mad, hrs: hrs, median: median, max: sorted[sorted.length - 1], min: sorted[0] };
  }

  // ---------------------------------------------------------- main entry ----
  /**
   * @param samples {times:[s], r:[], g:[], b:[]}
   * @param opts    {mode:"finger"|"face", fs:Number}
   */
  function analyse(samples, opts) {
    opts = opts || {};
    var mode = opts.mode || "face";
    var fs = opts.fs || 30;

    if (!samples.times || samples.times.length < fs * 4) {
      return { ok: false, reason: "Not enough data — need at least a few seconds of steady capture." };
    }

    var rs = resampleUniform(samples.times, [samples.r, samples.g, samples.b], fs);
    if (!rs) return { ok: false, reason: "Could not resample capture — frame timing was unusable." };
    var R = rs[0], G = rs[1], B = rs[2];

    var sig, est, method, agreement = null;
    if (mode === "finger") {
      var bc = bestChannel(R, G, B, fs);
      sig = bc.sig; est = bc.est; method = "contact PPG (" + bc.name + " channel)";
    } else {
      var hPos = bandpass(pos(R, G, B, fs), fs, HR_LO, HR_HI);
      var ePos = estimateHR(hPos, fs);
      var hChr = bandpass(chrom(R, G, B, fs), fs, HR_LO, HR_HI);
      var eChr = estimateHR(hChr, fs);
      /* Cross-algorithm agreement: POS and CHROM make different assumptions,
         so a genuine pulse shows up in both. A motion artifact often does not. */
      if (ePos.hr !== null && eChr.hr !== null) agreement = Math.abs(ePos.hr - eChr.hr);
      if (ePos.snr >= eChr.snr) { sig = hPos; est = ePos; method = "POS"; }
      else { sig = hChr; est = eChr; method = "CHROM"; }
    }

    if (est.hr === null) return { ok: false, reason: "No pulse-band signal found." };

    var consist = temporalConsistency(sig, fs);

    /* ---- signal quality ----
       Three independent gates, because any one alone can be fooled:
         SNR         — is there a clean peak at all?
         consistency — is it STABLE (a pulse) or wandering (an artifact)?
         agreement   — do two different algorithms see the same thing?
       Consistency is weighted heavily: it is the only one that reliably
       catches a confident lock onto a motion artifact. */
    var quality, qLabel, qNote;
    var unstable = consist.spread !== null && consist.spread > 6;
    var veryUnstable = consist.spread !== null && consist.spread > 12;
    var disagree = agreement !== null && agreement > 8;

    if (est.snr >= 3 && !unstable && !disagree) {
      quality = "good"; qLabel = "Good signal";
      qNote = "Clear, stable pulse-band peak.";
    } else if (est.snr >= -1 && !veryUnstable && !(unstable && disagree)) {
      quality = "fair"; qLabel = "Fair signal";
      qNote = "Pulse detected but noisy — treat the value as approximate.";
    } else {
      quality = "poor"; qLabel = "Poor signal — not reliable";
      qNote = veryUnstable
        ? "The detected rate drifted between measurement windows, which usually means motion rather than a pulse was tracked. Hold still, improve lighting, and try again."
        : "Too noisy to trust. Hold still, improve lighting, and try again.";
    }

    var hrv = null;
    if (quality !== "poor") hrv = hrvMetrics(findPeaks(sig, fs, est.freq), fs);

    // brightness stability — a proxy for motion/lighting disturbance
    var lum = new Array(G.length), i;
    for (i = 0; i < G.length; i++) lum[i] = 0.299 * R[i] + 0.587 * G[i] + 0.114 * B[i];
    var lumMean = mean(lum);
    var stability = lumMean > 0 ? (std(detrend(lum)) / lumMean) * 100 : 0;

    return {
      ok: true,
      hr: est.hr,
      snr: est.snr,
      quality: quality,
      qualityLabel: qLabel,
      qualityNote: qNote,
      method: method,
      hrv: hrv,
      consistencySpread: consist.spread,
      windowHRs: consist.hrs,
      crossMethodDelta: agreement,
      waveform: sig,
      fs: fs,
      spectrum: est.spectrum,
      durationSec: samples.times[samples.times.length - 1] - samples.times[0],
      meanLuma: lumMean,
      stabilityPct: stability
    };
  }

  // ----------------------------------------- synthetic self-test signal ----
  /* Used by the built-in self-test and by the Node validation suite.
     melanin: 0 (light) → 1 (very dark); attenuates both DC reflectance and,
     more importantly, the AC pulse amplitude, which is the documented cause of
     rPPG accuracy loss on darker skin. */
  function synth(opts) {
    opts = opts || {};
    var bpm = opts.bpm || 72, dur = opts.dur || 20, fs = opts.fs || 30;
    var melanin = opts.melanin === undefined ? 0 : opts.melanin;
    var noise = opts.noise === undefined ? 0.15 : opts.noise;
    var seed = opts.seed === undefined ? 1 : opts.seed;
    var n = Math.round(dur * fs), f = bpm / 60;
    var times = [], r = [], g = [], b = [], i;

    var s = seed;
    function rnd() { s = (s * 1103515245 + 12345) & 0x7fffffff; return (s / 0x7fffffff) * 2 - 1; }

    var atten = Math.pow(1 - 0.75 * melanin, 1.5);   // AC amplitude loss
    var dc = 150 * (1 - 0.72 * melanin);             // baseline reflectance
    // hemoglobin absorbs green most, then blue, then red
    var aG = 1.6 * atten, aB = 0.9 * atten, aR = 0.5 * atten;

    /* Motion artifact — in practice the dominant real-world noise source, and
       critically it is NOT attenuated by melanin, so as the pulse shrinks the
       artifact-to-signal ratio grows. This is why accuracy degrades on darker
       skin in the field far more than sensor noise alone would predict. */
    var motion = opts.motion === undefined ? 0 : opts.motion;
    var walk = 0;
    /* In-band interferer: quasi-periodic motion (head sway, hand tremor,
       rocking) landing inside 0.7–4 Hz. This is the artifact that actually
       causes gross HR errors, because the estimator can lock onto it instead
       of the pulse once the pulse is attenuated. */
    var ibFreq = opts.interfFreq === undefined ? 1.45 : opts.interfFreq;
    var ibAmp = opts.interf === undefined ? 0 : opts.interf;

    for (i = 0; i < n; i++) {
      var t = i / fs;
      // pulse waveform: fundamental + dicrotic-ish harmonic
      var p = Math.sin(2 * Math.PI * f * t) + 0.35 * Math.sin(4 * Math.PI * f * t + 0.6);
      var drift = 2.0 * Math.sin(2 * Math.PI * 0.06 * t);      // slow illumination drift
      var resp = 0.6 * Math.sin(2 * Math.PI * 0.25 * t);       // breathing
      walk = 0.92 * walk + motion * rnd();                     // correlated motion
      var jolt = (Math.abs(rnd()) > 0.985) ? motion * 4 * rnd() : 0;  // occasional lurch
      // wandering-phase in-band interferer (not perfectly periodic, like real motion)
      /* Real motion wanders in frequency (sway, shifting posture, tremor),
         unlike a metronome. `interfWander` sets how much, in Hz. */
      var wander = opts.interfWander === undefined ? 0 : opts.interfWander;
      var instFreq = ibFreq + wander * Math.sin(2 * Math.PI * 0.045 * t + 0.7);
      var ib = ibAmp * Math.sin(2 * Math.PI * instFreq * t + 1.3 * Math.sin(2 * Math.PI * 0.11 * t));
      var mo = walk + jolt;
      /* Motion is only partly achromatic: changing skin-to-light geometry and
         ROI content also shifts colour balance, so a fraction of the artifact
         carries a pulse-like chromatic signature that POS cannot project out. */
      var ibChrom = opts.interfChrom === undefined ? 0.45 : opts.interfChrom;
      times.push(t);
      r.push(dc * 1.00 + aR * p + drift + resp * 0.4 + mo * 1.00 + ib * (1 - ibChrom + ibChrom * 0.31) + noise * rnd());
      g.push(dc * 0.78 + aG * p + drift + resp * 0.5 + mo * 0.95 + ib * (1 - ibChrom + ibChrom * 1.00) + noise * rnd());
      b.push(dc * 0.65 + aB * p + drift + resp * 0.3 + mo * 0.90 + ib * (1 - ibChrom + ibChrom * 0.56) + noise * rnd());
    }
    return { times: times, r: r, g: g, b: b };
  }

  var API = {
    analyse: analyse, synth: synth, pos: pos, chrom: chrom, bandpass: bandpass,
    estimateHR: estimateHR, spectrum: spectrum, hrvMetrics: hrvMetrics,
    findPeaks: findPeaks, resampleUniform: resampleUniform, fft: fft, ifft: ifft
  };

  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.RPPG = API;
})(typeof window !== "undefined" ? window : this);
