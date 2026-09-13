/*
 * Camera capture for the rhythm check.
 *
 * Finger over the rear camera with the torch on. This is CONTACT photoplethysmography
 * with active illumination -- a different and far more robust measurement than
 * face rPPG in ambient light, which is why it is the only camera mode offered here.
 *
 * Reuses the tested DSP in rppg_core.js for filtering and quality gating, then layers
 * the rhythm analysis from af_core.js on top.
 */
(function (root) {
  "use strict";

  // The AF classifier was trained on 60-second windows. Shorter captures give too few
  // beats for the interval statistics to be stable, so this is a floor, not a preference.
  var TARGET_SECONDS = 60;
  var MIN_SECONDS = 45;

  function AFCapture(videoEl, canvasEl) {
    this.video = videoEl;
    this.canvas = canvasEl || document.createElement("canvas");
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.stream = null;
    this.track = null;
    this.samples = { times: [], r: [], g: [], b: [] };
    this.running = false;
    this.t0 = 0;
    this.onTick = null;
    this.torchOn = false;
  }

  AFCapture.prototype.start = async function () {
    this.samples = { times: [], r: [], g: [], b: [] };
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 640 }, height: { ideal: 480 },
        frameRate: { ideal: 60, min: 15 }     // 60 if the device allows; 30 is fine
      },
      audio: false
    });
    this.video.srcObject = this.stream;
    this.video.setAttribute("playsinline", "");
    await this.video.play().catch(function () {});

    this.track = this.stream.getVideoTracks()[0];
    // Torch is what makes this a contact measurement rather than a hopeful one.
    try {
      var caps = this.track.getCapabilities ? this.track.getCapabilities() : {};
      if (caps && caps.torch) {
        await this.track.applyConstraints({ advanced: [{ torch: true }] });
        this.torchOn = true;
      }
    } catch (e) { /* torch unsupported; capture still works, just noisier */ }

    this.canvas.width = 64;
    this.canvas.height = 48;
    this.running = true;
    this.t0 = performance.now();
    this._loop();
    return { torch: this.torchOn };
  };

  AFCapture.prototype._grab = function (nowMs) {
    if (!this.video.videoWidth) return;
    this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    var d;
    try {
      d = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height).data;
    } catch (e) { return; }
    var r = 0, g = 0, b = 0, n = 0;
    // central region only: the edges of a fingertip press are the least stable
    var W = this.canvas.width, H = this.canvas.height;
    for (var y = (H >> 2); y < H - (H >> 2); y++) {
      for (var x = (W >> 2); x < W - (W >> 2); x++) {
        var i = (y * W + x) * 4;
        r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
      }
    }
    if (!n) return;
    this.samples.times.push((nowMs - this.t0) / 1000);
    this.samples.r.push(r / n);
    this.samples.g.push(g / n);
    this.samples.b.push(b / n);
  };

  AFCapture.prototype._loop = function () {
    var self = this;
    function step(now) {
      if (!self.running) return;
      self._grab(now);
      var el = self.samples.times.length ? self.samples.times[self.samples.times.length - 1] : 0;
      if (self.onTick) self.onTick(el, self.samples);
      if (el >= TARGET_SECONDS) { self.stop(); return; }
      requestAnimationFrame(step);
    }
    // requestVideoFrameCallback gives true frame timing where available
    if (this.video.requestVideoFrameCallback) {
      var vstep = function (now) {
        if (!self.running) return;
        self._grab(now);
        var el = self.samples.times.length ? self.samples.times[self.samples.times.length - 1] : 0;
        if (self.onTick) self.onTick(el, self.samples);
        if (el >= TARGET_SECONDS) { self.stop(); return; }
        self.video.requestVideoFrameCallback(vstep);
      };
      this.video.requestVideoFrameCallback(vstep);
    } else {
      requestAnimationFrame(step);
    }
  };

  AFCapture.prototype.stop = function () {
    this.running = false;
    if (this.track && this.torchOn) {
      try {
        var p = this.track.applyConstraints({ advanced: [{ torch: false }] });
        if (p && p.catch) p.catch(function () {});
      } catch (e) { /* ignore */ }
    }
    if (this.stream) this.stream.getTracks().forEach(function (t) { t.stop(); });
    this.stream = null; this.track = null;
    if (this.video) this.video.srcObject = null;
  };

  /** Estimate the true capture rate from frame timestamps, rather than assuming 30. */
  function measuredFps(times) {
    if (times.length < 10) return 30;
    var d = [], i;
    for (i = 1; i < times.length; i++) d.push(times[i] - times[i - 1]);
    d.sort(function (a, b) { return a - b; });
    var med = d[d.length >> 1];
    return (med > 0 && isFinite(med)) ? Math.min(120, Math.max(10, 1 / med)) : 30;
  }

  /* ---- quality gating for rhythm analysis --------------------------------
   * This deliberately does NOT reuse rppg_core's headline quality verdict.
   *
   * That verdict leans hard on temporal consistency -- whether the estimated heart
   * rate stays stable across sliding windows -- because a wandering rate usually
   * means the estimator locked onto a motion artifact. That is exactly right when
   * the goal is a heart rate.
   *
   * It is exactly wrong here. In atrial fibrillation the rate genuinely wanders
   * between windows, so that gate rejects the very signal we are trying to detect.
   * Testing confirmed it: a clean synthetic AF waveform was thrown out as "poor".
   *
   * Spectral SNR fails here for the same underlying reason. SNR measures how sharp the
   * dominant frequency peak is, and atrial fibrillation has no sharp peak BY DEFINITION:
   * irregular intervals spread the pulse energy across the band. A clean synthetic AF
   * waveform measured SNR -2.6 while a regular one passed easily. Peak sharpness is a
   * rhythm measurement wearing a quality measurement's clothes.
   *
   * So the gates here are all either time-domain or band-aggregate, never peak-based:
   *   - in-band power fraction: is there a pulse at all, at ANY rate within the band?
   *   - brightness stability: did the finger shift or the light change?
   *   - beat amplitude consistency: real beats look alike; artifacts do not.
   *   - implausible-interval rate: failed beat detection, as distinct from irregularity.
   * -------------------------------------------------------------------- */
  function inBandFraction(raw, waveform) {
    // share of the signal's variance that sits in the pulse band. Unlike SNR this does
    // not care whether that energy is at one frequency or smeared across the band.
    function detrendVar(a) {
      var n = a.length, i, m = 0;
      for (i = 0; i < n; i++) m += a[i];
      m /= n;
      // remove linear drift as well as the mean
      var sx = 0, sxy = 0, mx = (n - 1) / 2;
      for (i = 0; i < n; i++) { sx += (i - mx) * (i - mx); sxy += (i - mx) * (a[i] - m); }
      var slope = sx > 0 ? sxy / sx : 0, v = 0, d;
      for (i = 0; i < n; i++) { d = a[i] - m - slope * (i - mx); v += d * d; }
      return v / n;
    }
    var vTotal = detrendVar(raw);
    if (!(vTotal > 0)) return 0;
    var vBand = detrendVar(waveform);
    return Math.min(1, vBand / vTotal);
  }

  function afQuality(res, samples) {
    var frac = inBandFraction(samples.r, res.waveform);
    if (frac < 0.02) {
      return { ok: false, reason: "No pulse was detectable in the recording — almost all "
             + "of the variation was drift or noise rather than a heartbeat." };
    }
    if (res.stabilityPct > 8) {
      return { ok: false, reason: "The image brightness moved around too much, which "
             + "usually means the finger shifted or the lighting changed." };
    }

    var RPPG = root.RPPG, AF = root.BrainShieldAF;
    var peaks = AF.refinePeaks(res.waveform, RPPG.findPeaks(res.waveform, res.fs, (res.hr || 70) / 60));
    if (peaks.length < 12) {
      return { ok: false, reason: "Too few heartbeats were detected in the recording." };
    }

    // Beat amplitudes should be similar. Motion produces wildly varying peak heights;
    // atrial fibrillation does not -- it varies the SPACING, not the shape.
    var amps = [], i, idx;
    for (i = 0; i < peaks.length; i++) {
      idx = Math.round(peaks[i]);
      if (idx >= 0 && idx < res.waveform.length) amps.push(Math.abs(res.waveform[idx]));
    }
    var m = 0;
    for (i = 0; i < amps.length; i++) m += amps[i];
    m /= amps.length;
    var v = 0;
    for (i = 0; i < amps.length; i++) v += (amps[i] - m) * (amps[i] - m);
    var ampCV = m > 0 ? Math.sqrt(v / amps.length) / m : 99;
    if (ampCV > 0.75) {
      return { ok: false, reason: "The pulse waveform was too irregular in shape to trust, "
             + "which points to movement rather than a steady reading." };
    }

    // Intervals outside 250-2500 ms mean beat detection failed, which is different
    // from a genuinely irregular rhythm within the plausible range.
    var all = [];
    for (i = 1; i < peaks.length; i++) all.push((peaks[i] - peaks[i - 1]) / res.fs * 1000);
    var bad = 0;
    for (i = 0; i < all.length; i++) if (all[i] <= 250 || all[i] >= 2500) bad++;
    if (all.length && bad / all.length > 0.2) {
      return { ok: false, reason: "Many beats could not be timed reliably." };
    }
    return { ok: true, amp_cv: ampCV, n_peaks: peaks.length, in_band_fraction: frac };
  }

  /**
   * Turn a finished capture into a rhythm decision.
   * Returns the same three-outcome shape as af_core.decide().
   */
  function analyseCapture(samples, afModel) {
    var RPPG = root.RPPG, AF = root.BrainShieldAF;
    var dur = samples.times.length ? samples.times[samples.times.length - 1] : 0;

    if (dur < MIN_SECONDS) {
      return {
        outcome: "unusable", sets_af: null, probability: null,
        headline: "Recording too short",
        detail: "The rhythm check needs at least " + MIN_SECONDS + " seconds of steady "
              + "contact to gather enough heartbeats. Please try again and keep your "
              + "finger still on the lens."
      };
    }

    var fps = measuredFps(samples.times);
    var res = RPPG.analyse(samples, { mode: "finger", fs: Math.round(fps) });

    if (!res.ok) {
      return { outcome: "unusable", sets_af: null, probability: null,
               headline: "Couldn't read your pulse", detail: res.reason };
    }

    var q = afQuality(res, samples);
    if (!q.ok) {
      return {
        outcome: "unusable", sets_af: null, probability: null,
        headline: "Signal wasn't clear enough",
        detail: q.reason + " Try again: cover the lens and flash completely with the pad of "
              + "your finger, press gently and evenly, and rest your hand on a table.",
        hr: res.hr, quality: res.quality
      };
    }

    var peaksRaw = RPPG.findPeaks(res.waveform, res.fs, (res.hr || 70) / 60);
    var peaks = AF.refinePeaks(res.waveform, peaksRaw);
    var rr = AF.rrFromPeaks(peaks, res.fs);

    if (rr.length < afModel.min_beats - 1) {
      return {
        outcome: "unusable", sets_af: null, probability: null,
        headline: "Not enough clean heartbeats",
        detail: "Found " + rr.length + " usable intervals; the rhythm check needs at least "
              + (afModel.min_beats - 1) + ". Try a longer, steadier recording.",
        hr: res.hr
      };
    }

    var feats = AF.afFeatures(rr);
    var decision = AF.decide(feats, afModel, { ok: true });
    decision.hr = res.hr;
    decision.quality = res.quality;
    decision.n_intervals = rr.length;
    decision.fps = fps;
    decision.duration = dur;
    decision.features = feats;
    return decision;
  }

  var API = {
    AFCapture: AFCapture, analyseCapture: analyseCapture, measuredFps: measuredFps,
    afQuality: afQuality, inBandFraction: inBandFraction,
    TARGET_SECONDS: TARGET_SECONDS, MIN_SECONDS: MIN_SECONDS
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.BrainShieldCapture = API;
})(typeof globalThis !== "undefined" ? globalThis : this);
