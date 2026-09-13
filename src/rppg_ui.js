/* ============================================================================
   BrainShield — camera capture UI for rPPG
   ========================================================================== */
(function () {
  "use strict";

  var CAPTURE_SEC = 25;
  var mode = "finger";
  var stream = null, track = null, rafId = null, running = false;
  var samples = { times: [], r: [], g: [], b: [] };
  var t0 = 0, liveBuf = [];

  var video = document.getElementById("cam");
  var overlay = document.getElementById("camOverlay");
    var waveCv = document.getElementById("wave");
  var waveCtx = waveCv.getContext("2d");
  var startBtn = document.getElementById("startPulse");
  var stopBtn = document.getElementById("stopPulse");
  var selfBtn = document.getElementById("selfTest");
  var elapsedEl = document.getElementById("elapsed");
  var frameEl = document.getElementById("frameCount");
  var lumaEl = document.getElementById("lumaVal");
  var warnEl = document.getElementById("liveWarn");
  var modeHelp = document.getElementById("modeHelp");

  var work = document.createElement("canvas");
  work.width = 64; work.height = 48;
  var wctx = work.getContext("2d", { willReadFrequently: true });

  var HELP = {
    finger: "Cover the <b>rear camera lens and flash</b> completely with your fingertip — light pressure, don't press hard. Keep your hand still for " + CAPTURE_SEC + " seconds. On phones the flash turns on automatically; on a laptop, use a bright lamp instead."
  };

  // Fingertip-with-flash is the only mode; the face-mode selector was removed.

  function clearWave() {
    waveCtx.fillStyle = "#0E1420";
    waveCtx.fillRect(0, 0, waveCv.width, waveCv.height);
  }
  clearWave();

  function drawLive() {
    var w = waveCv.width, h = waveCv.height;
    waveCtx.fillStyle = "#0E1420";
    waveCtx.fillRect(0, 0, w, h);
    if (liveBuf.length < 4) return;
    var n = Math.min(liveBuf.length, 220);
    var seg = liveBuf.slice(liveBuf.length - n);
    var mn = Math.min.apply(null, seg), mx = Math.max.apply(null, seg);
    var rng = (mx - mn) || 1;
    waveCtx.strokeStyle = "#6FD6C5";
    waveCtx.lineWidth = 2;
    waveCtx.beginPath();
    for (var i = 0; i < n; i++) {
      var x = (i / (n - 1)) * w;
      var y = h - 12 - ((seg[i] - mn) / rng) * (h - 24);
      if (i === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
    }
    waveCtx.stroke();
  }

  /* Live trace: subtract a moving mean so the tiny AC pulse is visible
     instead of being flattened by the much larger DC brightness. */
  function pushLive(g) {
    liveBuf.push(g);
    if (liveBuf.length > 600) liveBuf.shift();
  }
  function acTrace() {
    var out = [], win = 45, i, j, s, c;
    for (i = 0; i < liveBuf.length; i++) {
      s = 0; c = 0;
      for (j = Math.max(0, i - win); j <= Math.min(liveBuf.length - 1, i + win); j++) { s += liveBuf[j]; c++; }
      out.push(liveBuf[i] - s / c);
    }
    return out;
  }

  function sampleFrame(ts) {
    var vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    var s = Math.min(vw, vh) * 0.7;
    wctx.drawImage(video, (vw - s) / 2, (vh - s) / 2, s, s, 0, 0, work.width, work.height);
    var d = wctx.getImageData(0, 0, work.width, work.height).data;
    var r = 0, g = 0, b = 0, n = work.width * work.height, i;
    for (i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; b += d[i + 2]; }
    r /= n; g /= n; b /= n;

    var t = (ts - t0) / 1000;
    samples.times.push(t); samples.r.push(r); samples.g.push(g); samples.b.push(b);
    pushLive(g);

    var luma = 0.299 * r + 0.587 * g + 0.114 * b;
    elapsedEl.textContent = t.toFixed(1) + "s";
    frameEl.textContent = String(samples.times.length);
    lumaEl.textContent = luma.toFixed(0);

    var w = "";
    if (r > 250) w = "Camera is saturating — ease off the pressure or block the flash slightly.";
    else if (luma < 22) w = "Too dark — the pulse signal will be very weak. Add more light.";
    else if (mode === "finger" && luma > 30 && r < 90) w = "Make sure your fingertip fully covers the lens.";
    warnEl.textContent = w;

    if (samples.times.length % 3 === 0) { var tr = acTrace(); liveBufDraw(tr); }
    if (t >= CAPTURE_SEC) { finish(); return; }
  }

  function liveBufDraw(tr) {
    var w = waveCv.width, h = waveCv.height;
    waveCtx.fillStyle = "#0E1420";
    waveCtx.fillRect(0, 0, w, h);
    var n = Math.min(tr.length, 220);
    if (n < 4) return;
    var seg = tr.slice(tr.length - n);
    var mn = Math.min.apply(null, seg), mx = Math.max.apply(null, seg);
    var rng = (mx - mn) || 1;
    waveCtx.strokeStyle = "#6FD6C5";
    waveCtx.lineWidth = 2;
    waveCtx.beginPath();
    for (var i = 0; i < n; i++) {
      var x = (i / (n - 1)) * w;
      var y = h - 12 - ((seg[i] - mn) / rng) * (h - 24);
      if (i === 0) waveCtx.moveTo(x, y); else waveCtx.lineTo(x, y);
    }
    waveCtx.stroke();
    waveCtx.fillStyle = "#5B6472";
    waveCtx.font = "11px Calibri, sans-serif";
    waveCtx.fillText("live signal (pulse component)", 10, 16);
  }

  function loop() {
    if (!running) return;
    sampleFrame(performance.now());
    rafId = requestAnimationFrame(loop);
  }

  function useVFC() {
    if (!running) return;
    video.requestVideoFrameCallback(function (now, meta) {
      if (!running) return;
      sampleFrame(now);
      if (running) useVFC();
    });
  }

  async function start() {
    samples = { times: [], r: [], g: [], b: [] };
    liveBuf = [];
    document.getElementById("pulseResults").style.display = "none";
    warnEl.textContent = "";
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: mode === "finger" ? { ideal: "environment" } : { ideal: "user" },
          width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 }
        }, audio: false
      });
    } catch (e) {
      overlay.style.display = "flex";
      overlay.textContent = "Camera unavailable: " + (e && e.name ? e.name : "error");
      warnEl.textContent = "Could not access the camera. You can still run the self-test below.";
      return;
    }
    video.srcObject = stream;
    await video.play().catch(function () {});
    overlay.style.display = "none";

    track = stream.getVideoTracks()[0];
    if (mode === "finger" && track && track.getCapabilities) {
      try {
        var caps = track.getCapabilities();
        if (caps && caps.torch) await track.applyConstraints({ advanced: [{ torch: true }] });
        else warnEl.textContent = "Flash not available on this device — use a bright lamp against your fingertip.";
      } catch (e) { /* torch unsupported; continue */ }
    }

    running = true;
    t0 = performance.now();
    startBtn.style.display = "none";
    stopBtn.style.display = "inline-block";
    if (video.requestVideoFrameCallback) useVFC(); else loop();
  }

  function stopCamera() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
    if (track && track.applyConstraints) {
      // fire-and-forget: needs its own .catch, an async rejection escapes try/catch
      try { var pr = track.applyConstraints({ advanced: [{ torch: false }] }); if (pr && pr.catch) pr.catch(function () {}); }
      catch (e) { /* torch unsupported on this device */ }
    }
    if (stream) stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null; track = null;
    video.srcObject = null;
    overlay.style.display = "flex";
    overlay.textContent = "Camera is off";
    startBtn.style.display = "inline-block";
    stopBtn.style.display = "none";
  }

  function finish() {
    var s = samples;
    stopCamera();
    var res = window.RPPG.analyse(s, { mode: mode, fs: 30 });
    render(res, false);
  }

  function render(res, isSelfTest) {
    var box = document.getElementById("pulseResults");
    box.style.display = "block";
    if (!res.ok) {
      document.getElementById("bpmVal").textContent = "—";
      document.getElementById("qualBadge").textContent = "Measurement failed";
      document.getElementById("qualBadge").style.background = "#FBE7E5";
      document.getElementById("qualBadge").style.color = "#96271C";
      document.getElementById("qualNote").textContent = res.reason;
      box.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    document.getElementById("bpmVal").textContent = res.hr.toFixed(0);
    var badge = document.getElementById("qualBadge");
    badge.textContent = (isSelfTest ? "SELF-TEST — " : "") + res.qualityLabel;
    if (res.quality === "good") { badge.style.background = "#E4F3EE"; badge.style.color = "#1E7B5C"; }
    else if (res.quality === "fair") { badge.style.background = "#FDF2DC"; badge.style.color = "#8A5A00"; }
    else { badge.style.background = "#FBE7E5"; badge.style.color = "#96271C"; }
    document.getElementById("qualNote").textContent = res.qualityNote;

    document.getElementById("snrVal").textContent = res.snr.toFixed(1) + " dB";
    document.getElementById("stabVal").textContent =
      res.consistencySpread === null ? "n/a" : "±" + res.consistencySpread.toFixed(1) + " bpm";
    document.getElementById("methodVal").textContent = res.method;
    document.getElementById("beatsVal").textContent = res.hrv ? res.hrv.nBeats : "—";
    document.getElementById("sdnnVal").textContent = res.hrv ? res.hrv.sdnn.toFixed(0) + " ms" : "—";
    document.getElementById("rmssdVal").textContent = res.hrv ? res.hrv.rmssd.toFixed(0) + " ms" : "—";

    drawResultWave(res);
    box.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function drawResultWave(res) {
    var cv = document.getElementById("resultWave");
    var ctx = cv.getContext("2d");
    var w = cv.width, h = cv.height;
    ctx.fillStyle = "#0E1420";
    ctx.fillRect(0, 0, w, h);
    var sig = res.waveform;
    if (!sig || sig.length < 4) return;
    var show = Math.min(sig.length, res.fs * 10);
    var seg = sig.slice(0, show);
    var mn = Math.min.apply(null, seg), mx = Math.max.apply(null, seg);
    var rng = (mx - mn) || 1;
    ctx.strokeStyle = res.quality === "poor" ? "#B85042" : "#6FD6C5";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < show; i++) {
      var x = (i / (show - 1)) * w;
      var y = h - 14 - ((seg[i] - mn) / rng) * (h - 28);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "#8FA6B8";
    ctx.font = "11px Calibri, sans-serif";
    ctx.fillText("recovered pulse waveform — first 10 seconds", 10, 17);
  }

  startBtn.addEventListener("click", start);
  stopBtn.addEventListener("click", function () { stopCamera(); });

  selfBtn.addEventListener("click", function () {
    var bpm = 60 + Math.floor(Math.random() * 45);
    var syn = window.RPPG.synth({ bpm: bpm, dur: 20, fs: 30, melanin: 0.6, noise: 0.4, motion: 0.5, interf: 0.6, interfWander: 0.2 });
    var res = window.RPPG.analyse(syn, { mode: "finger", fs: 30 });
    render(res, true);
    document.getElementById("qualNote").textContent =
      "Synthetic signal generated at " + bpm + " bpm; the pipeline recovered " +
      (res.ok ? res.hr.toFixed(1) : "n/a") + " bpm. This verifies the processing code — it is not a measurement of you.";
  });
})();
