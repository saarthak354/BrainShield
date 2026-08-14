#!/usr/bin/env python3
"""
End-to-end test of the BROWSER capture path.

Chromium's --use-file-for-fake-video-capture device is unavailable in this
container, so instead we substitute a real MediaStream produced by
canvas.captureStream(). The pulse is drawn into the canvas pixels with
per-pixel dithering noise, exactly as a sensor would deliver it.

This genuinely exercises: <video> playback -> drawImage -> getImageData ->
per-frame RGB averaging -> irregular browser frame timing -> resampling ->
POS/CHROM -> quality gating -> rendered result.

It does NOT test getUserMedia device negotiation or torch control (no hardware
here), and it is NOT a validation on real human skin.
"""
import sys, os
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "http://127.0.0.1:8899/index.html"

INJECT = r"""
(cfg) => {
  const W = 160, H = 120;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(W, H);
  const f = cfg.bpm / 60;

  let base, amp;
  if (cfg.mode === 'finger') { base = [205, 62, 34]; amp = [7, 10, 4]; }
  else {
    const at = Math.pow(1 - 0.75 * cfg.melanin, 1.5);
    const dc = 1 - 0.72 * cfg.melanin;
    base = [168 * dc, 128 * dc, 112 * dc];
    amp  = [0.55 * at, 1.75 * at, 1.00 * at];
  }

  let seed = 12345;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff) * 2 - 1; }

  const t0 = performance.now();
  function draw() {
    const t = (performance.now() - t0) / 1000;
    const p = Math.sin(2*Math.PI*f*t) + 0.35*Math.sin(4*Math.PI*f*t + 0.6);
    const drift = 1.2 * Math.sin(2*Math.PI*0.05*t);
    const wob = cfg.motion ? cfg.motion * Math.sin(2*Math.PI*cfg.motionFreq*t) : 0;
    const r = base[0] + amp[0]*p + drift + wob;
    const g = base[1] + amp[1]*p + drift + wob*0.95;
    const b = base[2] + amp[2]*p + drift + wob*0.90;
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      // per-pixel dithering noise: without it a uniform frame quantises to a
      // staircase and the sub-1% pulse would be destroyed by 8-bit rounding
      d[i]   = Math.max(0, Math.min(255, r + rnd()*3.5));
      d[i+1] = Math.max(0, Math.min(255, g + rnd()*3.5));
      d[i+2] = Math.max(0, Math.min(255, b + rnd()*3.5));
      d[i+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    requestAnimationFrame(draw);
  }
  draw();

  const stream = cv.captureStream(30);
  navigator.mediaDevices.getUserMedia = async () => stream;
  window.__injected = true;
}
"""

CASES = [
    {"mode": "face",   "bpm": 72, "melanin": 0.0,  "motion": 0,   "motionFreq": 0,
     "label": "face mode, lighter skin, still"},
    {"mode": "finger", "bpm": 96, "melanin": 0.0,  "motion": 0,   "motionFreq": 0,
     "label": "finger mode (flash-lit contact)"},
    {"mode": "face",   "bpm": 64, "melanin": 0.85, "motion": 0,   "motionFreq": 0,
     "label": "face mode, SIMULATED dark skin (melanin 0.85)"},
    {"mode": "finger", "bpm": 78, "melanin": 0.85, "motion": 0,   "motionFreq": 0,
     "label": "finger mode, SIMULATED dark skin (melanin 0.85)"},
    {"mode": "face",   "bpm": 68, "melanin": 0.85, "motion": 2.2, "motionFreq": 1.6,
     "label": "face, dark skin + MOTION (should warn, not lie)"},
]

rows = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=["--autoplay-policy=no-user-gesture-required"])
    for c in CASES:
        ctx = browser.new_context(permissions=["camera"], viewport={"width": 1200, "height": 1500})
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console:" + m.text) if m.type == "error" else None)
        pg.goto(URL)
        pg.wait_for_timeout(400)
        pg.evaluate(INJECT, c)
        pg.click('.mode-btn[data-mode="%s"]' % c["mode"])
        pg.click("#startPulse")
        pg.wait_for_selector("#pulseResults", state="visible", timeout=90000)
        pg.wait_for_timeout(1200)

        est_txt = pg.inner_text("#bpmVal")
        try: est = float(est_txt)
        except ValueError: est = None
        rows.append({
            "label": c["label"], "truth": c["bpm"], "est": est,
            "err": abs(est - c["bpm"]) if est is not None else None,
            "badge": pg.inner_text("#qualBadge"), "snr": pg.inner_text("#snrVal"),
            "stab": pg.inner_text("#stabVal"), "method": pg.inner_text("#methodVal"),
            "frames": pg.inner_text("#frameCount"), "beats": pg.inner_text("#beatsVal"),
            "errs": errs, "expect_warn": c["motion"] > 0,
        })
        if c["mode"] == "finger" and c["bpm"] == 96:
            pg.screenshot(path=os.path.join(HERE, "qa_pulse_result.png"), full_page=True)
        ctx.close()
    browser.close()

print("\n===== END-TO-END BROWSER CAPTURE TEST =====")
print("(synthetic frames through a real MediaStream — not human validation)\n")
fails = 0
for r in rows:
    print("  %-48s truth %3d -> est %s   err %s" % (
        r["label"], r["truth"],
        ("%.0f" % r["est"]) if r["est"] is not None else "n/a",
        ("%.1f" % r["err"]) if r["err"] is not None else "n/a"))
    print("       [%s]  SNR %s | stability %s | %s | frames %s | beats %s"
          % (r["badge"], r["snr"], r["stab"], r["method"], r["frames"], r["beats"]))
    if r["errs"]:
        print("       JS ERRORS:", r["errs"][:3]); fails += 1
    accurate = r["err"] is not None and r["err"] < 3
    warned = "Poor" in r["badge"] or "not reliable" in r["badge"]
    if r["expect_warn"]:
        if accurate or warned:
            print("       -> PASS (%s)" % ("accurate" if accurate else "correctly flagged unreliable"))
        else:
            print("       -> FAIL: confidently wrong under motion"); fails += 1
    else:
        if accurate: print("       -> PASS")
        elif warned: print("       -> ACCEPTABLE (flagged unreliable instead of reporting a wrong number)")
        else: print("       -> FAIL: wrong value reported confidently"); fails += 1
    print()

print("===== %s =====\n" % ("ALL OK" if fails == 0 else "%d FAILURE(S)" % fails))
sys.exit(1 if fails else 0)
