"""
End-to-end test of the three-step assessment in a real browser.

The camera cannot be driven headlessly, so the rhythm step is exercised two ways:
the skip path through the real UI, and the analysis path by injecting synthetic
captures directly into the same functions the camera feeds.
"""
import re, sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/assess.html"
fails, passes = [], []

def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def fill_step1(page, age="68", sex="men", sbp="150", **kw):
    page.fill("#age", age)
    page.click(f'.toggle[data-field="sex"] button[data-value="{sex}"]')
    if sbp:
        page.fill("#sbp", sbp)
    else:
        page.click('.toggle[data-field="sbp_known"] button[data-value="0"]')
    for f, v in kw.items():
        page.click(f'.toggle[data-field="{f}"] button[data-value="{v}"]')
    page.click("#toStep2")

with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--use-fake-ui-for-media-stream"])
    page = b.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL, wait_until="networkidle")

    check("page loads with no JS errors", not errors, "; ".join(errors[:2]))
    check("all three engines loaded", page.evaluate(
        "!!(window.BrainShieldRisk && window.BrainShieldAF && window.BrainShieldCapture "
        "&& window.BrainShieldDocScan && window.RPPG)"))
    check("emergency guidance shown first", "call emergency services" in page.inner_text(".emergency"))
    check("starts on step 1", page.is_visible("#step1") and not page.is_visible("#step2"))

    # ---- step 1 -> 2 ----
    fill_step1(page, on_htn_meds="0", diabetes="0", current_smoker="1",
               prevalent_cvd="0", atrial_fib="null", smokeless_tobacco="0")
    page.wait_for_selector("#step2", state="visible")
    check("advances to the pulse step", page.is_visible("#step2"))
    check("step rail marks step 1 done", "done" in page.get_attribute("#rail1", "class"))

    # ---- rhythm analysis on synthetic captures (the camera's own code path) ----
    def synth(kind):
        """
        Build the waveform from explicit BEAT TIMES, then place a pulse at each.

        An earlier version modulated the carrier's instantaneous frequency per sample,
        which the 0.7-3.5 Hz pulse-band filter averages out within each beat -- so
        'AF' came back as regular. Atrial fibrillation is irregularity of the
        interval between beats, so the beat times are what must be irregular.
        """
        return page.evaluate("""(kind) => {
            const fs = 30, dur = 62;
            // 1. beat times
            let t = 0.5; const beats = [];
            let seed = 12345;
            const rnd = () => { seed = (seed*1103515245 + 12345) & 0x7fffffff; return seed/0x7fffffff; };
            while (t < dur) {
                let rr;
                if (kind === 'af') {
                    // irregularly irregular: wide, memoryless interval scatter
                    rr = 0.86 + (rnd() - 0.5) * 0.52;
                } else {
                    // normal sinus with gentle respiratory variation
                    rr = 0.857 + 0.022 * Math.sin(t * 2 * Math.PI / 4.0) + (rnd() - 0.5) * 0.012;
                }
                beats.push(t); t += rr;
            }
            // 2. render a pulse waveform at those beat times
            const n = fs * dur, times = [], r = [], g = [], b = [];
            for (let i = 0; i < n; i++) {
                const tt = i / fs;
                let v = 0;
                for (let k = 0; k < beats.length; k++) {
                    const d = tt - beats[k];
                    if (d > -0.25 && d < 0.55) {
                        // asymmetric pulse: fast upstroke, slower decay
                        v += Math.exp(-Math.pow(d / 0.085, 2)) - 0.32 * Math.exp(-Math.pow((d - 0.2) / 0.14, 2));
                    }
                }
                times.push(tt); r.push(150 + 9 * v); g.push(90 + 3 * v); b.push(70 + 2 * v);
            }
            const res = window.BrainShieldCapture.analyseCapture({times,r,g,b}, window.AF_MODEL);
            return {outcome: res.outcome, sets_af: res.sets_af, hr: res.hr,
                    n: res.n_intervals, headline: res.headline, detail: res.detail,
                    rmssd_norm: res.features ? res.features.rmssd_norm : null};
        }""", kind)

    reg = synth("regular")
    check("a steady synthetic pulse is analysable", reg["outcome"] in ("regular", "irregular"),
          f"outcome={reg['outcome']} hr={reg['hr'] and round(reg['hr'])} n={reg['n']}")
    if reg["outcome"] == "regular":
        check("a regular rhythm rules AF out", reg["sets_af"] is False)
    check("regular synthetic rhythm is classified regular", reg["outcome"] == "regular",
          f"outcome={reg['outcome']} rmssd_norm={reg['rmssd_norm'] and round(reg['rmssd_norm'],4)}")
    af = synth("af")
    check("irregular synthetic rhythm is detected as irregular", af["outcome"] == "irregular",
          f"outcome={af['outcome']} hr={af['hr'] and round(af['hr'])} "
          f"rmssd_norm={af['rmssd_norm'] and round(af['rmssd_norm'],4)}")
    check("an irregular rhythm never sets AF true", af["sets_af"] is not True,
          f"outcome={af['outcome']}")
    if af["outcome"] == "irregular":
        check("an irregular rhythm refers to an ECG", "ECG" in (af["headline"] + af["detail"]))
        check("an irregular rhythm denies being a diagnosis",
              "does NOT mean you have atrial fibrillation" in af["detail"])

    check("a too-short recording is rejected", page.evaluate("""() => {
        const fs=30,n=fs*10,times=[],r=[],g=[],b=[];
        for(let i=0;i<n;i++){times.push(i/fs);r.push(150+9*Math.sin(i*0.25));g.push(90);b.push(70);}
        return window.BrainShieldCapture.analyseCapture({times,r,g,b},window.AF_MODEL).outcome;
    }""") == "unusable")

    # ---- skip camera -> step 3 ----
    page.click("#skipCam")
    page.wait_for_selector("#step3", state="visible")
    check("skipping the pulse step advances to documents", page.is_visible("#step3"))
    check("documents step is labelled optional", "optional" in page.inner_text("#step3").lower())

    # ---- document scanning (OCR text path, no image needed) ----
    props = page.evaluate("""() => {
        const out = window.BrainShieldDocScan.analyseText(
            'GLYCOMET 500 SR Metformin  TELMA 40 Telmisartan  HbA1c : 8.1 %',
            window.MEDICATIONS);
        return out.proposals.map(p => ({field:p.field, value:p.value, reason:p.reason}));
    }""")
    fields = [p["field"] for p in props]
    check("metformin proposes diabetes", "diabetes" in fields, str(fields))
    check("telmisartan proposes BP medication", "on_htn_meds" in fields)
    check("HbA1c 8.1 is picked up", any("8.1" in (p["reason"] or "") for p in props))

    # ---- results ----
    page.click("#toResults")
    page.wait_for_selector("#step4", state="visible")
    disp = page.inner_text("#riskDisplay")
    check("a risk figure is shown", re.search(r"\d+\.\d+\s*%", disp) is not None, disp.strip())
    check("unknown AF still yields a range", "–" in disp, disp.strip())
    contrib = page.inner_text("#contribList")
    check("contribution panel lists the questionnaire", "Questionnaire" in contrib)
    check("contribution panel records the skipped pulse check", "skipped" in contrib.lower())
    check("contribution panel records unused documents", "not used" in contrib.lower())
    badge = page.inner_text("#modelBadge")
    check("rhythm model performance is disclosed", "MIT-BIH" in badge and "AUROC" in badge)
    check("capture-path limitation is disclosed", "has not been validated" in badge)
    check("calibration inequity is disclosed", "Black participants" in badge)
    check("privacy is stated", "No answer, video frame or photograph is uploaded" in badge)

    # ---- a fully-known profile gives a point estimate ----
    page.goto(URL, wait_until="networkidle")
    fill_step1(page, age="70", sex="men", sbp="150", on_htn_meds="0", diabetes="0",
               current_smoker="1", prevalent_cvd="0", atrial_fib="0", smokeless_tobacco="0")
    page.click("#skipCam"); page.wait_for_selector("#step3", state="visible")
    page.click("#toResults"); page.wait_for_selector("#step4", state="visible")
    d2 = page.inner_text("#riskDisplay")
    m = re.search(r"([\d.]+)\s*%", d2)
    check("complete profile gives a single number", "–" not in d2, d2.strip())
    check("value matches the validated Python model (7.37%)",
          m and abs(float(m.group(1)) - 7.37) < 0.05, f"got {m.group(1) if m else None}%")
    check("counterfactuals appear for a complete profile", page.is_visible("#cfPanel"))
    check("quitting smoking is offered", "smok" in page.inner_text("#cfList").lower())
    check("starting medication is never offered",
          not re.search(r"start .*medication", page.inner_text("#cfList"), re.I))

    # ---- smokeless tobacco: the India-specific risk factor ----
    page.goto(URL, wait_until="networkidle")
    check("the questionnaire asks about chewed tobacco",
          page.is_visible('.toggle[data-field="smokeless_tobacco"]'))
    check("it names the Indian products by name",
          "gutkha" in page.inner_text("#step1").lower()
          and "khaini" in page.inner_text("#step1").lower())
    fill_step1(page, age="60", sex="men", sbp="130", on_htn_meds="0", diabetes="0",
               current_smoker="0", prevalent_cvd="0", atrial_fib="0", smokeless_tobacco="1")
    page.click("#skipCam"); page.wait_for_selector("#step3", state="visible")
    page.click("#toResults"); page.wait_for_selector("#step4", state="visible")
    d3 = page.inner_text("#riskDisplay")
    m3 = re.search(r"([\d.]+)\s*%", d3)
    slt_risk = float(m3.group(1)) if m3 else None

    page.goto(URL, wait_until="networkidle")
    fill_step1(page, age="60", sex="men", sbp="130", on_htn_meds="0", diabetes="0",
               current_smoker="0", prevalent_cvd="0", atrial_fib="0", smokeless_tobacco="0")
    page.click("#skipCam"); page.wait_for_selector("#step3", state="visible")
    page.click("#toResults"); page.wait_for_selector("#step4", state="visible")
    m4 = re.search(r"([\d.]+)\s*%", page.inner_text("#riskDisplay"))
    base_risk = float(m4.group(1)) if m4 else None
    check("chewed tobacco raises risk by the published 1.35x",
          slt_risk and base_risk and abs(slt_risk / base_risk - 1.35) < 0.05,
          f"{base_risk}% -> {slt_risk}%  ({slt_risk/base_risk:.2f}x)" if slt_risk and base_risk else "n/a")
    # re-enter the chewer profile and assert the counterfactual actually appears
    page.goto(URL, wait_until="networkidle")
    fill_step1(page, age="60", sex="men", sbp="130", on_htn_meds="0", diabetes="0",
               current_smoker="0", prevalent_cvd="0", atrial_fib="0", smokeless_tobacco="1")
    page.click("#skipCam"); page.wait_for_selector("#step3", state="visible")
    page.click("#toResults"); page.wait_for_selector("#step4", state="visible")
    cf_txt = page.inner_text("#cfList")
    check("quitting chewed tobacco is offered as a lever",
          "chewing tobacco" in cf_txt.lower(), cf_txt.split("\n")[0][:60])
    check("it is presented as weaker than quitting smoking would be",
          "meta-analysis" in cf_txt.lower())

    page.screenshot(path="/tmp/assess_result.png", full_page=True)
    b.close()

print(f"\n{len(passes)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
