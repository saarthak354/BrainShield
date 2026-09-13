"""
End-to-end test of risk5.html in a real browser.

Drives the actual form the way a person would and asserts on rendered output, so a
broken selector or a JS error fails the build rather than silently showing a dash.
Run a local server first:  python3 -m http.server 8000
"""
import re, sys
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000/risk5.html"
fails, passes = [], []

def check(name, ok, detail=""):
    (passes if ok else fails).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def pct(text):
    m = re.search(r"([\d.]+)\s*%", text)
    return float(m.group(1)) if m else None

with sync_playwright() as pw:
    b = pw.chromium.launch()
    page = b.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(URL, wait_until="networkidle")

    check("page loads with no JS errors", not errors, "; ".join(errors[:2]))
    check("emergency guidance is above the fold",
          "call emergency services" in page.inner_text(".emergency"))

    # ---- case 1: fully specified profile -> single number ----
    page.fill("#age", "70")
    page.click('.toggle[data-field="sex"] button[data-value="men"]')
    page.fill("#sbp", "150")
    page.click('.toggle[data-field="on_htn_meds"] button[data-value="0"]')
    page.click('.toggle[data-field="diabetes"] button[data-value="0"]')
    page.click('.toggle[data-field="current_smoker"] button[data-value="1"]')
    page.click('.toggle[data-field="prevalent_cvd"] button[data-value="0"]')
    page.click('.toggle[data-field="atrial_fib"] button[data-value="0"]')
    page.click('.toggle[data-field="smokeless_tobacco"] button[data-value="0"]')
    page.click("#submitBtn")
    page.wait_for_selector("#results", state="visible")

    disp = page.inner_text("#riskDisplay")
    v = pct(disp)
    check("complete profile shows a single percentage", v is not None and "–" not in disp, disp.strip())
    check("value matches the Python model (7.37%)", v is not None and abs(v - 7.37) < 0.05, f"got {v}%")
    check("'what would change this' panel appears",
          page.is_visible("#cfPanel"))
    check("quitting smoking is offered as a lever",
          "smok" in page.inner_text("#cfList").lower())
    check("starting medication is NOT offered as a lever",
          not re.search(r"start.*medication|begin.*medication", page.inner_text("#cfList"), re.I))
    check("'what would sharpen this' is hidden when nothing is unknown",
          not page.is_visible("#voiPanel"))
    check("plain-language framing is shown",
          "1,000 people" in page.inner_text("#plainly"))

    # ---- case 2: unknowns -> a range, plus guidance ----
    page.reload(wait_until="networkidle")
    page.fill("#age", "70")
    page.click('.toggle[data-field="sex"] button[data-value="men"]')
    page.click('.toggle[data-field="sbp_known"] button[data-value="0"]')
    page.click("#submitBtn")
    page.wait_for_selector("#results", state="visible")

    disp2 = page.inner_text("#riskDisplay")
    check("unknown inputs produce a range, not a fabricated point", "–" in disp2, disp2.strip())
    check("BP field is disabled once 'not sure' is chosen", page.is_disabled("#sbp"))
    check("'what would sharpen this' panel appears", page.is_visible("#voiPanel"))
    voi = page.inner_text("#voiList")
    check("guidance says how to find each one out", "pharmacy" in voi.lower() or "ecg" in voi.lower())
    check("range is explained in words", "range" in page.inner_text("#plainly").lower())

    # ---- case 3: under 55 -> extrapolation warning ----
    page.reload(wait_until="networkidle")
    page.fill("#age", "40")
    page.click('.toggle[data-field="sex"] button[data-value="women"]')
    page.fill("#sbp", "118")
    page.click('.toggle[data-field="on_htn_meds"] button[data-value="0"]')
    page.click('.toggle[data-field="diabetes"] button[data-value="0"]')
    page.click('.toggle[data-field="current_smoker"] button[data-value="0"]')
    page.click('.toggle[data-field="prevalent_cvd"] button[data-value="0"]')
    page.click('.toggle[data-field="atrial_fib"] button[data-value="0"]')
    page.click('.toggle[data-field="smokeless_tobacco"] button[data-value="0"]')
    page.click("#submitBtn")
    page.wait_for_selector("#results", state="visible")
    check("under-55 shows an extrapolation warning",
          "validated range" in page.inner_text("#warnings"))

    # ---- case 4: very high BP raises its own flag ----
    page.reload(wait_until="networkidle")
    page.fill("#age", "68")
    page.click('.toggle[data-field="sex"] button[data-value="men"]')
    page.fill("#sbp", "195")
    page.click('.toggle[data-field="on_htn_meds"] button[data-value="0"]')
    page.click('.toggle[data-field="diabetes"] button[data-value="0"]')
    page.click('.toggle[data-field="current_smoker"] button[data-value="0"]')
    page.click('.toggle[data-field="prevalent_cvd"] button[data-value="0"]')
    page.click('.toggle[data-field="atrial_fib"] button[data-value="0"]')
    page.click('.toggle[data-field="smokeless_tobacco"] button[data-value="0"]')
    page.click("#submitBtn")
    page.wait_for_selector("#results", state="visible")
    check("SBP above 180 triggers a separate medical note",
          "180" in page.inner_text("#warnings"))
    check("limitations and citation are shown with the result",
          "c-statistic" in page.inner_text("#modelBadge"))
    check("calibration inequity is disclosed to the user",
          "Black participants" in page.inner_text("#modelBadge"))

    page.screenshot(path="/tmp/risk5_result.png", full_page=True)
    b.close()

print(f"\n{len(passes)} passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
