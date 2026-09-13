#!/usr/bin/env python3
"""Inline all three modalities into a single self-contained assess.html."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)

parts = {
    "__RISK_MODEL_DATA__":  "window.RISK_MODEL = "  + json.dumps(json.load(open(p("risk","risk_model_export.json")))) + ";",
    "__AF_MODEL_DATA__":    "window.AF_MODEL = "    + json.dumps(json.load(open(p("af","af_model_export.json")))) + ";",
    "__MEDICATIONS_DATA__": "window.MEDICATIONS = " + json.dumps(json.load(open(p("docscan","medications.json")))) + ";",
    "__RISK_CORE__":   open(p("risk","risk_core.js")).read(),
    "__RPPG_CORE__":   open(p("rppg_core.js")).read(),
    "__AF_CORE__":     open(p("af","af_core.js")).read(),
    "__AF_CAPTURE__":  open(p("af","af_capture.js")).read(),
    "__DOCSCAN_CORE__":open(p("docscan","docscan_core.js")).read(),
    "__APP__":         open(p("assess_app.js")).read(),
}

tpl = open(p("assess_template.html")).read()
for marker, content in parts.items():
    tpl = tpl.replace("/*" + marker + "*/", content)
for marker in parts:
    assert marker not in tpl, "unreplaced placeholder: " + marker

out = p("..", "assess.html")
open(out, "w").write(tpl)
print("wrote assess.html  %.1f KB" % (len(tpl) / 1024))
