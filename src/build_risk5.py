#!/usr/bin/env python3
"""Inline the risk model and its JS core into a single self-contained risk5.html."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)

model = json.load(open(p("risk", "risk_model_export.json")))
core = open(p("risk", "risk_core.js")).read()
tpl = open(p("risk5_template.html")).read()

tpl = tpl.replace("/*__RISK_MODEL_DATA__*/", "window.RISK_MODEL = " + json.dumps(model) + ";")
tpl = tpl.replace("/*__RISK_CORE__*/", core)

for marker in ["__RISK_MODEL_DATA__", "__RISK_CORE__"]:
    assert marker not in tpl, "unreplaced placeholder: " + marker

out = p("..", "risk5.html")
open(out, "w").write(tpl)
print("wrote risk5.html  %.1f KB" % (len(tpl) / 1024))
