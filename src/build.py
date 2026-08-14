#!/usr/bin/env python3
"""Inline the model, baseline, and rPPG code into a single self-contained index.html."""
import json, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)

model = json.load(open(p("model_export.json")))
baseline = json.load(open(p("baseline_lr.json")))
rows = list(csv.reader(open(p("..", "results", "shap_importance.csv"))))[1:]
top_feats = [r[0] for r in rows[:8]]

tpl = open(p("index_template.html")).read()
tpl = tpl.replace("/*__MODEL_DATA__*/", "const MODEL = " + json.dumps(model) + ";")
tpl = tpl.replace("/*__BASELINE_DATA__*/", "const BASELINE = " + json.dumps(baseline) + ";")
tpl = tpl.replace("/*__SHAP_DATA__*/", "const SHAP_TOP = " + json.dumps(top_feats) + ";")
tpl = tpl.replace("/*__RPPG_CSS__*/", open(p("rppg_section.css")).read())
tpl = tpl.replace("<!--__RPPG_SECTION__-->", open(p("rppg_section.html")).read())
tpl = tpl.replace("/*__RPPG_CORE__*/", open(p("rppg_core.js")).read())
tpl = tpl.replace("/*__RPPG_UI__*/", open(p("rppg_ui.js")).read())

# sanity: no unreplaced placeholders
for marker in ["__MODEL_DATA__", "__BASELINE_DATA__", "__SHAP_DATA__",
               "__RPPG_CSS__", "__RPPG_SECTION__", "__RPPG_CORE__", "__RPPG_UI__"]:
    assert marker not in tpl, "unreplaced placeholder: " + marker

open(p("index.html"), "w").write(tpl)
print("wrote index.html  %.2f MB" % (len(tpl) / 1e6))
