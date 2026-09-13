/*
 * Modality 3 — medical document scanning.
 *
 * Reads a photographed medication box or lab report and proposes values for the risk
 * model. OCR runs client-side (Tesseract.js), so the image never leaves the device.
 *
 * HARD RULE: this module returns PROPOSALS, never answers. Every proposal is shown to
 * the user with the text it was read from, and only takes effect if they confirm it.
 * That is what makes imperfect OCR acceptable — a misread becomes a question the user
 * corrects, not a silently wrong risk estimate.
 */
(function (root) {
  "use strict";

  /* ---- fuzzy matching ----------------------------------------------------
   * OCR on a curved, glossy medicine box mangles characters predictably: TELMA ->
   * TEIMA, 0 <-> O, 1 <-> l. Exact matching would miss most real photographs, so
   * tokens are normalised and compared with a bounded edit distance.
   * -------------------------------------------------------------------- */
  function normalise(s) {
    return String(s).toLowerCase()
      .replace(/[0o]/g, "0").replace(/[1il|]/g, "1").replace(/5/g, "s")
      .replace(/[^a-z0-9]/g, "");
  }

  function editDistance(a, b, cap) {
    if (Math.abs(a.length - b.length) > cap) return cap + 1;
    var prev = [], cur = [], i, j;
    for (j = 0; j <= b.length; j++) prev[j] = j;
    for (i = 1; i <= a.length; i++) {
      cur[0] = i;
      var best = i;
      for (j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1,
                          prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
        if (cur[j] < best) best = cur[j];
      }
      if (best > cap) return cap + 1;
      for (j = 0; j <= b.length; j++) prev[j] = cur[j];
    }
    return prev[b.length];
  }

  /** Tolerance scales with word length: short names must match nearly exactly. */
  function tolerance(len) { return len <= 4 ? 0 : (len <= 6 ? 1 : 2); }

  function tokenMatches(token, target) {
    var t = normalise(token), g = normalise(target);
    if (!t || !g) return false;
    if (t === g) return true;
    if (g.length >= 6 && t.indexOf(g) !== -1) return true;   // "telma40" contains "telma"
    return editDistance(t, g, tolerance(g.length)) <= tolerance(g.length);
  }

  /* ---- medication extraction -------------------------------------------- */
  function findMedications(text, dict) {
    var tokens = String(text).split(/[\s,;()\[\]\/\\|:+]+/).filter(function (t) {
      return t && t.length >= 4;
    });
    var found = [], seen = {};

    Object.keys(dict.classes).forEach(function (className) {
      var cls = dict.classes[className];
      Object.keys(cls.drugs).forEach(function (generic) {
        var names = [generic].concat(cls.drugs[generic]);
        for (var n = 0; n < names.length; n++) {
          for (var t = 0; t < tokens.length; t++) {
            if (tokenMatches(tokens[t], names[n])) {
              var key = className + "|" + generic;
              if (seen[key]) return;
              seen[key] = true;
              found.push({
                generic: generic,
                matched_text: tokens[t],
                matched_name: names[n],
                is_brand: n > 0,
                drug_class: className,
                implies: cls.implies,
                confidence: cls.confidence,
                prompt: cls.prompt
              });
              return;
            }
          }
        }
      });
    });
    return found;
  }

  /* ---- lab value extraction ---------------------------------------------
   * Looks for "<alias> ... <number>" within a short window, so a value is only
   * picked up when it sits next to its own label.
   * -------------------------------------------------------------------- */
  function findLabValues(text, dict) {
    var flat = String(text).replace(/\s+/g, " ").toLowerCase();
    var out = [];
    Object.keys(dict.lab_tests).forEach(function (key) {
      var spec = dict.lab_tests[key];
      for (var i = 0; i < spec.aliases.length; i++) {
        var alias = spec.aliases[i];
        var idx = flat.indexOf(alias);
        if (idx === -1) continue;
        var tail = flat.slice(idx + alias.length, idx + alias.length + 40);
        var m = tail.match(/(\d+(?:\.\d+)?)/);
        if (!m) continue;
        var val = parseFloat(m[1]);
        var lo = spec.plausible_range[0], hi = spec.plausible_range[1];
        if (!(val >= lo && val <= hi)) continue;     // reject OCR garbage
        out.push({
          test: key, value: val, unit: spec.unit,
          implies: spec.implies || null,
          threshold: spec.threshold || null,
          exceeds_threshold: spec.threshold ? val >= spec.threshold : null,
          rule: spec.rule || null,
          matched_text: alias + " " + m[1]
        });
        break;
      }
    });
    return out;
  }

  /* ---- turn findings into proposals the user must confirm ---------------- */
  function buildProposals(meds, labs) {
    var proposals = [], seen = {};

    function add(field, value, reason, confidence, prompt) {
      var key = field + "=" + value;
      if (seen[key]) {
        // Same conclusion reached a second way (e.g. metformin AND a high HbA1c).
        // Show both reasons rather than silently discarding the corroboration.
        seen[key].reason += "; also " + reason.charAt(0).toLowerCase() + reason.slice(1);
        if (confidence === "high") seen[key].confidence = "high";
        return;
      }
      var prop = { field: field, value: value, reason: reason,
                   confidence: confidence, prompt: prompt, accepted: null };
      seen[key] = prop;
      proposals.push(prop);
    }

    meds.forEach(function (m) {
      var label = m.generic + (m.is_brand ? " (read as “" + m.matched_text + "”)" : "");
      if (m.confidence === "signal" || !m.implies) {
        proposals.push({
          field: m.implies || null, value: null, question_only: true,
          reason: "Found " + label, confidence: m.confidence,
          prompt: m.prompt, accepted: null
        });
      } else {
        add(m.implies, true, "Found " + label, m.confidence, m.prompt);
      }
    });

    labs.forEach(function (l) {
      if (l.test === "systolic_bp") {
        add("sbp", l.value, "Read blood pressure " + l.value + " " + l.unit, "medium",
            "Is this your current blood pressure?");
      } else if (l.implies === "diabetes" && l.exceeds_threshold) {
        add("diabetes", true,
            l.test.replace(/_/g, " ") + " of " + l.value + " " + l.unit +
            " (" + l.rule + ")", "high",
            "This lab value is in the diabetic range. Have you been diagnosed?");
      }
    });

    return proposals;
  }

  function analyseText(text, dict) {
    var meds = findMedications(text, dict);
    var labs = findLabValues(text, dict);
    return {
      raw_text_length: String(text).length,
      medications: meds,
      lab_values: labs,
      proposals: buildProposals(meds, labs),
      nothing_found: meds.length === 0 && labs.length === 0
    };
  }

  /* ---- OCR --------------------------------------------------------------- */
  async function ocrImage(file, onProgress) {
    if (typeof Tesseract === "undefined") {
      throw new Error("OCR engine not loaded");
    }
    var res = await Tesseract.recognize(file, "eng", {
      logger: function (m) {
        if (onProgress && m.status === "recognizing text") onProgress(m.progress);
      }
    });
    return res.data.text;
  }

  async function scanImage(file, dict, onProgress) {
    var text = await ocrImage(file, onProgress);
    var out = analyseText(text, dict);
    out.raw_text = text;
    return out;
  }

  var API = {
    normalise: normalise, editDistance: editDistance, tokenMatches: tokenMatches,
    findMedications: findMedications, findLabValues: findLabValues,
    buildProposals: buildProposals, analyseText: analyseText, scanImage: scanImage
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  root.BrainShieldDocScan = API;
})(typeof globalThis !== "undefined" ? globalThis : this);
