/*
 * Wizard controller: questionnaire -> pulse rhythm check -> documents -> risk.
 *
 * The profile is built up across the three steps. Each step may only ever RESOLVE an
 * unknown; no step is allowed to overwrite something the user stated directly, and the
 * rhythm check is allowed to set atrial fibrillation to false but never to true.
 */
(function () {
  "use strict";

  var R = window.BrainShieldRisk;
  var AF = window.BrainShieldAF;
  var CAP = window.BrainShieldCapture;
  var DOC = window.BrainShieldDocScan;

  var state = {
    answered: {},          // what the user typed in step 1
    profile: {},           // working profile, refined by later steps
    af: null,              // rhythm-check outcome
    docFindings: null,
    docApplied: [],        // proposals the user accepted
    estimates: {}          // snapshots after each step, for the contribution panel
  };

  var ui = {};
  ["age","sbp","camStage","cam","countdown","camStatus","wave","startCam","skipCam",
   "camResult","camNextRow","toStep2","toStep3","toResults","drop","fileInput","ocrStatus",
   "ocrResult","riskDisplay","riskLabel","riskBand","plainly","optimalVal","cohortVal",
   "contribList","voiPanel","voiList","cfPanel","cfList","warnings","modelBadge","restart"]
    .forEach(function (id) { ui[id] = document.getElementById(id); });

  /* ---------------------------------------------------------- step 1 ---- */
  var toggles = { sbp_known: 1, sex: null, on_htn_meds: null, diabetes: null,
                  current_smoker: null, prevalent_cvd: null, atrial_fib: null,
                  smokeless_tobacco: null };

  function parseVal(v) { return v === "null" ? null : (v === "1" ? true : (v === "0" ? false : v)); }

  document.querySelectorAll(".toggle").forEach(function (group) {
    var field = group.getAttribute("data-field");
    group.querySelectorAll("button").forEach(function (btn) {
      btn.addEventListener("click", function () {
        group.querySelectorAll("button").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        toggles[field] = parseVal(btn.getAttribute("data-value"));
        if (field === "sbp_known") {
          var known = btn.getAttribute("data-value") === "1";
          ui.sbp.disabled = !known;
          if (!known) ui.sbp.value = "";
        }
      });
    });
  });

  function showStep(n) {
    [1, 2, 3, 4].forEach(function (i) {
      document.getElementById("step" + i).classList.toggle("active", i === n);
      var rail = document.getElementById("rail" + i);
      rail.classList.toggle("active", i === n);
      rail.classList.toggle("done", i < n);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function collectStep1() {
    var age = parseFloat(ui.age.value);
    if (!age) { alert("Please enter your age."); return null; }
    if (!toggles.sex) { alert("Please select a sex — the model uses a different equation for each."); return null; }
    var sbpRaw = ui.sbp.value;
    return {
      age: age, sex: toggles.sex,
      sbp: (toggles.sbp_known && sbpRaw !== "") ? parseFloat(sbpRaw) : null,
      on_htn_meds: toggles.on_htn_meds, diabetes: toggles.diabetes,
      current_smoker: toggles.current_smoker, prevalent_cvd: toggles.prevalent_cvd,
      atrial_fib: toggles.atrial_fib, smokeless_tobacco: toggles.smokeless_tobacco
    };
  }

  ui.toStep2.addEventListener("click", function () {
    var p = collectStep1();
    if (!p) return;
    state.answered = JSON.parse(JSON.stringify(p));
    state.profile = JSON.parse(JSON.stringify(p));
    state.estimates.questionnaire = R.estimate(state.profile, { n: 20000 });
    showStep(2);
  });

  /* ---------------------------------------------------------- step 2 ---- */
  var capture = null;

  function drawWave(samples) {
    var cv = ui.wave, ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    var r = samples.r;
    if (r.length < 4) return;
    var n = Math.min(r.length, 300), start = r.length - n;
    var slice = r.slice(start), mn = Math.min.apply(null, slice), mx = Math.max.apply(null, slice);
    if (mx - mn < 1e-9) return;
    ctx.beginPath();
    ctx.strokeStyle = "#1C7293"; ctx.lineWidth = 2;
    for (var i = 0; i < n; i++) {
      var x = (i / (n - 1)) * cv.width;
      var y = cv.height - ((slice[i] - mn) / (mx - mn)) * (cv.height - 10) - 5;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }

  ui.startCam.addEventListener("click", async function () {
    ui.startCam.disabled = true;
    ui.camResult.innerHTML = "";
    ui.camStatus.textContent = "Requesting camera…";
    capture = new CAP.AFCapture(ui.cam);
    capture.onTick = function (elapsed, samples) {
      var left = Math.max(0, CAP.TARGET_SECONDS - elapsed);
      ui.countdown.textContent = left.toFixed(0) + "s";
      if (elapsed > 1.5) ui.camStatus.textContent = "Measuring — keep your finger still.";
      drawWave(samples);
      if (left <= 0) finishCapture();
    };
    try {
      var info = await capture.start();
      ui.camStage.classList.add("live");
      ui.camStatus.textContent = info.torch
        ? "Flash on. Keep your finger over the lens."
        : "Couldn't turn the flash on — keep your finger still in good light.";
    } catch (e) {
      ui.startCam.disabled = false;
      ui.camStatus.textContent = "";
      ui.camResult.innerHTML = warn("bad", "Camera unavailable",
        "Your browser refused camera access (" + (e && e.name ? e.name : "unknown") + "). "
      + "The camera needs an https:// page and permission. You can skip this step — the "
      + "rest of the assessment works without it.");
      showCamNext();
    }
  });

  var finished = false;
  function finishCapture() {
    if (finished) return;
    finished = true;
    capture.stop();
    ui.camStage.classList.remove("live");
    ui.camStatus.textContent = "Analysing rhythm…";
    setTimeout(function () {
      var res;
      try {
        res = CAP.analyseCapture(capture.samples, window.AF_MODEL);
      } catch (e) {
        res = { outcome: "unusable", sets_af: null,
                headline: "Something went wrong analysing the recording",
                detail: String(e && e.message || e) };
      }
      state.af = res;
      renderCamResult(res);
      ui.camStatus.textContent = "";
      ui.countdown.textContent = "—";
      ui.startCam.disabled = false;
      ui.startCam.textContent = "Record again";
      finished = false;
      showCamNext();
    }, 30);
  }

  function warn(kind, title, body) {
    return '<div class="warnbox ' + kind + '"><b>' + title + '</b>' + body + '</div>';
  }

  function renderCamResult(res) {
    var hr = res.hr ? ("  Heart rate " + Math.round(res.hr) + " bpm.") : "";
    if (res.outcome === "regular") {
      ui.camResult.innerHTML = warn("good", res.headline, res.detail + hr);
      // A regular reading resolves AF only if the user hadn't already answered it.
      if (state.profile.atrial_fib === null || state.profile.atrial_fib === undefined) {
        state.profile.atrial_fib = false;
      }
    } else if (res.outcome === "irregular") {
      ui.camResult.innerHTML = warn("bad", res.headline, res.detail + hr);
      // deliberately does not touch state.profile.atrial_fib
    } else {
      ui.camResult.innerHTML = warn("", res.headline, res.detail);
    }
    state.estimates.afterCamera = R.estimate(state.profile, { n: 20000 });
  }

  function showCamNext() { ui.camNextRow.style.display = "flex"; }

  ui.skipCam.addEventListener("click", function () {
    state.af = { outcome: "skipped", sets_af: null,
                 headline: "Pulse check skipped", detail: "" };
    state.estimates.afterCamera = R.estimate(state.profile, { n: 20000 });
    if (capture) capture.stop();
    showStep(3);
  });
  ui.toStep3.addEventListener("click", function () {
    if (capture) capture.stop();
    showStep(3);
  });

  /* ---------------------------------------------------------- step 3 ---- */
  ui.drop.addEventListener("click", function () { ui.fileInput.click(); });
  ui.fileInput.addEventListener("change", async function () {
    var f = ui.fileInput.files && ui.fileInput.files[0];
    if (!f) return;
    ui.ocrStatus.textContent = "Reading the image on your device…";
    ui.ocrResult.innerHTML = "";
    try {
      var out = await DOC.scanImage(f, window.MEDICATIONS, function (p) {
        ui.ocrStatus.textContent = "Reading… " + Math.round(p * 100) + "%";
      });
      ui.ocrStatus.textContent = "";
      state.docFindings = out;
      renderProposals(out);
    } catch (e) {
      ui.ocrStatus.textContent = "";
      ui.ocrResult.innerHTML = warn("", "Couldn't read that image",
        "The text reader failed (" + String(e && e.message || e) + "). You can try a "
      + "sharper, better-lit photo, or just continue — this step is optional.");
    }
  });

  function renderProposals(out) {
    if (out.nothing_found) {
      ui.ocrResult.innerHTML = warn("", "Nothing recognised",
        "No medicines or lab values were found in that image. Try a sharper photo of the "
      + "box front or the results table — or skip this step, it's optional.");
      return;
    }
    var html = "";
    out.proposals.forEach(function (p, i) {
      html += '<div class="proposal" data-i="' + i + '">'
            + '<div class="p-reason">' + p.reason + '</div>'
            + '<div class="p-q">' + p.prompt + '</div>'
            + '<div class="toggle prop-toggle" data-i="' + i + '">'
            + '<button type="button" data-v="yes">Yes</button>'
            + '<button type="button" data-v="no">No</button>'
            + '<button type="button" data-v="skip" class="unsure">Skip</button>'
            + '</div></div>';
    });
    ui.ocrResult.innerHTML = html;

    ui.ocrResult.querySelectorAll(".prop-toggle").forEach(function (g) {
      var idx = parseInt(g.getAttribute("data-i"), 10);
      g.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () {
          g.querySelectorAll("button").forEach(function (x) { x.classList.remove("active"); });
          b.classList.add("active");
          applyProposal(out.proposals[idx], b.getAttribute("data-v"));
        });
      });
    });
  }

  function applyProposal(p, answer) {
    state.docApplied = state.docApplied.filter(function (a) { return a.p !== p; });
    if (answer === "skip") return;
    state.docApplied.push({ p: p, answer: answer });

    if (!p.field) return;
    if (p.question_only) {
      // a hint the user has now confirmed or denied outright
      if (answer === "yes") state.profile[p.field] = true;
      else if (answer === "no" && state.profile[p.field] === null) state.profile[p.field] = false;
      return;
    }
    if (answer === "yes") {
      state.profile[p.field] = (p.value === null ? true : p.value);
    } else if (answer === "no" && p.value === true) {
      state.profile[p.field] = false;
    }
  }

  /* --------------------------------------------------------- results ---- */
  function fmt(x) { return x.toFixed(x < 1 ? 2 : 1) + "%"; }

  function bandFor(pct, floorPct) {
    var ratio = floorPct > 0 ? pct / floorPct : 1;
    if (ratio < 1.25) return ["Close to the floor for your age", "greenbg", "green"];
    if (ratio < 2.0) return ["Moderately above the floor for your age", "amberbg", "amber"];
    return ["Well above the floor for your age", "redbg", "red"];
  }

  ui.toResults.addEventListener("click", function () { renderResults(); showStep(4); });

  function renderResults() {
    var p = state.profile;
    var est = R.estimate(p, { n: 20000 });
    state.estimates.final = est;

    var cohort = 100 * window.RISK_MODEL.sets[window.RISK_MODEL.default_set]
                          .sexes[p.sex].cohort_avg_5yr_risk;
    var floorProfile = { age: p.age, sex: p.sex, sbp: 110, on_htn_meds: false,
                         diabetes: false, current_smoker: false, prevalent_cvd: false,
                         atrial_fib: false, smokeless_tobacco: false };
    var floorPct = 100 * R.risk(floorProfile);

    if (est.is_certain) {
      ui.riskDisplay.innerHTML = '<div class="risk-pct">' + fmt(est.point_pct) + "</div>";
      ui.riskLabel.textContent = "estimated probability of a first stroke in the next 5 years";
    } else {
      ui.riskDisplay.innerHTML = '<div class="risk-range">' + fmt(est.low_pct) + " – "
                               + fmt(est.high_pct) + "</div>";
      ui.riskLabel.textContent = "likely range for your 5-year stroke risk, given what you told us";
    }

    var b = bandFor(est.point_pct, floorPct);
    var cs = getComputedStyle(document.documentElement);
    ui.riskBand.textContent = b[0];
    ui.riskBand.style.background = cs.getPropertyValue("--" + b[1]);
    ui.riskBand.style.color = cs.getPropertyValue("--" + b[2]);

    var per1000 = Math.round(est.point_pct * 10);
    ui.plainly.innerHTML = "Out of 1,000 people who answered exactly as you did, about <b>"
      + per1000 + "</b> would be expected to have a stroke in the next five years, and about <b>"
      + (1000 - per1000) + "</b> would not."
      + (est.is_certain ? "" : " Because some answers are still unknown, the honest answer is a range.");

    ui.optimalVal.textContent = fmt(floorPct);
    ui.cohortVal.textContent = fmt(cohort);

    renderContributions();

    if (est.value_of_information && est.value_of_information.length) {
      ui.voiList.innerHTML = est.value_of_information.map(function (r) {
        var s = Math.round(100 * r.variance_share);
        return '<div class="voi-row"><div class="voi-bar"><i style="width:'
             + Math.max(2, s) + '%"></i></div><div class="voi-share">' + s + '%</div>'
             + '<div class="voi-text">Find out <b>' + r.label + '</b>'
             + '<span class="how">' + r.how_to_find_out + '</span></div></div>';
      }).join("");
      ui.voiPanel.style.display = "block";
    } else ui.voiPanel.style.display = "none";

    if (est.is_certain) {
      var cf = R.counterfactuals(p);
      if (cf.options.length) {
        ui.cfList.innerHTML = cf.options.map(function (o) {
          var h = '<div class="cf-row"><div class="cf-head"><b>' + o.change + "</b>";
          if (o.absolute_reduction_pct_points !== null)
            h += '<span class="cf-delta">' + fmt(o.new_risk_pct) + "</span>";
          h += "</div>";
          if (o.absolute_reduction_pct_points !== null)
            h += '<div class="cf-caveat">Would lower the estimate by '
               + o.absolute_reduction_pct_points.toFixed(2) + " percentage points ("
               + Math.round(100 * o.relative_reduction) + "% lower). " + (o.caveat || "") + "</div>";
          else if (o.caveat) h += '<div class="cf-caveat">' + o.caveat + "</div>";
          return h + "</div>";
        }).join("");
        ui.cfPanel.style.display = "block";
      } else ui.cfPanel.style.display = "none";
    } else ui.cfPanel.style.display = "none";

    var w = "";
    est.applicability.notes.forEach(function (n) { w += warn("", "Worth knowing", n); });
    if (est.applicability.status === "extrapolation") {
      w += warn("", "Outside the model's validated range",
        "Everyone used to build this model was 55 or older. Below that age the figure is an "
      + "extrapolation rather than a calibrated probability — treat it as a direction, not a number.");
    }
    if (state.af && state.af.outcome === "irregular") {
      w += warn("bad", "Your pulse reading was irregular",
        "This has deliberately NOT been counted as atrial fibrillation in the number above, "
      + "because a phone camera cannot confirm it. Please ask a doctor for an ECG. If AF were "
      + "confirmed, your 5-year risk would be materially higher than shown.");
    }
    ui.warnings.innerHTML = w;

    var afm = window.AF_MODEL;
    ui.modelBadge.innerHTML =
      "<b>How this was produced.</b> Risk: Cox proportional-hazards model, "
    + "1 &minus; S&#8320;(5)^exp(LP &minus; M), coefficients from "
    + window.RISK_MODEL.sets[window.RISK_MODEL.default_set].label + " ("
    + window.RISK_MODEL.citation.paper + ", <a href='"
    + window.RISK_MODEL.citation.open_access_full_text + "' target='_blank' rel='noopener'>open access</a>). "
    + "Published c-statistic 0.62&ndash;0.78 &mdash; useful for ranking, far from certainty for an "
    + "individual, and calibration is meaningfully worse for Black participants in the source data. "
    + "<br><br><b>Rhythm check:</b> logistic model on pulse-interval features, trained on the "
    + "MIT-BIH Atrial Fibrillation Database and validated leave-one-patient-out (AUROC "
    + afm.performance.auroc_record_held_out.toFixed(3) + ", sensitivity "
    + Math.round(100 * afm.performance.sensitivity_at_threshold) + "%, specificity "
    + Math.round(100 * afm.performance.specificity_at_threshold) + "%). Those figures come from "
    + "clinical ECG recordings, not from phone video of your finger, and this app's capture path "
    + "has not been validated against human subjects. That is why a regular reading is used to "
    + "rule AF out and an irregular one is only ever a referral."
    + "<br><br><b>Privacy:</b> the risk model, the rhythm analysis and the text recognition all run "
    + "in this browser. No answer, video frame or photograph is uploaded.";
  }

  function renderContributions() {
    var q = state.estimates.questionnaire, f = state.estimates.final;
    function span(e) {
      return e.is_certain ? fmt(e.point_pct) : fmt(e.low_pct) + "–" + fmt(e.high_pct);
    }
    var rows = [];

    rows.push({ on: true, title: "Questionnaire",
      note: "Age, sex, blood pressure and history gave a starting estimate of <b>"
          + span(q) + "</b>." });

    if (!state.af || state.af.outcome === "skipped") {
      rows.push({ on: false, title: "Pulse rhythm check — skipped",
        note: "Atrial fibrillation stayed unknown, which keeps the range wider than it needs to be." });
    } else if (state.af.outcome === "regular") {
      rows.push({ on: true, title: "Pulse rhythm check — regular",
        note: "Your pulse looked regular, so atrial fibrillation was ruled out"
            + (state.af.hr ? " (heart rate " + Math.round(state.af.hr) + " bpm)" : "") + "." });
    } else if (state.af.outcome === "irregular") {
      rows.push({ on: false, title: "Pulse rhythm check — irregular, not counted",
        note: "An irregular reading cannot confirm AF on a phone camera, so the estimate was "
            + "left unchanged rather than raised. Get an ECG." });
    } else {
      rows.push({ on: false, title: "Pulse rhythm check — unusable",
        note: "The signal wasn't clean enough to judge the rhythm, so nothing was changed." });
    }

    if (!state.docApplied.length) {
      rows.push({ on: false, title: "Documents — not used",
        note: "No medicines or lab values were confirmed." });
    } else {
      var applied = state.docApplied.filter(function (a) { return a.answer !== "skip"; });
      rows.push({ on: true, title: "Documents — " + applied.length + " confirmed",
        note: applied.map(function (a) {
          return a.p.reason + " → you answered <b>" + a.answer + "</b>";
        }).join("<br>") });
    }

    rows.push({ on: true, title: "Final estimate",
      note: "<b>" + span(f) + "</b>"
          + (span(f) === span(q) ? " — unchanged from the questionnaire alone."
                                 : " — refined from " + span(q) + " by the steps above.") });

    ui.contribList.innerHTML = rows.map(function (r) {
      return '<div class="contrib"><div class="dot' + (r.on ? "" : " off") + '"></div>'
           + '<div><b>' + r.title + '</b><span class="c-note">' + r.note + "</span></div></div>";
    }).join("");
  }

  ui.restart.addEventListener("click", function () { location.reload(); });
})();
