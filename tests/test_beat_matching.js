/*
 * Self-test for the beat-matching used by tests/validate_butppg.js.
 *
 * Written after a harness bug produced a completely false validation result: RR error
 * was computed by subtracting RR sequences element-wise, with no beat matching, so a
 * single missed beat shifted the whole sequence and the reported "error" was simply the
 * length of one heartbeat (~267 ms). Every number in that report was meaningless.
 *
 * These cases use synthetic beats with KNOWN ground truth, so the measuring instrument
 * is checked before it is used to measure anything. A second bug was caught here too:
 * the lag estimator tie-broke on the wrong criterion and reported 155 ms for a true
 * 250 ms transit lag.
 *
 *   node tests/test_beat_matching.js
 */
const MATCH_TOL = 0.10;
function bestLag(ppgT, ecgT) {
    const rr=[]; for(let i=1;i<ecgT.length;i++) rr.push(ecgT[i]-ecgT[i-1]);
    rr.sort((a,b)=>a-b);
    const medRR = rr.length ? rr[Math.floor(rr.length/2)] : 0.8;
    const span = Math.max(0.8, 1.2*medRR);
    let best = {lag: 0, n: -1, cost: Infinity};
    for (let lag = -span; lag <= span; lag += 0.005) {
        let n = 0, cost = 0;
        for (const p of ppgT) { const t = p - lag;
            let bd = Infinity;
            for (const e of ecgT) { const d = Math.abs(e - t); if (d < bd) bd = d; }
            if (bd <= MATCH_TOL) { n++; cost += bd; } }
        if (n > best.n || (n === best.n && cost < best.cost)) best = {lag, n, cost};
    }
    return best;
}
function matchBeats(ppgT, ecgT, lag) {
    const used = new Set(), pairs = [];
    ppgT.forEach((p, pi) => { const t = p - lag;
        let bi = -1, bd = Infinity;
        ecgT.forEach((e, ei) => { if (used.has(ei)) return;
            const d = Math.abs(e - t); if (d < bd) { bd = d; bi = ei; } });
        if (bi >= 0 && bd <= MATCH_TOL) { used.add(bi); pairs.push({pi, ei: bi, d: bd}); } });
    pairs.sort((a,b)=>a.pi-b.pi);
    return pairs;
}
function rrError(ppgT, ecgT, pairs) {
    const errs = [];
    for (let i = 1; i < pairs.length; i++) {
        const a = pairs[i-1], b = pairs[i];
        if (b.ei !== a.ei + 1) continue;
        errs.push((ppgT[b.pi]-ppgT[a.pi])*1000 - (ecgT[b.ei]-ecgT[a.ei])*1000);
    }
    return errs.length ? errs.reduce((s,e)=>s+Math.abs(e),0)/errs.length : NaN;
}
let pass=0, fail=0;
function check(name, ok, detail) {
    if (ok) {pass++; console.log(`[PASS] ${name}${detail?"  -- "+detail:""}`);}
    else {fail++; console.error(`[FAIL] ${name}${detail?"  -- "+detail:""}`);}
}

// ground truth: 14 ECG beats, irregular-ish intervals
const ecg = [0.5]; const rrs=[0.80,0.75,0.82,0.78,0.85,0.72,0.79,0.83,0.76,0.81,0.77,0.84,0.80];
rrs.forEach(r => ecg.push(ecg[ecg.length-1]+r));

// case 1: perfect PPG, constant 250ms transit lag
let ppg = ecg.map(t => t + 0.25);
let {lag} = bestLag(ppg, ecg); let pairs = matchBeats(ppg, ecg, lag);
check("recovers a constant pulse transit lag", Math.abs(lag-0.25) < 0.01, `lag=${(lag*1000).toFixed(0)}ms`);
check("matches every beat when PPG is perfect", pairs.length === ecg.length, `${pairs.length}/${ecg.length}`);
check("RR error is ~0 for a perfect PPG", rrError(ppg,ecg,pairs) < 0.5, `${rrError(ppg,ecg,pairs).toFixed(3)} ms`);

// case 2: one beat MISSED - the failure the old harness mis-scored as a huge RR error
ppg = ecg.map(t=>t+0.25).filter((_,i)=>i!==5);
({lag} = bestLag(ppg, ecg)); pairs = matchBeats(ppg, ecg, lag);
check("a missed beat lowers sensitivity, not RR accuracy",
  pairs.length === ecg.length-1 && rrError(ppg,ecg,pairs) < 0.5,
  `sens=${(100*pairs.length/ecg.length).toFixed(0)}%  RRerr=${rrError(ppg,ecg,pairs).toFixed(2)}ms`);

// case 3: one SPURIOUS extra peak
ppg = ecg.map(t=>t+0.25); ppg.splice(7, 0, ecg[7]+0.25-0.33); ppg.sort((a,b)=>a-b);
({lag} = bestLag(ppg, ecg)); pairs = matchBeats(ppg, ecg, lag);
check("a spurious peak lowers PPV", pairs.length < ppg.length,
  `ppv=${(100*pairs.length/ppg.length).toFixed(0)}%`);

// case 4: real jitter of a known size must be recovered as that size
const JIT = 12;
let seed=7; const rnd=()=>{seed=(seed*1103515245+12345)&0x7fffffff; return seed/0x7fffffff-0.5;};
ppg = ecg.map(t => t + 0.25 + (JIT/1000)*rnd()*2);
({lag} = bestLag(ppg, ecg)); pairs = matchBeats(ppg, ecg, lag);
const e = rrError(ppg,ecg,pairs);
check(`recovers a known ${JIT}ms jitter as the right magnitude`, e > 2 && e < 4*JIT, `measured ${e.toFixed(1)} ms`);

// case 5: THE OLD BUG - naive element-wise comparison must look far worse
const naive = (()=>{const a=[],b=[];for(let i=1;i<ppg.length;i++)a.push((ppg[i]-ppg[i-1])*1000);
  for(let i=1;i<ecg.length;i++)b.push((ecg[i]-ecg[i-1])*1000);
  const shifted=a.slice(1); const k=Math.min(shifted.length,b.length); let s=0;
  for(let i=0;i<k;i++) s+=Math.abs(shifted[i]-b[i]); return s/k;})();
check("the old element-wise method inflates the error several-fold",
  naive > 3*e, `matched ${e.toFixed(1)}ms vs unmatched ${naive.toFixed(1)}ms`);

// case 6: a large NEGATIVE offset, as seen in real BUT PPG records (-243 ms observed).
// A lag window that does not reach this silently fails to match the whole recording,
// which is what collapsed measured beat sensitivity to 41.7% in the first real run.
ppg = ecg.map(t => t - 0.243);
({lag} = bestLag(ppg, ecg)); pairs = matchBeats(ppg, ecg, lag);
check("matches a large NEGATIVE offset (-243ms, seen in real data)",
  pairs.length === ecg.length && Math.abs(lag + 0.243) < 0.01,
  `lag=${(lag*1000).toFixed(0)}ms  matched ${pairs.length}/${ecg.length}`);
check("a constant offset does not corrupt RR intervals", rrError(ppg,ecg,pairs) < 0.5,
  `${rrError(ppg,ecg,pairs).toFixed(3)} ms`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail?1:0);
