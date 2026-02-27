// PorousPromptBO interactive GitHub Pages demo
// Runs a ridge surrogate (trained on the repo's simulator) in the browser.

let MODEL = null;
let BLOCKS = [];
let BLOCK_MAP = {};
let EXPERIMENTS = [];
let debounceTimer = null;



function guessGitHubContext(){
  try{
    const host = window.location.hostname || "";
    const path = (window.location.pathname || "/").split("/").filter(Boolean);
    if (host.endsWith("github.io") && path.length >= 1){
      const user = host.split(".")[0];
      const repo = path[0];
      return {user, repo};
    }
  }catch(e){}
  return null;
}

function updateRepoLink(fallbackRepo){
  const a = document.getElementById("repoLink");
  if (!a) return;
  const ctx = guessGitHubContext();
  if (ctx){
    const url = `https://github.com/${ctx.user}/${ctx.repo}`;
    a.href = url;
    a.textContent = url;
  }else if (fallbackRepo){
    a.href = `https://github.com/<your-username>/${fallbackRepo}`;
  }
}

function fmt(x, digits=2){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function clamp(x, lo, hi){
  return Math.min(Math.max(x, lo), hi);
}

function dot(a,b){
  let s = 0.0;
  for (let i=0;i<a.length;i++) s += a[i]*b[i];
  return s;
}

async function loadJSON(path){
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
  return await r.json();
}

function stepFor(id){
  const map = {
    "temperature_C": 1,
    "concentration_M": 0.01
  };
  return map[id] ?? 0.1;
}

function setSlider(id, min, max, value){
  const el = document.getElementById(id);
  el.min = String(min);
  el.max = String(max);
  el.step = String(stepFor(id));
  el.value = String(value);
}

function updateValLabel(id){
  const el = document.getElementById(id);
  const out = document.getElementById(`${id}_val`);
  if (!el || !out) return;
  const v = el.tagName.toLowerCase() === "select" ? el.value : parseFloat(el.value);
  out.textContent = (typeof v === "number") ? fmt(v, stepFor(id) < 0.1 ? 2 : 0) : String(v);
}

function median(arr){
  const a = [...arr].sort((x,y) => x-y);
  if (a.length === 0) return NaN;
  const mid = Math.floor(a.length/2);
  return (a.length % 2) ? a[mid] : 0.5*(a[mid-1]+a[mid]);
}

function setStats(){
  document.getElementById("statN").textContent = String(EXPERIMENTS.length);
  document.getElementById("statBlocks").textContent = String(BLOCKS.length);
  const r2vals = Object.values(MODEL.r2_test ?? {});
  const med = median(r2vals);
  document.getElementById("statR2").textContent = Number.isFinite(med) ? fmt(med, 2) : "—";
}

function updateModelInfo(){
  const lines = [];
  lines.push(`<b>${MODEL.name}</b>`);
  if (MODEL.note) lines.push(MODEL.note);
  if (MODEL.r2_test){
    const parts = [];
    for (const [k,v] of Object.entries(MODEL.r2_test)){
      parts.push(`${k}: ${fmt(v, 2)}`);
    }
    lines.push(`Test R² — ${parts.join(" · ")}`);
  }
  document.getElementById("modelInfo").innerHTML = lines.join("<br/>");
}

function blockSummary(b){
  if (!b) return "—";
  return `${b.block_id} · MW ${fmt(b.mw, 1)} · aromatic ${b.aromatic_rings} · flex ${fmt(b.flexibility, 3)}`;
}

function updateBlockInfo(){
  const A = BLOCK_MAP[document.getElementById("block_A").value];
  const B = BLOCK_MAP[document.getElementById("block_B").value];
  const html = [
    `<b>Selected descriptors (used by model)</b>`,
    `A: ${blockSummary(A)}`,
    `B: ${blockSummary(B)}`
  ].join("<br/>");
  document.getElementById("blockInfo").innerHTML = html;
}

function buildFeatureObject(){
  const A = BLOCK_MAP[document.getElementById("block_A").value];
  const B = BLOCK_MAP[document.getElementById("block_B").value];
  const solvent = document.getElementById("solvent").value;
  const catalyst = document.getElementById("catalyst").value;
  const T = parseFloat(document.getElementById("temperature_C").value);
  const C = parseFloat(document.getElementById("concentration_M").value);

  const f = {};
  f["aromatic_sum"] = (A.aromatic_rings + B.aromatic_rings);
  f["flex_avg"] = 0.5*(A.flexibility + B.flexibility);
  f["mw_sum"] = (A.mw + B.mw);
  f["temperature_C"] = T;
  f["temperature_C2"] = T*T;
  f["concentration_M"] = C;
  f["conc_high"] = Math.max(C - 0.25, 0.0);

  for (const s of MODEL.solvents){
    f[`solvent_${s}`] = (solvent === s) ? 1.0 : 0.0;
  }
  for (const c of MODEL.catalysts){
    f[`catalyst_${c}`] = (catalyst === c) ? 1.0 : 0.0;
  }
  return f;
}

function featureVectorFromObject(fobj){
  const x = [];
  for (const name of MODEL.feature_names){
    x.push(Number(fobj[name] ?? 0.0));
  }
  return x;
}

function predictFromVector(x){
  const z = x.map((v,i) => (v - MODEL.scaler.mean[i]) / MODEL.scaler.scale[i]);
  const out = {};
  for (const t of MODEL.targets){
    const m = MODEL.models[t];
    let y = m.intercept + dot(m.coef, z);
    const r = MODEL.output_ranges?.[t];
    if (r){
      y = clamp(y, r.min, r.max);
    }
    out[t] = y;
  }
  return out;
}

function heuristicScore(pred){
  // Prefer high surface area and crystallinity; keep yield acceptable
  function norm(v, lo, hi){
    return (v - lo) / (hi - lo + 1e-9);
  }
  const r = MODEL.output_ranges;
  const sa = norm(pred.surface_area_m2_g, r.surface_area_m2_g.min, r.surface_area_m2_g.max);
  const cr = norm(pred.crystallinity_score, r.crystallinity_score.min, r.crystallinity_score.max);
  const y  = norm(pred.yield_pct, r.yield_pct.min, r.yield_pct.max);

  // Soft penalty if yield < ~50%
  const yieldPenalty = pred.yield_pct < 50 ? 0.15 : 0.0;
  return 0.50*sa + 0.35*cr + 0.15*y - yieldPenalty;
}

function renderPredictions(pred){
  document.getElementById("out_yield_pct").textContent = fmt(pred.yield_pct, 1);
  document.getElementById("out_surface_area_m2_g").textContent = fmt(pred.surface_area_m2_g, 0);
  document.getElementById("out_crystallinity_score").textContent = fmt(pred.crystallinity_score, 3);
  document.getElementById("out_score").textContent = fmt(heuristicScore(pred), 3);
}

function predictAndRender(){
  const fobj = buildFeatureObject();
  const x = featureVectorFromObject(fobj);
  const pred = predictFromVector(x);
  renderPredictions(pred);
}

function predictDebounced(){
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => predictAndRender(), 120);
}

function suggestExperiment(){
  const best = {score: -1e9, A: null, B: null, solvent: null, catalyst: null, T: null, C: null, pred: null};

  for (let i=0;i<500;i++){
    const A = BLOCKS[Math.floor(Math.random()*BLOCKS.length)].block_id;
    const B = BLOCKS[Math.floor(Math.random()*BLOCKS.length)].block_id;
    const solvent = MODEL.solvents[Math.floor(Math.random()*MODEL.solvents.length)];
    const catalyst = MODEL.catalysts[Math.floor(Math.random()*MODEL.catalysts.length)];
    const T = MODEL.input_ranges.temperature_C.min + Math.random()*(MODEL.input_ranges.temperature_C.max - MODEL.input_ranges.temperature_C.min);
    const C = MODEL.input_ranges.concentration_M.min + Math.random()*(MODEL.input_ranges.concentration_M.max - MODEL.input_ranges.concentration_M.min);

    // quick feasibility gate (demo): avoid "none" catalyst at low temp (slow)
    if (catalyst === "none" && T < 55) continue;

    const fobj = {
      ...(() => {
        const a = BLOCK_MAP[A], b = BLOCK_MAP[B];
        const o = {};
        o["aromatic_sum"] = a.aromatic_rings + b.aromatic_rings;
        o["flex_avg"] = 0.5*(a.flexibility + b.flexibility);
        o["mw_sum"] = a.mw + b.mw;
        o["temperature_C"] = T;
        o["temperature_C2"] = T*T;
        o["concentration_M"] = C;
        o["conc_high"] = Math.max(C - 0.25, 0.0);
        for (const s of MODEL.solvents) o[`solvent_${s}`] = (solvent === s) ? 1.0 : 0.0;
        for (const c2 of MODEL.catalysts) o[`catalyst_${c2}`] = (catalyst === c2) ? 1.0 : 0.0;
        return o;
      })()
    };

    const x = featureVectorFromObject(fobj);
    const pred = predictFromVector(x);
    const score = heuristicScore(pred);

    if (score > best.score){
      best.score = score;
      best.A = A; best.B = B; best.solvent = solvent; best.catalyst = catalyst; best.T = T; best.C = C;
      best.pred = pred;
    }
  }

  if (!best.A){
    alert("No feasible candidate found. Try again.");
    return;
  }

  document.getElementById("block_A").value = best.A;
  document.getElementById("block_B").value = best.B;
  document.getElementById("solvent").value = best.solvent;
  document.getElementById("catalyst").value = best.catalyst;
  document.getElementById("temperature_C").value = String(best.T);
  document.getElementById("concentration_M").value = String(best.C);

  ["block_A","block_B","solvent","catalyst","temperature_C","concentration_M"].forEach(updateValLabel);
  updateBlockInfo();
  renderPredictions(best.pred);

  const mi = document.getElementById("modelInfo");
  mi.innerHTML = mi.innerHTML + `<br/><span style="color:var(--accent2)">Suggested candidate score: ${fmt(best.score, 3)}</span>`;
}

function renderTable(rows){
  const tb = document.querySelector("#expTable tbody");
  tb.innerHTML = "";
  const show = rows.slice(0, 140);
  for (const r of show){
    const tr = document.createElement("tr");
    const cells = [
      r.block_A,
      r.block_B,
      r.solvent,
      r.catalyst,
      fmt(r.temperature_C, 1),
      fmt(r.concentration_M, 3),
      fmt(r.yield_pct, 1),
      fmt(r.surface_area_m2_g, 0),
      fmt(r.crystallinity_score, 3),
    ];
    for (const c of cells){
      const td = document.createElement("td");
      td.textContent = String(c);
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
}

function initControls(){
  // populate block selects
  const selA = document.getElementById("block_A");
  const selB = document.getElementById("block_B");
  selA.innerHTML = ""; selB.innerHTML = "";
  for (const b of BLOCKS){
    const optA = document.createElement("option");
    optA.value = b.block_id;
    optA.textContent = `${b.block_id} — ${b.block_name}`;
    selA.appendChild(optA);

    const optB = document.createElement("option");
    optB.value = b.block_id;
    optB.textContent = `${b.block_id} — ${b.block_name}`;
    selB.appendChild(optB);
  }

  // solvents/catalysts
  const selS = document.getElementById("solvent");
  selS.innerHTML = "";
  for (const s of MODEL.solvents){
    const opt = document.createElement("option");
    opt.value = s; opt.textContent = s;
    selS.appendChild(opt);
  }

  const selC = document.getElementById("catalyst");
  selC.innerHTML = "";
  for (const c of MODEL.catalysts){
    const opt = document.createElement("option");
    opt.value = c; opt.textContent = c;
    selC.appendChild(opt);
  }

  // default values
  selA.value = BLOCKS[0].block_id;
  selB.value = BLOCKS[Math.min(1, BLOCKS.length-1)].block_id;
  selS.value = MODEL.solvents[0];
  selC.value = MODEL.catalysts[0];

  setSlider("temperature_C", MODEL.input_ranges.temperature_C.min, MODEL.input_ranges.temperature_C.max, 75);
  setSlider("concentration_M", MODEL.input_ranges.concentration_M.min, MODEL.input_ranges.concentration_M.max, 0.20);

  ["block_A","block_B","solvent","catalyst","temperature_C","concentration_M"].forEach(id => updateValLabel(id));

  // listeners
  selA.addEventListener("change", () => { updateValLabel("block_A"); updateBlockInfo(); predictDebounced(); });
  selB.addEventListener("change", () => { updateValLabel("block_B"); updateBlockInfo(); predictDebounced(); });
  selS.addEventListener("change", () => { updateValLabel("solvent"); predictDebounced(); });
  selC.addEventListener("change", () => { updateValLabel("catalyst"); predictDebounced(); });

  document.getElementById("temperature_C").addEventListener("input", () => { updateValLabel("temperature_C"); predictDebounced(); });
  document.getElementById("concentration_M").addEventListener("input", () => { updateValLabel("concentration_M"); predictDebounced(); });

  document.getElementById("predictBtn").addEventListener("click", () => predictAndRender());
  document.getElementById("suggestBtn").addEventListener("click", () => suggestExperiment());
  document.getElementById("resetBtn").addEventListener("click", () => {
    initControls();
    updateBlockInfo();
    updateModelInfo();
    predictAndRender();
  });

  document.getElementById("filterBtn").addEventListener("click", () => {
    const q = (document.getElementById("filterInput").value || "").trim().toLowerCase();
    if (!q){ renderTable(EXPERIMENTS); return; }
    const rows = EXPERIMENTS.filter(r => {
      const s = `${r.block_A} ${r.block_B} ${r.solvent} ${r.catalyst}`.toLowerCase();
      return s.includes(q);
    });
    renderTable(rows);
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    document.getElementById("filterInput").value = "";
    renderTable(EXPERIMENTS);
  });
}

async function main(){
  try{
    MODEL = await loadJSON("assets/model.json");
    const b = await loadJSON("assets/building_blocks.json");
    const e = await loadJSON("assets/experiments.json");

    BLOCKS = b.blocks ?? [];
    BLOCK_MAP = {};
    for (const bb of BLOCKS){
      BLOCK_MAP[bb.block_id] = bb;
    }
    EXPERIMENTS = e.experiments ?? [];

    setStats();
    updateModelInfo();
    initControls();
    updateBlockInfo();
    predictAndRender();
    renderTable(EXPERIMENTS);
    updateRepoLink("porouspromptbo");

  }catch(err){
    console.error(err);
    document.getElementById("modelInfo").textContent = "Failed to load assets. Check console.";
  }
}

main();
