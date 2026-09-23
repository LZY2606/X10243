const $ = (s) => document.querySelector(s);
const SVGNS = "http://www.w3.org/2000/svg";
let AS_OF = null;
let SELECTED_CAND = null;

function el(tag, attrs, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
const fmt = (v, d = 3) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d);

async function load() {
  const branch = $("#branch").value;
  const qs = new URLSearchParams();
  if (AS_OF !== null) qs.set("as_of", AS_OF);
  if (branch) qs.set("branch", branch);
  const [state, curve] = await Promise.all([
    fetch("/api/state?" + qs).then(r => r.json()),
    fetch("/api/curve?" + qs).then(r => r.ok ? r.json() : null),
  ]);
  renderTimeline(state);
  renderCurve(state, curve);
  renderInjections(state);
  renderFit(state);
  renderDiagnostics(state);
  renderMaterials(state);
  renderDecisions(state);
  $("#asof-label").textContent =
    `视图截至事件 evt-${String(state.as_of).padStart(4, "0")} / 最新 evt-${String(state.latest_seq).padStart(4, "0")}`;
}

function renderTimeline(state) {
  const box = $("#timeline");
  box.innerHTML = "";
  for (const e of state.events) {
    const d = document.createElement("span");
    d.className = "evt" + (e.seq === state.as_of ? " active" : "");
    d.innerHTML = `<span class="seq">${e.seq}</span>${e.type}`;
    d.title = `${e.id} ← 父事件: ${e.parents.join(", ") || "无"}`;
    d.onclick = () => { AS_OF = (AS_OF === e.seq ? null : e.seq); SELECTED_CAND = null; load(); };
    box.appendChild(d);
  }
}

function renderCurve(state, curve) {
  const svg = $("#curve");
  svg.innerHTML = "";
  $("#curve-meta").textContent = curve
    ? `mat-power v${curve.power_version} · 单位 ${curve.power_unit} · 时间基准 ${curve.time_base}` : "";
  if (!curve) return;
  const W = 960, H = 260, PAD = 34;
  let tMin = Infinity, tMax = -Infinity, pMin = Infinity, pMax = -Infinity;
  for (const seg of curve.segments) {
    tMin = Math.min(tMin, seg.t[0]); tMax = Math.max(tMax, seg.t[seg.t.length - 1]);
    for (const p of seg.p) { pMin = Math.min(pMin, p); pMax = Math.max(pMax, p); }
  }
  const x = (t) => PAD + (t - tMin) / (tMax - tMin) * (W - 2 * PAD);
  const y = (p) => H - PAD - (p - pMin) / (pMax - pMin || 1) * (H - 2 * PAD);
  for (const [g0, g1] of curve.gaps) {
    svg.appendChild(el("rect", { x: x(g0), y: PAD, width: Math.max(x(g1) - x(g0), 2),
      height: H - 2 * PAD, fill: "#f6d7d4" }));
    const label = el("text", { x: x(g0) + 3, y: PAD + 12, "font-size": 10, fill: "#b3261e" },
      `数据缺口 ${g0}–${g1} s`);
    svg.appendChild(label);
  }
  for (const w of curve.windows) {
    const color = w.status !== "ok" ? "rgba(179,38,30,0.18)"
      : w.excluded ? "rgba(107,118,134,0.15)" : "rgba(11,107,203,0.10)";
    const r = el("rect", { x: x(w.window[0]), y: PAD, width: x(w.window[1]) - x(w.window[0]),
      height: H - 2 * PAD, fill: color, "data-inj": w.index });
    r.classList.add("win");
    svg.appendChild(r);
  }
  for (const seg of curve.segments) {
    const pts = seg.t.map((t, i) => `${x(t)},${y(seg.p[i])}`).join(" ");
    svg.appendChild(el("polyline", { points: pts, fill: "none",
      stroke: "#1c2430", "stroke-width": 1 }));
  }
  for (const inj of state.injections) {
    if (inj.heat_ucal === null) continue;
    const c = el("circle", { cx: x(inj.time), cy: y(inj.heat_ucal / 90 + 0.6),
      r: 3.5, fill: inj.excluded ? "#6b7686" : "#0b6bcb", "data-inj": inj.index });
    c.innerHTML = `<title>inj-${String(inj.index).padStart(2, "0")} · ${fmt(inj.heat_ucal)} µcal</title>`;
    svg.appendChild(c);
  }
  svg.appendChild(el("text", { x: PAD, y: H - 6, "font-size": 10, fill: "#6b7686" },
    `t / ${curve.time_unit}（${curve.time_base}）`));
  svg.appendChild(el("text", { x: 4, y: PAD, "font-size": 10, fill: "#6b7686" },
    curve.power_unit));
  $("#curve-legend").textContent =
    "蓝底=有效积分窗 · 灰底=舍弃（首注） · 红底=积分窗跨越数据缺口（不积分） · 红带=数据缺口";
}

function renderInjections(state) {
  const tb = $("#injections tbody");
  tb.innerHTML = "";
  const integ = state.integration;
  $("#integration-meta").innerHTML = integ
    ? `<span class="id">${integ.id}</span> 状态 <b class="status-${integ.status}">${integ.status}</b>` +
      (integ.stale ? ' <span class="badge stale">已过期</span>' : "") +
      ` · 参数 <code>${JSON.stringify(integ.params)}</code>` : "";
  for (const inj of state.injections) {
    const tr = document.createElement("tr");
    tr.dataset.inj = inj.index;
    if (inj.status !== "ok") tr.classList.add("invalid");
    if (inj.excluded) tr.classList.add("excluded");
    const reasons = [inj.exclude_reason, ...inj.diagnostics.map(d => d.code)]
      .filter(Boolean).join("; ") || (inj.peak_overlap ? "峰重叠(允许)" : "ok");
    tr.innerHTML =
      `<td><span class="id">${inj.id}</span></td><td>${fmt(inj.time, 0)}</td>` +
      `<td>${fmt(inj.volume, 1)}</td><td>[${fmt(inj.window[0], 0)}, ${fmt(inj.window[1], 0)}]</td>` +
      `<td>${fmt(inj.heat_ucal)}</td><td>${fmt(inj.blank_ucal)}</td>` +
      `<td>${fmt(inj.corrected_ucal)}</td><td>${reasons}</td>`;
    tb.appendChild(tr);
  }
  linkHover();
}

function renderFit(state) {
  const box = $("#candidates");
  box.innerHTML = "";
  const fit = state.fit;
  if (!fit || !fit.payload.candidates) {
    $("#fit-meta").textContent = fit ? `状态 ${fit.status}` : "尚无拟合";
    return;
  }
  $("#fit-meta").innerHTML =
    `<span class="id">${fit.id}</span>` +
    (fit.stale ? ' <span class="badge stale">已过期：依赖已修订</span>' : "") +
    ` · 求值 ${fit.payload.evaluations}/${fit.payload.budget}`;
  if (!SELECTED_CAND || !fit.payload.candidates.some(c => c.id === SELECTED_CAND))
    SELECTED_CAND = fit.payload.candidates[0].id;
  for (const cand of fit.payload.candidates) {
    const div = document.createElement("div");
    div.className = "cand" + (cand.id === SELECTED_CAND ? " hl" : "");
    const params = Object.entries(cand.params)
      .map(([k, v]) => `${k}=${v}`).join("  ");
    const dec = cand.decision
      ? `<span class="badge dec-${cand.decision.action}">${cand.decision.action} @${cand.decision.branch}</span>` : "";
    div.innerHTML =
      `<div class="row"><div><b>${cand.family}</b>` +
      (cand.equipotent ? '<span class="badge eq">等优</span>' : "") + dec +
      `<div class="params">${cand.id} · SSR=${fmt(cand.ssr, 4)} · rank ${cand.rank}</div>` +
      `<div class="params">${params}</div></div>` +
      `<div><button data-act="accept">接受</button> <button data-act="reject">否定</button></div></div>`;
    div.onclick = (e) => {
      if (e.target.tagName === "BUTTON") return;
      SELECTED_CAND = cand.id; load();
    };
    for (const btn of div.querySelectorAll("button"))
      btn.onclick = async () => {
        const rationale = prompt(`对 ${cand.id} 的${btn.dataset.act === "accept" ? "接受" : "否定"}理由：`) || "";
        await fetch("/api/decision", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ branch: $("#branch").value || "main",
            target: cand.id, action: btn.dataset.act, rationale }) });
        load();
      };
    box.appendChild(div);
  }
  renderResiduals(state, fit.payload.candidates.find(c => c.id === SELECTED_CAND));
}

function renderResiduals(state, cand) {
  const svg = $("#residuals");
  svg.innerHTML = "";
  if (!cand || !cand.residuals) return;
  const W = 960, H = 180, PAD = 30;
  const entries = Object.entries(cand.residuals).map(([i, r]) => [Number(i), r]);
  if (!entries.length) return;
  const iMin = Math.min(...entries.map(e => e[0]));
  const iMax = Math.max(...entries.map(e => e[0]));
  const rMax = Math.max(0.5, ...entries.map(e => Math.abs(e[1])));
  const x = (i) => PAD + (i - iMin) / ((iMax - iMin) || 1) * (W - 2 * PAD);
  const y = (r) => H / 2 - r / rMax * (H / 2 - PAD);
  svg.appendChild(el("line", { x1: PAD, x2: W - PAD, y1: y(0), y2: y(0), stroke: "#d7dde5" }));
  for (const [i, r] of entries) {
    svg.appendChild(el("line", { x1: x(i), x2: x(i), y1: y(0), y2: y(r),
      stroke: "#0b6bcb", "stroke-width": 1.5 }));
    const c = el("circle", { cx: x(i), cy: y(r), r: 4, fill: "#0b6bcb", "data-inj": i });
    c.innerHTML = `<title>inj-${String(i).padStart(2, "0")} 残差 ${fmt(r, 4)} µcal</title>`;
    svg.appendChild(c);
    svg.appendChild(el("text", { x: x(i) - 8, y: H - 6, "font-size": 9, fill: "#6b7686" },
      `inj-${String(i).padStart(2, "0")}`));
  }
  svg.appendChild(el("text", { x: 4, y: 14, "font-size": 10, fill: "#6b7686" },
    `拟合残差 (µcal) · ${cand.id}`));
  linkHover();
}

function renderDiagnostics(state) {
  const ul = $("#diagnostics");
  ul.innerHTML = "";
  const seen = new Set();
  for (const d of state.diagnostics) {
    const key = d.code + JSON.stringify(d.location);
    if (seen.has(key)) continue;
    seen.add(key);
    const li = document.createElement("li");
    li.innerHTML = `<code>${d.code}</code> @ <code>${JSON.stringify(d.location)}</code> — ${d.message}`;
    ul.appendChild(li);
  }
  if (!ul.children.length) ul.innerHTML = "<li class='muted'>无诊断</li>";
}

function renderMaterials(state) {
  const ul = $("#materials");
  ul.innerHTML = "";
  for (const m of state.materials) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="id">${m.id} v${m.version}</span> · ${m.kind} · 来自事件 <span class="id">${m.event_id}</span>`;
    ul.appendChild(li);
  }
}

function renderDecisions(state) {
  const ul = $("#decisions");
  ul.innerHTML = "";
  for (const d of state.decisions) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="id">${d.id}</span> [${d.branch}] <b>${d.action}</b> ` +
      `<span class="id">${d.target}</span> — ${d.rationale || "（无理由）"} ` +
      `<span class="muted">事件 ${d.event_id}</span>`;
    ul.appendChild(li);
  }
  if (!ul.children.length) ul.innerHTML = "<li class='muted'>当前分支尚无决定</li>";
}

function linkHover() {
  const rows = document.querySelectorAll("#injections tbody tr");
  const marks = document.querySelectorAll("svg [data-inj]");
  rows.forEach(tr => {
    tr.onmouseenter = () => marks.forEach(m => {
      if (m.dataset.inj === tr.dataset.inj) m.setAttribute("r", 7);
    });
    tr.onmouseleave = () => marks.forEach(m => {
      if (m.dataset.inj === tr.dataset.inj) m.setAttribute("r", 4);
    });
  });
}

$("#branch").onchange = () => load();
$("#btn-export").onclick = async () => {
  const data = await fetch("/api/export").then(r => r.json());
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" }));
  a.download = "itc-review-export.json";
  a.click();
};
$("#btn-import").onclick = () => $("#file-import").click();
$("#file-import").onchange = async (e) => {
  const text = await e.target.files[0].text();
  const resp = await fetch("/api/import", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: text });
  const out = await resp.json();
  alert(out.status === "ok" ? `导入完成：${JSON.stringify(out)}` : `导入失败：${out.status} ${out.detail}`);
  load();
};
$("#btn-reset").onclick = async () => {
  await fetch("/api/reset", { method: "POST" });
  AS_OF = null; SELECTED_CAND = null; load();
};
load();
