const $ = s => document.querySelector(s);
let state = null;
const urlp = new URLSearchParams(location.search);

function qs() {
  const q = new URLSearchParams();
  for (const k of ["as_of", "integrals", "fit", "candidate"])
    if (urlp.get(k) !== null) q.set(k, urlp.get(k));
  return q.toString();
}

async function load() {
  const res = await fetch("/api/state?" + qs());
  state = await res.json();
  render();
}

function jump(next) {
  const q = new URLSearchParams();
  if (next.as_of !== undefined) q.set("as_of", next.as_of);
  if (next.fit !== undefined) q.set("fit", next.fit);
  if (next.candidate !== undefined) q.set("candidate", next.candidate);
  history.replaceState(null, "", "/?" + q.toString());
  urlp.set("as_of", next.as_of ?? urlp.get("as_of"));
  for (const k of ["fit", "candidate", "as_of"]) {
    if (next[k] === null || next[k] === undefined) {
      if (k === "fit" || k === "candidate") urlp.delete(k);
    } else urlp.set(k, next[k]);
  }
  load();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"]/g, c =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
}
const fmt = v => v === null || v === undefined ? "—" : Number(v).toFixed(2);
const title = s => s ? `title="${esc(s)}"` : "";

function highlightInj(id) {
  document.querySelectorAll("[data-inj]").forEach(el =>
    el.classList.toggle("highlight", el.getAttribute("data-inj") === id));
  document.querySelectorAll("tr[data-inj]").forEach(el =>
    el.classList.toggle("active", el.getAttribute("data-inj") === id));
}

function render() {
  $("#case-title").textContent = state.case.title;
  $("#narrative").innerHTML = state.case.narrative
    .map(t => `<li>${esc(t)}</li>`).join("");
  $("#view-pos").textContent =
    `查看事件 ${state.as_of} / 最新 ${state.head_seq}`;
  renderMaterials();
  renderIntegrals();
  renderFits();
  renderDiagnostics();
  renderDecisions();
  renderTimeline();
  renderTargets();
}

function renderMaterials() {
  $("#materials-table").querySelector("tbody").innerHTML = state.materials
    .map(m => `<tr ${title(m.id)}><td>${m.kind}</td><td>v${m.version}</td>
      <td class="oid">${esc(m.id.slice(-12))}</td>
      <td>${m.current ? "●当前" : ""}</td></tr>`).join("");
}

function renderIntegrals() {
  const sel = state.selection.integrals;
  $("#power-svg").innerHTML = sel ? sel.power_svg : "<p>无积分结果</p>";
  $("#power-units").textContent = sel
    ? `单位: ${Object.entries(sel.units).map(([k, v]) => `${k}=${v}`).join(", ")}; 时间基准: ${sel.time_basis}` : "";
  $("#integrals-meta").textContent = sel
    ? `结果身份 ${sel.id} · 状态 ${sel.status}` +
      (sel.stale ? " · 已失效: " + sel.stale_reasons.join("; ") : "") : "";
  const tb = $("#integrals-table").querySelector("tbody");
  if (!sel) { tb.innerHTML = ""; return; }
  tb.innerHTML = sel.injections.map(i => `
    <tr data-inj="${i.injection_id}" class="${i.status !== "ok" ? "inj-err" : ""}">
      <td ${title(i.injection_id)}>${i.injection_id}</td>
      <td>${fmt(i.t_s)}</td>
      <td>${i.vol_ok ? fmt(i.vol_uL) : "原始:" + esc(i.vol_uL) + " (不可转换)"}</td>
      <td>${i.window_s ? i.window_s.map(fmt).join("–") : "—"}</td>
      <td>${i.heat_uJ === null ? "—" : fmt(i.heat_uJ)}</td>
      <td>${i.flags.join(",")}</td><td>${i.status}</td>
    </tr>`).join("");
  tb.querySelectorAll("tr").forEach(tr => tr.addEventListener(
    "click", () => highlightInj(tr.getAttribute("data-inj"))));
}

const MODEL_LABEL = {one_site: "一位点", two_site: "多位点(两位点)",
                     dilution_constant: "稀释假设(常量)"};

function renderFits() {
  const tabs = $("#model-tabs");
  const fits = state.selection.fits;
  tabs.innerHTML = Object.keys(fits).map(m => {
    const f = fits[m];
    const active = state.selection.fit &&
      state.selection.fit.id === f.id ? "active" : "";
    return `<button class="${active} ${f.stale ? "stale-tab" : ""}"
      data-fit="${f.id}" ${title(f.id + (f.stale ? " " + f.stale_reasons.join(";") : ""))}>
      ${MODEL_LABEL[m] || m}(${f.n_candidates}候选${f.stale ? ",过期" : ""})</button>`;
  }).join("");
  tabs.querySelectorAll("button").forEach(b => b.addEventListener(
    "click", () => jump({as_of: state.as_of, fit: b.dataset.fit, candidate: null})));
  const fv = state.selection.fit;
  if (!fv) { $("#candidate-box").innerHTML = ""; $("#residuals-svg").innerHTML = "";
             $("#points-table").querySelector("tbody").innerHTML = ""; return; }
  $("#fit-stale").textContent = fv.stale
    ? "该拟合已过期: " + fv.stale_reasons.join("; ") : "";
  const ci = fv.candidate_index || 0;
  $("#candidate-box").innerHTML =
    `<label>等优候选(${fv.candidates.length}, 按RSS稳定排序):
     <select id="cand-select">${fv.candidates.map((c, i) =>
       `<option value="${i}" ${i === ci ? "selected" : ""}
         ${title(c.candidate_id)}>
         ${i + 1}. ${esc(c.candidate_id)} RSS=${c.rss.toFixed(3)}
         ${Object.entries(c.params).map(([k, v]) => `${k}=${v.toFixed(3)}`).join(" ")}
       </option>`).join("")}</select></label>
     <span class="oid">${esc(fv.id)}</span>`;
  $("#cand-select").addEventListener("change", e =>
    jump({as_of: state.as_of, fit: fv.id, candidate: +e.target.value}));
  $("#residuals-svg").innerHTML = fv.residuals_svg;
  const pts = (fv.points || []).filter(p => p.include);
  $("#points-table").querySelector("tbody").innerHTML = pts.map(p => `
    <tr data-inj="${p.injection_id}" ${title("摩尔热派生自积分热/(体积×注射器浓度)")}>
      <td>${p.injection_id}</td><td>${fmt(p.q_kJmol)}</td>
      <td>${fmt(p.heat_uJ)}</td></tr>`).join("");
  $("#points-table").querySelectorAll("tr").forEach(tr => tr.addEventListener(
    "click", () => highlightInj(tr.getAttribute("data-inj"))));
}

function renderDiagnostics() {
  const diags = state.diagnostics.slice();
  const fv = state.selection.fit;
  if (fv && fv.diagnostics)
    diags.push(...fv.diagnostics.map(d => ({...d, source: fv.id})));
  $("#diagnostics").innerHTML = diags.length === 0
    ? "<li>无诊断</li>"
    : diags.map(d => `<li class="${d.severity === "error" ? "error" : ""}"
        ${title(d.source)}>[${d.code}] ${esc(d.message)}
        ${d.position ? "@" + esc(JSON.stringify(d.position)) : ""} <span class="oid">${esc((d.source || "").slice(-12))}</span></li>`).join("");
}

function renderDecisions() {
  $("#decisions").innerHTML = state.decisions.map(e => {
    const d = e.summary;
    return `<li ${title(e.id + " 父级: " + JSON.stringify(e.parents))}>
      #${e.seq} ${esc(d)}</li>`;
  }).join("");
}

function renderTimeline() {
  $("#timeline").innerHTML = state.events.map(e => {
    let cls = "evt";
    if (e.kind === "milestone") cls += " milestone";
    if (e.kind === "scope_check") cls += e.status === "ok"
      ? " scope-ok" : " scope-failed";
    if (e.seq === state.as_of) cls += " current";
    return `<span class="${cls}" data-seq="${e.seq}"
      ${title(e.id + (e.param_version ? " 参数版本 " + e.param_version : "")
        + " 父级 " + JSON.stringify(e.parents))}>
      ${e.seq}.${esc(e.kind.replace(/_/g, ""))}</span>`;
  }).join("");
  $("#timeline").querySelectorAll(".evt").forEach(el => el.addEventListener(
    "click", () => jump({as_of: +el.dataset.seq, fit: null, candidate: null})));
}

function renderTargets() {
  const sel = $("#dec-target");
  const fits = state.selection.fits;
  sel.innerHTML = Object.values(fits).map(f =>
    `<option value="result:${f.id}">${esc(f.id)}</option>`).join("")
    + (state.selection.integrals
       ? state.selection.integrals.injections.map(i =>
         `<option value="inj:${i.injection_id}">注射 ${i.injection_id}</option>`).join("")
       : "");
}

$("#btn-decide").addEventListener("click", async () => {
  const reviewer = $("#dec-reviewer").value.trim();
  const action = $("#dec-action").value;
  const target = $("#dec-target").value;
  const rationale = $("#dec-rationale").value;
  const body = {reviewer, action, rationale};
  if (target.startsWith("result:")) body.target_result = target.slice(7);
  else body.target_injection = target.slice(4);
  const res = await fetch("/api/decide", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)});
  if (!res.ok) { alert("决定被拒绝: " + (await res.text())); return; }
  urlp.set("as_of", String((await res.json()).head_seq));
  load();
});
$("#btn-head").addEventListener("click", () => {
  urlp.delete("as_of"); urlp.delete("fit"); urlp.delete("candidate"); load();
});
$("#btn-latest").addEventListener("click", () => {
  urlp.set("as_of", String(state.head_seq));
  urlp.delete("fit"); urlp.delete("candidate"); load();
});
$("#btn-export").addEventListener("click", () => { location.href = "/api/export"; });
$("#file-import").addEventListener("change", async e => {
  const text = await e.target.files[0].text();
  const res = await fetch("/api/import", {method: "POST", body: text});
  const body = await res.json();
  if (body.status === "format_corrupt") {
    alert("格式损坏(format_corrupt): " + body.detail);
  } else {
    urlp.set("as_of", String(body.head_seq)); load();
  }
  e.target.value = "";
});

load();
