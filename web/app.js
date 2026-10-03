// CDU-100 Digital Twin — browser client
// 3D plant (Blender GLB assets + 2K PBR textures) · live telemetry over /ws · asset panel · LangGraph assistant
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { createSky } from "./sky.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { OutlinePass } from "three/addons/postprocessing/OutlinePass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const $ = (s) => document.querySelector(s);
const STATUS = { OK: "#34c77b", STANDBY: "#5b8cff", WARN: "#f5a524", ALARM: "#ff4d4f", TRIPPED: "#ff2d55" };
const UNITS = { vibration_mm_s: "mm/s", vib_1x_mm_s: "mm/s", vib_2x_mm_s: "mm/s", vib_axial_mm_s: "mm/s", vib_bpfo_mm_s: "mm/s", vib_hf_env_g: "g", vib_broadband_mm_s: "mm/s", suction_bar: "bar", bearing_temp_c: "°C", motor_current_a: "A", discharge_bar: "bar", flow_m3h: "m³/h",
  speed_rpm: "rpm", fouling_m2k_kw: "m²K/kW", shell_dp_bar: "bar", crude_out_c: "°C", hot_in_c: "°C", duty_mw: "MW",
  coil_outlet_c: "°C", tube_skin_c: "°C", stack_c: "°C", o2_pct: "%", fuel_gas_t_h: "t/h", top_c: "°C", flash_zone_c: "°C",
  top_bar: "bar", bottom_level_pct: "%", tray_dp_mbar: "mbar", level_pct: "%", temp_c: "°C", level_m: "m" };
const CHARTS = { pump: ["vibration_mm_s", "vib_hf_env_g", "bearing_temp_c"], exchanger: ["crude_out_c", "shell_dp_bar"],
  furnace: ["tube_skin_c", "stack_c"], column: ["tray_dp_mbar", "bottom_level_pct"], tank: ["level_pct", "temp_c"] };
const LIMITS = { vibration_mm_s: [[4.5, "#f5a524"], [7.1, "#ff4d4f"]] };
const FAULTS = { pump: ["bearing_fault", "imbalance", "misalignment", "cavitation"], exchanger: ["fouling"], furnace: ["degradation"], column: ["degradation"], tank: [] };
const KIND = { SM_Pump_Centrifugal: "pump", SM_HeatExchanger: "exchanger", SM_DistillationColumn: "column", SM_Furnace: "furnace", SM_StorageTank: "tank" };
const ue = (x, y, z) => new THREE.Vector3(x / 100, z / 100, y / 100); // Unreal cm -> three m (Y up)
const fmt = (v) => v == null ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 0.01 ? v.toFixed(2) : v.toExponential(1);
const label = (k) => k.replace(/_(mm_s|c|a|bar|m3h|rpm|m2k_kw|mw|pct|t_h|mbar|m|g)$/, "").replace(/^vib_/, "vibration ").replace(/_/g, " ").replace("hf env", "HF envelope").replace("bpfo", "bearing band (BPFO)").replace("1x", "1×").replace("2x", "2×");

// =========================================================== renderer / scene
const host = $("#scene");
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);
const css = new CSS2DRenderer();
Object.assign(css.domElement.style, { position: "absolute", inset: "0", pointerEvents: "none" });
host.appendChild(css.domElement);

const scene = new THREE.Scene();
renderer.toneMapping = THREE.ACESFilmicToneMapping;     // filmic roll-off suits a bright sunset sky

const key = new THREE.DirectionalLight(0xffb067, 2.6);   // the sun
key.castShadow = true;
key.shadow.mapSize.set(4096, 4096);
Object.assign(key.shadow.camera, { left: -140, right: 140, top: 140, bottom: -140, near: 10, far: 700 });
key.shadow.bias = -0.0004; key.shadow.normalBias = 0.05;
scene.add(key);
const hemi = new THREE.HemisphereLight(0x8d9ab4, 0x4a3426, 0.75);
scene.add(hemi);

// far ground to the horizon (the paved unit sits on top of it)
const farGround = new THREE.Mesh(new THREE.CircleGeometry(1400, 64), new THREE.MeshStandardMaterial({ color: 0x3b3430, roughness: 1 }));
farGround.rotation.x = -Math.PI / 2; farGround.position.y = -0.35; farGround.receiveShadow = true;
scene.add(farGround);

const sky = createSky(renderer, scene, { key, hemi, groundMesh: farGround });
let skyName = "sunset";
try { skyName = localStorage.getItem("twin.sky") || "sunset"; } catch { /* storage unavailable */ }
sky.apply(skyName);

const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.maxPolarAngle = Math.PI * 0.495; controls.minDistance = 4; controls.maxDistance = 330;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const outline = new OutlinePass(new THREE.Vector2(1, 1), scene, camera);
Object.assign(outline, { edgeStrength: 4, edgeThickness: 1.5, edgeGlow: 0.2 });
outline.visibleEdgeColor.set("#4fb3ff"); outline.hiddenEdgeColor.set("#1d4d73");
composer.addPass(outline);
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.35, 0.35, 0.94);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h); composer.setSize(w, h); css.setSize(w, h);
  outline.resolution.set(w, h);
}
new ResizeObserver(resize).observe(host);

// =========================================================== materials & assets
let MATS = null;
const texLoader = new THREE.TextureLoader(), texCache = {}, shared = {};
function tex(set, kind, rep) {
  const k = `${set}${kind}${rep}`;
  if (texCache[k]) return texCache[k];
  const t = texLoader.load(`/content/Textures/T_${set}_${kind}.png`);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(rep, rep); t.anisotropy = 8;
  t.colorSpace = kind === "BaseColor" ? THREE.SRGBColorSpace : THREE.NoColorSpace;
  return (texCache[k] = t);
}
function makeMat(key) {
  const s = MATS.materials[key] || MATS.materials.MI_SteelDark, rep = s.uv_scale || 1, orm = tex(s.set, "ORM", rep);
  const tint = s.tint.map((c) => (s.set === "Concrete" ? c * 0.5 : c)); // darker ground so equipment reads clearly
  return new THREE.MeshStandardMaterial({ name: key, color: new THREE.Color().setRGB(...tint), map: tex(s.set, "BaseColor", rep),
    normalMap: tex(s.set, "Normal", rep), normalScale: new THREE.Vector2(0.8, -0.8), roughnessMap: orm, metalnessMap: orm,
    roughness: 1, metalness: 0.85, emissive: 0x000000, emissiveIntensity: 0 });
}
const matKey = (n) => Object.keys(MATS.materials).find((k) => n && n.startsWith(k)) || "MI_SteelDark";
const gltf = new GLTFLoader(), meshCache = {};
const loadMesh = (n) => (meshCache[n] ||= gltf.loadAsync(`/content/Web/${n}.glb`).then((g) => g.scene));
function inst(src, own) {
  const obj = src.clone(true), mats = [];
  obj.traverse((o) => {
    if (!o.isMesh) return;
    o.castShadow = o.receiveShadow = true;
    const conv = (m) => { const k = matKey(m.name); if (own) { const mm = makeMat(k); mats.push(mm); return mm; } return (shared[k] ||= makeMat(k)); };
    o.material = Array.isArray(o.material) ? o.material.map(conv) : conv(o.material);
  });
  return { obj, mats };
}

const assets = {}, pickables = [], areas = [];
let selected = null, hovered = null, labelMode = "smart";

async function buildPlant() {
  const [layout, mats] = await Promise.all([fetch("/config/plant_layout.json").then((r) => r.json()), fetch("/config/materials.json").then((r) => r.json())]);
  MATS = mats;
  const g = layout.ground, tile = await loadMesh(g.mesh);
  for (let i = 0; i < g.nx; i++) for (let j = 0; j < g.ny; j++) {
    const o = inst(tile, false).obj; o.traverse((m) => m.isMesh && (m.castShadow = false));
    o.position.copy(ue(g.origin[0] + g.tile_size_cm * (i + 0.5), g.origin[1] + g.tile_size_cm * (j + 0.5), 0)); scene.add(o);
  }
  const pr = layout.pipe_racks, rack = await loadMesh(pr.mesh);
  for (let i = 0; i < pr.count; i++) { const o = inst(rack, false).obj; o.position.copy(ue(pr.x_start + i * pr.segment_cm, pr.y, 0)); scene.add(o); }

  for (const e of layout.equipment) {
    const { obj, mats } = inst(await loadMesh(e.mesh), true);
    obj.position.copy(ue(...e.loc)); obj.rotation.y = -THREE.MathUtils.degToRad(e.yaw || 0); scene.add(obj);
    const size = new THREE.Box3().setFromObject(obj).getSize(new THREE.Vector3());
    const el = document.createElement("div"); el.className = "lbl"; el.innerHTML = `<i></i>${e.tag}<em></em>`;
    el.onclick = () => select(e.tag, true);
    const lbl = new CSS2DObject(el); lbl.position.set(0, size.y + 1.0, 0); obj.add(lbl);
    const r = Math.max(size.x, size.z) * 0.6 + 0.5;
    const ring = new THREE.Mesh(new THREE.RingGeometry(r, r + 0.25, 72), new THREE.MeshBasicMaterial({ color: STATUS.OK, transparent: true, opacity: 0.3, toneMapped: false }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.05; obj.add(ring);
    obj.traverse((o) => { if (o.isMesh && o !== ring) { o.userData.tag = e.tag; pickables.push(o); } });
    assets[e.tag] = { spec: e, kind: KIND[e.mesh], obj, mats, lbl, el, ring, size, data: null, hist: {} };
  }
  const hk = layout.pump_hookups, valve = await loadMesh(hk.valve_mesh), pipe = await loadMesh(hk.pipe_mesh), edge = pr.y + 300;
  for (const e of layout.equipment.filter((x) => x.mesh === "SM_Pump_Centrifugal")) {
    const [dx, dy, dz] = hk.discharge_offset, x = e.loc[0] + dx, y = e.loc[1] + dy, vz = e.loc[2] + dz + 60, from = vz + 30;
    const v = inst(valve, false).obj; v.position.copy(ue(x, y, vz)); v.rotation.z = Math.PI / 2; scene.add(v);
    const a = inst(pipe, false).obj; a.position.copy(ue(x, y, (from + hk.rack_pipe_z) / 2)); a.scale.y = (hk.rack_pipe_z - from) / 100; scene.add(a);
    const b = inst(pipe, false).obj; b.position.copy(ue(x, (y + edge) / 2, hk.rack_pipe_z)); b.rotation.x = Math.PI / 2; b.scale.y = Math.abs(y - edge) / 100; scene.add(b);
  }
  // area labels for orientation when zoomed out
  for (const [name, pos] of [["Pump bay", ue(50, -2400, 450)], ["Exchanger train", ue(2600, 200, 500)], ["Tank farm", ue(7600, 0, 1800)],
    ["Pipe rack", ue(-2000, -4200, 1100)], ["Fired heater", ue(-3800, 200, 4800)], ["Distillation column", ue(0, 0, 4900)]]) {
    const el = document.createElement("div"); el.className = "area"; el.textContent = name;
    const o = new CSS2DObject(el); o.position.copy(pos); scene.add(o); areas.push(o);
  }
}

// =========================================================== camera
const PRESETS = {
  Skyline: [[-118, 5, -92], [10, 22, 6]],
  Overview: [[-70, 75, -95], [8, 5, 0]], Pumps: [[-4, 11, -4], [1, 1, -24]], Exchangers: [[8, 13, 18], [26, 1.5, 2]],
  Column: [[-28, 26, -34], [0, 16, 0]], Heater: [[-60, 22, -30], [-38, 10, 2]], Tanks: [[45, 30, -45], [76, 6, 0]],
};
let flight = null;
function flyTo(pos, target) { flight = { t: 0, p0: camera.position.clone(), p1: new THREE.Vector3(...pos), q0: controls.target.clone(), q1: new THREE.Vector3(...target) }; }
function flyToAsset(tag) {
  const a = assets[tag], c = new THREE.Box3().setFromObject(a.obj).getCenter(new THREE.Vector3());
  const d = Math.max(a.size.x, a.size.y * 0.8, a.size.z) * 1.8 + 5;
  const dir = camera.position.clone().sub(controls.target).setY(0).normalize();
  const p = c.clone().add(dir.multiplyScalar(d)); p.y = c.y + d * 0.5;
  flyTo(p.toArray(), c.toArray());
}
camera.position.set(...PRESETS.Overview[0]); controls.target.set(...PRESETS.Overview[1]);
$("#presets").innerHTML = Object.keys(PRESETS).map((k) => `<button data-p="${k}">${k}</button>`).join("");
$("#presets").onclick = (e) => { const p = e.target.dataset.p; if (p) flyTo(...PRESETS[p]); };
$("#skymode").innerHTML = sky.presets.map((m) => `<button data-k="${m}" class="${m === skyName ? "on" : ""}">${m[0].toUpperCase() + m.slice(1)}</button>`).join("");
$("#skymode").onclick = (e) => {
  const k = e.target.dataset.k; if (!k) return;
  sky.apply(k); $("#skymode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
  try { localStorage.setItem("twin.sky", k); } catch { /* ignore */ }
};
$("#labelmode").innerHTML = ["smart", "all", "none"].map((m) => `<button data-m="${m}" class="${m === labelMode ? "on" : ""}">Labels: ${m}</button>`).join("");
$("#labelmode").onclick = (e) => { const m = e.target.dataset.m; if (!m) return; labelMode = m; $("#labelmode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === m)); };

// =========================================================== selection & panels
function select(tag, fly) {
  selected = tag; $("#list").querySelectorAll(".row").forEach((r) => r.classList.toggle("sel", r.dataset.tag === tag));
  if (fly) flyToAsset(tag);
  setTab("asset"); backfill(tag); renderAsset(true);
}
async function backfill(tag) {
  const a = assets[tag];
  for (const s of CHARTS[a.kind] || []) {
    try {
      const r = await fetch(`/api/assets/${tag}/history?sensor=${s}&points=240`).then((x) => x.json());
      if (r.points?.length) a.hist[s] = r.points.map((p) => p.v);
    } catch { /* ignore */ }
  }
  renderAsset(true);
}
function setTab(t) { document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t)); document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x.id === `tab-${t}`)); }
document.querySelector(".tabs").onclick = (e) => e.target.dataset.tab && setTab(e.target.dataset.tab);

function buildList() {
  const groups = [["column", "Column"], ["furnace", "Fired heater"], ["exchanger", "Heat exchangers"], ["pump", "Pumps"], ["tank", "Tanks"]];
  $("#list").innerHTML = groups.map(([k, t]) => `<div class="sec">${t}</div>` + Object.entries(assets).filter(([, a]) => a.kind === k).map(([tag, a]) =>
    `<div class="row" data-tag="${tag}"><span class="d" id="d-${tag}"></span><span class="t">${tag}</span><span class="n">${a.spec.name}</span><span class="h" id="h-${tag}">—</span></div>`).join("")).join("");
  $("#list").onclick = (e) => { const r = e.target.closest(".row"); if (r) select(r.dataset.tag, true); };
}

let lastDetail = 0;
function renderAsset(force) {
  if (!selected) return;
  const now = performance.now(); if (!force && now - lastDetail < 900) return; lastDetail = now;
  const a = assets[selected], f = a.data, el = $("#tab-asset");
  if (!f) { el.innerHTML = `<div class="ahead"><h2>${selected}</h2><p>${a.spec.name}</p></div><div class="empty">Waiting for data…</div>`; return; }
  const col = STATUS[f.status];
  el.innerHTML = `
    <div class="ahead"><h2>${selected}<span class="chip" style="background:${col}">${f.status}</span></h2>
      <p>${f.name}${f.fault && f.degradation > 0.1 ? ` · tracking <b>${f.fault.replace("_", " ")}</b>` : ""}</p></div>
    <div class="stats">
      <div class="stat"><b style="color:${col}">${f.health.toFixed(0)}%</b><span>Health</span></div>
      <div class="stat"><b>${f.rul_h != null ? f.rul_h.toFixed(0) + "h" : "—"}</b><span>Remaining life</span></div>
      <div class="stat"><b>${f.anomaly.toFixed(1)}σ</b><span>Anomaly</span></div></div>
    ${(CHARTS[f.kind] || []).map((s) => `<div class="chartbox"><div class="ct"><span>${label(s)}</span><span class="mono">${fmt(f.sensors[s])} ${UNITS[s] || ""}</span></div><canvas id="ch-${s}"></canvas></div>`).join("")}
    <table class="sensors">${Object.entries(f.sensors).map(([k, v]) => `<tr><td>${label(k)}</td><td>${fmt(v)} ${UNITS[k] || ""}</td></tr>`).join("")}</table>
    <div class="sec">Scenario sandbox (simulation only)</div>
    <div class="actions">${(FAULTS[f.kind] || []).map((m) => `<button class="btn warn" data-f="${m}">Simulate ${m.replace("_", " ")}</button>`).join("")}
      <button class="btn" data-r="1">Reset</button><button class="btn" data-ask="1">Ask assistant</button></div>`;
  el.querySelectorAll("[data-f]").forEach((b) => b.onclick = () => send({ cmd: "inject_fault", tag: selected, mode: b.dataset.f, severity: 0.45, hours_to_fail: 12 }));
  el.querySelector("[data-r]").onclick = () => send({ cmd: "reset", tag: selected });
  el.querySelector("[data-ask]").onclick = () => { setTab("chat"); ask(`Diagnose ${selected}. What is happening and what should we do?`); };
  for (const s of CHARTS[f.kind] || []) drawChart($(`#ch-${s}`), a.hist[s] || [], col, LIMITS[s]);
}
function drawChart(cv, data, color, limits) {
  if (!cv) return;
  const W = (cv.width = cv.clientWidth * devicePixelRatio), H = (cv.height = cv.clientHeight * devicePixelRatio), g = cv.getContext("2d");
  if (data.length < 2) { g.fillStyle = "#5d6d80"; g.font = `${11 * devicePixelRatio}px IBM Plex Sans`; g.fillText("collecting…", 8, H / 2); return; }
  let lo = Math.min(...data), hi = Math.max(...data);
  for (const [v] of limits || []) if (v < hi * 1.6) hi = Math.max(hi, v);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; } const pad = (hi - lo) * 0.12; lo -= pad; hi += pad;
  const Y = (v) => H - ((v - lo) / (hi - lo)) * H;
  g.strokeStyle = "#233041"; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, (H * i) / 4); g.lineTo(W, (H * i) / 4); g.stroke(); }
  for (const [v, c] of limits || []) { if (v > hi) continue; g.setLineDash([6, 5]); g.strokeStyle = c; g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(W, Y(v)); g.stroke(); g.setLineDash([]); }
  g.beginPath(); data.forEach((v, i) => { const x = (i / (data.length - 1)) * W; i ? g.lineTo(x, Y(v)) : g.moveTo(x, Y(v)); });
  g.strokeStyle = color; g.lineWidth = 2 * devicePixelRatio; g.stroke();
}

// =========================================================== events
const events = [];
function addEvents(list) {
  if (!list.length) return;
  events.unshift(...list.slice().reverse()); events.length = Math.min(events.length, 100);
  const c = (l) => ({ WARN: "var(--warn)", ALARM: "var(--alarm)", TRIP: "var(--trip)" }[l] || "var(--accent)");
  $("#events").innerHTML = events.map((e) => `<div class="ev"><span class="tiny">${e.ts.slice(5, 16).replace("T", " ")}</span><span style="color:${c(e.level)}">${e.level}</span><b>${e.tag}</b><span>${e.msg}</span></div>`).join("");
  const tk = $("#ticker");
  for (const e of list) {
    const d = document.createElement("div"); d.innerHTML = `<span style="color:${c(e.level)}">${e.level}</span> ${e.tag} — ${e.msg}`;
    tk.prepend(d); setTimeout(() => d.remove(), 12000);
    while (tk.children.length > 3) tk.lastChild.remove();
  }
}

// =========================================================== telemetry
let ws = null;
const send = (m) => ws?.readyState === 1 && ws.send(JSON.stringify(m));
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { $("#conn").classList.add("on"); $("#conn-t").textContent = "live"; };
  ws.onclose = () => { $("#conn").classList.remove("on"); $("#conn-t").textContent = "offline"; setTimeout(connect, 2000); };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type !== "telemetry") return;
    $("#k-time").textContent = m.sim_time.slice(5, 16).replace("T", " ");
    $("#speeds").querySelectorAll("button").forEach((b) => b.classList.toggle("on", +b.dataset.s === m.speed));
    let ok = 0, warn = 0, bad = 0;
    for (const [tag, f] of Object.entries(m.assets)) {
      const a = assets[tag]; if (!a) continue;
      a.data = f;
      for (const [k, v] of Object.entries(f.sensors)) { const h = (a.hist[k] ||= []); h.push(v); if (h.length > 240) h.shift(); }
      f.status === "WARN" ? warn++ : ["ALARM", "TRIPPED"].includes(f.status) ? bad++ : ok++;
      const col = STATUS[f.status];
      $(`#d-${tag}`).style.background = col; $(`#h-${tag}`).textContent = f.status === "STANDBY" ? "stby" : `${f.health.toFixed(0)}%`;
      a.el.querySelector("i").style.background = col;
      a.el.querySelector("em").textContent = ["WARN", "ALARM", "TRIPPED"].includes(f.status) ? `${f.status} ${f.health.toFixed(0)}%` : "";
    }
    $("#k-ok").textContent = ok; $("#k-warn").textContent = warn; $("#k-alarm").textContent = bad;
    addEvents(m.events || []);
    renderAsset(false);
  };
}
const SPEEDS = [60, 300, 1200, 3600];
$("#speeds").innerHTML = SPEEDS.map((s) => `<button data-s="${s}">×${s}</button>`).join("");
$("#speeds").onclick = (e) => e.target.dataset.s && send({ cmd: "speed", value: +e.target.dataset.s });
$("#reset-all").onclick = () => send({ cmd: "reset_all" });

// =========================================================== assistant
const session = Math.random().toString(36).slice(2);
const SUGGEST = ["Which assets need attention?", "Diagnose P-101A", "Show the vibration trend of P-101A over 24 hours",
  "What happens if E-102 fouls? Simulate it", "Summarise recent alarms"];
$("#suggest").innerHTML = SUGGEST.map((s) => `<button>${s}</button>`).join("");
$("#suggest").onclick = (e) => e.target.tagName === "BUTTON" && ask(e.target.textContent);
$("#ask").onsubmit = (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) { $("#q").value = ""; ask(q); } };
function md(t) {
  const esc = t.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const lines = esc.split("\n"); let out = "", inList = false;
  for (const l of lines) {
    const li = l.match(/^\s*[-*•]\s+(.*)/) || l.match(/^\s*\d+\.\s+(.*)/);
    if (li) { if (!inList) { out += "<ul>"; inList = true; } out += `<li>${li[1]}</li>`; continue; }
    if (inList) { out += "</ul>"; inList = false; }
    if (l.trim()) out += `<p>${l.replace(/^#+\s*/, "")}</p>`;
  }
  if (inList) out += "</ul>";
  return out.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`(.+?)`/g, "<code>$1</code>");
}
function addMsg(cls, html) { const d = document.createElement("div"); d.className = `msg ${cls}`; d.innerHTML = html; $("#msgs").appendChild(d); $("#msgs").scrollTop = 1e9; return d; }
async function ask(q) {
  setTab("chat"); addMsg("u", md(q)); const t = addMsg("a think", "Thinking…");
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: q, session_id: session }) }).then((x) => x.json());
    t.className = "msg a";
    t.innerHTML = md(r.reply || "(no answer)") + (r.tools?.length ? `<div class="used">${[...new Set(r.tools)].map((x) => `<span>${x}</span>`).join("")}</div>` : "");
    $("#llm").textContent = `Model: ${r.llm}`;
    const focus = (r.actions || []).filter((a) => a.type === "focus").pop();
    if (focus && assets[focus.tag]) { selected = focus.tag; flyToAsset(focus.tag); backfill(focus.tag); $("#list").querySelectorAll(".row").forEach((x) => x.classList.toggle("sel", x.dataset.tag === focus.tag)); }
  } catch (e) { t.className = "msg a"; t.textContent = `Request failed: ${e.message}`; }
}
fetch("/api/llm").then((r) => r.json()).then((r) => ($("#llm").textContent = `Model: ${r.llm}`)).catch(() => {});
addMsg("a", md("Hi — I'm the CDU-100 plant assistant. Ask me about equipment health, trends, alarms or run a what-if scenario."));

// =========================================================== interaction & render loop
const ray = new THREE.Raycaster(), ptr = new THREE.Vector2(); let down = null;
function pick(e) { const r = renderer.domElement.getBoundingClientRect(); ptr.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ptr, camera); return ray.intersectObjects(pickables, false)[0]?.object.userData.tag || null; }
renderer.domElement.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
renderer.domElement.addEventListener("pointerup", (e) => { if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 5) { const t = pick(e); if (t) select(t, true); } });
let lastHover = 0;
renderer.domElement.addEventListener("pointermove", (e) => { if (performance.now() - lastHover < 60) return; lastHover = performance.now(); hovered = pick(e); renderer.domElement.style.cursor = hovered ? "pointer" : ""; });

const clock = new THREE.Clock(), tmp = new THREE.Color(), wp = new THREE.Vector3();
function loop() {
  requestAnimationFrame(loop);
  const dt = clock.getDelta(), t = clock.elapsedTime;
  if (flight) {
    flight.t = Math.min(1, flight.t + dt / 1.2); const k = flight.t < 0.5 ? 2 * flight.t ** 2 : 1 - (-2 * flight.t + 2) ** 2 / 2;
    camera.position.lerpVectors(flight.p0, flight.p1, k); controls.target.lerpVectors(flight.q0, flight.q1, k); if (flight.t >= 1) flight = null;
  }
  const camDist = camera.position.distanceTo(controls.target);
  for (const [tag, a] of Object.entries(assets)) {
    const f = a.data, bad = f && ["WARN", "ALARM", "TRIPPED"].includes(f.status);
    if (f) {
      const pulse = 0.6 + 0.4 * Math.sin(t * f.pulse);
      tmp.set(STATUS[f.status]);
      const glow = bad ? (f.status === "WARN" ? 0.35 : 0.7) * pulse : 0;
      for (const m of a.mats) { m.emissive.copy(tmp); m.emissiveIntensity = glow; }
      a.ring.material.color.copy(tmp); a.ring.material.opacity = bad ? 0.55 + 0.4 * pulse : 0.28;
    }
    // smart labels: only near the camera, for problems, or for the selected/hovered asset
    a.obj.getWorldPosition(wp);
    const near = camera.position.distanceTo(wp) < 32 + a.size.y * 1.2;
    const show = labelMode === "all" || (labelMode === "smart" && (near || bad || tag === selected || tag === hovered));
    a.lbl.visible = show; a.el.classList.toggle("sel", tag === selected);
  }
  for (const o of areas) o.visible = labelMode !== "none" && camDist > 60;
  outline.selectedObjects = [selected, hovered].filter(Boolean).map((t) => assets[t].obj);
  sky.update(t, camera);
  controls.update(); composer.render(); css.render(scene, camera);
}

buildPlant().then(() => { buildList(); resize(); $("#loading").remove(); loop(); connect(); })
  .catch((e) => { $("#loading").innerHTML = `Could not load 3D assets: ${e.message}<br>Run START.bat (it exports them from Blender).`; console.error(e); });
