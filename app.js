// Smart Crosswalk Monitor
// Detects pedestrians and vehicles in traffic video, tracks them, and flags
// vehicles that enter a marked crosswalk while a pedestrian is on it.
import { FilesetResolver, ObjectDetector } from "./vendor/mediapipe/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const abs = (p) => new URL(p, import.meta.url).href;

// ---------- settings ----------
const MODELS = { lite0: "models/efficientdet_lite0.tflite", lite2: "models/efficientdet_lite2.tflite" };
const SAMPLE = {
  webm: "media/sample-crosswalk.webm",
  mp4: "media/sample-crosswalk.mp4",
  // crosswalk drawn for the sample clip (fractions of width / height)
  zone: [[0.03, 0.37], [0.97, 0.37], [0.97, 0.70], [0.03, 0.70]],
};
const DEFAULT_ZONE = [[0.08, 0.55], [0.92, 0.55], [0.92, 0.78], [0.08, 0.78]];
const VEHICLE_LABELS = new Set(["car", "truck", "bus", "motorcycle", "train"]);
const NEVER_VEHICLE = new Set(["person", "bicycle", "traffic light", "stop sign", "fire hydrant", "parking meter", "bench", "dining table", "bed"]);
const MAX_W = 960;            // analysis resolution cap
const TRACK_TTL = 1.2;        // seconds a lost track is kept
const CONFIRM_HITS = 3;       // detections before a track counts
const WAIT_TO_YIELD = 0.8;    // seconds stopped near the crossing that count as yielding
const DECIDE_WINDOW = 0.6;    // seconds after entering the crossing to decide the outcome

// ---------- elements ----------
const video = $("video"), view = $("view"), ctx = view.getContext("2d");
const raw = document.createElement("canvas"), rctx = raw.getContext("2d", { willReadFrequently: true });
const small = document.createElement("canvas"), sctx = small.getContext("2d", { willReadFrequently: true });
const motionCanvas = document.createElement("canvas"), mctx = motionCanvas.getContext("2d");

// ---------- state ----------
let files = null, detector = null, detectorKey = null, delegateUsed = "";
let mode = null;              // "file" | "camera"
let running = false, camStart = 0, lastT = -1, lastTs = 0, detectMs = 60;
let zone = SAMPLE.zone.map((p) => p.slice());
let drawing = false, drawPts = [], zoneBefore = null;
let tracks = [], nextId = { ped: 1, veh: 1 }, counts, events, bgModel = null, motionMask = null;
let alertUntil = 0, alertText = "";
let ocrWorker = null;
let forceCpu = false, framesOnDelegate = 0, switching = false;
resetCounts();

function resetCounts() {
  counts = { peds: 0, cross: 0, veh: 0, viol: 0, yield: 0, clear: 0 };
  events = [];
  tracks = [];
  nextId = { ped: 1, veh: 1 };
  bgModel = null;
  alertUntil = 0;
  $("banner").hidden = true;
  $("events").innerHTML = '<li class="empty">Vehicles that enter the crosswalk while someone is on it will show up here with a snapshot.</li>';
  renderStats();
}

// ---------- detector ----------
async function ensureDetector(key) {
  if (detector && detectorKey === key) return;
  showLoading(key === "lite2" ? "Loading accurate model…" : "Loading detector…");
  if (!files) files = await FilesetResolver.forVisionTasks(abs("./vendor/mediapipe/wasm"));
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: abs(MODELS[key]), delegate },
    runningMode: "VIDEO", scoreThreshold: 0.3, maxResults: 15,
  });
  if (detector) { detector.close(); detector = null; }
  const wantCpu = forceCpu || new URLSearchParams(location.search).has("cpu");
  try {
    if (wantCpu) throw new Error("cpu requested");
    detector = await ObjectDetector.createFromOptions(files, opts("GPU")); delegateUsed = "GPU";
  } catch { detector = await ObjectDetector.createFromOptions(files, opts("CPU")); delegateUsed = "CPU"; }
  framesOnDelegate = 0; detectMs = 60;
  detectorKey = key;
  showLoading(null);
}

function showLoading(text) {
  $("loading").hidden = !text;
  if (text) $("loadingText").textContent = text;
}

// ---------- sources ----------
async function startSample() {
  stopCamera();
  zone = SAMPLE.zone.map((p) => p.slice());
  $("chkTopDown").checked = true;
  const webm = video.canPlayType('video/webm; codecs="vp9"');
  await startFile(abs(webm ? SAMPLE.webm : SAMPLE.mp4), "Sample clip");
}

async function startUpload(file) {
  stopCamera();
  zone = DEFAULT_ZONE.map((p) => p.slice());
  $("chkTopDown").checked = false;
  await startFile(URL.createObjectURL(file), file.name, true);
}

async function startFile(src, name, askZone = false) {
  running = false;
  mode = "file";
  video.srcObject = null;
  video.src = src;
  video.loop = false;
  await once(video, "loadeddata");
  sizeCanvases(video.videoWidth, video.videoHeight);
  resetCounts();
  $("placeholder").hidden = true;
  await ensureDetector($("selModel").value);
  setStatus(`${name} loaded.`);
  enableControls(true);
  if (askZone) {
    video.currentTime = 0;
    await once(video, "seeked").catch(() => {});
    grabFrame(); render();
    beginDraw(true);
    return;
  }
  play();
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
    running = false;
    mode = "camera";
    video.removeAttribute("src");
    video.srcObject = stream;
    await video.play();
    sizeCanvases(video.videoWidth, video.videoHeight);
    resetCounts();
    zone = DEFAULT_ZONE.map((p) => p.slice());
    $("chkTopDown").checked = false;
    $("placeholder").hidden = true;
    await ensureDetector($("selModel").value);
    enableControls(true);
    camStart = performance.now();
    running = true; lastT = -1;
    $("btnPlay").textContent = "Pause";
    setStatus("Live camera. Draw the crosswalk to match your view.");
    loop();
  } catch (e) {
    setStatus("Camera unavailable: " + e.message);
  }
}

function stopCamera() {
  const s = video.srcObject;
  if (s && s.getTracks) s.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}

function sizeCanvases(w, h) {
  const scale = Math.min(1, MAX_W / w);
  const W = Math.round(w * scale), H = Math.round(h * scale);
  for (const c of [view, raw]) { c.width = W; c.height = H; }
  small.width = 160; small.height = Math.max(1, Math.round(160 * H / W));
  motionCanvas.width = small.width; motionCanvas.height = small.height;
  $("screen").style.aspectRatio = `${W} / ${H}`;
}

function once(el, ev) {
  return new Promise((res, rej) => {
    const ok = () => { el.removeEventListener("error", bad); res(); };
    const bad = () => { el.removeEventListener(ev, ok); rej(new Error("media error")); };
    el.addEventListener(ev, ok, { once: true });
    el.addEventListener("error", bad, { once: true });
  });
}

function enableControls(on) {
  $("btnPlay").disabled = !on;
  $("btnRestart").disabled = !on || mode === "camera";
}

// ---------- playback / loop ----------
function play() {
  if (mode !== "file") return;
  running = true;
  lastT = -1;
  video.play().catch(() => {});
  $("btnPlay").textContent = "Pause";
  loop();
}

function pause() {
  running = false;
  if (mode === "file") video.pause();
  $("btnPlay").textContent = "Play";
}

function loop() {
  if (!running) return;
  if (video.requestVideoFrameCallback) {
    video.requestVideoFrameCallback((now, meta) => { onFrame(meta.mediaTime); loop(); });
  } else {
    requestAnimationFrame(() => { onFrame(video.currentTime); loop(); });
  }
}

function onFrame(mediaTime) {
  if (!running || !detector) return;
  const t = mode === "camera" ? (performance.now() - camStart) / 1000 : mediaTime;
  if (t === lastT) return;
  const dt = lastT < 0 ? 0 : Math.max(0, Math.min(0.5, t - lastT));
  lastT = t;
  processFrame(t, dt);
}

video.addEventListener("ended", () => {
  if (mode !== "file") return;
  running = false;
  finalizeAll(lastT + 1);
  $("btnPlay").textContent = "Play again";
  const v = counts.viol;
  setStatus(`Finished. ${v} vehicle${v === 1 ? "" : "s"} did not yield, ${counts.yield} yielded, ${counts.clear} passed with the crossing clear.`);
  $("banner").hidden = true;
  render();
});

// ---------- per-frame work ----------
function grabFrame() {
  rctx.drawImage(video, 0, 0, raw.width, raw.height);
}

function processFrame(t, dt) {
  grabFrame();
  updateMotion();

  const t0 = performance.now();
  const ts = Math.max(performance.now(), lastTs + 1);
  lastTs = ts;
  let result;
  try { result = detector.detectForVideo(raw, ts); } catch (e) { console.warn(e); return; }
  const ms = performance.now() - t0;
  detectMs = framesOnDelegate === 0 ? ms : detectMs * 0.85 + ms * 0.15;
  // Some machines expose only a slow software GPU; fall back to the CPU there.
  if (++framesOnDelegate === 8 && delegateUsed === "GPU" && detectMs > 180 && !switching) {
    switching = true; forceCpu = true;
    const was = running; if (was) pause();
    detector.close(); detector = null; detectorKey = null;
    ensureDetector($("selModel").value).then(() => { switching = false; if (was) { if (mode === "camera") { running = true; loop(); } else play(); } });
    return;
  }

  const { peds, vehs } = classify(result.detections || []);
  updateTracks("ped", peds, t);
  updateTracks("veh", vehs, t);
  applyRules(t, dt);
  render(t);
  adaptSpeed();
  if (mode === "file") {
    setStatus(`Analyzing ${fmtTime(t)} / ${fmtTime(video.duration || 0)} · ${Math.round(1000 / detectMs)} detections/s (${delegateUsed}) · playback ${video.playbackRate}x`);
  }
}

function adaptSpeed() {
  if (mode !== "file") return;
  // Aim to analyse at least ~10 frames per second of video.
  const ideal = Math.min(1, Math.max(0.1, 1000 / detectMs / 10));
  const rate = ideal >= 0.25 ? Math.round(ideal * 4) / 4 : Math.round(ideal * 20) / 20;
  if (Math.abs(video.playbackRate - rate) >= 0.05) video.playbackRate = rate;
}

function classify(dets) {
  const W = raw.width, H = raw.height, area = W * H;
  const topDown = $("chkTopDown").checked;
  const peds = [], vehs = [];
  for (const d of dets) {
    const c = d.categories[0];
    const b = d.boundingBox;
    const box = { x: b.originX, y: b.originY, w: b.width, h: b.height, score: c.score, label: c.categoryName };
    const frac = (box.w * box.h) / area;
    if (c.categoryName === "person") {
      if (c.score >= 0.35 && frac < 0.25) peds.push(box);
    } else if (VEHICLE_LABELS.has(c.categoryName)) {
      if (frac < 0.6) vehs.push(box);
    } else if (topDown && !NEVER_VEHICLE.has(c.categoryName) && frac >= 0.02 && frac < 0.45) {
      // From above, cars are often labelled "cell phone", "boat" and so on.
      box.label = "vehicle";
      vehs.push(box);
    }
  }
  return { peds: nms(peds), vehs: nms(vehs) };
}

function nms(list) {
  list.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of list) if (!keep.some((k) => iou(k, b) > 0.45 || containment(b, k) > 0.8)) keep.push(b);
  return keep;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a.w * a.h + b.w * b.h - inter || 1);
}
function containment(a, b) { // share of a inside b
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h);
  return (Math.max(0, x2 - x1) * Math.max(0, y2 - y1)) / (a.w * a.h || 1);
}

// ---------- tracking ----------
function updateTracks(kind, dets, t) {
  const live = tracks.filter((k) => k.kind === kind);
  const pairs = [];
  for (let i = 0; i < live.length; i++) {
    const tr = live[i];
    const gap = t - tr.last;
    for (let j = 0; j < dets.length; j++) {
      const d = dets[j];
      const o = iou(tr.box, d);
      const dist = Math.hypot(cx(tr.box) - cx(d), cy(tr.box) - cy(d));
      const reach = Math.max(tr.box.w, tr.box.h) * (0.6 + gap);
      const s = o > 0.15 ? 1 + o : dist < reach ? 1 - dist / reach : 0;
      if (s > 0) pairs.push([s, i, j]);
    }
  }
  pairs.sort((a, b) => b[0] - a[0]);
  const usedT = new Set(), usedD = new Set();
  for (const [, i, j] of pairs) {
    if (usedT.has(i) || usedD.has(j)) continue;
    usedT.add(i); usedD.add(j);
    const tr = live[i], d = dets[j];
    const a = 0.65;
    tr.box = { x: lerp(tr.box.x, d.x, a), y: lerp(tr.box.y, d.y, a), w: lerp(tr.box.w, d.w, a), h: lerp(tr.box.h, d.h, a) };
    tr.last = t; tr.hits++;
    if (d.label !== "vehicle") tr.label = d.label;
    tr.hist.push({ t, x: cx(tr.box), y: cy(tr.box) });
    while (tr.hist.length && t - tr.hist[0].t > 2) tr.hist.shift();
    if (!tr.id && tr.hits >= CONFIRM_HITS) {
      tr.id = nextId[kind]++;
      if (kind === "ped") counts.peds++; else counts.veh++;
    }
  }
  dets.forEach((d, j) => {
    if (usedD.has(j)) return;
    tracks.push({
      kind, id: 0, box: { x: d.x, y: d.y, w: d.w, h: d.h }, label: d.label, first: t, last: t, hits: 1,
      hist: [{ t, x: cx(d), y: cy(d) }], inZone: false, everInZone: false, zoneEnterT: null,
      waitT: 0, enterT: null, outcome: null, moving: false, motion: 0,
    });
  });
  // retire lost tracks
  tracks = tracks.filter((tr) => {
    if (t - tr.last <= TRACK_TTL || t < tr.last) return true;
    if (tr.kind === "veh") finalize(tr, t);
    return false;
  });
}

const cx = (b) => b.x + b.w / 2;
const cy = (b) => b.y + b.h / 2;
const lerp = (a, b, k) => a + (b - a) * k;

// ---------- crosswalk rules ----------
function applyRules(t, dt) {
  const W = raw.width, H = raw.height, diag = Math.hypot(W, H);
  const poly = zone.map(([x, y]) => [x * W, y * H]);

  const pedsOn = [];
  for (const p of tracks) {
    if (p.kind !== "ped" || !p.id) continue;
    const fresh = t - p.last < 0.5;
    const inZ = fresh && pointInPoly(cx(p.box), p.box.y + p.box.h * 0.92, poly);
    if (inZ && !p.inZone) p.zoneEnterT = t;
    if (!inZ) p.zoneEnterT = null;
    p.inZone = inZ;
    if (inZ && !p.everInZone) { p.everInZone = true; counts.cross++; }
    if (inZ) pedsOn.push(p);
  }

  for (const v of tracks) {
    if (v.kind !== "veh" || !v.id) continue;
    const fresh = t - v.last < 0.5;
    // movement: displacement over ~0.5 s, backed up by background subtraction
    const past = v.hist.find((h) => t - h.t <= 0.6) || v.hist[0];
    const span = Math.max(0.15, t - past.t);
    const speed = Math.hypot(cx(v.box) - past.x, cy(v.box) - past.y) / diag / span; // diagonals per second
    v.motion = motionIn(v.box);
    v.moving = speed > 0.025 || (v.motion > 0.3 && speed > 0.01);

    const frac = fractionInPoly(v.box, poly);
    const inZ = fresh && frac >= 0.3;
    const near = !inZ && fractionInPoly(grow(v.box, 0.35), poly) > 0;

    if (!inZ && near && !v.moving && pedsOn.length && v.outcome == null) v.waitT += dt;

    if (inZ && v.enterT == null) v.enterT = t;
    // ignore a one-frame brush with the edge of the crossing
    if (!inZ && v.enterT != null && v.outcome == null && t - v.enterT < 0.25) v.enterT = null;
    if (v.enterT != null && v.outcome == null) {
      const hadPriority = pedsOn.filter((p) => p.zoneEnterT != null && p.zoneEnterT <= v.enterT + 0.3);
      if (v.moving && hadPriority.length && v.waitT < WAIT_TO_YIELD) {
        decide(v, "violation", t, hadPriority.length);
      } else if (t - v.enterT > DECIDE_WINDOW || !inZ) {
        finalize(v, t);
      }
    }
    v.inZone = inZ;
  }
  const show = pedsOn.length > 0;
  const banner = $("banner");
  if (t < alertUntil) { banner.hidden = false; banner.className = "banner alert"; banner.textContent = alertText; }
  else if (show) { banner.hidden = false; banner.className = "banner"; banner.textContent = pedsOn.length > 1 ? `${pedsOn.length} pedestrians on the crossing` : "Pedestrian on the crossing"; }
  else banner.hidden = true;
}

function finalize(v, t) {
  if (v.outcome != null || v.enterT == null || !v.id) return;
  decide(v, v.waitT >= WAIT_TO_YIELD ? "yield" : "clear", t, 0);
}
function finalizeAll(t) { for (const v of tracks) if (v.kind === "veh") finalize(v, t); renderStats(); }

function decide(v, outcome, t, nPeds) {
  v.outcome = outcome;
  const ev = { t, clock: mode === "camera" ? new Date() : null, vehicle: v.id, outcome, peds: nPeds, waited: +v.waitT.toFixed(1), plate: "" };
  events.push(ev);
  if (outcome === "violation") {
    counts.viol++;
    alertUntil = t + 1.8;
    alertText = `Vehicle ${v.id} did not yield!`;
    render(t, v);
    addEvidence(ev, v);
  } else if (outcome === "yield") {
    counts.yield++;
    render(t, v);
    addEvidence(ev, v);
  } else counts.clear++;
  renderStats();
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function fractionInPoly(b, poly) {
  let n = 0;
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) {
    if (pointInPoly(b.x + ((i + 0.5) / 6) * b.w, b.y + ((j + 0.5) / 6) * b.h, poly)) n++;
  }
  return n / 36;
}
function grow(b, k) { return { x: b.x - b.w * k / 2, y: b.y - b.h * k / 2, w: b.w * (1 + k), h: b.h * (1 + k) }; }

// ---------- background subtraction (running average) ----------
function updateMotion() {
  const w = small.width, h = small.height;
  sctx.drawImage(raw, 0, 0, w, h);
  const px = sctx.getImageData(0, 0, w, h).data;
  if (!bgModel || bgModel.length !== w * h) {
    bgModel = new Float32Array(w * h);
    motionMask = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) bgModel[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
  }
  for (let i = 0; i < w * h; i++) {
    const g = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
    motionMask[i] = Math.abs(g - bgModel[i]) > 20 ? 1 : 0;
    bgModel[i] += (g - bgModel[i]) * 0.04;
  }
}
function motionIn(b) {
  if (!motionMask) return 0;
  const sx = small.width / raw.width, sy = small.height / raw.height;
  const x0 = Math.max(0, Math.floor(b.x * sx)), x1 = Math.min(small.width, Math.ceil((b.x + b.w) * sx));
  const y0 = Math.max(0, Math.floor(b.y * sy)), y1 = Math.min(small.height, Math.ceil((b.y + b.h) * sy));
  let n = 0, m = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { n++; m += motionMask[y * small.width + x]; }
  return n ? m / n : 0;
}

// ---------- drawing ----------
let zebra = null;
function zebraPattern() {
  if (zebra) return zebra;
  const c = document.createElement("canvas"); c.width = 24; c.height = 24;
  const g = c.getContext("2d");
  g.fillStyle = "rgba(255,255,255,0.22)"; g.fillRect(0, 0, 12, 24);
  zebra = ctx.createPattern(c, "repeat");
  return zebra;
}

function render(t = lastT, highlight = null) {
  const W = view.width, H = view.height;
  ctx.drawImage(raw, 0, 0);
  if ($("chkMotion").checked && motionMask) {
    const img = mctx.createImageData(small.width, small.height);
    for (let i = 0; i < motionMask.length; i++) if (motionMask[i]) { img.data[i * 4] = 255; img.data[i * 4 + 1] = 40; img.data[i * 4 + 2] = 90; img.data[i * 4 + 3] = 170; }
    mctx.putImageData(img, 0, 0);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(motionCanvas, 0, 0, W, H); ctx.restore();
  }
  const lw = Math.max(2, W / 320);
  // crosswalk
  const pts = drawing ? drawPts : zone;
  if (pts.length) {
    ctx.save();
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x * W, y * H) : ctx.moveTo(x * W, y * H)));
    if (!drawing || pts.length > 2) ctx.closePath();
    const pedOn = tracks.some((p) => p.kind === "ped" && p.inZone);
    if (!drawing) { ctx.fillStyle = zebraPattern(); ctx.fill(); }
    ctx.lineWidth = lw;
    ctx.setLineDash(drawing ? [8, 6] : []);
    ctx.strokeStyle = t < alertUntil ? "#ff4545" : pedOn ? "#ffd21f" : "rgba(255,255,255,0.75)";
    ctx.stroke();
    if (drawing) { ctx.fillStyle = "#ffd21f"; for (const [x, y] of pts) { ctx.beginPath(); ctx.arc(x * W, y * H, lw * 2.5, 0, 7); ctx.fill(); } }
    ctx.restore();
  }
  if (drawing) return;
  // tracks
  ctx.font = `600 ${Math.max(12, W / 60)}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  for (const tr of tracks) {
    if (!tr.id || t - tr.last > 0.4) continue;
    let color, label;
    if (tr.kind === "ped") { color = tr.inZone ? "#ffd21f" : "#4cc3ff"; label = `Person ${tr.id}`; }
    else {
      color = "#ffae3b"; label = `Vehicle ${tr.id}`;
      if (tr.outcome === "violation") { color = "#ff4545"; label += " · DID NOT YIELD"; }
      else if (tr.outcome === "yield") { color = "#2fd27a"; label += " · yielded"; }
      else if (tr.waitT > 0.3 && !tr.moving) { color = "#2fd27a"; label += " · waiting"; }
    }
    const b = tr.box;
    ctx.lineWidth = tr === highlight ? lw * 2 : lw;
    ctx.strokeStyle = color;
    ctx.strokeRect(b.x, b.y, b.w, b.h);
    const tw = ctx.measureText(label).width + 10, th = Math.max(16, W / 48);
    const ty = b.y - th < 0 ? b.y + th : b.y;
    ctx.fillStyle = color; ctx.fillRect(b.x - lw / 2, ty - th, tw, th);
    ctx.fillStyle = "#111"; ctx.fillText(label, b.x + 5, ty - 2);
  }
}

function renderStats() {
  $("sViol").textContent = counts.viol;
  $("sYield").textContent = counts.yield;
  $("sClear").textContent = counts.clear;
  $("sPeds").textContent = counts.peds;
  $("sCross").textContent = counts.cross;
  $("sVeh").textContent = counts.veh;
  const n = counts.viol + counts.yield;
  $("rate").innerHTML = n ? `Yield rate when someone was crossing: <b>${Math.round((100 * counts.yield) / n)}%</b> (${counts.yield} of ${n})` : "Yield rate: <b>no data yet</b>";
  $("btnCsv").disabled = events.length === 0;
}

// ---------- evidence ----------
function addEvidence(ev, v) {
  const list = $("events");
  list.querySelector(".empty")?.remove();
  // full frame with overlays
  const snapW = Math.min(640, view.width);
  const snap = document.createElement("canvas");
  snap.width = snapW; snap.height = Math.round((view.height * snapW) / view.width);
  snap.getContext("2d").drawImage(view, 0, 0, snap.width, snap.height);
  ev.snapshot = snap.toDataURL("image/jpeg", 0.85);
  // zoomed crop of the vehicle from the clean frame
  const b = grow(v.box, 0.15);
  const x = Math.max(0, b.x), y = Math.max(0, b.y);
  const w = Math.min(raw.width - x, b.w), h = Math.min(raw.height - y, b.h);
  const k = Math.min(3, Math.max(1, 260 / Math.max(1, w)));
  const crop = document.createElement("canvas");
  crop.width = Math.max(1, Math.round(w * k)); crop.height = Math.max(1, Math.round(h * k));
  const cc = crop.getContext("2d");
  cc.imageSmoothingQuality = "high";
  cc.drawImage(raw, x, y, w, h, 0, 0, crop.width, crop.height);
  ev.crop = crop.toDataURL("image/jpeg", 0.9);

  const li = document.createElement("li");
  li.className = "ev " + ev.outcome;
  const when = ev.clock ? ev.clock.toLocaleTimeString() : fmtTime(ev.t);
  const title = ev.outcome === "violation" ? `<b class="v">Vehicle ${ev.vehicle} did not yield</b>` : `<b class="y">Vehicle ${ev.vehicle} yielded</b>`;
  const detail = ev.outcome === "violation"
    ? `Entered the crosswalk while ${ev.peds === 1 ? "a pedestrian was" : ev.peds + " pedestrians were"} crossing.`
    : `Waited ${ev.waited}s for the pedestrian to cross.`;
  li.innerHTML = `
    <div class="h">${title}<span class="t">${when}</span></div>
    <div class="d">${detail}</div>
    <div class="imgs"><img alt="Vehicle close-up" src="${ev.crop}"><img alt="Frame at the moment" src="${ev.snapshot}"></div>
    ${ev.outcome === "violation" ? `<div class="plate"><input placeholder="Number plate" aria-label="Number plate"><button class="read">Read plate</button><a class="dl" download="crosswalk-vehicle-${ev.vehicle}.jpg" href="${ev.snapshot}"><button>Save</button></a></div>` : ""}`;
  li.querySelectorAll("img").forEach((img) => img.addEventListener("click", () => lightbox(img.src)));
  const input = li.querySelector("input");
  if (input) {
    input.addEventListener("input", () => (ev.plate = input.value.toUpperCase()));
    li.querySelector(".read").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true; btn.textContent = "Reading…";
      try {
        const text = await readPlate(crop);
        input.value = text; ev.plate = text;
        btn.textContent = text ? "Read again" : "Not readable";
      } catch (err) {
        console.warn(err); btn.textContent = "OCR failed";
      }
      btn.disabled = false;
    });
  }
  list.prepend(li);
}

async function readPlate(canvas) {
  if (!window.Tesseract) throw new Error("OCR library not loaded");
  if (!ocrWorker) {
    ocrWorker = await Tesseract.createWorker("eng", 1, {
      workerPath: abs("vendor/tesseract/worker.min.js"),
      corePath: abs("vendor/tesseract/core"),
      langPath: abs("vendor/tesseract/lang"),
      gzip: true,
    });
    await ocrWorker.setParameters({ tessedit_char_whitelist: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ", tessedit_pageseg_mode: "11" });
  }
  // grayscale + contrast stretch, upscaled
  const k = 2, c = document.createElement("canvas");
  c.width = canvas.width * k; c.height = canvas.height * k;
  const g = c.getContext("2d");
  g.drawImage(canvas, 0, 0, c.width, c.height);
  const img = g.getImageData(0, 0, c.width, c.height), d = img.data;
  let lo = 255, hi = 0;
  for (let i = 0; i < d.length; i += 4) { const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; d[i] = y; if (y < lo) lo = y; if (y > hi) hi = y; }
  const s = 255 / Math.max(1, hi - lo);
  for (let i = 0; i < d.length; i += 4) { const y = (d[i] - lo) * s; d[i] = d[i + 1] = d[i + 2] = y; }
  g.putImageData(img, 0, 0);
  const { data } = await ocrWorker.recognize(c);
  // Prefer a line that looks like a plate: 4-10 letters/digits including at least one digit.
  const lines = (data.text || "").split("\n").map((l) => l.toUpperCase().replace(/[^A-Z0-9 -]/g, "").trim()).filter(Boolean);
  const cands = lines.map((l) => l.replace(/\s+/g, " ")).filter((l) => /\d/.test(l) && l.replace(/[ -]/g, "").length >= 4 && l.replace(/[ -]/g, "").length <= 10);
  cands.sort((x, y) => y.replace(/[ -]/g, "").length - x.replace(/[ -]/g, "").length);
  return (data.confidence ?? 100) > 30 ? cands[0] || "" : "";
}

function lightbox(src) {
  const d = document.createElement("div");
  d.className = "lightbox";
  d.innerHTML = `<img alt="" src="${src}">`;
  d.addEventListener("click", () => d.remove());
  document.body.appendChild(d);
}

function downloadCsv() {
  const rows = [["video_time_s", "clock_time", "vehicle", "outcome", "pedestrians_on_crossing", "waited_s", "plate"]];
  for (const e of events) rows.push([e.t.toFixed(2), e.clock ? e.clock.toISOString() : "", e.vehicle, e.outcome, e.peds, e.waited, e.plate || ""]);
  rows.push([]);
  rows.push(["summary", "", "", `did_not_yield=${counts.viol}`, `yielded=${counts.yield}`, `clear_passes=${counts.clear}`, `pedestrians=${counts.peds}; vehicles=${counts.veh}`]);
  const csv = rows.map((r) => r.map((x) => `"${String(x).replace(/"/g, '""')}"`).join(",")).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "crosswalk-report.csv";
  a.click();
}

// ---------- crosswalk drawing ----------
function beginDraw(fromUpload = false) {
  if (!mode) return;
  zoneBefore = zone.map((p) => p.slice());
  drawing = true; drawPts = [];
  if (running) pause();
  $("screen").classList.add("drawing");
  $("drawHint").hidden = false;
  $("btnDraw").hidden = true; $("btnDrawDone").hidden = false; $("btnDrawCancel").hidden = false;
  $("btnDrawDone").disabled = true;
  if (fromUpload) setStatus("Mark the crosswalk first: click its corners on the video.");
  render();
}
function endDraw(save) {
  if (save && drawPts.length >= 3) zone = drawPts.map((p) => p.slice()); else zone = zoneBefore;
  drawing = false;
  $("screen").classList.remove("drawing");
  $("drawHint").hidden = true;
  $("btnDraw").hidden = false; $("btnDrawDone").hidden = true; $("btnDrawCancel").hidden = true;
  render();
  if (mode === "camera") { running = true; loop(); }
  else if (mode === "file" && !video.ended) play();
}
view.addEventListener("click", (e) => {
  if (!drawing) return;
  const r = view.getBoundingClientRect();
  drawPts.push([(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]);
  $("btnDrawDone").disabled = drawPts.length < 3;
  render();
});

// ---------- misc ----------
function fmtTime(s) {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60), r = s - m * 60;
  return `${m}:${r.toFixed(1).padStart(4, "0")}`;
}
function setStatus(s) { $("status").textContent = s; }

// ---------- wire up ----------
$("btnSample").addEventListener("click", () => startSample().catch((e) => setStatus("Could not start: " + e.message)));
$("fileInput").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) startUpload(f).catch((err) => setStatus("Could not open video: " + err.message)); e.target.value = ""; });
$("btnCamera").addEventListener("click", startCamera);
$("btnPlay").addEventListener("click", () => {
  if (mode === "camera") { if (running) { running = false; $("btnPlay").textContent = "Resume"; } else { running = true; $("btnPlay").textContent = "Pause"; loop(); } return; }
  if (video.ended) { restart(); return; }
  running ? pause() : play();
});
function restart() {
  if (mode !== "file") return;
  resetCounts();
  video.currentTime = 0;
  lastT = -1;
  setStatus("Restarted.");
  play();
}
$("btnRestart").addEventListener("click", restart);
$("btnDraw").addEventListener("click", () => beginDraw());
$("btnDrawDone").addEventListener("click", () => endDraw(true));
$("btnDrawCancel").addEventListener("click", () => endDraw(false));
$("chkMotion").addEventListener("change", () => { if (!running) render(); });
$("btnCsv").addEventListener("click", downloadCsv);
$("selModel").addEventListener("change", async (e) => {
  const was = running;
  if (was) pause();
  await ensureDetector(e.target.value);
  if (was) { if (mode === "camera") { running = true; loop(); } else play(); }
});

// Expose a small hook for automated checks.
window.__crosswalk = { get counts() { return { ...counts }; }, get events() { return events.map(({ snapshot, crop, ...e }) => e); }, readPlate };
