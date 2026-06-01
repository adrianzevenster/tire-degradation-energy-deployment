const state = {
  running: false,
  timer: null,
  predictions: [],
  metrics: {},
  metricHistory: [],
  readiness: null,
  alerts: { health_score: 0, alerts: [] },
  comparison: [],
  artifacts: [],
  strategy: null,
  latestTelemetry: null,
  history: [],
  selectedRunId: null,
  historySortKey: "updated_at_ms",
  historySortDirection: "desc",
};

const el = (id) => document.getElementById(id);

function pct(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(0)}%`;
}

function num(value, digits = 2) {
  if (!Number.isFinite(value)) return "-";
  return Number(value).toFixed(digits);
}

function parseMetrics(text) {
  const metrics = {};
  for (const line of text.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const [name, value] = line.trim().split(/\s+/);
    const parsed = Number(value);
    if (Number.isFinite(parsed)) metrics[name] = parsed;
  }
  return metrics;
}

function metricValue(prefix) {
  const entry = Object.entries(state.metrics).find(([name]) => name.startsWith(prefix));
  return entry ? entry[1] : Number.NaN;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function setNotice(message, tone = "info") {
  const notice = el("appNotice");
  if (!message) {
    notice.hidden = true;
    notice.textContent = "";
    notice.className = "notice";
    return;
  }
  notice.hidden = false;
  notice.textContent = message;
  notice.className = `notice ${tone}`;
}

function setControls(status = null) {
  el("startButton").disabled = state.running;
  el("pauseButton").disabled = !state.running;
  el("tickButton").disabled = state.running;
  if (status) {
    el("simulationProgress").textContent =
      status.session_id ? `${status.index} / ${status.total} · ${status.session_id}` : `${status.index} / ${status.total}`;
  }
}

function setActiveTab(name) {
  const live = name === "live";
  const ops = name === "ops";
  el("liveTab").classList.toggle("active", live);
  el("opsTab").classList.toggle("active", ops);
  el("historyTab").classList.toggle("active", !live && !ops);
  el("liveTabButton").classList.toggle("active", live);
  el("opsTabButton").classList.toggle("active", ops);
  el("historyTabButton").classList.toggle("active", !live && !ops);
  if (ops) refreshOps();
  if (!live && !ops) refreshHistory();
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    el("modelValue").textContent = health.model_backend || "-";
    el("modelValue").title =
      `Feature schema ${health.feature_schema_version} / ${health.feature_schema_hash}`;
    el("serviceState").textContent =
      `${health.env} / ${health.persistence_backend} / ${health.model_backend} / schema ${String(health.feature_schema_hash || "").slice(0, 8)}`;
  } catch (error) {
    el("serviceState").textContent = "Offline";
    setNotice(`Service unavailable: ${error.message}`, "error");
  }
}

async function refreshModels() {
  try {
    const payload = await api("/models");
    el("modelSelect").value = payload.configured_backend || "auto";
    el("modelValue").textContent = payload.active_backend || "-";
    await refreshArtifacts();
  } catch (error) {
    setNotice(`Could not load model list: ${error.message}`, "error");
  }
}

async function refreshArtifacts() {
  const payload = await api("/artifacts");
  state.artifacts = payload.artifacts || [];
  const select = el("artifactSelect");
  const active = payload.active_artifact_id || "unregistered";
  select.innerHTML = `<option value="">Unregistered</option>`;
  for (const artifact of state.artifacts) {
    const option = document.createElement("option");
    option.value = artifact.artifact_id;
    option.textContent = `${artifact.status} · ${artifact.artifact_id}`;
    option.disabled = artifact.status !== "promoted";
    select.appendChild(option);
  }
  select.value = active === "unregistered" ? "" : active;
  select.title = active;
}

async function changeModelBackend() {
  const backend = el("modelSelect").value;
  try {
    state.running = false;
    clearTimeout(state.timer);
    const payload = await api(`/model/backend?backend=${encodeURIComponent(backend)}`, {
      method: "POST",
    });
    state.predictions = [];
    state.metrics = {};
    state.strategy = null;
    state.latestTelemetry = null;
    resetDashboard();
    setControls({ index: 0, total: 0, session_id: "" });
    el("modelValue").textContent = payload.active_backend || backend;
    el("artifactSelect").value = "";
    setNotice(`Model switched to ${payload.active_backend}. Simulation state was reset.`);
    await refreshHealth();
    await refreshArtifacts();
  } catch (error) {
    setNotice(`Could not switch model: ${error.message}`, "error");
    await refreshModels();
  }
}

async function changeModelArtifact() {
  const artifactId = el("artifactSelect").value;
  if (!artifactId) return;
  try {
    state.running = false;
    clearTimeout(state.timer);
    const payload = await api(`/model/artifact?artifact_id=${encodeURIComponent(artifactId)}`, {
      method: "POST",
    });
    state.predictions = [];
    state.metrics = {};
    state.strategy = null;
    state.latestTelemetry = null;
    resetDashboard();
    setControls({ index: 0, total: 0, session_id: "" });
    el("modelValue").textContent = payload.active_backend || "-";
    setNotice(`Artifact loaded: ${payload.active_artifact_id}. Simulation state was reset.`);
    await refreshHealth();
    await refreshArtifacts();
  } catch (error) {
    setNotice(`Could not load artifact: ${error.message}`, "error");
    await refreshArtifacts();
  }
}

async function refreshMetrics() {
  const text = await api("/metrics");
  state.metrics = parseMetrics(text);
  state.metricHistory.push({
    latency: state.metrics.f1_inference_latency_ms_p95,
    throughput: state.metrics.f1_throughput_events_total,
    health: metricValue("f1_model_health_score"),
    ready: metricValue("f1_deployment_ready"),
    mae: metricValue("f1_model_mae_lap_delta_s"),
    rmse: metricValue("f1_model_rmse_lap_delta_s"),
    coverage: metricValue("f1_model_interval_coverage_pct"),
    alerts: metricValue("f1_model_alerts_total"),
  });
  state.metricHistory = state.metricHistory.slice(-80);
  renderMetrics();
  renderOpsCharts();
}

async function refreshOps() {
  try {
    const [readiness, alerts] = await Promise.all([
      api("/deployment/readiness"),
      api("/monitoring/alerts"),
      refreshMetrics(),
    ]);
    state.readiness = readiness;
    state.alerts = alerts;
    state.comparison = (await api("/monitoring/model-comparison")).models || [];
    renderOps();
  } catch (error) {
    setNotice(`Could not load operations view: ${error.message}`, "error");
  }
}

async function startSimulation() {
  try {
    const laps = Number(el("lapsInput").value || 18);
    const seed = Number(el("seedInput").value || 7);
    const status = await api(`/simulation/start?laps=${laps}&seed=${seed}`, { method: "POST" });
    state.predictions = [];
    state.strategy = null;
    state.running = true;
    setNotice("");
    render(status);
    setControls(status);
    schedule();
  } catch (error) {
    state.running = false;
    setControls();
    setNotice(`Could not start simulation: ${error.message}`, "error");
  }
}

async function stopSimulation() {
  try {
    state.running = false;
    const status = await api("/simulation/stop", { method: "POST" });
    clearTimeout(state.timer);
    setControls(status);
  } catch (error) {
    setNotice(`Could not pause simulation: ${error.message}`, "error");
  }
}

async function resetSimulation() {
  try {
    state.running = false;
    clearTimeout(state.timer);
    const status = await api("/simulation/reset", { method: "POST" });
    state.predictions = [];
    state.metrics = {};
    state.metricHistory = [];
    state.readiness = null;
    state.alerts = { health_score: 0, alerts: [] };
    state.strategy = null;
    resetDashboard();
    render(status);
    setControls(status);
    await refreshHistory();
  } catch (error) {
    setNotice(`Could not reset simulation: ${error.message}`, "error");
  }
}

async function tickSimulation() {
  try {
    const batch = Number(el("batchInput").value || 1);
    const data = await api(`/simulation/tick?batch_size=${batch}&remaining_laps=30`, {
      method: "POST",
    });
    for (const prediction of data.predictions || []) {
      state.predictions.push(prediction);
    }
    if (data.telemetry?.length) {
      state.latestTelemetry = data.telemetry[data.telemetry.length - 1];
    }
    if (data.strategy) state.strategy = data.strategy;
    state.metrics = parseMetrics(data.metrics || "");
    await refreshOps();
    render(data.status);
    setControls(data.status);
    if (data.status.complete) {
      state.running = false;
      clearTimeout(state.timer);
      setControls(data.status);
      await refreshHistory();
    }
  } catch (error) {
    state.running = false;
    clearTimeout(state.timer);
    setControls();
    setNotice(`Simulation request failed: ${error.message}`, "error");
  }
}

function schedule() {
  clearTimeout(state.timer);
  if (!state.running) return;
  state.timer = setTimeout(async () => {
    try {
      await tickSimulation();
      schedule();
    } catch (error) {
      state.running = false;
      setNotice(`Simulation stopped: ${error.message}`, "error");
      setControls();
    }
  }, 700);
}

function latestPrediction() {
  return state.predictions[state.predictions.length - 1] || null;
}

function riskClass(value) {
  if (value >= 0.6) return "risk-high";
  if (value >= 0.3) return "risk-medium";
  return "risk-low";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function tireLoadColor(load) {
  const normalized = clamp(load / 100, 0, 1);
  const hue = 140 - normalized * 140;
  const lightness = 32 + normalized * 22;
  return `hsl(${hue} 78% ${lightness}%)`;
}

function renderTirePressureMap() {
  const event = state.latestTelemetry;
  if (!event) {
    el("pressureState").textContent = "Waiting";
    for (const id of ["tireFL", "tireFR", "tireRL", "tireRR"]) {
      const node = el(id);
      node.style.setProperty("--load-color", "#16212b");
      node.querySelector("strong").textContent = "-";
      node.querySelector("small").textContent = "-";
    }
    return;
  }

  const tempLoads = {
    FL: clamp((event.tire_temp_fl - 70) * 1.7, 0, 100),
    FR: clamp((event.tire_temp_fr - 70) * 1.7, 0, 100),
    RL: clamp((event.tire_temp_rl - 68) * 1.7, 0, 100),
    RR: clamp((event.tire_temp_rr - 68) * 1.7, 0, 100),
  };
  const lateral = clamp(Math.abs(event.lateral_g) / 5, 0, 1);
  const steering = clamp(event.steering_angle / 30, -1, 1);
  const rightBias = steering >= 0 ? lateral * 16 : -lateral * 16;
  const frontBias = event.brake * 22;
  const rearBias = event.throttle * 7;
  const slipBias = clamp(event.slip_angle / 8, 0, 1) * 12;
  const loads = {
    FL: tempLoads.FL + frontBias - rightBias + slipBias,
    FR: tempLoads.FR + frontBias + rightBias + slipBias,
    RL: tempLoads.RL + rearBias - rightBias * 0.45 + slipBias * 0.6,
    RR: tempLoads.RR + rearBias + rightBias * 0.45 + slipBias * 0.6,
  };
  const temps = {
    FL: event.tire_temp_fl,
    FR: event.tire_temp_fr,
    RL: event.tire_temp_rl,
    RR: event.tire_temp_rr,
  };
  for (const [corner, load] of Object.entries(loads)) {
    const node = el(`tire${corner}`);
    const clamped = clamp(load, 0, 100);
    node.style.setProperty("--load-color", tireLoadColor(clamped));
    node.querySelector("strong").textContent = `${num(clamped, 0)}%`;
    node.querySelector("small").textContent = `${num(temps[corner], 1)} C`;
  }
  const hottest = Object.entries(loads).sort((a, b) => b[1] - a[1])[0][0];
  el("pressureState").textContent = `${hottest} loaded · lap ${event.lap}.${event.sector}`;
  el("pressureBrake").textContent = pct(event.brake);
  el("pressureLateral").textContent = `${num(event.lateral_g, 2)}g`;
  el("pressureSlip").textContent = `${num(event.slip_angle, 1)} deg`;
}

function render(status = null) {
  const prediction = latestPrediction();
  if (prediction) {
    el("lapValue").textContent = prediction.lap;
    el("wearValue").textContent = `${num(prediction.tire_wear_pct, 1)}%`;
    el("cliffValue").textContent = pct(prediction.cliff_probability);
    el("cliffValue").className = riskClass(prediction.cliff_probability);
    el("ersValue").textContent = num(prediction.ers_efficiency, 2);
    el("deltaValue").textContent = `${num(prediction.next_lap_delta_s, 3)}s`;
    el("tireBand").textContent =
      prediction.cliff_probability > 0.5 ? "Cliff Risk" : "Thermal Window";
    el("paceBand").textContent = `Lap ${prediction.lap}`;
  }
  if (status) {
    setControls(status);
  }
  renderStrategy();
  renderMetrics();
  renderTirePressureMap();
  drawCharts();
  renderOpsCharts();
}

function resetDashboard() {
  for (const id of [
    "lapValue",
    "wearValue",
    "cliffValue",
    "ersValue",
    "deltaValue",
    "latencyValue",
    "modelValue",
    "pressureBrake",
    "pressureLateral",
    "pressureSlip",
    "pitWindowValue",
    "pitTarget",
    "undercutValue",
    "safetyValue",
    "paceTarget",
    "throughputValue",
    "intervalValue",
    "brakeValue",
    "overheatValue",
  ]) {
    el(id).textContent = "-";
  }
  el("eventCount").textContent = "0 events";
  el("tireBand").textContent = "Stable";
  el("paceBand").textContent = "Live";
  el("deploymentBars").innerHTML = "";
  el("reasonList").innerHTML = "";
  el("driftRows").innerHTML = `<tr><td>No baseline fitted</td><td>-</td></tr>`;
  el("metricStream").innerHTML = "";
  el("deploymentReadyValue").textContent = "-";
  el("healthScoreValue").textContent = "-";
  el("alertCountValue").textContent = "-";
  el("rollbackValue").textContent = "-";
  el("readinessChecks").innerHTML = "";
  el("alertList").innerHTML = "";
  el("comparisonRows").innerHTML = `<tr><td colspan="9">No evaluated models</td></tr>`;
  el("comparisonState").textContent = "No evaluated models";
  el("readinessState").textContent = "Waiting";
  el("alertState").textContent = "0 active";
  state.latestTelemetry = null;
  renderTirePressureMap();
  drawCharts();
}

function renderStrategy() {
  const strategy = state.strategy;
  if (!strategy) return;
  const pit = strategy.pit_window;
  el("pitWindowValue").textContent = `${pit.earliest_lap}-${pit.latest_lap}`;
  el("pitTarget").textContent = pit.target_lap;
  el("undercutValue").textContent = pct(pit.undercut_success_probability);
  el("safetyValue").textContent = pct(pit.safety_car_sensitivity);
  el("paceTarget").textContent = `${num(strategy.pace_target_delta_s, 3)}s`;

  const bars = el("deploymentBars");
  bars.innerHTML = "";
  const deployment = strategy.energy_plan.sector_deployment_kw || {};
  for (const [sector, value] of Object.entries(deployment)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span>Sector ${sector}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, value)}%"></div></div>
      <strong>${num(value, 1)} kW</strong>
    `;
    bars.appendChild(row);
  }

  const reasons = el("reasonList");
  reasons.innerHTML = "";
  for (const reason of strategy.reasons || []) {
    const item = document.createElement("li");
    item.textContent = reason;
    reasons.appendChild(item);
  }
}

function renderMetrics() {
  const metrics = state.metrics;
  el("latencyValue").textContent = `${num(metrics.f1_inference_latency_ms_p95, 3)} ms`;
  el("eventCount").textContent = `${num(metrics.f1_throughput_events_total, 0)} events`;
  el("throughputValue").textContent = `${num(metrics.f1_throughput_events_per_second, 1)}/s`;
  el("intervalValue").textContent = `${num(metrics.f1_prediction_interval_width_s, 3)}s`;
  el("brakeValue").textContent = `${num(metrics.f1_brake_temp_next_lap_c, 0)} C`;
  el("overheatValue").textContent = pct(metrics.f1_overheating_probability);

  const drift = Object.entries(metrics)
    .filter(([name]) => name.startsWith("f1_drift_z_score_"))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  const rows = el("driftRows");
  rows.innerHTML = "";
  if (!drift.length) {
    rows.innerHTML = `<tr><td>No baseline fitted</td><td>-</td></tr>`;
  } else {
    for (const [name, value] of drift) {
      const label = name.replace("f1_drift_z_score_", "").replaceAll("_", " ");
      rows.innerHTML += `<tr><td>${label}</td><td>${num(value, 2)}</td></tr>`;
    }
  }

  const stream = el("metricStream");
  stream.innerHTML = "";
  const selected = [
    "f1_tire_wear_pct",
    "f1_tire_cliff_probability",
    "f1_ers_efficiency",
    "f1_next_lap_delta_s",
    "f1_prediction_interval_width_s",
    "f1_undercut_success_probability",
    "f1_ers_expected_lap_gain_s",
    "f1_drift_alerts_total",
  ];
  for (const name of selected) {
    const chip = document.createElement("div");
    chip.className = "metric-chip";
    chip.innerHTML = `<span>${name.replace("f1_", "")}</span><strong>${num(metrics[name], 3)}</strong>`;
    stream.appendChild(chip);
  }
}

function renderOps() {
  const readiness = state.readiness;
  const alerts = state.alerts?.alerts || readiness?.alerts || [];
  if (!readiness) {
    el("readinessState").textContent = "Waiting";
    return;
  }
  el("deploymentReadyValue").textContent = readiness.ready ? "Ready" : "Blocked";
  el("deploymentReadyValue").className = readiness.ready ? "risk-low" : "risk-high";
  el("healthScoreValue").textContent = `${num(readiness.health_score, 0)}`;
  el("alertCountValue").textContent = readiness.alert_count;
  const rollback = readiness.rollback_candidate;
  el("rollbackValue").textContent = rollback ? shortRunId(rollback.artifact_id) : "None";
  el("rollbackValue").title = rollback?.artifact_id || "";
  el("readinessState").textContent =
    `${readiness.active_backend} / ${readiness.active_artifact_id}`;
  el("alertState").textContent = `${alerts.length} active`;

  const checks = el("readinessChecks");
  checks.innerHTML = "";
  for (const [name, passed] of Object.entries(readiness.checks || {})) {
    const item = document.createElement("div");
    item.className = `check-item ${passed ? "pass" : "fail"}`;
    item.innerHTML = `<span>${name.replaceAll("_", " ")}</span><strong>${passed ? "Pass" : "Fail"}</strong>`;
    checks.appendChild(item);
  }

  const list = el("alertList");
  list.innerHTML = "";
  if (!alerts.length) {
    list.innerHTML = `<div class="alert-item"><span>No active alerts</span><strong>Clear</strong></div>`;
  } else {
    for (const alert of alerts) {
      const item = document.createElement("div");
      item.className = `alert-item ${alert.severity}`;
      item.innerHTML = `
        <span>${alert.type.replaceAll("_", " ")}</span>
        <strong>${alert.severity}</strong>
        <small>${alert.message}</small>
      `;
      list.appendChild(item);
    }
  }
  renderOpsCharts();
  renderModelComparison();
}

function renderModelComparison() {
  const rows = state.comparison || [];
  const table = el("comparisonRows");
  table.innerHTML = "";
  el("comparisonState").textContent = rows.length ? `${rows.length} evaluated` : "No evaluated models";
  if (!rows.length) {
    table.innerHTML = `<tr><td colspan="9">Run at least one full lap per model</td></tr>`;
    drawComparisonChart([]);
    return;
  }
  for (const row of rows) {
    const item = document.createElement("tr");
    item.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.backend}</td>
      <td title="${row.artifact_id}">${shortRunId(row.artifact_id)}</td>
      <td>${row.evaluations}</td>
      <td>${num(row.mae_lap_delta_s, 4)}s</td>
      <td>${num(row.rmse_lap_delta_s, 4)}s</td>
      <td>${num(row.interval_coverage_pct, 1)}%</td>
      <td>${num(row.health_score, 0)}</td>
      <td>${row.alert_count}</td>
    `;
    table.appendChild(item);
  }
  drawComparisonChart(rows);
}

function renderOpsCharts() {
  const rows = state.metricHistory;
  el("modelMetricBand").textContent = rows.length ? `${rows.length} samples` : "Waiting";
  el("serviceMetricBand").textContent = rows.length ? `${rows.length} samples` : "Waiting";
  drawLineChart(el("modelMetricsChart"), rows, [
    { key: "health", label: "Health", color: "#58d68d", fixedMax: 100, unit: "" },
    { key: "mae", label: "MAE", color: "#ff6b6b", unit: "s" },
    { key: "rmse", label: "RMSE", color: "#f5c45c", unit: "s" },
    { key: "coverage", label: "Coverage", color: "#62a8ff", fixedMax: 100, unit: "%" },
  ], { xLabel: "Prometheus scrapes" });
  drawLineChart(el("serviceMetricsChart"), rows, [
    { key: "latency", label: "p95 latency", color: "#62a8ff", unit: "ms" },
    { key: "ready", label: "Ready", color: "#58d68d", fixedMax: 1, unit: "" },
    { key: "alerts", label: "Alerts", color: "#ff6b6b", unit: "" },
    { key: "throughput", label: "Events", color: "#49c6d1", unit: "" },
  ], { xLabel: "Prometheus scrapes" });
}

function drawComparisonChart(rows) {
  const chartRows = rows.map((row) => ({
    mae: row.mae_lap_delta_s * 100,
    rmse: row.rmse_lap_delta_s * 100,
    coverage: row.interval_coverage_pct,
    health: row.health_score,
  }));
  drawLineChart(el("comparisonChart"), chartRows, [
    { key: "mae", label: "MAE x100", color: "#ff6b6b", unit: "" },
    { key: "rmse", label: "RMSE x100", color: "#f5c45c", unit: "" },
    { key: "coverage", label: "Coverage", color: "#62a8ff", fixedMax: 100, unit: "%" },
    { key: "health", label: "Health", color: "#58d68d", fixedMax: 100, unit: "" },
  ], { xLabel: "Ranked models" });
}

function drawCharts() {
  const tire = state.predictions.map((p) => ({
    wear: p.tire_wear_pct,
    cliff: p.cliff_probability * 100,
    heat: p.overheating_probability * 100,
  }));
  const pace = state.predictions.map((p) => ({
    delta: p.next_lap_delta_s,
    low: p.uncertainty_low_s,
    high: p.uncertainty_high_s,
    ers: p.ers_efficiency,
  }));
  drawLineChart(el("tireChart"), tire, [
    { key: "wear", label: "Wear %", color: "#ff6b6b", fixedMax: 100, unit: "%" },
    { key: "cliff", label: "Cliff %", color: "#f5c45c", fixedMax: 100, unit: "%" },
    { key: "heat", label: "Overheat %", color: "#62a8ff", fixedMax: 100, unit: "%" },
  ], { yUnit: "%", xLabel: "Telemetry events" });
  drawLineChart(el("paceChart"), pace, [
    { key: "delta", label: "Delta", color: "#62a8ff", unit: "s" },
    { key: "low", label: "Low", color: "#92a4b2", unit: "s" },
    { key: "high", label: "High", color: "#92a4b2", unit: "s" },
    { key: "ers", label: "ERS", color: "#58d68d", unit: "" },
  ], { yUnit: "s / ratio", xLabel: "Telemetry events" });
}

async function refreshHistory() {
  try {
    const payload = await api("/history/runs?limit=10");
    state.history = payload.runs || [];
    if (!state.selectedRunId && state.history.length) {
      state.selectedRunId = state.history[0].session_id;
    }
    const backend = payload.persistence_backend || "unknown";
    el("historyState").textContent =
      backend === "none" ? "Install DuckDB extra for durable history" : `${backend} · ${state.history.length} runs`;
    renderHistory();
  } catch (error) {
    el("historyState").textContent = "History unavailable";
    setNotice(`Could not load run history: ${error.message}`, "error");
  }
}

function renderHistory() {
  const rows = el("historyRows");
  rows.innerHTML = "";
  renderHistorySummary();
  if (!state.history.length) {
    rows.innerHTML = `<tr><td colspan="12">No saved runs</td></tr>`;
    renderSelectedRun(null);
    drawHistoryChart();
    return;
  }
  const sortedHistory = sortedRuns();
  if (!sortedHistory.some((run) => run.session_id === state.selectedRunId)) {
    state.selectedRunId = sortedHistory[0].session_id;
  }
  for (const run of sortedHistory) {
    const strategy = run.latest_strategy || {};
    const pit = strategy.pit_target_lap || "-";
    const range = `${num(run.min_lap_delta_s, 3)} to ${num(run.max_lap_delta_s, 3)}s`;
    const row = document.createElement("tr");
    row.className = run.session_id === state.selectedRunId ? "selected" : "";
    row.tabIndex = 0;
    row.title = "Select run";
    row.innerHTML = `
      <td title="${run.session_id}">${shortRunId(run.session_id)}</td>
      <td>${run.prediction_count}</td>
      <td>${run.latest_lap}</td>
      <td>${num(run.max_tire_wear_pct, 1)}%</td>
      <td>${pct(run.max_cliff_probability)}</td>
      <td>${pct(run.max_overheating_probability)}</td>
      <td>${num(run.avg_lap_delta_s, 3)}s</td>
      <td>${range}</td>
      <td>${num(run.avg_ers_efficiency, 3)}</td>
      <td>${num(run.avg_prediction_interval_width_s, 3)}s</td>
      <td>${num(run.avg_latency_ms, 2)} / ${num(run.max_latency_ms, 2)} ms</td>
      <td>${pit}</td>
    `;
    row.addEventListener("click", () => selectRun(run.session_id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectRun(run.session_id);
      }
    });
    rows.appendChild(row);
  }
  renderSelectedRun(sortedHistory.find((run) => run.session_id === state.selectedRunId));
  drawHistoryChart();
}

function sortedRuns() {
  const direction = state.historySortDirection === "asc" ? 1 : -1;
  return state.history.slice().sort((left, right) => {
    const leftValue = left[state.historySortKey];
    const rightValue = right[state.historySortKey];
    if (leftValue === rightValue) return 0;
    return leftValue > rightValue ? direction : -direction;
  });
}

function selectRun(sessionId) {
  state.selectedRunId = sessionId;
  renderHistory();
}

function setHistorySort(key) {
  state.historySortKey = key;
  renderHistory();
}

function toggleHistorySortDirection() {
  state.historySortDirection = state.historySortDirection === "desc" ? "asc" : "desc";
  el("historySortDirectionButton").textContent =
    state.historySortDirection === "desc" ? "Desc" : "Asc";
  renderHistory();
}

function renderHistorySummary() {
  el("historyRunCount").textContent = state.history.length;
  if (!state.history.length) {
    el("historyBestDelta").textContent = "-";
    el("historyWorstCliff").textContent = "-";
    el("historyAvgLatency").textContent = "-";
    return;
  }
  const bestDelta = Math.min(...state.history.map((run) => run.avg_lap_delta_s));
  const worstCliff = Math.max(...state.history.map((run) => run.max_cliff_probability));
  const avgLatency =
    state.history.reduce((total, run) => total + run.avg_latency_ms, 0) / state.history.length;
  el("historyBestDelta").textContent = `${num(bestDelta, 3)}s`;
  el("historyWorstCliff").textContent = pct(worstCliff);
  el("historyAvgLatency").textContent = `${num(avgLatency, 2)} ms`;
}

function renderSelectedRun(run) {
  const details = el("selectedRunDetails");
  details.innerHTML = "";
  if (!run) {
    el("selectedRunTitle").textContent = "Click a run";
    details.innerHTML = `<div class="detail-item"><span>Status</span><strong>No run selected</strong></div>`;
    return;
  }
  const strategy = run.latest_strategy || {};
  const prediction = run.latest_prediction || {};
  el("selectedRunTitle").textContent = run.session_id;
  const items = [
    ["Samples", run.prediction_count],
    ["Latest Lap", run.latest_lap],
    ["Max Wear", `${num(run.max_tire_wear_pct, 1)}%`],
    ["Max Cliff", pct(run.max_cliff_probability)],
    ["Max Overheat", pct(run.max_overheating_probability)],
    ["Avg Delta", `${num(run.avg_lap_delta_s, 3)}s`],
    ["Delta Range", `${num(run.min_lap_delta_s, 3)} to ${num(run.max_lap_delta_s, 3)}s`],
    ["Avg ERS", num(run.avg_ers_efficiency, 3)],
    ["Interval Width", `${num(run.avg_prediction_interval_width_s, 3)}s`],
    ["Avg Latency", `${num(run.avg_latency_ms, 2)} ms`],
    ["Max Latency", `${num(run.max_latency_ms, 2)} ms`],
    ["Target Pit", strategy.pit_target_lap || "-"],
    ["Pit Window", strategy.pit_window ? `${strategy.pit_window.earliest_lap}-${strategy.pit_window.latest_lap}` : "-"],
    ["Latest Tire", `${num(prediction.tire_wear_pct, 1)}%`],
    ["Latest Delta", `${num(prediction.next_lap_delta_s, 3)}s`],
    ["Latest ERS", num(prediction.ers_efficiency, 3)],
    ["Car", run.car_id],
    ["Run", shortRunId(run.session_id)],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    details.appendChild(item);
  }
}

function shortRunId(value) {
  if (!value) return "-";
  return value.length > 18 ? `...${value.slice(-15)}` : value;
}

function drawHistoryChart() {
  const rows = sortedRuns()
    .slice()
    .reverse()
    .map((run) => ({
      wear: run.max_tire_wear_pct,
      cliff: run.max_cliff_probability * 100,
      overheat: run.max_overheating_probability * 100,
      delta: run.avg_lap_delta_s * 25,
    }));
  drawLineChart(el("historyChart"), rows, [
    { key: "wear", label: "Max wear %", color: "#ff6b6b", fixedMax: 100, unit: "%" },
    { key: "cliff", label: "Max cliff %", color: "#f5c45c", fixedMax: 100, unit: "%" },
    { key: "overheat", label: "Max overheat %", color: "#62a8ff", fixedMax: 100, unit: "%" },
    { key: "delta", label: "Avg delta x25", color: "#49c6d1", fixedMax: 100, unit: "" },
  ], { yUnit: "% / scaled", xLabel: "Saved runs" });
}

function formatAxisValue(value, unit = "") {
  if (!Number.isFinite(value)) return "-";
  const digits = Math.abs(value) >= 10 ? 0 : 2;
  return `${Number(value).toFixed(digits)}${unit}`;
}

function drawLineChart(canvas, rows, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const plot = {
    left: 58,
    right: width - 18,
    top: 42,
    bottom: height - 34,
  };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101820";
  ctx.fillRect(0, 0, width, height);

  ctx.font = "20px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#314452";
  ctx.textAlign = "center";
  ctx.fillText(rows.length < 2 ? "Waiting for data" : "", width / 2, height / 2);

  ctx.strokeStyle = "#243442";
  ctx.lineWidth = 1;

  const values = [];
  for (const row of rows) {
    for (const item of series) {
      const value = row[item.key];
      if (Number.isFinite(value)) {
        values.push(item.fixedMax ? Math.min(item.fixedMax, value) : value);
      }
    }
  }
  const fixedMax = series.every((item) => item.fixedMax === 100);
  const min = fixedMax ? 0 : Math.min(...values, -0.2);
  const max = fixedMax ? 100 : Math.max(...values, 1.2);
  const span = max - min || 1;

  for (let i = 0; i < 5; i += 1) {
    const y = plot.top + i * ((plot.bottom - plot.top) / 4);
    const tickValue = max - i * (span / 4);
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.fillStyle = "#92a4b2";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(formatAxisValue(tickValue, fixedMax ? "%" : ""), plot.left - 8, y + 4);
  }

  ctx.fillStyle = "#92a4b2";
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(options.xLabel || "", plot.left, height - 8);
  ctx.textAlign = "right";
  ctx.fillText(rows.length ? `n=${rows.length}` : "n=0", plot.right, height - 8);

  const latest = rows[rows.length - 1] || {};
  let legendX = plot.left;
  for (const item of series) {
    const value = latest[item.key];
    const label = `${item.label}: ${formatAxisValue(value, item.unit)}`;
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, 14, 16, 3);
    ctx.fillStyle = "#c4d2dc";
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(label, legendX + 21, 18);
    legendX += Math.min(190, 48 + label.length * 7);
  }

  if (rows.length < 2) return;

  const xFor = (index) =>
    plot.left + index * ((plot.right - plot.left) / Math.max(1, rows.length - 1));
  const yFor = (value) => plot.bottom - ((value - min) / span) * (plot.bottom - plot.top);

  for (const item of series) {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.key === "low" || item.key === "high" ? 1.5 : 2.5;
    ctx.beginPath();
    rows.forEach((row, index) => {
      const value = item.fixedMax ? Math.min(item.fixedMax, row[item.key]) : row[item.key];
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const lastValue = item.fixedMax
      ? Math.min(item.fixedMax, latest[item.key])
      : latest[item.key];
    if (Number.isFinite(lastValue)) {
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(xFor(rows.length - 1), yFor(lastValue), 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

el("startButton").addEventListener("click", startSimulation);
el("pauseButton").addEventListener("click", stopSimulation);
el("tickButton").addEventListener("click", tickSimulation);
el("resetButton").addEventListener("click", resetSimulation);
el("modelSelect").addEventListener("change", changeModelBackend);
el("artifactSelect").addEventListener("change", changeModelArtifact);
el("historyRefreshButton").addEventListener("click", refreshHistory);
el("historySortSelect").addEventListener("change", (event) => setHistorySort(event.target.value));
el("historySortDirectionButton").addEventListener("click", toggleHistorySortDirection);
el("liveTabButton").addEventListener("click", () => setActiveTab("live"));
el("historyTabButton").addEventListener("click", () => setActiveTab("history"));
el("opsTabButton").addEventListener("click", () => setActiveTab("ops"));

refreshHealth();
refreshModels();
refreshMetrics().catch(() => {});
refreshOps().catch(() => {});
refreshHistory();
setControls();
setInterval(refreshHealth, 3000);
setInterval(refreshOps, 5000);
