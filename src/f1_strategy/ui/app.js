const state = {
  running: false,
  timer: null,
  predictions: [],
  metrics: {},
  metricHistory: [],
  readiness: null,
  alerts: { health_score: 0, alerts: [] },
  comparison: [],
  replayEvaluation: null,
  replaySuite: null,
  replayBenchmark: null,
  replayRun: null,
  regressionRun: null,
  smokeRun: null,
  runChecks: { latest: { replay: null, regression: null, smoke: null }, history: { replay: [], regression: [], smoke: [] } },
  replayDatasets: [],
  selectedReplayDataset: "examples/replay_telemetry.csv",
  trainingReplayDataset: "examples/replay_telemetry.csv",
  datasetManifest: null,
  artifacts: [],
  artifactRegistry: { promoted: {}, active_artifact_id: "unregistered" },
  selectedArtifactId: null,
  artifactDetail: null,
  strategy: null,
  latestTelemetry: null,
  history: [],
  selectedRunId: null,
  historySortKey: "updated_at_ms",
  historySortDirection: "desc",
  shadow: null,
  health: null,
  liveMode: "replay",        // "replay" | "live"
  liveStatus: null,
  livePredictions: [],
  liveTimer: null,
  trainingJobs: [],
  activeTrainingJobId: null,
  trainingPollTimer: null,
  externalLinks: [],
  featureImportance: null,
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

function replayTrust(provenance = {}) {
  const signal = provenance.validation_signal || "-";
  const source = provenance.source || "";
  const label = provenance.lap_time_label || "-";
  if (provenance.production_validation_ready === true) {
    return { label: "Production", className: "trust-production", title: "Observed public lap-time validation" };
  }
  if (source === "benchmark-fixture") {
    return { label: "Benchmark", className: "trust-benchmark", title: "Manifested benchmark fixture, not production validation" };
  }
  if (signal === "synthetic" || label === "synthetic") {
    return { label: "Synthetic", className: "trust-synthetic", title: "Simulator or synthetic fixture" };
  }
  if (signal === "unprovenanced") {
    return { label: "No Manifest", className: "trust-missing", title: "No sidecar provenance manifest" };
  }
  if (signal === "proxy-heavy") {
    return { label: "Proxy", className: "trust-proxy", title: "Manifested data with proxy-heavy validation fields" };
  }
  return { label: signal, className: "trust-unknown", title: signal };
}

function trustBadge(provenance = {}) {
  const trust = replayTrust(provenance);
  return `<span class="trust-badge ${trust.className}" title="${trust.title}">${trust.label}</span>`;
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

function latestRun(kind) {
  return state.runChecks?.latest?.[kind] || null;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const type = response.headers.get("content-type") || "";
    if (type.includes("application/json")) {
      const body = await response.json();
      const err = new Error(`${response.status} ${response.statusText}`);
      err.detail = body.detail;
      throw err;
    }
    throw new Error(`${response.status} ${response.statusText}`);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response.text();
}

function serializeUrl(url) {
  const path = url.pathname === "/" ? "" : url.pathname;
  return `${url.protocol}//${url.host}${path}${url.search}${url.hash}`;
}

function resolveInfraUrl(rawUrl) {
  if (!rawUrl) return "";
  try {
    const url = new URL(rawUrl, window.location.origin);
    return serializeUrl(url);
  } catch (_error) {
    return rawUrl;
  }
}

function defaultExternalServices() {
  return [
    {
      id: "mlflow",
      label: "MLflow",
      url: "http://localhost:5000",
      status: "available",
      external: true,
      hint: "MLflow tracking UI",
    },
    {
      id: "grafana",
      label: "Grafana",
      url: "http://localhost:3000",
      status: "available",
      external: true,
      hint: "Grafana dashboards",
    },
    {
      id: "prometheus",
      label: "Prometheus",
      url: "http://localhost:9090",
      status: "available",
      external: true,
      hint: "Prometheus metrics browser",
    },
    { id: "api-docs", label: "API Docs", url: "/docs", status: "available", external: false, hint: "" },
    { id: "metrics", label: "Metrics", url: "/metrics", status: "available", external: false, hint: "" },
  ];
}

function mergeExternalServices(baseServices, fetchedServices) {
  const fetchedById = new Map((fetchedServices || []).map((service) => [service.id, service]));
  return baseServices.map((service) => ({ ...service, ...(fetchedById.get(service.id) || {}) }));
}

function renderExternalLinks(services = []) {
  const row = el("externalLinksRow");
  row.innerHTML = "";
  const downCount = services.filter((service) => service.external && service.status !== "available").length;
  el("externalLinksState").textContent = downCount ? `${downCount} unchecked` : "Available";

  for (const service of services) {
    const configured = Boolean(service.url);
    const resolvedUrl = configured ? resolveInfraUrl(service.url) : "";
    const node = document.createElement(configured ? "a" : "span");
    node.className = `ext-link ${configured ? "" : "disabled"}`.trim();
    node.textContent = service.label;
    node.title = configured
      ? `${resolvedUrl}${service.status === "unavailable" ? " (not reachable from API probe)" : ""}`
      : `${service.label} unavailable. ${service.hint || ""}`.trim();
    node.dataset.status = service.status || "unknown";
    if (configured) {
      node.setAttribute("href", resolvedUrl);
      node.target = "_blank";
      node.rel = "noopener";
    }
    row.appendChild(node);
  }
}

async function refreshExternalLinks() {
  try {
    const payload = await api("/integrations/external-links");
    state.externalLinks = mergeExternalServices(defaultExternalServices(), payload.services || []);
    renderExternalLinks(state.externalLinks);
  } catch (_error) {
    el("externalLinksState").textContent = "Unavailable";
    renderExternalLinks(defaultExternalServices());
  }
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

function setTabVisibility(id, active) {
  const node = el(id);
  node.hidden = !active;
  node.style.display = active ? "grid" : "none";
}

function setActiveTab(name) {
  const live = name === "live";
  const evaluation = name === "evaluation";
  const promotion = name === "promotion";
  const training = name === "training";
  setTabVisibility("liveTab", live);
  setTabVisibility("evaluationTab", evaluation);
  setTabVisibility("promotionTab", promotion);
  setTabVisibility("trainingTab", training);
  setTabVisibility("historyTab", !live && !evaluation && !promotion && !training);
  el("liveTabButton").classList.toggle("active", live);
  el("evaluationTabButton").classList.toggle("active", evaluation);
  el("promotionTabButton").classList.toggle("active", promotion);
  el("trainingTabButton").classList.toggle("active", training);
  el("historyTabButton").classList.toggle("active", !live && !evaluation && !promotion && !training);
  if (evaluation || promotion || training) {
    refreshOps();
    refreshBenchmark();
  }
  if (!live && !evaluation && !promotion && !training) refreshHistory();
}

async function refreshHealth() {
  try {
    const health = await api("/health");
    state.health = health;
    el("modelValue").textContent = health.model_backend || "-";
    el("modelValue").title =
      `Feature schema ${health.feature_schema_version} / ${health.feature_schema_hash}`;
    const fs = health.feature_store_backend || "memory";
    const driftState = health.drift_baseline_fitted
      ? "drift active"
      : `drift warmup ${health.drift_ingest_count || 0}/30`;
    el("serviceState").textContent =
      `${health.env} / ${health.persistence_backend} / fs:${fs} / ${health.model_backend} / ${driftState}`;
    renderDriftWarmup(health.drift_baseline_fitted, health.drift_ingest_count || 0);
  } catch (error) {
    el("serviceState").textContent = "Offline";
    setNotice(`Service unavailable: ${error.message}`, "error");
  }
}

function renderDriftWarmup(fitted, ingestCount) {
  const bar = el("driftWarmupBar");
  if (!bar) return;
  if (fitted) {
    bar.hidden = true;
    return;
  }
  const pct = Math.min(100, Math.round((ingestCount / 30) * 100));
  bar.hidden = false;
  el("driftWarmupLabel").textContent = `Fitting baseline (${ingestCount}/30 events)`;
  el("driftWarmupFill").style.width = `${pct}%`;
  el("driftWarmupPct").textContent = `${pct}%`;
}

async function refreshModels() {
  try {
    const payload = await api("/models");
    el("modelSelect").value = payload.configured_backend || "auto";
    el("modelValue").textContent = payload.active_backend || "-";
  } catch (error) {
    setNotice(`Could not load model list: ${error.message}`, "error");
    return;
  }

  try {
    await refreshArtifacts();
  } catch (error) {
    setNotice(`Could not load artifact list: ${error.message}`, "error");
  }
}

async function refreshArtifacts() {
  const payload = await api("/artifacts");
  state.artifacts = payload.artifacts || [];
  state.artifactRegistry = payload;
  const select = el("artifactSelect");
  const active = payload.active_artifact_id || "unregistered";
  select.innerHTML = `<option value="">Unregistered</option>`;
  for (const artifact of state.artifacts) {
    const option = document.createElement("option");
    option.value = artifact.artifact_id;
    option.textContent = `${artifact.status} · ${artifact.artifact_id}`;
    select.appendChild(option);
  }
  select.value = active === "unregistered" ? "" : active;
  select.title = active;
  const selected =
    (active !== "unregistered" && state.artifacts.some((item) => item.artifact_id === active))
      ? active
      : state.selectedArtifactId;
  if (!selected && state.artifacts.length) {
    state.selectedArtifactId = state.artifacts[0].artifact_id;
  } else if (selected) {
    state.selectedArtifactId = selected;
  }
  await refreshArtifactDetail();
}

async function refreshArtifactDetail() {
  if (!state.selectedArtifactId) {
    state.artifactDetail = null;
    renderArtifactRelease();
    return;
  }
  try {
    state.artifactDetail = await api(`/artifacts/${encodeURIComponentPath(state.selectedArtifactId)}`);
  } catch (error) {
    state.artifactDetail = null;
    setNotice(`Could not load artifact detail: ${error.message}`, "error");
  }
  renderArtifactRelease();
}

function encodeURIComponentPath(value) {
  return String(value).split("/").map(encodeURIComponent).join("/");
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
  state.metrics = await api("/monitoring/metrics-summary");
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

async function settledRequest(label, promise) {
  try {
    return { label, status: "fulfilled", value: await promise };
  } catch (error) {
    return { label, status: "rejected", reason: error };
  }
}

async function refreshOps() {
  try {
    const datasetPayload = await api("/data-sources/replay-datasets");
    state.replayDatasets = datasetPayload.datasets || [];
    if (
      !state.selectedReplayDataset
      || !state.replayDatasets.some((item) => item.path === state.selectedReplayDataset)
    ) {
      state.selectedReplayDataset =
      state.replayDatasets.find((item) => item.path === "examples/replay_telemetry.csv")?.path
      || state.replayDatasets[0]?.path
      || "examples/replay_telemetry.csv";
    }
    const [
      readinessResult,
      alertsResult,
      replayEvaluationResult,
      replaySuiteResult,
      trainingJobsResult,
      runChecksResult,
      metricsResult,
    ] = await Promise.all([
      settledRequest("deployment/readiness", api("/deployment/readiness?mode=production")),
      settledRequest("monitoring/alerts", api("/monitoring/alerts")),
      settledRequest(
        "evaluation/replay",
        api(`/evaluation/replay?dataset_path=${encodeURIComponent(state.selectedReplayDataset)}`)
      ),
      settledRequest("evaluation/replay-suite", api("/evaluation/replay-suite")),
      settledRequest("training/jobs", api("/training/jobs")),
      settledRequest("ops/run-checks", api("/ops/run-checks")),
      settledRequest("metrics", refreshMetrics()),
    ]);
    state.readiness = readinessResult.status === "fulfilled" ? readinessResult.value : state.readiness;
    state.alerts = alertsResult.status === "fulfilled" ? alertsResult.value : state.alerts;
    state.replayEvaluation =
      replayEvaluationResult.status === "fulfilled" ? replayEvaluationResult.value : state.replayEvaluation;
    state.replaySuite = replaySuiteResult.status === "fulfilled" ? replaySuiteResult.value : state.replaySuite;
    state.trainingJobs = trainingJobsResult.status === "fulfilled" ? (trainingJobsResult.value.jobs || []) : state.trainingJobs;
    state.runChecks = runChecksResult.status === "fulfilled" ? (runChecksResult.value || state.runChecks) : state.runChecks;
    state.replayRun = latestRun("replay");
    state.regressionRun = latestRun("regression");
    state.smokeRun = latestRun("smoke");
    await refreshDatasetManifest();
    const comparison = await api("/monitoring/model-comparison");
    state.comparison = comparison.models || [];
    try {
      state.featureImportance = await api("/monitoring/feature-importance");
    } catch (_err) {
      // feature importance optional — no model trained yet
    }
    try {
      state.shadow = await api("/shadow/status");
    } catch (_err) {
      // shadow endpoint optional
    }
    if (readinessResult.status === "rejected" || alertsResult.status === "rejected" || replayEvaluationResult.status === "rejected" || replaySuiteResult.status === "rejected" || trainingJobsResult.status === "rejected" || runChecksResult.status === "rejected" || metricsResult.status === "rejected") {
      const failures = [
        readinessResult,
        alertsResult,
        replayEvaluationResult,
        replaySuiteResult,
        trainingJobsResult,
        runChecksResult,
        metricsResult,
      ]
        .filter((result) => result.status === "rejected")
        .map((result) => `${result.label}: ${result.reason?.detail || result.reason?.message || String(result.reason)}`);
      setNotice(`Evaluation view loaded with partial data: ${failures[0]}`, "error");
    }
    renderOps();
  } catch (error) {
    setNotice(`Could not load operations view: ${error.message}`, "error");
  }
}

async function refreshBenchmark() {
  try {
    state.replayBenchmark = await api("/evaluation/replay-benchmark");
    renderOps();
  } catch (error) {
    setNotice(`Could not load benchmark: ${error.message}`, "error");
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

async function refreshDatasetManifest() {
  const selected = state.replayDatasets.find((item) => item.path === state.selectedReplayDataset);
  if (!selected?.has_manifest) {
    state.datasetManifest = null;
    return;
  }
  try {
    state.datasetManifest = await api(
      `/data-sources/replay-datasets/${encodeURIComponentPath(state.selectedReplayDataset)}/manifest`
    );
  } catch (_error) {
    state.datasetManifest = null;
  }
}

async function selectReplayDataset(path) {
  state.selectedReplayDataset = path;
  await refreshDatasetManifest();
  await refreshOps();
}

async function runReplayCheck() {
  const kind = el("replayRunKindSelect").value;
  const payload = {
    kind,
    dataset_path: state.selectedReplayDataset,
  };
  try {
    el("replayRunButton").disabled = true;
    el("runCheckState").textContent = "Running replay…";
    const result = await api("/evaluation/replay/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.replayRun = result;
    el("runCheckState").textContent = `Replay ${kind}`;
    renderRunChecks();
  } catch (error) {
    setNotice(`Could not run replay check: ${error.message}`, "error");
    el("runCheckState").textContent = "Replay error";
  } finally {
    el("replayRunButton").disabled = false;
  }
}

async function runRegressionCheck() {
  const payload = {
    laps: Number(el("regressionLapsInput").value || 18),
    seed: Number(el("regressionSeedInput").value || 11),
    target_latency_ms: el("regressionLatencyInput").value.trim() ? Number(el("regressionLatencyInput").value) : null,
    max_temporal_oscillation_s: el("regressionOscillationInput").value.trim() ? Number(el("regressionOscillationInput").value) : null,
    min_calibration_width_s: Number(el("regressionMinWidthInput").value || 0.2),
    max_calibration_width_s: el("regressionMaxWidthInput").value.trim() ? Number(el("regressionMaxWidthInput").value) : null,
  };
  try {
    el("regressionRunButton").disabled = true;
    el("runCheckState").textContent = "Running regression…";
    const result = await api("/regression/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.regressionRun = result;
    el("runCheckState").textContent = result.passed ? "Regression pass" : "Regression fail";
    renderRunChecks();
  } catch (error) {
    setNotice(`Could not run regression suite: ${error.message}`, "error");
    el("runCheckState").textContent = "Regression error";
  } finally {
    el("regressionRunButton").disabled = false;
  }
}

async function runSmokeCheck() {
  const payload = {
    replay_dataset_path: state.selectedReplayDataset,
    regression_laps: Number(el("regressionLapsInput").value || 18),
    regression_seed: Number(el("regressionSeedInput").value || 11),
    probe_external_links: true,
  };
  try {
    el("smokeRunButton").disabled = true;
    el("runCheckState").textContent = "Running smoke…";
    const result = await api("/deployment/smoke", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.smokeRun = result;
    el("runCheckState").textContent = result.passed ? "Smoke pass" : "Smoke fail";
    renderRunChecks();
  } catch (error) {
    setNotice(`Could not run deployment smoke: ${error.message}`, "error");
    el("runCheckState").textContent = "Smoke error";
  } finally {
    el("smokeRunButton").disabled = false;
  }
}

function setExportSourceTab(source) {
  el("fastf1ExportForm").hidden = source !== "fastf1";
  el("openf1ExportForm").hidden = source !== "openf1";
  el("fastf1TabBtn").classList.toggle("active", source === "fastf1");
  el("openf1TabBtn").classList.toggle("active", source === "openf1");
}

async function exportOpenF1Session() {
  const payload = {
    year: Number(el("openf1YearInput").value || 2024),
    event: el("openf1EventInput").value || "Bahrain",
    session: el("openf1SessionInput").value || "Race",
    driver: el("openf1DriverInput").value || "VER",
  };
  try {
    el("openf1ExportButton").disabled = true;
    el("openf1ExportButton").textContent = "Exporting…";
    const manifest = await api("/data-sources/openf1/export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.selectedReplayDataset = manifest.output;
    const laps = manifest.lap_count || "?";
    const ageObs = manifest.tyre_age_observed ? " · tyre_age observed ✓" : "";
    setNotice(`OpenF1 exported: ${manifest.output} (${laps} laps${ageObs})`);
    await refreshOps();
  } catch (error) {
    setNotice(`OpenF1 export failed: ${error.message}`, "error");
  } finally {
    el("openf1ExportButton").disabled = false;
    el("openf1ExportButton").textContent = "Export";
  }
}

async function exportOpenF1FleetIntervals() {
  const payload = {
    year: Number(el("openf1YearInput").value || 2024),
    event: el("openf1EventInput").value || "Bahrain",
    session: el("openf1SessionInput").value || "Race",
  };
  try {
    el("openf1FleetExportButton").disabled = true;
    el("openf1FleetExportButton").textContent = "Exporting…";
    const manifest = await api("/data-sources/openf1/fleet-export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    const drivers = manifest.driver_count || "?";
    const laps = manifest.lap_count || "?";
    setNotice(`Fleet intervals exported: ${manifest.output} (${drivers} drivers × ${laps} laps)`);
    await refreshOps();
  } catch (error) {
    setNotice(`Fleet intervals export failed: ${error.message}`, "error");
  } finally {
    el("openf1FleetExportButton").disabled = false;
    el("openf1FleetExportButton").textContent = "Export Fleet";
  }
}

async function exportFastF1Replay() {
  const payload = {
    year: Number(el("fastf1YearInput").value || 2024),
    event: el("fastf1EventInput").value || "Bahrain",
    session: el("fastf1SessionInput").value || "R",
    driver: el("fastf1DriverInput").value || "VER",
    output: el("fastf1OutputInput").value || null,
    cache_dir: "data/fastf1-cache",
  };
  try {
    el("fastf1ExportButton").disabled = true;
    const manifest = await api("/data-sources/fastf1/export", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.selectedReplayDataset = manifest.output;
    setNotice(`FastF1 replay exported: ${manifest.output}`);
    await refreshOps();
  } catch (error) {
    setNotice(`FastF1 export failed: ${error.message}`, "error");
  } finally {
    el("fastf1ExportButton").disabled = false;
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
  el("replayGateValue").textContent = "-";
  el("featureStoreValue").textContent = "-";
  el("activeArtifactValue").textContent = "-";
  el("promotedArtifactValue").textContent = "-";
  if (el("driftWarmupBar")) el("driftWarmupBar").hidden = true;
  el("readinessChecks").innerHTML = "";
  el("alertList").innerHTML = "";
  el("replaySummary").innerHTML = "";
  el("replayGates").innerHTML = "";
  el("replaySuiteRows").innerHTML = `<tr><td colspan="9">No replay suite</td></tr>`;
  el("replayBenchmarkRows").innerHTML = `<tr><td colspan="9">No benchmark suite</td></tr>`;
  el("replayRunSummary").innerHTML = "";
  el("replayRunGates").innerHTML = "";
  el("regressionSummary").innerHTML = "";
  el("regressionChecks").innerHTML = "";
  el("smokeSummary").innerHTML = "";
  el("smokeChecks").innerHTML = "";
  el("runCheckState").textContent = "Idle";
  el("replayRunState").textContent = "Ready";
  el("regressionRunState").textContent = "Ready";
  el("smokeRunState").textContent = "Ready";
  state.replayRun = null;
  state.regressionRun = null;
  state.smokeRun = null;
  el("dataSourceState").textContent = "No datasets";
  el("replayDatasetRows").innerHTML = `<tr><td colspan="8">No replay datasets</td></tr>`;
  el("datasetDetailGrid").innerHTML = "";
  el("fieldProvenanceList").innerHTML = "";
  el("artifactRows").innerHTML = `<tr><td colspan="8">No registered artifacts</td></tr>`;
  el("artifactDetailGrid").innerHTML = "";
  el("artifactBlockers").innerHTML = "";
  el("artifactSuiteRows").innerHTML = `<tr><td colspan="9">No replay suite</td></tr>`;
  el("artifactReleaseState").textContent = "No artifacts";
  el("comparisonRows").innerHTML = `<tr><td colspan="9">No evaluated models</td></tr>`;
  el("comparisonState").textContent = "No evaluated models";
  el("readinessState").textContent = "Waiting";
  el("alertState").textContent = "0 active";
  el("replayState").textContent = "Waiting";
  state.latestTelemetry = null;
  renderTirePressureMap();
  drawCharts();
  renderShadow();
  renderRunChecks();
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

  const cliffEl = el("cliffLapEstimate");
  if (cliffEl) {
    cliffEl.textContent = strategy.cliff_lap_estimate > 0 ? `Lap ${strategy.cliff_lap_estimate}` : "-";
  }
  const undercutWinEl = el("undercutWindowValue");
  if (undercutWinEl) {
    undercutWinEl.textContent = pit.undercut_window_laps > 0 ? `${pit.undercut_window_laps} lap(s)` : "None";
  }
  const overcutWinEl = el("overcutWindowValue");
  if (overcutWinEl) {
    overcutWinEl.textContent = pit.overcut_window_laps > 0 ? `${pit.overcut_window_laps} lap(s)` : "None";
  }

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

  // PSI and concept drift keys land in feature_scores as psi_<name> and concept_drift_z,
  // which monitoring exports as drift_z_score_psi_<name> / drift_z_score_concept_drift_z.
  const allDrift = Object.fromEntries(
    Object.entries(metrics)
      .filter(([name]) => name.startsWith("f1_drift_z_score_"))
      .map(([name, value]) => [name.replace("f1_drift_z_score_", ""), value])
  );
  const zScores = Object.fromEntries(
    Object.entries(allDrift).filter(([k]) => !k.startsWith("psi_") && k !== "concept_drift_z")
  );
  const psiScores = Object.fromEntries(
    Object.entries(allDrift)
      .filter(([k]) => k.startsWith("psi_"))
      .map(([k, v]) => [k.replace("psi_", ""), v])
  );
  const conceptZ = allDrift["concept_drift_z"] ?? Number.NaN;
  const driftNames = Object.keys(zScores).sort((a, b) => (zScores[b] || 0) - (zScores[a] || 0));
  const rows = el("driftRows");
  rows.innerHTML = "";
  if (!driftNames.length && !Number.isFinite(conceptZ)) {
    rows.innerHTML = `<tr><td colspan="4">No baseline fitted</td></tr>`;
  } else {
    for (const name of driftNames.slice(0, 8)) {
      const z = zScores[name] ?? Number.NaN;
      const psi = psiScores[name] ?? Number.NaN;
      const status = z >= 3 || psi >= 0.2 ? "alert" : psi >= 0.1 ? "warn" : "ok";
      const statusLabel = status === "alert" ? "Alert" : status === "warn" ? "Warn" : "OK";
      rows.innerHTML += `<tr>
        <td>${name.replaceAll("_", " ")}</td>
        <td>${Number.isFinite(z) ? num(z, 2) : "-"}</td>
        <td>${Number.isFinite(psi) ? num(psi, 3) : "-"}</td>
        <td class="drift-${status}">${statusLabel}</td>
      </tr>`;
    }
    if (Number.isFinite(conceptZ)) {
      const status = conceptZ >= 3 ? "alert" : conceptZ >= 1.5 ? "warn" : "ok";
      rows.innerHTML += `<tr>
        <td>concept drift</td>
        <td>${num(conceptZ, 2)}</td>
        <td>-</td>
        <td class="drift-${status}">${status === "alert" ? "Alert" : status === "warn" ? "Warn" : "OK"}</td>
      </tr>`;
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
    el("deploymentReadyValue").textContent = "-";
    el("healthScoreValue").textContent = "-";
    el("alertCountValue").textContent = "-";
    el("rollbackValue").textContent = "-";
    el("featureStoreValue").textContent = "-";
    el("activeArtifactValue").textContent = "-";
    el("promotedArtifactValue").textContent = "-";
    el("readinessChecks").innerHTML = "";
    el("alertList").innerHTML = "";
  } else {
    el("deploymentReadyValue").textContent = readiness.ready ? "Ready" : "Blocked";
    el("deploymentReadyValue").className = readiness.ready ? "risk-low" : "risk-high";
    el("healthScoreValue").textContent = `${num(readiness.health_score, 0)}`;
    el("alertCountValue").textContent = readiness.alert_count;
    const rollback = readiness.rollback_candidate;
    el("rollbackValue").textContent = rollback ? shortRunId(rollback.artifact_id) : "None";
    el("rollbackValue").title = rollback?.artifact_id || "";
    const health = state.health;
    if (health) {
      const fs = health.feature_store_backend || "memory";
      el("featureStoreValue").textContent = fs;
      el("featureStoreValue").className = fs === "redis" ? "risk-low" : "risk-medium";
      renderDriftWarmup(health.drift_baseline_fitted, health.drift_ingest_count || 0);
    }
    renderArtifactAlignment(readiness);
    el("readinessState").textContent =
      `${readiness.mode || "local"} / ${readiness.active_backend} / ${readiness.active_artifact_id}`;
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
  }
  renderReplayDataSources();
  populateTrainingReplayDatasetSelect();
  renderReplayEvaluation();
  renderRunChecks();
  renderOpsCharts();
  renderModelComparison();
  renderFeatureImportance();
  renderArtifactRelease();
  renderShadow();
  renderLiveData();
  renderTrainingJobs();
  renderRunChecks();
}

function renderRunChecks() {
  renderReplayRun();
  renderRegressionRun();
  renderSmokeRun();
}

function renderReplayRun() {
  const payload = state.replayRun;
  const summary = el("replayRunSummary");
  const gates = el("replayRunGates");
  if (!summary || !gates) return;
  if (!payload) {
    el("replayRunState").textContent = "Ready";
    summary.innerHTML = "";
    gates.innerHTML = "";
    return;
  }
  el("replayRunState").textContent = payload.kind || "dataset";
  summary.innerHTML = "";
  gates.innerHTML = "";
  const suite = payload.suite || null;
  const report = payload.report || null;
  const items = report
    ? [
        ["Dataset", shortRunId(report.dataset_path)],
        ["Trust", replayTrust(report.data_provenance || {}).label],
        ["Signal", report.data_provenance?.validation_signal || "-"],
        ["Labels", `${report.labeled_event_count} / ${report.event_count}`],
        ["MAE", `${num(report.scenario?.mae_lap_delta_s, 4)}s`],
        ["Coverage", `${num(report.scenario?.coverage_pct, 1)}%`],
        ["Calibration", `${num(report.scenario?.calibration_error_pct, 1)}%`],
        ["Width", `${num(report.scenario?.mean_interval_width_s, 3)}s`],
      ]
    : suite
      ? [
          ["Suite", suite.suite_name || payload.kind || "-"],
          ["Splits", suite.split_count ?? "-"],
          ["Events", suite.total_event_count ?? "-"],
          ["Labels", suite.total_labeled_event_count ?? "-"],
          ["MAE", `${num(suite.mean_mae_lap_delta_s, 4)}s`],
          ["Coverage", `${num(suite.mean_coverage_pct, 1)}%`],
        ]
      : [];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    summary.appendChild(item);
  }
  const gatesData = report?.gates || suite?.splits?.reduce((acc, split, index) => {
    acc[`split ${index + 1}`] = split.passed;
    return acc;
  }, {}) || {};
  for (const [name, passed] of Object.entries(gatesData)) {
    const item = document.createElement("div");
    item.className = `check-item ${passed ? "pass" : "fail"}`;
    item.innerHTML = `<span>${name}</span><strong>${passed ? "Pass" : "Fail"}</strong>`;
    gates.appendChild(item);
  }
}

function renderRegressionRun() {
  const payload = state.regressionRun;
  const summary = el("regressionSummary");
  const checks = el("regressionChecks");
  if (!summary || !checks) return;
  if (!payload) {
    el("regressionRunState").textContent = "Ready";
    summary.innerHTML = "";
    checks.innerHTML = "";
    return;
  }
  el("regressionRunState").textContent = payload.passed ? "Pass" : "Fail";
  summary.innerHTML = "";
  checks.innerHTML = "";
  const items = [
    ["Laps", payload.config?.laps ?? "-"],
    ["Seed", payload.config?.seed ?? "-"],
    ["Latency", payload.config?.target_latency_ms != null ? `${num(payload.config.target_latency_ms, 1)} ms` : "auto"],
    ["Oscillation", payload.config?.max_temporal_oscillation_s != null ? `${num(payload.config.max_temporal_oscillation_s, 2)} s` : "auto"],
    ["Min Width", `${num(payload.config?.min_calibration_width_s, 2)} s`],
    ["Max Width", payload.config?.max_calibration_width_s != null ? `${num(payload.config.max_calibration_width_s, 2)} s` : "auto"],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    summary.appendChild(item);
  }
  for (const result of payload.results || []) {
    const item = document.createElement("div");
    item.className = `check-item ${result.passed ? "pass" : "fail"}`;
    item.innerHTML = `<span>${result.name.replaceAll("_", " ")}</span><strong>${num(result.value, 4)} / ${num(result.threshold, 4)}</strong>`;
    checks.appendChild(item);
  }
}

function renderSmokeRun() {
  const payload = state.smokeRun;
  const summary = el("smokeSummary");
  const checks = el("smokeChecks");
  if (!summary || !checks) return;
  if (!payload) {
    el("smokeRunState").textContent = "Ready";
    summary.innerHTML = "";
    checks.innerHTML = "";
    return;
  }
  el("smokeRunState").textContent = payload.passed ? "Pass" : "Fail";
  summary.innerHTML = "";
  checks.innerHTML = "";
  const items = [
    ["Model", payload.model_backend || "-"],
    ["Artifact", payload.artifact_id || "-"],
    ["Replay", payload.replay?.scenario?.source || "-"],
    ["Benchmark", payload.benchmark?.suite_name || "-"],
    ["Regression", payload.regression?.passed ? "Pass" : "Fail"],
    ["Links", payload.external_links?.filter((item) => item.external).length ?? 0],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    summary.appendChild(item);
  }
  for (const check of payload.checks || []) {
    const item = document.createElement("div");
    item.className = `check-item ${check.passed ? "pass" : "fail"}`;
    item.innerHTML = `<span>${check.name.replaceAll("_", " ")}</span><strong>${check.passed ? "Pass" : "Fail"}</strong>`;
    checks.appendChild(item);
  }
}

function renderArtifactAlignment(readiness) {
  const active = readiness?.active_artifact_id || state.artifactRegistry.active_artifact_id || "unregistered";
  const promoted = promotedArtifactForBackend(readiness?.active_backend);
  const aligned = active !== "unregistered" && promoted && active === promoted;
  el("activeArtifactValue").textContent = shortRunId(active);
  el("activeArtifactValue").title = active;
  el("activeArtifactValue").className = aligned ? "risk-low" : "risk-medium";
  el("promotedArtifactValue").textContent = promoted ? shortRunId(promoted) : "None";
  el("promotedArtifactValue").title = promoted || "";
  el("promotedArtifactValue").className = promoted ? "risk-low" : "risk-medium";
}

function promotedArtifactForBackend(activeBackend) {
  const promoted = state.artifactRegistry.promoted || {};
  const normalized = String(activeBackend || "").replace("-torch", "");
  return promoted[normalized] || promoted.sequence || Object.values(promoted)[0] || "";
}

function renderReplayDataSources() {
  const rows = el("replayDatasetRows");
  const select = el("replayDatasetSelect");
  const datasets = state.replayDatasets || [];
  rows.innerHTML = "";
  select.innerHTML = "";
  el("dataSourceState").textContent = datasets.length
    ? `${datasets.length} datasets`
    : "No datasets";
  populateLiveDatasetSelect();
  populateTrainingRealDataSelect();
  if (!datasets.length) {
    rows.innerHTML = `<tr><td colspan="9">No replay datasets</td></tr>`;
    renderDatasetDetail(null);
    return;
  }
  for (const dataset of datasets) {
    const provenance = dataset.data_provenance || {};
    const productionReady = provenance.production_validation_ready === true;
    const hasFleet = dataset.has_fleet_intervals === true;
    const option = document.createElement("option");
    option.value = dataset.path;
    option.textContent = dataset.path;
    select.appendChild(option);

    const row = document.createElement("tr");
    row.className = dataset.path === state.selectedReplayDataset ? "selected" : "";
    row.tabIndex = 0;
    row.title = dataset.path;
    row.innerHTML = `
      <td>${shortRunId(dataset.path)}</td>
      <td>${dataset.source || "replay"}</td>
      <td>${dataset.event_count ?? "-"}</td>
      <td>${dataset.labeled_event_count ?? "-"}</td>
      <td>${trustBadge(provenance)}</td>
      <td>${provenance.lap_time_label || "-"}</td>
      <td class="${productionReady ? "risk-low" : "risk-medium"}">${productionReady ? "Yes" : "No"}</td>
      <td class="${dataset.has_manifest ? "risk-low" : "risk-medium"}">${dataset.has_manifest ? "Yes" : "No"}</td>
      <td class="${hasFleet ? "risk-low" : "risk-medium"}" title="${hasFleet ? dataset.fleet_intervals_path : "Run Export Fleet to generate"}">${hasFleet ? "Yes" : "No"}</td>
    `;
    row.addEventListener("click", () => selectReplayDataset(dataset.path));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectReplayDataset(dataset.path);
      }
    });
    rows.appendChild(row);
  }
  select.value = state.selectedReplayDataset;
  renderDatasetDetail(datasets.find((item) => item.path === state.selectedReplayDataset));
}

function renderDatasetDetail(dataset) {
  const grid = el("datasetDetailGrid");
  grid.innerHTML = "";
  if (!dataset) {
    grid.innerHTML = `<div class="detail-item"><span>Dataset</span><strong>No dataset selected</strong></div>`;
    renderFieldProvenance({});
    return;
  }
  const manifest = state.datasetManifest || {};
  const provenance = dataset.data_provenance || {};
  const items = [
    ["Dataset", shortRunId(dataset.path)],
    ["Source", dataset.source || "replay"],
    ["Trust", replayTrust(provenance).label],
    ["Signal", provenance.validation_signal || dataset.validation_signal || "-"],
    ["Label Type", provenance.lap_time_label || "-"],
    ["Prod Ready", provenance.production_validation_ready ? "Yes" : "No"],
    ["Reference Lap", provenance.reference_lap_time_s ? `${num(provenance.reference_lap_time_s, 3)}s` : "-"],
    ["Observed Fields", provenance.observed_field_count ?? "-"],
    ["Proxy Fields", provenance.proxy_diagnostic_field_count ?? "-"],
    ["Rows", dataset.event_count ?? "-"],
    ["Labels", dataset.labeled_event_count ?? "-"],
    ["Laps", dataset.lap_count ?? manifest.lap_count ?? "-"],
    ["Manifest", dataset.has_manifest ? "Available" : "Missing"],
    ["Fleet Intervals", dataset.has_fleet_intervals ? "Available" : "Missing — use Export Fleet"],
    ["Fingerprint", shortRunId(dataset.dataset_fingerprint || manifest.dataset_fingerprint || "-")],
    ["Generated", manifest.generated_at ? shortRunId(manifest.generated_at) : "-"],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    grid.appendChild(item);
  }
  renderFieldProvenance(manifest.field_provenance || dataset.field_provenance || {});
  renderProvenanceLimitations(provenance.limitations || manifest.limitations || []);
}

function renderFieldProvenance(provenance) {
  const list = el("fieldProvenanceList");
  list.innerHTML = "";
  const entries = Object.entries(provenance);
  if (!entries.length) {
    list.innerHTML = `<div class="check-item"><span>Field provenance</span><strong>Unavailable</strong></div>`;
    return;
  }
  const groups = { observed: 0, synthetic: 0, derived: 0, fallback: 0, unavailable: 0, other: 0 };
  for (const [, description] of entries) {
    const kind = String(description).split(":")[0];
    if (kind in groups) groups[kind] += 1;
    else groups.other += 1;
  }
  for (const [kind, count] of Object.entries(groups).filter(([, count]) => count > 0)) {
    const item = document.createElement("div");
    item.className = `check-item ${kind === "observed" ? "pass" : kind === "synthetic" ? "warn" : "fail"}`;
    item.innerHTML = `<span>${kind} fields</span><strong>${count}</strong>`;
    list.appendChild(item);
  }
}

function renderProvenanceLimitations(limitations) {
  const list = el("fieldProvenanceList");
  for (const limitation of limitations.slice(0, 3)) {
    const item = document.createElement("div");
    item.className = "check-item fail";
    item.innerHTML = `<span title="${limitation}">${limitation}</span><strong>Limit</strong>`;
    list.appendChild(item);
  }
}

function renderReplayEvaluation() {
  const report = state.replayEvaluation;
  const benchmark = state.replayBenchmark;
  if (!report) {
    el("replayGateValue").textContent = "-";
    el("replayState").textContent = "Waiting";
    return;
  }
  const scenario = report.scenario || {};
  const provenance = report.data_provenance || {};
  const benchmarkPassed = benchmark?.passed === true;
  el("replayGateValue").textContent = benchmark ? (benchmarkPassed ? "Pass" : "Fail") : "-";
  el("replayGateValue").className = benchmarkPassed ? "risk-low" : "risk-high";
  el("replayState").textContent =
    benchmark
      ? `${benchmark.suite_name || "benchmark"} / ${benchmark.split_count || 0} splits`
      : `${shortRunId(report.dataset_path)} / ${String(report.dataset_fingerprint || "").slice(0, 8)}`;
  el("replayState").title = benchmark
    ? `${benchmark.total_labeled_event_count || 0} labeled events`
    : report.dataset_fingerprint || "";

  const summary = el("replaySummary");
  summary.innerHTML = "";
  const items = [
    ["Source", scenario.source || "replay"],
    ["Trust", replayTrust(provenance).label],
    ["Signal", provenance.validation_signal || "-"],
    ["Label Type", provenance.lap_time_label || "-"],
    ["Prod Ready", provenance.production_validation_ready ? "Yes" : "No"],
    ["Reference Lap", provenance.reference_lap_time_s ? `${num(provenance.reference_lap_time_s, 3)}s` : "-"],
    ["Observed Fields", provenance.observed_field_count ?? "-"],
    ["Proxy Fields", provenance.proxy_diagnostic_field_count ?? "-"],
    ["Events", report.event_count],
    ["Labels", `${report.labeled_event_count} / ${report.event_count}`],
    ["Sessions", report.session_count],
    ["MAE", `${num(scenario.mae_lap_delta_s, 4)}s`],
    ["RMSE", `${num(scenario.rmse_lap_delta_s, 4)}s`],
    ["Coverage", `${num(scenario.coverage_pct, 1)}%`],
    ["Calibration", `${num(scenario.calibration_error_pct, 1)}%`],
    ["Sharpness", `${num(scenario.mean_interval_width_s, 3)}s`],
    ["Pit Error", `${num(scenario.pit_target_error_laps, 2)} laps`],
    ["Regret", `${num(scenario.strategy_regret_s, 3)}s`],
    ["p95 Latency", `${num(scenario.latency_p95_ms, 3)} ms`],
    ["Wear Violations", scenario.monotonic_wear_violations],
    ["Missing Targets", `${num(report.missing_target_pct, 1)}%`],
    ["Schema", String(report.feature_schema_hash || "").slice(0, 8)],
    ["Dataset", shortRunId(report.dataset_path)],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    summary.appendChild(item);
  }

  const gates = el("replayGates");
  gates.innerHTML = "";
  for (const [name, passed] of Object.entries(report.gates || {})) {
    const item = document.createElement("div");
    item.className = `check-item ${passed ? "pass" : "fail"}`;
    item.innerHTML = `<span>${name.replaceAll("_", " ")}</span><strong>${passed ? "Pass" : "Fail"}</strong>`;
    gates.appendChild(item);
  }
  renderReplaySuite(state.replaySuite, "replaySuiteRows");
  renderReplaySuite(state.replayBenchmark, "replayBenchmarkRows");
}

function renderReplaySuite(suite, targetId) {
  const rows = el(targetId);
  rows.innerHTML = "";
  const splits = suite?.splits || [];
  if (!splits.length) {
    rows.innerHTML = `<tr><td colspan="9">No replay suite</td></tr>`;
    return;
  }
  for (const split of splits) {
    const scenario = split.scenario || {};
    const provenance = split.data_provenance || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${scenario.scenario || "-"}</td>
      <td class="${split.passed ? "risk-low" : "risk-high"}">${split.passed ? "Pass" : "Fail"}</td>
      <td>${trustBadge(provenance)}</td>
      <td>${num(scenario.mae_lap_delta_s, 4)}s</td>
      <td>${num(scenario.coverage_pct, 1)}%</td>
      <td>${num(scenario.calibration_error_pct, 1)}%</td>
      <td>${num(scenario.mean_interval_width_s, 3)}s</td>
      <td>${num(scenario.pit_target_error_laps, 2)}</td>
      <td>${num(scenario.strategy_regret_s, 3)}s</td>
    `;
    rows.appendChild(row);
  }
}

function activeComparisonBackend() {
  return state.readiness?.active_backend || state.health?.model_backend || "";
}

function comparisonVerdict(row, bestRow) {
  if (!row) return { label: "-", className: "performance-muted" };
  const bestMae = bestRow?.mae_lap_delta_s ?? row.mae_lap_delta_s;
  const maeGap = row.mae_lap_delta_s - bestMae;
  if (row.rank === 1 || maeGap <= 0.005) {
    return { label: "Leading", className: "performance-good" };
  }
  if (maeGap <= 0.03 || row.health_score >= 80) {
    return { label: "Competitive", className: "performance-warn" };
  }
  return { label: "Lagging", className: "performance-bad" };
}

function comparisonRecommendation(row, bestRow) {
  const recommendation = row?.recommendation;
  if (recommendation?.action) {
    return {
      label: recommendation.action,
      className: `performance-action-${recommendation.action}`,
      title: recommendation.reason || recommendation.action,
    };
  }
  if (!row) {
    return { label: "-", className: "performance-muted", title: "-" };
  }
  const bestMae = bestRow?.mae_lap_delta_s ?? row.mae_lap_delta_s;
  const maeGap = row.mae_lap_delta_s - bestMae;
  if (row.rank === 1 && row.health_score >= 80 && row.interval_coverage_pct >= 90 && row.alert_count === 0) {
    return { label: "promote", className: "performance-action-promote", title: "Best observed performer with strong coverage and no alerts" };
  }
  if (row.critical_alert_count > 0 || row.health_score < 55 || row.interval_coverage_pct < 75 || maeGap > 0.08) {
    return { label: "retire", className: "performance-action-retire", title: "High error, weak coverage, or critical alerts" };
  }
  return { label: "hold", className: "performance-action-hold", title: "Needs more evidence before promotion" };
}

function renderComparisonSummary(rows) {
  const summary = el("comparisonSummary");
  const active = activeComparisonBackend();
  const activeRow = rows.find((row) => row.backend === active) || rows[0] || null;
  const bestRow = rows[0] || null;
  if (!summary) return;
  if (!rows.length || !activeRow) {
    el("comparisonModelValue").textContent = active || "-";
    el("comparisonRankValue").textContent = "-";
    el("comparisonGapValue").textContent = "-";
    el("comparisonVerdictValue").textContent = "-";
    el("comparisonActionValue").textContent = "-";
    summary.querySelectorAll(".performance-chip").forEach((chip) => {
      chip.className = "performance-chip";
    });
    return;
  }

  const verdict = comparisonVerdict(activeRow, bestRow);
  const action = comparisonRecommendation(activeRow, bestRow);
  const bestMae = bestRow?.mae_lap_delta_s ?? activeRow.mae_lap_delta_s;
  const maeGap = activeRow.mae_lap_delta_s - bestMae;
  const coverageGap = (bestRow?.interval_coverage_pct ?? activeRow.interval_coverage_pct) - activeRow.interval_coverage_pct;
  el("comparisonModelValue").textContent = activeRow.backend || "-";
  el("comparisonRankValue").textContent = `#${activeRow.rank ?? "-"}`;
  el("comparisonGapValue").textContent = `${maeGap > 0 ? "+" : ""}${num(maeGap, 4)}s / ${num(coverageGap, 1)}% cov`;
  el("comparisonVerdictValue").textContent = verdict.label;
  el("comparisonActionValue").textContent = action.label;
  el("comparisonActionValue").title = action.title;

  const chips = summary.querySelectorAll(".performance-chip");
  if (chips[0]) chips[0].className = `performance-chip ${activeRow.rank === 1 ? "performance-good" : "performance-muted"}`;
  if (chips[1]) chips[1].className = `performance-chip ${activeRow.rank === 1 ? "performance-good" : activeRow.rank <= 3 ? "performance-warn" : "performance-bad"}`;
  if (chips[2]) chips[2].className = `performance-chip ${maeGap <= 0.005 ? "performance-good" : maeGap <= 0.03 ? "performance-warn" : "performance-bad"}`;
  if (chips[3]) chips[3].className = `performance-chip ${verdict.className}`;
  if (chips[4]) chips[4].className = `performance-chip ${action.className}`;
}

function renderModelComparison() {
  const rows = state.comparison || [];
  const table = el("comparisonRows");
  table.innerHTML = "";
  const active = activeComparisonBackend();
  el("comparisonState").textContent = rows.length
    ? `${rows.length} evaluated${active ? ` · active ${active}` : ""}`
    : "No evaluated models";
  if (!rows.length) {
    table.innerHTML = `<tr><td colspan="10">Run at least one full lap per model</td></tr>`;
    renderComparisonSummary([]);
    drawComparisonChart([]);
    return;
  }
  for (const row of rows) {
    const item = document.createElement("tr");
    item.className = row.backend === active ? "selected" : "";
    item.innerHTML = `
      <td>${row.rank}</td>
      <td>${row.backend}</td>
      <td title="${row.artifact_id}">${shortRunId(row.artifact_id)}</td>
      <td>${row.evaluations}</td>
      <td>${num(row.mae_lap_delta_s, 4)}s</td>
      <td>${num(row.rmse_lap_delta_s, 4)}s</td>
      <td>${num(row.interval_coverage_pct, 1)}%</td>
      <td><span class="decision-pill decision-${row.recommendation?.action || "hold"}" title="${row.recommendation?.reason || ""}">${row.recommendation?.action || "hold"}</span></td>
      <td>${num(row.health_score, 0)}</td>
      <td>${row.alert_count}</td>
    `;
    table.appendChild(item);
  }
  renderComparisonSummary(rows);
  drawComparisonChart(rows);
}

function renderFeatureImportance() {
  const container = el("featureImportanceBars");
  const stateEl = el("featureImportanceState");
  const data = state.featureImportance;
  if (!data || !data.available) {
    stateEl.textContent = "Unavailable";
    container.innerHTML = `<span class="muted">Feature importance not yet available — train a model first.</span>`;
    return;
  }
  const items = (data.importance || []).slice(0, 20);
  stateEl.textContent = `${data.backend} · top ${items.length} features`;
  container.innerHTML = "";
  const maxPct = items[0]?.pct || 1;
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "fi-row";
    const barWidth = Math.max(2, (item.pct / maxPct) * 100).toFixed(1);
    row.innerHTML = `
      <span class="fi-rank">#${item.rank}</span>
      <span class="fi-name" title="${item.name}">${item.name}</span>
      <div class="fi-bar-wrap"><div class="fi-bar" style="width:${barWidth}%"></div></div>
      <span class="fi-pct">${num(item.pct, 1)}%</span>
    `;
    container.appendChild(row);
  }
}

function renderArtifactRelease() {
  const rows = el("artifactRows");
  const artifacts = state.artifacts || [];
  rows.innerHTML = "";
  el("artifactReleaseState").textContent = artifacts.length
    ? `${artifacts.length} registered`
    : "No artifacts";
  if (!artifacts.length) {
    rows.innerHTML = `<tr><td colspan="8">No registered artifacts</td></tr>`;
    renderArtifactDetail();
    return;
  }
  if (!state.selectedArtifactId || !artifacts.some((item) => item.artifact_id === state.selectedArtifactId)) {
    state.selectedArtifactId = artifacts[0].artifact_id;
  }
  for (const artifact of artifacts) {
    const replayPassed = artifact.replay_passed;
    const replayState = replayPassed === true ? "Pass" : replayPassed === false ? "Fail" : "Missing";
    const replayStateClass = replayPassed === true ? "risk-low" : replayPassed === false ? "risk-high" : "risk-medium";
    const productionReady = artifact.production_validation_ready === true;
    const item = document.createElement("tr");
    item.className = artifact.artifact_id === state.selectedArtifactId ? "selected" : "";
    item.tabIndex = 0;
    item.title = artifact.artifact_id;
    item.innerHTML = `
      <td>${shortRunId(artifact.artifact_id)}</td>
      <td>${artifact.status || "-"}</td>
      <td class="${replayStateClass}">${replayState}</td>
      <td>${artifact.replay_validation_signal || "-"}</td>
      <td class="${productionReady ? "risk-low" : "risk-medium"}">${productionReady ? "Yes" : "No"}</td>
      <td>${num(artifact.replay_mae_lap_delta_s, 4)}s</td>
      <td>${num(artifact.replay_coverage_pct, 1)}%</td>
      <td title="${artifact.replay_dataset_fingerprint || ""}">${shortRunId(artifact.replay_dataset_fingerprint || "-")}</td>
    `;
    item.addEventListener("click", () => selectArtifact(artifact.artifact_id));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectArtifact(artifact.artifact_id);
      }
    });
    rows.appendChild(item);
  }
  renderArtifactDetail();
}

async function selectArtifact(artifactId) {
  state.selectedArtifactId = artifactId;
  renderArtifactRelease();
  await refreshArtifactDetail();
}

function renderArtifactDetail() {
  const detail = state.artifactDetail;
  const grid = el("artifactDetailGrid");
  const blockers = el("artifactBlockers");
  grid.innerHTML = "";
  blockers.innerHTML = "";
  if (!state.selectedArtifactId) {
    grid.innerHTML = `<div class="detail-item"><span>Status</span><strong>No artifact selected</strong></div>`;
    return;
  }
  if (!detail) {
    grid.innerHTML = `<div class="detail-item"><span>Artifact</span><strong>${shortRunId(state.selectedArtifactId)}</strong></div>`;
    blockers.innerHTML = `<div class="check-item fail"><span>Detail</span><strong>Unavailable</strong></div>`;
    return;
  }
  const manifest = detail.manifest || {};
  const evalMetrics = manifest.evaluation_metrics || {};
  const replayMetrics = manifest.replay_evaluation_metrics || {};
  const replayProvenance = manifest.replay_data_provenance || {};
  const items = [
    ["Artifact", shortRunId(detail.artifact_id)],
    ["Backend", manifest.backend || "-"],
    ["Status", manifest.status || "-"],
    ["Promotion", detail.promotion_ready ? "Ready" : "Blocked"],
    ["Git SHA", manifest.git_sha || "-"],
    ["Created", manifest.created_at || "-"],
    ["Schema", String(manifest.feature_schema_hash || "").slice(0, 8)],
    ["Simulator MAE", `${num(evalMetrics.mean_mae_lap_delta_s, 4)}s`],
    ["Simulator Coverage", `${num(evalMetrics.mean_coverage_pct, 1)}%`],
    ["Max Cal Error", `${num(evalMetrics.max_calibration_error_pct, 1)}%`],
    ["Max Width", `${num(evalMetrics.max_mean_interval_width_s, 3)}s`],
    ["Max Pit Error", `${num(evalMetrics.max_pit_target_error_laps, 2)} laps`],
    ["Max Regret", `${num(evalMetrics.max_strategy_regret_s, 3)}s`],
    ["Replay MAE", `${num(replayMetrics.mae_lap_delta_s, 4)}s`],
    ["Replay Trust", replayTrust(replayProvenance).label],
    ["Replay Signal", replayMetrics.validation_signal || replayProvenance.validation_signal || "-"],
    ["Replay Label", replayProvenance.lap_time_label || "-"],
    ["Prod Validation", replayMetrics.production_validation_ready ? "Yes" : "No"],
    ["Reference Lap", replayProvenance.reference_lap_time_s ? `${num(replayProvenance.reference_lap_time_s, 3)}s` : "-"],
    ["Replay Observed", replayProvenance.observed_field_count ?? "-"],
    ["Replay Proxy", replayProvenance.proxy_diagnostic_field_count ?? "-"],
    ["Replay Coverage", `${num(replayMetrics.coverage_pct, 1)}%`],
    ["Replay Cal Error", `${num(replayMetrics.calibration_error_pct, 1)}%`],
    ["Replay Width", `${num(replayMetrics.mean_interval_width_s, 3)}s`],
    ["Replay Pit Error", `${num(replayMetrics.pit_target_error_laps, 2)} laps`],
    ["Replay Regret", `${num(replayMetrics.strategy_regret_s, 3)}s`],
    ["Suite", manifest.replay_suite_metrics?.passed ? "Pass" : "Blocked"],
    ["Suite Splits", manifest.replay_suite_metrics?.split_count || "-"],
    ["Replay Dataset", shortRunId(manifest.replay_dataset_fingerprint || "-")],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    item.className = "detail-item";
    item.innerHTML = `<span>${label}</span><strong title="${value}">${value}</strong>`;
    grid.appendChild(item);
  }
  const failures = detail.promotion_failures || [];
  if (!failures.length) {
    blockers.innerHTML = `<div class="check-item pass"><span>Promotion blockers</span><strong>Clear</strong></div>`;
  } else {
    for (const failure of failures.slice(0, 8)) {
      const item = document.createElement("div");
      item.className = "check-item fail";
      item.innerHTML = `<span title="${failure}">${failure}</span><strong>Block</strong>`;
      blockers.appendChild(item);
    }
  }
  renderArtifactSuite(detail.replay_suite);
}

function renderArtifactSuite(suite) {
  const rows = el("artifactSuiteRows");
  rows.innerHTML = "";
  const splits = suite?.splits || [];
  if (!splits.length) {
    rows.innerHTML = `<tr><td colspan="8">No replay suite</td></tr>`;
    return;
  }
  for (const split of splits) {
    const scenario = split.scenario || {};
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${scenario.scenario || "-"}</td>
      <td class="${split.passed ? "risk-low" : "risk-high"}">${split.passed ? "Pass" : "Fail"}</td>
      <td>${num(scenario.mae_lap_delta_s, 4)}s</td>
      <td>${num(scenario.coverage_pct, 1)}%</td>
      <td>${num(scenario.calibration_error_pct, 1)}%</td>
      <td>${num(scenario.mean_interval_width_s, 3)}s</td>
      <td>${num(scenario.pit_target_error_laps, 2)}</td>
      <td>${num(scenario.strategy_regret_s, 3)}s</td>
      <td title="${split.dataset_fingerprint || ""}">${shortRunId(split.dataset_path || "-")}</td>
    `;
    rows.appendChild(row);
  }
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

async function enableShadow() {
  const backend = el("shadowBackendSelect").value;
  try {
    el("shadowEnableButton").disabled = true;
    state.shadow = await api(
      `/shadow/configure?backend=${encodeURIComponent(backend)}`,
      { method: "POST" }
    );
    el("shadowEnableButton").disabled = false;
    el("shadowDisableButton").disabled = false;
    setNotice(`Shadow deployment enabled: challenger=${backend}`);
    renderShadow();
  } catch (error) {
    el("shadowEnableButton").disabled = false;
    setNotice(`Could not enable shadow: ${error.message}`, "error");
  }
}

async function disableShadow() {
  try {
    await api("/shadow", { method: "DELETE" });
    state.shadow = { active: false, challenger_backend: "none", total_predictions: 0, recent: [] };
    el("shadowDisableButton").disabled = true;
    el("shadowEnableButton").disabled = false;
    setNotice("Shadow deployment disabled.");
    renderShadow();
  } catch (error) {
    setNotice(`Could not disable shadow: ${error.message}`, "error");
  }
}

function renderShadow() {
  const shadow = state.shadow;
  const active = shadow?.active === true;
  el("shadowState").textContent = active
    ? `Active · ${shadow.challenger_backend}`
    : "Inactive";
  el("shadowState").className = active ? "risk-low" : "";
  el("shadowBackendValue").textContent = active ? (shadow.challenger_backend || "none") : "None";
  el("shadowTotalValue").textContent = shadow?.total_predictions ?? 0;
  el("shadowDivergenceValue").textContent = shadow
    ? pct(shadow.divergence_rate ?? 0)
    : "-";
  el("shadowDivergenceValue").className = (shadow?.divergence_rate ?? 0) > 0.3
    ? "risk-high"
    : (shadow?.divergence_rate ?? 0) > 0.1
      ? "risk-medium"
      : "risk-low";
  el("shadowMeanDeltaValue").textContent = shadow
    ? `${num(shadow.mean_abs_delta_s ?? 0, 3)}s`
    : "-";
  el("shadowEnableButton").disabled = active;
  el("shadowDisableButton").disabled = !active;

  const candidate = shadow?.promotion_candidate;
  const banner = el("shadowPromotionBanner");
  if (candidate) {
    banner.hidden = false;
    el("shadowPromotionDetail").textContent =
      `${candidate.challenger_backend} is ${num(candidate.improvement_s * 1000, 0)}ms better · ` +
      `${num(candidate.improvement_pct, 1)}% improvement over ${candidate.window_size} predictions`;
  } else {
    banner.hidden = true;
  }

  const rows = el("shadowRecentRows");
  rows.innerHTML = "";
  const recent = (shadow?.recent || []).slice().reverse();
  if (!recent.length) {
    rows.innerHTML = `<tr><td colspan="6">${active ? "Waiting for predictions…" : "Shadow deployment inactive"}</td></tr>`;
    drawShadowChart([]);
    return;
  }
  for (const item of recent.slice(0, 20)) {
    const absDelta = Math.abs(item.delta_s ?? 0);
    const cls = absDelta > 0.15 ? "strongly-diverged" : absDelta > 0.05 ? "diverged" : "";
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${item.lap ?? "-"}</td>
      <td>${num(item.champion_delta_s, 3)}s</td>
      <td>${num(item.challenger_delta_s, 3)}s</td>
      <td class="${cls}">${num(item.delta_s, 3)}s</td>
      <td>${num(item.champion_wear, 1)}%</td>
      <td>${num(item.challenger_wear, 1)}%</td>
    `;
    rows.appendChild(row);
  }
  drawShadowChart(shadow.recent || []);
}

function drawShadowChart(data) {
  const canvas = el("shadowChart");
  if (!canvas) return;
  const rows = data.slice(-80);
  if (!rows.length) {
    el("shadowChartBand").textContent = "Waiting";
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#101820";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#314452";
    ctx.font = "20px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for shadow data", canvas.width / 2, canvas.height / 2);
    return;
  }
  el("shadowChartBand").textContent = `${rows.length} observations`;
  const series = [
    { key: "champion_delta_s", label: "Champion Δ", color: "#62a8ff" },
    { key: "challenger_delta_s", label: "Challenger Δ", color: "#f5c45c" },
  ];
  drawLineChart(canvas, rows, series, { xLabel: "Predictions", yUnit: "s" });
}

el("startButton").addEventListener("click", startSimulation);
el("pauseButton").addEventListener("click", stopSimulation);
el("tickButton").addEventListener("click", tickSimulation);
el("resetButton").addEventListener("click", resetSimulation);
el("modelSelect").addEventListener("change", changeModelBackend);
el("artifactSelect").addEventListener("change", changeModelArtifact);
el("fastf1ExportButton").addEventListener("click", exportFastF1Replay);
el("openf1ExportButton").addEventListener("click", exportOpenF1Session);
el("fastf1TabBtn").addEventListener("click", () => setExportSourceTab("fastf1"));
el("openf1TabBtn").addEventListener("click", () => setExportSourceTab("openf1"));
el("validateDatasetButton").addEventListener("click", refreshOps);
el("replayDatasetSelect").addEventListener("change", (event) => selectReplayDataset(event.target.value));
el("historyRefreshButton").addEventListener("click", refreshHistory);
el("historySortSelect").addEventListener("change", (event) => setHistorySort(event.target.value));
el("historySortDirectionButton").addEventListener("click", toggleHistorySortDirection);
// ── Live data ──────────────────────────────────────────────────────────────

function setLiveMode(mode) {
  state.liveMode = mode;
  el("replayModeBtn").classList.toggle("active", mode === "replay");
  el("liveModeBtn").classList.toggle("active", mode === "live");
  el("replayControls").hidden = mode !== "replay";
  el("liveTimingControls").hidden = mode !== "live";
}

function populateLiveDatasetSelect() {
  const select = el("liveReplayDatasetSelect");
  const current = select.value;
  select.innerHTML = `<option value="">Select dataset…</option>`;
  for (const dataset of state.replayDatasets || []) {
    const opt = document.createElement("option");
    opt.value = dataset.path;
    opt.textContent = dataset.path;
    select.appendChild(opt);
  }
  if (current) select.value = current;
}

async function startLiveReplay() {
  const dataset = el("liveReplayDatasetSelect").value;
  if (!dataset) {
    setNotice("Select a replay dataset first.", "error");
    return;
  }
  const speed = Number(el("liveSpeedInput").value || 5);
  try {
    el("liveReplayStartBtn").disabled = true;
    state.liveStatus = await api("/live-data/replay/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ dataset_path: dataset, speed_multiplier: speed }),
    });
    el("liveStopBtn").disabled = false;
    state.livePredictions = [];
    setNotice(`Replay started: ${dataset} at ${speed}× speed`);
    renderLiveData();
    scheduleLivePoll();
  } catch (error) {
    el("liveReplayStartBtn").disabled = false;
    setNotice(`Could not start replay: ${error.message}`, "error");
  }
}

async function startLiveTiming() {
  const driver = el("liveDriverInput").value.trim() || "VER";
  const sessionId = el("liveSessionIdInput").value.trim() || "live-session";
  const noAuth = el("liveNoAuthInput").checked;
  try {
    el("liveTimingStartBtn").disabled = true;
    state.liveStatus = await api("/live-data/live/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ driver, session_id: sessionId, no_auth: noAuth }),
    });
    el("liveTimingStopBtn").disabled = false;
    state.livePredictions = [];
    setNotice(`Live timing connecting: driver=${driver} session=${sessionId}`);
    renderLiveData();
    scheduleLivePoll();
  } catch (error) {
    el("liveTimingStartBtn").disabled = false;
    setNotice(`Could not start live timing: ${error.message}`, "error");
  }
}

async function stopLiveData() {
  clearTimeout(state.liveTimer);
  try {
    await api("/live-data", { method: "DELETE" });
    state.liveStatus = null;
    el("liveReplayStartBtn").disabled = false;
    el("liveTimingStartBtn").disabled = false;
    el("liveStopBtn").disabled = true;
    el("liveTimingStopBtn").disabled = true;
    setNotice("Live data stopped.");
    renderLiveData();
  } catch (error) {
    setNotice(`Could not stop live data: ${error.message}`, "error");
  }
}

function scheduleLivePoll() {
  clearTimeout(state.liveTimer);
  const interval = state.liveStatus?.connected ? 200 : 900;
  state.liveTimer = setTimeout(async () => {
    try {
      state.liveStatus = await api("/live-data/status");
      const pred = state.liveStatus?.latest_prediction;
      if (pred) {
        state.livePredictions.push(pred);
        state.livePredictions = state.livePredictions.slice(-80);
      }
      renderLiveData();
      if (state.liveStatus?.connected) scheduleLivePoll();
      else {
        el("liveReplayStartBtn").disabled = false;
        el("liveTimingStartBtn").disabled = false;
        el("liveStopBtn").disabled = true;
        el("liveTimingStopBtn").disabled = true;
      }
    } catch (_err) {
      scheduleLivePoll();
    }
  }, interval);
}

function renderLiveData() {
  const s = state.liveStatus;
  const badge = el("liveDataState");

  if (!s || s.mode === "idle") {
    badge.textContent = "Idle";
    badge.className = "live-badge";
    el("liveModeBadge").textContent = "—";
    el("liveEventCount").textContent = "0";
    el("liveEventRate").textContent = "—";
    el("liveLatestLap").textContent = "—";
    el("liveLapTime").textContent = "—";
    el("liveCompound").textContent = "—";
    el("liveProgress").textContent = "—";
    el("liveMessages").textContent = "—";
    el("liveError").hidden = true;
    el("liveFeedBand").textContent = "Waiting";
    el("liveFeedRows").innerHTML = `<tr><td colspan="7">No live data yet</td></tr>`;
    drawLiveFeedChart([]);
    return;
  }

  const isReplay = s.mode === "replay";
  badge.textContent = s.connected
    ? (isReplay ? `Streaming ${s.speed_multiplier}×` : "Live")
    : (s.error ? "Error" : "Done");
  badge.className = `live-badge ${s.connected ? (isReplay ? "streaming" : "live") : ""}`;

  el("liveModeBadge").textContent = s.mode;
  el("liveEventCount").textContent = s.events_ingested;
  el("liveEventRate").textContent = `${num(s.events_per_second, 1)}/s`;
  el("liveLatestLap").textContent = s.latest_lap || "—";
  el("liveLapTime").textContent = s.latest_lap_time_s != null
    ? `${num(s.latest_lap_time_s, 3)}s`
    : "—";
  el("liveCompound").textContent = s.current_compound || "—";
  el("liveProgress").textContent = isReplay
    ? `${num(s.progress_pct, 0)}%`
    : (s.message_count ? `${s.message_count} msg` : "—");
  el("liveMessages").textContent = s.message_count || "—";
  el("liveFeedBand").textContent = s.session_id
    ? `${s.session_id} · ${s.driver}`
    : "Waiting";

  if (s.error) {
    el("liveError").hidden = false;
    el("liveError").textContent = s.error;
  } else {
    el("liveError").hidden = true;
  }

  renderLiveFeedRows();
  drawLiveFeedChart(state.livePredictions);
}

function renderLiveFeedRows() {
  const rows = el("liveFeedRows");
  const predictions = state.livePredictions.slice().reverse().slice(0, 24);
  rows.innerHTML = "";
  if (!predictions.length) {
    rows.innerHTML = `<tr><td colspan="7">No predictions yet</td></tr>`;
    return;
  }
  for (const p of predictions) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${p.lap}</td>
      <td>—</td>
      <td>${num(p.tire_wear_pct, 1)}%</td>
      <td class="${riskClass(p.cliff_probability)}">${pct(p.cliff_probability)}</td>
      <td>${num(p.next_lap_delta_s, 3)}s</td>
      <td>${num(p.ers_efficiency, 2)}</td>
      <td>${p.model_backend || "—"}</td>
    `;
    rows.appendChild(row);
  }
}

function drawLiveFeedChart(predictions) {
  const canvas = el("liveFeedChart");
  if (!canvas) return;
  const rows = predictions.slice(-60).map((p) => ({
    wear: p.tire_wear_pct,
    cliff: p.cliff_probability * 100,
    delta: p.next_lap_delta_s * 20,
  }));
  el("liveFeedBand").textContent = rows.length
    ? `${rows.length} predictions`
    : "Waiting";
  drawLineChart(canvas, rows, [
    { key: "wear", label: "Wear %", color: "#ff6b6b", fixedMax: 100, unit: "%" },
    { key: "cliff", label: "Cliff %", color: "#f5c45c", fixedMax: 100, unit: "%" },
    { key: "delta", label: "Δ ×20", color: "#49c6d1", fixedMax: 100, unit: "" },
  ], { xLabel: "Predictions", yUnit: "%" });
}

async function promoteShadowChallenger() {
  const c = state.shadow?.promotion_candidate;
  if (!c) {
    setNotice("No promotion candidate available yet.", "error");
    return;
  }
  try {
    el("shadowPromoteViewBtn").disabled = true;
    const result = await api("/shadow/promote", { method: "POST" });
    state.shadow = { active: false, challenger_backend: "none", total_predictions: 0, recent: [] };
    setNotice(
      `Promoted ${result.challenger_backend} → champion. ` +
      `Improvement: ${num(result.improvement_s * 1000, 0)}ms (${num(result.improvement_pct, 1)}%). ` +
      `Active: ${result.active_artifact_id}`
    );
    await refreshHealth();
    await refreshArtifacts();
    renderShadow();
  } catch (error) {
    setNotice(`Promotion failed: ${error.message}`, "error");
  } finally {
    el("shadowPromoteViewBtn").disabled = false;
  }
}

function populateTrainingRealDataSelect() {
  const select = el("trainingRealDataSelect");
  const prevSelected = new Set(
    Array.from(select.selectedOptions).map((o) => o.value)
  );
  select.innerHTML = "";
  for (const dataset of state.replayDatasets || []) {
    const opt = document.createElement("option");
    opt.value = dataset.path;
    opt.textContent = dataset.path;
    if (prevSelected.has(dataset.path)) opt.selected = true;
    select.appendChild(opt);
  }
}

function populateTrainingReplayDatasetSelect() {
  const select = el("trainingReplayDatasetSelect");
  if (!select) return;
  const prevValue = select.value || state.trainingReplayDataset || state.selectedReplayDataset;
  select.innerHTML = "";
  for (const dataset of state.replayDatasets || []) {
    const opt = document.createElement("option");
    opt.value = dataset.path;
    opt.textContent = dataset.path;
    select.appendChild(opt);
  }
  const fallback =
    state.replayDatasets.find((dataset) => dataset.path === state.selectedReplayDataset)?.path
    || state.replayDatasets[0]?.path
    || "examples/replay_telemetry.csv";
  select.value = state.replayDatasets.some((dataset) => dataset.path === prevValue) ? prevValue : fallback;
  state.trainingReplayDataset = select.value || fallback;
}

async function runTraining() {
  const baselapRaw = el("trainingBaselapInput").value.trim();
  const selectedPaths = Array.from(el("trainingRealDataSelect").selectedOptions).map((o) => o.value);
  const replayDatasetPath = el("trainingReplayDatasetSelect").value || state.trainingReplayDataset || state.selectedReplayDataset;
  const payload = {
    backend: el("trainingBackendSelect").value,
    laps: Number(el("trainingLapsInput").value || 28),
    seeds: Number(el("trainingSeedsInput").value || 64),
    rounds: Number(el("trainingRoundsInput").value || 140),
    base_lap_time_s: baselapRaw ? Number(baselapRaw) : null,
    real_data: selectedPaths.length ? selectedPaths : null,
    replay_dataset_path: replayDatasetPath,
    use_mlflow: el("trainingMlflowInput").checked,
    register_artifact: el("trainingRegisterInput").checked,
  };
  try {
    el("trainingRunButton").disabled = true;
    el("trainingState").textContent = "Starting…";
    state.trainingReplayDataset = replayDatasetPath;
    const result = await api("/training/run", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.activeTrainingJobId = result.job_id;
    setNotice(`Training job started: ${result.job_id} (${result.backend}) · replay ${shortRunId(replayDatasetPath)}`);
    scheduleTrainingPoll();
  } catch (error) {
    el("trainingRunButton").disabled = false;
    el("trainingState").textContent = "Error";
    setNotice(`Could not start training: ${error.message}`, "error");
  }
}

function scheduleTrainingPoll() {
  clearTimeout(state.trainingPollTimer);
  if (!state.activeTrainingJobId) return;
  state.trainingPollTimer = setTimeout(async () => {
    try {
      const jobs = await api("/training/jobs");
      state.trainingJobs = jobs.jobs || [];
      renderTrainingJobs();
      const active = state.trainingJobs.find((j) => j.job_id === state.activeTrainingJobId);
      if (active?.status === "running") {
        scheduleTrainingPoll();
      } else {
        el("trainingRunButton").disabled = false;
        if (active?.status === "done") {
          setNotice(
            `Training complete: ${active.backend}` +
            (active.artifact_id ? ` · artifact: ${active.artifact_id}` : "")
          );
          await refreshArtifacts();
        } else if (active?.status === "error") {
          setNotice(`Training failed: ${active.error}`, "error");
          el("trainingState").textContent = "Error";
        }
        state.activeTrainingJobId = null;
      }
    } catch (_err) {
      scheduleTrainingPoll();
    }
  }, 2000);
}

function renderTrainingJobs() {
  const jobs = state.trainingJobs || [];
  const list = el("trainingJobList");
  list.innerHTML = "";
  const active = jobs.find((j) => j.job_id === state.activeTrainingJobId);
  el("trainingState").textContent = active
    ? `${active.backend} running…`
    : jobs.length ? `${jobs.length} job(s)` : "Idle";
  if (!jobs.length) {
    list.innerHTML = `<div class="training-job-item"><span>No training jobs yet</span></div>`;
    return;
  }
  for (const job of jobs.slice(0, 6)) {
    const item = document.createElement("div");
    const statusClass = job.status === "done" ? "pass" : job.status === "error" ? "fail" : "running";
    item.className = `training-job-item ${statusClass}`;
    const elapsed = job.completed_at
      ? `${num((job.completed_at - job.started_at), 0)}s`
      : "…";
    item.innerHTML = `
      <div class="training-job-head">
        <strong>${job.backend}</strong>
        <span class="training-job-id">${job.job_id}</span>
        <span class="training-job-status">${job.status}</span>
        <span class="training-job-elapsed">${elapsed}</span>
      </div>
      ${job.replay_dataset_path ? `<div class="training-job-artifact" title="${job.replay_dataset_path}">replay: ${shortRunId(job.replay_dataset_path)}</div>` : ""}
      ${job.artifact_id ? `<div class="training-job-artifact" title="${job.artifact_id}">artifact: ${shortRunId(job.artifact_id)}</div>` : ""}
      ${job.error ? `<div class="training-job-error">${job.error}</div>` : ""}
      ${(job.log || []).slice(-2).map((line) => `<div class="training-job-log">${line}</div>`).join("")}
    `;
    list.appendChild(item);
  }
}

el("promoteArtifactBtn").addEventListener("click", () => promoteSelectedArtifact(false));
el("forcePromoteArtifactBtn").addEventListener("click", () => promoteSelectedArtifact(true));
el("deployArtifactBtn").addEventListener("click", deploySelectedArtifact);
async function promoteSelectedArtifact(force) {
  const id = state.selectedArtifactId;
  if (!id) { setNotice("Select an artifact first.", "error"); return; }
  try {
    el("promoteArtifactBtn").disabled = true;
    el("forcePromoteArtifactBtn").disabled = true;
    const url = `/artifacts/${encodeURIComponentPath(id)}/promote${force ? "?force=true" : ""}`;
    await api(url, { method: "POST" });
    setNotice(`Promoted: ${id}`);
    await refreshArtifacts();
  } catch (error) {
    const detail = error.detail;
    let msg;
    if (detail && typeof detail === "object" && Array.isArray(detail.failures)) {
      msg = detail.failures.join(" · ");
    } else if (typeof detail === "string") {
      msg = detail;
    } else {
      msg = error.message;
    }
    setNotice(`Promotion blocked — ${msg}`, "error");
  } finally {
    el("promoteArtifactBtn").disabled = false;
    el("forcePromoteArtifactBtn").disabled = false;
  }
}

async function deploySelectedArtifact() {
  const id = state.selectedArtifactId;
  if (!id) { setNotice("Select an artifact first.", "error"); return; }
  try {
    el("deployArtifactBtn").disabled = true;
    const payload = await api(`/model/artifact?artifact_id=${encodeURIComponent(id)}`, { method: "POST" });
    el("modelValue").textContent = payload.active_backend || "-";
    setNotice(`Deployed: ${payload.active_artifact_id} (${payload.active_backend})`);
    await refreshHealth();
    await refreshArtifacts();
  } catch (error) {
    setNotice(`Deploy failed: ${error.message}`, "error");
  } finally {
    el("deployArtifactBtn").disabled = false;
  }
}

el("shadowEnableButton").addEventListener("click", enableShadow);
el("shadowDisableButton").addEventListener("click", disableShadow);
el("shadowPromoteViewBtn").addEventListener("click", promoteShadowChallenger);
el("trainingRunButton").addEventListener("click", runTraining);
el("openf1FleetExportButton").addEventListener("click", exportOpenF1FleetIntervals);
el("validateDatasetButton").addEventListener("click", runReplayCheck);
el("replayRunButton").addEventListener("click", runReplayCheck);
el("regressionRunButton").addEventListener("click", runRegressionCheck);
el("smokeRunButton").addEventListener("click", runSmokeCheck);
el("replayModeBtn").addEventListener("click", () => setLiveMode("replay"));
el("liveModeBtn").addEventListener("click", () => setLiveMode("live"));
el("liveReplayStartBtn").addEventListener("click", startLiveReplay);
el("liveStopBtn").addEventListener("click", stopLiveData);
el("liveTimingStartBtn").addEventListener("click", startLiveTiming);
el("liveTimingStopBtn").addEventListener("click", stopLiveData);
el("liveTabButton").addEventListener("click", () => setActiveTab("live"));
el("evaluationTabButton").addEventListener("click", () => setActiveTab("evaluation"));
el("promotionTabButton").addEventListener("click", () => setActiveTab("promotion"));
el("trainingTabButton").addEventListener("click", () => setActiveTab("training"));
el("historyTabButton").addEventListener("click", () => setActiveTab("history"));

setActiveTab("live");
refreshHealth();
refreshModels();
refreshMetrics().catch(() => {});
refreshOps().catch(() => {});
refreshBenchmark().catch(() => {});
refreshHistory();
refreshExternalLinks();
setControls();
setInterval(refreshHealth, 3000);
setInterval(refreshOps, 5000);
setInterval(refreshExternalLinks, 30000);
