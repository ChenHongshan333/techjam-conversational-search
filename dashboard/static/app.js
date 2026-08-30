const state = {
  session: null,
  testCases: [],
  revealTarget: false,
  evaluationPoll: null,
};

const $ = (selector) => document.querySelector(selector);

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed: ${response.status}`);
  return payload;
}

function formatNumber(value, digits = 3) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function formatPercent(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadMetrics() {
  const { baseline, current } = await request("/api/metrics");
  $("#metric-hit-baseline").textContent = `Baseline ${formatPercent(baseline.hit_rate_at_10)}`;
  $("#metric-mrr-baseline").textContent = `Baseline ${formatNumber(baseline.mrr)}`;
  $("#metric-mttc-baseline").textContent = `Baseline ${formatNumber(baseline.mttc, 2)}`;
  if (!current) return;
  $("#metric-score").textContent = formatNumber(current.recommended_technical_score, 4);
  $("#metric-score-delta").textContent = `+${formatNumber(current.recommended_technical_score - baseline.technical_score, 4)} from baseline`;
  $("#metric-hit").textContent = formatPercent(current.hit_rate_at_10);
  $("#metric-mrr").textContent = formatNumber(current.mrr, 4);
  $("#metric-mttc").textContent = formatNumber(current.mttc, 2);
  renderScenarioTable(current.scenario_metrics || {});
}

function renderScenarioTable(metrics) {
  const order = ["buying", "browsing", "intent_override", "boundary"];
  $("#scenario-table").innerHTML = order.filter((name) => metrics[name]).map((name) => {
    const row = metrics[name];
    return `<tr>
      <td>${escapeHtml(name.replaceAll("_", " "))}</td>
      <td>${row.sample_count}</td>
      <td>${formatPercent(row.hit_rate_at_10)}</td>
      <td>${formatNumber(row.mrr, 4)}</td>
      <td>${formatNumber(row.mttc, 2)}</td>
    </tr>`;
  }).join("");
}

async function loadTestCases() {
  const scenario = $("#scenario-filter").value;
  const query = scenario ? `?scenario=${encodeURIComponent(scenario)}` : "";
  const payload = await request(`/api/test-cases${query}`);
  state.testCases = payload.test_cases;
  $("#case-select").innerHTML = state.testCases.map((item) => (
    `<option value="${escapeHtml(item.sample_id)}">${escapeHtml(item.sample_id)} · ${escapeHtml(item.difficulty_bucket)} · ${escapeHtml(item.target_title)}</option>`
  )).join("");
  $("#start-case").disabled = state.testCases.length === 0;
}

async function startReplay() {
  const sampleId = $("#case-select").value;
  if (!sampleId) return;
  setFooter("STARTING REPLAY");
  const snapshot = await request("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ sample_id: sampleId }),
  });
  state.session = snapshot;
  renderSession(snapshot);
  setFooter("REPLAY READY");
}

function renderSession(snapshot) {
  $("#conversation").classList.remove("empty-state");
  $("#conversation").innerHTML = "";
  $("#turn-counter").textContent = `Turn ${snapshot.turn} / ${snapshot.max_turns}`;
  $("#next-turn").disabled = snapshot.finished;
  $("#auto-run").disabled = snapshot.finished;
  $("#case-meta").classList.remove("hidden");
  $("#case-meta").textContent = `${snapshot.sample_id} / ${snapshot.scenario_type.replaceAll("_", " ")} / ${snapshot.difficulty_bucket}`;
  $("#target-title").textContent = snapshot.target.title;
  $("#target-asin").textContent = snapshot.target.parent_asin;
  toggleTarget();
  renderResult(snapshot.summary);
}

async function nextTurn() {
  if (!state.session || state.session.finished) return;
  disableReplayActions(true);
  setFooter("RUNNING TURN");
  try {
    const event = await request(`/api/sessions/${state.session.session_id}/step`, { method: "POST" });
    appendEvent(event);
    state.session.finished = event.finished;
    state.session.summary = event.summary;
    state.session.turn = event.finished ? event.turn : event.turn + 1;
    $("#turn-counter").textContent = event.finished ? `Finished at turn ${event.turn}` : `Turn ${event.turn + 1} / 10`;
    renderDiagnostics(event.diagnostics);
    renderResult(event.summary);
  } finally {
    disableReplayActions(Boolean(state.session?.finished));
    setFooter(state.session?.finished ? "REPLAY COMPLETE" : "READY");
  }
}

async function autoRun() {
  if (!state.session || state.session.finished) return;
  disableReplayActions(true);
  setFooter("AUTO RUNNING");
  try {
    const payload = await request(`/api/sessions/${state.session.session_id}/run`, { method: "POST" });
    for (const event of payload.events) {
      appendEvent(event);
      renderDiagnostics(event.diagnostics);
      await new Promise((resolve) => setTimeout(resolve, 180));
    }
    state.session = payload.snapshot;
    $("#turn-counter").textContent = state.session.summary.hit
      ? `Hit at turn ${state.session.summary.first_hit_turn}`
      : "Finished / miss";
    renderResult(state.session.summary);
  } finally {
    disableReplayActions(true);
    setFooter("REPLAY COMPLETE");
  }
}

function appendEvent(event) {
  const targetClass = (recommendation) => (
    state.revealTarget && recommendation.is_target ? " target" : ""
  );
  const targetMark = (recommendation) => (
    state.revealTarget && recommendation.is_target ? '<span class="target-mark">TARGET</span>' : ""
  );
  const recommendations = event.recommendations.map((item) => (
    `<div class="recommendation${targetClass(item)}" data-target="${item.is_target}">
      <span class="recommendation-rank">${String(item.rank).padStart(2, "0")}</span>
      <div>
        <div class="recommendation-title">${escapeHtml(item.title)}</div>
        <code>${escapeHtml(item.parent_asin)}</code>
      </div>
      ${targetMark(item)}
    </div>`
  )).join("");
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="turn-label"><span>Turn ${event.turn}</span><span>${event.scored_hit ? "Target found" : "No hit"}</span></div>
    <div class="message message-user">${escapeHtml(event.user_message)}</div>
    <div class="message message-agent">${escapeHtml(event.agent_message)}</div>
    <span class="attribute-badge">ask_attribute / ${escapeHtml(event.ask_attribute ?? "null")}</span>
    <div class="recommendations">${recommendations}</div>`;
  $("#conversation").appendChild(turn);
  $("#conversation").scrollTop = $("#conversation").scrollHeight;
}

function renderDiagnostics(diagnostics = {}) {
  const constraints = diagnostics.active_constraints || [];
  $("#active-constraints").innerHTML = constraints.length
    ? constraints.map((item) => `<span class="tag">${escapeHtml(item.attribute)} / ${escapeHtml(item.value)}</span>`).join("")
    : '<span class="muted">None</span>';
  const values = [
    diagnostics.exact_candidate_count,
    diagnostics.intersection_candidate_count,
    diagnostics.bm25_and_candidate_count,
    diagnostics.bm25_or_candidate_count,
    diagnostics.dense_identity_candidate_count,
    diagnostics.dense_attribute_candidate_count,
    diagnostics.fused_candidate_count,
  ];
  $("#candidate-flow").querySelectorAll("dd").forEach((node, index) => {
    node.textContent = values[index] ?? "—";
  });
  $("#query-terms").textContent = (diagnostics.query_terms || []).join(" · ") || "—";
  $("#semantic-query").textContent = diagnostics.semantic_query || "—";
  const statuses = [
    diagnostics.llm_rewrite_used ? "LLM rewrite" : "deterministic query",
    diagnostics.dense_retrieval_enabled ? "dense on" : "dense off",
    diagnostics.rerank_enabled ? "rerank on" : "rerank off",
    diagnostics.candidate_rotation_active ? "rotation on" : "rotation off",
  ];
  $("#retrieval-status").innerHTML = statuses
    .map((value) => `<span class="tag">${escapeHtml(value)}</span>`)
    .join("");
  const errors = [diagnostics.llm_rewrite_error, diagnostics.dense_retrieval_error, diagnostics.rerank_error]
    .filter(Boolean);
  $("#retrieval-status").title = errors.join("\n");
}

function renderResult(summary = {}) {
  if (!summary.finished) {
    $("#session-result").innerHTML = '<span class="muted">Session in progress</span>';
    return;
  }
  $("#session-result").innerHTML = `<div class="result-grid">
    <div><span>Outcome</span><strong>${summary.hit ? "HIT" : "MISS"}</strong></div>
    <div><span>Turn</span><strong>${summary.first_hit_turn ?? "11"}</strong></div>
    <div><span>Rank</span><strong>${summary.best_rank ?? "—"}</strong></div>
    <div><span>Score</span><strong>${formatNumber(summary.technical_score, 3)}</strong></div>
  </div>`;
}

function toggleTarget() {
  state.revealTarget = $("#show-target").checked;
  $("#target-details").classList.toggle("hidden", !state.revealTarget || !state.session);
  document.querySelectorAll(".recommendation[data-target='true']").forEach((row) => {
    row.classList.toggle("target", state.revealTarget);
    const existing = row.querySelector(".target-mark");
    if (state.revealTarget && !existing) row.insertAdjacentHTML("beforeend", '<span class="target-mark">TARGET</span>');
    if (!state.revealTarget && existing) existing.remove();
  });
}

function disableReplayActions(disabled) {
  $("#next-turn").disabled = disabled;
  $("#auto-run").disabled = disabled;
}

async function runEvaluation() {
  $("#run-evaluation").disabled = true;
  $("#evaluation-status").textContent = "Starting evaluation…";
  const job = await request("/api/evaluations", { method: "POST" });
  pollEvaluation(job.evaluation_id);
}

async function pollEvaluation(evaluationId) {
  clearTimeout(state.evaluationPoll);
  const job = await request(`/api/evaluations/${evaluationId}`);
  $("#evaluation-status").textContent = job.status === "running" ? "Evaluating 200 cases…" : job.status;
  if (job.status === "completed") {
    $("#run-evaluation").disabled = false;
    $("#evaluation-status").textContent = "Evaluation complete";
    await loadMetrics();
    setFooter("SCORE UPDATED");
    return;
  }
  if (job.status === "failed") {
    $("#run-evaluation").disabled = false;
    $("#evaluation-status").textContent = job.error || "Evaluation failed";
    return;
  }
  state.evaluationPoll = setTimeout(() => pollEvaluation(evaluationId), 1000);
}

function setFooter(value) {
  $("#footer-status").textContent = value;
}

async function initialize() {
  try {
    await Promise.all([loadMetrics(), loadTestCases()]);
    setFooter("READY");
  } catch (error) {
    setFooter("ERROR");
    $("#conversation").textContent = error.message;
  }
}

$("#scenario-filter").addEventListener("change", loadTestCases);
$("#start-case").addEventListener("click", () => startReplay().catch((error) => setFooter(error.message)));
$("#next-turn").addEventListener("click", () => nextTurn().catch((error) => setFooter(error.message)));
$("#auto-run").addEventListener("click", () => autoRun().catch((error) => setFooter(error.message)));
$("#show-target").addEventListener("change", toggleTarget);
$("#run-evaluation").addEventListener("click", () => runEvaluation().catch((error) => {
  $("#evaluation-status").textContent = error.message;
  $("#run-evaluation").disabled = false;
}));

initialize();
