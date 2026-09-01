const state = {
  session: null,
  replayEvents: [],
  replayIndex: 0,
  finalSnapshot: null,
  testCases: [],
  revealTarget: false,
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

function humanize(value) {
  return String(value || "—").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function sectorScore(row) {
  const efficiency = Math.max(0, Math.min(1, (11 - Number(row.mttc || 11)) / 10));
  return .5 * Number(row.hit_rate_at_10 || 0) + .3 * Number(row.mrr || 0) + .2 * efficiency;
}

async function loadMetrics() {
  const { baseline, current } = await request("/api/metrics");
  $("#metric-hit-baseline").textContent = `Baseline ${formatPercent(baseline.hit_rate_at_10)}`;
  $("#metric-mrr-baseline").textContent = `Baseline ${formatNumber(baseline.mrr)}`;
  $("#metric-mttc-baseline").textContent = `Baseline ${formatNumber(baseline.mttc, 2)}`;
  if (!current) return;
  const delta = current.recommended_technical_score - baseline.technical_score;
  $("#metric-score").textContent = formatNumber(current.recommended_technical_score, 4);
  $("#metric-score-delta").textContent = `${delta >= 0 ? "+" : ""}${formatNumber(delta, 4)} from baseline`;
  $("#metric-hit").textContent = formatPercent(current.hit_rate_at_10);
  $("#metric-mrr").textContent = formatNumber(current.mrr, 4);
  $("#metric-mttc").textContent = formatNumber(current.mttc, 2);
  renderScenarioTable(current.scenario_metrics || {});
  $("#run-evaluation").disabled = true;
  $("#run-evaluation").textContent = "Validated offline";
  $("#evaluation-status").textContent = "Validated on 200 public cases";
}

function renderScenarioTable(metrics) {
  const order = ["buying", "browsing", "intent_override", "boundary"];
  const rows = order.filter((name) => metrics[name]).map((name) => {
    const row = metrics[name];
    return `<tr>
      <td>${escapeHtml(humanize(name))}</td>
      <td>${row.sample_count}</td>
      <td>${formatPercent(row.hit_rate_at_10)}</td>
      <td>${formatNumber(row.mrr, 4)}</td>
      <td>${formatNumber(row.mttc, 2)}</td>
      <td>${formatNumber(sectorScore(row), 4)}</td>
    </tr>`;
  });
  $("#scenario-table").innerHTML = rows.length
    ? rows.join("")
    : '<tr><td colspan="6" class="empty-cell">Run an evaluation to see scenario metrics.</td></tr>';
}

async function loadTestCases() {
  const scenario = $("#scenario-filter").value;
  const query = scenario ? `?scenario=${encodeURIComponent(scenario)}` : "";
  const payload = await request(`/api/test-cases${query}`);
  state.testCases = payload.test_cases;
  $("#case-select").innerHTML = state.testCases.map((item) => (
    `<option value="${escapeHtml(item.sample_id)}">${escapeHtml(item.sample_id)} · ${escapeHtml(humanize(item.difficulty_bucket))} · ${escapeHtml(item.target_title)}</option>`
  )).join("");
  $("#start-case").disabled = state.testCases.length === 0;
}

async function startReplay() {
  const sampleId = $("#case-select").value;
  if (!sampleId) return;
  const startButton = $("#start-case");
  const originalLabel = startButton.textContent;
  startButton.disabled = true;
  startButton.textContent = "Starting · allow 20–30s";
  setFooter("Initializing hosted catalog · local dashboard has no delay");
  try {
    const payload = await request("/api/replay", {
      method: "POST",
      body: JSON.stringify({ sample_id: sampleId }),
    });
    state.session = payload.initial;
    state.replayEvents = payload.events || [];
    state.replayIndex = 0;
    state.finalSnapshot = payload.snapshot;
    renderSession(payload.initial);
    setFooter("Conversation ready");
  } finally {
    startButton.disabled = false;
    startButton.textContent = originalLabel;
  }
}

function renderSession(snapshot) {
  $("#conversation").classList.remove("empty-state");
  $("#conversation").innerHTML = "";
  renderPendingUser(snapshot.current_user_message);
  renderProfile(snapshot.user_profile || {}, snapshot.sample_id);
  resetInsights();
  $("#turn-counter").textContent = `Turn ${snapshot.turn} of ${snapshot.max_turns}`;
  $("#next-turn").disabled = snapshot.finished;
  $("#auto-run").disabled = snapshot.finished;
  $("#case-meta").classList.remove("hidden");
  $("#case-meta").textContent = `${snapshot.sample_id} · ${humanize(snapshot.scenario_type)} · ${humanize(snapshot.difficulty_bucket)}`;
  $("#target-title").textContent = snapshot.target.title;
  $("#target-asin").textContent = snapshot.target.parent_asin;
  toggleTarget();
  renderResult(snapshot.summary);
}

function renderProfile(profile, sampleId) {
  const tags = profile.preference_tags || [];
  $("#customer-avatar").textContent = String(sampleId || "G").replace("public_", "").slice(-2);
  $("#profile-title").textContent = "Guest shopper";
  $("#customer-id").textContent = `Anonymous profile · ${sampleId}`;
  $("#profile-summary").textContent = profile.summary || "No aggregate shopping summary is available.";
  $("#profile-tags").innerHTML = tags.length
    ? tags.map((tag) => `<span class="pill active">${escapeHtml(humanize(tag))}</span>`).join("")
    : '<span class="empty-copy">No profile priorities</span>';
  $("#profile-frequency").textContent = profile.purchase_frequency || "Not available";
  $("#profile-rating").textContent = profile.average_prior_rating == null
    ? "Not available"
    : `${Number(profile.average_prior_rating).toFixed(1)} / 5`;
  $("#profile-style").textContent = humanize(profile.rating_style || "Not available");
}

function resetInsights() {
  $("#intent-label").textContent = "Waiting for the agent";
  $("#intent-detail").textContent = "Intent will be classified from the customer's language on the next turn.";
  $("#intent-confidence").textContent = "—";
  $("#intent-confidence-bar").style.width = "0%";
  $("#intent-meta").innerHTML = "";
  $("#active-constraints").innerHTML = '<span class="empty-copy">No requirements yet</span>';
  $("#why-summary").textContent = "Ranking evidence will appear after the first response.";
  $("#why-signals").innerHTML = "";
  $("#profile-use").textContent = "";
  $("#retrieval-status").innerHTML = "";
}

function renderPendingUser(message) {
  if (!message) return;
  document.querySelector(".pending-user-turn")?.remove();
  const pending = document.createElement("article");
  pending.className = "turn pending-user-turn pending-user";
  pending.innerHTML = `<div class="turn-label"><span>Customer reply ready</span></div>
    <div class="message-row customer">
      <div class="message message-user">${escapeHtml(message)}</div>
      <div class="message-avatar">ME</div>
    </div>`;
  $("#conversation").appendChild(pending);
  scrollConversation();
}

async function nextTurn() {
  if (!state.session || state.session.finished || state.replayIndex >= state.replayEvents.length) return;
  disableReplayActions(true);
  setFooter("Seekly is responding");
  try {
    const event = state.replayEvents[state.replayIndex];
    state.replayIndex += 1;
    appendEvent(event);
    state.session.finished = event.finished;
    state.session.summary = event.summary;
    state.session.turn = event.finished ? event.turn : event.turn + 1;
    $("#turn-counter").textContent = event.finished ? `Finished at turn ${event.turn}` : `Turn ${event.turn + 1} of 10`;
    renderDiagnostics(event.diagnostics, event.intent, event.ranking_explanation);
    renderResult(event.summary);
  } finally {
    disableReplayActions(Boolean(state.session?.finished));
    setFooter(state.session?.finished ? "Conversation complete" : "Ready");
  }
}

async function autoRun() {
  if (!state.session || state.session.finished) return;
  disableReplayActions(true);
  setFooter("Playing conversation");
  try {
    while (state.replayIndex < state.replayEvents.length) {
      const event = state.replayEvents[state.replayIndex];
      state.replayIndex += 1;
      appendEvent(event);
      renderDiagnostics(event.diagnostics, event.intent, event.ranking_explanation);
      renderResult(event.summary);
      await new Promise((resolve) => setTimeout(resolve, 230));
    }
    state.session = state.finalSnapshot;
    $("#turn-counter").textContent = state.session.summary.hit
      ? `Found at turn ${state.session.summary.first_hit_turn}`
      : "Conversation complete";
    renderResult(state.session.summary);
  } finally {
    disableReplayActions(true);
    setFooter("Conversation complete");
  }
}

function productCard(item) {
  const explanation = item.explanation || {};
  const reasons = (explanation.reasons || [])
    .map((reason) => `<li>${escapeHtml(reason)}</li>`)
    .join("");
  const categories = (item.categories || []).slice(-2).join(" › ");
  const rating = Number(item.average_rating || 0);
  const targetMark = state.revealTarget && item.is_target ? '<span class="target-mark">TARGET</span>' : "";
  const targetClass = state.revealTarget && item.is_target ? " target" : "";
  return `<article class="recommendation${targetClass}" data-target="${item.is_target}">
    <div class="product-heading">
      <span class="recommendation-rank">${String(item.rank).padStart(2, "0")}</span>
      <div class="recommendation-title">${escapeHtml(item.title)}</div>
      ${targetMark || (rating ? `<span class="rating">★ ${rating.toFixed(1)}</span>` : "")}
    </div>
    <p class="recommendation-category">${escapeHtml(categories || item.parent_asin)}</p>
    <details class="product-signals">
      <summary>${explanation.matched_count || 0}/${explanation.constraint_count || 0} requirements matched · Why ranked here</summary>
      <ul>${reasons}</ul>
    </details>
  </article>`;
}

function appendEvent(event) {
  document.querySelector(".pending-user-turn")?.remove();
  const recommendations = (event.recommendations || []).map(productCard).join("");
  const recommendationBlock = recommendations
    ? `<div class="result-intro"><strong>Recommended for you</strong><span>${event.recommendations.length} ranked result${event.recommendations.length === 1 ? "" : "s"}</span></div>
       <div class="recommendations">${recommendations}</div>`
    : "";
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="turn-label"><span>Turn ${event.turn}</span><span>${event.scored_hit ? "Target found" : "Search refined"}</span></div>
    <div class="message-row customer">
      <div class="message message-user">${escapeHtml(event.user_message)}</div>
      <div class="message-avatar">ME</div>
    </div>
    <div class="message-row agent">
      <div class="message-avatar">S</div>
      <div class="message message-agent">${escapeHtml(event.agent_message)}</div>
    </div>
    ${event.ask_attribute ? `<div class="attribute-badge">Clarifying: ${escapeHtml(humanize(event.ask_attribute))}</div>` : ""}
    ${recommendationBlock}`;
  $("#conversation").appendChild(turn);
  if (event.next_user_message && !event.finished) renderPendingUser(event.next_user_message);
  scrollConversation();
}

function renderDiagnostics(diagnostics = {}, intent = {}, explanation = {}) {
  const constraints = diagnostics.active_constraints || [];
  const superseded = diagnostics.superseded_constraints || [];
  $("#active-constraints").innerHTML = constraints.length || superseded.length
    ? [
        ...constraints.map((item) => `<span class="pill active">${escapeHtml(item.value)}</span>`),
        ...superseded.map((item) => `<span class="pill retracted">${escapeHtml(item.value)}</span>`),
      ].join("")
    : '<span class="empty-copy">No concrete requirements yet</span>';

  renderIntent(intent);
  renderExplanation(explanation);

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
    diagnostics.llm_rewrite_used ? "LLM rewrite" : "Deterministic query",
    diagnostics.dense_retrieval_applied ? "Dense applied" : "Dense not applied",
    diagnostics.rerank_applied ? "Rerank applied" : "Rerank not applied",
  ];
  $("#retrieval-status").innerHTML = statuses
    .map((value) => `<span class="pill">${escapeHtml(value)}</span>`)
    .join("");
  const errors = [diagnostics.llm_rewrite_error, diagnostics.dense_retrieval_error, diagnostics.rerank_error]
    .filter(Boolean);
  $("#retrieval-status").title = errors.join("\n");
}

function renderIntent(intent = {}) {
  const confidence = Math.max(0, Math.min(1, Number(intent.confidence || 0)));
  $("#intent-label").textContent = intent.label || "Intent forming";
  $("#intent-detail").textContent = intent.detail || "Seekly is collecting more evidence.";
  $("#intent-confidence").textContent = `${Math.round(confidence * 100)}%`;
  $("#intent-confidence-bar").style.width = `${Math.round(confidence * 100)}%`;
  const metadata = [
    intent.strategy ? humanize(intent.strategy) : null,
    intent.source ? `Source: ${humanize(intent.source)}` : null,
  ].filter(Boolean);
  $("#intent-meta").innerHTML = metadata.map((value) => `<span>${escapeHtml(value)}</span>`).join("");
}

function renderExplanation(explanation = {}) {
  $("#why-summary").textContent = explanation.summary || "Ranking evidence is not available yet.";
  $("#why-signals").innerHTML = (explanation.signals || [])
    .map((signal) => `<li>${escapeHtml(signal)}</li>`)
    .join("");
  $("#profile-use").textContent = explanation.profile_note || "";
}

function renderResult(summary = {}) {
  if (!summary.finished) {
    $("#session-result").innerHTML = '<span class="empty-copy">Session in progress · the first valid hit ends the replay</span>';
    return;
  }
  $("#session-result").innerHTML = `<div class="result-heading">
      <span class="rail-label">Session outcome</span>
      <strong class="${summary.hit ? "hit" : "miss"}">${summary.hit ? "Hit" : "Miss"}</strong>
    </div>
    <div class="result-grid">
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
    if (state.revealTarget && !existing) row.querySelector(".product-heading")?.insertAdjacentHTML("beforeend", '<span class="target-mark">TARGET</span>');
    if (!state.revealTarget && existing) existing.remove();
  });
}

function disableReplayActions(disabled) {
  $("#next-turn").disabled = disabled;
  $("#auto-run").disabled = disabled;
}

function scrollConversation() {
  const conversation = $("#conversation");
  conversation.scrollTop = conversation.scrollHeight;
}

function setFooter(value) {
  $("#footer-status").textContent = value;
}

async function initialize() {
  try {
    await Promise.all([loadMetrics(), loadTestCases()]);
    setFooter("Ready");
  } catch (error) {
    setFooter("Error");
    $("#conversation").textContent = error.message;
  }
}

$("#scenario-filter").addEventListener("change", loadTestCases);
$("#start-case").addEventListener("click", () => startReplay().catch((error) => setFooter(error.message)));
$("#next-turn").addEventListener("click", () => nextTurn().catch((error) => setFooter(error.message)));
$("#auto-run").addEventListener("click", () => autoRun().catch((error) => setFooter(error.message)));
$("#show-target").addEventListener("change", toggleTarget);
initialize();
