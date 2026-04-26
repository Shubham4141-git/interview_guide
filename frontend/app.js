const API_BASE = (window.API_BASE_URL || "http://localhost:8000") + "/api";

const form = document.querySelector("[data-form]");
const input = document.querySelector("[data-input]");
const userInput = document.querySelector("[data-user]");
const stream = document.getElementById("chat-stream");
const emptyState = document.getElementById("empty-state");
const statusEl = document.getElementById("status-message");
const turnTemplate = document.getElementById("chat-turn-template");
const composerSurface = document.querySelector("[data-composer]");
const profileContent = document.querySelector("[data-profile]");
const resourcesContent = document.querySelector("[data-resources]");
const userDropdown = document.querySelector("[data-user-dropdown]");
const profileCard = document.getElementById("profile-card");
const profileModal = document.getElementById("profile-modal");
const profileDashboard = document.querySelector("[data-profile-dashboard]");
const profileCloseButtons = document.querySelectorAll("[data-profile-close]");

const QUESTION_TYPES = ["Technical", "System design", "Behavioural", "Conceptual"];
const COMPOSER_MAX_HEIGHT = 220;
const COLLAPSIBLE_MESSAGE_MIN_LENGTH = 500;
const COLLAPSIBLE_MESSAGE_MIN_LINES = 10;
const PROFILE_DISTRIBUTION_ORDER = [
  { key: "excellent", label: "Excellent" },
  { key: "strong", label: "Strong" },
  { key: "average", label: "Average" },
  { key: "weak", label: "Weak" },
  { key: "very weak", label: "Very weak" },
];

let isLoading = false;
const turns = [];
let currentProfile = {};
let lastProfileTrigger = null;

const getLastState = () => {
  if (!turns.length) return undefined;
  return turns[turns.length - 1].state;
};

function syncComposerHeight() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, COMPOSER_MAX_HEIGHT)}px`;
  input.style.overflowY = input.scrollHeight > COMPOSER_MAX_HEIGHT ? "auto" : "hidden";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatSkillName(skill) {
  return String(skill || "")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => {
      if (part.toUpperCase() === part) {
        return part;
      }
      return `${part.charAt(0).toUpperCase()}${part.slice(1)}`;
    })
    .join(" ");
}

function truncateLabel(label, max = 18) {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function getSkillEntries(skillStats = {}) {
  return Object.entries(skillStats || {})
    .map(([skill, stats]) => ({
      skill,
      avg: Number(stats?.avg || 0),
      count: Number(stats?.count || 0),
    }))
    .filter((item) => item.skill)
    .sort((a, b) => b.count - a.count || b.avg - a.avg || a.skill.localeCompare(b.skill));
}

function getProfileHeadline(avgScore) {
  if (avgScore >= 4.2) {
    return "Interview-ready";
  }
  if (avgScore >= 3.4) {
    return "Strong momentum";
  }
  if (avgScore >= 2.6) {
    return "Solid baseline";
  }
  return "Needs focused practice";
}

function buildRadarChart(skillStats = {}) {
  const entries = getSkillEntries(skillStats).slice(0, 6);
  if (!entries.length) {
    return `
      <div class="profile-dashboard__empty-chart">
        Skill coverage appears here after a few scored answers.
      </div>
    `;
  }

  const size = 360;
  const center = size / 2;
  const radius = 112;
  const levels = [1, 2, 3, 4, 5];
  const angleStep = (Math.PI * 2) / entries.length;

  const pointAt = (level, idx, distance = radius) => {
    const angle = -Math.PI / 2 + idx * angleStep;
    const ratio = level / 5;
    return {
      x: center + Math.cos(angle) * distance * ratio,
      y: center + Math.sin(angle) * distance * ratio,
    };
  };

  const grid = levels
    .map((level) => {
      const points = entries
        .map((_, idx) => {
          const point = pointAt(level, idx);
          return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
        })
        .join(" ");
      return `<polygon class="radar-chart__grid" points="${points}"></polygon>`;
    })
    .join("");

  const spokes = entries
    .map((_, idx) => {
      const point = pointAt(5, idx);
      return `<line class="radar-chart__spoke" x1="${center}" y1="${center}" x2="${point.x.toFixed(
        2
      )}" y2="${point.y.toFixed(2)}"></line>`;
    })
    .join("");

  const dataPoints = entries
    .map((entry, idx) => {
      const point = pointAt(entry.avg, idx);
      return `${point.x.toFixed(2)},${point.y.toFixed(2)}`;
    })
    .join(" ");

  const dots = entries
    .map((entry, idx) => {
      const point = pointAt(entry.avg, idx);
      return `<circle class="radar-chart__dot" cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(
        2
      )}" r="4"></circle>`;
    })
    .join("");

  const labels = entries
    .map((entry, idx) => {
      const angle = -Math.PI / 2 + idx * angleStep;
      const distance = radius + 28;
      const x = center + Math.cos(angle) * distance;
      const y = center + Math.sin(angle) * distance;
      const anchor = Math.cos(angle) > 0.25 ? "start" : Math.cos(angle) < -0.25 ? "end" : "middle";
      const skill = truncateLabel(formatSkillName(entry.skill), 16);
      return `
        <text class="radar-chart__label" x="${x.toFixed(2)}" y="${(y - 5).toFixed(
          2
        )}" text-anchor="${anchor}">${escapeHtml(skill)}</text>
        <text class="radar-chart__value" x="${x.toFixed(2)}" y="${(y + 11).toFixed(
          2
        )}" text-anchor="${anchor}">${entry.avg.toFixed(1)}/5</text>
      `;
    })
    .join("");

  const rings = levels
    .map((level) => {
      const y = center - (radius * level) / 5;
      return `<text class="radar-chart__ring-label" x="${center}" y="${(y - 4).toFixed(
        2
      )}" text-anchor="middle">${level}</text>`;
    })
    .join("");

  return `
    <svg class="radar-chart__svg" viewBox="0 0 ${size} ${size}" aria-label="Skill radar chart">
      ${grid}
      ${spokes}
      ${rings}
      <polygon class="radar-chart__shape" points="${dataPoints}"></polygon>
      ${dots}
      ${labels}
    </svg>
  `;
}

function buildDistributionRows(profile = {}) {
  const distribution = profile.distribution || {};
  const total = Number(profile.total_answers || 0);
  return PROFILE_DISTRIBUTION_ORDER.map(({ key, label }) => {
    const count = Number(distribution[key] || 0);
    const pct = total ? Math.round((count / total) * 100) : 0;
    return `
      <div class="profile-distribution__row">
        <span class="profile-distribution__label">${label}</span>
        <div class="profile-distribution__track">
          <div class="profile-distribution__fill" style="width: ${count ? Math.max(pct, 8) : 0}%"></div>
        </div>
        <span class="profile-distribution__value">${count}</span>
      </div>
    `;
  }).join("");
}

function buildSkillPills(items = [], variant = "strength") {
  if (!items.length) {
    return `<span class="profile-pill profile-pill--muted">Not enough signal yet</span>`;
  }
  return items
    .slice(0, 5)
    .map(
      (item) =>
        `<span class="profile-pill profile-pill--${variant}">${escapeHtml(formatSkillName(item))}</span>`
    )
    .join("");
}

function buildRecentRows(profile = {}) {
  const recent = Array.isArray(profile.recent) ? profile.recent.slice(0, 4) : [];
  if (!recent.length) {
    return `
      <div class="profile-recent__empty">
        Recent scored answers will appear here after evaluation.
      </div>
    `;
  }

  return recent
    .map((item) => {
      const label = item.question_text || item.skill || "Recent answer";
      const score = Number(item.score || 0);
      const tone = score >= 4 ? "high" : score >= 3 ? "mid" : "low";
      return `
        <div class="profile-recent__item">
          <div class="profile-recent__copy">
            <strong>${escapeHtml(truncateLabel(label, 56))}</strong>
            <p>${escapeHtml(item.feedback || "Feedback will appear after evaluation.")}</p>
          </div>
          <span class="profile-recent__score profile-recent__score--${tone}">${score}/5</span>
        </div>
      `;
    })
    .join("");
}

function renderProfileDashboard(profile = {}) {
  if (!profileDashboard) {
    return;
  }

  if (!profile.total_answers) {
    profileDashboard.innerHTML = `
      <div class="profile-dashboard__empty">
        <p class="profile-dashboard__empty-title">No profile data yet</p>
        <p class="profile-dashboard__empty-copy">
          Submit a few answers for evaluation to unlock your progress dashboard.
        </p>
      </div>
    `;
    return;
  }

  const avgScore = Number(profile.avg_score || 0);
  const headline = getProfileHeadline(avgScore);
  const skillEntries = getSkillEntries(profile.skill_stats || {});
  const strongestSkill = profile.strength_skills?.[0] || skillEntries[0]?.skill || "Building";
  const focusSkill =
    profile.weakness_skills?.[0] ||
    [...skillEntries].sort((a, b) => a.avg - b.avg || b.count - a.count)[0]?.skill ||
    "None yet";

  profileDashboard.innerHTML = `
    <div class="profile-dashboard__header">
      <div>
        <p class="profile-dashboard__eyebrow">Profile Dashboard</p>
        <h2 class="profile-dashboard__title" id="profile-dashboard-title">${headline}</h2>
        <p class="profile-dashboard__sub">
          A quick view of where this user is strong, where they need focus, and how recent evaluations are trending.
        </p>
      </div>
    </div>

    <div class="profile-kpis">
      <div class="profile-kpi">
        <span class="profile-kpi__label">Answers scored</span>
        <strong class="profile-kpi__value">${profile.total_answers}</strong>
      </div>
      <div class="profile-kpi">
        <span class="profile-kpi__label">Average score</span>
        <strong class="profile-kpi__value">${avgScore.toFixed(1)}/5</strong>
      </div>
      <div class="profile-kpi">
        <span class="profile-kpi__label">Top strength</span>
        <strong class="profile-kpi__value">${escapeHtml(formatSkillName(strongestSkill))}</strong>
      </div>
      <div class="profile-kpi">
        <span class="profile-kpi__label">Main focus</span>
        <strong class="profile-kpi__value">${escapeHtml(formatSkillName(focusSkill))}</strong>
      </div>
    </div>

    <div class="profile-dashboard__grid">
      <section class="profile-panel">
        <div class="profile-panel__head">
          <h3>Skill map</h3>
          <p>Radar view of the most active skills in this profile.</p>
        </div>
        <div class="radar-chart">
          ${buildRadarChart(profile.skill_stats || {})}
        </div>
      </section>

      <section class="profile-panel">
        <div class="profile-panel__head">
          <h3>Score distribution</h3>
          <p>How recent answers are spread across performance bands.</p>
        </div>
        <div class="profile-distribution">
          ${buildDistributionRows(profile)}
        </div>
      </section>

      <section class="profile-panel">
        <div class="profile-panel__head">
          <h3>Strengths</h3>
          <p>Skills that are consistently scoring well.</p>
        </div>
        <div class="profile-pill-list">
          ${buildSkillPills(profile.strength_skills || [], "strength")}
        </div>
      </section>

      <section class="profile-panel">
        <div class="profile-panel__head">
          <h3>Focus next</h3>
          <p>Best opportunities for the next round of practice.</p>
        </div>
        <div class="profile-pill-list">
          ${buildSkillPills(profile.weakness_skills || [], "weakness")}
        </div>
      </section>

      <section class="profile-panel profile-panel--wide">
        <div class="profile-panel__head">
          <h3>Recent evaluations</h3>
          <p>Latest scoring feedback, trimmed down for quick review.</p>
        </div>
        <div class="profile-recent">
          ${buildRecentRows(profile)}
        </div>
      </section>
    </div>
  `;
}

function openProfileDashboard() {
  if (!profileModal) {
    return;
  }
  lastProfileTrigger = document.activeElement;
  renderProfileDashboard(currentProfile);
  profileModal.hidden = false;
  document.body.classList.add("has-modal");
  profileModal.classList.add("is-open");
  profileModal.querySelector(".profile-modal__close")?.focus();
}

function closeProfileDashboard() {
  if (!profileModal) {
    return;
  }
  profileModal.classList.remove("is-open");
  profileModal.hidden = true;
  document.body.classList.remove("has-modal");
  if (lastProfileTrigger instanceof HTMLElement) {
    lastProfileTrigger.focus();
  }
}

if (profileCard) {
  profileCard.addEventListener("click", openProfileDashboard);
  profileCard.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openProfileDashboard();
    }
  });
}

profileCloseButtons.forEach((button) => {
  button.addEventListener("click", closeProfileDashboard);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && profileModal && !profileModal.hidden) {
    closeProfileDashboard();
  }
});

if (emptyState) {
  emptyState.querySelectorAll("[data-fill]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const fill = btn.dataset.fill || "";
      input.value = fill;
      syncComposerHeight();
      input.focus();
      if (!fill) {
        input.placeholder = "Paste the full job description here and press Send…";
      }
    });
  });
}

stream.addEventListener("click", (e) => {
  const toggleBtn = e.target.closest(".bubble-toggle");
  if (toggleBtn) {
    handleBubbleToggle({ currentTarget: toggleBtn });
    return;
  }

  const submitBtn = e.target.closest(".question-card__submit");
  if (submitBtn) {
    handleQuestionCardSubmit({ currentTarget: submitBtn });
    return;
  }

  const evaluateBtn = e.target.closest(".question-card-list__evaluate");
  if (evaluateBtn) {
    handleBatchEvaluate({ currentTarget: evaluateBtn });
  }
});

stream.addEventListener("input", (e) => {
  const ta = e.target.closest(".question-card__ta");
  if (ta) {
    handleQuestionCardInput(ta);
  }
});

function setStatus(message) {
  statusEl.textContent = message;
}

function setLoading(flag) {
  isLoading = flag;
  composerSurface.classList.toggle("is-loading", flag);
  setStatus(flag ? "Interview Guide is thinking…" : "Ready when you are.");
}

function formatProfile(profile = {}) {
  currentProfile = profile || {};
  renderProfileDashboard(currentProfile);

  if (!profile.total_answers) {
    profileContent.innerHTML = `
      <div class="profile-preview profile-preview--empty">
        <p class="profile-preview__empty">Run an evaluation to see insights.</p>
        <span class="profile-preview__hint">Click to open the progress dashboard.</span>
      </div>
    `;
    return;
  }

  const avgScore = Number(profile.avg_score || 0);
  const strongest = profile.strength_skills?.[0] || "Building";
  const focus = profile.weakness_skills?.[0] || "None yet";
  profileContent.innerHTML = `
    <div class="profile-preview">
      <div class="profile-preview__stats">
        <div class="profile-preview__stat">
          <span class="profile-preview__label">Answers</span>
          <strong class="profile-preview__value">${profile.total_answers}</strong>
        </div>
        <div class="profile-preview__stat">
          <span class="profile-preview__label">Avg score</span>
          <strong class="profile-preview__value">${avgScore.toFixed(1)}/5</strong>
        </div>
      </div>
      <div class="profile-preview__bands">
        <div class="profile-preview__band">
          <span class="profile-preview__band-label">Best at</span>
          <strong>${escapeHtml(formatSkillName(strongest))}</strong>
        </div>
        <div class="profile-preview__band">
          <span class="profile-preview__band-label">Focus next</span>
          <strong>${escapeHtml(formatSkillName(focus))}</strong>
        </div>
      </div>
      <span class="profile-preview__hint">Click to open your progress dashboard.</span>
    </div>
  `;
}

function formatResources(resources = []) {
  if (!resources.length) {
    resourcesContent.textContent = "Ask for resources to populate this panel.";
    return;
  }
  resourcesContent.innerHTML = "";
  resources.slice(0, 5).forEach((res) => {
    const block = document.createElement("div");
    block.className = "resource-pill";
    const anchor = document.createElement("a");
    anchor.href = res.url || "#";
    anchor.textContent = res.title || "Resource";
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer";
    block.appendChild(anchor);
    resourcesContent.appendChild(block);
  });
}

async function loadUserList(selectedId) {
  if (!userDropdown) {
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/users`);
    if (!response.ok) {
      throw new Error("Failed to load users");
    }
    const payload = await response.json();
    userDropdown.innerHTML = '<option value="">Select saved user</option>';
    (payload.users || []).forEach((id) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = id;
      if (id === selectedId) {
        option.selected = true;
      }
      userDropdown.appendChild(option);
    });
  } catch (error) {
    console.error(error);
  }
}

async function fetchUserOverview(userId) {
  if (!userId) {
    return;
  }
  try {
    const response = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}`);
    if (!response.ok) {
      throw new Error("Failed to load user overview");
    }
    const payload = await response.json();
    formatProfile(payload.profile || {});
    formatResources(payload.resources || []);
  } catch (error) {
    console.error(error);
  }
}

function createBubble(text, variant = "agent") {
  const bubble = document.createElement("div");
  bubble.className = `bubble bubble--${variant}`;
  bubble.innerHTML = text;
  return bubble;
}

function createActionButton(label, payload) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "action-chip";
  btn.textContent = label;
  btn.addEventListener("click", () => {
    input.value = payload;
    input.focus();
  });
  return btn;
}

function isCollapsibleMessage(text) {
  const value = String(text || "");
  const lines = value.split("\n").length;
  return value.length >= COLLAPSIBLE_MESSAGE_MIN_LENGTH || lines >= COLLAPSIBLE_MESSAGE_MIN_LINES;
}

function renderUserBubbleContent(bubble, text) {
  bubble.innerHTML = "";
  const value = String(text || "");

  if (!isCollapsibleMessage(value)) {
    bubble.textContent = value;
    return;
  }

  bubble.classList.add("bubble--collapsible");

  const content = document.createElement("div");
  content.className = "bubble__content bubble__content--collapsed";
  content.textContent = value;

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "bubble-toggle";
  toggle.setAttribute("aria-expanded", "false");
  toggle.textContent = "Show more";

  bubble.append(content, toggle);
}

function handleBubbleToggle(event) {
  const btn = event.currentTarget;
  const bubble = btn.closest(".bubble--collapsible");
  const content = bubble?.querySelector(".bubble__content");
  if (!bubble || !content) {
    return;
  }

  const collapsed = content.classList.toggle("bubble__content--collapsed");
  btn.setAttribute("aria-expanded", String(!collapsed));
  btn.textContent = collapsed ? "Show more" : "Show less";
}

function generateFollowUps(state) {
  const actions = document.createElement("div");
  actions.className = "action-list";

  const base = [];
  if (state.questions?.length) {
    base.push("Evaluate these answers next...");
  }
  if (state.profile && state.profile.total_answers) {
    base.push("Recommend resources based on my weaknesses.");
  }
  if (!base.length) {
    base.push(
      "Generate interview questions for cloud architecture.",
      "Ingest this JD: Senior backend engineer with Go experience."
    );
  }

  base.forEach((text) => actions.appendChild(createActionButton(text, text)));
  return actions;
}

function updateQuestionBatchState(wrapper) {
  if (!wrapper) return;

  const cards = Array.from(wrapper.querySelectorAll(".question-card"));
  const total = cards.length;
  const submitted = cards.filter((card) => card.dataset.submitted === "true").length;
  const evaluated = cards.filter((card) => card.dataset.evaluated === "true").length;
  const ready = cards.filter(
    (card) => card.dataset.submitted === "true" && card.dataset.evaluated !== "true"
  ).length;
  const unanswered = Math.max(0, total - submitted);

  const summary = wrapper.querySelector(".question-card-list__summary");
  if (summary) {
    summary.textContent = total
      ? `${evaluated}/${total} evaluated · ${ready} ready · ${unanswered} unanswered`
      : "No questions available.";
  }

  const btn = wrapper.querySelector(".question-card-list__evaluate");
  if (btn) {
    btn.disabled = ready === 0 || isLoading;
    btn.textContent = ready
      ? `Evaluate submitted answers (${ready})`
      : evaluated === total && total
        ? "All submitted answers evaluated"
        : "Evaluate submitted answers";
  }
}

function buildQuestionCards(questions, sessionId, userId, topic = "") {
  const wrapper = document.createElement("div");
  wrapper.className = "question-card-list";
  wrapper.dataset.session = sessionId || "";
  wrapper.dataset.user = userId;
  wrapper.dataset.topic = topic || "";

  questions.forEach((q, idx) => {
    const text = q.text || String(q);
    const type = QUESTION_TYPES[idx % QUESTION_TYPES.length];
    const card = document.createElement("div");
    card.className = "question-card";
    card.dataset.idx = String(idx + 1);
    card.dataset.question = text;
    card.dataset.submitted = "false";
    card.dataset.evaluated = "false";

    const taId = `qta-${Date.now()}-${idx}`;

    const head = document.createElement("div");
    head.className = "question-card__head";

    const headMain = document.createElement("div");
    headMain.className = "question-card__head-main";

    const num = document.createElement("span");
    num.className = "question-card__num";
    num.textContent = `Q${idx + 1}`;

    const typeEl = document.createElement("span");
    typeEl.className = "question-card__type";
    typeEl.textContent = type;

    const meta = document.createElement("span");
    meta.className = "question-card__meta";
    meta.textContent = "Not submitted";

    headMain.append(num, typeEl);
    head.append(headMain, meta);

    const textEl = document.createElement("div");
    textEl.className = "question-card__text";
    textEl.textContent = text;

    const answer = document.createElement("div");
    answer.className = "question-card__answer";

    const textarea = document.createElement("textarea");
    textarea.id = taId;
    textarea.className = "question-card__ta";
    textarea.placeholder = "Type your answer here…";
    textarea.rows = 3;

    const button = document.createElement("button");
    button.className = "question-card__submit";
    button.type = "button";
    button.dataset.ta = taId;
    button.dataset.qtext = text;
    button.dataset.session = sessionId || "";
    button.dataset.user = userId;
    button.textContent = "Submit answer";

    answer.append(textarea, button);
    card.append(head, textEl, answer);
    wrapper.appendChild(card);
  });

  const footer = document.createElement("div");
  footer.className = "question-card-list__footer";

  const summary = document.createElement("div");
  summary.className = "question-card-list__summary";

  const evaluateBtn = document.createElement("button");
  evaluateBtn.className = "question-card-list__evaluate";
  evaluateBtn.type = "button";

  footer.append(summary, evaluateBtn);
  wrapper.appendChild(footer);
  updateQuestionBatchState(wrapper);

  return wrapper.outerHTML;
}

function handleQuestionCardInput(textarea) {
  const card = textarea.closest(".question-card");
  if (!card || card.dataset.evaluated === "true") {
    return;
  }

  const wrapper = textarea.closest(".question-card-list");
  const btn = card.querySelector(".question-card__submit");
  const meta = card.querySelector(".question-card__meta");
  const current = textarea.value.trim();
  const saved = card.dataset.answer || "";

  if (!current) {
    card.dataset.submitted = "false";
    if (btn) btn.textContent = "Submit answer";
    if (meta) meta.textContent = "Not submitted";
  } else if (current !== saved) {
    card.dataset.submitted = "false";
    if (btn) btn.textContent = "Submit answer";
    if (meta) meta.textContent = saved ? "Edited since save" : "Unsaved draft";
  } else if (card.dataset.submitted === "true") {
    if (btn) btn.textContent = "Answer saved";
    if (meta) meta.textContent = "Answer saved";
  }

  updateQuestionBatchState(wrapper);
}

function handleQuestionCardSubmit(event) {
  const btn = event.currentTarget;
  const taId = btn.dataset.ta;
  const ta = document.getElementById(taId);
  const answer = ta?.value?.trim();
  const card = btn.closest(".question-card");
  const wrapper = btn.closest(".question-card-list");
  const meta = card?.querySelector(".question-card__meta");

  if (!answer) {
    ta?.focus();
    return;
  }

  if (card) {
    card.dataset.answer = answer;
    card.dataset.submitted = "true";
  }
  btn.textContent = "Answer saved";
  if (meta) {
    meta.textContent = "Answer saved";
  }
  updateQuestionBatchState(wrapper);
}

async function handleBatchEvaluate(event) {
  const btn = event.currentTarget;
  const wrapper = btn.closest(".question-card-list");
  if (!wrapper || isLoading) return;

  const cards = Array.from(wrapper.querySelectorAll(".question-card")).filter(
    (card) => card.dataset.submitted === "true" && card.dataset.evaluated !== "true"
  );

  if (!cards.length) {
    const firstOpen = wrapper.querySelector(".question-card__ta:not(:disabled)");
    firstOpen?.focus();
    return;
  }

  const answers = cards.map((card) => card.dataset.answer || "");
  const questionTexts = cards.map((card) => card.dataset.question || "");
  const questionIndices = cards.map((card) => Number(card.dataset.idx));
  const questionButtons = cards.map((card) => card.querySelector(".question-card__submit"));
  const questionTextareas = cards.map((card) => card.querySelector(".question-card__ta"));
  const questionMeta = cards.map((card) => card.querySelector(".question-card__meta"));
  const query = `Evaluate submitted answers (${answers.length})`;
  const userId = wrapper.dataset.user || (userInput.value || "anon").trim() || "anon";
  const rawSessionId = wrapper.dataset.session || getLastState()?.session_id;
  const sessionId = rawSessionId ? Number(rawSessionId) : undefined;
  const topic = wrapper.dataset.topic || "";

  btn.disabled = true;
  btn.textContent = "Evaluating…";
  questionTextareas.forEach((ta) => {
    if (ta) ta.disabled = true;
  });
  questionButtons.forEach((button) => {
    if (button) {
      button.disabled = true;
      button.textContent = "Evaluating…";
    }
  });
  questionMeta.forEach((meta) => {
    if (meta) meta.textContent = "Evaluating";
  });

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/agent/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        user_id: userId,
        session_id: sessionId,
        intent: "EVALUATE_ANSWERS",
        slots: {
          answers,
          question_indices: questionIndices,
          question_texts: questionTexts,
          topic,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const payload = await response.json();
    if (payload.state?.session_id) {
      wrapper.dataset.session = String(payload.state.session_id);
    }

    cards.forEach((card) => {
      card.dataset.evaluated = "true";
      card.dataset.submitted = "true";
      const ta = card.querySelector(".question-card__ta");
      const submitBtn = card.querySelector(".question-card__submit");
      const meta = card.querySelector(".question-card__meta");
      if (ta) ta.disabled = true;
      if (submitBtn) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Evaluated";
      }
      if (meta) meta.textContent = "Evaluated";
    });

    turns.push({ query, state: payload.state });
    renderTurn(turns[turns.length - 1]);
    formatProfile(payload.state.profile);
    formatResources(payload.state.resources || []);
    setStatus("Done.");
    loadUserList(userId);
    fetchUserOverview(userId);
  } catch (error) {
    questionTextareas.forEach((ta) => {
      if (ta) ta.disabled = false;
    });
    questionButtons.forEach((button) => {
      if (button) {
        button.disabled = false;
        button.textContent = "Answer saved";
      }
    });
    questionMeta.forEach((meta) => {
      if (meta) meta.textContent = "Answer saved";
    });
    setStatus("Please try again.");
  } finally {
    setLoading(false);
    updateQuestionBatchState(wrapper);
  }
}

function renderTurn(turn) {
  const node = turnTemplate.content.firstElementChild.cloneNode(true);
  renderUserBubbleContent(node.querySelector(".bubble--user"), turn.query);

  const agentBubble = node.querySelector(".bubble--agent");
  const fragments = [];
  if (turn.state.result) {
    fragments.push(`<p class="bubble-text">${turn.state.result}</p>`);
  }
  if (turn.state.questions?.length) {
    const sessionId = turn.state.session_id;
    const userId = (userInput.value || "anon").trim();
    const topic = turn.state.slots?.topic || turn.state.questions[0]?.topic || "";
    fragments.push(buildQuestionCards(turn.state.questions, sessionId, userId, topic));
  }
  if (turn.state.scores?.length) {
    const scoreCards = turn.state.scores
      .map((s) => {
        const pct = Math.round((s.score / 5) * 100);
        const level = pct >= 80 ? "high" : pct >= 50 ? "mid" : "low";
        const label = pct >= 80 ? "Strong" : pct >= 50 ? "Good" : "Needs work";
        const qText = s.question_text
          ? `<div class="score-card__question">${s.question_text}</div>`
          : "";
        return `
          <div class="score-card score-card--${level}">
            ${qText}
            <div class="score-card__top">
              <span class="score-card__badge">${s.score}/5</span>
              <span class="score-card__label">${label}</span>
            </div>
            <div class="score-card__feedback">${s.feedback || ""}</div>
          </div>
        `;
      })
      .join("");
    fragments.push(`<div class="score-card-list">${scoreCards}</div>`);
  }
  if (turn.state.resources?.length) {
    const topResources = turn.state.resources.slice(0, 4);
    fragments.push(
      `<div><strong>Resources</strong><ul class="inline-resource-list">${topResources
        .map(
          (r) =>
            `<li><a href="${r.url || "#"}" target="_blank" rel="noopener noreferrer">${
              r.title || "Resource"
            }</a></li>`
        )
        .join("")}</ul></div>`
    );
  } else if (turn.state.skip_recommendations) {
    fragments.push("<div><strong>Resources</strong><p>No weaknesses detected. No resources recommended.</p></div>");
  }
  if (turn.state.profile && turn.state.profile.total_answers) {
    const prof = turn.state.profile;
    const profileBlock = document.createElement("div");
    profileBlock.innerHTML = `
      <strong>Profile</strong>
      <p>Total answers: ${prof.total_answers} · Avg score: ${prof.avg_score ?? 0}</p>
      ${
        prof.strength_skills?.length
          ? `<p>Strengths: ${prof.strength_skills.slice(0, 3).join(", ")}</p>`
          : ""
      }
      ${
        prof.weakness_skills?.length
          ? `<p>Focus: ${prof.weakness_skills.slice(0, 3).join(", ")}</p>`
          : ""
      }
    `;
    fragments.push(profileBlock.outerHTML);
  }

  agentBubble.innerHTML = fragments.join("");

  if (turn.state.examples?.length) {
    const actions = document.createElement("div");
    actions.className = "action-list";
    turn.state.examples.forEach((text) => actions.appendChild(createActionButton(text, text)));
    agentBubble.appendChild(actions);
  } else {
    agentBubble.appendChild(generateFollowUps(turn.state));
  }

  stream.appendChild(node);
  stream.scrollTo({ top: stream.scrollHeight, behavior: "smooth" });
}

async function handleSubmit(event) {
  event.preventDefault();
  if (isLoading) return;

  const query = input.value.trim();
  if (!query) {
    input.focus();
    return;
  }

  const userId = (userInput.value || "anon").trim() || "anon";
  const lastState = getLastState();

  if (emptyState) {
    emptyState.style.display = "none";
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE}/agent/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        user_id: userId,
        session_id: lastState?.session_id,
      }),
    });

    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }

    const payload = await response.json();
    turns.push({ query, state: payload.state });
    renderTurn(turns[turns.length - 1]);
    formatProfile(payload.state.profile);
    formatResources(payload.state.resources || []);
    setStatus("Done.");
    input.value = "";
    syncComposerHeight();
    loadUserList(userId);
    fetchUserOverview(userId);
  } catch (error) {
    const errorBubble = document.createElement("div");
    errorBubble.className = "error-banner";
    errorBubble.textContent = `Something went wrong. ${error.message}`;
    stream.appendChild(errorBubble);
    setStatus("Please try again.");
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", handleSubmit);

input.addEventListener("input", () => {
  syncComposerHeight();
});

if (userDropdown) {
  userDropdown.addEventListener("change", (event) => {
    const selected = event.target.value;
    if (selected) {
      userInput.value = selected;
      fetchUserOverview(selected);
    }
  });
}

if (userInput) {
  userInput.addEventListener("input", () => {
    if (userDropdown) {
      userDropdown.value = "";
    }
  });
}

const initialUserId = (userInput.value || "").trim();
syncComposerHeight();
loadUserList(initialUserId);
if (initialUserId) {
  fetchUserOverview(initialUserId);
}
