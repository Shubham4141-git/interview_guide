const API_BASE = (window.API_BASE_URL || "http://localhost:8000") + "/api";

const form = document.querySelector("[data-form]");
const input = document.querySelector("[data-input]");
const userInput = document.querySelector("[data-user]");
const stream = document.getElementById("chat-stream");
const statusEl = document.getElementById("status-message");
const turnTemplate = document.getElementById("chat-turn-template");
const composerSurface = document.querySelector("[data-composer]");
const profileContent = document.querySelector("[data-profile]");
const resourcesContent = document.querySelector("[data-resources]");
const userDropdown = document.querySelector("[data-user-dropdown]");

let isLoading = false;
const turns = [];

const getLastState = () => {
  if (!turns.length) return undefined;
  return turns[turns.length - 1].state;
};

function setStatus(message) {
  statusEl.textContent = message;
}

function setLoading(flag) {
  isLoading = flag;
  composerSurface.classList.toggle("is-loading", flag);
  setStatus(flag ? "Interview Guide is thinking…" : "Ready when you are.");
}

function formatProfile(profile = {}) {
  if (!profile.total_answers) {
    profileContent.textContent = "Run an evaluation to see insights.";
    return;
  }
  profileContent.innerHTML = `
    <p><strong>Total answers:</strong> ${profile.total_answers}</p>
    <p><strong>Average score:</strong> ${profile.avg_score ?? 0}</p>
    ${
      profile.strength_skills?.length
        ? `<p><strong>Strengths:</strong> ${profile.strength_skills.join(", ")}</p>`
        : ""
    }
    ${
      profile.weakness_skills?.length
        ? `<p><strong>Focus areas:</strong> ${profile.weakness_skills.join(", ")}</p>`
        : ""
    }
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

function renderTurn(turn) {
  const node = turnTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector(".bubble--user").textContent = turn.query;

  const agentBubble = node.querySelector(".bubble--agent");
  const fragments = [];
  if (turn.state.result) {
    fragments.push(`<p class="bubble-text">${turn.state.result}</p>`);
  }
  if (turn.state.questions?.length) {
    fragments.push(
      `<div><strong>Questions</strong><ul>${turn.state.questions
        .map((q) => `<li>${q.text || q}</li>`)
        .join("")}</ul></div>`
    );
  }
  if (turn.state.skill_groups?.length) {
    const groups = turn.state.skill_groups
      .map((group, idx) => {
        const items = Array.isArray(group) ? group.join(", ") : String(group);
        return `<li><strong>Group ${idx + 1}:</strong> ${items}</li>`;
      })
      .join("");
    fragments.push(`<div><strong>Skill clusters</strong><ul>${groups}</ul></div>`);
  }
  if (turn.state.scores?.length) {
    fragments.push(
      `<div><strong>Scores</strong><ul>${turn.state.scores
        .map((s) => `<li>${s.score}/5 ${s.skill ? `(${s.skill})` : ""} — ${s.feedback}</li>`)
        .join("")}</ul></div>`
    );
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
    input.style.height = "auto";
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
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
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
loadUserList(initialUserId);
if (initialUserId) {
  fetchUserOverview(initialUserId);
}
