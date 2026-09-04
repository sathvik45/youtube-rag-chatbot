"use strict";

const STORAGE_TOKEN = "rag-eval-access-token";
const STORAGE_EMAIL = "rag-eval-email";
const POLL_DELAY_MS = 3000;

const state = {
  token: sessionStorage.getItem(STORAGE_TOKEN) || "",
  email: sessionStorage.getItem(STORAGE_EMAIL) || "",
  sources: [],
  threads: [],
  selectedThreadId: null,
  selectedThreadTitle: "",
  sourcePollTimer: null,
};

const elements = {
  authView: document.querySelector("#auth-view"),
  workspace: document.querySelector("#workspace"),
  notice: document.querySelector("#notice"),
  loginForm: document.querySelector("#login-form"),
  registerForm: document.querySelector("#register-form"),
  logoutButton: document.querySelector("#logout-button"),
  currentEmail: document.querySelector("#current-email"),
  sourceForm: document.querySelector("#source-form"),
  sourceList: document.querySelector("#source-list"),
  refreshSources: document.querySelector("#refresh-sources"),
  threadList: document.querySelector("#thread-list"),
  refreshThreads: document.querySelector("#refresh-threads"),
  chatTitle: document.querySelector("#chat-title"),
  chatSubtitle: document.querySelector("#chat-subtitle"),
  chatStatus: document.querySelector("#chat-status"),
  messageList: document.querySelector("#message-list"),
  messageForm: document.querySelector("#message-form"),
  messageContent: document.querySelector("#message-content"),
  sendButton: document.querySelector("#send-button"),
};

function setNotice(message, kind = "info") {
  elements.notice.textContent = message;
  elements.notice.dataset.kind = kind;
}

function clearNotice() {
  setNotice("");
}

function detailFrom(payload, fallback) {
  if (typeof payload === "object" && payload && "detail" in payload) {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
  }
  return fallback;
}

async function request(
  path,
  { method = "GET", body, authenticated = true, formEncoded = false } = {},
) {
  const headers = new Headers();
  if (authenticated && state.token) {
    headers.set("Authorization", "Bearer " + state.token);
  }

  let requestBody;
  if (body !== undefined) {
    if (formEncoded) {
      headers.set("Content-Type", "application/x-www-form-urlencoded");
      requestBody = body;
    } else {
      headers.set("Content-Type", "application/json");
      requestBody = JSON.stringify(body);
    }
  }

  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: requestBody,
    });
  } catch {
    throw new Error("Could not connect to the local API.");
  }

  const responseText = await response.text();
  let payload = null;
  if (responseText) {
    try {
      payload = JSON.parse(responseText);
    } catch {
      payload = responseText;
    }
  }

  if (!response.ok) {
    if (response.status === 401 && authenticated) {
      leaveWorkspace("Your session expired. Please sign in again.", "error");
    }
    throw new Error(
      detailFrom(payload, "The request failed with status " + response.status + "."),
    );
  }

  return payload;
}

function setFormBusy(form, busy) {
  for (const control of form.querySelectorAll("button, input, textarea")) {
    control.disabled = busy;
  }
}

function formatDate(value) {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString();
}

function safeCitationUrl(value) {
  if (typeof value !== "string") {
    return null;
  }
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    const isYouTube =
      host === "youtu.be" ||
      host === "youtube.com" ||
      host.endsWith(".youtube.com");
    return url.protocol === "https:" && isYouTube ? url.href : null;
  } catch {
    return null;
  }
}

function statusText(source) {
  const progress = source.progress;
  if (!progress) {
    return source.status || "unknown";
  }
  const ready = progress.video_counts ? progress.video_counts.ready || 0 : 0;
  return (
    progress.status +
    " · " +
    progress.completed_videos +
    "/" +
    progress.total_videos +
    " processed · " +
    ready +
    " ready"
  );
}

function sourceCanCreateThread(source) {
  if (!source.progress) {
    return false;
  }
  const status = source.progress.status;
  const ready = source.progress.video_counts
    ? source.progress.video_counts.ready || 0
    : 0;
  return (status === "ready" || status === "partial") && ready > 0;
}

function hasProcessingSource() {
  return state.sources.some((source) => {
    const status = source.progress ? source.progress.status : source.status;
    return status === "pending" || status === "processing";
  });
}

function scheduleSourcePoll() {
  window.clearTimeout(state.sourcePollTimer);
  state.sourcePollTimer = null;

  if (!state.token || !hasProcessingSource()) {
    return;
  }

  state.sourcePollTimer = window.setTimeout(async () => {
    try {
      await loadSources({ quiet: true });
    } catch (error) {
      setNotice(error.message, "error");
    }
  }, POLL_DELAY_MS);
}

function clearList(list, message) {
  list.replaceChildren();
  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = message;
  list.append(empty);
}

function renderSources() {
  if (!state.sources.length) {
    clearList(
      elements.sourceList,
      "Add a YouTube video or playlist to start a source-scoped chat.",
    );
    return;
  }

  elements.sourceList.replaceChildren();
  for (const source of state.sources) {
    const card = document.createElement("article");
    card.className = "source-card";

    const title = document.createElement("p");
    title.className = "source-title";
    title.textContent = source.submitted_value;

    const meta = document.createElement("p");
    meta.className = "source-meta";
    meta.textContent = statusText(source);

    const create = document.createElement("div");
    create.className = "source-create";

    const chatTitle = document.createElement("input");
    chatTitle.type = "text";
    chatTitle.maxLength = 255;
    chatTitle.placeholder = "Chat title";
    chatTitle.value = "Video chat";
    chatTitle.setAttribute("aria-label", "Title for a new chat");

    const createButton = document.createElement("button");
    createButton.type = "button";
    createButton.textContent = "Create chat";
    createButton.disabled = !sourceCanCreateThread(source);
    if (createButton.disabled) {
      createButton.title = "Wait for at least one video to finish processing.";
    }
    createButton.addEventListener("click", async () => {
      await createThread(source.id, chatTitle.value);
    });

    create.append(chatTitle, createButton);
    card.append(title, meta, create);
    elements.sourceList.append(card);
  }
}

function renderThreads() {
  if (!state.threads.length) {
    clearList(
      elements.threadList,
      "Create a chat from a ready source to begin asking questions.",
    );
    return;
  }

  elements.threadList.replaceChildren();
  for (const thread of state.threads) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "thread-button";
    if (thread.id === state.selectedThreadId) {
      button.classList.add("is-selected");
    }

    const title = document.createElement("strong");
    title.textContent = thread.title;
    const updated = document.createElement("span");
    updated.textContent = "Updated " + formatDate(thread.updated_at);
    button.append(title, updated);
    button.addEventListener("click", async () => {
      await selectThread(thread.id, thread.title);
    });
    elements.threadList.append(button);
  }
}

function renderMessages(messages) {
  if (!messages.length) {
    clearList(
      elements.messageList,
      "This chat is empty. Ask a specific question about its uploaded videos.",
    );
    return;
  }

  elements.messageList.replaceChildren();
  for (const message of messages) {
    const article = document.createElement("article");
    article.className =
      "message " + (message.role === "user" ? "user" : "assistant");

    const role = document.createElement("span");
    role.className = "message-role";
    role.textContent = message.role === "user" ? "You" : "Assistant";

    const content = document.createElement("div");
    content.className = "message-content";
    content.textContent = message.content;
    article.append(role, content);

    if (message.role === "assistant") {
      const outcome = document.createElement("p");
      outcome.className = "message-status";
      if (message.status === "refused_no_context") {
        outcome.textContent =
          "No matching evidence was found in this thread's uploaded videos.";
      } else if (message.status === "temporary_error") {
        outcome.textContent =
          "This answer could not be completed. You can try again.";
      } else {
        outcome.textContent = message.grounded
          ? "Grounded answer"
          : "Answer status: " + message.status;
      }
      article.append(outcome);

      if (Array.isArray(message.citations) && message.citations.length) {
        const citations = document.createElement("div");
        citations.className = "citation-list";
        for (const citation of message.citations) {
          const citationItem = document.createElement("div");
          citationItem.className = "citation";

          const label =
            "Citation " +
            citation.position +
            " (" +
            Math.floor(citation.start_ms / 1000) +
            "s)";
          const citationUrl = safeCitationUrl(citation.youtube_url);
          if (citationUrl) {
            const link = document.createElement("a");
            link.href = citationUrl;
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = "Open " + label.toLowerCase();
            citationItem.append(link);
          } else {
            const text = document.createElement("span");
            text.textContent = label;
            citationItem.append(text);
          }

          if (citation.quote_text) {
            const quote = document.createElement("blockquote");
            quote.textContent = citation.quote_text;
            citationItem.append(quote);
          }
          citations.append(citationItem);
        }
        article.append(citations);
      }
    }

    elements.messageList.append(article);
  }

  elements.messageList.scrollTop = elements.messageList.scrollHeight;
}

function resetChat() {
  state.selectedThreadId = null;
  state.selectedThreadTitle = "";
  elements.chatTitle.textContent = "Choose a chat";
  elements.chatSubtitle.textContent =
    "Your messages and answers stay inside the selected thread.";
  elements.chatStatus.classList.add("hidden");
  elements.messageContent.disabled = true;
  elements.sendButton.disabled = true;
  clearList(
    elements.messageList,
    "Create or select a chat to view its saved conversation.",
  );
}

function setChatLoading(loading) {
  elements.messageContent.disabled = loading || !state.selectedThreadId;
  elements.sendButton.disabled = loading || !state.selectedThreadId;
  elements.chatStatus.classList.toggle("hidden", !loading);
  elements.chatStatus.textContent = loading ? "Working…" : "";
}

async function loadSources({ quiet = false } = {}) {
  const listing = await request("/sources?limit=20");
  const sources = await Promise.all(
    listing.items.map(async (source) => {
      try {
        const progress = await request("/sources/" + source.id);
        return { ...source, progress };
      } catch (error) {
        return { ...source, progress: null, progressError: error.message };
      }
    }),
  );
  state.sources = sources;
  renderSources();
  scheduleSourcePoll();
  if (!quiet) {
    clearNotice();
  }
}

async function loadThreads() {
  const listing = await request("/threads?limit=50");
  state.threads = listing.items;
  const selectedStillExists = state.threads.some(
    (thread) => thread.id === state.selectedThreadId,
  );
  if (state.selectedThreadId && !selectedStillExists) {
    resetChat();
  }
  renderThreads();
}

async function loadHistory() {
  if (!state.selectedThreadId) {
    return;
  }
  const history = await request(
    "/threads/" + state.selectedThreadId + "/messages?limit=100",
  );
  renderMessages(history.items);
}

async function createThread(sourceId, title) {
  const trimmedTitle = title.trim() || "Video chat";
  try {
    const thread = await request("/threads", {
      method: "POST",
      body: { source_id: sourceId, title: trimmedTitle },
    });
    setNotice("Chat created. You can ask a question now.", "success");
    await loadThreads();
    await selectThread(thread.id, thread.title);
  } catch (error) {
    setNotice(error.message, "error");
  }
}

async function selectThread(threadId, title) {
  state.selectedThreadId = threadId;
  state.selectedThreadTitle = title;
  elements.chatTitle.textContent = title;
  elements.chatSubtitle.textContent =
    "Answers are limited to this thread's uploaded source.";
  elements.messageContent.disabled = false;
  elements.sendButton.disabled = false;
  renderThreads();

  try {
    setChatLoading(true);
    await loadHistory();
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setChatLoading(false);
  }
}

function leaveWorkspace(message = "", kind = "info") {
  state.token = "";
  state.email = "";
  state.sources = [];
  state.threads = [];
  window.clearTimeout(state.sourcePollTimer);
  state.sourcePollTimer = null;
  sessionStorage.removeItem(STORAGE_TOKEN);
  sessionStorage.removeItem(STORAGE_EMAIL);
  elements.workspace.classList.add("hidden");
  elements.authView.classList.remove("hidden");
  resetChat();
  if (message) {
    setNotice(message, kind);
  } else {
    clearNotice();
  }
}

async function enterWorkspace() {
  if (!state.token) {
    return;
  }
  elements.currentEmail.textContent = state.email;
  elements.authView.classList.add("hidden");
  elements.workspace.classList.remove("hidden");
  resetChat();
  try {
    await Promise.all([loadSources(), loadThreads()]);
    setNotice("Signed in. Add a source or select an existing chat.", "success");
  } catch (error) {
    if (state.token) {
      setNotice(error.message, "error");
    }
  }
}

async function signIn(email, password) {
  const form = new URLSearchParams();
  form.set("username", email.trim());
  form.set("password", password);
  const token = await request("/auth/token", {
    method: "POST",
    body: form,
    authenticated: false,
    formEncoded: true,
  });
  state.token = token.access_token;
  state.email = email.trim();
  sessionStorage.setItem(STORAGE_TOKEN, state.token);
  sessionStorage.setItem(STORAGE_EMAIL, state.email);
  await enterWorkspace();
}

elements.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(elements.loginForm);
  setFormBusy(elements.loginForm, true);
  try {
    await signIn(
      String(data.get("email") || ""),
      String(data.get("password") || ""),
    );
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setFormBusy(elements.loginForm, false);
  }
});

elements.registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(elements.registerForm);
  const email = String(data.get("email") || "").trim();
  const password = String(data.get("password") || "");
  setFormBusy(elements.registerForm, true);
  try {
    await request("/auth/register", {
      method: "POST",
      body: { email, password },
      authenticated: false,
    });
    await signIn(email, password);
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setFormBusy(elements.registerForm, false);
  }
});

elements.logoutButton.addEventListener("click", () => {
  leaveWorkspace("You have been logged out.", "success");
});

elements.sourceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(elements.sourceForm);
  const youtubeUrl = String(data.get("youtube_url") || "").trim();
  setFormBusy(elements.sourceForm, true);
  try {
    const submission = await request("/sources", {
      method: "POST",
      body: { youtube_url: youtubeUrl },
    });
    elements.sourceForm.reset();
    setNotice(
      "Source added. " +
        submission.video_ids.length +
        " video(s) are ready or queued for processing.",
      "success",
    );
    await loadSources({ quiet: true });
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setFormBusy(elements.sourceForm, false);
  }
});

elements.refreshSources.addEventListener("click", async () => {
  try {
    await loadSources();
  } catch (error) {
    setNotice(error.message, "error");
  }
});

elements.refreshThreads.addEventListener("click", async () => {
  try {
    await loadThreads();
    setNotice("Chats refreshed.", "success");
  } catch (error) {
    setNotice(error.message, "error");
  }
});

elements.messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.selectedThreadId) {
    return;
  }
  const content = elements.messageContent.value.trim();
  if (!content) {
    return;
  }

  try {
    setChatLoading(true);
    await request("/threads/" + state.selectedThreadId + "/messages", {
      method: "POST",
      body: { content },
    });
    elements.messageContent.value = "";
    await Promise.all([loadHistory(), loadThreads()]);
  } catch (error) {
    setNotice(error.message, "error");
  } finally {
    setChatLoading(false);
  }
});

if (state.token) {
  enterWorkspace();
} else {
  resetChat();
}
