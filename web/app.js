/* Mental Health Info Assistant - client.
 * Security: user text is only ever inserted with textContent; model output is rendered as markdown and
 * sanitised with DOMPurify before insertion (fixes the v1 XSS, CHANGELOG B15).
 * Privacy: the conversation lives in sessionStorage (this tab only) and is sent with each request as history.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("composer"), input = $("msg"), sendBtn = $("send"), stopBtn = $("stop");
  const modelSel = $("model"), regionSel = $("region");
  const STORE = "mhrag.conversation.v1";
  const PREFS = "mhrag.prefs.v1";

  let history = [];          // [{role, content}]
  let controller = null;     // AbortController for the in-flight request
  let helplineCache = {};

  const store = {
    get(k, d) { try { const v = sessionStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
    local(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    setLocal(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* ignore */ } },
  };

  // ------------------------------------------------------------------ markdown (sanitised)
  if (window.marked) marked.setOptions({ gfm: true, breaks: true });
  function renderMarkdown(el, text) {
    const html = window.marked ? marked.parse(text) : text.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
    const clean = window.DOMPurify
      ? DOMPurify.sanitize(html, { USE_PROFILES: { html: true }, FORBID_TAGS: ["style", "img", "iframe", "form", "input"], FORBID_ATTR: ["style"] })
      : "";
    el.innerHTML = clean; // eslint-disable-line no-unsanitized/property -- sanitised above
    el.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
    // Turn [1] style citations into links to the source list.
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach((n) => {
      if (!/\[\d{1,2}\]/.test(n.nodeValue) || n.parentElement.closest("a,code,pre")) return;
      const frag = document.createDocumentFragment();
      n.nodeValue.split(/(\[\d{1,2}\])/).forEach((part) => {
        const m = part.match(/^\[(\d{1,2})\]$/);
        if (m) {
          const a = document.createElement("a");
          a.className = "cite"; a.href = "#"; a.dataset.n = m[1]; a.textContent = `[${m[1]}]`;
          a.setAttribute("aria-label", `Source ${m[1]}`);
          frag.appendChild(a);
        } else if (part) frag.appendChild(document.createTextNode(part));
      });
      n.replaceWith(frag);
    });
  }

  // ------------------------------------------------------------------ message DOM
  function hideWelcome() { const w = $("welcome"); if (w) w.remove(); }
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  function addUser(text) {
    hideWelcome();
    const row = document.createElement("div"); row.className = "msg user";
    const b = document.createElement("div"); b.className = "bubble"; b.textContent = text;
    row.appendChild(b); chat.appendChild(row); scrollDown();
  }

  function addAssistant() {
    hideWelcome();
    const row = document.createElement("div"); row.className = "msg bot";
    const b = document.createElement("div"); b.className = "bubble";
    const body = document.createElement("div"); body.className = "body";
    const typing = document.createElement("div"); typing.className = "typing";
    typing.setAttribute("aria-label", "Assistant is typing");
    typing.innerHTML = "<span></span><span></span><span></span>";
    body.appendChild(typing);
    b.appendChild(body); row.appendChild(b); chat.appendChild(row); scrollDown();
    return { row, bubble: b, body };
  }

  function renderSources(bubble, sources) {
    bubble.querySelector("details.sources")?.remove();
    if (!sources || !sources.length) return;
    const d = document.createElement("details"); d.className = "sources";
    const s = document.createElement("summary"); s.textContent = `Sources (${sources.length})`;
    const ol = document.createElement("ol");
    sources.forEach((src) => {
      const li = document.createElement("li"); li.id = `src-${bubble.dataset.turn}-${src.n}`;
      const label = src.section && src.section !== src.title ? `${src.title} — ${src.section}` : src.title;
      if (src.url) {
        const a = document.createElement("a"); a.href = src.url; a.target = "_blank"; a.rel = "noopener noreferrer";
        a.textContent = label; li.appendChild(a);
      } else {
        li.appendChild(document.createTextNode(label));
      }
      const badge = document.createElement("span"); badge.className = "badge";
      const kinds = { mind_web: "Mind", mind_booklet: `Mind booklet${src.year ? " " + src.year : ""}`, faq: "FAQ dataset",
        kb_fact: "KB dataset", web_article: "web article", counselling: "counselling Q&A" };
      badge.textContent = kinds[src.source_type] || src.source_type;
      li.appendChild(badge);
      if (src.source_type === "mind_booklet") {
        const note = document.createElement("span"); note.className = "badge"; note.textContent = "link: Mind info hub";
        li.appendChild(note);
      }
      ol.appendChild(li);
    });
    d.append(s, ol); bubble.appendChild(d);
  }

  function renderMeta(bubble, done) {
    const m = document.createElement("div"); m.className = "meta";
    const t = done.timing || {};
    const parts = [];
    if (done.model) parts.push(done.model);
    if (t.ttft_ms) parts.push(`first token ${(t.ttft_ms / 1000).toFixed(1)}s`);
    if (t.total_ms) parts.push(`total ${(t.total_ms / 1000).toFixed(1)}s`);
    parts.forEach((p) => { const s = document.createElement("span"); s.textContent = p; m.appendChild(s); });
    bubble.appendChild(m);
  }

  function addContinue(bubble) {
    const b = document.createElement("button"); b.type = "button"; b.className = "btn small retry";
    b.textContent = "Continue";
    b.setAttribute("aria-label", "Continue this answer");
    b.addEventListener("click", () => {
      if (controller) return;
      b.remove();
      const text = "Please continue your previous answer.";
      addUser(text); ask(text);
    });
    bubble.appendChild(b);
  }

  function showError(view, message, retry) {
    view.bubble.classList.add("error");
    view.body.textContent = message;
    const b = document.createElement("button"); b.type = "button"; b.className = "btn small retry"; b.textContent = "Retry";
    b.addEventListener("click", () => { view.row.remove(); retry(); });
    view.bubble.appendChild(b);
  }

  // ------------------------------------------------------------------ crisis banner
  function helplineItems(list) {
    const ul = $("crisis-list"); ul.textContent = "";
    list.forEach((reg, i) => {
      if (i > 1 && reg.region !== "INTL") return; // selected region + next region + generic fallback
      reg.services.forEach((svc) => {
        const li = document.createElement("li");
        const strong = document.createElement("strong"); strong.textContent = `${reg.name}: `;
        li.appendChild(strong);
        li.appendChild(document.createTextNode(svc.name));
        if (svc.phone) {
          li.appendChild(document.createTextNode(" — "));
          const a = document.createElement("a"); a.href = `tel:${svc.phone.replace(/[^0-9+]/g, "")}`; a.textContent = svc.phone;
          li.appendChild(a);
          if (svc.alt_phone) li.appendChild(document.createTextNode(` / ${svc.alt_phone}`));
        }
        ul.appendChild(li);
      });
    });
  }
  async function showCrisis(label, helplines) {
    const titles = {
      crisis: "You don't have to face this alone",
      harmful_request: "Please reach out for support",
      third_party: "Worried about someone's safety?",
      elevated: "Support is available",
      manual: "Free, confidential helplines",
    };
    $("crisis-title").textContent = titles[label] || titles.crisis;
    if (!helplines || !helplines.length) helplines = await loadHelplines(regionSel.value);
    helplineItems(helplines);
    $("crisis").hidden = false;
    if (label !== "manual") $("crisis").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
  async function loadHelplines(region) {
    if (helplineCache[region]) return helplineCache[region];
    const r = await fetch(`/api/helplines?region=${encodeURIComponent(region || "")}`);
    const j = await r.json();
    helplineCache[region] = j.helplines;
    return j.helplines;
  }

  // ------------------------------------------------------------------ SSE over fetch
  async function* sse(response) {
    const reader = response.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i); buf = buf.slice(i + 2);
        let event = "message", data = "";
        block.split("\n").forEach((ln) => {
          if (ln.startsWith("event:")) event = ln.slice(6).trim();
          else if (ln.startsWith("data:")) data += ln.slice(5).trim();
        });
        if (data) yield { event, data: JSON.parse(data) };
      }
    }
  }

  function setBusy(busy) {
    sendBtn.disabled = busy; stopBtn.hidden = !busy; input.setAttribute("aria-busy", String(busy));
  }

  let turn = 0;
  async function ask(text) {
    const view = addAssistant();
    view.bubble.dataset.turn = String(++turn);
    const sentHistory = history.slice(-12);
    controller = new AbortController();
    setBusy(true);
    let answer = "", sources = [], rendered = false, pending = false;
    const paint = () => { pending = false; renderMarkdown(view.body, answer); scrollDown(); };
    const schedule = () => { if (!pending) { pending = true; requestAnimationFrame(paint); } };
    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: sentHistory, model: modelSel.value || null, region: regionSel.value || null }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        let msg = `Request failed (${resp.status}).`;
        try { const j = await resp.json(); msg = j.error || (j.detail && j.detail[0] && j.detail[0].msg) || msg; } catch { /* not json */ }
        throw new Error(msg);
      }
      for await (const { event, data } of sse(resp)) {
        if (event === "status") { if (!answer) view.body.textContent = data.text; }
        else if (event === "gate" && data.label && data.label !== "none") showCrisis(data.label, data.helplines);
        else if (event === "sources") sources = data.sources || [];
        else if (event === "token") { answer += data.text; rendered = true; schedule(); }
        else if (event === "replace") { answer = data.text; schedule(); }
        else if (event === "append") { answer += data.text; schedule(); }
        else if (event === "error") throw new Error(data.message || "Something went wrong.");
        else if (event === "done") {
          answer = data.answer || answer;
          paint(); renderSources(view.bubble, data.sources || sources); renderMeta(view.bubble, data);
          if (data.truncated) addContinue(view.bubble);
        }
      }
      if (!rendered && !answer) throw new Error("No answer was returned. Please try again.");
      history.push({ role: "user", content: text }, { role: "assistant", content: answer });
      store.set(STORE, history);
    } catch (err) {
      if (err.name === "AbortError") {
        if (answer) { paint(); history.push({ role: "user", content: text }, { role: "assistant", content: answer }); store.set(STORE, history); }
        else view.row.remove();
      } else {
        showError(view, err.message && !/Failed to fetch|NetworkError/i.test(err.message)
          ? err.message : "Couldn't reach the server. Check your connection and try again.", () => ask(text));
      }
    } finally {
      controller = null; setBusy(false); input.focus();
    }
  }

  // ------------------------------------------------------------------ events
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || controller) return;
    input.value = ""; autosize();
    addUser(text);
    ask(text);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 180) + "px"; }
  input.addEventListener("input", autosize);
  stopBtn.addEventListener("click", () => controller && controller.abort());
  chat.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) { input.value = chip.textContent; form.requestSubmit(); return; }
    const cite = e.target.closest("a.cite");
    if (cite) {
      e.preventDefault();
      const bubble = cite.closest(".bubble");
      const det = bubble.querySelector("details.sources");
      if (det) { det.open = true; const li = bubble.querySelector(`#src-${bubble.dataset.turn}-${cite.dataset.n}`); li && li.scrollIntoView({ block: "nearest" }); }
    }
  });
  $("new-chat").addEventListener("click", () => {
    if (controller) controller.abort();
    history = []; store.set(STORE, history); location.reload();
  });
  $("crisis-close").addEventListener("click", () => { $("crisis").hidden = true; });
  $("show-helplines").addEventListener("click", () => showCrisis("manual"));
  regionSel.addEventListener("change", async () => {
    const prefs = store.local(PREFS, {}); prefs.region = regionSel.value; store.setLocal(PREFS, prefs);
    if (!$("crisis").hidden) helplineItems(await loadHelplines(regionSel.value));
  });
  modelSel.addEventListener("change", () => { const prefs = store.local(PREFS, {}); prefs.model = modelSel.value; store.setLocal(PREFS, prefs); });

  // theme
  const themeBtn = $("theme");
  function applyTheme(t) {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    themeBtn.setAttribute("aria-pressed", String(dark));
  }
  themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    const prefs = store.local(PREFS, {}); prefs.theme = next; store.setLocal(PREFS, prefs); applyTheme(next);
  });

  // ------------------------------------------------------------------ init
  async function init() {
    const prefs = store.local(PREFS, {});
    applyTheme(prefs.theme);
    try {
      const [models, hl] = await Promise.all([fetch("/api/models").then((r) => r.json()), fetch("/api/helplines").then((r) => r.json())]);
      // One <optgroup> per model family; models that cannot run here stay visible but disabled, with the reason.
      const groups = new Map();
      models.models.forEach((m) => {
        if (!groups.has(m.group_label)) {
          const g = document.createElement("optgroup"); g.label = m.group_label || "Models";
          groups.set(m.group_label, g); modelSel.appendChild(g);
        }
        const o = document.createElement("option"); o.value = m.key;
        const size = m.params_b ? ` · ${m.params_b < 1 ? Math.round(m.params_b * 1000) + "M" : Math.round(m.params_b * 10) / 10 + "B"}` : "";
        let text = m.label + size;
        if (!m.available) text += ` (unavailable: ${m.status})`;
        else if (m.status && m.status !== "ok") text += ` (${m.status})`;
        o.textContent = text;
        o.title = [m.label, m.license ? `licence: ${m.license}` : "", m.available ? m.status : `unavailable: ${m.status}`].filter(Boolean).join("\n");
        o.disabled = !m.available;
        groups.get(m.group_label).appendChild(o);
      });
      const nAvail = models.models.filter((m) => m.available).length;
      modelSel.setAttribute("aria-description", `${nAvail} of ${models.models.length} models available on this server`);
      const want = prefs.model && models.models.some((m) => m.key === prefs.model && m.available) ? prefs.model : models.default;
      modelSel.value = want;
      hl.regions.forEach((r) => { const o = document.createElement("option"); o.value = r.code; o.textContent = r.name; regionSel.appendChild(o); });
      regionSel.value = prefs.region || hl.regions[0].code;
    } catch {
      const o = document.createElement("option"); o.textContent = "server unavailable"; modelSel.appendChild(o);
    }
    history = store.get(STORE, []);
    for (let i = 0; i < history.length; i += 2) {
      addUser(history[i].content);
      if (history[i + 1]) { const v = addAssistant(); v.bubble.dataset.turn = String(++turn); renderMarkdown(v.body, history[i + 1].content); }
    }
    input.focus();
  }
  init();
})();
