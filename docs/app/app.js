/* Patient Journey Assistant: front end.
   Live mode calls the backend API (Render, free tier). If the API is asleep or down, the page
   switches to fallback mode: keyword (BM25) search over the shipped passages with copied
   sentences, run entirely in the browser, and says so clearly. No cookies, no analytics. */
(function () {
  "use strict";
  const API = "https://patient-journey-assistant-api.onrender.com";
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const STAGE = { understand: "Understanding", manage: "Day-to-day", warning: "Warning signs" };
  const COND = { asthma: "Asthma", diabetes: "Type 2 diabetes" };
  const EXAMPLES = [
    "Which vaccines are recommended if I have asthma and diabetes?",
    "Can the steroid tablets used for asthma affect blood sugar?",
    "What are the symptoms of low blood sugar?",
    "When does an asthma attack need the emergency room?",
    "Can insulin be taken with an inhaler?",
    "How can I keep exercising if activity sets off my asthma?",
    "How much does insulin cost in India?",
    "How many puffs of my reliever can I take in a day?",
  ];
  let mode = "checking";
  let fallback = null; // {data, idx}
  let lastQuery = null;
  let routed = false; // move focus to the heading only on navigation, not on first load

  // ---------------- views ----------------
  function show(view) {
    document.querySelectorAll(".view").forEach((v) => { v.hidden = v.id !== "view-" + view; });
    document.querySelectorAll(".journey a").forEach((a) => a.setAttribute("aria-current", a.dataset.view === view ? "true" : "false"));
    if (view === "results") loadResults();
    const h = $("#view-" + view + " h1");
    if (h && routed) { h.setAttribute("tabindex", "-1"); h.focus({ preventScroll: true }); }
    routed = true;
    window.scrollTo(0, 0);
  }
  function route() {
    const v = (location.hash || "#ask").slice(1);
    show(["ask", "compare", "results", "about"].includes(v) ? v : "ask");
  }
  window.addEventListener("hashchange", route);

  // ---------------- service status ----------------
  function setStatus(kind, text) {
    $("#status").innerHTML = `<span class="pill ${kind}">${esc(text)}</span>`;
  }
  async function fetchT(url, opt = {}, ms = 8000) {
    const c = new AbortController(); const t = setTimeout(() => c.abort(), ms);
    try { return await fetch(url, { ...opt, signal: c.signal, credentials: "omit", referrerPolicy: "no-referrer" }); }
    finally { clearTimeout(t); }
  }
  async function checkService() {
    setStatus("checking", "Connecting to the answer service…");
    const start = Date.now();
    while (Date.now() - start < 75000) {
      try {
        const r = await fetchT(API + "/health", {}, 8000);
        if (r.ok) { const j = await r.json(); if (j.ok) { mode = "live"; setStatus("live", "Live: answers written by a language model from the sources, then checked sentence by sentence"); return; } }
      } catch (e) { /* asleep or down; keep trying for a while */ }
      setStatus("checking", "Waking the answer service. This can take up to a minute…");
      await new Promise((r) => setTimeout(r, 5000));
    }
    await enterFallback();
  }
  async function enterFallback() {
    mode = "fallback";
    setStatus("fallback", "Fallback mode: the answer service is unavailable, so this page is using simple keyword search in your browser. Answers are copied sentences, not checked, and less reliable.");
    await loadFallback();
  }
  async function loadFallback() {
    if (fallback) return fallback;
    const data = await (await fetch("data/index.json")).json();
    fallback = { data, idx: Lexical.load(data) };
    return fallback;
  }

  // ---------------- asking ----------------
  function readForm() {
    const q = $("#q").value.trim();
    const conds = [...document.querySelectorAll('input[name="cond"]:checked')].map((i) => i.value);
    return { question: q, conditions: conds.length ? conds : null, stage: $("#stage").value || null };
  }
  async function ask(compare) {
    const f = readForm();
    if (f.question.length < 3) { $("#q").focus(); return; }
    lastQuery = f;
    const target = compare ? $("#cmpout") : $("#out");
    if (compare) location.hash = "#compare";
    target.innerHTML = `<p class="muted" role="status">Searching the sources and writing a checked answer. On the free server this can take up to a minute…</p>`;
    setBusy(true);
    try {
      if (mode === "live") {
        const r = await fetchT(API + (compare ? "/compare" : "/ask"), {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(f),
        }, 150000);
        if (r.status === 429) { const j = await r.json().catch(() => ({})); target.innerHTML = callout("note", "Please wait a moment", j.detail || "Too many questions in a short time. Try again in a minute."); return; }
        if (!r.ok) throw new Error("service error " + r.status);
        const j = await r.json();
        target.innerHTML = compare ? renderCompare(j, f) : renderResult(j, f, "live");
      } else {
        await loadFallback();
        const res = fallbackAnswer(f);
        target.innerHTML = compare ? callout("note", "Compare needs the live service", "The answer service is unavailable, so only the in-browser keyword search can run. Here is its answer.") + renderResult(res, f, "fallback")
                                   : renderResult(res, f, "fallback");
      }
    } catch (e) {
      await enterFallback();
      const res = fallbackAnswer(f);
      target.innerHTML = renderResult(res, f, "fallback");
    } finally { setBusy(false); }
  }
  function setBusy(b) { $("#askbtn").disabled = b; $("#cmpbtn").disabled = b; }

  function fallbackAnswer(f) {
    const { data, idx } = fallback;
    const s = Lexical.safetyCheck(f.question);
    if (s.kind !== "ok") return { route: s.kind, message: s.message };
    const inferred = Lexical.classify(f.question);
    const meta = { conditions: f.conditions || inferred.conditions, stage: f.stage || inferred.stage };
    const order = Lexical.search(idx, f.question, meta).slice(0, 4);
    const passages = order.map((i, n) => ({ label: "P" + (n + 1), ...data.passages[i] }));
    const out = { route: "ok", inferred, used_meta: meta, passages, system: "fallback (BM25 keyword search)" };
    if (!passages.length) return { ...out, abstained: true, abstain_reason: "No passage shares words with the question." };
    const a = Lexical.extractive(f.question, passages, idx.idfText);
    if (a.abstained) return { ...out, abstained: true, abstain_reason: "No sentence in the passages matches the question." };
    const sentences = Lexical.splitSentences(a.text).map((t) => ({ sentence: t, cited: (t.match(/\[(P\d+)\]/g) || []).map((x) => x.slice(1, -1)), supported: null }));
    return { ...out, abstained: false, answer: a.text, sentences };
  }

  // ---------------- rendering ----------------
  function callout(kind, title, body) {
    return `<div class="callout ${kind}" role="${kind === "urgent" ? "alert" : "note"}"><h2 class="h-small">${esc(title)}</h2><p>${esc(body)}</p></div>`;
  }
  function tags(conds, stage) {
    const c = (conds || []).length === 2 ? `<span class="tag both">Both conditions</span>` : (conds || []).map((x) => `<span class="tag ${x}">${esc(COND[x] || x)}</span>`).join("");
    return c + (stage ? `<span class="tag stage">${esc(STAGE[stage] || stage)}</span>` : "");
  }
  let PASSAGES = {};
  function renderResult(j, f, how, title) {
    if (j.route === "emergency") return callout("urgent", "Get urgent help now", j.message);
    if (j.route === "dosing") return callout("note", "This needs your own care team", j.message);
    if (j.route === "diagnosis") return callout("note", "Only a health professional can tell you that", j.message);
    (j.passages || []).forEach((p) => { PASSAGES[p.id] = p; });
    const head = title ? `<h2>${esc(title)}</h2>` : `<h2>${j.abstained ? "No answer from these sources" : "Answer"}</h2>`;
    const modeline = j.abstained ? `<p class="modeline">${j.system === "dense" ? "Plain semantic search" : how === "fallback" ? "Fallback keyword search" : "Hybrid search with condition matching and re-ranking"} found the passages below, but they don't cover the question.</p>` : how === "fallback"
      ? `<p class="modeline">Fallback mode: sentences copied from the top keyword-search passages. Not checked by the answer checker.</p>`
      : `<p class="modeline">${j.system === "dense" ? "Plain semantic search" : "Hybrid search with condition matching and re-ranking"}; answer written by gpt-oss-120b from the passages below, then checked.</p>`;
    let body;
    if (j.abstained) {
      const extra = j.llm_available === false ? " The daily limit for written answers has been reached; the passages found are listed below." : "";
      body = `<div class="callout ok"><p><strong>The sources here don't cover this well enough to answer.</strong> ${esc(j.abstain_reason || "")}${esc(extra)}</p><p class="small">For questions outside these pages, ask your doctor, nurse or pharmacist.</p></div>`;
    } else {
      body = (j.sentences || []).map((s) => {
        const labs = s.cited || [];
        const text = s.sentence.replace(/\s*\[P\d+\]/g, "");
        const btns = labs.map((l) => { const p = (j.passages || []).find((x) => x.label === l); return p ? `<button type="button" class="cite" data-pid="${esc(p.id)}" aria-label="Open source ${esc(l)}: ${esc(p.source)}, ${esc(p.title)}">${esc(l)}</button>` : ""; }).join("");
        const chk = s.supported === true ? `<span class="check yes" title="An independent checker found this sentence in the cited passage">✓ found in source</span>`
                  : s.supported === false ? `<span class="check no" title="The small checker model on this free server could not confirm this sentence from the cited passage. Open the source to check it yourself.">checker unsure</span>` : "";
        return `<p class="sent">${esc(text)}${btns}${chk}</p>`;
      }).join("") + (j.scope_notes || []).map((n) => `<p class="sent muted">${esc(n)}</p>`).join("");
      if (j.fully_grounded === false) body += `<p class="small muted">“Checker unsure” means the automatic checker could not confirm a sentence. This free server runs a small checker model that often misses sentences that are in the source, so open the cited passage and read it before relying on that sentence.</p>`;
    }
    const why = `<details class="why"><summary>Why this answer</summary>
      <dl class="kv"><dt>Detected from your question</dt><dd>${tags(j.inferred?.conditions, j.inferred?.stage) || "no clear condition or stage"}</dd>
      <dt>Used for the search</dt><dd>${tags(j.used_meta?.conditions, j.used_meta?.stage) || "none"}</dd>
      ${j.gate_score != null ? `<dt>Match score</dt><dd>${j.gate_score.toFixed(2)} (answers need at least ${j.gate_threshold.toFixed(2)})</dd>` : ""}</dl>
      <ol class="plist">${(j.passages || []).map((p) => `<li><div class="ptitle">${esc(p.label)} · ${esc(p.source)}: ${esc(p.title)}${p.section ? " — " + esc(p.section) : ""}</div>
        <div>${tags(p.conditions, p.stage)}</div><p class="ptext">${esc(p.text.slice(0, 260))}${p.text.length > 260 ? "…" : ""}</p>
        <button type="button" class="cite" data-pid="${esc(p.id)}">Read full passage</button></li>`).join("")}</ol></details>`;
    return `<article class="answer">${head}${modeline}${body}${why}<p class="notice">Not medical advice. Educational information from public health sources; check with your health care team before acting on it.</p></article>`;
  }
  function renderCompare(j, f) {
    const n = (r) => r.route !== "ok" ? null : r;
    if (!n(j.full)) return renderResult(j.full, f, "live");
    return `<p class="small muted">Question: “${esc(f.question)}”</p><div class="cmpgrid">${renderResult(j.plain, f, "live", "Plain semantic search")}${renderResult(j.full, f, "live", "Journey-aware system")}</div>`;
  }

  // passage dialog
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.cite"); if (!b) return;
    const p = PASSAGES[b.dataset.pid]; if (!p) return;
    $("#pd-title").textContent = p.source + ": " + p.title;
    $("#pd-body").innerHTML = `<p>${tags(p.conditions, p.stage)}</p>${p.section ? `<p class="muted">${esc(p.section)}</p>` : ""}
      <blockquote>${esc(p.text)}</blockquote>
      <p><a href="${esc(p.url)}" target="_blank" rel="noopener noreferrer">Open the original page on ${esc(new URL(p.url).hostname)}</a></p>
      <p class="small muted">Text as retrieved on 24 October 2026; the source page may have changed since. Credit: ${esc(p.source)}. Use does not imply endorsement.</p>`;
    $("#passage").showModal();
  });

  // ---------------- precomputed results ----------------
  let resultsLoaded = false;
  async function loadResults() {
    if (resultsLoaded) return; resultsLoaded = true;
    try {
      const r = await (await fetch("data/results.json")).json();
      $("#resout").innerHTML = renderResults(r);
    } catch (e) { $("#resout").innerHTML = `<p>The results file could not be loaded. They are also on the <a href="../#results">project page</a>.</p>`; }
  }
  function pct(x) { return x == null ? "–" : (100 * x).toFixed(0) + "%"; }
  function f3(x) { return x == null ? "–" : x.toFixed(2); }
  function ci(o, p = false) { return o ? `${p ? pct(o.mean ?? o.value) : f3(o.mean ?? o.value)} <span class="ci">${p ? pct(o.lo) : f3(o.lo)}–${p ? pct(o.hi) : f3(o.hi)}</span>` : "–"; }
  function renderResults(r) {
    const sys = r.systems;
    const rows = sys.map((s) => `<tr><td>${esc(s.label)}</td><td class="num">${ci(s.mrr)}</td><td class="num">${ci(s.success5, true)}</td><td class="num">${ci(s.wrong_condition, true)}</td><td class="num">${ci(s.wrong_stage, true)}</td></tr>`).join("");
    return `<p>${esc(r.headline)}</p>
      <h2>Finding the right passages</h2>
      <div class="tw"><table><caption class="small muted" style="caption-side:bottom;text-align:left;padding:8px 12px">Test questions with an answer in the sources (n = ${r.n_retrieval}). MRR: mean reciprocal rank of the first right passage (1 = always first). Small figures: 95% bootstrap intervals.</caption>
      <thead><tr><th>Search method</th><th class="num">MRR</th><th class="num">Right passage in top 5</th><th class="num">Wrong condition (top 5)</th><th class="num">Wrong stage (top 5)</th></tr></thead><tbody>${rows}</tbody></table></div>
      <h2>Knowing when not to answer</h2>
      ${r.coverage_svg}
      <p class="small muted">${esc(r.coverage_caption)}</p>
      <h2>Are the answers backed by the sources?</h2>
      <div class="tw"><table><thead><tr><th>Answer writer</th><th class="num">Sentences confirmed in the cited passage</th><th class="num">Answers with every sentence confirmed</th><th class="num">Answers</th></tr></thead>
      <tbody>${r.grounding.map((g) => `<tr><td>${esc(g.label)}</td><td class="num">${ci(g.sentences, true)}</td><td class="num">${ci(g.full, true)}</td><td class="num">${g.n}</td></tr>`).join("")}</tbody></table></div>
      <h2>Safety routing</h2><p>${esc(r.safety_text)}</p>
      <h2>Where it went wrong</h2><div class="cases">${r.failures.map((c) => `<div class="case"><div class="q">${esc(c.question)}</div><p class="small">${esc(c.what)}</p></div>`).join("")}</div>`;
  }

  // ---------------- init ----------------
  $("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="secondary">${esc(q)}</button>`).join("");
  $("#examples").addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; $("#q").value = b.textContent; $("#qcount").textContent = b.textContent.length; ask(false); });
  $("#q").addEventListener("input", () => { $("#qcount").textContent = $("#q").value.length; });
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); ask(false); });
  $("#cmpbtn").addEventListener("click", () => ask(true));
  route();
  if (new URLSearchParams(location.search).get("fallback") === "1") enterFallback(); // test or try the in-browser mode on purpose
  else checkService();
})();
