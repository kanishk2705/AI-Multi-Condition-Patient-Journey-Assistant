/* Browser fallback: BM25 + keyword metadata + extractive answers, used only when the backend is
   asleep or down. Line-for-line port of src/pjassist/{text,bm25,classifier}.py and the extractive
   answerer in generate.py; tests/test_parity.py checks the rankings and answers match Python.
   Works in the browser (window.Lexical) and in Node (module.exports). */
(function (root) {
  "use strict";
  let STOP = new Set();
  const SPLIT = /[^a-z0-9]+/;

  function stem(t) {
    if (t.length > 5 && t.endsWith("ies")) return t.slice(0, -3) + "y";
    if (t.length > 5 && t.endsWith("ing")) return t.slice(0, -3);
    if (t.length > 4 && t.endsWith("ed")) return t.slice(0, -2);
    if (t.length > 4 && t.endsWith("es") && !t.endsWith("ses")) return t.slice(0, -2);
    if (t.length > 3 && t.endsWith("s") && !t.endsWith("ss")) return t.slice(0, -1);
    return t;
  }
  function tokens(text) {
    const out = [];
    for (const t of text.toLowerCase().split(SPLIT)) {
      if (t.length < 2 || STOP.has(t)) continue;
      out.push(stem(t));
    }
    return out;
  }
  function counter(arr) { const m = new Map(); for (const x of arr) m.set(x, (m.get(x) || 0) + 1); return m; }

  const K1 = 1.2, B = 0.75;
  function BM25(texts) {
    this.docs = texts.map((t) => counter(tokens(t)));
    this.len = this.docs.map((d) => { let s = 0; d.forEach((v) => { s += v; }); return s; });
    this.avgdl = this.len.reduce((a, b) => a + b, 0) / Math.max(1, this.len.length);
    const df = new Map();
    for (const d of this.docs) for (const k of d.keys()) df.set(k, (df.get(k) || 0) + 1);
    const n = this.docs.length;
    this.idf = new Map();
    df.forEach((f, t) => this.idf.set(t, Math.log(1 + (n - f + 0.5) / (f + 0.5))));
  }
  BM25.prototype.scores = function (query) {
    const q = tokens(query);
    return this.docs.map((d, i) => {
      let s = 0;
      for (const t of q) {
        const tf = d.get(t) || 0;
        if (tf) s += this.idf.get(t) * tf * (K1 + 1) / (tf + K1 * (1 - B + B * this.len[i] / this.avgdl));
      }
      return s;
    });
  };

  // ---- keyword classifier (cues come from the index file, exported from classifier.py) ----
  let CUES = null;
  const STAGE_ORDER = ["warning", "manage", "understand"];
  function phrases(list) { return list.map((c) => tokens(c)).filter((p) => p.length); }
  function count(tok, ph) {
    let n = 0;
    for (const p of ph) for (let i = 0; i + p.length <= tok.length; i++) {
      let ok = true; for (let j = 0; j < p.length; j++) if (tok[i + j] !== p[j]) { ok = false; break; }
      if (ok) n++;
    }
    return n;
  }
  function classify(question) {
    const tok = tokens(question);
    const low = " " + question.toLowerCase().replace(/’/g, "'").replace(/[^a-z0-9' ]+/g, " ").split(/\s+/).filter(Boolean).join(" ") + " ";
    const hits = {};
    for (const k of ["asthma", "diabetes", "both"]) hits[k] = count(tok, CUES.condPh[k]);
    const conds = new Set();
    if (hits.both) { conds.add("asthma"); conds.add("diabetes"); }
    for (const c of ["asthma", "diabetes"]) if (hits[c]) conds.add(c);
    const sh = {};
    for (const k of STAGE_ORDER) {
      let raw = 0;
      for (const c of CUES.stageRaw[k]) { let idx = low.indexOf(" " + c + " "); while (idx !== -1) { raw++; idx = low.indexOf(" " + c + " ", idx + 1); } }
      sh[k] = count(tok, CUES.stagePh[k]) + raw;
    }
    const best = Math.max(...STAGE_ORDER.map((k) => sh[k]));
    const stage = best === 0 ? null : STAGE_ORDER.find((k) => sh[k] === best);
    return { conditions: [...conds].sort(), stage };
  }

  // ---- sentence splitting and extractive answer (same rules as grounding.py / generate.py) ----
  function splitSentences(text) {
    text = text.trim().replace(/\s+/g, " ");
    return text.replace(/([.!?](?:\s*\[P\d+\])*)\s+(?=[A-Z0-9"'(])/g, "$1\n").split("\n").map((s) => s.trim()).filter(Boolean);
  }
  function extractive(question, shown, idf, maxSents = 3, topPassages = 3) {
    const q = counter(tokens(question));
    const cands = [];
    shown.slice(0, topPassages).forEach((p, pi) => {
      splitSentences(p.text).forEach((s, j) => {
        const st = new Set(tokens(s));
        let sc = 0; q.forEach((_, t) => { if (st.has(t)) sc += idf.get(t) || 0; });
        cands.push([sc / Math.sqrt(1 + st.size), pi + 1, j, s]);
      });
    });
    const order = cands.map((c, i) => [c, i]).sort((a, b) => (b[0][0] - a[0][0]) || (a[1] - b[1])).map((x) => x[0]);
    const chosen = order.slice(0, maxSents).filter((c) => c[0] > 0);
    if (!chosen.length) return { text: "", abstained: true };
    chosen.sort((a, b) => (a[1] - b[1]) || (a[2] - b[2]));
    return { text: chosen.map((c) => `${c[3].replace(/\s+$/, "")} [P${c[1]}]`).join(" "), abstained: false };
  }

  // ---- fallback retrieval: BM25 with a simple metadata preference (an approximation, not the evaluated system) ----
  function search(index, question, meta) {
    const s = index.bm25.scores(question);
    const qc = new Set(meta.conditions || []);
    const scored = index.passages.map((p, i) => {
      let x = s[i];
      if (x > 0 && qc.size) {
        const pc = p.conditions;
        if (pc.some((c) => qc.has(c))) x *= 1.25; else if (pc.length) x *= 0.5;
      }
      if (x > 0 && meta.stage && p.stage === meta.stage) x *= 1.1;
      return [x, i];
    });
    scored.sort((a, b) => (b[0] - a[0]) || (a[1] - b[1]));
    return scored.filter((r) => r[0] > 0).map((r) => r[1]);
  }

  // ---- safety layer (patterns exported from safety.py; same order: emergency, dosing, diagnosis) ----
  let SAFETY = null;
  function safetyCheck(question) {
    let q = question.toLowerCase().replace(/\u2019/g, "'");
    for (const p of SAFETY.neutralise) q = q.replace(new RegExp(p, "g"), " device ");
    q = q.replace(/\s+/g, " ");
    for (const kind of ["emergency", "dosing", "diagnosis"]) {
      for (const p of SAFETY[kind]) if (new RegExp(p).test(q)) return { kind, message: SAFETY.messages[kind], rule: p };
    }
    return { kind: "ok", message: "" };
  }

  function load(data) {
    SAFETY = data.safety;
    STOP = new Set(data.stopwords);
    CUES = { condPh: {}, stagePh: {}, stageRaw: {} };
    for (const k in data.condition_cues) CUES.condPh[k] = phrases(data.condition_cues[k]);
    for (const k in data.stage_cues) {
      CUES.stagePh[k] = phrases(data.stage_cues[k]);
      CUES.stageRaw[k] = data.stage_cues[k].filter((c) => tokens(c).length === 0);
    }
    const passages = data.passages;
    return { passages, bm25: new BM25(passages.map((p) => p.title + " " + (p.section || "") + " " + p.text)),
             idfText: new BM25(passages.map((p) => p.text)).idf };
  }

  const api = { load, tokens, stem, BM25, classify, splitSentences, extractive, search, safetyCheck };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.Lexical = api;
})(typeof window !== "undefined" ? window : globalThis);
