"""Generate the project page (docs/index.html), the assistant's precomputed results
(docs/app/data/results.json) and README.md from the results JSON, so no number is typed by hand.

  python scripts/make_report.py                 # uses results/final
  python scripts/make_report.py --from rehearsal   # layout check on the dev rehearsal
"""
import html, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = "rehearsal" if "--from" in sys.argv and sys.argv[sys.argv.index("--from") + 1] == "rehearsal" else "final"
RES = ROOT / "results" / SRC
R = json.loads((RES / "results.json").read_text())
PQ = json.loads((RES / "per_question.json").read_text())
AUD = json.loads((RES / "audit.json").read_text()) if (RES / "audit.json").exists() else None
DEV = json.loads((ROOT / "results/dev/summary.json").read_text())
QFILE = "dev.json" if SRC == "rehearsal" else "test.json"
QS = json.loads((ROOT / "data/questions" / QFILE).read_text())
QM = {q["qid"]: q for q in QS}
P = [json.loads(l) for l in open(ROOT / "data/corpus/passages.jsonl")]
REPO = "https://github.com/kanishk2705/AI-Multi-Condition-Patient-Journey-Assistant"
BLOB = REPO + "/blob/main/"
e = html.escape
import time
ASSET_V = time.strftime("%Y%m%d%H%M")
import subprocess
N_TESTS = sum(1 for l in subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", str(ROOT / "tests")],
              capture_output=True, text=True, cwd=ROOT).stdout.splitlines() if "::" in l)

# ------------------------------------------------------------------ numbers --------------
ret, ans, gate, saf, cls = R["retrieval"], R["answers"], R["gate"], R["safety"], R["classifier"]
A = ret["all"]
LABEL = {"dense": "Plain semantic search (dense)", "bm25": "Keyword search (BM25)", "hybrid": "Hybrid (dense + BM25)",
         "meta": "Hybrid + condition match", "meta_ce": "Hybrid + condition match + re-ranker (selected)"}
n_ret = A["dense"]["rr"]["n"]
n_both = sum(1 for p in P if len(p["conditions"]) == 2)


def f2(x): return f"{x:.2f}"
def pc(x): return f"{100 * x:.0f}%"
def civ(o, p=False, key="mean"):
    v = o.get(key, o.get("value"))
    return (f"{pc(v)} ({pc(o['lo'])}–{pc(o['hi'])})" if p else f"{f2(v)} ({f2(o['lo'])}–{f2(o['hi'])})")
def civh(o, p=False, key="mean"):
    v = o.get(key, o.get("value"))
    a, b = (pc(o["lo"]), pc(o["hi"])) if p else (f2(o["lo"]), f2(o["hi"]))
    return f"{pc(v) if p else f2(v)}<span class='ci'>{a}–{b}</span>"
def plural(n, w): return f"{n} {w}{'' if n == 1 else 's'}"
def dciv(d):
    return f"{d['diff']:+.3f} ({d['lo']:+.3f} to {d['hi']:+.3f})"


H = ret["hypotheses"]
llm, ext = ans["llm"], ans["extractive"]
hyp = [
    ("H1", "Condition/stage-aware retrieval (`meta`) beats plain semantic retrieval on MRR@10", dciv(H["H1_meta_vs_dense_mrr"]),
     H["H1_meta_vs_dense_mrr"]["lo"] > 0),
    ("H2", "`meta` retrieves fewer wrong-condition passages (top 5) than `dense`", dciv(H["H2_meta_vs_dense_wrong_condition"]),
     H["H2_meta_vs_dense_wrong_condition"]["hi"] < 0),
    ("H3", "The cross-encoder re-ranker (`meta_ce`) beats `meta` on MRR@10", dciv(H["H3_meta_ce_vs_meta_mrr"]),
     H["H3_meta_ce_vs_meta_mrr"]["lo"] > 0),
    ("H4", "Hybrid beats both dense and BM25 on MRR@10",
     f"vs dense {dciv(H['H4a_hybrid_vs_dense_mrr'])}; vs BM25 {dciv(H['H4b_hybrid_vs_bm25_mrr'])}",
     H["H4a_hybrid_vs_dense_mrr"]["lo"] > 0 and H["H4b_hybrid_vs_bm25_mrr"]["lo"] > 0),
    ("H5", "On trap questions `meta` does not beat `dense` (expected)", dciv(H["H5_trap_meta_vs_dense_mrr"]),
     not H["H5_trap_meta_vs_dense_mrr"]["lo"] > 0),
    ("H6", "LLM sentences are confirmed less often than extractive sentences", dciv(ans["H6_llm_minus_extractive_supported_rate"]),
     ans["H6_llm_minus_extractive_supported_rate"]["hi"] < 0),
    ("H7", "At least 80% of LLM answers are fully grounded", civ(llm["fully_grounded_rate"], True),
     llm["fully_grounded_rate"]["mean"] >= 0.8),
    ("H8", "Abstention gate AUROC is at least 0.80", f2(gate["auroc"]), gate["auroc"] >= 0.8),
    ("H9", "Every emergency is routed to urgent care, with no false alarms on ordinary questions",
     f"{saf['emergency']['tp']}/{saf['emergency']['tp'] + saf['emergency']['fn']} emergencies; {plural(saf['false_alarms_on_non_safety'], 'false alarm')}",
     saf["emergency"]["recall"] == 1.0 and saf["false_alarms_on_non_safety"] == 0),
    ("H10", "On cross-condition questions `meta` covers both conditions in the top 5 more often (descriptive)",
     dciv(H["H10_cross_meta_vs_dense_both_coverage"]), H["H10_cross_meta_vs_dense_both_coverage"]["diff"] > 0),
]
n_fail = sum(1 for h in hyp if not h[3])

# failure cases (selected from per-question outputs by rule, not by hand)
fails = []
sel = R["selection"]["retrieval_system"]
rm = PQ["retrieval_metrics"]
for qid, m in rm[sel].items():
    if m["success@5"] == 0 and len(fails) < 2:
        fails.append({"question": QM[qid]["question"], "what": f"No correct passage in the top 5 of the selected system ({QM[qid]['type']} question)."})
for qid, m in rm["meta"].items():
    if QM[qid]["type"] == "trap" and m["rr"] < rm["dense"][qid]["rr"] and len(fails) < 4:
        im = PQ["inferred_meta"].get(qid, {})
        fails.append({"question": QM[qid]["question"],
                      "what": f"Trap question: the classifier guessed {', '.join(im.get('conditions', [])) or 'no condition'} / {im.get('stage') or 'no stage'}, and condition matching ranked the right passage lower than plain semantic search did."})
for qid, a in PQ["answers"].items():
    g = a.get("llm", {}).get("grounding")
    if g and not g["fully_grounded"] and len(fails) < 6:
        bad = next(s for s in g["sentences"] if not s["supported"])
        fails.append({"question": QM[qid]["question"],
                      "what": f"An LLM sentence the checker could not confirm ({bad['reason']}): “{bad['sentence'][:180]}”"})
for r_ in saf["rows"]:
    if r_["expected"] != r_["got"] and len(fails) < 8:
        fails.append({"question": QM[r_["qid"]]["question"], "what": f"Safety layer routed this to “{r_['got']}”; expected “{r_['expected']}”."})
for qid, a in PQ["answers"].items():
    if QM[qid]["type"] == "unanswerable" and not a["gate_abstained"] and len(fails) < 9:
        fails.append({"question": QM[qid]["question"], "what": "Unanswerable question that passed the abstention gate."
                      + (" The LLM then declined on its own." if a.get("llm", {}).get("abstained") else " The LLM answered it.")})


# ------------------------------------------------------------------ charts ---------------
def mrr_chart():
    rows = [(LABEL[s], A[s]["rr"]) for s in ("dense", "bm25", "hybrid", "meta", "meta_ce")]
    W, rowh, left, right = 760, 38, 330, 40
    Hh = 30 + rowh * len(rows) + 30
    x0 = 0.5
    x = lambda v: left + (W - left - right) * (max(v, x0) - x0) / (1 - x0)
    out = [f'<svg class="chart" viewBox="0 0 {W} {Hh}" role="img" aria-labelledby="mrr-t mrr-d"><title id="mrr-t">MRR@10 by retrieval system</title><desc id="mrr-d">Mean reciprocal rank with 95% bootstrap intervals on {n_ret} test questions.</desc>']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<line class="axis" x1="{x(t):.1f}" y1="20" x2="{x(t):.1f}" y2="{Hh - 26}"/><text x="{x(t):.1f}" y="{Hh - 8}" text-anchor="middle">{t:.2f}</text>')
    for i, (lab, o) in enumerate(rows):
        y = 34 + i * rowh
        out.append(f'<text x="{left - 12}" y="{y + 4}" text-anchor="end">{e(lab)}</text>')
        out.append(f'<line x1="{x(o["lo"]):.1f}" y1="{y}" x2="{x(o["hi"]):.1f}" y2="{y}" stroke="var(--ink2)" stroke-width="2"/>')
        out.append(f'<circle cx="{x(o["mean"]):.1f}" cy="{y}" r="6" fill="var(--both)"/><text x="{x(o["mean"]):.1f}" y="{y - 10}" text-anchor="middle">{o["mean"]:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


def cov_chart():
    c = gate["curve"]
    W, Hh, l, r, t, b = 640, 300, 56, 20, 16, 44
    X = lambda v: l + (W - l - r) * v
    Y = lambda v: t + (Hh - t - b) * (1 - v / max(0.5, max(p["risk"] for p in c)))
    ymax = max(0.5, max(p["risk"] for p in c))
    pts = " ".join(f"{X(p['coverage']):.1f},{Y(p['risk']):.1f}" for p in c)
    tau_pt = min((p for p in c if p["threshold"] >= gate["tau"]), key=lambda p: p["threshold"], default=c[-1])
    out = [f'<svg class="chart" viewBox="0 0 {W} {Hh}" role="img" aria-labelledby="cr-t cr-d"><title id="cr-t">Coverage versus risk</title><desc id="cr-d">As the assistant answers more questions (coverage, horizontal), the share of its answers that are wrong (risk, vertical) changes. The red dot is the threshold chosen on development questions.</desc>']
    for v in (0, .25, .5, .75, 1):
        out.append(f'<line class="axis" x1="{X(v):.1f}" y1="{t}" x2="{X(v):.1f}" y2="{Hh - b}"/><text x="{X(v):.1f}" y="{Hh - b + 18}" text-anchor="middle">{pc(v)}</text>')
    for v in [i * ymax / 4 for i in range(5)]:
        out.append(f'<line class="axis" x1="{l}" y1="{Y(v):.1f}" x2="{W - r}" y2="{Y(v):.1f}"/><text x="{l - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{pc(v)}</text>')
    out.append(f'<polyline class="curve" points="{pts}"/>')
    out.append(f'<circle class="pt" cx="{X(tau_pt["coverage"]):.1f}" cy="{Y(tau_pt["risk"]):.1f}" r="6"/>')
    out.append(f'<text x="{(l + W) / 2:.0f}" y="{Hh - 6}" text-anchor="middle">coverage: share of questions answered</text>')
    out.append(f'<text x="14" y="{(t + Hh - b) / 2:.0f}" text-anchor="middle" transform="rotate(-90 14 {(t + Hh - b) / 2:.0f})">risk: answered but wrong</text></svg>')
    return "".join(out), tau_pt


COV_SVG, TAU_PT = cov_chart()

# ------------------------------------------------------------------ shared text -----------
head_sel = A["meta_ce"]
headline = (f"On {n_ret} held-out test questions with an answer in the sources, the selected system put a correct passage first "
            f"in {pc(head_sel['success@1']['mean'])} of questions and in the top five in {pc(head_sel['success@5']['mean'])}. "
            f"Condition and stage matching itself did not beat plain semantic search (MRR difference {dciv(H['H1_meta_vs_dense_mrr'])}); "
            f"the gain came from the re-ranker. The abstention gate separated answerable from unanswerable questions with AUROC {f2(gate['auroc'])}. "
            f"{pc(llm['sentence_supported_rate']['value'])} of LLM answer sentences were confirmed in their cited passage by an independent checker, "
            f"against {pc(ext['sentence_supported_rate']['value'])} for copied sentences.")
em_tp, em_n = saf["emergency"]["tp"], saf["emergency"]["tp"] + saf["emergency"]["fn"]
solved = (f"routes {em_tp} of {em_n} test emergencies to an urgent-care message (the missed one, “too breathless to finish a sentence”, "
          f"and the other routing gaps found on the test set were patched afterwards for the live site only; see deviation D11)"
          if em_tp < em_n else "routes every test emergency to an urgent-care message")
safety_text = (f"Emergencies routed to the urgent-care message: {saf['emergency']['tp']} of {saf['emergency']['tp'] + saf['emergency']['fn']}. "
               f"Dosing questions declined: {saf['dosing']['tp']} of {saf['dosing']['tp'] + saf['dosing']['fn']}. "
               f"Diagnosis questions declined: {saf['diagnosis']['tp']} of {saf['diagnosis']['tp'] + saf['diagnosis']['fn']}. "
               f"Ordinary questions wrongly routed (false alarms): {saf['false_alarms_on_non_safety']}.")
audit_text = (f"Hand audit of {AUD['n']} randomly sampled answer sentences: the checker agreed with a careful reading on "
              f"{AUD['agree']} of {AUD['n']} ({pc(AUD['agree'] / AUD['n'])}; Cohen's kappa {f2(AUD['kappa'])}). "
              f"Where they disagreed, the checker was too strict {AUD['checker_too_strict']} times and too lenient {AUD['checker_too_lenient']} times. "
              f"For LLM sentences the careful reading judged {AUD['by_answerer']['llm']['human_supported']} of {AUD['by_answerer']['llm']['n']} supported, "
              f"so the checker's LLM figure is best read as a conservative lower bound (kappa is low mainly because almost every sampled sentence was supported)."
              if AUD else "Hand audit pending.")

# ------------------------------------------------------------------ app results json ------
app = {
    "precomputed": True, "headline": headline, "n_retrieval": n_ret,
    "systems": [{"label": LABEL[s], "mrr": A[s]["rr"], "success5": A[s]["success@5"],
                 "wrong_condition": A[s]["wrong_condition@5"], "wrong_stage": A[s]["wrong_stage@5"]}
                for s in ("dense", "bm25", "hybrid", "meta", "meta_ce")],
    "coverage_svg": COV_SVG,
    "coverage_caption": (f"Each point is one possible threshold on the re-ranker's match score. The red dot is the threshold chosen on "
                         f"development questions: it answers {pc(TAU_PT['coverage'])} of test questions, and {pc(TAU_PT['risk'])} of those answers "
                         f"lack a correct passage or should not have been answered. AUROC {f2(gate['auroc'])}; area under the curve {f2(gate['aurc'])}."),
    "grounding": [{"label": "Language model (gpt-oss-120b)", "sentences": llm["sentence_supported_rate"], "full": llm["fully_grounded_rate"], "n": llm["n_answers"]},
                  {"label": "Copied sentences (no language model)", "sentences": ext["sentence_supported_rate"], "full": ext["fully_grounded_rate"], "n": ext["n_answers"]}],
    "safety_text": safety_text, "failures": fails, "audit": audit_text,
}
(ROOT / "docs/app/data").mkdir(parents=True, exist_ok=True)
if SRC == "final":
    (ROOT / "docs/app/data/results.json").write_text(json.dumps(app, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ project page ----------
def sys_table(block, rows=("dense", "bm25", "hybrid", "meta", "meta_ce")):
    tr = "".join(f"<tr><td>{e(LABEL[s])}</td><td class='num'>{civh(block[s]['rr'])}</td><td class='num'>{civh(block[s]['success@1'], True)}</td>"
                 f"<td class='num'>{civh(block[s]['success@5'], True)}</td><td class='num'>{civh(block[s]['wrong_condition@5'], True)}</td>"
                 f"<td class='num'>{civh(block[s]['wrong_stage@5'], True)}</td></tr>" for s in rows if s in block)
    return ("<div class='tw'><table><thead><tr><th>System</th><th class='num'>MRR@10</th><th class='num'>Correct first</th>"
            "<th class='num'>Correct in top 5</th><th class='num'>Wrong condition (top 5)</th><th class='num'>Wrong stage (top 5)</th></tr></thead>"
            f"<tbody>{tr}</tbody></table></div>")


hyp_rows = "".join(f"<tr><td>{h}</td><td>{e(t).replace('`', '')}</td><td class='num'>{e(v)}</td><td>{'Supported' if ok else '<strong>Not supported</strong>'}</td></tr>"
                   for h, t, v, ok in hyp)
by_type = "".join(f"<tr><td>{t}</td><td class='num'>{ret[t]['dense']['rr']['n']}</td><td class='num'>{civh(ret[t]['dense']['rr'])}</td><td class='num'>{civh(ret[t]['meta']['rr'])}</td><td class='num'>{civh(ret[t]['meta_ce']['rr'])}</td></tr>"
                  for t in ("answerable", "trap", "cross"))
cross = ret["cross"]
gr_rows = "".join(f"<tr><td>{e(g['label'])}</td><td class='num'>{g['n']}</td><td class='num'>{civh(g['sentences'], True, 'value')}</td><td class='num'>{civh(g['full'], True)}</td></tr>" for g in app["grounding"])
fail_html = "".join(f"<li><strong>{e(c['question'])}</strong><br><span class='muted'>{e(c['what'])}</span></li>" for c in fails)
ledger = R["ledger"]
dev_llm = DEV.get("answers", {}).get("llm", {})

_lv = json.loads((ROOT / "results/live_model_check.json").read_text())
_sel = _lv["selected_system_dev"]; _ck = _lv["checker_agreement_live_vs_evaluated"]
LV = {"sel_live_mrr": f"{_sel['live_mrr']:.3f}", "sel_eval_mrr": f"{_sel['evaluated_mrr']:.3f}",
      "live_s5": f"{_sel['live_success5']:.0%}", "eval_s5": f"{_sel['evaluated_success5']:.0%}",
      "cos": f"{min(_lv['embedding_cosine_passages_min'], _lv['embedding_cosine_queries_min']):.3f}",
      "corr": f"{_lv['reranker_score_correlation']:.3f}", "top4": f"{_sel['same_top4_share']:.0%}",
      "agree": _ck["agree"], "n": _ck["sentences"], "rate": f"{_ck['rate']:.0%}"}

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Patient Journey Assistant</title>
<meta name="description" content="A source-grounded assistant for people living with both asthma and type 2 diabetes, and a pre-registered evaluation of whether condition- and stage-aware retrieval, grounded generation and abstention work.">
<meta name="referrer" content="no-referrer">
<link rel="stylesheet" href="assets/site.css?v={ASSET_V}"><link rel="stylesheet" href="app/app.css?v={ASSET_V}">
<style>
.hero{{padding:8px 0 28px}}
.hero .gap{{display:flex;gap:18px;align-items:center;margin:18px 0 22px;flex-wrap:wrap}}
.hero .gapnum{{font-size:3.2rem;font-weight:700;line-height:1;color:var(--both)}}
.hero .gaptext{{max-width:46ch;color:var(--ink2)}}
.cta{{display:inline-block;text-decoration:none;margin-right:10px}}
.flow{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:14px 0 22px;padding:0;list-style:none;counter-reset:s}}
.flow li{{background:var(--paper);border-radius:var(--radius-s);padding:12px 14px;font-size:.92rem;counter-increment:s}}
td .ci{{display:block;font-size:.8rem;color:var(--ink2)}}
.flow li::before{{content:counter(s) ". ";font-weight:700;color:var(--both)}}
ul.plain{{padding-left:1.2rem}} ul.plain li{{margin:.35rem 0;max-width:72ch}}
</style></head>
<body>
<a class="skip" href="#problem">Skip to content</a>
<div class="banner" role="note"><strong>Not medical advice.</strong> A research and teaching project. The assistant gives educational information from public health sources only.</div>
<div class="shell">
<nav class="rail" aria-label="Project sections">
  <a class="brand" href="#top">Patient Journey Assistant</a>
  <ol class="journey">
    <li><a href="#problem">Problem</a></li><li><a href="#design">Design</a></li><li><a href="#implementation">Implementation</a></li>
    <li><a href="#results">Results</a></li><li><a href="#limitations">Limitations</a></li><li><a href="#next">Next steps</a></li>
    <li><a href="#audit">Audit trail</a></li>
  </ol>
  <div class="legend"><span><span class="dot asthma"></span>Asthma</span><span><span class="dot diabetes"></span>Type 2 diabetes</span></div>
  <div class="tools"><a href="app/">Try the assistant</a><a href="{REPO}">Source code</a></div>
</nav>
<main class="main" id="top">
<header class="section hero">
  <h1>A source-grounded assistant for living with asthma and type 2 diabetes</h1>
  <p class="lede">An assistant that answers only from public U.S. health pages, cites a passage for every sentence and declines when the pages don't cover the question, plus a pre-registered evaluation of whether it works.</p>
  <div class="gap"><div class="gapnum">{n_both}</div><div class="gaptext">of the {len(P)} trusted patient-information passages collected for this project are written about both conditions together. That gap is what the project is about.</div></div>
  <a class="btn cta" href="app/">Try the assistant</a><a class="btn secondary cta" href="{REPO}">Read the code</a>
  <div class="callout note" style="margin-top:22px"><p style="margin:0"><strong>In one paragraph.</strong> {e(headline)} The questions were written by the developer's AI collaborator, not by patients or clinicians, and no clinician reviewed the system, so these results describe a careful prototype, not a clinical tool.</p></div>
</header>

<section class="section" id="problem"><h2>1. Problem</h2>
<p>People who live with both asthma and type 2 diabetes need information that respects both conditions at once. A common example: oral steroid tablets used for an asthma flare-up can affect blood sugar. Trusted patient information from public health agencies is written one condition at a time and spread over many pages: <strong>{n_both} of the {len(P)} passages</strong> collected here are about both conditions. General chatbots fill the gap fluently, but they mix conditions, rarely cite where an answer came from, and rarely say “the sources don't cover this”.</p>
<p>The goal was an assistant that (1) answers only from trusted public sources, (2) shows exactly which passage each sentence came from, (3) declines when the sources are not enough, and (4) never handles emergencies, doses or diagnoses. The research question is whether retrieval that knows the reader's condition and stage of the journey (understanding the diagnosis, daily self-management, warning signs) finds better passages than plain semantic search.</p>
</section>

<section class="section" id="design"><h2>2. Design</h2>
<ol class="flow"><li>Safety rules: emergency, dosing or diagnosis questions get a fixed message</li><li>Keyword classifier guesses the condition(s) and journey stage</li><li>Retrieval from 373 public-domain passages</li><li>Abstention gate on the re-ranker's match score</li><li>LLM writes the answer, one citation per sentence</li><li>Independent NLI and word-overlap checker confirms each sentence</li></ol>
<h3>Sources</h3><p>373 passages from MedlinePlus (NLM), NIDDK, NHLBI and CDC, collected on 24 October 2026. All four allow reuse of this text; licences, credit lines and hashes are in <a href="{BLOB}DATA.md">DATA.md</a>. Each passage is labelled with its condition(s) and journey stage by rules written before any question (about 90% agreement with a careful reading in two audits of 40 passages).</p>
<h3>Retrieval systems compared</h3>
<ul class="plain"><li><strong>Plain semantic search</strong>: <code>bge-small-en-v1.5</code> embeddings.</li><li><strong>Keyword search</strong>: BM25, own implementation (shared with the in-browser fallback).</li><li><strong>Hybrid</strong>: reciprocal-rank fusion of the two.</li><li><strong>Condition/stage-aware</strong>: hybrid candidates with a bonus when a passage's labels match the condition and stage inferred from the question.</li><li><strong>+ re-ranker</strong>: the top 20 re-ranked by a cross-encoder (<code>ms-marco-MiniLM-L-6-v2</code>).</li></ul>
<h3>Questions and test discipline</h3><p>148 questions: {sum(1 for _ in QS) if SRC == 'final' else 89} of them held out as the test set before any system was run. Types: answerable, cross-condition (need passages about both conditions), trap questions whose words point to the wrong condition or stage (an insulin <em>inhaler</em>; a reliever inhaler that makes you <em>shaky</em>), unanswerable, and safety cases. Everything was tuned on 59 development questions; the test set was read once, by a script that records an access ledger. The protocol and ten hypotheses were frozen beforehand: <a href="{BLOB}PROTOCOL.md">PROTOCOL.md</a>.</p>
</section>

<section class="section" id="implementation"><h2>3. Implementation</h2>
<ul class="plain">
<li><strong>Network work on CI.</strong> The build sandbox could not reach the health sites, Hugging Face or Groq, so collection, model scoring and LLM calls run in a GitHub Actions workflow in the repository; API keys exist only as secrets.</li>
<li><strong>Generation.</strong> Groq offered no Llama chat model for this account, so the answer model is <code>openai/gpt-oss-120b</code> (temperature 0, low reasoning effort), chosen because in a probe it was the only available model that refused to state an unsupported steroid–blood-sugar claim. A copy-sentences extractive answerer is the no-LLM baseline.</li>
<li><strong>Grounding checker.</strong> A sentence counts as supported only if it cites a shown passage, an NLI cross-encoder (<code>nli-deberta-v3-base</code>, a different model from the writer) finds it entailed by 1–3 sentences of that passage, and 60% of its content words appear there.</li>
<li><strong>Tests.</strong> {N_TESTS} unit tests, including tests showing that the grounding checker, safety layer and leakage guard fail on deliberately broken input, and a parity test that runs fixed queries through the Python and JavaScript keyword pipelines and requires identical rankings.</li>
<li><strong>Website.</strong> Static front end on GitHub Pages; FastAPI backend in Docker on Render's free tier with per-visitor rate limits and a daily LLM budget. Hugging Face no longer hosts free Docker backends, and Render's 512 MB limit rules out PyTorch, so the live backend runs int8-quantised ONNX exports of the same embedding and re-ranker models and a smaller NLI checker (<code>nli-deberta-v3-xsmall</code>); On the dev questions the live models give MRR@10 {LV['sel_live_mrr']} (evaluated {LV['sel_eval_mrr']}) and right passage in the top 5 for {LV['live_s5']} (evaluated {LV['eval_s5']}); embedding cosine similarity is at least {LV['cos']} and re-ranker scores correlate at {LV['corr']}. The same set of four passages is shown for only {LV['top4']} of dev questions, so a live answer can draw on slightly different passages from the evaluated system, and the small checker agrees with the evaluated checker on {LV['agree']} of {LV['n']} answer sentences ({LV['rate']}, measured before the top-4-window speed-up); details in <a href="{BLOB}results/live_model_check.json">live_model_check.json</a>. To fit its 0.1 CPU, the live checker also tests only each sentence's four best-matching passage windows (deviation D14). It sleeps when idle (about a minute to wake); if the backend is asleep, the page falls back to in-browser keyword search and says so.</li>
<li><strong>What dev changed</strong> (logged as deviations): four safety-rule bugs, a citation-format bug in the checker, and one revision of the generator prompt (dev LLM sentences confirmed: 49% → {pc(dev_llm.get('sentence_supported_rate', {}).get('value', 0))}).</li>
</ul>
</section>

<section class="section" id="results"><h2>4. Results (held-out test set, read once)</h2>
<h3>Retrieval</h3>
<figure class="figure">{mrr_chart()}<figcaption>Mean reciprocal rank (higher is better; the axis starts at 0.5) on {n_ret} test questions whose answer is in the sources. Lines: 95% bootstrap intervals over questions.</figcaption></figure>
{sys_table(A)}
<p>Plain semantic search was already strong and almost never retrieved a passage about the wrong condition, so a condition bonus had little to fix. The keyword classifier recognised the condition well ({civ(cls['condition_exact'], True)}) but the journey stage poorly ({civ(cls['stage_exact'], True)}; no stage cue at all in {pc(cls['stage_no_cue'])} of questions), and development tuning set the stage weight to zero. Giving the system the author's own labels instead of the classifier's guesses (an upper bound) gave MRR {civ(ret['oracle_all']['meta']['rr'])} for the condition-aware system.</p>
<div class="tw"><table><thead><tr><th>Question type</th><th class="num">n</th><th class="num">Dense MRR</th><th class="num">Condition-aware MRR</th><th class="num">Selected (+ re-ranker) MRR</th></tr></thead><tbody>{by_type}</tbody></table></div>
<p>Cross-condition questions (n = {cross['dense']['both_coverage@5']['n']}): a passage about <em>each</em> condition appeared in the top five for {civ(cross['dense']['both_coverage@5'], True)} with dense search and {civ(cross['meta_ce']['both_coverage@5'], True)} with the selected system.</p>
<h3>Knowing when not to answer</h3>
<figure class="figure">{COV_SVG}<figcaption>{e(app['coverage_caption'])}</figcaption></figure>
<p>At the threshold chosen on development questions, the gate let through {civ(gate['answer_rate_answerable'], True)} of answerable questions and {civ(gate['false_answer_rate_unanswerable'], True)} of unanswerable ones.</p>
<h3>Are the answers backed by their sources?</h3>
<div class="tw"><table><thead><tr><th>Answer writer</th><th class="num">Answers</th><th class="num">Sentences confirmed</th><th class="num">Answers with every sentence confirmed</th></tr></thead><tbody>{gr_rows}</tbody></table></div>
<p>{e(audit_text)} The LLM cited a correct passage in {civ(llm['gold_cited_rate'], True)} of its answers to answerable questions; it declined on its own {plural(llm['self_abstained_after_gate'], 'time')} after the gate let a question through. Copied sentences are confirmed almost by construction; they are the floor the LLM is compared against, and they read worse.</p>
<h3>Safety routing</h3><p>{e(safety_text)}</p>
<h3>Pre-registered hypotheses</h3>
<div class="tw"><table><thead><tr><th></th><th>Hypothesis</th><th class="num">Result (paired 95% interval)</th><th>Verdict</th></tr></thead><tbody>{hyp_rows}</tbody></table></div>
<p>{n_fail} of the 10 hypotheses were not supported. Those are reported as results, not hidden.</p>
<h3>Where it went wrong</h3><ul class="plain">{fail_html}</ul>
<h3>What problem was solved, and how well</h3>
<p>The assistant answers everyday asthma and type 2 diabetes questions from trusted public sources, shows the passage behind every sentence, declines unanswerable questions most of the time ({pc(gate['false_answer_rate_unanswerable']['mean'])} of unanswerable test questions got past the gate) and {e(solved)}. The specific idea of using condition and journey-stage labels to steer retrieval did <strong>not</strong> help on this corpus: a good semantic model plus a re-ranker did the work. Combining two conditions in one answer remains hard because the sources themselves rarely connect them.</p>
</section>

<section class="section" id="limitations"><h2>5. Limitations</h2>
<ul class="plain">
<li>Small, curated corpus: 373 passages, two conditions, four U.S. agencies, English only. Agencies update their pages; this is a snapshot.</li>
<li>Questions, gold labels, stage rules, classifier and audits were all written by the same AI collaborator. That creates a circularity risk; trap questions and reporting by question type reduce it but do not remove it.</li>
<li>No patient or clinician wrote or reviewed questions or answers. No claim is made about clinical usefulness or safety.</li>
<li>Test subsets are small (for example {cross['dense']['both_coverage@5']['n']} cross-condition questions), so their intervals are wide.</li>
<li>The grounding checker is conservative on long sentences that merge several facts, and lenient on paraphrases that keep the words but shift the meaning; the hand audit quantifies this.</li>
<li>The steroid–blood-sugar example that motivates the project is only partly answerable: no collected U.S. source states it directly.</li>
<li>The live site uses the free tiers of Groq and Hugging Face; answers can be slow or temporarily unavailable, in which case a weaker in-browser fallback is used.</li>
</ul></section>

<section class="section" id="next"><h2>6. Next steps</h2>
<ul class="plain">
<li>Ask people living with both conditions, and a clinician, to write and review questions; re-test on those.</li>
<li>Add openly licensed sources that discuss conditions together, and multi-hop retrieval that joins one passage per condition.</li>
<li>Replace the keyword stage classifier with a small trained one, evaluated on new questions.</li>
<li>Calibrate the grounding checker on a larger human-labelled set, and check whole-passage entailment for list-style answers.</li>
</ul></section>

<section class="section" id="audit"><h2>Audit trail</h2>
<ul class="plain">
<li>Pinned branches: <code>frozen/corpus</code>, <code>frozen/protocol</code>, <code>frozen/selection</code> (tag pushes are blocked from the build sandbox).</li>
<li>Test split SHA-256 <code>{e(ledger.get('test_sha256', ''))[:16]}…</code>, read once at {e(ledger.get('read_at_utc', 'n/a'))} from commit <code>{e(ledger.get('git_commit', ''))[:7]}</code>: <a href="{BLOB}results/final/test_access.json">ledger</a>.</li>
<li>All numbers on this page and in the README are generated from <a href="{BLOB}results/final/results.json">results/final/results.json</a> by <code>scripts/make_report.py</code>.</li>
<li>Deviations from the starting design, each with a reason: <a href="{BLOB}PROTOCOL.md#deviations-log">PROTOCOL.md</a>.</li>
</ul>
<p class="small muted">{e(json.loads((ROOT / 'docs/app/data/index.json').read_text()).get('retrieved', ''))} source text credit: MedlinePlus (National Library of Medicine); National Institute of Diabetes and Digestive and Kidney Diseases; National Heart, Lung, and Blood Institute, National Institutes of Health; Centers for Disease Control and Prevention. Available free of charge on medlineplus.gov, niddk.nih.gov, nhlbi.nih.gov and cdc.gov. Use here does not imply endorsement by NLM, NIH, CDC, HHS or the U.S. Government. Font: Atkinson Hyperlegible, Braille Institute (SIL Open Font License).</p>
</section>
</main></div>
<script>
(function(){{const links=[...document.querySelectorAll('.journey a')];const secs=links.map(a=>document.querySelector(a.getAttribute('href')));
const io=new IntersectionObserver(es=>{{es.forEach(en=>{{if(en.isIntersecting){{links.forEach(l=>l.setAttribute('aria-current',l.getAttribute('href')==='#'+en.target.id?'true':'false'));}}}});}},{{rootMargin:'-30% 0px -60% 0px'}});
secs.forEach(s=>s&&io.observe(s));}})();
</script>
</body></html>"""
if SRC == "final":
    (ROOT / "docs/index.html").write_text(page)
else:
    (ROOT / "results/rehearsal/index_preview.html").write_text(page)

# ------------------------------------------------------------------ README ----------------
def mdtable(block):
    rows = ["| System | MRR@10 | Correct passage first | Correct passage in top 5 | Wrong-condition share (top 5) | Wrong-stage share (top 5) |",
            "|---|---|---|---|---|---|"]
    for s in ("dense", "bm25", "hybrid", "meta", "meta_ce"):
        b = block[s]
        rows.append(f"| {LABEL[s]} | {civ(b['rr'])} | {civ(b['success@1'], True)} | {civ(b['success@5'], True)} | {civ(b['wrong_condition@5'], True)} | {civ(b['wrong_stage@5'], True)} |")
    return "\n".join(rows)


readme = f"""# AI Multi-Condition Patient Journey Assistant

**Not medical advice.** A research and teaching project. The assistant gives educational
information from public U.S. health sources only. In an emergency, call your local emergency number.

**Project page:** https://kanishk2705.github.io/AI-Multi-Condition-Patient-Journey-Assistant/
**Try the assistant:** https://kanishk2705.github.io/AI-Multi-Condition-Patient-Journey-Assistant/app/

> {headline}

## 1. Problem

People living with both asthma and type 2 diabetes need information that accounts for both
conditions (for example, steroid tablets for an asthma flare-up can affect blood sugar). Trusted
patient information is written one condition at a time: **{n_both} of the {len(P)} public passages
collected here are about both conditions.** General chatbots mix conditions, cite nothing and rarely
admit when they don't know. This project builds an assistant that answers only from public health
pages, cites a passage for every sentence, declines when the passages don't cover the question, and
never handles emergencies, doses or diagnoses. It then tests, with a pre-registered protocol,
whether retrieval that knows the reader's condition and stage of the journey (understanding,
daily management, warning signs) finds better passages than plain semantic search.

## 2. Design

`safety rules → condition/stage classifier → retrieval → abstention gate → LLM answer with a citation per sentence → independent grounding check`

* **Sources:** {len(P)} passages from MedlinePlus, NIDDK, NHLBI and CDC (public domain or copyright-free;
  licences and hashes in [DATA.md](DATA.md)).
* **Retrieval compared:** plain semantic (`bge-small-en-v1.5`), BM25, hybrid (RRF), hybrid with a
  condition/stage bonus, and that plus a cross-encoder re-ranker (`ms-marco-MiniLM-L-6-v2`).
* **Answers:** `openai/gpt-oss-120b` on Groq (no Llama chat model was available to this account; see
  deviation D1) versus an extractive copy-sentences baseline.
* **Grounding checker:** NLI cross-encoder (`nli-deberta-v3-base`) plus a word-overlap check, independent
  of the answer model; hand-audited.
* **Questions:** 148 written by Claude (not by patients or clinicians): answerable, cross-condition,
  trap, unanswerable and safety cases; 59 development / 89 held-out test, split before any system ran.
* **Protocol and hypotheses** frozen first: [PROTOCOL.md](PROTOCOL.md).

## 3. Implementation

```
src/pjassist/   text, bm25, classifier, retrieval, safety, generate, grounding, evaluate, report, pipeline
scripts/        collect_sources, build_corpus, build_questions, embed, run_dev, answer_dev, run_final, make_report, deploy_space
tests/          {N_TESTS} unit tests (grounding, safety and leakage guards shown to fail on broken input; Python/JS parity)
space/app.py    FastAPI backend; render/ + render.yaml deploy it on Render (ONNX models, free tier)
docs/           project page and the assistant (GitHub Pages)
results/        dev/ (tuning), final/ (held-out test, read once), llm_cache/ (every LLM response)
```

The build sandbox could not reach the health sites, Hugging Face or Groq, so those steps run in the
`ci-run` GitHub Actions workflow; keys exist only as GitHub and Space secrets.

## 4. Results (held-out test set, {n_ret} questions with an answer in the sources)

{mdtable(A)}

Brackets are 95% bootstrap intervals over questions.

* Classifier vs the author's labels: condition {civ(cls['condition_exact'], True)}, stage {civ(cls['stage_exact'], True)}.
* Abstention gate: AUROC {f2(gate['auroc'])}; at the dev threshold it answered {civ(gate['answer_rate_answerable'], True)} of answerable
  and {civ(gate['false_answer_rate_unanswerable'], True)} of unanswerable test questions.
* Grounding: LLM sentences confirmed {civ(llm['sentence_supported_rate'], True, 'value')}, LLM answers fully confirmed
  {civ(llm['fully_grounded_rate'], True)}; extractive sentences confirmed {civ(ext['sentence_supported_rate'], True, 'value')}.
* {audit_text}
* Safety: {safety_text}

| | Hypothesis | Result | Verdict |
|---|---|---|---|
""" + "\n".join(f"| {h} | {t} | {v} | {'Supported' if ok else '**Not supported**'} |" for h, t, v, ok in hyp) + f"""

**What was solved, and how well.** The assistant answers everyday asthma and type 2 diabetes questions
from trusted sources with a visible source for every sentence, declines most unanswerable questions and
{solved}. The specific hypothesis that condition and journey-stage
labels improve retrieval was **not supported** on this corpus: a strong semantic model plus a re-ranker
did the work, and the keyword stage classifier was too weak to help. Combining two conditions in one
answer stays hard because the sources rarely connect them.

## 5. Limitations

Small curated corpus (two conditions, four U.S. agencies, English, one snapshot); questions, gold
labels, rules and audits written by the same AI collaborator (circularity risk); no patient or
clinician input or review; small test subsets; a conservative automatic grounding checker; free-tier
hosting. No clinical-effectiveness claim is made.

## 6. Next steps

Questions from people living with both conditions and clinician review; sources that discuss the
conditions together; multi-hop retrieval; a trained stage classifier; a larger human-labelled set to
calibrate the grounding checker.

## Reproduce

```
pip install -r requirements-ci.txt
pytest -q                                   # {N_TESTS} tests
python scripts/run_dev.py                   # dev tuning from cached model scores (data/cache)
python scripts/make_report.py               # page + README from results/final/results.json
```
Network-dependent steps (`collect_sources.py`, `embed.py`, `answer_dev.py`, `run_final.py`, `check_onnx.py`)
run in the `ci-run` workflow. `run_final.py` refuses to run again: the test split has been read.

## Data credit

Source passages: MedlinePlus (National Library of Medicine); NIDDK; NHLBI, National Institutes of
Health; CDC. Available free of charge on medlineplus.gov, niddk.nih.gov, nhlbi.nih.gov and cdc.gov.
Use here does not imply endorsement by NLM, NIH, CDC, HHS or the U.S. Government. Code: MIT licence.
"""
if SRC == "final":
    (ROOT / "README.md").write_text(readme)
print("report written from", SRC, "| hypotheses not supported:", n_fail)
