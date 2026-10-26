"""Turn downloaded source pages into labelled passages: data/corpus/passages.jsonl.

Each passage = one section (or part of a long section) of one page, verbatim text, with:
  id, source, title, section, url, conditions, stage, license, retrieved_utc, page_sha256
Labelling rules (PROTOCOL.md section 3) are deterministic and use only the page and its
headings; no question or gold label is used. Text is kept verbatim except: whitespace is
normalised, and list items get a full stop if they have none (so they split into sentences).
"""
from __future__ import annotations
import hashlib, json, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup, NavigableString, Tag

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "corpus"
MAX_W, MIN_W = 170, 25

LICENSES = {
    "MedlinePlus": "Public domain (NLM health-topic summary). Credit: MedlinePlus, National Library of Medicine.",
    "NIDDK": "Copyright-free per NIDDK (niddk.nih.gov/copyright). Credit: National Institute of Diabetes and Digestive and Kidney Diseases.",
    "NHLBI": "Public domain unless noted (NHLBI). Credit: National Heart, Lung, and Blood Institute; National Institutes of Health.",
    "CDC": "Public domain (CDC agency materials policy). Credit: Centers for Disease Control and Prevention. Use does not imply endorsement by CDC, HHS or the U.S. Government.",
}
SKIP_HEAD = re.compile(r"language switcher|book traversal|on this page|clinical trial|references|resources|related|"
                       r"sources|print|share|reviewed|last reviewed|content source|more information|"
                       r"what have we learned|find a|spanish|español|for more|stay informed|get email|"
                       r"recursos|key resources|related pages|health professionals|feedback", re.I)
BOILER = re.compile(r"^(Home|English|Español|Espanol|In this section:?|On this page:?|Print|Share|Related)\.?$|^In this section", re.I)
NAV = re.compile(r"< Back To|^Health Topics\.?$|^MORE INFORMATION\.?$|^/\.?$|"
                 r"^(Asthma|What Is Asthma\??|Symptoms|Asthma Attack|Causes and Triggers|Diagnosis|"
                 r"Treatment and Action Plan|Managing Asthma|Asthma in Children|Living With)\.?$|"
                 r"This page was last updated|^Last Reviewed", re.I)
SKIP_TEXT = re.compile(r"This content is provided as a service of the National Institute|NIDDK would like to thank|"
                       r"Javascript is disabled|Sign up for", re.I)

ASTHMA_T = re.compile(r"\basthma", re.I)
DIAB_T = re.compile(r"\bdiabet|\bblood (sugar|glucose)\b|\bhypoglyc|\bhyperglyc|\bprediabet|\binsulin\b", re.I)


def page_base(source: str, title: str, url: str) -> set:
    t = (title + " " + url).lower()
    if "asthma" in t:
        return {"asthma"}
    if any(w in t for w in ("diabet", "blood sugar", "glucose", "hypoglyc", "hyperglyc", "a1c")):
        return {"diabetes"}
    return set()


def label_conditions(base: set, text: str) -> list:
    c = set(base)
    if ASTHMA_T.search(text):
        c.add("asthma")
    if DIAB_T.search(text):
        c.add("diabetes")
    return sorted(c)


WARN_H = re.compile(r"when (to|should i) (call|go to|see|seek)|get (help|care|medical)|emergency|\bER\b|warning|severe|attack|urgent|"
                    r"unawareness|call 911|treat(ing)? (low|high)|\bketo|getting worse|medical alert|15-15", re.I)
MANAGE_H = re.compile(r"treat|manag|prevent|protect|control|medicine|diet|food|eat|drink|quit|check|track|"
                      r"\bplan\b|vaccin|flu shot|physical activity|weight|sleep|mental health|monitor|lifestyle|"
                      r"trigger|insulin|pen\b|pump|syringe|inhaler|injector|surgery|lower my chances|"
                      r"health care team|tips|special situations|ABCs|is for", re.I)
UNDERSTAND_H = re.compile(r"what (is|are|causes|happens|else can cause)|\bcauses?\b|risk|diagnos|\btests?\b|"
                          r"types of|how common|who (gets|is|should be tested)|overview|key points|about|"
                          r"symptoms|why|how (does|do) .* (cause|affect)|health problems|side effects|"
                          r"what (health|eye|other) problems|facts|at a glance|background|what to know", re.I)
WARN_PAGE = re.compile(r"attack|emergency|hypoglyc|low-blood|low blood|hyperglyc|high blood", re.I)
UNDERSTAND_PAGE = re.compile(r"what is|symptoms|causes|diagnosis|about", re.I)
OFF_SCOPE = re.compile(r"research|stud(y|ies)|participate|right for you|fact sheets|checklist for home visitors|"
                       r"artificial pancreas|islet transplant|bronchial thermoplasty|covid", re.I)


def label_stage(title: str, section: str, url: str, parent: str = "") -> str:
    """First match wins: own heading (warning, manage, understand), then parent h2, then page."""
    for h in ([section] if section else []) + ([parent] if parent else []):
        if WARN_H.search(h):
            return "warning"
        if re.search(r"symptom", h, re.I) and WARN_PAGE.search(title + " " + url):
            return "warning"
        if MANAGE_H.search(h):
            return "manage"
        if UNDERSTAND_H.search(h):
            return "understand"
    if WARN_PAGE.search(title) and re.search(r"attack", title, re.I):
        return "warning"
    if UNDERSTAND_PAGE.search(title):
        return "understand"
    return "manage"


PARENT: dict = {}


def sections_from_html(root: Tag) -> list[tuple[str, list[str]]]:
    """Assign every leaf paragraph / list item to the nearest preceding h2/h3 inside root."""
    for bad in root.find_all(["nav", "script", "style", "aside", "footer", "form", "button", "figure", "noscript"]):
        bad.decompose()
    for bad in root.select(".glossarytext, .usa-modal, .sr-only, .visually-hidden, .book-navigation, "
                           ".language-switcher, .breadcrumb, .usa-breadcrumb, .dk-box-related, .dk-sidebar"):
        bad.decompose()
    order, secs = [], {}
    for node in root.find_all(["p", "li", "dd", "tr"]):
        if node.find_parent("tr") and node.name != "tr":
            continue
        own = None
        if node.name == "li" and node.find(["ul", "ol"]):
            # a list item that heads a nested list: keep its own lead-in text (e.g. a test's name)
            import copy
            c = copy.copy(node)
            for sub in c.find_all(["ul", "ol"]):
                sub.decompose()
            own = re.sub(r"\s+", " ", c.get_text(" ")).strip()
        elif node.name != "tr" and node.find(["p", "li", "dd"]):
            continue
        h = node.find_previous(["h2", "h3"])
        head = re.sub(r"\s+", " ", h.get_text(" ")).strip() if h and root in h.parents else ""
        if h is not None and h.name == "h3" and root in h.parents:
            h2 = h.find_previous("h2")
            if h2 is not None and root in h2.parents:
                PARENT[head] = re.sub(r"\s+", " ", h2.get_text(" ")).strip()
        if node.name == "tr":  # table row: keep the row header with its cells
            cells = [re.sub(r"\s+", " ", c.get_text(" ")).strip() for c in node.find_all(["th", "td"])]
            t = ": ".join(c for c in cells if c)
            hdr = node.find_parent("table").find("tr") if node.find_parent("table") else None
            if hdr is node or not t:
                continue
            if hdr is not None:
                cols = [re.sub(r"\s+", " ", c.get_text(" ")).strip() for c in hdr.find_all(["th", "td"])]
                if len(cols) == len(cells) and len(cols) > 1:
                    t = "; ".join(f"{h}: {c}" if h else c for h, c in zip(cols, cells))
            t = t.rstrip(".") + "."
        else:
            t = own if own is not None else re.sub(r"\s+", " ", node.get_text(" ")).strip()
        t = re.sub(r"\s+([.,;:!?)])", r"\1", t)
        if not t or SKIP_TEXT.search(t) or BOILER.match(t) or NAV.search(t):
            continue
        if head == "" and node.name == "li" and t.endswith("?"):
            continue  # "In this section" question menus before the first heading
        if node.name == "li" and not re.search(r"[.!?:]$", t):
            t += "."
        if head not in secs:
            secs[head] = []; order.append(head)
        secs[head].append(t)
    heads = set(order)
    return [(h, [t for t in secs[h] if t.rstrip(".") not in heads]) for h in order]


def dedupe(paras: list[str]) -> list[str]:
    seen, out = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def chunk(paras: list[str]) -> list[str]:
    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > MAX_W:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(p); n += w
    if cur:
        if chunks and n < MIN_W and len(chunks[-1].split()) + n <= MAX_W + 40:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "data" / "sources.json").read_text())
    manifest = json.loads((RAW / "manifest.json").read_text())
    passages = []

    def add(source, title, url, section, text_chunks, retrieved, sha):
        base = page_base(source, title, url)
        for k, t in enumerate(text_chunks):
            if len(t.split()) < 8:
                continue
            pid = f"{source.lower()}-{re.sub(r'[^a-z0-9]+', '-', (title + '-' + (section or 'summary')).lower()).strip('-')[:70]}-{k}"
            passages.append({"id": pid, "source": source, "title": title, "section": section or None, "url": url,
                             "text": t, "conditions": label_conditions(base, t),
                             "stage": label_stage(title, section, url, PARENT.get(section or "", "")),
                             "parent_section": PARENT.get(section or "") , "license": LICENSES[source],
                             "retrieved_utc": retrieved, "page_sha256": sha})

    # MedlinePlus XML
    mx = next(m for m in manifest if m["file"] == "medlineplus_topics.xml")
    root = ET.parse(RAW / "medlineplus_topics.xml").getroot()
    want = set(cfg["medlineplus_topics"])
    for ht in root.findall("health-topic"):
        if ht.get("language") != "English" or ht.get("title") not in want:
            continue
        soup = BeautifulSoup("<div>" + (ht.find("full-summary").text or "") + "</div>", "lxml").div
        for sec, paras in sections_from_html(soup):
            if sec and OFF_SCOPE.search(sec):
                continue
            add("MedlinePlus", ht.get("title"), ht.get("url"), sec, chunk(dedupe(paras)), mx["retrieved_utc"], mx["sha256"])

    for m in manifest:
        if m["file"].endswith(".xml") or m.get("status") != 200:
            continue
        html = (RAW / m["file"]).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        host = m["final_url"]
        source = "NIDDK" if "niddk" in host else "NHLBI" if "nhlbi" in host else "CDC" if "cdc.gov" in host else None
        if not source:
            continue
        main_el = soup.select_one("article.dk-content") or soup.find("main") or soup.body
        h1 = main_el.find("h1") or soup.find("h1")
        title = re.sub(r"\s+", " ", h1.get_text(" ")).strip() if h1 else soup.title.get_text()
        title = re.sub(r"^Asthma\s+(?=\S)", "Asthma: ", title) if source == "NHLBI" and title != "Asthma" else title
        if h1:
            h1.decompose()
        for sec, paras in sections_from_html(main_el):
            if sec and (SKIP_HEAD.search(sec) or OFF_SCOPE.search(sec)):
                continue
            add(source, title, m["final_url"].split("?")[0], sec, chunk(dedupe(paras)), m["retrieved_utc"], m["sha256"])

    # drop exact-duplicate texts (CDC pages repeat "key points")
    seen, uniq = set(), []
    for p in passages:
        h = hashlib.sha1(p["text"].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h); uniq.append(p)
    ids = {}
    for p in uniq:  # make ids unique and stable
        b = p["id"]; ids[b] = ids.get(b, -1) + 1
        if ids[b]:
            p["id"] = f"{b}-{ids[b]}"
    with open(OUT / "passages.jsonl", "w") as f:
        for p in uniq:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(uniq), "passages;", Counter(p["source"] for p in uniq), Counter(p["stage"] for p in uniq),
          Counter("+".join(p["conditions"]) or "general" for p in uniq))


if __name__ == "__main__":
    main()
