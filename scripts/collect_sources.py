"""Download source documents (runs on CI; the build sandbox has no access to these hosts).

Writes data/raw/<slug>.{html,xml}, data/raw/manifest.json (url, final url, status,
retrieval time UTC, sha256, bytes) and data/raw/links.json (same-site links per page,
used only to correct URLs that moved). Polite: one request at a time, 2 s apart.
"""
from __future__ import annotations
import hashlib, json, re, sys, time, io, zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "Mozilla/5.0 (educational research; github.com/kanishk2705/AI-Multi-Condition-Patient-Journey-Assistant)"}

def slug(url: str) -> str:
    p = urlparse(url)
    return re.sub(r"[^a-z0-9]+", "-", (p.netloc.replace("www.", "") + p.path).lower()).strip("-")[:120]

def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "data" / "sources.json").read_text())
    manifest, links = [], {}
    # MedlinePlus health-topic XML (public-domain summaries; NLM invites download and reuse)
    idx = requests.get("https://medlineplus.gov/xml.html", headers=UA, timeout=60)
    m = re.search(r'href="(https://medlineplus\.gov/xml/mplus_topics_\d{4}-\d{2}-\d{2}\.xml)"', idx.text)
    print("medlineplus index", idx.status_code, m.group(1) if m else None)
    if m:
        r = requests.get(m.group(1), headers=UA, timeout=300)
        (RAW / "medlineplus_topics.xml").write_bytes(r.content)
        manifest.append(dict(url=m.group(1), final_url=r.url, status=r.status_code, file="medlineplus_topics.xml",
                             retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content)))
    for url in cfg["pages"]:
        time.sleep(2)
        try:
            r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
        except Exception as e:  # record and continue
            manifest.append(dict(url=url, status=None, error=repr(e))); print("ERR", url, e); continue
        name = slug(url) + ".html"
        (RAW / name).write_bytes(r.content)
        manifest.append(dict(url=url, final_url=r.url, status=r.status_code, file=name,
                             retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             sha256=hashlib.sha256(r.content).hexdigest(), bytes=len(r.content)))
        print(r.status_code, len(r.content), url, "->", r.url)
        try:
            soup = BeautifulSoup(r.text, "html.parser")
            host = urlparse(r.url).netloc
            links[url] = sorted({urljoin(r.url, a["href"]).split("#")[0] for a in soup.find_all("a", href=True)
                                 if urlparse(urljoin(r.url, a["href"])).netloc == host})
        except Exception:
            pass
    (RAW / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (RAW / "links.json").write_text(json.dumps(links, indent=1))

if __name__ == "__main__":
    main()
