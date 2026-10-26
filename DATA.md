# Data: sources, licences, collection and labelling

**Not medical advice.** Everything here is public patient-education text collected for a
research and teaching project. It is not reviewed by clinicians for this use.

## 1. What the corpus is

373 passages (`data/corpus/passages.jsonl`) cut from 30 public web pages and 20 MedlinePlus
health-topic summaries, all collected on **2026-10-24 (UTC)**. Two conditions are covered,
asthma and type 2 diabetes, plus a few general topics that matter to both (flu shot, smoking,
allergy, steroids).

| Source | Pages / topics | Passages |
|---|---|---|
| MedlinePlus (U.S. National Library of Medicine) | 20 health-topic summaries from the daily XML file | 103 |
| NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases) | 9 pages | 151 |
| CDC (Centers for Disease Control and Prevention) | 13 pages | 81 |
| NHLBI (National Heart, Lung, and Blood Institute) | 8 pages | 38 |

By condition label: diabetes 275, asthma 83, general 15. By journey stage: understand 157,
manage 191, warning 25. **No passage is labelled with both conditions**: public patient
information is written one condition at a time, which is the problem this project is about.

## 2. Licences and terms (checked 2026-10-24)

Only sources whose terms allow reuse were collected. Each passage carries its licence string.

| Source | Terms found | What this project does |
|---|---|---|
| **MedlinePlus** ([using content](https://medlineplus.gov/about/using/usingcontent/), [XML files](https://medlineplus.gov/xml.html)) | Health-topic summaries are public domain; NLM invites download and use of the XML files and asks for credit "Source: MedlinePlus, National Library of Medicine". A.D.A.M. encyclopedia articles and ASHP drug monographs are copyrighted and licensed to MedlinePlus. | Uses only the health-topic `full-summary` text. No encyclopedia or drug-monograph text is used. Credit given on every passage. |
| **NIDDK** ([copyright](https://www.niddk.nih.gov/copyright)) | Most information is copyright-free and may be downloaded and reproduced; credit NIDDK; content must not imply endorsement or be used to recommend specific medical advice; co-sponsored documents and third-party graphics may be restricted. | Text only, unchanged, credited. Answers carry a "not medical advice" notice. No images. |
| **NHLBI** ([public-domain statement](https://www.nhlbi.nih.gov/about/contact/trademark-branding-and-logo)) | Content on nhlbi.nih.gov is public domain unless noted; do not change publications; no endorsement or advertising use; credit "National Heart, Lung, and Blood Institute; National Institutes of Health". | Text only, unchanged, credited. Glossary pop-ups and navigation removed (not part of the article text). |
| **CDC** ([agency materials policy](https://www.cdc.gov/other/agencymaterials.html)) | Most content is public domain; credit CDC/HHS; show a disclaimer that use does not imply endorsement by CDC, HHS or the U.S. Government; do not change substantive content; say the material is available free on cdc.gov; contractor or third-party material may be restricted. | Text only, unchanged, credited, with the no-endorsement statement in the README, on the website and in each passage's licence field. |

General NIH guidance ([NIH FAQ](https://www.nih.gov/about-nih/frequently-asked-questions)) says the
same: most NIH website content is public domain, with marked exceptions. None of the collected text
carried a copyright notice. Because all four sources allow redistribution of this text, the
passages are shipped in this repository and on the website. Agency names and logos are not used
as branding; the project is not affiliated with or endorsed by NLM, NIH, NIDDK, NHLBI, CDC or HHS.

**Not used:** NHS, Diabetes UK, Asthma + Lung UK and similar pages. They state the steroid–blood
glucose link more directly than any U.S. source found here, but they are not openly licensed.

## 3. How it was collected

The build sandbox could not reach any of these hosts (the proxy refused the connection), so
collection ran on GitHub Actions in this repository (`scripts/collect_sources.py`, via the
`ci-run` workflow). One request at a time, 2 seconds apart, with a User-Agent naming this
repository. The workflow log and outputs are on branch `ci/run`.

Every downloaded file is listed in `data/corpus/source_manifest.json` with its URL, final
URL after redirects, HTTP status, retrieval time and SHA-256. Raw HTML is not committed to
`main` (it is page chrome around the text); it is on branch `ci/run` under `data/raw/`.

The first collection pass found three wrong URLs: two pages had moved (NIDDK hypoglycaemia,
CDC sick days) and one redirected to a page already collected. They were replaced with the
current URLs found in the collected pages' own links, and everything was downloaded again in
one pass so that all files share the same retrieval date.

## 4. From pages to passages (`scripts/build_corpus.py`)

1. **Main content only:** `article.dk-content` (NIDDK) or `<main>` (CDC, NHLBI); MedlinePlus
   uses the XML `full-summary`. Navigation, glossary pop-ups, footers, "last updated" lines and
   "in this section" menus are removed.
2. **Sections:** every paragraph or list item belongs to the nearest preceding `h2`/`h3`. Sections
   about research studies, clinical trials, the artificial pancreas, islet transplants,
   bronchial thermoplasty and COVID-19 are dropped as outside a patient's everyday journey.
3. **Passages:** paragraphs are packed into passages of at most about 170 words; a short tail is
   merged into the previous passage. Exact duplicate texts (CDC repeats its key points) are kept once.
4. **Verbatim text:** text is unchanged except whitespace normalisation; list items get a full stop
   if they have none, and table rows are written as "header: cell" pairs so a value keeps its label.

## 5. Metadata labels (rule-based, written before any question)

* **Condition:** the page's condition (from its title or URL), plus any condition the passage text
  itself mentions (`asthma`; `diabet…`, `blood sugar/glucose`, `hypo/hyperglyc…`, `insulin`,
  `prediabet…`). General pages (flu shot, smoking, steroids, allergy) get only what their text mentions.
* **Journey stage**, decided from the section heading, then its parent heading, then the page title:
  * **warning**: recognising and responding to acute problems: warning signs, emergencies,
    attacks, treating a low or high, when to call a provider or go to the ER, action-plan alert zones;
  * **manage**: day-to-day self-management: treatment and medicines, food, activity, monitoring,
    triggers, vaccines, sick-day planning, preventing complications;
  * **understand**: understanding the condition and its diagnosis: what it is, causes, risk factors,
    symptoms in general, tests and what results mean.

**Label audit.** Two random samples of 40 passages were checked by hand (by Claude, the same
author that wrote the rules, so this is a consistency check, not an independent one).

| Sample | Condition labels agreed | Stage labels agreed | Notes |
|---|---|---|---|
| 1 (seed 20261024), before fixes | 40/40 | 36/39 | 1 sampled passage was navigation text; one stage error came from a rule bug ("When should I *get* the flu shot?" matched a warning rule) |
| 2 (seed 777), final corpus | 40/40 | 37/40 | disagreements are borderline understand/warning cases |

After sample 1 the rule bug, the navigation leak and a table-header problem were fixed; sample 2
was drawn afterwards from the final corpus. Stage labels are therefore roughly 90% consistent
with a careful reading, and the retrieval results must be read with that noise in mind.

## 6. Questions

The evaluation questions (`data/questions/`) were written by Claude for this project. They are
not from patients or clinicians. See PROTOCOL.md section 4.

## 7. Attribution text used on the website

> Source passages: MedlinePlus (National Library of Medicine); National Institute of Diabetes and
> Digestive and Kidney Diseases (NIDDK); National Heart, Lung, and Blood Institute (NHLBI), National
> Institutes of Health; Centers for Disease Control and Prevention (CDC). These materials are
> available free of charge on medlineplus.gov, niddk.nih.gov, nhlbi.nih.gov and cdc.gov. Their use
> here does not imply endorsement by NLM, NIH, CDC, HHS or the U.S. Government. Text retrieved
> 2026-10-24 and may since have been updated at the source.
