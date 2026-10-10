# Data, Source, Safety, and Privacy Policy

## Approved source policy used for Phase 2

The first ingestion set is deliberately narrow and restricted to selected MedlinePlus content types. The official [MedlinePlus content-use policy](https://medlineplus.gov/about/using/usingcontent/) states that health-topic page summaries and medical-test information are public domain. The implementation therefore supports only two extraction modes:

- **medlineplus_topic_summary:** captures only the section under the Summary heading on a health-topic page. The validator restricts this mode to MedlinePlus topic paths, excluding lab-test and Medical Encyclopedia paths.
- **medlineplus_medical_test:** captures the text body of a MedlinePlus medical-test page, and the validator requires the URL path to start with /lab-tests/.

The extractor must fail closed if the Summary heading cannot be found or if too little text is extracted. Do not ingest MedlinePlus Medical Encyclopedia articles, drug monographs, images, videos, or separately copyrighted content under these modes.

## Sources not yet approved

- **WHO:** content remains link-only/deferred unless the exact page's licence permits the intended reuse or written permission is obtained. WHO's [website terms](https://www.who.int/about/policies/terms-of-use) note copyright protection and require permission for substantial reproduction or broader uses.
- **NHS:** candidate only. The NHS terms state that most website text is released under the Open Government Licence, subject to exceptions. Each page still requires review for identified third-party content and any excluded assets or material before being marked approved. See [NHS terms and conditions](https://www.nhs.uk/our-policies/terms-and-conditions/).

A source is not approved merely because it appears in a search result. Keep its canonical URL, title, publisher, checked date, extraction mode, reuse basis, and stage metadata in data/source_manifest.csv. The ingestion tool fetches only rows with status=approved.

## Access policy

- HTTPS only and only approved public domains.
- Check the source host's robots.txt before requesting a page; if the file cannot be retrieved or the policy disallows access, fail closed. An HTTP 404/410 for robots.txt is treated as no published rules.
- Use a descriptive User-Agent and conservative timeout, response-size limit, and inter-request delay.
- Reject redirects to other domains, non-HTML responses, and oversized responses.
- The command is dry-run by default. Network requests require an explicit --fetch-approved option.

## Data minimization and attribution

- Keep source provenance on every chunk: source ID, title, publisher, URL, source-content hash, chunk index, extraction mode, and fetch time.
- Keep generated text chunks and ingestion reports locally under data/local/; these outputs are git-ignored and must not be committed.
- Do not commit full copied pages, raw source dumps, personal health information, API keys, or user questions.
- Display the source title and canonical link with any future generated answer.
- The project must not imply that publishers endorse this project.

## Safety boundaries

The system provides general health education and information navigation only. It is not a medical device and is not validated for clinical use.

It must not:
- diagnose a disease or estimate personal disease risk;
- triage symptoms or decide whether a condition is an emergency;
- prescribe or recommend starting, stopping, or changing medication or dosage;
- provide individualized treatment plans;
- claim that retrieved information replaces a clinician.

The initial source set focuses on topic summaries and medical-test education, not emergency protocols. If future sources introduce emergency or treatment content, the source and chunk must be reviewed for safety boundaries before use. For urgent/emergency queries, the eventual interface must not generate routine education in place of seeking urgent help; it should advise contacting local emergency services or an appropriate healthcare professional without hard-coding an unverified local number.

When evidence is missing, irrelevant to the selected condition/stage, or contradictory, the assistant should state that the approved collection cannot answer reliably and direct the user to a qualified healthcare professional.

## Privacy

- Do not ask for names, addresses, phone numbers, identity numbers, medical-record files, or other identifying data.
- Do not persist user questions or assistant responses.
- Do not log prompt text, retrieved passages, or potentially sensitive user-entered data.
- Use provider API keys only via environment variables or hosting secrets. Never commit them.
- Explain that a question is processed to produce a response and is not intentionally stored by this app; do not make unverified claims about third-party model-provider retention.

## Limitations to display

The UI and portfolio must state that the system uses a small curated collection, may omit relevant information, can generate errors, and is intended for educational demonstration only—not diagnosis, emergency response, or personal treatment advice.
