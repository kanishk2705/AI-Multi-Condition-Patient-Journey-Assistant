# Data, Source, Safety, and Privacy Policy

## Approved source types

Prefer public information published by government health agencies and recognized public-health organizations. Initial candidates include:
- World Health Organization (WHO)
- MedlinePlus / U.S. National Library of Medicine
- NHS health information pages, where relevant and clear about the healthcare system described

A source is not automatically approved merely because it appears in a search result. Before ingestion, check that the specific page is relevant, current enough for its topic, reachable, attributable, and usable under the source site's terms and access rules. Record the date checked and any page-level restrictions.

## Data minimization and attribution

- Keep the source title, publisher, exact canonical URL, retrieved/checked date, and relevant dates/metadata.
- Store only the content excerpts needed to build the prototype and evaluation.
- Do not commit downloaded full webpages, large source text dumps, or private or identifiable health information to Git.
- Keep generated/local source material in git-ignored directories.
- Show the source title and link on each answer. Never imply the project owns, authored, or clinically endorses source content.
- If terms, robots policy, or reuse restrictions are unclear, do not bulk scrape the website. Use permitted access or manually prepare brief notes with links and attribution.

## Safety boundaries

The system provides general health education and information navigation only. It is not a medical device and is not validated for clinical use.

It must not:
- diagnose a disease or estimate personal disease risk;
- triage symptoms or decide whether a condition is an emergency;
- prescribe or recommend starting, stopping, or changing medication or dosage;
- provide individualized treatment plans;
- claim that retrieved information replaces a clinician.

For urgent/emergency symptoms or requests, the system must not continue a routine answer flow; it must display a short instruction to seek urgent help through local emergency services or a qualified healthcare professional. Avoid hard-coding a country-specific emergency number unless the user's location and number have been verified. This prototype does not need to determine which emergency category a user is in.

When evidence is missing, not relevant to the chosen condition/stage, or contradictory, the assistant should state that the approved collection cannot answer reliably and direct the user to a qualified healthcare professional.

## Privacy

- Do not request or accept names, addresses, phone numbers, identity numbers, medical-record files, or other identifying data.
- Do not persist user questions or assistant responses.
- Do not log prompt text, retrieved passages, or potentially sensitive user-entered data. Log only technical errors without query contents.
- Use provider API keys only via environment variables or hosting secrets. Never commit them.
- Explain to visitors that their question is processed to generate a response and is not intentionally stored by this application. Do not make broader provider-level data-retention claims that have not been verified.

## Limitations to display

The portfolio and UI must state that the system uses a small curated information collection, may omit relevant information, can generate errors, and is intended for educational demonstration only—not diagnosis, emergency response, or personal treatment advice.
