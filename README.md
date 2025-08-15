# Illinois Public Sentiment Map (Counties & AHJs)

**Live demo:** https://sophie-wzr.github.io/il-sentiment-map/

A single-page, client-side map that visualizes public sentiment toward solar development across Illinois by **County** and **AHJ (Authority Having Jurisdiction)**. It fuses project records with GeoJSON boundaries, computes a reproducible sentiment score per AHJ, and surfaces project details and sources for outreach and siting work.

---

## Repository Contents

- `index.html` — the web app.
- `illinois_counties.geojson` — county polygons.
- `illinois_ahj.geojson` — AHJ polygons.
- `sentiment.csv` — computed sentiment per AHJ/County (consumed by the app).
- `build_sentiment.py` — script to compute `sentiment.csv` from the source sheet.
- `Ameren Muni Public Sentiment Research - Sentiment Research.csv` — source project data.

> AHJ = “Authority Having Jurisdiction” (cities, villages, some counties, etc.).

---

## Features

- **Bivariate view** — toggle **Counties** and/or **AHJs**.
- **Color scale** — Positive (green), Mixed (orange), Negative (red), No data (gray).
- **Search** — fuzzy search by county or AHJ.
- **Smart sidebar**
  - Selection summary with **moratorium** status and numeric **sentiment** (Positive/Mixed/Negative pill).
  - Jurisdiction info: AHJ type, utility, comprehensive plan, zoning map link, past solar project flag.
  - Ordered **project list** showing status/year, council votes, zoning (highlighted), size, concerns, parcel, and **source links**.
- **Legend modes** — overall sentiment distribution; switches to **project distribution** when a place is selected.
- **Method transparency** — a “How is the score calculated?” link summarizing the scoring.

---

## Scoring (Short Version)

For each project, compute an **approval affinity** by blending:
- a **vote-derived signal** (parsed council tally; if missing, a *shrunk* Planning Commission/ZBA recommendation), and  
- **status** (Approved = 1, Denied = 0, else 0.5).

Combine that with a **concern score** that decays with counts of citizen/official concerns (officials weighted more). Add **+3** when zoning indicates **Agriculture (AG)**. The per-project score is: project_score = 100 × (0.7 × approval_affinity + 0.3 × concern_score)


Per-project **weight** multiplies:
- a smooth **size band** peaking at **5–20 MW** (boost 1.6, gated by approval),
- **recency** (≈3-year half-life with a modest ≤2-year boost), and
- a small **approval tilt**.

At the **AHJ level**, take the weighted mean and apply bounded nudges:
- **experience count bonus** (more projects → diminishing boost),
- a **very-recent swing** for clear approvals/denials (this year + last year), and
- **+2** if any past solar project.

AHJs with no history default to **50** (neutral). Final scores are clipped to **0–100**.
