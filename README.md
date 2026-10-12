# Crime Detection & Fatality Prediction System

DBMS + AI mini project. One app, two roles: **Police** (operational) and **Journalist** (anonymised, public view).

## Setup (2 minutes)
```bash
pip install -r requirements.txt
python generate_data.py      # creates crime.db (SQLite) with 6000 synthetic crimes
python train_models.py       # trains both models into ./models
streamlit run app.py
```
Logins: `police1 / police123`, `journalist1 / press123`, `admin / admin123`

## What it does
| Feature | Role | Tech |
|---|---|---|
| **Live Operations Center** (auto-refresh feed, KPIs, map, 5-min chart) | Police | `st.fragment(run_every=...)`, `realtime.py` |
| **Spike detection** (area suddenly above its normal rate) | Police, Journalist | Poisson tail test vs historical baseline |
| **High fatality-risk alerts** (popup + banner at p >= 50%) | Police | RandomForest probability on every new case |
| **Live Newsroom** (anonymised feed + "developing situations") | Journalist | View-based, unverified-flagged |
| **Incident simulator** (demo mode) + **one-click model retrain** | Police/Admin | `simulate_incident()`, `retrain_models()` |
| Hotspot detection (map + table) | Police | DBSCAN (haversine) on coordinates |
| High-risk area/time forecast | Police, Journalist | GradientBoosting classifier |
| Per-incident fatality prediction | Police | RandomForest pipeline |
| Register case + auto prediction | Police | SQL INSERT + trigger audit log |
| Case records, filters, status update, CSV | Police | SQL views |
| Trends, Story Finder, risk heatmap | Journalist | Anonymised view `v_public_summary` |
| DBMS Console (JOIN, GROUP BY, subquery, window fn) | Admin/Police | Read-only SQL |

## DBMS highlights (for viva)
- 8 tables in 3NF: `roles, users, locations, crime_types, crimes, victims, predictions, audit_log`
- Primary/foreign keys, CHECK constraints, UNIQUE constraints, indexes
- **Views**: `v_police_cases` (full detail), `v_public_summary` (hides weapon, status, description, exact coordinates), `v_area_stats`
- **Triggers**: audit log on INSERT/DELETE of crimes; a Fatal victim auto-marks the crime fatal
- Role-based access: users -> roles, each role sees different pages and views

## Team split
1. **Database**: `schema.sql`, `generate_data.py`, ER diagram, queries in the DBMS Console
2. **Crime ML**: hotspot page + `train_risk` in `train_models.py`
3. **Fatality ML**: `train_fatality`, metrics, feature importance
4. **App/UI**: `app.py`, login, integration, PPT and report

## Using a real dataset
Keep the schema. Write a loader that maps your CSV (NCRB / Kaggle) into `locations`, `crime_types` and `crimes` (plus `victims` if available), then rerun `train_models.py`. Columns the models expect: crime type, weapon, city, area, lat/lon, timestamp, victim count, fatal flag.

## Honest notes (mention in viva)
- Data is **synthetic**, so metrics reflect patterns we generated. Only about 11% of cases are fatal, so at the 0.5 cut-off recall is low (about 36%) while precision is high (about 78%). The app therefore shows the probability and a LOW / MEDIUM (>=20%) / HIGH (>=50%) band rather than a hard yes/no.
- We predict fatality of an **incident**, not of any individual person, which avoids profiling people.
- Passwords use a salted SHA-256 for demo purposes; production should use bcrypt/argon2.
