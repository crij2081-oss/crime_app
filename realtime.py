"""Real-time layer: live feed, spike detection, high-risk alerts, incident simulator, model retraining.

Pages (auto-refresh with st.fragment):
  police_live_page()      -> Live Operations Center
  journalist_live_page()  -> Live Newsroom (anonymised)
"""
import json
import math
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
DB_PATH = BASE / "crime.db"
MODEL_DIR = BASE / "models"
SIM_TAG = "Simulated live incident"
WINDOW_MIN = 60            # spike-detection window
HIGH_RISK = 0.5            # fatality probability that triggers an alert
NON_VIOLENT = {"Cyber Crime", "Fraud", "Theft", "Vehicle Theft"}
WEAPONS = ["Knife", "Firearm", "Blunt Object", "Other"]
FMT = "%Y-%m-%d %H:%M:%S"


# ------------------------------------------------------------------ db helpers
def _conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _q(sql, params=()):
    c = _conn()
    try:
        return pd.read_sql_query(sql, c, params=params)
    finally:
        c.close()


def _exec(sql, params=()):
    c = _conn()
    try:
        cur = c.execute(sql, params)
        c.commit()
        return cur.lastrowid
    finally:
        c.close()


@st.cache_resource
def _fatality_model():
    p = MODEL_DIR / "fatality_model.joblib"
    return joblib.load(p) if p.exists() else None


def _predict(fat, tname, weapon, city, area, severity, victims, ts):
    h = ts.hour
    row = pd.DataFrame([{"type_name": tname, "weapon_used": weapon, "city": city, "area": area,
                         "severity": int(severity), "victim_count": int(victims), "hour": h,
                         "weekday": ts.weekday(), "month": ts.month, "is_night": int(h >= 22 or h <= 4)}])
    return float(fat.predict_proba(row)[0, 1])


# ------------------------------------------------------------------ analytics
def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam), summed from the tail for numerical accuracy."""
    if k <= 0:
        return 1.0
    term = math.exp(-lam) * lam ** k / math.factorial(k)
    total = 0.0
    for i in range(k, k + 60):
        total += term
        term *= lam / (i + 1)
    return min(total, 1.0)


def detect_spikes(window_min: int = WINDOW_MIN, alpha: float = 0.01) -> pd.DataFrame:
    """Flag areas whose incident count in the last `window_min` minutes is statistically unusual.

    Baseline = each area's historical average rate. Test = Poisson tail probability.
    """
    now = datetime.now()
    since = (now - timedelta(minutes=window_min)).strftime(FMT)
    hist = _q("""SELECT l.location_id, l.city, l.area, COUNT(*) AS n
                 FROM crimes c JOIN locations l ON l.location_id = c.location_id
                 WHERE c.description IS NULL OR c.description <> ?
                 GROUP BY l.location_id""", (SIM_TAG,))
    span = _q("SELECT MIN(occurred_at) AS a, MAX(occurred_at) AS b FROM crimes "
              "WHERE description IS NULL OR description <> ?", (SIM_TAG,))
    days = max((pd.to_datetime(span.b[0]) - pd.to_datetime(span.a[0])).days + 1, 1)
    recent = _q("""SELECT c.location_id, COUNT(*) AS recent
                   FROM crimes c WHERE c.occurred_at >= ? GROUP BY c.location_id""", (since,))
    if recent.empty:
        return pd.DataFrame(columns=["city", "area", "recent", "expected", "ratio", "p_value"])
    df = hist.merge(recent, on="location_id")
    df["expected"] = (df["n"] / days * window_min / 1440).clip(lower=1e-6)
    df["ratio"] = df["recent"] / df["expected"]
    df["p_value"] = [poisson_sf(int(r), e) for r, e in zip(df["recent"], df["expected"])]
    out = df[(df["recent"] >= 2) & (df["p_value"] < alpha)].sort_values("p_value")
    return out[["city", "area", "recent", "expected", "ratio", "p_value"]].reset_index(drop=True)


# ------------------------------------------------------------------ simulator
def simulate_incident(fat, focus=None) -> dict:
    """Insert one synthetic live incident (plus its fatality prediction). Used for demos."""
    types = _q("SELECT * FROM crime_types")
    locs = _q("SELECT * FROM locations")
    if focus is not None and random.random() < 0.6 and (locs.location_id == focus).any():
        loc = locs[locs.location_id == focus].iloc[0]
    else:
        loc = locs.sample(1).iloc[0]
    weights = ((6 - types.severity) ** 2).tolist()
    t = types.iloc[random.choices(range(len(types)), weights=weights)[0]]
    weapon = "None" if t.type_name in NON_VIOLENT else random.choice(WEAPONS)
    victims = random.choice([1, 1, 1, 2, 3])
    ts = datetime.now()
    cid = _exec("""INSERT INTO crimes(type_id,location_id,occurred_at,weapon_used,victim_count,description,reported_by)
                   VALUES (?,?,?,?,?,?,?)""",
                (int(t.type_id), int(loc.location_id), ts.strftime(FMT), weapon, victims, SIM_TAG, 2))
    p = None
    if fat is not None:
        p = _predict(fat, t.type_name, weapon, loc.city, loc.area, t.severity, victims, ts)
        _exec("INSERT INTO predictions(crime_id,model_name,fatality_prob,predicted_label,made_by) VALUES (?,?,?,?,?)",
              (cid, "RandomForest", p, int(p >= HIGH_RISK), 2))
    return {"crime_id": cid, "type": t.type_name, "area": loc.area, "city": loc.city, "p": p}


def _tick_simulation(fat):
    tick = st.session_state.get("tick", 0) + 1
    st.session_state["tick"] = tick
    if tick % 15 == 1 or st.session_state.get("focus_loc") is None:   # new "problem area" every ~15 ticks
        st.session_state["focus_loc"] = int(_q("SELECT location_id FROM locations").sample(1).iloc[0, 0])
    for _ in range(random.choice([1, 1, 2])):
        simulate_incident(fat, st.session_state["focus_loc"])


# ------------------------------------------------------------------ shared pieces
def _counts():
    now = datetime.now()
    k = _q("SELECT SUM(occurred_at >= ?) AS h1, SUM(occurred_at >= ?) AS h24 FROM crimes",
           ((now - timedelta(hours=1)).strftime(FMT), (now - timedelta(hours=24)).strftime(FMT)))
    return int(k.h1[0] or 0), int(k.h24[0] or 0)


def _spike_text(r):
    return (f"{r.area}, {r.city}: {int(r.recent)} incidents in the last {WINDOW_MIN} min "
            f"(normally ~{r.expected:.2f}), about {r.ratio:.0f}x the usual rate")


# ------------------------------------------------------------------ police live page
def _render_police():
    h1, h24 = _counts()
    spikes = detect_spikes()
    now = datetime.now()
    hi = _q("""SELECT c.crime_id, ct.type_name, l.area, l.city, c.occurred_at, p.fatality_prob
               FROM predictions p JOIN crimes c ON c.crime_id = p.crime_id
               JOIN crime_types ct ON ct.type_id = c.type_id JOIN locations l ON l.location_id = c.location_id
               WHERE c.occurred_at >= ? AND p.fatality_prob >= ?
               ORDER BY c.crime_id DESC LIMIT 5""",
            ((now - timedelta(hours=1)).strftime(FMT), HIGH_RISK))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Incidents (last hour)", h1)
    k2.metric("Incidents (last 24h)", h24)
    k3.metric("High fatality-risk (1h)", len(hi))
    k4.metric("Active spikes", len(spikes))
    st.caption(f"Last updated {now.strftime('%H:%M:%S')}")

    for r in spikes.itertuples():
        st.error("🚨 **Spike detected** - " + _spike_text(r))
    for r in hi.itertuples():
        st.warning(f"⚠️ **High fatality risk ({r.fatality_prob:.0%})** - {r.type_name} at {r.area}, {r.city} "
                   f"({r.occurred_at[11:16]}) - case #{r.crime_id}")
    last = st.session_state.get("last_alert_id", 0)
    new = hi[hi.crime_id > last]
    if not new.empty:
        r = new.iloc[0]
        st.toast(f"High-risk incident: {r.type_name} at {r.area}", icon="⚠️")
        st.session_state["last_alert_id"] = int(hi.crime_id.max())

    left, right = st.columns([3, 2])
    feed = _q("""SELECT c.crime_id AS id, substr(c.occurred_at,12,8) AS time, ct.type_name AS type,
                        l.area, l.city, c.weapon_used AS weapon,
                        ROUND(100*p.fatality_prob,1) AS "fatality_%", c.status
                 FROM crimes c JOIN crime_types ct ON ct.type_id = c.type_id
                 JOIN locations l ON l.location_id = c.location_id
                 LEFT JOIN predictions p ON p.crime_id = c.crime_id
                 ORDER BY c.crime_id DESC LIMIT 12""")
    left.subheader("Latest incidents")
    left.dataframe(feed, use_container_width=True, hide_index=True)
    pts = _q("SELECT latitude, longitude FROM v_police_cases ORDER BY crime_id DESC LIMIT 30")
    right.subheader("Recent locations")
    right.map(pts, latitude="latitude", longitude="longitude")

    recent = _q("SELECT occurred_at FROM crimes WHERE occurred_at >= ?",
                ((now - timedelta(hours=2)).strftime(FMT),))
    if recent.empty:
        st.info("No incidents in the last 2 hours. Turn on the simulator above or register a case.")
    else:
        t = pd.to_datetime(recent.occurred_at).dt.floor("5min")
        b = t.value_counts().sort_index().reset_index()
        b.columns = ["time", "incidents"]
        st.plotly_chart(px.bar(b, x="time", y="incidents", title="Incidents per 5 minutes (last 2 hours)"),
                        use_container_width=True)


def retrain_models():
    import train_models as tm
    df = tm.load_cases()
    fat = tm.train_fatality(df)
    risk = tm.train_risk(df)
    out = {"fatality_model": {k: round(float(v), 4) for k, v in fat.items()},
           "risk_model": {k: round(float(v), 4) for k, v in risk.items()}}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    st.cache_resource.clear()
    return out


def _admin_tools():
    with st.expander("Admin tools"):
        c1, c2 = st.columns(2)
        if c1.button("Retrain models on latest data"):
            with st.spinner("Training..."):
                out = retrain_models()
            st.success("Models retrained and reloaded")
            st.json(out)
        if c2.button("Clear simulated incidents"):
            _exec("DELETE FROM predictions WHERE crime_id IN (SELECT crime_id FROM crimes WHERE description = ?)",
                  (SIM_TAG,))
            _exec("DELETE FROM crimes WHERE description = ?", (SIM_TAG,))
            st.success("Simulated incidents removed")
        mt = MODEL_DIR / "fatality_model.joblib"
        if mt.exists():
            st.caption("Fatality model last trained: " + datetime.fromtimestamp(mt.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))


def police_live_page():
    st.header("🔴 Live Operations Center")
    fat = _fatality_model()
    c1, c2 = st.columns(2)
    c1.toggle("Simulate incoming incidents (demo mode)", key="sim_on")
    interval = c2.select_slider("Refresh every (seconds)", options=[2, 3, 5, 10], value=3)
    if st.session_state.get("sim_on"):
        st.caption("Simulator ON: new incidents arrive continuously, concentrated around one area at a time so spike alerts trigger.")

    @st.fragment(run_every=f"{interval}s")
    def live():
        if st.session_state.get("sim_on"):
            _tick_simulation(fat)
        _render_police()

    live()
    _admin_tools()


# ------------------------------------------------------------------ journalist live page
def _render_journalist():
    h1, h24 = _counts()
    spikes = detect_spikes()
    k1, k2, k3 = st.columns(3)
    k1.metric("Reported (last hour)", h1)
    k2.metric("Reported (last 24h)", h24)
    k3.metric("Developing situations", len(spikes))
    st.caption(f"Last updated {datetime.now().strftime('%H:%M:%S')}")

    if spikes.empty:
        st.success("No unusual activity detected right now.")
    for r in spikes.itertuples():
        st.warning("📰 **Developing situation (unverified)** - " + _spike_text(r))

    feed = _q("""SELECT substr(c.occurred_at,12,5) AS time, ct.type_name AS type, l.area, l.city
                 FROM crimes c JOIN crime_types ct ON ct.type_id = c.type_id
                 JOIN locations l ON l.location_id = c.location_id
                 ORDER BY c.crime_id DESC LIMIT 15""")
    st.subheader("Latest reported incidents")
    st.dataframe(feed, use_container_width=True, hide_index=True)

    now = datetime.now()
    d = _q("""SELECT l.city, COUNT(*) AS incidents FROM crimes c JOIN locations l ON l.location_id = c.location_id
              WHERE c.occurred_at >= ? GROUP BY l.city ORDER BY incidents DESC""",
           ((now - timedelta(hours=24)).strftime(FMT),))
    if not d.empty:
        st.plotly_chart(px.bar(d, x="city", y="incidents", title="Incidents by city (last 24h)"),
                        use_container_width=True)


def journalist_live_page():
    st.header("🔴 Live Newsroom")
    st.caption("Auto-refreshing, anonymised feed. Items are system-detected and unverified: confirm with police before publishing.")

    @st.fragment(run_every="5s")
    def live():
        _render_journalist()

    live()
