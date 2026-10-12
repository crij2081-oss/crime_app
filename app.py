"""Crime Detection & Fatality Prediction System - Streamlit app.

Run:  streamlit run app.py
Logins: police1 / police123   journalist1 / press123   admin / admin123
"""
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN

import realtime
import style

BASE = Path(__file__).parent
DB_PATH = BASE / "crime.db"
MODEL_DIR = BASE / "models"
SALT = "crime_app_demo_salt"

st.set_page_config(page_title="Crime Intelligence System", page_icon="🚨", layout="wide")


@st.cache_resource(show_spinner="First run: building the database and training the models (about 30 seconds)...")
def bootstrap():
    """On a fresh deployment, create crime.db and train the models automatically."""
    if not DB_PATH.exists():
        import generate_data
        generate_data.main()
    if not ((MODEL_DIR / "fatality_model.joblib").exists() and (MODEL_DIR / "risk_model.joblib").exists()):
        import train_models
        train_models.main()
    return True


bootstrap()
style.inject()


# ---------------------------------------------------------------- helpers
def conn():
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA foreign_keys = ON")
    return c


def q(sql, params=()):
    c = conn()
    try:
        return pd.read_sql_query(sql, c, params=params)
    finally:
        c.close()


def execute(sql, params=()):
    c = conn()
    try:
        cur = c.execute(sql, params)
        c.commit()
        return cur.lastrowid
    finally:
        c.close()


def hash_pw(pw):
    return hashlib.sha256((SALT + pw).encode()).hexdigest()


@st.cache_resource
def load_models():
    f, r = MODEL_DIR / "fatality_model.joblib", MODEL_DIR / "risk_model.joblib"
    if not (f.exists() and r.exists()):
        return None, None
    return joblib.load(f), joblib.load(r)


def load_metrics():
    p = MODEL_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def bar_block(h):
    return 0 if h <= 5 else 1 if h <= 11 else 2 if h <= 17 else 3


BLOCK_NAMES = {0: "Night (00-05)", 1: "Morning (06-11)", 2: "Afternoon (12-17)", 3: "Evening (18-23)"}


# ---------------------------------------------------------------- login
def login_page():
    style.hero("🚨 Crime Intelligence System",
               "AI-powered crime detection and fatality prediction for Police and Journalists",
               ["Live alerts", "Hotspot maps", "Fatality risk AI", "Spike detection", "Role-based access"])
    if not DB_PATH.exists():
        st.error("Database not found. Run `python generate_data.py` and `python train_models.py` first.")
        st.stop()
    col, side = st.columns([1, 1])
    with side:
        style.feature_card("🔴 Live Operations Center", "Auto-refreshing incident feed with instant alerts for police")
        style.feature_card("📍 Hotspot detection", "Machine learning finds crime clusters and high-risk times")
        style.feature_card("⚠️ Fatality prediction", "Every new case gets a fatality-risk probability")
        style.feature_card("📰 Live Newsroom", "Anonymised trends and developing situations for journalists")
    with col:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            ok = st.form_submit_button("Login")
        if ok:
            df = q("""SELECT u.user_id, u.username, u.full_name, r.role_name
                      FROM users u JOIN roles r ON r.role_id = u.role_id
                      WHERE u.username = ? AND u.password_hash = ?""", (u.strip().lower(), hash_pw(p.strip())))
            if df.empty:
                st.error("Invalid username or password")
            else:
                st.session_state["user"] = df.iloc[0].to_dict()
                st.rerun()
        st.info("Demo logins: `police1 / police123`, `journalist1 / press123`, `admin / admin123`")


# ---------------------------------------------------------------- shared charts
def trend_charts(df, date_col="crime_date"):
    df = df.copy()
    df["month"] = pd.to_datetime(df[date_col]).dt.to_period("M").astype(str)
    c1, c2 = st.columns(2)
    m = df.groupby("month").agg(crimes=("crime_id", "count"), fatal=("is_fatal", "sum")).reset_index()
    c1.plotly_chart(px.line(m, x="month", y=["crimes", "fatal"], title="Monthly crimes and fatal incidents"),
                    use_container_width=True)
    t = df.groupby("type_name").size().reset_index(name="count").sort_values("count", ascending=False)
    c2.plotly_chart(px.bar(t, x="type_name", y="count", title="Crimes by type"), use_container_width=True)
    c3, c4 = st.columns(2)
    hr = df.groupby("crime_hour").size().reset_index(name="count") if "crime_hour" in df else None
    if hr is not None:
        c3.plotly_chart(px.bar(hr, x="crime_hour", y="count", title="Crimes by hour of day"), use_container_width=True)
    ar = df.groupby(["city", "area"]).agg(crimes=("crime_id", "count"), fatal=("is_fatal", "sum")).reset_index()
    ar["label"] = ar["area"] + ", " + ar["city"]
    ar = ar.sort_values("fatal", ascending=False).head(12)
    c4.plotly_chart(px.bar(ar, x="label", y="fatal", title="Areas with most fatal incidents"), use_container_width=True)


# ---------------------------------------------------------------- police pages
def police_dashboard():
    st.header("Police Dashboard")
    df = q("SELECT * FROM v_police_cases")
    df["crime_date"] = df["occurred_at"].str[:10]
    df["crime_hour"] = pd.to_datetime(df["occurred_at"]).dt.hour
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total cases", f"{len(df):,}")
    k2.metric("Fatal incidents", f"{int(df.is_fatal.sum()):,}")
    k3.metric("Open / under investigation", f"{int((df.status != 'Closed').sum()):,}")
    k4.metric("Fatality rate", f"{100 * df.is_fatal.mean():.1f}%")
    trend_charts(df)


def hotspot_page():
    st.header("Crime Hotspot Detection")
    df = q("SELECT * FROM v_police_cases")
    c1, c2, c3 = st.columns(3)
    city = c1.selectbox("City", ["All"] + sorted(df.city.unique()))
    ctype = c2.selectbox("Crime type", ["All"] + sorted(df.type_name.unique()))
    only_fatal = c3.checkbox("Fatal incidents only")
    if city != "All":
        df = df[df.city == city]
    if ctype != "All":
        df = df[df.type_name == ctype]
    if only_fatal:
        df = df[df.is_fatal == 1]
    if len(df) < 10:
        st.warning("Too few records for the selected filters.")
        return

    # DBSCAN on coordinates: haversine metric, eps ~ 3 km
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_km = st.slider("Cluster radius (km)", 1, 15, 5)
    min_s = st.slider("Minimum incidents in a hotspot", 20, 300, 80)
    labels = DBSCAN(eps=eps_km / 6371.0, min_samples=min_s, metric="haversine", algorithm="ball_tree").fit_predict(coords)
    df = df.assign(cluster=labels)
    hot = (df[df.cluster >= 0].groupby("cluster")
           .agg(incidents=("crime_id", "count"), fatal=("is_fatal", "sum"),
                lat=("latitude", "mean"), lon=("longitude", "mean"),
                areas=("area", lambda s: ", ".join(s.value_counts().head(3).index)))
           .reset_index().sort_values("incidents", ascending=False))
    st.write(f"**{len(hot)} hotspot(s) detected** from {len(df):,} incidents")
    if not hot.empty:
        m = hot.rename(columns={"lat": "latitude", "lon": "longitude"}).assign(size=lambda d: d.incidents * 3)
        st.map(m, latitude="latitude", longitude="longitude", size="size")
        st.dataframe(hot, use_container_width=True, hide_index=True)

    st.subheader("Area risk ranking")
    st.dataframe(q("SELECT city, area, total_crimes, fatal_crimes, fatality_pct FROM v_area_stats "
                   "ORDER BY total_crimes DESC"), use_container_width=True, hide_index=True)


def risk_forecast_page():
    st.header("High-Risk Area / Time Forecast")
    _, risk = load_models()
    if risk is None:
        st.error("Models not trained. Run `python train_models.py`.")
        return
    locs = q("SELECT * FROM locations ORDER BY city, area")
    rows = []
    for _, l in locs.iterrows():
        for b in range(4):
            for w in range(7):
                rows.append((l.city, l.area, l.latitude, l.longitude, b, w))
    g = pd.DataFrame(rows, columns=["city", "area", "latitude", "longitude", "block", "weekday"])
    g["risk_prob"] = risk.predict_proba(g[["latitude", "longitude", "block", "weekday"]])[:, 1]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    c1, c2 = st.columns(2)
    wd = c1.selectbox("Day of week", range(7), format_func=lambda i: days[i])
    bl = c2.selectbox("Time block", range(4), format_func=lambda i: BLOCK_NAMES[i])
    sel = g[(g.weekday == wd) & (g.block == bl)].sort_values("risk_prob", ascending=False)
    sel["label"] = sel["area"] + ", " + sel["city"]
    st.plotly_chart(px.bar(sel.head(15), x="label", y="risk_prob", range_y=[0, 1],
                           title=f"Top risk areas - {days[wd]}, {BLOCK_NAMES[bl]}",
                           labels={"risk_prob": "Probability of HIGH crime risk"}), use_container_width=True)
    st.caption("Use this to plan patrol deployment. The model learns which area/time slots historically see unusually many crimes.")


def fatality_page():
    st.header("Fatality Prediction (per incident)")
    fat, _ = load_models()
    if fat is None:
        st.error("Models not trained. Run `python train_models.py`.")
        return
    types = q("SELECT * FROM crime_types ORDER BY type_name")
    locs = q("SELECT * FROM locations ORDER BY city, area")
    with st.form("pred"):
        c1, c2, c3 = st.columns(3)
        tname = c1.selectbox("Crime type", types.type_name)
        weapon = c2.selectbox("Weapon used", ["None", "Knife", "Firearm", "Blunt Object", "Other"])
        victims = c3.number_input("Number of victims", 1, 20, 1)
        c4, c5, c6 = st.columns(3)
        city = c4.selectbox("City", sorted(locs.city.unique()))
        area = c5.selectbox("Area", sorted(locs[locs.city == city].area))
        when = c6.time_input("Time of incident", datetime.now().time())
        date = st.date_input("Date", datetime.now().date())
        save = st.checkbox("Save this prediction to the database")
        go = st.form_submit_button("Predict")
    if go:
        sev = int(types.loc[types.type_name == tname, "severity"].iloc[0])
        h = when.hour
        row = pd.DataFrame([{"type_name": tname, "weapon_used": weapon, "city": city, "area": area,
                             "severity": sev, "victim_count": victims, "hour": h, "weekday": date.weekday(),
                             "month": date.month, "is_night": int(h >= 22 or h <= 4)}])
        p = float(fat.predict_proba(row)[0, 1])
        label = int(p >= 0.5)
        level = "HIGH" if p >= 0.5 else "MEDIUM" if p >= 0.2 else "LOW"
        st.metric("Probability of fatality", f"{100 * p:.1f}%")
        (st.error if level == "HIGH" else st.warning if level == "MEDIUM" else st.success)(f"Fatality risk: **{level}**")
        if save:
            execute("INSERT INTO predictions(model_name,fatality_prob,predicted_label,made_by) VALUES (?,?,?,?)",
                    ("RandomForest", p, label, st.session_state["user"]["user_id"]))
            st.toast("Prediction saved")
    imp = MODEL_DIR / "fatality_importance.csv"
    if imp.exists():
        with st.expander("What drives the model? (feature importance)"):
            d = pd.read_csv(imp, index_col=0).reset_index().rename(columns={"index": "feature"})
            st.plotly_chart(px.bar(d, x="importance", y=d.columns[0], orientation="h"), use_container_width=True)


def add_case_page():
    st.header("Register New Case")
    types = q("SELECT * FROM crime_types ORDER BY type_name")
    locs = q("SELECT * FROM locations ORDER BY city, area")
    fat, _ = load_models()
    with st.form("new_case", clear_on_submit=True):
        c1, c2 = st.columns(2)
        tname = c1.selectbox("Crime type", types.type_name)
        weapon = c2.selectbox("Weapon used", ["None", "Knife", "Firearm", "Blunt Object", "Other"])
        c3, c4 = st.columns(2)
        city = c3.selectbox("City", sorted(locs.city.unique()))
        area = c4.selectbox("Area", sorted(locs.area.unique()))
        c5, c6, c7 = st.columns(3)
        date = c5.date_input("Date")
        time = c6.time_input("Time")
        victims = c7.number_input("Victims", 1, 20, 1)
        desc = st.text_area("Description")
        go = st.form_submit_button("Save case")
    if go:
        loc = locs[(locs.city == city) & (locs.area == area)]
        if loc.empty:
            st.error("That area does not belong to the selected city.")
            return
        type_id = int(types.loc[types.type_name == tname, "type_id"].iloc[0])
        ts = datetime.combine(date, time).strftime("%Y-%m-%d %H:%M:%S")
        cid = execute("""INSERT INTO crimes(type_id,location_id,occurred_at,weapon_used,victim_count,description,reported_by)
                         VALUES (?,?,?,?,?,?,?)""",
                      (type_id, int(loc.location_id.iloc[0]), ts, weapon, int(victims), desc,
                       st.session_state["user"]["user_id"]))
        msg = f"Case #{cid} saved. (An audit log entry was created by a database trigger.)"
        if fat is not None:
            sev = int(types.loc[types.type_name == tname, "severity"].iloc[0])
            h = time.hour
            row = pd.DataFrame([{"type_name": tname, "weapon_used": weapon, "city": city, "area": area,
                                 "severity": sev, "victim_count": int(victims), "hour": h, "weekday": date.weekday(),
                                 "month": date.month, "is_night": int(h >= 22 or h <= 4)}])
            p = float(fat.predict_proba(row)[0, 1])
            execute("INSERT INTO predictions(crime_id,model_name,fatality_prob,predicted_label,made_by) VALUES (?,?,?,?,?)",
                    (cid, "RandomForest", p, int(p >= 0.5), st.session_state["user"]["user_id"]))
            msg += f" Predicted fatality risk: {100 * p:.1f}%."
        st.success(msg)


def records_page():
    st.header("Case Records")
    df = q("SELECT * FROM v_police_cases ORDER BY occurred_at DESC")
    c1, c2, c3 = st.columns(3)
    city = c1.multiselect("City", sorted(df.city.unique()))
    ctype = c2.multiselect("Crime type", sorted(df.type_name.unique()))
    status = c3.multiselect("Status", ["Open", "Under Investigation", "Closed"])
    if city:
        df = df[df.city.isin(city)]
    if ctype:
        df = df[df.type_name.isin(ctype)]
    if status:
        df = df[df.status.isin(status)]
    st.write(f"{len(df):,} records")
    st.dataframe(df.head(1000), use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False), "cases.csv", "text/csv")

    st.subheader("Update case status")
    c1, c2, c3 = st.columns([1, 2, 1])
    cid = c1.number_input("Case ID", min_value=1, step=1)
    new = c2.selectbox("New status", ["Open", "Under Investigation", "Closed"])
    if c3.button("Update"):
        execute("UPDATE crimes SET status = ? WHERE crime_id = ?", (new, int(cid)))
        st.success("Updated")


# ---------------------------------------------------------------- journalist pages
def journalist_dashboard():
    st.header("Crime Trends (public data)")
    st.caption("Anonymised view. Exact coordinates, weapons, case status and descriptions are hidden.")
    df = q("SELECT * FROM v_public_summary")
    k1, k2, k3 = st.columns(3)
    k1.metric("Recorded crimes", f"{len(df):,}")
    k2.metric("Fatal incidents", f"{int(df.is_fatal.sum()):,}")
    k3.metric("Fatality rate", f"{100 * df.is_fatal.mean():.1f}%")
    trend_charts(df)


def story_finder_page():
    st.header("Story Finder")
    df = q("SELECT * FROM v_public_summary")
    df["year"] = df.crime_date.str[:4]
    c1, c2 = st.columns(2)
    city = c1.selectbox("City", ["All"] + sorted(df.city.unique()))
    ctype = c2.selectbox("Crime type", ["All"] + sorted(df.type_name.unique()))
    if city != "All":
        df = df[df.city == city]
    if ctype != "All":
        df = df[df.type_name == ctype]
    yr = df.groupby("year").agg(crimes=("crime_id", "count"), fatal=("is_fatal", "sum")).reset_index()
    yr["change_%"] = (yr.crimes.pct_change() * 100).round(1)
    st.dataframe(yr, use_container_width=True, hide_index=True)
    if len(yr) >= 2 and pd.notna(yr["change_%"].iloc[-1]):
        ch = yr["change_%"].iloc[-1]
        word = "rose" if ch > 0 else "fell"
        st.info(f"Headline stat: in {yr.year.iloc[-1]}, reported {ctype if ctype != 'All' else 'crimes'} "
                f"in {city if city != 'All' else 'all cities'} {word} by {abs(ch):.1f}% compared with {yr.year.iloc[-2]}.")
    ar = (df.groupby(["city", "area"]).agg(crimes=("crime_id", "count"), fatal=("is_fatal", "sum")).reset_index()
            .assign(fatality_pct=lambda d: (100 * d.fatal / d.crimes).round(2)).sort_values("crimes", ascending=False))
    st.subheader("Area comparison")
    st.dataframe(ar, use_container_width=True, hide_index=True)
    st.download_button("Download data (CSV)", ar.to_csv(index=False), "area_stats.csv", "text/csv")


def risk_awareness_page():
    st.header("Public Risk Awareness")
    _, risk = load_models()
    if risk is None:
        st.error("Models not trained. Run `python train_models.py`.")
        return
    locs = q("SELECT * FROM locations ORDER BY city, area")
    city = st.selectbox("City", sorted(locs.city.unique()))
    sub = locs[locs.city == city]
    rows = [(r.area, r.latitude, r.longitude, b, w) for r in sub.itertuples() for b in range(4) for w in range(7)]
    g = pd.DataFrame(rows, columns=["area", "latitude", "longitude", "block", "weekday"])
    g["risk"] = risk.predict_proba(g[["latitude", "longitude", "block", "weekday"]])[:, 1]
    pivot = g.groupby(["area", "block"]).risk.mean().reset_index()
    pivot["block"] = pivot.block.map(BLOCK_NAMES)
    st.plotly_chart(px.density_heatmap(pivot, x="block", y="area", z="risk", color_continuous_scale="Reds",
                                       title="Average modelled crime risk by area and time of day"),
                    use_container_width=True)


# ---------------------------------------------------------------- DBMS / admin page
SAMPLE_QUERIES = {
    "JOIN: cases with type and location": """SELECT c.crime_id, ct.type_name, l.city, l.area, c.occurred_at
FROM crimes c JOIN crime_types ct ON ct.type_id = c.type_id
JOIN locations l ON l.location_id = c.location_id LIMIT 20;""",
    "GROUP BY + HAVING: types with >5% fatality": """SELECT ct.type_name, COUNT(*) AS total, SUM(c.is_fatal) AS fatal,
ROUND(100.0*SUM(c.is_fatal)/COUNT(*),2) AS fatality_pct
FROM crimes c JOIN crime_types ct ON ct.type_id = c.type_id
GROUP BY ct.type_name HAVING fatality_pct > 5 ORDER BY fatality_pct DESC;""",
    "SUBQUERY: areas above average crime count": """SELECT city, area, total_crimes FROM v_area_stats
WHERE total_crimes > (SELECT AVG(total_crimes) FROM v_area_stats) ORDER BY total_crimes DESC;""",
    "WINDOW: rank areas within each city": """SELECT city, area, total_crimes,
RANK() OVER (PARTITION BY city ORDER BY total_crimes DESC) AS rank_in_city
FROM v_area_stats;""",
    "VICTIM demographics (age group x gender)": """SELECT CASE WHEN age<18 THEN '<18' WHEN age<35 THEN '18-34' WHEN age<60 THEN '35-59' ELSE '60+' END AS age_group,
gender, COUNT(*) AS victims, SUM(injury_level='Fatal') AS fatalities
FROM victims GROUP BY age_group, gender ORDER BY age_group;""",
    "AUDIT LOG (written by triggers)": "SELECT * FROM audit_log ORDER BY log_id DESC LIMIT 20;",
    "PREDICTIONS stored": "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT 20;",
}


def sql_console_page():
    st.header("DBMS Console (read-only)")
    st.caption("Run SELECT queries - useful for the viva demo.")
    pick = st.selectbox("Sample query", list(SAMPLE_QUERIES))
    sql = st.text_area("SQL", SAMPLE_QUERIES[pick], height=160)
    if st.button("Run"):
        if not sql.strip().lower().startswith("select") and not sql.strip().lower().startswith("with"):
            st.error("Only SELECT queries are allowed here.")
            return
        try:
            c = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
            st.dataframe(pd.read_sql_query(sql, c), use_container_width=True, hide_index=True)
            c.close()
        except Exception as e:
            st.error(str(e))


def model_info_page():
    st.header("Model Performance")
    m = load_metrics()
    if not m:
        st.warning("No metrics found. Run `python train_models.py`.")
        return
    st.subheader("Fatality model (RandomForest)")
    st.json(m["fatality_model"])
    st.subheader("Crime risk model (GradientBoosting)")
    st.json(m["risk_model"])
    st.caption("Data is synthetic, so scores reflect the generated patterns. Replace with NCRB data for real results.")


# ---------------------------------------------------------------- main
PAGES = {
    "police": {"🔴 Live Center": realtime.police_live_page, "Dashboard": police_dashboard,
               "Hotspot Map": hotspot_page, "Risk Forecast": risk_forecast_page,
               "Fatality Prediction": fatality_page, "Register Case": add_case_page, "Case Records": records_page},
    "journalist": {"🔴 Live Newsroom": realtime.journalist_live_page, "Trends": journalist_dashboard,
                   "Story Finder": story_finder_page, "Risk Awareness": risk_awareness_page},
}
PAGES["admin"] = {**PAGES["police"], "Journalist View": journalist_dashboard, "DBMS Console": sql_console_page,
                  "Model Info": model_info_page}
PAGES["police"]["Model Info"] = model_info_page
PAGES["police"]["DBMS Console"] = sql_console_page


def main():
    user = st.session_state.get("user")
    if not user:
        login_page()
        return
    role = user["role_name"]
    st.sidebar.title("🚨 Crime Intelligence")
    with st.sidebar:
        style.user_card(user["full_name"], role)
    page = st.sidebar.radio("Navigate", list(PAGES[role]))
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()
    PAGES[role][page]()


main()
