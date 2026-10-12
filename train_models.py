"""Train the two ML models from the database.

Model 1 (fatality):  will a given incident turn fatal?  (RandomForest classifier)
Model 2 (crime risk): will an area + time-block be HIGH RISK for crime?  (GradientBoosting classifier)

Run:  python train_models.py
"""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).parent
DB_PATH = BASE / "crime.db"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_cases() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM v_police_cases", conn)
    conn.close()
    t = pd.to_datetime(df["occurred_at"])
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 4)).astype(int)
    return df


def train_fatality(df: pd.DataFrame) -> dict:
    cat = ["type_name", "weapon_used", "city", "area"]
    num = ["severity", "victim_count", "hour", "weekday", "month", "is_night"]
    X, y = df[cat + num], df["is_fatal"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    model = Pipeline([("pre", pre),
                      ("clf", RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
                                                      random_state=42, n_jobs=-1))])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred, zero_division=0),
               "recall": recall_score(y_te, pred), "f1": f1_score(y_te, pred), "roc_auc": roc_auc_score(y_te, proba),
               "train_rows": len(X_tr), "test_rows": len(X_te), "fatal_rate": float(y.mean())}
    joblib.dump(model, MODEL_DIR / "fatality_model.joblib")

    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
    imp.head(15).to_csv(MODEL_DIR / "fatality_importance.csv", header=["importance"])
    return metrics


def block_of(h):
    return pd.cut(h, bins=[-1, 5, 11, 17, 23], labels=[0, 1, 2, 3]).astype(int)  # night, morning, afternoon, evening


def train_risk(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["block"] = block_of(df["hour"])
    g = (df.groupby(["area", "city", "latitude", "longitude", "block", "weekday"]).size()
           .rename("n").reset_index())
    # complete grid (so slots with zero crimes are included)
    areas = df[["area", "city", "latitude", "longitude"]].drop_duplicates()
    grid = areas.merge(pd.DataFrame([(b, w) for b in range(4) for w in range(7)], columns=["block", "weekday"]), how="cross")
    g = grid.merge(g, how="left", on=["area", "city", "latitude", "longitude", "block", "weekday"]).fillna({"n": 0})
    thr = float(g["n"].quantile(0.75))
    g["high_risk"] = (g["n"] > thr).astype(int)

    feats = ["latitude", "longitude", "block", "weekday"]
    X_tr, X_te, y_tr, y_te = train_test_split(g[feats], g["high_risk"], test_size=0.25,
                                              stratify=g["high_risk"], random_state=42)
    clf = GradientBoostingClassifier(random_state=42).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
               "roc_auc": roc_auc_score(y_te, proba), "threshold_crimes_per_slot": thr}
    joblib.dump(clf, MODEL_DIR / "risk_model.joblib")
    return metrics


def main():
    df = load_cases()
    print(f"Loaded {len(df)} cases")
    fat = train_fatality(df)
    risk = train_risk(df)
    out = {"fatality_model": {k: round(float(v), 4) for k, v in fat.items()},
           "risk_model": {k: round(float(v), 4) for k, v in risk.items()}}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
