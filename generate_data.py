"""Create the SQLite database and fill it with realistic SYNTHETIC crime data.

Run:  python generate_data.py
To use real data instead (NCRB / Kaggle), see the README section "Using a real dataset".
"""
import hashlib
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
DB_PATH = BASE / "crime.db"
SEED = 42
N_CRIMES = 6000

random.seed(SEED)
np.random.seed(SEED)

# city -> {area: (lat, lon, risk_multiplier)}
CITIES = {
    "Mumbai": {"Andheri": (19.1136, 72.8697, 1.2), "Dharavi": (19.0420, 72.8553, 1.6),
               "Colaba": (18.9067, 72.8147, 0.7), "Bandra": (19.0596, 72.8295, 0.9),
               "Borivali": (19.2307, 72.8567, 0.8)},
    "Delhi": {"Connaught Place": (28.6315, 77.2167, 0.8), "Seelampur": (28.6696, 77.2670, 1.6),
              "Dwarka": (28.5921, 77.0460, 0.9), "Rohini": (28.7041, 77.1025, 1.1),
              "Saket": (28.5245, 77.2066, 0.7)},
    "Bengaluru": {"Whitefield": (12.9698, 77.7500, 0.8), "Koramangala": (12.9352, 77.6245, 0.9),
                  "Shivajinagar": (12.9857, 77.6057, 1.4), "Jayanagar": (12.9250, 77.5938, 0.7),
                  "Yeshwanthpur": (13.0285, 77.5400, 1.2)},
    "Chennai": {"T Nagar": (13.0418, 80.2341, 1.0), "Royapuram": (13.1067, 80.2958, 1.5),
                "Adyar": (13.0012, 80.2565, 0.7), "Velachery": (12.9815, 80.2180, 0.9)},
    "Kolkata": {"Park Street": (22.5535, 88.3520, 0.8), "Howrah": (22.5958, 88.2636, 1.6),
                "Salt Lake": (22.5800, 88.4180, 0.7), "Garden Reach": (22.5450, 88.2960, 1.4)},
    "Hyderabad": {"Charminar": (17.3616, 78.4747, 1.4), "Gachibowli": (17.4401, 78.3489, 0.7),
                  "Secunderabad": (17.4399, 78.4983, 1.1), "Kukatpally": (17.4948, 78.3996, 1.0)},
}

# type -> (severity, weight, base fatal probability)
CRIME_TYPES = {
    "Theft": (2, 28, 0.002), "Burglary": (3, 12, 0.01), "Vehicle Theft": (2, 10, 0.001),
    "Robbery": (4, 9, 0.06), "Assault": (3, 14, 0.07), "Cyber Crime": (2, 8, 0.0),
    "Fraud": (2, 7, 0.0), "Kidnapping": (4, 3, 0.10), "Domestic Violence": (3, 4, 0.08),
    "Sexual Assault": (5, 2, 0.06), "Arson": (4, 1.5, 0.12), "Murder": (5, 1.5, 1.0),
    "Drug Offence": (3, 4, 0.02),
}
WEAPONS = ["None", "Knife", "Firearm", "Blunt Object", "Other"]

PASSWORD_SALT = "crime_app_demo_salt"


def hash_pw(pw: str) -> str:
    return hashlib.sha256((PASSWORD_SALT + pw).encode()).hexdigest()


def pick_hour(ctype: str) -> int:
    night_heavy = {"Robbery", "Assault", "Murder", "Burglary", "Vehicle Theft", "Drug Offence"}
    if ctype in night_heavy:
        weights = [5, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 7]
    else:
        weights = [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2]
    return random.choices(range(24), weights=weights)[0]


def weapon_for(ctype: str) -> str:
    if ctype in {"Cyber Crime", "Fraud", "Theft", "Vehicle Theft"}:
        return "None"
    if ctype in {"Murder", "Robbery", "Assault", "Kidnapping", "Arson"}:
        return random.choices(WEAPONS, weights=[10, 30, 25, 20, 15])[0]
    return random.choices(WEAPONS, weights=[60, 15, 5, 12, 8])[0]


def fatal_probability(ctype, base, weapon, hour, risk, victims):
    if ctype == "Murder":
        return 1.0
    p = base
    p *= {"None": 1.0, "Knife": 2.5, "Firearm": 5.0, "Blunt Object": 2.0, "Other": 1.3}[weapon]
    if hour >= 22 or hour <= 4:
        p *= 1.6
    p *= risk
    p *= 1 + 0.15 * (victims - 1)
    return min(p, 0.95)


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript((BASE / "schema.sql").read_text())
    cur = conn.cursor()

    # roles & users
    cur.executemany("INSERT INTO roles(role_name) VALUES (?)", [("admin",), ("police",), ("journalist",)])
    users = [("admin", "admin123", "System Admin", 1),
             ("police1", "police123", "Inspector Rao", 2),
             ("police2", "police123", "Sub-Inspector Khan", 2),
             ("journalist1", "press123", "Anita Sharma (Reporter)", 3)]
    cur.executemany("INSERT INTO users(username,password_hash,full_name,role_id) VALUES (?,?,?,?)",
                    [(u, hash_pw(p), n, r) for u, p, n, r in users])

    # locations
    loc_risk = {}
    for city, areas in CITIES.items():
        for area, (lat, lon, risk) in areas.items():
            cur.execute("INSERT INTO locations(city,area,latitude,longitude) VALUES (?,?,?,?)",
                        (city, area, lat, lon))
            loc_risk[cur.lastrowid] = risk

    # crime types
    type_ids = {}
    for name, (sev, _, _) in CRIME_TYPES.items():
        cur.execute("INSERT INTO crime_types(type_name,severity) VALUES (?,?)", (name, sev))
        type_ids[name] = cur.lastrowid

    names = list(CRIME_TYPES)
    weights = [CRIME_TYPES[n][1] for n in names]
    loc_ids = list(loc_risk)
    loc_weights = [loc_risk[i] ** 2 for i in loc_ids]   # risky areas get more crimes

    start = datetime(2023, 1, 1)
    span_days = 3 * 365
    for _ in range(N_CRIMES):
        ctype = random.choices(names, weights=weights)[0]
        loc_id = random.choices(loc_ids, weights=loc_weights)[0]
        day = start + timedelta(days=random.randint(0, span_days - 1))
        hour = pick_hour(ctype)
        ts = day.replace(hour=hour, minute=random.randint(0, 59), second=0)
        weapon = weapon_for(ctype)
        n_victims = 1 if random.random() < 0.8 else random.randint(2, 5)
        p = fatal_probability(ctype, CRIME_TYPES[ctype][2], weapon, hour, loc_risk[loc_id], n_victims)
        fatal = 1 if random.random() < p else 0
        status = random.choices(["Open", "Under Investigation", "Closed"], weights=[25, 35, 40])[0]
        cur.execute(
            """INSERT INTO crimes(type_id,location_id,occurred_at,weapon_used,victim_count,is_fatal,status,description,reported_by)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (type_ids[ctype], loc_id, ts.strftime("%Y-%m-%d %H:%M:%S"), weapon, n_victims, 0, status,
             f"{ctype} reported (synthetic record)", 2))
        crime_id = cur.lastrowid
        for k in range(n_victims):
            if fatal and k == 0:
                injury = "Fatal"
            else:
                injury = random.choices(["None", "Minor", "Serious"], weights=[40, 40, 20])[0]
            cur.execute("INSERT INTO victims(crime_id,age,gender,injury_level) VALUES (?,?,?,?)",
                        (crime_id, int(np.clip(np.random.normal(33, 14), 5, 90)),
                         random.choices(["Male", "Female", "Other"], weights=[55, 43, 2])[0], injury))
        # the trigger trg_victim_check sets crimes.is_fatal when a Fatal victim is inserted

    conn.commit()
    cnt = cur.execute("SELECT COUNT(*), SUM(is_fatal) FROM crimes").fetchone()
    print(f"Database created: {DB_PATH.name} | crimes={cnt[0]} fatal={cnt[1]}")
    conn.close()


if __name__ == "__main__":
    main()
