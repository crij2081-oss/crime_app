-- Crime Detection & Fatality Prediction System : database schema (SQLite, 3NF)
PRAGMA foreign_keys = ON;

DROP VIEW    IF EXISTS v_police_cases;
DROP VIEW    IF EXISTS v_public_summary;
DROP VIEW    IF EXISTS v_area_stats;
DROP TRIGGER IF EXISTS trg_crime_insert_audit;
DROP TRIGGER IF EXISTS trg_crime_delete_audit;
DROP TRIGGER IF EXISTS trg_victim_check;
DROP TABLE IF EXISTS audit_log;
DROP TABLE IF EXISTS predictions;
DROP TABLE IF EXISTS victims;
DROP TABLE IF EXISTS crimes;
DROP TABLE IF EXISTS crime_types;
DROP TABLE IF EXISTS locations;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS roles;

CREATE TABLE roles (
    role_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    role_name TEXT NOT NULL UNIQUE            -- admin / police / journalist
);

CREATE TABLE users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name     TEXT NOT NULL,
    role_id       INTEGER NOT NULL,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(role_id)
);

CREATE TABLE locations (
    location_id INTEGER PRIMARY KEY AUTOINCREMENT,
    city        TEXT NOT NULL,
    area        TEXT NOT NULL,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    UNIQUE (city, area)
);

CREATE TABLE crime_types (
    type_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name TEXT NOT NULL UNIQUE,
    severity  INTEGER NOT NULL CHECK (severity BETWEEN 1 AND 5)
);

CREATE TABLE crimes (
    crime_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    type_id       INTEGER NOT NULL,
    location_id   INTEGER NOT NULL,
    occurred_at   TEXT NOT NULL,              -- ISO datetime
    weapon_used   TEXT NOT NULL DEFAULT 'None',
    victim_count  INTEGER NOT NULL DEFAULT 1 CHECK (victim_count >= 0),
    is_fatal      INTEGER NOT NULL DEFAULT 0 CHECK (is_fatal IN (0,1)),
    status        TEXT NOT NULL DEFAULT 'Open' CHECK (status IN ('Open','Under Investigation','Closed')),
    description   TEXT,
    reported_by   INTEGER,
    FOREIGN KEY (type_id)     REFERENCES crime_types(type_id),
    FOREIGN KEY (location_id) REFERENCES locations(location_id),
    FOREIGN KEY (reported_by) REFERENCES users(user_id)
);

CREATE TABLE victims (
    victim_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    crime_id    INTEGER NOT NULL,
    age         INTEGER CHECK (age BETWEEN 0 AND 120),
    gender      TEXT CHECK (gender IN ('Male','Female','Other')),
    injury_level TEXT CHECK (injury_level IN ('None','Minor','Serious','Fatal')),
    FOREIGN KEY (crime_id) REFERENCES crimes(crime_id) ON DELETE CASCADE
);

CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    crime_id      INTEGER,
    model_name    TEXT NOT NULL,
    fatality_prob REAL NOT NULL,
    predicted_label INTEGER NOT NULL,
    predicted_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    made_by       INTEGER,
    FOREIGN KEY (crime_id) REFERENCES crimes(crime_id) ON DELETE SET NULL,
    FOREIGN KEY (made_by)  REFERENCES users(user_id)
);

CREATE TABLE audit_log (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,
    table_name TEXT NOT NULL,
    record_id  INTEGER,
    logged_at  TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common filters
CREATE INDEX idx_crimes_time  ON crimes(occurred_at);
CREATE INDEX idx_crimes_loc   ON crimes(location_id);
CREATE INDEX idx_crimes_type  ON crimes(type_id);

-- Triggers
CREATE TRIGGER trg_crime_insert_audit AFTER INSERT ON crimes
BEGIN
    INSERT INTO audit_log(action, table_name, record_id) VALUES ('INSERT', 'crimes', NEW.crime_id);
END;

CREATE TRIGGER trg_crime_delete_audit AFTER DELETE ON crimes
BEGIN
    INSERT INTO audit_log(action, table_name, record_id) VALUES ('DELETE', 'crimes', OLD.crime_id);
END;

-- A fatal injury on a victim automatically marks the crime fatal
CREATE TRIGGER trg_victim_check AFTER INSERT ON victims
WHEN NEW.injury_level = 'Fatal'
BEGIN
    UPDATE crimes SET is_fatal = 1 WHERE crime_id = NEW.crime_id;
END;

-- Views (role-based access)
-- Police: full case detail
CREATE VIEW v_police_cases AS
SELECT c.crime_id, c.occurred_at, ct.type_name, ct.severity,
       l.city, l.area, l.latitude, l.longitude,
       c.weapon_used, c.victim_count, c.is_fatal, c.status, c.description
FROM crimes c
JOIN crime_types ct ON ct.type_id = c.type_id
JOIN locations   l  ON l.location_id = c.location_id;

-- Journalist: anonymised, no weapon / description / status / exact coordinates
CREATE VIEW v_public_summary AS
SELECT c.crime_id,
       substr(c.occurred_at, 1, 10) AS crime_date,
       CAST(substr(c.occurred_at, 12, 2) AS INTEGER) AS crime_hour,
       ct.type_name, l.city, l.area, c.is_fatal
FROM crimes c
JOIN crime_types ct ON ct.type_id = c.type_id
JOIN locations   l  ON l.location_id = c.location_id;

-- Area level statistics
CREATE VIEW v_area_stats AS
SELECT l.location_id, l.city, l.area, l.latitude, l.longitude,
       COUNT(c.crime_id)            AS total_crimes,
       COALESCE(SUM(c.is_fatal), 0) AS fatal_crimes,
       ROUND(100.0 * COALESCE(SUM(c.is_fatal),0) / MAX(COUNT(c.crime_id),1), 2) AS fatality_pct
FROM locations l
LEFT JOIN crimes c ON c.location_id = l.location_id
GROUP BY l.location_id;
