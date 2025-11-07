CREATE TABLE IF NOT EXISTS systems (
    system_name TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    timezone TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT NOT NULL,
    system_name TEXT NOT NULL REFERENCES systems(system_name),
    name TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    capacity INTEGER,
    region_id TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (station_id, system_name)
);

CREATE TABLE IF NOT EXISTS trips (
    trip_id TEXT NOT NULL,
    system_name TEXT NOT NULL REFERENCES systems(system_name),
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ NOT NULL,
    start_station_id TEXT,
    end_station_id TEXT,
    member_casual TEXT,
    rideable_type TEXT,
    duration_min NUMERIC,
    month DATE NOT NULL,
    raw_payload JSONB,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trip_id, system_name)
);

CREATE INDEX IF NOT EXISTS idx_trips_month ON trips(month);
CREATE INDEX IF NOT EXISTS idx_trips_member ON trips(member_casual);
CREATE INDEX IF NOT EXISTS idx_trips_rideable_type ON trips(rideable_type);
