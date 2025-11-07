-- Bikeshare Analytics Database Schema
-- Optimized for fast analytics queries and historical data storage

-- Systems table - metadata for bikeshare systems
CREATE TABLE IF NOT EXISTS systems (
    system_id SERIAL PRIMARY KEY,
    system_name VARCHAR(50) UNIQUE NOT NULL,
    city VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL DEFAULT 'USA',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Stations table - station information with coordinates
CREATE TABLE IF NOT EXISTS stations (
    station_id VARCHAR(50) PRIMARY KEY,
    system_name VARCHAR(50) NOT NULL,
    station_name VARCHAR(200),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    capacity INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (system_name) REFERENCES systems(system_name)
);

-- Main trips table - optimized for analytics queries
CREATE TABLE IF NOT EXISTS trips (
    trip_id VARCHAR(100) PRIMARY KEY,
    system_name VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration_minutes INTEGER,
    distance_miles DECIMAL(8, 3),
    start_station_id VARCHAR(50),
    end_station_id VARCHAR(50),
    start_station_name VARCHAR(200),
    end_station_name VARCHAR(200),
    start_latitude DECIMAL(10, 8),
    start_longitude DECIMAL(11, 8),
    end_latitude DECIMAL(10, 8),
    end_longitude DECIMAL(11, 8),
    member_type VARCHAR(20), -- 'member' or 'casual'
    bike_type VARCHAR(30), -- 'classic_bike', 'electric_bike', etc.
    user_type VARCHAR(20), -- for legacy compatibility
    birth_year INTEGER,
    gender VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (system_name) REFERENCES systems(system_name)
);

-- Data load tracking table
CREATE TABLE IF NOT EXISTS data_load_log (
    load_id SERIAL PRIMARY KEY,
    system_name VARCHAR(50) NOT NULL,
    data_month DATE NOT NULL, -- First day of the month for the data
    file_name VARCHAR(200),
    file_url TEXT,
    trips_loaded INTEGER,
    trips_rejected INTEGER,
    load_status VARCHAR(20) DEFAULT 'pending', -- pending, success, failed
    error_message TEXT,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    UNIQUE(system_name, data_month)
);

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_trips_system_time ON trips(system_name, start_time);
CREATE INDEX IF NOT EXISTS idx_trips_start_time ON trips(start_time);
CREATE INDEX IF NOT EXISTS idx_trips_end_time ON trips(end_time);
CREATE INDEX IF NOT EXISTS idx_trips_member_type ON trips(member_type);
CREATE INDEX IF NOT EXISTS idx_trips_bike_type ON trips(bike_type);
CREATE INDEX IF NOT EXISTS idx_trips_stations ON trips(start_station_id, end_station_id);
CREATE INDEX IF NOT EXISTS idx_trips_duration ON trips(duration_minutes);
CREATE INDEX IF NOT EXISTS idx_trips_distance ON trips(distance_miles);

-- Stations indexes
CREATE INDEX IF NOT EXISTS idx_stations_system ON stations(system_name);
CREATE INDEX IF NOT EXISTS idx_stations_location ON stations(latitude, longitude);

-- Data load log indexes
CREATE INDEX IF NOT EXISTS idx_load_log_system_month ON data_load_log(system_name, data_month);
CREATE INDEX IF NOT EXISTS idx_load_log_status ON data_load_log(load_status);

-- Insert default systems
INSERT INTO systems (system_name, city, country) VALUES
    ('baywheels', 'San Francisco Bay Area', 'USA'),
    ('citibike', 'New York City', 'USA')
ON CONFLICT (system_name) DO NOTHING;