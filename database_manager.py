"""
Database Manager for Bikeshare Analytics
Handles database connections, schema creation, and data operations
"""

import os
import psycopg2
import pandas as pd
from datetime import datetime, date
import logging
from typing import Optional, Dict, List, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_string = os.getenv('DATABASE_URL')
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable not found")
    
    def get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(self.connection_string)
    
    def execute_schema(self):
        """Execute the database schema creation"""
        try:
            with open('database_schema.sql', 'r') as f:
                schema_sql = f.read()
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(schema_sql)
                conn.commit()
            
            logger.info("Database schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            return False
    
    def check_data_loaded(self, system_name: str, data_month: date) -> bool:
        """Check if data for a specific month has already been loaded"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM data_load_log 
                    WHERE system_name = %s AND data_month = %s AND load_status = 'success'
                """, (system_name, data_month))
                return cur.fetchone()[0] > 0
    
    def get_available_months(self, system_name: str) -> List[date]:
        """Get list of months with successfully loaded data"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT data_month 
                    FROM data_load_log 
                    WHERE system_name = %s AND load_status = 'success'
                    ORDER BY data_month DESC
                """, (system_name,))
                return [row[0] for row in cur.fetchall()]
    
    def log_data_load_start(self, system_name: str, data_month: date, file_name: str, file_url: str) -> int:
        """Log the start of a data loading process"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO data_load_log (system_name, data_month, file_name, file_url, load_status)
                    VALUES (%s, %s, %s, %s, 'pending')
                    ON CONFLICT (system_name, data_month) 
                    DO UPDATE SET 
                        file_name = EXCLUDED.file_name,
                        file_url = EXCLUDED.file_url,
                        load_status = 'pending',
                        started_at = NOW(),
                        completed_at = NULL,
                        error_message = NULL
                    RETURNING load_id
                """, (system_name, data_month, file_name, file_url))
                load_id = cur.fetchone()[0]
            conn.commit()
        return load_id
    
    def log_data_load_complete(self, load_id: int, trips_loaded: int, trips_rejected: int, 
                             status: str = 'success', error_message: str = None):
        """Log the completion of a data loading process"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE data_load_log 
                    SET trips_loaded = %s, trips_rejected = %s, load_status = %s, 
                        error_message = %s, completed_at = NOW()
                    WHERE load_id = %s
                """, (trips_loaded, trips_rejected, status, error_message, load_id))
            conn.commit()
    
    def bulk_insert_trips(self, trips_df: pd.DataFrame, system_name: str):
        """Efficiently insert trips data using bulk operations"""
        # Prepare data for insertion
        trips_df = trips_df.copy()
        trips_df['system_name'] = system_name
        
        # Handle missing values
        trips_df = trips_df.where(pd.notnull(trips_df), None)
        
        # Convert to list of tuples for bulk insert
        columns = [
            'trip_id', 'system_name', 'start_time', 'end_time', 'duration_minutes',
            'distance_miles', 'start_station_id', 'end_station_id', 'start_station_name',
            'end_station_name', 'start_latitude', 'start_longitude', 'end_latitude',
            'end_longitude', 'member_type', 'bike_type', 'user_type', 'birth_year', 'gender'
        ]
        
        # Ensure all required columns exist
        for col in columns:
            if col not in trips_df.columns:
                trips_df[col] = None
        
        values = trips_df[columns].values.tolist()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Use execute_values for efficient bulk insert
                from psycopg2.extras import execute_values
                
                insert_sql = f"""
                    INSERT INTO trips ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT (trip_id) DO NOTHING
                """
                
                execute_values(
                    cur, insert_sql, values,
                    template=None, page_size=1000
                )
            conn.commit()
    
    def bulk_insert_stations(self, stations_df: pd.DataFrame, system_name: str):
        """Efficiently insert or update stations data"""
        stations_df = stations_df.copy()
        stations_df['system_name'] = system_name
        stations_df = stations_df.where(pd.notnull(stations_df), None)
        
        columns = [
            'station_id', 'system_name', 'station_name', 'latitude', 
            'longitude', 'capacity', 'is_active'
        ]
        
        for col in columns:
            if col not in stations_df.columns:
                if col == 'is_active':
                    stations_df[col] = True
                else:
                    stations_df[col] = None
        
        values = stations_df[columns].values.tolist()
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                from psycopg2.extras import execute_values
                
                insert_sql = f"""
                    INSERT INTO stations ({', '.join(columns)})
                    VALUES %s
                    ON CONFLICT (station_id) DO UPDATE SET
                        station_name = EXCLUDED.station_name,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        capacity = EXCLUDED.capacity,
                        updated_at = NOW()
                """
                
                execute_values(
                    cur, insert_sql, values,
                    template=None, page_size=1000
                )
            conn.commit()
    
    def get_trips_data(self, system_name: str = None, start_date: date = None, 
                      end_date: date = None, limit: int = None) -> pd.DataFrame:
        """Query trips data with optional filters"""
        query = "SELECT * FROM trips WHERE 1=1"
        params = []
        
        if system_name:
            query += " AND system_name = %s"
            params.append(system_name)
        
        if start_date:
            query += " AND start_time >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND start_time < %s"
            params.append(end_date)
        
        query += " ORDER BY start_time"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_monthly_metrics(self, system_name: str = None, start_date: date = None, 
                           end_date: date = None) -> pd.DataFrame:
        """Get aggregated monthly metrics"""
        query = """
            SELECT 
                system_name,
                DATE_TRUNC('month', start_time) as month,
                COUNT(*) as total_trips,
                SUM(CASE WHEN member_type = 'member' THEN 1 ELSE 0 END) as member_rides,
                SUM(CASE WHEN member_type = 'casual' THEN 1 ELSE 0 END) as casual_rides,
                AVG(duration_minutes) as avg_duration,
                AVG(distance_miles) as avg_distance,
                COUNT(DISTINCT start_station_id) as unique_start_stations,
                COUNT(DISTINCT end_station_id) as unique_end_stations
            FROM trips 
            WHERE 1=1
        """
        params = []
        
        if system_name:
            query += " AND system_name = %s"
            params.append(system_name)
        
        if start_date:
            query += " AND start_time >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND start_time < %s"
            params.append(end_date)
        
        query += """
            GROUP BY system_name, DATE_TRUNC('month', start_time)
            ORDER BY month
        """
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_data_quality_stats(self) -> Dict:
        """Get data quality and loading statistics"""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get overall stats
                cur.execute("""
                    SELECT 
                        system_name,
                        COUNT(*) as total_trips,
                        MIN(start_time) as earliest_trip,
                        MAX(start_time) as latest_trip,
                        COUNT(DISTINCT DATE_TRUNC('month', start_time)) as months_loaded
                    FROM trips 
                    GROUP BY system_name
                """)
                
                stats = {}
                for row in cur.fetchall():
                    system_name, total_trips, earliest, latest, months = row
                    stats[system_name] = {
                        'total_trips': total_trips,
                        'earliest_trip': earliest,
                        'latest_trip': latest,
                        'months_loaded': months
                    }
                
                # Get load log summary
                cur.execute("""
                    SELECT 
                        system_name,
                        load_status,
                        COUNT(*) as count,
                        SUM(COALESCE(trips_loaded, 0)) as total_loaded,
                        SUM(COALESCE(trips_rejected, 0)) as total_rejected
                    FROM data_load_log 
                    GROUP BY system_name, load_status
                """)
                
                for row in cur.fetchall():
                    system_name, status, count, loaded, rejected = row
                    if system_name not in stats:
                        stats[system_name] = {}
                    stats[system_name][f'loads_{status}'] = count
                    if loaded:
                        stats[system_name]['total_loaded'] = loaded
                    if rejected:
                        stats[system_name]['total_rejected'] = rejected
                
                return stats
    
    def get_top_routes(self, system_name: str, start_date: date, end_date: date, top_n: int = 10) -> pd.DataFrame:
        """Get top N routes using SQL aggregation for better performance"""
        query = """
            WITH route_counts AS (
                SELECT 
                    start_station_id,
                    end_station_id,
                    start_station_name,
                    end_station_name,
                    start_latitude,
                    start_longitude,
                    end_latitude,
                    end_longitude,
                    COUNT(*) as trip_count
                FROM trips
                WHERE system_name = %s
                    AND start_time >= %s
                    AND start_time < %s
                    AND start_station_id IS NOT NULL
                    AND end_station_id IS NOT NULL
                    AND start_station_id != end_station_id
                    AND start_latitude IS NOT NULL
                    AND start_longitude IS NOT NULL
                    AND end_latitude IS NOT NULL
                    AND end_longitude IS NOT NULL
                GROUP BY 
                    start_station_id, end_station_id,
                    start_station_name, end_station_name,
                    start_latitude, start_longitude,
                    end_latitude, end_longitude
            ),
            bidirectional_routes AS (
                SELECT 
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN start_station_id 
                        ELSE end_station_id 
                    END as station_a_id,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN end_station_id 
                        ELSE start_station_id 
                    END as station_b_id,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN start_station_name 
                        ELSE end_station_name 
                    END as station_a_name,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN end_station_name 
                        ELSE start_station_name 
                    END as station_b_name,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN start_latitude 
                        ELSE end_latitude 
                    END as station_a_lat,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN start_longitude 
                        ELSE end_longitude 
                    END as station_a_lon,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN end_latitude 
                        ELSE start_latitude 
                    END as station_b_lat,
                    CASE 
                        WHEN start_station_id < end_station_id 
                        THEN end_longitude 
                        ELSE start_longitude 
                    END as station_b_lon,
                    SUM(trip_count) as total_trips
                FROM route_counts
                GROUP BY 
                    station_a_id, station_b_id,
                    station_a_name, station_b_name,
                    station_a_lat, station_a_lon,
                    station_b_lat, station_b_lon
            )
            SELECT 
                station_a_id as start_station_id,
                station_b_id as end_station_id,
                station_a_name as start_station_name,
                station_b_name as end_station_name,
                station_a_lat as start_lat,
                station_a_lon as start_lon,
                station_b_lat as end_lat,
                station_b_lon as end_lon,
                total_trips as trip_count
            FROM bidirectional_routes
            ORDER BY total_trips DESC
            LIMIT %s
        """
        
        params = [system_name, start_date, end_date, top_n]
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

# Streamlit caching
@st.cache_resource
def get_database_manager():
    """Get cached database manager instance"""
    return DatabaseManager()