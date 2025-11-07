"""
Data Migration Script for Bikeshare Analytics
Migrates historical CSV data from S3 to PostgreSQL database
"""

import pandas as pd
import requests
import zipfile
import io
import logging
from datetime import datetime, date
from calendar import monthrange
import streamlit as st
from typing import Dict, List, Tuple, Optional
from database_manager import DatabaseManager
import os
import tempfile

logger = logging.getLogger(__name__)

class DataMigrator:
    def __init__(self):
        self.db = DatabaseManager()
        self.systems_config = {
            'baywheels': {
                'base_url': 'https://s3.amazonaws.com/baywheels-data',
                'file_pattern': '{year}{month:02d}-baywheels-tripdata.csv.zip',
                'date_range': (2017, 6),  # Start from June 2017
                'city': 'San Francisco'
            },
            'citibike': {
                'base_url': 'https://s3.amazonaws.com/tripdata',
                'file_pattern': '{year}{month:02d}-citibike-tripdata.csv.zip',
                'date_range': (2013, 6),  # Start from June 2013
                'city': 'New York City'
            }
        }
    
    def setup_database(self):
        """Initialize database schema"""
        return self.db.execute_schema()
    
    def clean_trip_data(self, df: pd.DataFrame, system_name: str) -> Tuple[pd.DataFrame, int]:
        """Apply data cleaning rules and return cleaned data + rejection count"""
        original_count = len(df)
        
        # Standardize column names based on system
        df = self._standardize_columns(df, system_name)
        
        # Convert datetime columns
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        if 'end_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        
        # Data quality filters
        # Remove trips with missing critical data
        df = df.dropna(subset=['start_time', 'end_time'])
        
        # Remove trips with invalid time ranges
        df = df[df['end_time'] > df['start_time']]
        
        # Calculate duration if not present
        if 'duration_minutes' not in df.columns:
            df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
        
        # Remove unrealistic durations
        df = df[(df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 1440)]  # 1 min to 24 hours
        
        # Convert coordinates to numeric, handling any non-numeric values
        coord_columns = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']
        for col in coord_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove trips with invalid coordinates
        if 'start_latitude' in df.columns and 'start_longitude' in df.columns:
            df = df[
                (df['start_latitude'].notna()) &
                (df['start_longitude'].notna()) &
                (df['start_latitude'].between(-90, 90)) & 
                (df['start_longitude'].between(-180, 180))
            ]
        
        if 'end_latitude' in df.columns and 'end_longitude' in df.columns:
            df = df[
                (df['end_latitude'].notna()) &
                (df['end_longitude'].notna()) &
                (df['end_latitude'].between(-90, 90)) & 
                (df['end_longitude'].between(-180, 180))
            ]
        
        # Calculate distance if coordinates are available
        if all(col in df.columns for col in ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude']):
            try:
                df['distance_miles'] = self._calculate_distance(
                    df['start_latitude'], df['start_longitude'],
                    df['end_latitude'], df['end_longitude']
                )
                # Remove unrealistic distances
                df = df[(df['distance_miles'].notna()) & (df['distance_miles'] >= 0.01) & (df['distance_miles'] <= 50)]
            except Exception as e:
                # If distance calculation fails, just skip it
                pass
        
        # Standardize member types
        if 'member_type' in df.columns:
            df['member_type'] = df['member_type'].str.lower().replace({
                'subscriber': 'member',
                'customer': 'casual'
            })
        
        # Generate trip IDs if not present
        if 'trip_id' not in df.columns:
            df['trip_id'] = df.apply(lambda row: f"{system_name}_{row['start_time'].strftime('%Y%m%d_%H%M%S')}_{hash(str(row.to_dict())) % 1000000}", axis=1)
        
        rejected_count = original_count - len(df)
        return df, rejected_count
    
    def _standardize_columns(self, df: pd.DataFrame, system_name: str) -> pd.DataFrame:
        """Standardize column names across different data sources"""
        # Common column mappings
        column_mappings = {
            # Time columns
            'started_at': 'start_time',
            'ended_at': 'end_time',
            'start time': 'start_time',
            'stop time': 'end_time',
            'starttime': 'start_time',
            'stoptime': 'end_time',
            
            # Station columns
            'start_station_id': 'start_station_id',
            'end_station_id': 'end_station_id',
            'start station id': 'start_station_id',
            'end station id': 'end_station_id',
            'start_station_name': 'start_station_name',
            'end_station_name': 'end_station_name',
            'start station name': 'start_station_name',
            'end station name': 'end_station_name',
            
            # Location columns
            'start_lat': 'start_latitude',
            'start_lng': 'start_longitude',
            'end_lat': 'end_latitude',
            'end_lng': 'end_longitude',
            'start station latitude': 'start_latitude',
            'start station longitude': 'start_longitude',
            'end station latitude': 'end_latitude',
            'end station longitude': 'end_longitude',
            
            # User/bike type columns
            'member_casual': 'member_type',
            'usertype': 'member_type',
            'user type': 'member_type',
            'rideable_type': 'bike_type',
            'bikeid': 'bike_id',
            'bike id': 'bike_id',
            
            # Duration
            'tripduration': 'duration_seconds',
            'trip duration': 'duration_seconds',
            
            # Demographics
            'birth year': 'birth_year',
        }
        
        # Apply column mappings
        df = df.rename(columns={k: v for k, v in column_mappings.items() if k in df.columns})
        
        # Convert duration from seconds to minutes if needed
        if 'duration_seconds' in df.columns:
            df['duration_minutes'] = df['duration_seconds'] / 60
            df = df.drop('duration_seconds', axis=1)
        
        return df
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        from math import radians, sin, cos, sqrt, asin
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in miles
        r = 3959
        
        return c * r
    
    def get_available_files(self, system_name: str) -> List[Tuple[date, str]]:
        """Get list of available files for a system"""
        config = self.systems_config[system_name]
        base_url = config['base_url']
        file_pattern = config['file_pattern']
        start_year, start_month = config['date_range']
        
        available_files = []
        current_date = datetime.now()
        
        # Generate list of potential files from start date to current month
        year = start_year
        month = start_month
        
        while year < current_date.year or (year == current_date.year and month <= current_date.month):
            file_name = file_pattern.format(year=year, month=month)
            file_url = f"{base_url}/{file_name}"
            month_date = date(year, month, 1)
            
            # Check if file exists (simple HEAD request)
            try:
                response = requests.head(file_url, timeout=10)
                if response.status_code == 200:
                    available_files.append((month_date, file_url))
            except:
                pass  # File doesn't exist or network error
            
            # Move to next month
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        return available_files
    
    def load_month_data(self, system_name: str, data_month: date, file_url: str, 
                       progress_callback=None) -> Tuple[bool, str]:
        """Load data for a specific month"""
        try:
            # Check if already loaded
            if self.db.check_data_loaded(system_name, data_month):
                return True, "Data already loaded"
            
            file_name = file_url.split('/')[-1]
            
            # Log start of loading
            load_id = self.db.log_data_load_start(system_name, data_month, file_name, file_url)
            
            if progress_callback:
                progress_callback(f"Downloading {file_name}...")
            
            # Download and extract CSV
            response = requests.get(file_url, timeout=300)
            response.raise_for_status()
            
            if progress_callback:
                progress_callback(f"Extracting {file_name}...")
            
            # Handle ZIP files
            if file_name.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    csv_name = file_name.replace('.zip', '')
                    csv_content = zip_file.read(csv_name)
                    df = pd.read_csv(io.StringIO(csv_content.decode('utf-8')))
            else:
                df = pd.read_csv(io.StringIO(response.text))
            
            if progress_callback:
                progress_callback(f"Cleaning {len(df):,} records...")
            
            # Clean the data
            cleaned_df, rejected_count = self.clean_trip_data(df, system_name)
            
            if progress_callback:
                progress_callback(f"Inserting {len(cleaned_df):,} clean records...")
            
            # Insert into database
            self.db.bulk_insert_trips(cleaned_df, system_name)
            
            # Extract and insert station data
            station_data = self._extract_station_data(cleaned_df, system_name)
            if not station_data.empty:
                self.db.bulk_insert_stations(station_data, system_name)
            
            # Log successful completion
            self.db.log_data_load_complete(
                load_id, len(cleaned_df), rejected_count, 'success'
            )
            
            success_msg = f"Loaded {len(cleaned_df):,} trips, rejected {rejected_count:,}"
            if progress_callback:
                progress_callback(success_msg)
            
            return True, success_msg
            
        except Exception as e:
            error_msg = f"Error loading {file_name}: {str(e)}"
            logger.error(error_msg)
            
            # Log failed completion
            try:
                self.db.log_data_load_complete(
                    load_id, 0, 0, 'failed', error_msg
                )
            except:
                pass
            
            return False, error_msg
    
    def _extract_station_data(self, trips_df: pd.DataFrame, system_name: str) -> pd.DataFrame:
        """Extract unique station data from trips"""
        stations = []
        
        # Extract start stations
        if all(col in trips_df.columns for col in ['start_station_id', 'start_station_name']):
            start_stations = trips_df[
                ['start_station_id', 'start_station_name', 'start_latitude', 'start_longitude']
            ].dropna(subset=['start_station_id']).drop_duplicates('start_station_id')
            
            start_stations = start_stations.rename(columns={
                'start_station_id': 'station_id',
                'start_station_name': 'station_name',
                'start_latitude': 'latitude',
                'start_longitude': 'longitude'
            })
            stations.append(start_stations)
        
        # Extract end stations
        if all(col in trips_df.columns for col in ['end_station_id', 'end_station_name']):
            end_stations = trips_df[
                ['end_station_id', 'end_station_name', 'end_latitude', 'end_longitude']
            ].dropna(subset=['end_station_id']).drop_duplicates('end_station_id')
            
            end_stations = end_stations.rename(columns={
                'end_station_id': 'station_id',
                'end_station_name': 'station_name',
                'end_latitude': 'latitude',
                'end_longitude': 'longitude'
            })
            stations.append(end_stations)
        
        if stations:
            all_stations = pd.concat(stations, ignore_index=True).drop_duplicates('station_id')
            return all_stations
        
        return pd.DataFrame()
    
    def migrate_system_data(self, system_name: str, max_months: int = None, 
                           progress_callback=None) -> Dict:
        """Migrate all available data for a system"""
        available_files = self.get_available_files(system_name)
        
        if not available_files:
            return {'success': False, 'message': 'No files found'}
        
        results = {
            'total_files': len(available_files),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'messages': []
        }
        
        # Limit number of months if specified
        if max_months:
            available_files = available_files[-max_months:]
        
        for i, (data_month, file_url) in enumerate(available_files):
            if progress_callback:
                progress_callback(f"Processing {data_month.strftime('%Y-%m')} ({i+1}/{len(available_files)})")
            
            success, message = self.load_month_data(
                system_name, data_month, file_url, progress_callback
            )
            
            results['processed'] += 1
            if success:
                results['successful'] += 1
            else:
                results['failed'] += 1
            
            results['messages'].append(f"{data_month.strftime('%Y-%m')}: {message}")
        
        results['success'] = results['failed'] == 0
        return results

# Streamlit interface functions
def run_migration_interface():
    """Streamlit interface for data migration"""
    st.header("🚀 Database Migration")
    
    # Initialize migrator
    migrator = DataMigrator()
    
    # Setup database button
    if st.button("Initialize Database Schema"):
        with st.spinner("Creating database schema..."):
            if migrator.setup_database():
                st.success("Database schema created successfully!")
            else:
                st.error("Failed to create database schema")
    
    st.subheader("Available Systems")
    
    # System selection
    system_name = st.selectbox(
        "Select System",
        options=['baywheels', 'citibike'],
        format_func=lambda x: f"{x.title()} ({migrator.systems_config[x]['city']})"
    )
    
    # Check available files
    if st.button("Check Available Files"):
        with st.spinner("Checking available files..."):
            available_files = migrator.get_available_files(system_name)
            
        if available_files:
            st.success(f"Found {len(available_files)} available files")
            
            # Show sample of files
            sample_files = available_files[-5:]  # Last 5 files
            st.write("Recent files:")
            for month_date, file_url in sample_files:
                st.write(f"- {month_date.strftime('%Y-%m')}: {file_url.split('/')[-1]}")
        else:
            st.warning("No available files found")
    
    # Migration options
    st.subheader("Migration Options")
    
    max_months = st.number_input(
        "Maximum months to process (0 = all)",
        min_value=0, max_value=120, value=3,
        help="Limit migration to recent months for testing"
    )
    
    # Run migration
    if st.button(f"Migrate {system_name.title()} Data"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(message):
            status_text.text(message)
        
        with st.spinner("Running migration..."):
            results = migrator.migrate_system_data(
                system_name, 
                max_months if max_months > 0 else None,
                progress_callback
            )
        
        progress_bar.progress(1.0)
        
        if results['success']:
            st.success(f"Migration completed! Processed {results['successful']}/{results['total_files']} files successfully")
        else:
            st.error(f"Migration completed with errors. {results['successful']}/{results['total_files']} files successful")
        
        # Show detailed results
        with st.expander("Detailed Results"):
            for message in results['messages']:
                st.write(f"- {message}")

if __name__ == "__main__":
    run_migration_interface()