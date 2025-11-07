import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import time
import zipfile
import io
import os
from urllib.parse import urlparse
import numpy as np
from math import radians, cos, sin, asin, sqrt
import psycopg2
from psycopg2.extras import RealDictCursor
import json

class DataLoader:
    """Handles loading and preprocessing of real-time bikeshare data from Bay Wheels and Citi Bike GBFS feeds"""
    
    def __init__(self):
        self.gbfs_bay_wheels = "https://gbfs.lyft.com/gbfs/2.3/bay/en"
        self.gbfs_citibike = "https://gbfs.lyft.com/gbfs/2.3/bkn/en"
    
    def load_station_data(self, system: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Load station information and real-time status for a bikeshare system"""
        try:
            base_url = self.gbfs_bay_wheels if system == "baywheels" else self.gbfs_citibike
            
            # Load station information (static data)
            station_info_url = f"{base_url}/station_information.json"
            station_status_url = f"{base_url}/station_status.json"
            
            with st.spinner(f"Loading {system} station data..."):
                # Get station information
                info_response = requests.get(station_info_url, timeout=10)
                info_response.raise_for_status()
                station_info = info_response.json()
                
                # Get station status
                status_response = requests.get(station_status_url, timeout=10)
                status_response.raise_for_status()
                station_status = status_response.json()
            
            # Convert to DataFrames
            info_df = pd.DataFrame(station_info['data']['stations'])
            status_df = pd.DataFrame(station_status['data']['stations'])
            
            # Merge station info and status
            merged_df = pd.merge(info_df, status_df, on='station_id', how='left')
            
            # Add calculated fields
            merged_df['utilization_rate'] = merged_df['num_bikes_available'] / (merged_df['num_bikes_available'] + merged_df['num_docks_available'])
            merged_df['utilization_rate'] = merged_df['utilization_rate'].fillna(0)
            
            # Categorize stations by availability
            merged_df['availability_status'] = merged_df.apply(self._categorize_availability, axis=1)
            
            # Add system identifier
            merged_df['system'] = system
            merged_df['city'] = "San Francisco Bay Area" if system == "baywheels" else "New York City"
            
            # Calculate summary metrics
            summary_metrics = self._calculate_system_metrics(merged_df)
            
            return {
                'stations': merged_df,
                'metrics': summary_metrics,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.error(f"Error loading {system} station data: {str(e)}")
            return None
    
    def _categorize_availability(self, row):
        """Categorize station availability status"""
        if not row.get('is_renting', 1) or not row.get('is_returning', 1):
            return "Offline"
        elif row.get('num_bikes_available', 0) == 0:
            return "Empty"
        elif row.get('num_docks_available', 0) == 0:
            return "Full"
        elif row.get('num_bikes_available', 0) <= 2:
            return "Low"
        else:
            return "Available"
    
    def _calculate_system_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate system-wide metrics from station data"""
        try:
            total_stations = len(df)
            active_stations = len(df[df['is_installed'] == 1])
            
            total_bikes = df['num_bikes_available'].sum()
            total_docks = df['num_docks_available'].sum()
            total_capacity = total_bikes + total_docks
            
            # E-bike metrics
            total_ebikes = df['num_ebikes_available'].sum()
            
            # Availability metrics
            empty_stations = len(df[df['num_bikes_available'] == 0])
            full_stations = len(df[df['num_docks_available'] == 0])
            
            # System utilization
            system_utilization = total_bikes / total_capacity if total_capacity > 0 else 0
            
            return {
                'total_stations': total_stations,
                'active_stations': active_stations,
                'total_bikes': total_bikes,
                'total_docks': total_docks,
                'total_capacity': total_capacity,
                'total_ebikes': total_ebikes,
                'empty_stations': empty_stations,
                'full_stations': full_stations,
                'system_utilization': system_utilization,
                'ebike_percentage': (total_ebikes / total_bikes * 100) if total_bikes > 0 else 0
            }
        except Exception as e:
            st.warning(f"Error calculating metrics: {str(e)}")
            return {}
    
    def load_system_info(self, system: str) -> Optional[Dict]:
        """Load general system information"""
        try:
            base_url = self.gbfs_bay_wheels if system == "baywheels" else self.gbfs_citibike
            system_info_url = f"{base_url}/system_information.json"
            
            response = requests.get(system_info_url, timeout=10)
            response.raise_for_status()
            
            return response.json()['data']
            
        except Exception as e:
            st.warning(f"Could not load system info for {system}: {str(e)}")
            return None
    
    def get_station_summary_by_region(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get station summary grouped by region"""
        try:
            if 'region_id' not in df.columns:
                # If no region info, create a single region
                df['region_id'] = 1
                df['region_name'] = df['city'].iloc[0] if not df.empty else "Unknown"
            
            summary = df.groupby('region_id').agg({
                'station_id': 'count',
                'num_bikes_available': 'sum',
                'num_docks_available': 'sum',
                'num_ebikes_available': 'sum',
                'capacity': 'sum',
                'utilization_rate': 'mean'
            }).round(3)
            
            summary.columns = ['stations', 'bikes_available', 'docks_available', 'ebikes_available', 'total_capacity', 'avg_utilization']
            
            return summary.reset_index()
            
        except Exception as e:
            st.warning(f"Error creating region summary: {str(e)}")
            return pd.DataFrame()
    
    def get_top_stations(self, df: pd.DataFrame, metric: str = 'num_bikes_available', n: int = 10) -> pd.DataFrame:
        """Get top N stations by specified metric"""
        try:
            if metric not in df.columns:
                return pd.DataFrame()
            
            # Include all the columns that might be needed for display
            base_columns = ['name', 'short_name', metric, 'lat', 'lon', 'availability_status']
            optional_columns = ['num_bikes_available', 'num_docks_available', 'num_ebikes_available', 'capacity', 'utilization_rate']
            
            # Add available optional columns
            columns_to_include = base_columns.copy()
            for col in optional_columns:
                if col in df.columns and col not in columns_to_include:
                    columns_to_include.append(col)
            
            top_stations = df.nlargest(n, metric)[columns_to_include]
            return top_stations
            
        except Exception as e:
            st.warning(f"Error getting top stations: {str(e)}")
            return pd.DataFrame()
    
    def compare_systems(self, bay_data: Dict, citibike_data: Dict) -> Dict:
        """Compare metrics between Bay Wheels and Citi Bike systems"""
        try:
            if not bay_data or not citibike_data:
                return {}
            
            bay_metrics = bay_data.get('metrics', {})
            citibike_metrics = citibike_data.get('metrics', {})
            
            comparison = {}
            
            for metric in bay_metrics.keys():
                if metric in citibike_metrics:
                    bay_val = bay_metrics[metric]
                    citibike_val = citibike_metrics[metric]
                    
                    comparison[metric] = {
                        'baywheels': bay_val,
                        'citibike': citibike_val,
                        'difference': bay_val - citibike_val if isinstance(bay_val, (int, float)) else None,
                        'ratio': bay_val / citibike_val if isinstance(bay_val, (int, float)) and citibike_val != 0 else None
                    }
            
            return comparison
            
        except Exception as e:
            st.warning(f"Error comparing systems: {str(e)}")
            return {}

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_historical_trip_data(_self, system: str, year_month: str) -> Optional[pd.DataFrame]:
        """Load historical trip data from CSV files in zip archives"""
        try:
            if system == "baywheels":
                base_url = "https://s3.amazonaws.com/baywheels-data"
                # Format: 202501-baywheels-tripdata.csv.zip
                filename = f"{year_month}-baywheels-tripdata.csv.zip"
            elif system == "citibike":
                base_url = "https://s3.amazonaws.com/tripdata"
                # Format: 202501-citibike-tripdata.csv.zip
                filename = f"{year_month}-citibike-tripdata.csv.zip"
            else:
                st.error(f"Unsupported system: {system}")
                return None
            
            url = f"{base_url}/{filename}"
            
            with st.spinner(f"Loading {system} trip data for {year_month}..."):
                # Download the zip file
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Extract CSV from zip
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    # Get the first CSV file in the zip
                    csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        st.error(f"No CSV files found in {filename}")
                        return None
                    
                    csv_filename = csv_files[0]
                    
                    # Read CSV data
                    with zip_file.open(csv_filename) as csv_file:
                        df = pd.read_csv(csv_file, low_memory=False)
                
                # Standardize column names for both systems
                df = _self._standardize_trip_columns(df, system)
                
                # Add system and month info
                df['system'] = system
                df['year_month'] = year_month
                df['city'] = "San Francisco Bay Area" if system == "baywheels" else "New York City"
                
                # Apply data cleansing
                df_clean = _self._cleanse_trip_data(df, system)
                
                return df_clean
                
        except requests.exceptions.RequestException as e:
            st.warning(f"Could not download {system} data for {year_month}: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error processing {system} trip data for {year_month}: {str(e)}")
            return None
    
    def _standardize_trip_columns(self, df: pd.DataFrame, system: str) -> pd.DataFrame:
        """Standardize column names across different bikeshare systems"""
        try:
            # Common column mappings
            if system == "baywheels":
                column_mapping = {
                    'started_at': 'start_time',
                    'ended_at': 'end_time', 
                    'start_station_name': 'start_station_name',
                    'end_station_name': 'end_station_name',
                    'start_station_id': 'start_station_id',
                    'end_station_id': 'end_station_id',
                    'rideable_type': 'bike_type',
                    'member_casual': 'user_type'
                }
            elif system == "citibike":
                column_mapping = {
                    'started_at': 'start_time',
                    'ended_at': 'end_time',
                    'start_station_name': 'start_station_name', 
                    'end_station_name': 'end_station_name',
                    'start_station_id': 'start_station_id',
                    'end_station_id': 'end_station_id',
                    'rideable_type': 'bike_type',
                    'member_casual': 'user_type'
                }
            else:
                return df
            
            # Rename columns that exist
            existing_mapping = {old: new for old, new in column_mapping.items() if old in df.columns}
            df = df.rename(columns=existing_mapping)
            
            # Convert datetime columns
            datetime_cols = ['start_time', 'end_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Calculate trip duration in minutes
            if 'start_time' in df.columns and 'end_time' in df.columns:
                df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
                # Filter out unreasonable trip durations (< 1 min or > 24 hours)
                df = df[(df['duration_minutes'] >= 1) & (df['duration_minutes'] <= 1440)]
            
            return df
            
        except Exception as e:
            st.warning(f"Error standardizing columns: {str(e)}")
            return df
    
    def get_available_months(self, system: str, start_year: int = 2023, end_year: int = 2025) -> List[str]:
        """Get list of available months for historical data"""
        months = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Don't include future months
                current_date = datetime.now()
                if year > current_date.year or (year == current_date.year and month > current_date.month):
                    break
                    
                year_month = f"{year}{month:02d}"
                months.append(year_month)
        
        return months
    
    def get_most_recent_available_data(self, system: str, start_year: int = 2023, end_year: int = 2025) -> Optional[str]:
        """Get the most recent data available in S3 bucket by checking file modification dates"""
        try:
            if system == "baywheels":
                base_url = "https://s3.amazonaws.com/baywheels-data"
            elif system == "citibike":
                base_url = "https://s3.amazonaws.com/tripdata"
            else:
                return None
            
            most_recent_date = None
            most_recent_month = None
            
            # Check files in reverse chronological order (most recent first)
            for year in range(end_year, start_year - 1, -1):
                for month in range(12, 0, -1):
                    # Don't check future months
                    current_date = datetime.now()
                    if year > current_date.year or (year == current_date.year and month > current_date.month):
                        continue
                        
                    year_month = f"{year}{month:02d}"
                    
                    if system == "baywheels":
                        filename = f"{year_month}-baywheels-tripdata.csv.zip"
                    else:
                        filename = f"{year_month}-citibike-tripdata.csv.zip"
                    
                    url = f"{base_url}/{filename}"
                    
                    try:
                        # Make HEAD request to check if file exists and get metadata
                        response = requests.head(url, timeout=10)
                        if response.status_code == 200:
                            # Get last modified date
                            last_modified = response.headers.get('Last-Modified')
                            if last_modified:
                                try:
                                    # Parse the date string
                                    file_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S %Z')
                                    
                                    # If this is the first valid file or it's more recent than our current best
                                    if most_recent_date is None or file_date > most_recent_date:
                                        most_recent_date = file_date
                                        most_recent_month = year_month
                                        
                                    # Since we're going in reverse chronological order,
                                    # if we found a file, this is likely the most recent
                                    # Continue checking a few more months to be thorough
                                    if most_recent_month:
                                        break
                                except ValueError:
                                    # If we can't parse the date, skip this file
                                    continue
                    except requests.RequestException:
                        # If request fails, continue to next month
                        continue
                
                # If we found a recent month, break out of year loop too
                if most_recent_month:
                    break
            
            return most_recent_month
            
        except Exception as e:
            st.warning(f"Could not determine most recent data: {str(e)}")
            return None
    
    def calculate_monthly_metrics(self, trip_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate monthly ridership metrics from trip data"""
        try:
            monthly_stats = []
            
            for year_month, df in trip_data.items():
                if df is None or df.empty:
                    continue
                
                stats = {
                    'year_month': year_month,
                    'system': df['system'].iloc[0] if 'system' in df.columns else 'unknown',
                    'city': df['city'].iloc[0] if 'city' in df.columns else 'unknown',
                    'total_rides': len(df),
                    'avg_duration': df['duration_minutes'].mean() if 'duration_minutes' in df.columns else None,
                    'avg_distance': (df['trip_distance_feet'].mean() / 5280) if 'trip_distance_feet' in df.columns else None,  # Convert to miles
                    'member_rides': len(df[df['user_type'] == 'member']) if 'user_type' in df.columns else None,
                    'casual_rides': len(df[df['user_type'] == 'casual']) if 'user_type' in df.columns else None,
                }
                
                # Add bike type breakdown if available
                if 'bike_type' in df.columns:
                    bike_types = df['bike_type'].value_counts()
                    for bike_type, count in bike_types.items():
                        stats[f'{bike_type}_rides'] = count
                
                monthly_stats.append(stats)
            
            return pd.DataFrame(monthly_stats)
            
        except Exception as e:
            st.error(f"Error calculating monthly metrics: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on earth in feet"""
        try:
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            # Radius of earth in feet (3959 miles * 5280 feet/mile)
            r_feet = 3959 * 5280
            
            return c * r_feet
            
        except:
            return np.nan
    
    def _cleanse_trip_data(self, df: pd.DataFrame, system: str) -> pd.DataFrame:
        """Apply data quality filters to trip data"""
        try:
            if df is None or df.empty:
                return df
                
            original_count = len(df)
            
            # Create a copy to avoid modifying original
            clean_df = df.copy()
            
            # Filter 1: Remove trips with invalid durations (already done in standardize_columns)
            # Duration filters: 1 min <= duration <= 24 hours are already applied
            
            # Filter 2: Remove trips with negative durations or future dates
            if 'start_time' in clean_df.columns and 'end_time' in clean_df.columns:
                clean_df = clean_df[clean_df['start_time'] <= clean_df['end_time']].copy()
                
                # Remove future trips (beyond current time)
                current_time = datetime.now()
                clean_df = clean_df[clean_df['start_time'] <= current_time].copy()
                
                # Remove very old trips (before 2010)
                cutoff_date = datetime(2010, 1, 1)
                clean_df = clean_df[clean_df['start_time'] >= cutoff_date].copy()
            
            # Filter 3: Remove trips with missing critical data
            critical_columns = []
            if 'start_station_id' in clean_df.columns:
                critical_columns.append('start_station_id')
            if 'end_station_id' in clean_df.columns:
                critical_columns.append('end_station_id')
            
            if critical_columns:
                clean_df = clean_df.dropna(subset=critical_columns).copy()
            
            # Filter 4: Calculate distance and apply distance-based filters
            lat_cols = [col for col in clean_df.columns if 'start_lat' in col.lower() or 'start_latitude' in col.lower()]
            lon_cols = [col for col in clean_df.columns if 'start_lng' in col.lower() or 'start_longitude' in col.lower()]
            end_lat_cols = [col for col in clean_df.columns if 'end_lat' in col.lower() or 'end_latitude' in col.lower()]
            end_lon_cols = [col for col in clean_df.columns if 'end_lng' in col.lower() or 'end_longitude' in col.lower()]
            
            start_lat_col = None
            start_lon_col = None
            end_lat_col = None
            end_lon_col = None
            
            if lat_cols and lon_cols and end_lat_cols and end_lon_cols:
                start_lat_col = lat_cols[0]
                start_lon_col = lon_cols[0]
                end_lat_col = end_lat_cols[0]
                end_lon_col = end_lon_cols[0]
                
                # Remove trips with invalid coordinates (0,0 or null)
                coord_mask = (
                    (clean_df[start_lat_col] != 0) & (clean_df[start_lon_col] != 0) &
                    (clean_df[end_lat_col] != 0) & (clean_df[end_lon_col] != 0) &
                    (clean_df[start_lat_col].notna()) & (clean_df[start_lon_col].notna()) &
                    (clean_df[end_lat_col].notna()) & (clean_df[end_lon_col].notna())
                )
                clean_df = clean_df[coord_mask].copy()
                
                # Calculate trip distance
                if len(clean_df) > 0:
                    clean_df['trip_distance_feet'] = clean_df.apply(
                        lambda row: self._calculate_haversine_distance(
                            row[start_lat_col], row[start_lon_col],
                            row[end_lat_col], row[end_lon_col]
                        ), axis=1
                    )
                    
                    # Filter by distance: remove very short trips (≤ 5 feet) and very long trips (> 50 miles)
                    distance_mask = (
                        (clean_df['trip_distance_feet'] > 5) & 
                        (clean_df['trip_distance_feet'] <= 50 * 5280)  # 50 miles in feet
                    )
                    clean_df = clean_df[distance_mask].copy()
                    
                    # Calculate average speed and filter unrealistic speeds
                    if 'duration_minutes' in clean_df.columns and len(clean_df) > 0:
                        clean_df['avg_speed_mph'] = (clean_df['trip_distance_feet'] / 5280) / (clean_df['duration_minutes'] / 60)
                        # Remove trips with average speed > 30 mph (unrealistic for bikes)
                        speed_mask = clean_df['avg_speed_mph'] <= 30
                        clean_df = clean_df[speed_mask].copy()
            
            # Filter 5: Remove same start/end station trips with very short duration
            if ('start_station_id' in clean_df.columns and 
                'end_station_id' in clean_df.columns and 
                'duration_minutes' in clean_df.columns and 
                len(clean_df) > 0):
                
                # Allow same station trips only if duration >= 2 minutes
                same_station_mask = (clean_df['start_station_id'] == clean_df['end_station_id'])
                valid_same_station = same_station_mask & (clean_df['duration_minutes'] >= 2)
                different_stations = ~same_station_mask
                
                clean_df = clean_df[valid_same_station | different_stations].copy()
            
            # Filter 6: Geographic bounds check
            if (start_lat_col and start_lon_col and end_lat_col and end_lon_col and 
                len(clean_df) > 0):
                
                if system == "baywheels":
                    # San Francisco Bay Area bounds (approximate)
                    lat_bounds = (37.0, 38.5)
                    lon_bounds = (-123.0, -121.0)
                elif system == "citibike":
                    # New York City bounds (approximate)  
                    lat_bounds = (40.4, 41.0)
                    lon_bounds = (-74.5, -73.5)
                else:
                    lat_bounds = None
                    lon_bounds = None
                
                if lat_bounds and lon_bounds:
                    bounds_mask = (
                        (clean_df[start_lat_col] >= lat_bounds[0]) & (clean_df[start_lat_col] <= lat_bounds[1]) &
                        (clean_df[start_lon_col] >= lon_bounds[0]) & (clean_df[start_lon_col] <= lon_bounds[1]) &
                        (clean_df[end_lat_col] >= lat_bounds[0]) & (clean_df[end_lat_col] <= lat_bounds[1]) &
                        (clean_df[end_lon_col] >= lon_bounds[0]) & (clean_df[end_lon_col] <= lon_bounds[1])
                    )
                    clean_df = clean_df[bounds_mask].copy()
            
            filtered_count = len(clean_df)
            removed_count = original_count - filtered_count
            
            # Store cleansing stats for display at bottom of page
            if not hasattr(st.session_state, 'cleansing_stats'):
                st.session_state.cleansing_stats = []
            
            if removed_count > 0:
                removal_pct = (removed_count / original_count) * 100
                cleansing_info = {
                    'system': system,
                    'removed': removed_count,
                    'remaining': filtered_count,
                    'percentage': removal_pct,
                    'original': original_count
                }
                st.session_state.cleansing_stats.append(cleansing_info)
            
            return clean_df
            
        except Exception as e:
            st.warning(f"Error during data cleansing: {str(e)}")
            return df
    
    def extract_cities_from_trip_data(self, df: pd.DataFrame, system: str) -> List[str]:
        """Extract unique cities from trip data using reverse geocoding"""
        cities = set()
        
        try:
            # Find coordinate columns
            start_lat_cols = [col for col in df.columns if 'start' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            start_lon_cols = [col for col in df.columns if 'start' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            end_lat_cols = [col for col in df.columns if 'end' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            end_lon_cols = [col for col in df.columns if 'end' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            
            # Process start coordinates
            if start_lat_cols and start_lon_cols:
                start_lat_col = start_lat_cols[0]
                start_lon_col = start_lon_cols[0]
                
                # Sample coordinates to find cities (don't process every row for performance)
                sample_size = min(1000, len(df))
                sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
                
                for _, row in sample_df.iterrows():
                    lat = row.get(start_lat_col)
                    lon = row.get(start_lon_col)
                    if pd.notna(lat) and pd.notna(lon):
                        city = self._reverse_geocode_city(lat, lon, system)
                        if city:
                            cities.add(city)
            
            # Process end coordinates
            if end_lat_cols and end_lon_cols:
                end_lat_col = end_lat_cols[0]
                end_lon_col = end_lon_cols[0]
                
                # Sample coordinates to find cities
                sample_size = min(1000, len(df))
                sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
                
                for _, row in sample_df.iterrows():
                    lat = row.get(end_lat_col)
                    lon = row.get(end_lon_col)
                    if pd.notna(lat) and pd.notna(lon):
                        city = self._reverse_geocode_city(lat, lon, system)
                        if city:
                            cities.add(city)
                
        except Exception as e:
            st.warning(f"Error extracting cities: {str(e)}")
            
        return sorted(list(cities)) if cities else []
    
    def _reverse_geocode_city(self, lat: float, lon: float, system: str) -> Optional[str]:
        """Reverse geocode coordinates to determine city using precise boundaries"""
        if pd.isna(lat) or pd.isna(lon):
            return None
            
        if system == "baywheels":
            # Precise Bay Area city boundaries
            city_bounds = {
                'San Francisco': {'lat': (37.708, 37.833), 'lon': (-122.515, -122.355)},
                'Oakland': {'lat': (37.754, 37.853), 'lon': (-122.330, -122.114)},
                'Berkeley': {'lat': (37.853, 37.906), 'lon': (-122.320, -122.234)},
                'Emeryville': {'lat': (37.831, 37.851), 'lon': (-122.310, -122.280)},
                'San Jose': {'lat': (37.250, 37.430), 'lon': (-121.950, -121.750)},
                'Redwood City': {'lat': (37.465, 37.515), 'lon': (-122.265, -122.200)},
                'Palo Alto': {'lat': (37.418, 37.475), 'lon': (-122.175, -122.100)},
                'Mountain View': {'lat': (37.380, 37.425), 'lon': (-122.125, -122.050)},
                'Sunnyvale': {'lat': (37.350, 37.405), 'lon': (-122.065, -121.980)},
                'Santa Clara': {'lat': (37.340, 37.365), 'lon': (-121.985, -121.930)},
                'Fremont': {'lat': (37.480, 37.565), 'lon': (-122.085, -121.940)},
                'Alameda': {'lat': (37.755, 37.785), 'lon': (-122.315, -122.235)},
                'Richmond': {'lat': (37.930, 37.970), 'lon': (-122.400, -122.330)},
                'Vallejo': {'lat': (38.090, 38.125), 'lon': (-122.275, -122.230)},
                'San Mateo': {'lat': (37.540, 37.575), 'lon': (-122.335, -122.295)},
                'Foster City': {'lat': (37.555, 37.570), 'lon': (-122.275, -122.250)},
                'Hayward': {'lat': (37.645, 37.685), 'lon': (-122.110, -122.055)},
                'Union City': {'lat': (37.590, 37.610), 'lon': (-122.045, -122.010)},
                'San Leandro': {'lat': (37.715, 37.735), 'lon': (-122.175, -122.150)}
            }
        elif system == "citibike":
            # NYC boroughs/areas
            city_bounds = {
                'Manhattan': {'lat': (40.700, 40.790), 'lon': (-74.020, -73.930)},
                'Brooklyn': {'lat': (40.570, 40.740), 'lon': (-74.050, -73.850)},
                'Queens': {'lat': (40.540, 40.800), 'lon': (-73.970, -73.700)},
                'Bronx': {'lat': (40.785, 40.915), 'lon': (-73.933, -73.765)},
                'Jersey City': {'lat': (40.695, 40.760), 'lon': (-74.085, -74.025)},
                'Hoboken': {'lat': (40.735, 40.755), 'lon': (-74.045, -74.015)}
            }
        else:
            return None
            
        # Find the city that contains these coordinates
        for city, bounds in city_bounds.items():
            if (bounds['lat'][0] <= lat <= bounds['lat'][1] and 
                bounds['lon'][0] <= lon <= bounds['lon'][1]):
                return city
                
        return None
    
    def _extract_cities_from_coordinates(self, df: pd.DataFrame) -> set:
        """Extract cities from geographic coordinates for Bay Area"""
        cities = set()
        
        # Find coordinate columns
        lat_cols = [col for col in df.columns if 'lat' in col.lower() and 'start' in col.lower()]
        lon_cols = [col for col in df.columns if ('lng' in col.lower() or 'lon' in col.lower()) and 'start' in col.lower()]
        
        if not lat_cols or not lon_cols:
            return cities
            
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        
        # Define approximate city boundaries for Bay Area
        city_bounds = {
            'San Francisco': {'lat': (37.7, 37.84), 'lon': (-122.52, -122.35)},
            'Oakland': {'lat': (37.75, 37.85), 'lon': (-122.32, -122.15)},
            'Berkeley': {'lat': (37.85, 37.89), 'lon': (-122.30, -122.23)},
            'San Jose': {'lat': (37.25, 37.45), 'lon': (-122.0, -121.75)},
            'Palo Alto': {'lat': (37.42, 37.47), 'lon': (-122.17, -122.10)},
            'Mountain View': {'lat': (37.38, 37.42), 'lon': (-122.12, -122.05)},
            'Redwood City': {'lat': (37.46, 37.51), 'lon': (-122.26, -122.20)}
        }
        
        for city, bounds in city_bounds.items():
            city_mask = (
                (df[lat_col] >= bounds['lat'][0]) & (df[lat_col] <= bounds['lat'][1]) &
                (df[lon_col] >= bounds['lon'][0]) & (df[lon_col] <= bounds['lon'][1])
            )
            if city_mask.any():
                cities.add(city)
                
        return cities
        
    def filter_data_by_city(self, df: pd.DataFrame, city: str, system: str) -> pd.DataFrame:
        """Filter trip data by selected city using reverse geocoding"""
        if city == "All Cities" or df.empty:
            return df
            
        try:
            # Find coordinate columns
            start_lat_cols = [col for col in df.columns if 'start' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            start_lon_cols = [col for col in df.columns if 'start' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            end_lat_cols = [col for col in df.columns if 'end' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            end_lon_cols = [col for col in df.columns if 'end' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            
            if not (start_lat_cols and start_lon_cols and end_lat_cols and end_lon_cols):
                return df
                
            start_lat_col = start_lat_cols[0]
            start_lon_col = start_lon_cols[0]
            end_lat_col = end_lat_cols[0]
            end_lon_col = end_lon_cols[0]
            
            # Add city columns to DataFrame for efficient filtering
            if f'start_city_{system}' not in df.columns:
                df[f'start_city_{system}'] = df.apply(
                    lambda row: self._reverse_geocode_city(row[start_lat_col], row[start_lon_col], system)
                    if pd.notna(row[start_lat_col]) and pd.notna(row[start_lon_col]) else None, 
                    axis=1
                )
                
            if f'end_city_{system}' not in df.columns:
                df[f'end_city_{system}'] = df.apply(
                    lambda row: self._reverse_geocode_city(row[end_lat_col], row[end_lon_col], system)
                    if pd.notna(row[end_lat_col]) and pd.notna(row[end_lon_col]) else None, 
                    axis=1
                )
            
            # Filter trips that start OR end in the selected city
            city_mask = (df[f'start_city_{system}'] == city) | (df[f'end_city_{system}'] == city)
            filtered_df = df[city_mask].copy()
            
            return filtered_df
            
        except Exception as e:
            st.warning(f"Error filtering by city: {str(e)}")
            return df
    
    def _filter_by_coordinates(self, df: pd.DataFrame, city: str) -> pd.DataFrame:
        """Filter data by city using coordinate boundaries"""
        # Find coordinate columns
        lat_cols = [col for col in df.columns if 'lat' in col.lower() and 'start' in col.lower()]
        lon_cols = [col for col in df.columns if ('lng' in col.lower() or 'lon' in col.lower()) and 'start' in col.lower()]
        
        if not lat_cols or not lon_cols:
            return df
            
        lat_col = lat_cols[0] 
        lon_col = lon_cols[0]
        
        # City boundaries (same as defined in _extract_cities_from_coordinates)
        city_bounds = {
            'San Francisco': {'lat': (37.7, 37.84), 'lon': (-122.52, -122.35)},
            'Oakland': {'lat': (37.75, 37.85), 'lon': (-122.32, -122.15)},
            'Berkeley': {'lat': (37.85, 37.89), 'lon': (-122.30, -122.23)},
            'San Jose': {'lat': (37.25, 37.45), 'lon': (-122.0, -121.75)},
            'Palo Alto': {'lat': (37.42, 37.47), 'lon': (-122.17, -122.10)},
            'Mountain View': {'lat': (37.38, 37.42), 'lon': (-122.12, -122.05)},
            'Redwood City': {'lat': (37.46, 37.51), 'lon': (-122.26, -122.20)}
        }
        
        if city not in city_bounds:
            return df
            
        bounds = city_bounds[city]
        city_mask = (
            (df[lat_col] >= bounds['lat'][0]) & (df[lat_col] <= bounds['lat'][1]) &
            (df[lon_col] >= bounds['lon'][0]) & (df[lon_col] <= bounds['lon'][1])
        )
        
        return df[city_mask]
    
    def extract_popular_routes(self, df: pd.DataFrame, system: str, top_n: int = 5) -> List[Dict]:
        """Extract top N most popular routes from trip data"""
        try:
            if df.empty:
                return []
                
            # Find required columns
            start_station_cols = [col for col in df.columns if 'start_station_id' in col.lower()]
            end_station_cols = [col for col in df.columns if 'end_station_id' in col.lower()]
            start_name_cols = [col for col in df.columns if 'start_station_name' in col.lower()]
            end_name_cols = [col for col in df.columns if 'end_station_name' in col.lower()]
            start_lat_cols = [col for col in df.columns if 'start' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            start_lon_cols = [col for col in df.columns if 'start' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            end_lat_cols = [col for col in df.columns if 'end' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
            end_lon_cols = [col for col in df.columns if 'end' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
            
            if not all([start_station_cols, end_station_cols, start_name_cols, end_name_cols, 
                       start_lat_cols, start_lon_cols, end_lat_cols, end_lon_cols]):
                return []
                
            start_station_col = start_station_cols[0]
            end_station_col = end_station_cols[0]
            start_name_col = start_name_cols[0]
            end_name_col = end_name_cols[0]
            start_lat_col = start_lat_cols[0]
            start_lon_col = start_lon_cols[0]
            end_lat_col = end_lat_cols[0]
            end_lon_col = end_lon_cols[0]
            
            # Create route pairs (considering bidirectional routes)
            routes = {}
            for _, row in df.iterrows():
                start_id = row[start_station_col]
                end_id = row[end_station_col]
                start_name = row[start_name_col]
                end_name = row[end_name_col]
                start_lat = row[start_lat_col]
                start_lon = row[start_lon_col]
                end_lat = row[end_lat_col]
                end_lon = row[end_lon_col]
                
                # Skip if any required data is missing
                if pd.isna(start_id) or pd.isna(end_id) or pd.isna(start_lat) or pd.isna(start_lon) or pd.isna(end_lat) or pd.isna(end_lon):
                    continue
                    
                # Create bidirectional route key (smaller ID first to treat A->B and B->A as same route)
                if start_id < end_id:
                    route_key = f"{start_id}_{end_id}"
                    route_start_id, route_end_id = start_id, end_id
                    route_start_name, route_end_name = start_name, end_name
                    route_start_lat, route_start_lon = start_lat, start_lon
                    route_end_lat, route_end_lon = end_lat, end_lon
                else:
                    route_key = f"{end_id}_{start_id}"
                    route_start_id, route_end_id = end_id, start_id
                    route_start_name, route_end_name = end_name, start_name
                    route_start_lat, route_start_lon = end_lat, end_lon
                    route_end_lat, route_end_lon = start_lat, start_lon
                    
                if route_key not in routes:
                    routes[route_key] = {
                        'start_station_id': route_start_id,
                        'end_station_id': route_end_id,
                        'start_station_name': route_start_name,
                        'end_station_name': route_end_name,
                        'start_lat': route_start_lat,
                        'start_lon': route_start_lon,
                        'end_lat': route_end_lat,
                        'end_lon': route_end_lon,
                        'trip_count': 0
                    }
                    
                routes[route_key]['trip_count'] += 1
            
            # Sort by trip count and get top N
            sorted_routes = sorted(routes.values(), key=lambda x: x['trip_count'], reverse=True)
            top_routes = sorted_routes[:top_n]
            
            # Store routes in database
            self._store_routes_in_database(top_routes, system)
            
            return top_routes
            
        except Exception as e:
            st.warning(f"Error extracting popular routes: {str(e)}")
            return []
    
    def _store_routes_in_database(self, routes: List[Dict], system: str):
        """Store route data in PostgreSQL database"""
        try:
            # Database connection details from environment
            conn = psycopg2.connect(
                host=os.getenv('PGHOST', 'localhost'),
                port=os.getenv('PGPORT', '5432'),
                database=os.getenv('PGDATABASE', 'postgres'),
                user=os.getenv('PGUSER', 'postgres'),
                password=os.getenv('PGPASSWORD', '')
            )
            
            cursor = conn.cursor()
            
            # Create routes table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS popular_routes (
                    id SERIAL PRIMARY KEY,
                    system VARCHAR(50) NOT NULL,
                    route_key VARCHAR(100) NOT NULL,
                    start_station_id VARCHAR(50) NOT NULL,
                    end_station_id VARCHAR(50) NOT NULL,
                    start_station_name TEXT,
                    end_station_name TEXT,
                    start_lat DECIMAL(10, 8),
                    start_lon DECIMAL(11, 8),
                    end_lat DECIMAL(10, 8),
                    end_lon DECIMAL(11, 8),
                    trip_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(system, route_key)
                );
            """)
            
            # Insert or update routes
            for route in routes:
                route_key = f"{route['start_station_id']}_{route['end_station_id']}"
                cursor.execute("""
                    INSERT INTO popular_routes 
                    (system, route_key, start_station_id, end_station_id, start_station_name, 
                     end_station_name, start_lat, start_lon, end_lat, end_lon, trip_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (system, route_key) 
                    DO UPDATE SET 
                        trip_count = EXCLUDED.trip_count,
                        created_at = CURRENT_TIMESTAMP;
                """, (
                    system, route_key, route['start_station_id'], route['end_station_id'],
                    route['start_station_name'], route['end_station_name'],
                    route['start_lat'], route['start_lon'], route['end_lat'], route['end_lon'],
                    route['trip_count']
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            st.warning(f"Error storing routes in database: {str(e)}")
    
    def get_bicycle_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Optional[List[Tuple[float, float]]]:
        """Get bicycle route coordinates using Valhalla routing API"""
        try:
            # Use Valhalla public API for bicycle routing with detailed polylines
            url = "https://valhalla1.openstreetmap.de/route"
            
            params = {
                'json': {
                    "locations": [
                        {"lat": start_lat, "lon": start_lon},
                        {"lat": end_lat, "lon": end_lon}
                    ],
                    "costing": "bicycle",
                    "costing_options": {
                        "bicycle": {
                            "bicycle_type": "Road",
                            "cycling_speed": 25,
                            "use_roads": 0.5,
                            "use_hills": 0.2
                        }
                    },
                    "directions_options": {
                        "units": "kilometers"
                    },
                    "shape_match": "map_snap",
                    "filters": {
                        "attributes": ["edge.length", "edge.speed"],
                        "action": "include"
                    }
                }
            }
            
            # Convert to URL parameter
            import json
            json_param = json.dumps(params['json'])
            
            response = requests.get(url, params={'json': json_param}, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'trip' in data and 'legs' in data['trip'] and len(data['trip']['legs']) > 0:
                    # Extract the encoded polyline
                    shape = data['trip']['legs'][0].get('shape')
                    if shape:
                        # Decode the polyline to get coordinates
                        route_coords = self._decode_polyline(shape)
                        if route_coords and len(route_coords) > 1:
                            return route_coords
            
            # Try alternative Valhalla instance
            url_alt = "https://valhalla.mapzen.com/route"
            response_alt = requests.get(url_alt, params={'json': json_param}, timeout=10)
            
            if response_alt.status_code == 200:
                data_alt = response_alt.json()
                if 'trip' in data_alt and 'legs' in data_alt['trip'] and len(data_alt['trip']['legs']) > 0:
                    shape = data_alt['trip']['legs'][0].get('shape')
                    if shape:
                        route_coords = self._decode_polyline(shape)
                        if route_coords and len(route_coords) > 1:
                            return route_coords
                            
        except Exception as e:
            pass  # Fall through to fallback
        
        # Fallback: Try OSRM public API for bicycle routing
        try:
            osrm_url = f"http://router.project-osrm.org/route/v1/bicycle/{start_lon},{start_lat};{end_lon},{end_lat}"
            params = {
                'overview': 'full',
                'geometries': 'polyline',
                'steps': 'false'
            }
            
            response = requests.get(osrm_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    geometry = data['routes'][0].get('geometry')
                    if geometry:
                        route_coords = self._decode_polyline(geometry)
                        if route_coords and len(route_coords) > 1:
                            return route_coords
                            
        except Exception as e:
            pass  # Fall through to final fallback
        
        # Final fallback: direct line
        return [(start_lat, start_lon), (end_lat, end_lon)]
    
    def _decode_polyline(self, polyline_str: str) -> List[Tuple[float, float]]:
        """Decode a polyline string to a list of (lat, lon) coordinates"""
        try:
            # Simple polyline decoder for Valhalla/OSRM encoded polylines
            coordinates = []
            index = 0
            lat = 0
            lng = 0
            
            while index < len(polyline_str):
                # Decode latitude
                result = 1
                shift = 0
                while True:
                    b = ord(polyline_str[index]) - 63 - 1
                    index += 1
                    result += b << shift
                    shift += 5
                    if b < 0x1f:
                        break
                lat += (~result >> 1) if (result & 1) != 0 else (result >> 1)
                
                # Decode longitude
                result = 1
                shift = 0
                while True:
                    b = ord(polyline_str[index]) - 63 - 1
                    index += 1
                    result += b << shift
                    shift += 5
                    if b < 0x1f:
                        break
                lng += (~result >> 1) if (result & 1) != 0 else (result >> 1)
                
                coordinates.append((lat / 1e5, lng / 1e5))
                
            return coordinates
            
        except Exception as e:
            return []