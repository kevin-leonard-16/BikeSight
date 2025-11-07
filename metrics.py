import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import streamlit as st

class MetricsCalculator:
    """Handles calculation of various bikeshare metrics and KPIs"""
    
    def __init__(self):
        pass
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic ridership metrics"""
        if df is None or df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Total rides
            metrics['total_rides'] = len(df)
            
            # Average trip duration
            if 'duration_minutes' in df.columns:
                metrics['avg_duration'] = df['duration_minutes'].mean()
                metrics['median_duration'] = df['duration_minutes'].median()
                metrics['total_duration_hours'] = df['duration_minutes'].sum() / 60
            
            # Unique stations
            if 'start_station_name' in df.columns:
                metrics['unique_start_stations'] = df['start_station_name'].nunique()
            
            if 'end_station_name' in df.columns:
                metrics['unique_end_stations'] = df['end_station_name'].nunique()
            
            # Peak hour
            if 'start_hour' in df.columns:
                peak_hour_series = df['start_hour'].mode()
                metrics['peak_hour'] = peak_hour_series.iloc[0] if not peak_hour_series.empty else 0
            
            # User type breakdown
            if 'user_type' in df.columns:
                user_type_counts = df['user_type'].value_counts(normalize=True)
                for user_type, percentage in user_type_counts.items():
                    metrics[f'pct_{user_type.lower().replace(" ", "_")}'] = percentage * 100
            
            # Bike type breakdown
            if 'bike_type' in df.columns:
                bike_type_counts = df['bike_type'].value_counts(normalize=True)
                for bike_type, percentage in bike_type_counts.items():
                    metrics[f'pct_{bike_type.lower().replace(" ", "_")}_bike'] = percentage * 100
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def calculate_temporal_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate temporal pattern metrics"""
        if df is None or df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Daily patterns
            if 'start_hour' in df.columns:
                hourly_counts = df.groupby('start_hour').size()
                metrics['peak_hour'] = hourly_counts.idxmax()
                metrics['peak_hour_count'] = hourly_counts.max()
                metrics['off_peak_hour'] = hourly_counts.idxmin()
                metrics['peak_to_offpeak_ratio'] = hourly_counts.max() / hourly_counts.min() if hourly_counts.min() > 0 else 0
            
            # Weekly patterns
            if 'start_day_of_week' in df.columns:
                daily_counts = df.groupby('start_day_of_week').size()
                metrics['busiest_day'] = daily_counts.idxmax()
                metrics['quietest_day'] = daily_counts.idxmin()
                
                # Weekend vs weekday usage
                weekend_days = ['Saturday', 'Sunday']
                weekday_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                
                weekend_rides = df[df['start_day_of_week'].isin(weekend_days)]
                weekday_rides = df[df['start_day_of_week'].isin(weekday_days)]
                
                if len(weekend_rides) > 0 and len(weekday_rides) > 0:
                    metrics['weekend_to_weekday_ratio'] = len(weekend_rides) / len(weekday_rides)
                    metrics['pct_weekend_usage'] = (len(weekend_rides) / len(df)) * 100
            
            # Monthly patterns
            if hasattr(df.index, 'month'):
                monthly_counts = df.groupby(df.index.month).size()
                metrics['busiest_month'] = monthly_counts.idxmax()
                metrics['quietest_month'] = monthly_counts.idxmin()
                metrics['seasonal_variation_coeff'] = monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating temporal metrics: {str(e)}")
            return {}
    
    def calculate_spatial_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate spatial/geographic metrics"""
        if df is None or df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Station popularity
            if 'start_station_name' in df.columns:
                station_counts = df['start_station_name'].value_counts()
                metrics['most_popular_start_station'] = station_counts.index[0] if not station_counts.empty else None
                metrics['most_popular_start_station_count'] = station_counts.iloc[0] if not station_counts.empty else 0
                
                # Station usage distribution
                metrics['station_usage_concentration'] = (station_counts.iloc[0] / station_counts.sum()) * 100 if not station_counts.empty else 0
                
                # Top 10 stations percentage
                top_10_pct = (station_counts.head(10).sum() / station_counts.sum()) * 100 if len(station_counts) >= 10 else 100
                metrics['top_10_stations_pct'] = top_10_pct
            
            # Route analysis
            if 'start_station_name' in df.columns and 'end_station_name' in df.columns:
                df_copy = df.copy()
                df_copy['route'] = df_copy['start_station_name'] + ' → ' + df_copy['end_station_name']
                route_counts = df_copy['route'].value_counts()
                
                metrics['most_popular_route'] = route_counts.index[0] if not route_counts.empty else None
                metrics['most_popular_route_count'] = route_counts.iloc[0] if not route_counts.empty else 0
                
                # Round trip analysis
                round_trips = df_copy[df_copy['start_station_name'] == df_copy['end_station_name']]
                metrics['round_trip_count'] = len(round_trips)
                metrics['round_trip_percentage'] = (len(round_trips) / len(df_copy)) * 100 if len(df_copy) > 0 else 0
            
            # Geographic spread
            if all(col in df.columns for col in ['start_latitude', 'start_longitude']):
                valid_coords = df[(df['start_latitude'] != 0) & (df['start_longitude'] != 0) & 
                                (df['start_latitude'].notna()) & (df['start_longitude'].notna())]
                
                if not valid_coords.empty:
                    lat_range = valid_coords['start_latitude'].max() - valid_coords['start_latitude'].min()
                    lon_range = valid_coords['start_longitude'].max() - valid_coords['start_longitude'].min()
                    
                    metrics['latitude_range'] = lat_range
                    metrics['longitude_range'] = lon_range
                    metrics['geographic_spread_area'] = lat_range * lon_range  # Rough approximation
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating spatial metrics: {str(e)}")
            return {}
    
    def calculate_user_behavior_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate user behavior and usage pattern metrics"""
        if df is None or df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Duration analysis
            if 'duration_minutes' in df.columns:
                duration_stats = df['duration_minutes'].describe()
                metrics['duration_25th_percentile'] = duration_stats['25%']
                metrics['duration_75th_percentile'] = duration_stats['75%']
                metrics['duration_iqr'] = duration_stats['75%'] - duration_stats['25%']
                
                # Short vs long trips
                short_trips = df[df['duration_minutes'] <= 30]  # 30 minutes or less
                long_trips = df[df['duration_minutes'] > 60]    # More than 1 hour
                
                metrics['short_trip_percentage'] = (len(short_trips) / len(df)) * 100
                metrics['long_trip_percentage'] = (len(long_trips) / len(df)) * 100
            
            # User type behavior
            if 'user_type' in df.columns and 'duration_minutes' in df.columns:
                user_duration = df.groupby('user_type')['duration_minutes'].agg(['mean', 'median', 'count'])
                
                for user_type in user_duration.index:
                    safe_user_type = user_type.lower().replace(' ', '_').replace('/', '_')
                    metrics[f'{safe_user_type}_avg_duration'] = user_duration.loc[user_type, 'mean']
                    metrics[f'{safe_user_type}_median_duration'] = user_duration.loc[user_type, 'median']
                    metrics[f'{safe_user_type}_trip_count'] = user_duration.loc[user_type, 'count']
            
            # Hourly usage patterns by user type
            if 'user_type' in df.columns and 'start_hour' in df.columns:
                user_hour_patterns = df.groupby(['user_type', 'start_hour']).size().unstack(fill_value=0)
                
                for user_type in user_hour_patterns.index:
                    peak_hour = user_hour_patterns.loc[user_type].idxmax()
                    safe_user_type = user_type.lower().replace(' ', '_').replace('/', '_')
                    metrics[f'{safe_user_type}_peak_hour'] = peak_hour
            
            # Bike type preferences
            if 'bike_type' in df.columns and 'user_type' in df.columns:
                bike_user_cross = pd.crosstab(df['user_type'], df['bike_type'], normalize='index') * 100
                
                for user_type in bike_user_cross.index:
                    for bike_type in bike_user_cross.columns:
                        safe_user_type = user_type.lower().replace(' ', '_').replace('/', '_')
                        safe_bike_type = bike_type.lower().replace(' ', '_')
                        metrics[f'{safe_user_type}_{safe_bike_type}_preference'] = bike_user_cross.loc[user_type, bike_type]
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating user behavior metrics: {str(e)}")
            return {}
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, any]:
        """Calculate system performance and efficiency metrics"""
        if df is None or df.empty:
            return {}
        
        metrics = {}
        
        try:
            # Rides per day
            if hasattr(df.index, 'date'):
                daily_rides = df.groupby(df.index.date).size()
                metrics['avg_rides_per_day'] = daily_rides.mean()
                metrics['max_rides_per_day'] = daily_rides.max()
                metrics['min_rides_per_day'] = daily_rides.min()
                metrics['rides_per_day_std'] = daily_rides.std()
            
            # Station utilization
            if 'start_station_name' in df.columns:
                station_usage = df['start_station_name'].value_counts()
                total_stations = station_usage.count()
                
                # Gini coefficient for station usage inequality
                if total_stations > 1:
                    sorted_usage = np.sort(station_usage.values)
                    n = len(sorted_usage)
                    cumsum = np.cumsum(sorted_usage)
                    metrics['station_usage_gini'] = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
                
                # Station efficiency metrics
                metrics['avg_rides_per_station'] = station_usage.mean()
                metrics['station_usage_coefficient_variation'] = station_usage.std() / station_usage.mean() if station_usage.mean() > 0 else 0
            
            # Network efficiency
            if 'start_station_name' in df.columns and 'end_station_name' in df.columns:
                unique_origins = df['start_station_name'].nunique()
                unique_destinations = df['end_station_name'].nunique()
                total_possible_routes = unique_origins * unique_destinations
                
                df_copy = df.copy()
                df_copy['route'] = df_copy['start_station_name'] + ' → ' + df_copy['end_station_name']
                actual_routes = df_copy['route'].nunique()
                
                metrics['route_utilization_rate'] = (actual_routes / total_possible_routes) * 100 if total_possible_routes > 0 else 0
                metrics['unique_routes'] = actual_routes
                metrics['total_possible_routes'] = total_possible_routes
            
            # Temporal efficiency
            if 'start_hour' in df.columns:
                hourly_usage = df.groupby('start_hour').size()
                peak_hour_usage = hourly_usage.max()
                off_peak_usage = hourly_usage.min()
                
                metrics['capacity_utilization_variance'] = hourly_usage.var()
                metrics['peak_utilization_efficiency'] = (hourly_usage.mean() / peak_hour_usage) * 100 if peak_hour_usage > 0 else 0
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def calculate_comparative_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Calculate comparative metrics between different systems"""
        if len(data) < 2:
            return {}
        
        metrics = {}
        
        try:
            system_metrics = {}
            
            # Calculate basic metrics for each system
            for system, df in data.items():
                if df is not None and not df.empty:
                    system_metrics[system] = self.calculate_basic_metrics(df)
            
            if len(system_metrics) < 2:
                return {}
            
            # Compare systems
            systems = list(system_metrics.keys())
            system1, system2 = systems[0], systems[1]
            
            # Ridership comparison
            if 'total_rides' in system_metrics[system1] and 'total_rides' in system_metrics[system2]:
                rides1 = system_metrics[system1]['total_rides']
                rides2 = system_metrics[system2]['total_rides']
                
                metrics['ridership_ratio'] = rides1 / rides2 if rides2 > 0 else 0
                metrics['ridership_difference'] = rides1 - rides2
                metrics['ridership_difference_pct'] = ((rides1 - rides2) / rides2) * 100 if rides2 > 0 else 0
            
            # Duration comparison
            if 'avg_duration' in system_metrics[system1] and 'avg_duration' in system_metrics[system2]:
                duration1 = system_metrics[system1]['avg_duration']
                duration2 = system_metrics[system2]['avg_duration']
                
                metrics['duration_ratio'] = duration1 / duration2 if duration2 > 0 else 0
                metrics['duration_difference'] = duration1 - duration2
                metrics['duration_difference_pct'] = ((duration1 - duration2) / duration2) * 100 if duration2 > 0 else 0
            
            # Station coverage comparison
            if 'unique_start_stations' in system_metrics[system1] and 'unique_start_stations' in system_metrics[system2]:
                stations1 = system_metrics[system1]['unique_start_stations']
                stations2 = system_metrics[system2]['unique_start_stations']
                
                metrics['station_coverage_ratio'] = stations1 / stations2 if stations2 > 0 else 0
                metrics['station_coverage_difference'] = stations1 - stations2
            
            # User type distribution comparison
            for user_type in ['subscriber', 'member', 'customer', 'casual']:
                key = f'pct_{user_type}'
                if key in system_metrics[system1] and key in system_metrics[system2]:
                    pct1 = system_metrics[system1][key]
                    pct2 = system_metrics[system2][key]
                    metrics[f'{user_type}_distribution_difference'] = pct1 - pct2
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating comparative metrics: {str(e)}")
            return {}
