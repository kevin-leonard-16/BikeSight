import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st

def format_number(num: float, precision: int = 0) -> str:
    """Format numbers with appropriate suffixes (K, M, B)"""
    if pd.isna(num) or num == 0:
        return "0"
    
    try:
        if abs(num) >= 1_000_000_000:
            return f"{num/1_000_000_000:.{precision}f}B"
        elif abs(num) >= 1_000_000:
            return f"{num/1_000_000:.{precision}f}M"
        elif abs(num) >= 1_000:
            return f"{num/1_000:.{precision}f}K"
        else:
            return f"{num:.{precision}f}"
    except (ValueError, TypeError):
        return str(num)

def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable format"""
    if pd.isna(minutes) or minutes <= 0:
        return "0 min"
    
    try:
        if minutes >= 1440:  # More than a day
            days = int(minutes // 1440)
            remaining_hours = int((minutes % 1440) // 60)
            return f"{days}d {remaining_hours}h"
        elif minutes >= 60:  # More than an hour
            hours = int(minutes // 60)
            remaining_minutes = int(minutes % 60)
            return f"{hours}h {remaining_minutes}m"
        else:
            return f"{int(minutes)}m"
    except (ValueError, TypeError):
        return str(minutes)

def format_percentage(value: float, precision: int = 1) -> str:
    """Format percentage values"""
    if pd.isna(value):
        return "N/A"
    
    try:
        return f"{value:.{precision}f}%"
    except (ValueError, TypeError):
        return str(value)

def get_date_range_options() -> List[str]:
    """Get predefined date range options"""
    return [
        "Last 7 Days",
        "Last 30 Days", 
        "Last 3 Months",
        "Last 6 Months",
        "Last Year",
        "Year to Date",
        "Custom"
    ]

def calculate_date_range(option: str, custom_start: Optional[datetime] = None, 
                        custom_end: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Calculate start and end dates based on the selected option"""
    end_date = datetime.now()
    
    if option == "Last 7 Days":
        start_date = end_date - timedelta(days=7)
    elif option == "Last 30 Days":
        start_date = end_date - timedelta(days=30)
    elif option == "Last 3 Months":
        start_date = end_date - timedelta(days=90)
    elif option == "Last 6 Months":
        start_date = end_date - timedelta(days=180)
    elif option == "Last Year":
        start_date = end_date - timedelta(days=365)
    elif option == "Year to Date":
        start_date = datetime(end_date.year, 1, 1)
    elif option == "Custom":
        start_date = custom_start if custom_start else end_date - timedelta(days=30)
        end_date = custom_end if custom_end else end_date
    else:
        start_date = end_date - timedelta(days=30)  # Default to last 30 days
    
    return start_date, end_date

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate that a dataframe has the required columns"""
    if df is None or df.empty:
        return False, ["DataFrame is empty or None"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, [f"Missing required columns: {', '.join(missing_columns)}"]
    
    return True, []

def clean_station_name(name: str) -> str:
    """Clean and standardize station names"""
    if pd.isna(name) or not isinstance(name, str):
        return "Unknown Station"
    
    # Remove extra whitespace
    cleaned = name.strip()
    
    # Standardize common abbreviations
    replacements = {
        ' St ': ' Street ',
        ' Ave ': ' Avenue ',
        ' Blvd ': ' Boulevard ',
        ' Rd ': ' Road ',
        ' Dr ': ' Drive ',
        ' Pl ': ' Place ',
        ' Ct ': ' Court ',
        '&': 'and'
    }
    
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    return cleaned

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers"""
    try:
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    except (ValueError, TypeError):
        return np.nan

def categorize_trip_duration(duration_minutes: float) -> str:
    """Categorize trip duration into buckets"""
    if pd.isna(duration_minutes):
        return "Unknown"
    
    if duration_minutes < 10:
        return "Very Short (< 10 min)"
    elif duration_minutes < 30:
        return "Short (10-30 min)"
    elif duration_minutes < 60:
        return "Medium (30-60 min)"
    elif duration_minutes < 120:
        return "Long (1-2 hours)"
    else:
        return "Very Long (> 2 hours)"

def categorize_time_of_day(hour: int) -> str:
    """Categorize hour into time of day periods"""
    if pd.isna(hour) or not isinstance(hour, (int, float)):
        return "Unknown"
    
    hour = int(hour)
    
    if 5 <= hour < 9:
        return "Morning Rush (5-9 AM)"
    elif 9 <= hour < 12:
        return "Late Morning (9-12 PM)"
    elif 12 <= hour < 17:
        return "Afternoon (12-5 PM)"
    elif 17 <= hour < 20:
        return "Evening Rush (5-8 PM)"
    elif 20 <= hour < 23:
        return "Evening (8-11 PM)"
    else:
        return "Night (11 PM-5 AM)"

def get_season(month: int) -> str:
    """Get season from month number"""
    if pd.isna(month) or not isinstance(month, (int, float)):
        return "Unknown"
    
    month = int(month)
    
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Unknown"

def detect_outliers(series: pd.Series, method: str = "iqr", multiplier: float = 1.5) -> pd.Series:
    """Detect outliers in a pandas Series"""
    if series.empty or series.isna().all():
        return pd.Series(dtype=bool)
    
    if method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > multiplier
    
    else:
        return pd.Series([False] * len(series), index=series.index)

def aggregate_by_time_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Aggregate data by time period"""
    if df.empty or not hasattr(df.index, 'to_period'):
        return df
    
    try:
        if period == "hour":
            return df.groupby(df.index.hour).size().to_frame('count')
        elif period == "day":
            return df.groupby(df.index.date).size().to_frame('count')
        elif period == "week":
            return df.groupby(df.index.to_period('W')).size().to_frame('count')
        elif period == "month":
            return df.groupby(df.index.to_period('M')).size().to_frame('count')
        elif period == "quarter":
            return df.groupby(df.index.to_period('Q')).size().to_frame('count')
        elif period == "year":
            return df.groupby(df.index.to_period('Y')).size().to_frame('count')
        else:
            return df
    except Exception:
        return df

def create_summary_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Create summary statistics for a column"""
    if df.empty or column not in df.columns:
        return {}
    
    series = df[column].dropna()
    
    if series.empty:
        return {}
    
    try:
        stats = {
            'count': len(series),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q25': series.quantile(0.25),
            'q75': series.quantile(0.75),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis()
        }
        
        # Add mode for categorical or discrete data
        mode_series = series.mode()
        if not mode_series.empty:
            stats['mode'] = mode_series.iloc[0]
        
        return stats
    except Exception:
        return {'count': len(series)}

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def cache_key_from_params(**kwargs) -> str:
    """Generate a cache key from parameters"""
    import hashlib
    
    # Convert all parameters to strings and sort for consistent hashing
    param_str = str(sorted(kwargs.items()))
    return hashlib.md5(param_str.encode()).hexdigest()

def memory_usage_mb(df: pd.DataFrame) -> float:
    """Calculate memory usage of DataFrame in MB"""
    try:
        return df.memory_usage(deep=True).sum() / 1024 / 1024
    except Exception:
        return 0.0

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    if df.empty:
        return df
    
    optimized_df = df.copy()
    
    try:
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Convert object columns to category if they have low cardinality
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique values
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    except Exception:
        return df

@st.cache_data
def get_system_info() -> Dict[str, Dict[str, Any]]:
    """Get information about bikeshare systems"""
    return {
        'baywheels': {
            'name': 'Bay Wheels',
            'city': 'San Francisco Bay Area',
            'operator': 'Lyft',
            'launch_year': 2013,
            'coverage_area': 'San Francisco, Oakland, Berkeley, San Jose',
            'website': 'https://www.lyft.com/bikes/bay-wheels',
            'data_license': 'Bay Wheels License Agreement'
        },
        'citibike': {
            'name': 'Citi Bike',
            'city': 'New York City',
            'operator': 'Lyft',
            'launch_year': 2013,
            'coverage_area': 'Manhattan, Brooklyn, Queens, Bronx, Jersey City, Hoboken',
            'website': 'https://citibikenyc.com',
            'data_license': 'NYCBS Data Use Policy'
        }
    }
