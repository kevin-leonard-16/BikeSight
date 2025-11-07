import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

from data_loader import DataLoader
from visualization import Visualizer
from metrics import MetricsCalculator
from utils import format_number
from migration_page import render_migration_page
from database_manager import get_database_manager

# Page configuration
st.set_page_config(
    page_title="Lyft Bikeshare Analytics Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize components
@st.cache_resource
def init_components():
    data_loader = DataLoader()
    visualizer = Visualizer()
    metrics_calc = MetricsCalculator()
    return data_loader, visualizer, metrics_calc

def render_live_data_page(data_loader, visualizer, metrics_calc):
    """Render the Live Data page with real-time GBFS data"""
    st.title("🚲 Live Bikeshare Data")
    st.markdown("Real-time station status and availability across BayWheels and Citi Bike systems")
    
    # System selection - now single selection only
    selected_system = st.selectbox(
        "Select Bikeshare System",
        options=["baywheels", "citibike"],
        index=0,
        format_func=lambda x: "Bay Wheels (SF Bay Area)" if x == "baywheels" else "Citi Bike (NYC)",
        help="Choose which bikeshare system to analyze"
    )
    
    if not selected_system:
        st.warning("Please select a bikeshare system to analyze.")
        return
    
    # Display options
    st.subheader("Display Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_offline_stations = st.checkbox("Include Offline Stations", value=False)
    with col2:
        min_bikes_filter = st.slider("Minimum Bikes Available", 0, 20, 0)
    with col3:
        if st.button("🔄 Refresh Data", help="Reload real-time station data"):
            st.cache_data.clear()
            st.rerun()
    
    # Load data for selected system
    system_data = {}
    
    with st.spinner("Loading real-time bikeshare data..."):
        data = data_loader.load_station_data(selected_system)
        if data:
            system_data[selected_system] = data
    
    if not system_data:
        st.error("No data could be loaded. Please check your internet connection and try again.")
        return
    
    # Apply filters
    filtered_data = {}
    if system_data and selected_system in system_data:
        data = system_data[selected_system]
        if data and 'stations' in data:
            df = data['stations'].copy()
            
            # Filter by minimum bikes
            if min_bikes_filter > 0:
                df = df[df['num_bikes_available'] >= min_bikes_filter]
            
            # Filter offline stations
            if not show_offline_stations:
                df = df[df['availability_status'] != 'Offline']
            
            filtered_data[selected_system] = {
                'stations': df,
                'metrics': data['metrics'],
                'last_updated': data['last_updated']
            }
    
    # System Overview Row
    st.header("📊 System Overview")
    
    if filtered_data and selected_system in filtered_data:
        data = filtered_data[selected_system]
        city_name = "San Francisco Bay Area" if selected_system == "baywheels" else "New York City"
        st.subheader(f"🏙️ {city_name}")
        
        metrics = data['metrics']
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Stations", format_number(metrics.get('total_stations', 0)))
        with col2:
            st.metric("Available Bikes", format_number(metrics.get('total_bikes', 0)))
        with col3:
            st.metric("Available Docks", format_number(metrics.get('total_docks', 0)))
        with col4:
            st.metric("E-Bikes", format_number(metrics.get('total_ebikes', 0)))
        
        # System utilization and E-bike percentage
        col1, col2 = st.columns(2)
        with col1:
            utilization = metrics.get('system_utilization', 0) * 100
            st.metric("System Utilization", f"{utilization:.1f}%")
        with col2:
            ebike_pct = metrics.get('ebike_percentage', 0)
            st.metric("E-Bike %", f"{ebike_pct:.1f}%")
    
    # Station Status Analysis
    st.header("🚦 Station Status Analysis")
    
    status_tab1, status_tab2 = st.tabs(["Availability Distribution", "System Health"])
    
    with status_tab1:
        if len(filtered_data) > 0:
            availability_chart = visualizer.create_availability_distribution(filtered_data)
            if availability_chart:
                st.plotly_chart(availability_chart, use_container_width=True)
    
    with status_tab2:
        if len(filtered_data) > 0:
            health_metrics = visualizer.create_system_health_chart(filtered_data)
            if health_metrics:
                st.plotly_chart(health_metrics, use_container_width=True)
    
    # Geographic Analysis
    st.header("🗺️ Station Network Map")
    
    geo_tab1, geo_tab2 = st.tabs(["Station Locations", "Utilization Heatmap"])
    
    with geo_tab1:
        if len(filtered_data) > 0:
            station_map = visualizer.create_station_map(filtered_data)
            if station_map:
                st.plotly_chart(station_map, use_container_width=True)
    
    with geo_tab2:
        if len(filtered_data) > 0:
            utilization_map = visualizer.create_utilization_heatmap(filtered_data)
            if utilization_map:
                st.plotly_chart(utilization_map, use_container_width=True)
    
    # Top Stations Analysis
    st.header("⭐ Top Stations")
    
    top_tab1, top_tab2, top_tab3 = st.tabs(["Most Bikes", "Highest Utilization", "Largest Capacity"])
    
    with top_tab1:
        for system, data in filtered_data.items():
            if data and 'stations' in data:
                city_name = "San Francisco Bay Area" if system == "baywheels" else "New York City"
                st.subheader(f"{city_name} - Stations with Most Bikes")
                
                top_stations = data_loader.get_top_stations(data['stations'], 'num_bikes_available', 10)
                if not top_stations.empty:
                    st.dataframe(
                        top_stations[['name', 'num_bikes_available', 'availability_status']],
                        column_config={
                            "name": "Station Name",
                            "num_bikes_available": "Available Bikes",
                            "availability_status": "Status"
                        },
                        hide_index=True
                    )
    
    with top_tab2:
        for system, data in filtered_data.items():
            if data and 'stations' in data:
                city_name = "San Francisco Bay Area" if system == "baywheels" else "New York City"
                st.subheader(f"{city_name} - Highest Utilization")
                
                top_stations = data_loader.get_top_stations(data['stations'], 'utilization_rate', 10)
                if not top_stations.empty:
                    # Add formatted utilization percentage
                    if 'utilization_rate' in top_stations.columns:
                        top_stations['utilization_pct'] = (top_stations['utilization_rate'] * 100).round(1).astype(str) + '%'
                    
                    # Select available columns for display
                    display_columns = ['name']
                    column_config = {"name": "Station Name"}
                    
                    if 'utilization_pct' in top_stations.columns:
                        display_columns.append('utilization_pct')
                        column_config["utilization_pct"] = "Utilization"
                    elif 'utilization_rate' in top_stations.columns:
                        display_columns.append('utilization_rate')
                        column_config["utilization_rate"] = "Utilization"
                    
                    if 'num_bikes_available' in top_stations.columns:
                        display_columns.append('num_bikes_available')
                        column_config["num_bikes_available"] = "Bikes"
                    
                    if 'num_docks_available' in top_stations.columns:
                        display_columns.append('num_docks_available')
                        column_config["num_docks_available"] = "Docks"
                    
                    st.dataframe(
                        top_stations[display_columns],
                        column_config=column_config,
                        hide_index=True
                    )
    
    with top_tab3:
        for system, data in filtered_data.items():
            if data and 'stations' in data:
                city_name = "San Francisco Bay Area" if system == "baywheels" else "New York City"
                st.subheader(f"{city_name} - Largest Stations")
                
                top_stations = data_loader.get_top_stations(data['stations'], 'capacity', 10)
                if not top_stations.empty:
                    st.dataframe(
                        top_stations[['name', 'capacity', 'num_bikes_available', 'num_docks_available']],
                        column_config={
                            "name": "Station Name",
                            "capacity": "Total Capacity",
                            "num_bikes_available": "Available Bikes",
                            "num_docks_available": "Available Docks"
                        },
                        hide_index=True
                    )
    
    # Service Area Map
    st.header("🗺️ Service Area Map")
    if filtered_data and selected_system in filtered_data:
        basemap = visualizer.create_service_area_basemap(selected_system, filtered_data[selected_system])
        if basemap:
            st.plotly_chart(basemap, use_container_width=True)

def render_historical_data_page(data_loader, visualizer, metrics_calc):
    """Render the Historical Data page with trip data analysis"""
    st.title("📈 BayWheels Analytics Dashboard")
    st.markdown("Explore bikeshare trip patterns and ridership trends from the San Francisco Bay Area")
    
    # Get database manager
    db = get_database_manager()
    selected_system = "baywheels"  # Hardcoded to BayWheels only
    
    # Get the most recent month of data from database
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT MAX(data_month) 
            FROM data_load_log 
            WHERE system_name = 'baywheels' 
            AND load_status = 'success'
        """)
        result = cursor.fetchone()
        latest_month_date = result[0] if result else None
    
    if not latest_month_date:
        st.error("No BayWheels data found in database. Please load data using the Database Migration page first.")
        return
    
    # Set default date range to the full most recent month
    from datetime import timedelta
    from calendar import monthrange
    
    default_start_date = latest_month_date.replace(day=1)
    days_in_month = monthrange(latest_month_date.year, latest_month_date.month)[1]
    default_end_date = default_start_date + timedelta(days=days_in_month - 1)
    
    # Date range picker
    st.subheader("📅 Select Date Range")
    
    # Initialize session state for dates if not set
    if 'selected_start_date' not in st.session_state:
        st.session_state.selected_start_date = default_start_date
    if 'selected_end_date' not in st.session_state:
        st.session_state.selected_end_date = default_end_date
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.selected_start_date,
            min_value=datetime(2023, 1, 1),
            max_value=datetime.now(),
            key="start_date_picker"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.selected_end_date,
            min_value=datetime(2023, 1, 1),
            max_value=datetime.now(),
            key="end_date_picker"
        )
    
    # Update session state when date inputs change
    st.session_state.selected_start_date = start_date
    st.session_state.selected_end_date = end_date
    
    # Event quick select buttons
    st.markdown("**Quick Select Events:**")
    event_col1, event_col2, event_col3 = st.columns([1, 1, 2])
    
    with event_col1:
        if st.button("🎵 Outside Lands", help="August 8-10, 2025", use_container_width=True):
            st.session_state.selected_start_date = datetime(2025, 8, 8).date()
            st.session_state.selected_end_date = datetime(2025, 8, 10).date()
            st.rerun()
    
    with event_col2:
        if st.button("🎪 Portola Festival", help="September 20-21, 2025", use_container_width=True):
            st.session_state.selected_start_date = datetime(2025, 9, 20).date()
            st.session_state.selected_end_date = datetime(2025, 9, 21).date()
            st.rerun()
    
    # Validate date range
    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        return
    
    # Query database for trip data within the date range
    with st.spinner("Loading trip data from database..."):
        # Query with date range (add 1 day to end_date to include the entire end day)
        from datetime import timedelta
        end_date_inclusive = end_date + timedelta(days=1)
        
        trip_df = db.get_trips_data(
            system_name=selected_system,
            start_date=start_date,
            end_date=end_date_inclusive
        )
    
    if trip_df.empty:
        st.error(f"No trip data found for the selected date range ({start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}).")
        return
    
    # Store in dictionary format for compatibility with existing visualization code
    trip_data = {"baywheels_daterange": trip_df}
    
    # Calculate metrics from the trip data
    total_trips_calculated = len(trip_df)
    
    # Calculate member/casual breakdown
    if 'member_type' in trip_df.columns:
        member_counts = trip_df['member_type'].value_counts()
        total_member = member_counts.get('member', 0)
        total_casual = member_counts.get('casual', 0)
    else:
        total_member = 0
        total_casual = 0
    
    # Calculate averages
    avg_duration = trip_df['duration_minutes'].mean() if 'duration_minutes' in trip_df.columns else None
    
    # Calculate unique routes
    unique_routes = 0
    if 'start_station_id' in trip_df.columns and 'end_station_id' in trip_df.columns:
        # Filter out null values and same-station trips
        valid_trips = trip_df[
            (trip_df['start_station_id'].notna()) & 
            (trip_df['end_station_id'].notna()) &
            (trip_df['start_station_id'] != trip_df['end_station_id'])
        ]
        if not valid_trips.empty:
            # Create route identifier and count unique combinations
            route_pairs = valid_trips['start_station_id'].astype(str) + '-' + valid_trips['end_station_id'].astype(str)
            unique_routes = route_pairs.nunique()
    
    # Show data summary as toast notification
    st.toast(f"📊 Data Loaded: {total_trips_calculated:,} trips from {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}", icon="✅")
    
    # Display summary metrics and charts
    st.header("📊 Summary Statistics")
    
    # Generate intelligent data summary
    def generate_data_summary(trip_df, total_member, total_casual, avg_duration, start_date, end_date):
        """Generate an intelligent summary of the data patterns - limited to 3 combined bullet points"""
        summary_parts = []
        
        # Date context - format the date range
        if start_date == end_date:
            time_context = f"On {start_date.strftime('%B %d, %Y')}"
        elif start_date.month == end_date.month and start_date.year == end_date.year:
            if start_date.day == 1 and end_date.day == monthrange(end_date.year, end_date.month)[1]:
                # Full month
                time_context = f"During {start_date.strftime('%B %Y')}"
            else:
                # Partial month
                time_context = f"From {start_date.strftime('%B %d')} to {end_date.strftime('%d, %Y')}"
        else:
            # Multi-month range
            time_context = f"From {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
        
        # Bullet 1: Volume and user breakdown combined
        total_rides = total_member + total_casual
        if total_rides > 0:
            volume_text = f"{time_context}, there were <strong>{total_rides:,} total bike rides</strong> in the San Francisco Bay Area"
            
            if total_member > 0 and total_casual > 0:
                member_pct = (total_member / total_rides) * 100
                if member_pct > 70:
                    volume_text += f", with <strong>members dominating usage at {member_pct:.0f}%</strong> of all trips, showing strong subscriber loyalty"
                elif member_pct > 50:
                    volume_text += f", where <strong>members made up {member_pct:.0f}%</strong> of trips and casual riders accounted for the remaining {100-member_pct:.0f}%"
                else:
                    casual_pct = 100 - member_pct
                    volume_text += f", with <strong>casual riders being very active at {casual_pct:.0f}%</strong> of trips, suggesting strong tourist or occasional usage"
            
            summary_parts.append(volume_text + ".")
        
        # Bullet 2: Trip duration
        if avg_duration and not pd.isna(avg_duration):
            if avg_duration < 10:
                duration_text = f"<strong>Trips were quite short</strong> with an average duration of {avg_duration:.1f} minutes, typical of commuting or quick errands"
            elif avg_duration < 20:
                duration_text = f"<strong>Average trip length was {avg_duration:.1f} minutes</strong>, indicating a mix of commuting and recreational riding"
            else:
                duration_text = f"<strong>Longer rides dominated</strong> with an average of {avg_duration:.1f} minutes, suggesting recreational or tourist usage"
            
            summary_parts.append(duration_text + ".")
        
        # Bullet 3: Peak hour analysis
        if 'start_time' in trip_df.columns and not trip_df.empty:
            if not pd.api.types.is_datetime64_any_dtype(trip_df['start_time']):
                trip_df['start_time'] = pd.to_datetime(trip_df['start_time'])
            
            hourly_counts = trip_df['start_time'].dt.hour.value_counts()
            if len(hourly_counts) > 0:
                peak_hour = hourly_counts.idxmax()
                
                if 7 <= peak_hour <= 9:
                    peak_text = f"<strong>Morning commute peak occurred at {peak_hour}:00</strong>, showing strong work-related usage patterns with riders likely traveling to offices and workplaces"
                elif 17 <= peak_hour <= 19:
                    peak_text = f"<strong>Evening rush hour peak was at {peak_hour}:00</strong>, indicating commuter return trips as people head home from work"
                elif 12 <= peak_hour <= 14:
                    peak_text = f"<strong>Lunch hour peak at {peak_hour}:00</strong> suggests recreational or meal-related trips, with people using bikes for midday activities"
                elif 20 <= peak_hour <= 23:
                    peak_text = f"<strong>Evening entertainment peak at {peak_hour}:00</strong> shows nightlife and leisure activity, with riders using bikes for social outings"
                else:
                    peak_text = f"<strong>Peak usage occurred at {peak_hour}:00</strong>, showing unique riding patterns that may indicate special events or local usage preferences"
                
                summary_parts.append(peak_text + ".")
        
        return summary_parts
    
    # Generate and display the summary
    summary_parts = generate_data_summary(trip_df, total_member, total_casual, avg_duration, start_date, end_date)
    
    # Display summary in a styled info box with bullet points
    if summary_parts:
        # Limit to 3 combined bullet points
        bullet_points = ""
        for i, part in enumerate(summary_parts[:3]):
            if part:
                bullet_points += f"<li style='margin-bottom: 12px;'>{part}</li>"
        
        st.markdown(
            f"""
            <div style="
                background: #e7f3ff;
                padding: 10px;
                border-radius: 15px;
                margin: 20px 0;
                border: 1px solid #bee5eb;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            ">
                <div style="
                    background: #e7f3ff;
                    padding: 20px;
                    border-radius: 10px;
                    color: #0c5460;
                    line-height: 1.7;
                    font-size: 16px;
                ">
                    <h4 style="color: #0c5460; margin-top: 0; margin-bottom: 15px; font-weight: 600;">
                        🔍 Data Insights
                    </h4>
                    <ul style="margin: 0; padding-left: 20px; list-style-type: disc;">
                        {bullet_points}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Metrics row layout: 5 metrics in a single row
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    # Total trips metric
    with col1:
        st.metric("Total Trips", f"{total_trips_calculated:,}")
    
    # Unique routes metric
    with col2:
        st.metric("Unique Routes", f"{unique_routes:,}")
    
    # Duration metric
    with col3:
        st.metric("Avg Duration", f"{avg_duration:.0f} min" if not pd.isna(avg_duration) else "N/A")
    
    # Member trips percentage
    with col4:
        if not pd.isna(total_member) and not pd.isna(total_casual) and total_trips_calculated > 0:
            member_pct = (total_member / total_trips_calculated) * 100
            st.metric("Member Trips", f"{member_pct:.0f}%")
        else:
            st.metric("Member Trips", "N/A")
    
    # E-bike trips percentage
    with col5:
        # Calculate bike type totals from raw trip data
        ebike_total = 0
        regular_total = 0
        
        for key, df in trip_data.items():
            if 'bike_type' in df.columns:
                bike_counts = df['bike_type'].value_counts()
                # Look for common e-bike identifiers
                for bike_type, count in bike_counts.items():
                    if any(keyword in str(bike_type).lower() for keyword in ['electric', 'ebike', 'e-bike', 'docked_ebike']):
                        ebike_total += count
                    else:
                        regular_total += count
            elif 'rideable_type' in df.columns:
                bike_counts = df['rideable_type'].value_counts()
                for bike_type, count in bike_counts.items():
                    if any(keyword in str(bike_type).lower() for keyword in ['electric', 'ebike', 'e-bike', 'docked_ebike']):
                        ebike_total += count
                    else:
                        regular_total += count
        
        bike_total = ebike_total + regular_total
        if bike_total > 0:
            ebike_pct = (ebike_total / bike_total) * 100
            st.metric("E-Bike Trips", f"{ebike_pct:.0f}%")
        else:
            st.metric("E-Bike Trips", "N/A")
    
    # Full-width hourly distribution histogram
    st.header("📊 Hourly Ride Distribution")
    
    # Extract hour data from trip data
    if 'start_time' in trip_df.columns and not trip_df.empty:
        # Ensure start_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(trip_df['start_time']):
            trip_df['start_time'] = pd.to_datetime(trip_df['start_time'])
        
        # Extract hour and count
        hourly_counts = trip_df['start_time'].dt.hour.value_counts().sort_index()
        
        # Create full 24-hour range
        hours = list(range(24))
        counts = [hourly_counts.get(h, 0) for h in hours]
        
        fig_hourly = go.Figure(data=[go.Bar(
            x=hours,
            y=counts,
            marker_color='#9467bd',
            name="Rides by Hour"
        )])
        
        fig_hourly.update_layout(
            title="When rides start throughout the day",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Rides",
            height=400,
            showlegend=False,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=4,  # Show every 4 hours
                tickfont=dict(size=12)
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    else:
        st.info("No time data available for hourly distribution")
    
    # Station Popularity Heat Map
    st.header("🗺️ Station Popularity Heat Map")
    st.caption("💡 This map shows station activity by combining trips that start and end at each location. A station with 1,000 trips means 1,000 total departures and arrivals combined.")
    
    # Calculate station popularity from trip data
    station_popularity = {}
    
    # Count trips starting at each station
    if 'start_station_id' in trip_df.columns and 'start_station_name' in trip_df.columns:
        start_counts = trip_df.groupby(['start_station_id', 'start_station_name']).size()
        for (station_id, station_name), count in start_counts.items():
            if station_id not in station_popularity:
                station_popularity[station_id] = {'name': station_name, 'trips': 0, 'start_trips': 0, 'end_trips': 0, 'coords': None}
            station_popularity[station_id]['start_trips'] += count
            station_popularity[station_id]['trips'] += count
    
    # Count trips ending at each station  
    if 'end_station_id' in trip_df.columns and 'end_station_name' in trip_df.columns:
        end_counts = trip_df.groupby(['end_station_id', 'end_station_name']).size()
        for (station_id, station_name), count in end_counts.items():
            if station_id not in station_popularity:
                station_popularity[station_id] = {'name': station_name, 'trips': 0, 'start_trips': 0, 'end_trips': 0, 'coords': None}
            station_popularity[station_id]['end_trips'] += count
            station_popularity[station_id]['trips'] += count
    
    # Get coordinates from trip data
    if not trip_df.empty:
        # Find available coordinate columns
        start_lat_cols = [col for col in trip_df.columns if 'start' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
        start_lon_cols = [col for col in trip_df.columns if 'start' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
        end_lat_cols = [col for col in trip_df.columns if 'end' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
        end_lon_cols = [col for col in trip_df.columns if 'end' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
        
        # Extract start station coordinates
        if start_lat_cols and start_lon_cols:
            start_lat_col = start_lat_cols[0]
            start_lon_col = start_lon_cols[0]
            
            start_coords = trip_df.groupby('start_station_id').agg({
                start_lat_col: 'first',
                start_lon_col: 'first'
            }).to_dict('index')
            
            for station_id, coords in start_coords.items():
                if station_id in station_popularity:
                    lat = coords.get(start_lat_col)
                    lon = coords.get(start_lon_col)
                    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                        station_popularity[station_id]['coords'] = (lat, lon)
        
        # Extract end station coordinates
        if end_lat_cols and end_lon_cols:
            end_lat_col = end_lat_cols[0]
            end_lon_col = end_lon_cols[0]
            
            end_coords = trip_df.groupby('end_station_id').agg({
                end_lat_col: 'first',
                end_lon_col: 'first'
            }).to_dict('index')
            
            for station_id, coords in end_coords.items():
                if station_id in station_popularity and not station_popularity[station_id]['coords']:
                    lat = coords.get(end_lat_col)
                    lon = coords.get(end_lon_col)
                    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                        station_popularity[station_id]['coords'] = (lat, lon)
    
    # Create 3D column map
    selected_city = "All Cities"  # Hardcoded since we removed city filter
    if station_popularity:
        column_map = visualizer.create_station_column_map(selected_system, station_popularity, selected_city)
        if column_map:
            # Display 3D map full width
            st.pydeck_chart(column_map, use_container_width=True)
            
            # Create top 10 stations table underneath the map
            # Sort stations by trip count
            sorted_stations = sorted(
                station_popularity.items(),
                key=lambda x: x[1]['trips'],
                reverse=True
            )[:10]
            
            # Prepare table data
            table_data = []
            for i, (station_id, data) in enumerate(sorted_stations, 1):
                table_data.append({
                    'Rank': i,
                    'Station Name': data['name'],
                    'Trip Starts': f"{data['start_trips']:,}",
                    'Trip Ends': f"{data['end_trips']:,}",
                    'Total Trips': f"{data['trips']:,}"
                })
            
            # Display as DataFrame
            stations_df = pd.DataFrame(table_data)
            st.dataframe(
                stations_df,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        help="Station popularity ranking"
                    ),
                    "Station Name": st.column_config.TextColumn(
                        "Station Name",
                        help="Name of the station"
                    ),
                    "Trip Starts": st.column_config.TextColumn(
                        "Trip Starts",
                        help="Number of trips that started at this station"
                    ),
                    "Trip Ends": st.column_config.TextColumn(
                        "Trip Ends",
                        help="Number of trips that ended at this station"
                    ),
                    "Total Trips": st.column_config.TextColumn(
                        "Total Trips",
                        help="Total number of trips starting or ending at this station"
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No station data available for map")
    
    # Top 10 Routes Map
    st.header("🚴 Top 10 Routes")
    st.caption("💡 Routes are bidirectional, meaning trips from Station A to Station B and from Station B to Station A are counted together as one route.")
    
    # Get popular routes using optimized SQL query
    popular_routes = []
    try:
        db_manager = get_database_manager()
        routes_df = db_manager.get_top_routes(selected_system, start_date, end_date_inclusive, top_n=10)
        
        # Convert DataFrame to list of dictionaries for compatibility with existing code
        if not routes_df.empty:
            popular_routes = routes_df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading routes: {str(e)}")
    
    # Create route map with Pydeck
    if popular_routes:
        route_map = visualizer.create_route_heatmap(selected_system, popular_routes, selected_city)
        if route_map:
            # Display map using pydeck_chart
            st.pydeck_chart(route_map, use_container_width=True)
            
            # Create table underneath the map (no header)
            # Prepare table data
            table_data = []
            for i, route in enumerate(popular_routes, 1):
                table_data.append({
                    'Rank': i,
                    'Start Station': route['start_station_name'],
                    'End Station': route['end_station_name'],
                    'Trips': f"{int(route['trip_count']):,}"
                })
            
            # Display as DataFrame
            routes_df = pd.DataFrame(table_data)
            st.dataframe(
                routes_df,
                column_config={
                    "Rank": st.column_config.NumberColumn(
                        "Rank",
                        help="Route popularity ranking"
                    ),
                    "Start Station": st.column_config.TextColumn(
                        "Start Station",
                        help="Origin station"
                    ),
                    "End Station": st.column_config.TextColumn(
                        "End Station",
                        help="Destination station"
                    ),
                    "Trips": st.column_config.TextColumn(
                        "Trips",
                        help="Number of trips on this route"
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.info("No route data available for map")

def main():
    # Initialize components
    data_loader, visualizer, metrics_calc = init_components()
    
    # Render the historical data page directly (no navigation)
    render_historical_data_page(data_loader, visualizer, metrics_calc)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard powered by BayWheels historical trip data stored in PostgreSQL*")

if __name__ == "__main__":
    main()