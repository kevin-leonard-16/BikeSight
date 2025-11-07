import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import streamlit as st
import pydeck as pdk

class Visualizer:
    """Handles creation of interactive visualizations for real-time bikeshare station data"""
    
    def __init__(self):
        self.color_scheme = {
            'baywheels': '#1f77b4',  # Blue
            'citibike': '#ff7f0e'    # Orange
        }
        self.city_names = {
            'baywheels': 'San Francisco Bay Area',
            'citibike': 'New York City'
        }
        self.availability_colors = {
            'Available': '#2E8B57',    # Sea Green
            'Low': '#FFD700',          # Gold
            'Empty': '#DC143C',        # Crimson
            'Full': '#4169E1',         # Royal Blue
            'Offline': '#696969'       # Dim Gray
        }
        
        # Service area boundaries (approximate)
        self.service_areas = {
            'baywheels': {
                'center': {'lat': 37.7749, 'lon': -122.4194},  # San Francisco
                'bounds': {
                    'north': 38.2,
                    'south': 37.1, 
                    'east': -121.7,
                    'west': -123.0
                },
                'zoom': 9
            },
            'citibike': {
                'center': {'lat': 40.7589, 'lon': -73.9851},  # Manhattan
                'bounds': {
                    'north': 40.9,
                    'south': 40.4,
                    'east': -73.4, 
                    'west': -74.3
                },
                'zoom': 10
            }
        }
    
    def create_availability_distribution(self, data: Dict) -> Optional[go.Figure]:
        """Create availability status distribution chart"""
        try:
            fig = make_subplots(
                rows=1, cols=len(data),
                specs=[[{"type": "pie"}] * len(data)],
                subplot_titles=[self.city_names.get(system, system) for system in data.keys()]
            )
            
            for idx, (system, system_data) in enumerate(data.items(), 1):
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    
                    # Calculate availability distribution
                    availability_counts = df['availability_status'].value_counts()
                    
                    colors = [self.availability_colors.get(status, '#999999') for status in availability_counts.index]
                    
                    fig.add_trace(
                        go.Pie(
                            labels=availability_counts.index,
                            values=availability_counts.values,
                            name=self.city_names.get(system, system),
                            marker_colors=colors,
                            textinfo='label+percent'
                        ),
                        row=1, col=idx
                    )
            
            fig.update_layout(
                title="Station Availability Status Distribution",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating availability distribution chart: {str(e)}")
            return None
    
    def create_system_health_chart(self, data: Dict) -> Optional[go.Figure]:
        """Create system health metrics chart"""
        try:
            fig = go.Figure()
            
            systems = []
            empty_stations = []
            full_stations = []
            offline_stations = []
            total_stations = []
            
            for system, system_data in data.items():
                if system_data and 'metrics' in system_data:
                    metrics = system_data['metrics']
                    systems.append(self.city_names.get(system, system))
                    empty_stations.append(metrics.get('empty_stations', 0))
                    full_stations.append(metrics.get('full_stations', 0))
                    total_stations.append(metrics.get('total_stations', 0))
                    
                    # Calculate offline stations
                    df = system_data['stations']
                    offline_count = len(df[df['availability_status'] == 'Offline'])
                    offline_stations.append(offline_count)
            
            # Create stacked bar chart
            fig.add_trace(go.Bar(
                x=systems,
                y=empty_stations,
                name='Empty Stations',
                marker_color=self.availability_colors['Empty']
            ))
            
            fig.add_trace(go.Bar(
                x=systems,
                y=full_stations,
                name='Full Stations',
                marker_color=self.availability_colors['Full']
            ))
            
            fig.add_trace(go.Bar(
                x=systems,
                y=offline_stations,
                name='Offline Stations',
                marker_color=self.availability_colors['Offline']
            ))
            
            fig.update_layout(
                title="System Health Overview",
                xaxis_title="System",
                yaxis_title="Number of Stations",
                barmode='stack',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating system health chart: {str(e)}")
            return None
    
    def create_station_map(self, data: Dict) -> Optional[go.Figure]:
        """Create interactive station locations map"""
        try:
            fig = go.Figure()
            
            for system, system_data in data.items():
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    
                    # Filter out stations with invalid coordinates
                    valid_coords = df[
                        (df['lat'] != 0) & (df['lon'] != 0) & 
                        (df['lat'].notna()) & (df['lon'].notna())
                    ]
                    
                    if valid_coords.empty:
                        continue
                    
                    # Map availability status to marker colors
                    colors = [self.availability_colors.get(status, '#999999') for status in valid_coords['availability_status']]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=valid_coords['lat'],
                        lon=valid_coords['lon'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors,
                            opacity=0.7
                        ),
                        text=[f"<b>{name}</b><br>Bikes: {bikes}<br>Docks: {docks}<br>Status: {status}" 
                              for name, bikes, docks, status in zip(
                                  valid_coords['name'], 
                                  valid_coords['num_bikes_available'],
                                  valid_coords['num_docks_available'],
                                  valid_coords['availability_status']
                              )],
                        hovertemplate='%{text}<extra></extra>',
                        name=self.city_names.get(system, system)
                    ))
            
            # Determine map center
            if len(data) == 1:
                system = list(data.keys())[0]
                if system == 'baywheels':
                    center_lat, center_lon = 37.7749, -122.4194  # San Francisco
                    zoom = 10
                else:
                    center_lat, center_lon = 40.7128, -74.0060  # New York
                    zoom = 10
            else:
                center_lat, center_lon = 39.2904, -98.8570  # Center of US
                zoom = 3
            
            fig.update_layout(
                title="Bikeshare Station Network",
                mapbox=dict(
                    style="carto-positron",  # Shortbread-style clean basemap
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                ),
                showlegend=True,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating station map: {str(e)}")
            return None
    
    def create_utilization_heatmap(self, data: Dict) -> Optional[go.Figure]:
        """Create station utilization heatmap"""
        try:
            fig = go.Figure()
            
            for system, system_data in data.items():
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    
                    # Filter out stations with invalid coordinates
                    valid_coords = df[
                        (df['lat'] != 0) & (df['lon'] != 0) & 
                        (df['lat'].notna()) & (df['lon'].notna()) &
                        (df['utilization_rate'].notna())
                    ]
                    
                    if valid_coords.empty:
                        continue
                    
                    # Create size based on capacity and color based on utilization
                    marker_sizes = np.sqrt(valid_coords['capacity']) * 2
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=valid_coords['lat'],
                        lon=valid_coords['lon'],
                        mode='markers',
                        marker=dict(
                            size=marker_sizes,
                            color=valid_coords['utilization_rate'],
                            colorscale='Viridis',
                            cmin=0,
                            cmax=1,
                            opacity=0.7,
                            colorbar=dict(title="Utilization Rate"),
                            sizemin=4
                        ),
                        text=[f"<b>{name}</b><br>Capacity: {capacity}<br>Utilization: {util:.1%}<br>Bikes: {bikes}" 
                              for name, capacity, util, bikes in zip(
                                  valid_coords['name'], 
                                  valid_coords['capacity'],
                                  valid_coords['utilization_rate'],
                                  valid_coords['num_bikes_available']
                              )],
                        hovertemplate='%{text}<extra></extra>',
                        name=self.city_names.get(system, system)
                    ))
            
            # Determine map center
            if len(data) == 1:
                system = list(data.keys())[0]
                if system == 'baywheels':
                    center_lat, center_lon = 37.7749, -122.4194  # San Francisco
                    zoom = 10
                else:
                    center_lat, center_lon = 40.7128, -74.0060  # New York
                    zoom = 10
            else:
                center_lat, center_lon = 39.2904, -98.8570  # Center of US
                zoom = 3
            
            fig.update_layout(
                title="Station Utilization Heatmap",
                mapbox=dict(
                    style="carto-positron",  # Shortbread-style clean basemap
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                ),
                showlegend=True,
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating utilization heatmap: {str(e)}")
            return None
    
    def create_system_comparison_chart(self, comparison_data: Dict) -> Optional[go.Figure]:
        """Create system comparison chart"""
        try:
            # Extract metrics for comparison
            metrics_to_compare = ['total_stations', 'total_bikes', 'total_docks', 'total_ebikes']
            
            bay_values = []
            citibike_values = []
            metric_labels = []
            
            for metric in metrics_to_compare:
                if metric in comparison_data:
                    bay_values.append(comparison_data[metric].get('baywheels', 0))
                    citibike_values.append(comparison_data[metric].get('citibike', 0))
                    # Clean up metric names for display
                    clean_name = metric.replace('total_', '').replace('_', ' ').title()
                    metric_labels.append(clean_name)
            
            fig = go.Figure()
            
            # Add bars for each system
            fig.add_trace(go.Bar(
                x=metric_labels,
                y=bay_values,
                name='Bay Wheels (SF)',
                marker_color=self.color_scheme['baywheels']
            ))
            
            fig.add_trace(go.Bar(
                x=metric_labels,
                y=citibike_values,
                name='Citi Bike (NYC)',
                marker_color=self.color_scheme['citibike']
            ))
            
            fig.update_layout(
                title="System Comparison - Key Metrics",
                xaxis_title="Metrics",
                yaxis_title="Count",
                barmode='group',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating system comparison chart: {str(e)}")
            return None
    
    def create_capacity_distribution(self, data: Dict) -> Optional[go.Figure]:
        """Create station capacity distribution histogram"""
        try:
            fig = go.Figure()
            
            for system, system_data in data.items():
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    
                    if 'capacity' in df.columns:
                        fig.add_trace(go.Histogram(
                            x=df['capacity'],
                            name=self.city_names.get(system, system),
                            marker_color=self.color_scheme.get(system, '#999999'),
                            opacity=0.7,
                            nbinsx=20
                        ))
            
            fig.update_layout(
                title="Station Capacity Distribution",
                xaxis_title="Station Capacity",
                yaxis_title="Number of Stations",
                barmode='overlay',
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating capacity distribution chart: {str(e)}")
            return None
    
    def create_ebike_analysis(self, data: Dict) -> Optional[go.Figure]:
        """Create e-bike availability analysis chart"""
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["E-Bike vs Regular Bike Distribution", "E-Bike Availability by System"],
                specs=[[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Pie chart for bike type distribution
            total_regular = 0
            total_ebikes = 0
            
            for system, system_data in data.items():
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    regular_bikes = (df['num_bikes_available'] - df['num_ebikes_available']).sum()
                    ebikes = df['num_ebikes_available'].sum()
                    
                    total_regular += max(0, regular_bikes)  # Ensure non-negative
                    total_ebikes += ebikes
            
            if total_regular + total_ebikes > 0:
                fig.add_trace(
                    go.Pie(
                        labels=['Regular Bikes', 'E-Bikes'],
                        values=[total_regular, total_ebikes],
                        name="Bike Types"
                    ),
                    row=1, col=1
                )
            
            # Bar chart for e-bike availability by system
            systems = []
            ebike_counts = []
            
            for system, system_data in data.items():
                if system_data and 'stations' in system_data:
                    df = system_data['stations']
                    systems.append(self.city_names.get(system, system))
                    ebike_counts.append(df['num_ebikes_available'].sum())
            
            if systems:
                colors = [self.color_scheme.get(sys.lower().replace(' ', '').replace('(', '').replace(')', '').split('-')[0] if '-' in sys else sys.lower().replace(' ', ''), '#999999') for sys in systems]
                
                fig.add_trace(
                    go.Bar(
                        x=systems,
                        y=ebike_counts,
                        name="E-Bikes Available",
                        marker_color=colors
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="E-Bike Analysis",
                height=400,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating e-bike analysis chart: {str(e)}")
            return None
    
    def create_service_area_basemap(self, system: str, system_data: Dict) -> Optional[go.Figure]:
        """Create service area basemap with station locations"""
        try:
            if system not in self.service_areas:
                st.warning(f"Service area not defined for system: {system}")
                return None
                
            service_area = self.service_areas[system]
            city_name = self.city_names.get(system, system)
            
            # Create base map
            fig = go.Figure()
            
            # Service area boundary removed per user request
            
            # Add stations if available
            if system_data and 'stations' in system_data:
                df = system_data['stations']
                
                # Filter out stations without coordinates
                valid_stations = df.dropna(subset=['lat', 'lon'])
                
                if not valid_stations.empty:
                    # Create hover text
                    hover_text = [
                        f"<b>{name}</b><br>" +
                        f"Available Bikes: {bikes}<br>" +
                        f"Available Docks: {docks}<br>" +
                        f"E-Bikes: {ebikes}<br>" +
                        f"Status: {status}"
                        for name, bikes, docks, ebikes, status in zip(
                            valid_stations['name'],
                            valid_stations['num_bikes_available'],
                            valid_stations['num_docks_available'],
                            valid_stations['num_ebikes_available'],
                            valid_stations['availability_status']
                        )
                    ]
                    
                    # Color stations by availability status
                    colors = [self.availability_colors.get(status, '#999999') 
                             for status in valid_stations['availability_status']]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=valid_stations['lat'],
                        lon=valid_stations['lon'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors,
                            opacity=0.8
                        ),
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        name='Stations',
                        showlegend=False
                    ))
            
            # Update layout with mapbox
            fig.update_layout(
                title=f"{city_name} Bikeshare Service Area",
                mapbox=dict(
                    accesstoken=None,  # Using open source tiles
                    style="carto-positron",  # Shortbread-style clean basemap
                    center=service_area['center'],
                    zoom=service_area['zoom']
                ),
                height=600,
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating service area basemap: {str(e)}")
            return None
    
    def create_station_heatmap(self, system: str, station_popularity: dict, city_filter: str = None) -> Optional[go.Figure]:
        """Create station popularity heat map with vibrant magenta gradient styling"""
        try:
            if system not in self.service_areas:
                st.warning(f"Service area not defined for system: {system}")
                return None
                
            service_area = self.service_areas[system]
            
            # Create base map
            fig = go.Figure()
            
            # Filter stations with coordinates and prepare data for heat map
            valid_stations = []
            for station_id, data in station_popularity.items():
                if data['coords'] and data['trips'] > 0:
                    lat, lon = data['coords']
                    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                        valid_stations.append({
                            'station_id': station_id,
                            'name': data['name'],
                            'trips': data['trips'],
                            'lat': lat,
                            'lon': lon
                        })
            
            if valid_stations:
                # Sort by trips for better visualization
                valid_stations.sort(key=lambda x: x['trips'], reverse=True)
                
                # Calculate dynamic bounds from data points
                lats = [station['lat'] for station in valid_stations]
                lons = [station['lon'] for station in valid_stations]
                
                # Calculate bounding box with padding
                min_lat, max_lat = min(lats), max(lats)
                min_lon, max_lon = min(lons), max(lons)
                
                lat_padding = (max_lat - min_lat) * 0.1
                lon_padding = (max_lon - min_lon) * 0.1
                
                dynamic_bounds = {
                    'north': max_lat + lat_padding,
                    'south': min_lat - lat_padding,
                    'east': max_lon + lon_padding,
                    'west': min_lon - lon_padding
                }
                
                dynamic_center = {
                    'lat': (min_lat + max_lat) / 2,
                    'lon': (min_lon + max_lon) / 2
                }
                
                # Create hover text for all stations
                hover_text = [
                    f"<b>{station['name']}</b><br>" +
                    f"Total Trips: {station['trips']:,}<br>" +
                    f"Station ID: {station['station_id']}"
                    for station in valid_stations
                ]
                
                # Normalize trip counts for color and size (0-1 scale)
                max_trips = max(station['trips'] for station in valid_stations)
                min_trips = min(station['trips'] for station in valid_stations)
                trip_range = max_trips - min_trips if max_trips > min_trips else 1
                
                # Create vibrant sizing and colors based on popularity
                colors = []
                sizes = []
                for station in valid_stations:
                    intensity = (station['trips'] - min_trips) / trip_range
                    colors.append(station['trips'])  # Use actual trip count for scale
                    # Dramatic sizing: 12-60 pixels with power scaling for more visual impact
                    # More popular stations are significantly bigger
                    size = 12 + (intensity ** 0.5) * 48
                    sizes.append(size)
                
                # Custom magenta gradient colorscale
                # Dark magenta (#4d0038) for low trips -> Bright vibrant magenta (#ff00bf) for high trips
                custom_colorscale = [
                    [0.0, '#4d0038'],   # Dark purple/magenta
                    [0.3, '#800060'],   # Medium dark magenta
                    [0.6, '#cc0096'],   # Medium magenta
                    [0.8, '#ff00bf'],   # Bright magenta
                    [1.0, '#ff66d9']    # Very bright, vibrant magenta
                ]
                
                # Add all stations as scatter points with vibrant magenta gradient
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors,
                        colorscale=custom_colorscale,
                        showscale=True,
                        colorbar=dict(
                            title=dict(
                                text="Trips",
                                font=dict(size=14)
                            ),
                            tickformat=',d',
                            thickness=20,
                            len=0.6,
                            x=1.02,
                            y=0.5
                        ),
                        opacity=0.5,  # 50% opacity
                        sizemode='diameter'
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name="Stations"
                ))
            
            # Calculate appropriate zoom level based on bounds
            lat_range = dynamic_bounds['north'] - dynamic_bounds['south']
            lon_range = dynamic_bounds['east'] - dynamic_bounds['west']
            
            max_range = max(lat_range, lon_range)
            if max_range > 1.0:
                zoom_level = 8
            elif max_range > 0.5:
                zoom_level = 9
            elif max_range > 0.25:
                zoom_level = 10
            elif max_range > 0.1:
                zoom_level = 11
            else:
                zoom_level = 12

            # Update layout with mapbox 
            fig.update_layout(
                title="",
                mapbox=dict(
                    accesstoken=None,
                    style="carto-positron",
                    center=dynamic_center,
                    zoom=zoom_level,
                    bearing=0,
                    pitch=50  # 3D angled view
                ),
                height=600,
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                showlegend=False,
                autosize=True,
                modebar=dict(
                    orientation="h",
                    bgcolor="rgba(255,255,255,0.8)",
                    color="black", 
                    activecolor="blue"
                )
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating station heat map: {str(e)}")
            return None
    
    def create_station_column_map(self, system: str, station_popularity: dict, city_filter: str = None):
        """Create 3D station popularity map with vertical columns using Pydeck ColumnLayer"""
        try:
            if system not in self.service_areas:
                st.warning(f"Service area not defined for system: {system}")
                return None
                
            service_area = self.service_areas[system]
            
            # Filter stations with coordinates and prepare data for column map
            valid_stations = []
            for station_id, data in station_popularity.items():
                if data['coords'] and data['trips'] > 0:
                    lat, lon = data['coords']
                    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                        valid_stations.append({
                            'station_id': station_id,
                            'name': data['name'],
                            'trips': data['trips'],
                            'lat': lat,
                            'lon': lon
                        })
            
            if not valid_stations:
                st.warning("No valid station data to display")
                return None
            
            # Convert to DataFrame for Pydeck
            df = pd.DataFrame(valid_stations)
            
            # Calculate dynamic bounds from data points
            min_lat, max_lat = df['lat'].min(), df['lat'].max()
            min_lon, max_lon = df['lon'].min(), df['lon'].max()
            
            dynamic_center = {
                'lat': (min_lat + max_lat) / 2,
                'lon': (min_lon + max_lon) / 2
            }
            
            # Calculate appropriate zoom level
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            max_range = max(lat_range, lon_range)
            
            if max_range > 1.0:
                zoom_level = 8
            elif max_range > 0.5:
                zoom_level = 9
            elif max_range > 0.25:
                zoom_level = 10
            elif max_range > 0.1:
                zoom_level = 11
            else:
                zoom_level = 12
            
            # Normalize trips for color intensity
            max_trips = df['trips'].max()
            min_trips = df['trips'].min()
            trip_range = max_trips - min_trips if max_trips > min_trips else 1
            
            # Create color column with magenta gradient based on intensity
            def get_magenta_color(trips):
                intensity = (trips - min_trips) / trip_range
                
                # Dark magenta (#4d0038) for low trips -> Bright magenta (#ff00bf) for high trips
                if intensity < 0.3:
                    r = int(77 + (128 - 77) * (intensity / 0.3))
                    g = 0
                    b = int(56 + (96 - 56) * (intensity / 0.3))
                elif intensity < 0.6:
                    r = int(128 + (204 - 128) * ((intensity - 0.3) / 0.3))
                    g = 0
                    b = int(96 + (150 - 96) * ((intensity - 0.3) / 0.3))
                elif intensity < 0.8:
                    r = int(204 + (255 - 204) * ((intensity - 0.6) / 0.2))
                    g = 0
                    b = int(150 + (191 - 150) * ((intensity - 0.6) / 0.2))
                else:
                    r = int(255)
                    g = int(0 + (102) * ((intensity - 0.8) / 0.2))
                    b = int(191 + (217 - 191) * ((intensity - 0.8) / 0.2))
                
                return [r, g, b, 255]  # 100% opacity (fully opaque)
            
            df['color'] = df['trips'].apply(get_magenta_color)
            
            # Create ScatterplotLayer with dynamic zoom-based radius scaling
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_radius="trips",
                radius_scale=0.01,
                radius_min_pixels=3,
                radius_max_pixels=50,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True
            )
            
            # Set ViewState for 2D top-down view
            # Use San Francisco center for BayWheels, otherwise use dynamic center
            if system == 'baywheels':
                center_lat = 37.7749
                center_lon = -122.4194
                zoom = 11  # Good zoom for SF geo-fenced area
            else:
                center_lat = dynamic_center['lat']
                center_lon = dynamic_center['lon']
                zoom = zoom_level
            
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
                bearing=0
            )
            
            # Tooltip - simplified without number formatting
            tooltip = {
                "html": "<b>{name}</b><br/>Total Trips: {trips}",
                "style": {"color": "white"}
            }
            
            # Create Deck - use light style (built-in, no token required)
            deck = pdk.Deck(
                layers=[scatter_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light"
            )
            
            return deck
            
        except Exception as e:
            st.error(f"Error creating station column map: {str(e)}")
            return None
    
    def create_animated_station_heatmap(self, system: str, trip_data: dict, city_filter: str = None) -> Optional[go.Figure]:
        """Create animated station heat map showing hourly activity throughout the day"""
        try:
            if system not in self.service_areas:
                st.warning(f"Service area not defined for system: {system}")
                return None
                
            service_area = self.service_areas[system]
            city_name = city_filter if city_filter and city_filter != "All Cities" else self.city_names.get(system, system)
            
            # Calculate hourly station activity
            hourly_station_data = {}
            all_stations = {}
            
            # Process trip data to extract hourly patterns
            for key, df in trip_data.items():
                if df.empty:
                    continue
                    
                # Ensure start_time is datetime
                if 'start_time' in df.columns:
                    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
                        df['start_time'] = pd.to_datetime(df['start_time'])
                    
                    # Extract hour from start_time
                    df['hour'] = df['start_time'].dt.hour
                    
                    # Get start station coordinates
                    start_lat_cols = [col for col in df.columns if 'start' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
                    start_lon_cols = [col for col in df.columns if 'start' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
                    end_lat_cols = [col for col in df.columns if 'end' in col.lower() and ('lat' in col.lower() or 'latitude' in col.lower())]
                    end_lon_cols = [col for col in df.columns if 'end' in col.lower() and ('lng' in col.lower() or 'lon' in col.lower() or 'longitude' in col.lower())]
                    
                    if start_lat_cols and start_lon_cols and 'start_station_id' in df.columns and 'start_station_name' in df.columns:
                        start_lat_col = start_lat_cols[0]
                        start_lon_col = start_lon_cols[0]
                        
                        # Process start stations by hour
                        for hour in range(24):
                            if hour not in hourly_station_data:
                                hourly_station_data[hour] = {}
                            
                            hour_df = df[df['hour'] == hour]
                            if not hour_df.empty:
                                # Count trips starting at each station for this hour
                                station_counts = hour_df.groupby(['start_station_id', 'start_station_name', start_lat_col, start_lon_col]).size().reset_index(name='trips')
                                
                                for _, row in station_counts.iterrows():
                                    station_id = row['start_station_id']
                                    station_name = row['start_station_name']
                                    lat = row[start_lat_col]
                                    lon = row[start_lon_col]
                                    trips = row['trips']
                                    
                                    if pd.notna(lat) and pd.notna(lon) and lat != 0 and lon != 0:
                                        if station_id not in hourly_station_data[hour]:
                                            hourly_station_data[hour][station_id] = {
                                                'name': station_name,
                                                'lat': lat,
                                                'lon': lon,
                                                'trips': 0
                                            }
                                        hourly_station_data[hour][station_id]['trips'] += trips
                                        
                                        # Keep track of all stations for consistent layout
                                        if station_id not in all_stations:
                                            all_stations[station_id] = {
                                                'name': station_name,
                                                'lat': lat,
                                                'lon': lon
                                            }
                    
                    # Process end stations by hour
                    if end_lat_cols and end_lon_cols and 'end_station_id' in df.columns and 'end_station_name' in df.columns:
                        end_lat_col = end_lat_cols[0]
                        end_lon_col = end_lon_cols[0]
                        
                        for hour in range(24):
                            if hour not in hourly_station_data:
                                hourly_station_data[hour] = {}
                            
                            hour_df = df[df['hour'] == hour]
                            if not hour_df.empty:
                                # Count trips ending at each station for this hour
                                station_counts = hour_df.groupby(['end_station_id', 'end_station_name', end_lat_col, end_lon_col]).size().reset_index(name='trips')
                                
                                for _, row in station_counts.iterrows():
                                    station_id = row['end_station_id']
                                    station_name = row['end_station_name']
                                    lat = row[end_lat_col]
                                    lon = row[end_lon_col]
                                    trips = row['trips']
                                    
                                    if pd.notna(lat) and pd.notna(lon) and lat != 0 and lon != 0:
                                        if station_id not in hourly_station_data[hour]:
                                            hourly_station_data[hour][station_id] = {
                                                'name': station_name,
                                                'lat': lat,
                                                'lon': lon,
                                                'trips': 0
                                            }
                                        hourly_station_data[hour][station_id]['trips'] += trips
                                        
                                        # Keep track of all stations for consistent layout
                                        if station_id not in all_stations:
                                            all_stations[station_id] = {
                                                'name': station_name,
                                                'lat': lat,
                                                'lon': lon
                                            }
            
            if not hourly_station_data or not all_stations:
                st.warning("No station data available for animation")
                return None
            
            # Calculate global bounds from all stations
            all_lats = [station['lat'] for station in all_stations.values()]
            all_lons = [station['lon'] for station in all_stations.values()]
            
            min_lat, max_lat = min(all_lats), max(all_lats)
            min_lon, max_lon = min(all_lons), max(all_lons)
            
            # Add padding
            lat_padding = (max_lat - min_lat) * 0.1
            lon_padding = (max_lon - min_lon) * 0.1
            
            dynamic_center = {
                'lat': (min_lat + max_lat) / 2,
                'lon': (min_lon + max_lon) / 2
            }
            
            # Calculate zoom level
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            max_range = max(lat_range, lon_range)
            
            if max_range > 1.0:
                zoom_level = 8
            elif max_range > 0.5:
                zoom_level = 9
            elif max_range > 0.25:
                zoom_level = 10
            elif max_range > 0.1:
                zoom_level = 11
            else:
                zoom_level = 12
            
            # Create animated figure
            fig = go.Figure()
            
            # Find maximum trips across all hours for consistent scaling
            max_trips = 0
            for hour_data in hourly_station_data.values():
                for station_data in hour_data.values():
                    max_trips = max(max_trips, station_data['trips'])
            
            if max_trips == 0:
                st.warning("No trip data found for animation")
                return None
            
            # Create frames for each hour
            frames = []
            for hour in range(24):
                frame_data = []
                
                if hour in hourly_station_data:
                    stations_this_hour = list(hourly_station_data[hour].values())
                    
                    if stations_this_hour:
                        # Sort by trips and take top 15 for performance
                        stations_this_hour.sort(key=lambda x: x['trips'], reverse=True)
                        top_stations = stations_this_hour[:15]
                        
                        lats = [s['lat'] for s in top_stations]
                        lons = [s['lon'] for s in top_stations]
                        trips = [s['trips'] for s in top_stations]
                        names = [s['name'] for s in top_stations]
                        
                        # Calculate sizes (15-50 pixels)
                        sizes = []
                        for trip_count in trips:
                            intensity = trip_count / max_trips if max_trips > 0 else 0
                            size = 15 + (intensity ** 0.7) * 35
                            sizes.append(size)
                        
                        # Create hover text
                        hover_text = [
                            f"<b>{name}</b><br>Trips: {trip_count:,}<br>Hour: {hour}:00"
                            for name, trip_count in zip(names, trips)
                        ]
                        
                        frame_data.append(go.Scattermapbox(
                            lat=lats,
                            lon=lons,
                            mode='markers',
                            marker=dict(
                                size=sizes,
                                color=trips,
                                colorscale='Plasma',
                                opacity=0.7,
                                sizemode='diameter'
                            ),
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>',
                            name=f'Hour {hour}',
                            showlegend=False
                        ))
                
                frames.append(go.Frame(
                    data=frame_data,
                    name=str(hour),
                    layout=dict(
                        title=f"Station Activity at {hour}:00"
                    )
                ))
            
            # Add initial data (hour 0)
            if 0 in hourly_station_data and hourly_station_data[0]:
                initial_stations = list(hourly_station_data[0].values())
                initial_stations.sort(key=lambda x: x['trips'], reverse=True)
                top_initial = initial_stations[:15]
                
                if top_initial:
                    initial_lats = [s['lat'] for s in top_initial]
                    initial_lons = [s['lon'] for s in top_initial]
                    initial_trips = [s['trips'] for s in top_initial]
                    initial_names = [s['name'] for s in top_initial]
                    
                    initial_sizes = []
                    for trip_count in initial_trips:
                        intensity = trip_count / max_trips if max_trips > 0 else 0
                        size = 15 + (intensity ** 0.7) * 35
                        initial_sizes.append(size)
                    
                    initial_hover = [
                        f"<b>{name}</b><br>Trips: {trip_count:,}<br>Hour: 0:00"
                        for name, trip_count in zip(initial_names, initial_trips)
                    ]
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=initial_lats,
                        lon=initial_lons,
                        mode='markers',
                        marker=dict(
                            size=initial_sizes,
                            color=initial_trips,
                            colorscale='Plasma',
                            opacity=0.7,
                            sizemode='diameter'
                        ),
                        text=initial_hover,
                        hovertemplate='%{text}<extra></extra>',
                        name='Stations',
                        showlegend=False
                    ))
            
            # Update layout with animation controls
            fig.update_layout(
                title="Hourly Station Activity Animation",
                mapbox=dict(
                    accesstoken=None,
                    style="carto-positron",
                    center=dynamic_center,
                    zoom=zoom_level
                ),
                height=600,
                margin={"r": 0, "t": 50, "l": 0, "b": 50},
                showlegend=False,
                updatemenus=[{
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 1000, "redraw": True},
                                           "fromcurrent": True, "transition": {"duration": 300}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate", "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }],
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Hour: ",
                        "visible": True,
                        "xanchor": "right"
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[str(hour)], {
                                "frame": {"duration": 300, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 300}
                            }],
                            "label": f"{hour}:00",
                            "method": "animate"
                        } for hour in range(24)
                    ]
                }]
            )
            
            # Add frames to figure
            fig.frames = frames
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating animated station heat map: {str(e)}")
            return None
    
    def create_route_heatmap(self, system: str, popular_routes: List[Dict], city_filter: str = None):
        """Create route popularity map with exactly 2 points per route (20 total for top 10)"""
        try:
            if system not in self.service_areas:
                st.warning(f"Service area not defined for system: {system}")
                return None
                
            if not popular_routes:
                st.warning("No route data available")
                return None
            
            # Create individual points for each route (2 per route: start and end)
            # Points have formatted label for tooltip: "Station - {name}"
            route_points = []
            for i, route in enumerate(popular_routes):
                rank = i + 1
                trip_count = route['trip_count']
                
                # Add start station point with formatted label
                route_points.append({
                    'tooltip': f"<b>{route['start_station_name']}</b>",
                    'lat': route['start_lat'],
                    'lon': route['start_lon'],
                    'trips': trip_count
                })
                
                # Add end station point with formatted label
                route_points.append({
                    'tooltip': f"<b>{route['end_station_name']}</b>",
                    'lat': route['end_lat'],
                    'lon': route['end_lon'],
                    'trips': trip_count
                })
            
            # Convert to DataFrame
            df = pd.DataFrame(route_points)
            
            if df.empty:
                st.warning("No valid route data to display")
                return None
            
            # Calculate dynamic bounds and center
            min_lat, max_lat = df['lat'].min(), df['lat'].max()
            min_lon, max_lon = df['lon'].min(), df['lon'].max()
            
            dynamic_center = {
                'lat': (min_lat + max_lat) / 2,
                'lon': (min_lon + max_lon) / 2
            }
            
            # Calculate appropriate zoom level
            lat_range = max_lat - min_lat
            lon_range = max_lon - min_lon
            max_range = max(lat_range, lon_range)
            
            if max_range > 1.0:
                zoom_level = 8
            elif max_range > 0.5:
                zoom_level = 9
            elif max_range > 0.25:
                zoom_level = 10
            elif max_range > 0.1:
                zoom_level = 11
            else:
                zoom_level = 12
            
            # Normalize trips for color intensity (across all routes)
            max_trips = df['trips'].max()
            min_trips = df['trips'].min()
            trip_range = max_trips - min_trips if max_trips > min_trips else 1
            
            # Create color column with magenta gradient based on route trip count
            def get_magenta_color(trips):
                intensity = (trips - min_trips) / trip_range
                
                # Dark magenta (#4d0038) for low trips -> Bright magenta (#ff00bf) for high trips
                if intensity < 0.3:
                    r = int(77 + (128 - 77) * (intensity / 0.3))
                    g = 0
                    b = int(56 + (96 - 56) * (intensity / 0.3))
                elif intensity < 0.6:
                    r = int(128 + (204 - 128) * ((intensity - 0.3) / 0.3))
                    g = 0
                    b = int(96 + (150 - 96) * ((intensity - 0.3) / 0.3))
                elif intensity < 0.8:
                    r = int(204 + (255 - 204) * ((intensity - 0.6) / 0.2))
                    g = 0
                    b = int(150 + (191 - 150) * ((intensity - 0.6) / 0.2))
                else:
                    r = int(255)
                    g = int(0 + (102) * ((intensity - 0.8) / 0.2))
                    b = int(191 + (217 - 191) * ((intensity - 0.8) / 0.2))
                
                return [r, g, b, 255]  # 100% opacity (fully opaque)
            
            df['color'] = df['trips'].apply(get_magenta_color)
            
            # Create line data for connecting start and end stations
            line_data = []
            for i, route in enumerate(popular_routes):
                rank = i + 1
                trip_count = route['trip_count']
                
                # Get color for this route
                intensity = (trip_count - min_trips) / trip_range
                color = get_magenta_color(trip_count)
                
                # Calculate line width based on trip count (scaled between 2 and 10)
                width_intensity = (trip_count - min_trips) / trip_range
                line_width = 2 + (width_intensity * 8)  # 2 to 10 pixels
                
                line_data.append({
                    'start_position': [route['start_lon'], route['start_lat']],
                    'end_position': [route['end_lon'], route['end_lat']],
                    'color': color,
                    'width': line_width,
                    'tooltip': f"<b>#{rank} - {route['start_station_name']} - {route['end_station_name']}</b><br/>Total Trips: {int(trip_count):,}"
                })
            
            # Create DataFrame for lines
            lines_df = pd.DataFrame(line_data)
            
            # Create LineLayer for route connections
            line_layer = pdk.Layer(
                "LineLayer",
                data=lines_df,
                get_source_position="start_position",
                get_target_position="end_position",
                get_color="color",
                get_width="width",
                width_min_pixels=2,
                width_max_pixels=10,
                pickable=True,
                auto_highlight=True
            )
            
            # Create ScatterplotLayer with dynamic zoom-based radius scaling
            # Use exact same parameters as station popularity map
            scatter_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=["lon", "lat"],
                get_radius="trips",
                radius_scale=0.01,
                radius_min_pixels=3,
                radius_max_pixels=50,
                get_fill_color="color",
                pickable=True,
                auto_highlight=True
            )
            
            # Set ViewState for 2D top-down view
            if system == 'baywheels':
                center_lat = 37.7749
                center_lon = -122.4194
                zoom = 11
            else:
                center_lat = dynamic_center['lat']
                center_lon = dynamic_center['lon']
                zoom = zoom_level
            
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
                bearing=0
            )
            
            # Tooltip uses pre-formatted labels from both layers
            tooltip = {
                "html": "{tooltip}",
                "style": {"color": "white"}
            }
            
            # Create Deck with line layer (underneath) and scatter layer (on top)
            deck = pdk.Deck(
                layers=[line_layer, scatter_layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light"
            )
            
            return deck
            
        except Exception as e:
            st.error(f"Error creating route heat map: {str(e)}")
            return None