# Lyft Bikeshare Analytics Dashboard

## Overview

This is a Streamlit-based analytics dashboard for analyzing Lyft bikeshare data across two major systems: BayWheels (San Francisco Bay Area) and Citi Bike (New York City). The application provides comprehensive insights into mobility patterns, ridership trends, and operational metrics through interactive visualizations and real-time data processing. The dashboard fetches data from public S3 buckets and GBFS (General Bikeshare Feed Specification) APIs to deliver up-to-date analytics on bike usage patterns, station popularity, trip durations, and user behavior across different time periods and geographic locations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the primary framework for building the interactive web dashboard. This choice provides rapid prototyping capabilities and built-in UI components optimized for data applications. The frontend follows a modular component-based architecture with cached resource initialization to optimize performance and reduce API calls.

### Data Processing Architecture
The system implements a **layered data processing approach** with four main components:

- **DataLoader**: Handles data acquisition from external APIs and S3 buckets with built-in caching mechanisms
- **MetricsCalculator**: Processes raw data to compute KPIs and statistical measures
- **Visualizer**: Creates interactive charts and visualizations using Plotly
- **Utils**: Provides shared formatting and utility functions

This separation of concerns allows for independent testing and maintenance of each component while maintaining clean interfaces between layers.

### Caching Strategy
The application uses **Streamlit's caching decorators** (@st.cache_data, @st.cache_resource) to minimize redundant API calls and improve user experience. Data is cached with TTL (Time To Live) settings to balance performance with data freshness requirements.

### Data Visualization Framework
**Plotly** was chosen as the visualization library to provide interactive charts with built-in zoom, hover, and filtering capabilities. The visualization component uses a consistent color scheme and styling across different chart types to maintain visual coherence.

## External Dependencies

### Data Sources
- **BayWheels S3 Bucket**: Historical trip data from San Francisco Bay Area bikeshare system (CSV format)
- **Citi Bike S3 Bucket**: Historical trip data from New York City bikeshare system (CSV format)
- **GBFS APIs**: Real-time station information and system status for both BayWheels and Citi Bike systems

### Third-Party Libraries
- **Streamlit**: Web application framework for the dashboard interface
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualization library (plotly.express and plotly.graph_objects)
- **NumPy**: Numerical computing for statistical calculations
- **Requests**: HTTP library for API calls and data fetching

### External APIs
- **BayWheels GBFS**: https://gbfs.baywheels.com/gbfs/2.3/gbfs.json
- **Citi Bike GBFS**: https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json

These APIs provide real-time station availability, system status, and operational information that complements the historical trip data analysis.