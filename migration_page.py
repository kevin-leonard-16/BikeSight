"""
Migration Page for Bikeshare Analytics Dashboard
Provides interface for database migration and management
"""

import streamlit as st
from data_migration import DataMigrator
from database_manager import get_database_manager
import pandas as pd

def render_migration_page():
    """Render the database migration page"""
    st.header("🚀 Database Migration & Management")
    
    # Initialize components
    try:
        migrator = DataMigrator()
        db = get_database_manager()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.info("Please ensure the PostgreSQL database is running and accessible.")
        return
    
    # Database Status
    st.subheader("📊 Database Status")
    
    try:
        stats = db.get_data_quality_stats()
        
        if stats:
            col1, col2 = st.columns(2)
            
            for system_name, system_stats in stats.items():
                with col1 if system_name == 'baywheels' else col2:
                    st.markdown(f"### {system_name.title()}")
                    
                    if 'total_trips' in system_stats:
                        st.metric("Total Trips", f"{system_stats['total_trips']:,}")
                        
                        if 'earliest_trip' in system_stats and 'latest_trip' in system_stats:
                            st.write(f"**Data Range:** {system_stats['earliest_trip'].strftime('%Y-%m-%d')} to {system_stats['latest_trip'].strftime('%Y-%m-%d')}")
                        
                        if 'months_loaded' in system_stats:
                            st.write(f"**Months Loaded:** {system_stats['months_loaded']}")
                    else:
                        st.info("No data loaded yet")
        else:
            st.info("Database is empty. Use the migration tools below to load data.")
    except Exception as e:
        st.warning(f"Could not fetch database stats: {e}")
    
    st.divider()
    
    # Migration Interface
    st.subheader("📥 Data Migration")
    
    # System selection
    system_name = st.selectbox(
        "Select Bikeshare System",
        options=['baywheels', 'citibike'],
        format_func=lambda x: f"{x.title()} ({migrator.systems_config[x]['city']})"
    )
    
    # Check available files
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Check Available Files"):
            with st.spinner("Checking S3 for available files..."):
                available_files = migrator.get_available_files(system_name)
            
            if available_files:
                st.success(f"Found {len(available_files)} available files")
                
                # Show recent files
                recent_files = available_files[-10:]  # Last 10 files
                st.markdown("**Recent Files:**")
                for month_date, file_url in recent_files:
                    # Check if already loaded
                    is_loaded = db.check_data_loaded(system_name, month_date)
                    status = "✅ Loaded" if is_loaded else "⏳ Available"
                    st.write(f"- {month_date.strftime('%Y-%m')}: {status}")
            else:
                st.warning("No available files found")
    
    with col2:
        if st.button("📋 Check Load History"):
            try:
                with db.get_connection() as conn:
                    query = """
                        SELECT data_month, load_status, trips_loaded, trips_rejected, completed_at
                        FROM data_load_log 
                        WHERE system_name = %s 
                        ORDER BY data_month DESC 
                        LIMIT 10
                    """
                    history_df = pd.read_sql_query(query, conn, params=[system_name])
                
                if not history_df.empty:
                    st.markdown("**Recent Load History:**")
                    for _, row in history_df.iterrows():
                        status_icon = "✅" if row['load_status'] == 'success' else "❌" if row['load_status'] == 'failed' else "⏳"
                        trips_text = f"({row['trips_loaded']:,} trips)" if pd.notna(row['trips_loaded']) else ""
                        st.write(f"- {status_icon} {row['data_month'].strftime('%Y-%m')} {trips_text}")
                else:
                    st.info("No load history found")
            except Exception as e:
                st.error(f"Error fetching load history: {e}")
    
    st.divider()
    
    # Migration Options
    st.subheader("⚙️ Migration Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_months = st.number_input(
            "Maximum months to process",
            min_value=1, max_value=12, value=3,
            help="Limit migration for testing (recommended: 3 for first run)"
        )
    
    with col2:
        load_mode = st.radio(
            "Load Mode",
            ["Recent Only", "Missing Only", "Force Reload"],
            help="Recent: load latest months, Missing: only unloaded months, Force: reload all"
        )
    
    # Run Migration
    if st.button(f"🚀 Migrate {system_name.title()} Data", type="primary"):
        # Warning for large loads
        if max_months > 6:
            st.warning("⚠️ Loading more than 6 months may take several minutes. Consider starting with 3 months for testing.")
        
        # Create confirmation modal
        st.markdown("---")
        st.markdown("### ⚠️ Confirm Migration")
        
        # Modal-style warning box
        st.error(f"""
        **You are about to:**
        - Download {max_months} months of {system_name.title()} data from S3
        - Process and clean historical CSV files  
        - Load data into PostgreSQL database
        
        **This may take several minutes depending on data volume.**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("❌ Cancel", key="cancel_migration"):
                st.info("Migration cancelled.")
                st.rerun()
        
        with col2:
            if st.button("✅ Confirm & Start", type="primary", key="confirm_migration"):
                # Create containers for progress tracking
                st.markdown("---")
                st.markdown("### 🚀 Migration in Progress")
                
                progress_bar = st.progress(0)
                status_placeholder = st.empty()
                log_placeholder = st.empty()
                
                # Initialize tracking variables
                total_files = 0
                current_file = 0
                
                def progress_callback(message):
                    nonlocal current_file, total_files
                    
                    # Update progress based on message type
                    if "Processing" in message and "/" in message:
                        # Extract current/total from "Processing 2024-01 (1/3)" format
                        try:
                            parts = message.split("(")[1].split(")")[0].split("/")
                            current_file = int(parts[0])
                            total_files = int(parts[1])
                            progress = current_file / total_files
                            progress_bar.progress(progress, text=f"File {current_file} of {total_files}")
                        except:
                            pass
                    
                    # Update status display
                    status_placeholder.info(f"🔄 **{message}**")
                    
                    # Force Streamlit to update (this helps with real-time display)
                    import time
                    time.sleep(0.1)
                
                try:
                    # Start migration
                    status_placeholder.info("🚀 **Starting migration process...**")
                    
                    results = migrator.migrate_system_data(
                        system_name, 
                        max_months,
                        progress_callback
                    )
                    
                    # Complete progress
                    progress_bar.progress(1.0, text="Migration completed!")
                    
                    # Clear progress status and show final results
                    status_placeholder.empty()
                    log_placeholder.empty()
                    
                    # Success/failure header
                    if results['success']:
                        st.success("🎉 **Migration Completed Successfully!**")
                        st.balloons()
                        
                        # Clear explanation of what happened
                        st.info(f"""
                        **✅ Database has been updated with {system_name.title()} data:**
                        - Historical trip data is now loaded and indexed
                        - Queries will be 10-100x faster than CSV processing
                        - Data is persistent and ready for analysis
                        - You can now use the Historical Data page for instant analytics
                        """)
                    else:
                        st.error("❌ **Migration Completed with Errors**")
                        st.warning("Some files may have failed to load. Check details below.")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Files Processed", results['total_files'])
                    with col2:
                        st.metric("Successful Loads", results['successful'])
                    with col3:
                        st.metric("Failed Loads", results['failed'])
                    
                    # What to do next
                    if results['successful'] > 0:
                        st.markdown("### 🎯 Next Steps")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info("**📊 View Your Data**\nGo to Historical Data page to see instant analytics")
                        with col2:
                            st.info("**⚡ Test Performance**\nUse 'Test Query Performance' below to verify speed")
                    
                    # Detailed results
                    with st.expander("📋 Detailed Results", expanded=results['failed'] > 0):
                        for message in results['messages']:
                            if 'Error' in message or 'failed' in message.lower():
                                st.error(f"❌ {message}")
                            elif 'already loaded' in message:
                                st.info(f"ℹ️ {message}")
                            else:
                                st.success(f"✅ {message}")
                
                except Exception as e:
                    progress_bar.progress(1.0, text="Migration failed!")
                    status_placeholder.error(f"❌ **Migration Failed:** {str(e)}")
                    st.error("Migration encountered an error. Please try again or check the system logs.")
    
    st.divider()
    
    # Database Management
    st.subheader("🛠️ Database Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 View Sample Data"):
            try:
                sample_df = db.get_trips_data(system_name=system_name, limit=100)
                if not sample_df.empty:
                    st.markdown("**Sample Trip Data:**")
                    st.dataframe(
                        sample_df[['start_time', 'end_time', 'duration_minutes', 'member_type', 'start_station_name', 'end_station_name']].head(10),
                        use_container_width=True
                    )
                else:
                    st.info("No trip data found")
            except Exception as e:
                st.error(f"Error fetching sample data: {e}")
    
    with col2:
        if st.button("📊 Test Query Performance"):
            try:
                import time
                start_time = time.time()
                
                # Test query
                metrics_df = db.get_monthly_metrics(system_name=system_name)
                
                query_time = time.time() - start_time
                
                st.success(f"✅ Query completed in {query_time:.2f} seconds")
                
                if not metrics_df.empty:
                    st.write(f"Retrieved {len(metrics_df)} months of data")
                    total_trips = metrics_df['total_trips'].sum()
                    st.write(f"Total trips analyzed: {total_trips:,}")
                
            except Exception as e:
                st.error(f"Query test failed: {e}")
    
    with col3:
        if st.button("🧹 Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")
    
    # Tips
    st.subheader("💡 Tips")
    st.markdown("""
    **Getting Started:**
    1. **First Time:** Start with 3 months of recent data to test the system
    2. **Check Files:** Use "Check Available Files" to see what's available
    3. **Monitor Progress:** Watch the progress log during migration
    4. **Test Performance:** Use "Test Query Performance" to verify speed improvements
    
    **Performance Benefits:**
    - 🚀 **10-100x faster** queries compared to CSV processing
    - 📊 **Complex analytics** possible with SQL aggregations
    - 🔄 **Incremental updates** - only load new data as needed
    - 💾 **Persistent storage** - data survives app restarts
    """)

if __name__ == "__main__":
    render_migration_page()