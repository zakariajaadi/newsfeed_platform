import streamlit as st
import plotly.express as px
from typing import Dict, List

from src.logging_setup import configure_logging
from src.dashboard.dashboard_service import DashboardService
from src.orchestration.ingestion_service_factory import IngestionServiceFactory

# Configure logging
configure_logging()

# Page configuration
st.set_page_config(
    page_title="IT Newsfeed Dashboard",
    page_icon="📰",
    layout="wide"
)


class StreamlitDashboardUI:
    """Streamlit UI - purely presentation layer."""

    def __init__(self):
        self.apply_custom_styles()

    @staticmethod
    def apply_custom_styles():
        """Apply custom CSS styles."""
        st.markdown("""
        <style>
        
        .css-1d391kg {
            width: 150px !important;
        }
        
        .css-1lcbmhc {
            width: 150px !important;
        }
        
        section[data-testid="stSidebar"] {
            width: 150px !important;
        }
        
        .metric-container {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .info-box {
            background: #d1ecf1;
            border: 1px solid #bee5eb;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .ingestion-note {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def get_services():
        """Initialize and cache services - simple version."""
        shared_storage = IngestionServiceFactory.create_shared_storage()
        dashboard_service = DashboardService(vector_storage=shared_storage)
        return dashboard_service

    @staticmethod
    def get_storage_modification_time():
        """Get the last modification time of storage files."""
        import os
        try:
            shared_storage = IngestionServiceFactory.create_shared_storage()
            index_path = shared_storage.index_file_path
            metadata_path = shared_storage.metadata_file_path

            index_mtime = os.path.getmtime(index_path) if os.path.exists(index_path) else 0
            metadata_mtime = os.path.getmtime(metadata_path) if os.path.exists(metadata_path) else 0

            return max(index_mtime, metadata_mtime)
        except:
            return 0

    def render_sidebar(self) -> tuple:
        """Render sidebar controls."""

        st.sidebar.header("⚙️ Settings ")

        # Refresh button in sidebar
        st.sidebar.subheader("Data")
        if st.sidebar.button("🔄 Refresh Data", help="Reload data from storage"):
            st.cache_resource.clear()
            st.success("✅ Data refreshed!")
            st.rerun()

        # Display options
        st.sidebar.subheader("Display Options")
        max_events = st.sidebar.number_input("Max events per tab", min_value=5, max_value=100, value=20)
        show_timestamps = st.sidebar.checkbox("Show timestamps", value=True)

        return max_events, show_timestamps

    def render_ingestion_notice(self):
        """Show notice about using main.py for ingestion."""
        st.markdown("""
        <div class="ingestion-note">
            <h4>💡 Data Ingestion</h4>
            <p>To populate data, use the command line:</p>
            <ul>
                <li><strong>Manual:</strong> <code>python main.py manual</code></li>
                <li><strong>Background:</strong> <code>python main.py scheduler</code></li>
                <li><strong>Status:</strong> <code>python main.py status</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


    def render_dashboard_section(self, dashboard_service: DashboardService,
                                 max_events: int, show_timestamps: bool):
        """Render the events section."""
        st.header("📰 Events")

        # Get all events for display
        all_events = dashboard_service.get_all_events()

        if not all_events:
            st.info("No events in storage. Use main.py to run ingestion.")
            self.render_ingestion_notice()

        # Format events for display
        formatted_events = dashboard_service.format_events_for_display(all_events)

        # Filter controls
        #st.subheader("Filters")
        col1, col2 = st.columns(2)

        with col1:
            sources = dashboard_service.get_available_sources(all_events)
            source_filter = st.selectbox("Filter by source", ["All"] + sources)
            source_filter = source_filter if source_filter != "All" else None

        with col2:
            time_options = {"All": None, "Last hour": 1, "Last 6 hours": 6, "Last 24 hours": 24}
            time_filter = st.selectbox("Filter by time", list(time_options.keys()))
            time_filter_hours = time_options[time_filter]

        # Apply filters
        if source_filter or time_filter_hours:
            filtered_events = dashboard_service.apply_display_filters(
                all_events, source_filter, time_filter_hours)
            formatted_events = dashboard_service.format_events_for_display(filtered_events)

        # Display events
        self.render_events_display(formatted_events, "Events", max_events, show_timestamps)

    def render_statistics(self, stats: Dict):
        """Render statistics widgets."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📊 Total Events", stats['total_events'])

        with col2:
            st.metric("🕐 Recent (1hr)", stats['recent_events'])

        with col3:
            sources = stats.get('sources', {})
            unique_sources = len(sources)
            st.metric("📡 Sources", unique_sources)

        with col4:
            if stats['newest_event'] and stats['oldest_event']:
                age = (stats['newest_event'] - stats['oldest_event']).total_seconds() / 3600
                if age < 24:
                    st.metric("⏱️ Span (hrs)", f"{age:.1f}")
                else:
                    days = age / 24
                    st.metric("⏱️ Span (days)", f"{days:.0f}")
            else:
                st.metric("⏱️ Span (hrs)", "0")

        # Charts
        if stats['total_events'] > 0:
            col1, col2 = st.columns(2)

            with col1:
                # Source distribution
                sources = stats.get('sources', {})
                if sources:
                    fig = px.pie(
                        values=list(sources.values()),
                        names=list(sources.keys()),
                        title="Events by Source"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Time distribution
                recent = stats['recent_events']
                older = stats['total_events'] - recent
                if recent > 0 or older > 0:
                    fig = px.pie(
                        values=[recent, older],
                        names=['Recent (<1hr)', 'Older'],
                        title="Event Recency"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def render_events_display(self, events: List[Dict], title: str,
                              max_events: int, show_timestamps: bool):
        """Render events list."""
        st.subheader(f"{title} ({len(events)} total)")

        if not events:
            st.info("No events to display.")
            return

        # Pagination
        display_events = events[:max_events]
        if len(events) > max_events:
            st.info(f"Showing first {max_events} of {len(events)} events")

        # Events display
        for i, event in enumerate(display_events, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(f"### {i}. {event['title']}")
                    st.write(event['body'])

                    caption_parts = [f"Source: {event['source']}"]
                    if show_timestamps:
                        caption_parts.append(f"Published: {event['published_at']}")

                    st.caption(" | ".join(caption_parts))

                with col2:
                    age = event.get('age_minutes', 0)
                    if age < 60:
                        st.metric("Age", f"{age:.0f} minutes")
                    elif age < 1440:  # Less than 24 hours (24 * 60 = 1440 minutes)
                        st.metric("Age", f"{age / 60:.1f} hours")
                    else:  # 24 hours or more
                        st.metric("Age", f"{age / 1440:.0f} days")

                st.divider()

    def render_search_section(self, dashboard_service: DashboardService):
        """Render search functionality."""
        st.header("🔍 Search Events")

        # Search input
        query = st.text_input(
            "Search query:",
            placeholder="e.g., 'server error', 'security breach', 'authentication'"
        )

        if query:
            with st.spinner("Searching..."):
                results = dashboard_service.search_events(query, limit=10)

            if results:
                st.success(f"🎯 Found {len(results)} results")
                formatted_results = dashboard_service.format_events_for_display(results)
                self.render_events_display(formatted_results, "Search Results", 10, True)
            else:
                st.info("No matching events found.")

        # Storage management
        with st.expander("🔧 Storage Management"):
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Show Storage Stats"):
                    stats = dashboard_service.get_storage_stats()
                    st.json(stats)

            with col2:
                if st.button("🔄 Refresh Data", help="Clear cache and reload data from storage"):
                    st.cache_resource.clear()
                    st.success("Cache cleared! Data refreshed.")
                    st.rerun()

            with col3:
                if st.button("🗑️ Clear Storage", type="secondary"):
                    if st.button("⚠️ Confirm Clear", type="secondary"):
                        dashboard_service.clear_storage()
                        st.success("Storage cleared!")
                        st.rerun()


def main():
    """Main application."""
    ui = StreamlitDashboardUI()

    # Header
    st.header(" 📰 IT Newsfeed Dashboard")
    st.markdown("Monitor and manage IT-related news and events with semantic filtering.")

    # Get services (no modification time tracking needed)
    dashboard_service = ui.get_services()

    # Rest of your existing code stays exactly the same...
    max_events, show_timestamps = ui.render_sidebar()

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📰 Events", "🔍 Search"])

    with tab1:
        all_events = dashboard_service.get_all_events()
        if not all_events:
            st.info("No events in storage. Use main.py to run ingestion.")
            ui.render_ingestion_notice()
        else:
            stats = dashboard_service.get_display_statistics(all_events)
            ui.render_statistics(stats)

    with tab2:
        ui.render_dashboard_section(dashboard_service, max_events, show_timestamps)

    with tab3:
        ui.render_search_section(dashboard_service)




if __name__ == "__main__":
    main()