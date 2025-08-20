import streamlit as st
import plotly.express as px
from typing import Dict, List

from src.logging_setup import configure_logging
from src.dashboard.dashboard_service import DashboardService
from src.orchestration.ingestion_service_factory import IngestionServiceFactory
from src.ranking.ranking_engine import RankingEngine
from src.config import get_config

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

        .event-body {
            background-color: #f0f8ff;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
        }

        .clear-button {
            background-color: #dc3545;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 0.5rem;
        }

        /* Make all sidebar elements same width */
        .stButton > button {
            width: 100% !important;
        }

        .stNumberInput > div > div > input {
            width: 100% !important;
        }

        .stSlider > div > div {
            width: 100% !important;
        }

        .stSelectbox > div > div {
            width: 100% !important;
        }

        section[data-testid="stSidebar"] .stButton > button,
        section[data-testid="stSidebar"] .stNumberInput > div > div > input,
        section[data-testid="stSidebar"] .stSlider > div > div,
        section[data-testid="stSidebar"] .stSelectbox > div > div {
            width: 100% !important;
            box-sizing: border-box !important;
        }

        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    @st.cache_resource
    def get_services():
        """Initialize and cache services - simple version."""
        shared_storage = IngestionServiceFactory.create_shared_storage()
        ranking_engine = IngestionServiceFactory.create_ranking_engine()

        dashboard_service = DashboardService(vector_storage=shared_storage,
                                             ranking_engine=ranking_engine)
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

    def render_sidebar(self, dashboard_service: DashboardService) -> tuple:
        """Render sidebar controls."""

        st.sidebar.header("⚙️ Settings ")

        # Data management section
        st.sidebar.subheader("Data Management")

        # Refresh button (original layout)
        if st.sidebar.button("🔄 Refresh Data", help="Reload data from storage", use_container_width=True):
            st.cache_resource.clear()
            st.success("✅ Data refreshed!")
            st.rerun()

        # Clear button with two-step confirmation
        if 'confirm_clear' not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.sidebar.button("🗑️ Clear All Data", help="Clear all stored events", type="secondary",
                                 use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.sidebar.markdown("⚠️ **Are you sure?**")
            col_yes, col_no = st.sidebar.columns(2)

            with col_yes:
                if st.button("✅ Yes", type="primary", use_container_width=True, key="confirm_yes"):
                    if dashboard_service.clear_storage():
                        st.cache_resource.clear()
                        st.success("✅ All data cleared!")
                        st.session_state.confirm_clear = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear data")
                        st.session_state.confirm_clear = False

            with col_no:
                if st.button("❌ Cancel", use_container_width=True, key="confirm_no"):
                    st.session_state.confirm_clear = False
                    st.rerun()

        # Warning text only during confirmation
        if st.session_state.get('confirm_clear', False):
            st.sidebar.markdown("⚠️ **This action cannot be undone!**")

        # Display options
        st.sidebar.subheader("Display Options")
        max_events = st.sidebar.number_input("Max events per tab", min_value=5, max_value=100, value=20,
                                             key="max_events_input")
        show_timestamps = st.sidebar.checkbox("Show timestamps", value=True, key="show_timestamps_checkbox")

        # Importance vs Recency slider
        st.sidebar.subheader("Ranking balance")
        importance_weight = st.sidebar.slider(
            "Importance x Recency",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="importance_weight_slider"
        )
        recency_weight = 1.0 - importance_weight
        st.sidebar.markdown(f"<small>Importance weight= {importance_weight:.2f} </small>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<small>Recency weight= {recency_weight:.2f} </small>", unsafe_allow_html=True)

        return max_events, show_timestamps, importance_weight, recency_weight

    def render_ingestion_notice(self):
        """Show notice about using main.py for ingestion."""
        st.markdown("""
                <div class="ingestion-note">
                    <h4>📭 No Events In Storage ! </h4>
                    <p>To populate data:</p>
                    <ul>
                        <li>🐳 <strong>Make sure the "scheduler" Docker container is running</strong></li>
                        <li>⏱️ <strong>Wait 5 minutes</strong> (or the configured interval) for automatic data collection</li>
                    </ul>
                    <p><em>The scheduler automatically fetches IT news by default every 5 minutes from Reddit, and RSS feeds.</em></p>
                </div>
                """, unsafe_allow_html=True)

    def render_dashboard_section(self, dashboard_service: DashboardService,
                                 max_events: int,
                                 show_timestamps: bool,
                                 importance_weight: float = 0.7,
                                 recency_weight: float = 0.3):
        """Render the events section."""
        st.header("📰 Events")

        # Get all events for display
        ### all_events = dashboard_service.get_all_events()

        all_events = dashboard_service.get_all_events_reranked(
            importance_weight=importance_weight,
            recency_weight=recency_weight
        )

        if not all_events:
            #st.info("No events in storage.")
            self.render_ingestion_notice()

        # Format events for display
        formatted_events = dashboard_service.format_events_for_display(all_events)

        # Filter controls
        # st.subheader("Filters")
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
        """Render events"""

        st.subheader(f"{title} ({len(events)} total)")

        if not events:
            st.info("No events to display.")
            return

        display_events = events[:max_events]
        if len(events) > max_events:
            st.info(f"Showing first {max_events} of {len(events)} events")

        for i, event in enumerate(display_events, 1):
            with st.container():
                # Event title
                st.markdown(f"### {i}. {event['title']}")

                snippet_length = 400
                body_snippet = event['body'][:snippet_length]

                # If body is long, show snippet plus expander
                if len(event['body']) > snippet_length:
                    st.markdown(f"<div class='event-body'>{body_snippet}...</div>", unsafe_allow_html=True)
                    with st.expander("Read full event"):
                        st.markdown(f"<div class='event-body'>{event['body']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='event-body'>{event['body']}</div>", unsafe_allow_html=True)

                # Caption with age
                caption_parts = [f"📡 Source: {event['source']}"]
                if show_timestamps:
                    caption_parts.append(f"📅 Published: {event['published_at']}")

                age = event.get('age_minutes', 0)
                if age < 60:
                    age_str = f"{age:.0f}m ago"
                elif age < 1440:
                    age_str = f"{age / 60:.1f}h ago"
                else:
                    age_str = f"{age / 1440:.0f}d ago"
                caption_parts.append(f"⏱️ {age_str}")

                st.caption(" | ".join(caption_parts))
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


def main():
    """Main application."""
    ui = StreamlitDashboardUI()

    # Header
    st.header(" 📰 IT Newsfeed Dashboard")
    st.markdown("Monitor and manage IT-related news and events with semantic filtering.")

    # Get services (no modification time tracking needed)
    dashboard_service = ui.get_services()

    max_events, show_timestamps, importance_weight, recency_weight = ui.render_sidebar(dashboard_service)

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📰 Events", "🔍 Search"])

    with tab1:
        all_events = dashboard_service.get_all_events_ranked()
        if not all_events:
            #st.info("No events in storage.")
            ui.render_ingestion_notice()
        else:
            stats = dashboard_service.get_display_statistics(all_events)
            ui.render_statistics(stats)

    with tab2:
        ui.render_dashboard_section(dashboard_service, max_events, show_timestamps, importance_weight, recency_weight)

    with tab3:
        ui.render_search_section(dashboard_service)


if __name__ == "__main__":
    main()