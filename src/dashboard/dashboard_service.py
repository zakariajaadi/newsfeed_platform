import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from src.logging_setup import configure_logging
from src.models import Event, EnrichedEvent
from src.storage.storage_service import VectorStorageService

# Set logging
logger = logging.getLogger(__name__)
configure_logging()

class DashboardService:
    """
        Dashboard service for handling UI-related operations.

        This service is responsible for:
        - Formatting data for display in the dashboard
        - Searching and filtering events
        - Providing statistics for dashboard widgets
        - Managing display-related storage operations

        The service acts as a bridge between the storage layer and the UI,
        focusing solely on presentation and display concerns.
        """

    def __init__(self, vector_storage: Optional[VectorStorageService] = None):
        """ Initialize the dashboard service with vector storage. """
        self.vector_storage = vector_storage


    def search_events(self, query: str, limit: int = 5) -> List[Event]:
        """ Search for events in storage based on query string."""
        try:
            # Check if there are any events in storage before searching
            if self.vector_storage.get_index_stats()['total_vectors'] > 0:
                return self.vector_storage.search(query, top_k=limit)
            else:
                # Log error and return empty list to prevent UI crashes
                logger.info("No events in storage for search")
                return []
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def get_all_events_ranked(self) -> List[EnrichedEvent]:
        """Get all stored events for dashboard display."""
        try:
            return self.vector_storage.get_all_events_ranked()
        except Exception as e:
            # Log error and return empty list to maintain UI stability
            logger.error(f"Error retrieving events: {e}")
            return []

    def get_all_events_reranked(self, importance_weight: float = 0.7, recency_weight: float = 0.3) -> List[EnrichedEvent]:
        """Get all stored events for dashboard display."""
        try:
            return self.vector_storage.get_all_events_reranked(importance_weight, recency_weight)
        except Exception as e:
            # Log error and return empty list to maintain UI stability
            logger.error(f"Error retrieving events: {e}")
            return []

    def get_storage_stats(self) -> Dict:
        """Get storage statistics for dashboard display."""
        try:
            return self.vector_storage.get_index_stats()
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {'total_vectors': 0, 'total_events': 0, 'next_id': 0, 'embedding_dim': 0}

    def _get_current_time(self) -> datetime:
        """Get current time with proper timezone handling."""
        return datetime.now(timezone.utc)

    def _calculate_age_minutes(self, published_at: Optional[datetime]) -> float:
        """Calculate the age of an event in minutes from its publication time.
        """
        # Return 0 if no publication time is provided
        if not published_at:
            return 0

        # Calculate age
        age_delta = self._get_current_time() - published_at
        return age_delta.total_seconds() / 60

    def format_events_for_display(self, events: List[Event]) -> List[Dict]:
        """ Format events for UI display by converting to dictionary format.
        """
        formatted = []
        for event in events:
            formatted.append({
                'id': event.id,
                'title': event.title,
                'body': event.body,
                'source': event.source,
                'published_at': event.published_at.strftime('%Y-%m-%d %H:%M:%S') if event.published_at else 'Unknown',
                'age_minutes': self._calculate_age_minutes(event.published_at)
            })
        return formatted

    def get_display_statistics(self, events: List[Event]) -> Dict:
        """ Calculate comprehensive statistics for dashboard widgets."""
        # Handle empty event list case
        if not events:
            return {
                'total_events': 0,
                'sources': {},
                'recent_events': 0,
                'oldest_event': None,
                'newest_event': None
            }

        # Count events by source for source distribution widget
        sources = {}
        for event in events:
            sources[event.source] = sources.get(event.source, 0) + 1

        # Calculate time-based statistics for dashboard widgets
        current_time = self._get_current_time()
        recent_events = 0

        # Count recent events (last hour)
        for event in events:
            if event.published_at:

                # Handle timezone-naive datetimes
                event_time = event.published_at

                # Check if event is within the last hour (3600 seconds)
                if (current_time - event_time).total_seconds() < 3600:
                    recent_events += 1

        # Find oldest and newest events for timeline display
        events_with_time = [e for e in events if e.published_at]
        oldest = min(events_with_time, key=lambda x: x.published_at) if events_with_time else None
        newest = max(events_with_time, key=lambda x: x.published_at) if events_with_time else None

        return {
            'total_events': len(events),
            'sources': sources,
            'recent_events': recent_events,
            'oldest_event': oldest.published_at if oldest else None,
            'newest_event': newest.published_at if newest else None
        }

    def apply_display_filters(self, events: List[Event],
                              source_filter: Optional[str] = None,
                              time_filter_hours: Optional[int] = None) -> List[Event]:
        """ Apply UI filters to event list for dashboard display.
            This method allows users to filter events by source and time range
            for focused dashboard views."""

        filtered = events

        # Apply source filter
        if source_filter:
            filtered = [e for e in filtered if e.source == source_filter]

        # Apply time filter
        if time_filter_hours:
            current_time = self._get_current_time()
            # Calculate cutoff time (current time minus filter hours)
            cutoff = current_time - timedelta(hours=time_filter_hours)

            filtered_by_time = []
            for event in filtered:
                if event.published_at:
                    # Handle timezone-naive datetimes
                    event_time = event.published_at
                    if event_time.tzinfo is None:
                        event_time = event_time.replace(tzinfo=timezone.utc)

                    # Include event if it's newer than the cutoff time
                    if event_time >= cutoff:
                        filtered_by_time.append(event)

            filtered = filtered_by_time

        return filtered

    def get_available_sources(self, events: List[Event]) -> List[str]:
        """Get unique source names for populating filter dropdowns."""
        # Extract unique sources and sort alphabetically for consistent UI
        return sorted(set(event.source for event in events if event.source))

    def clear_storage(self) -> None:
        """Clear all storage data - dashboard management function."""
        try:
            self.vector_storage.clear_index()
            logger.info("Storage cleared via dashboard")
        except Exception as e:
            logger.error(f"Error clearing storage: {e}")