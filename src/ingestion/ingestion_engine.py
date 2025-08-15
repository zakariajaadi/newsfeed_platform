import logging
from datetime import datetime,timezone
from typing import List, Dict, Union, Optional


from src.filtering.semantic_filtering_engine import SemanticContentFilter
from src.logging_setup import configure_logging
from src.models import Event, EnrichedEvent
from src.ranking.ranking_engine import RankingEngine
from src.storage.storage_service import VectorStorageService

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()


class IngestionEngine:
    """
    Central engine for processing and ingesting events through a complete pipeline.

    The IngestionEngine orchestrates the entire event processing workflow:

    1. Semantic Filtering: Uses ML-based semantic analysis to determine if events
       are relevant to IT operations
    2. Ranking/Scoring: Calculates importance scores for relevant events
    3. Enrichment: Adds metadata and processing information to events
    4. Storage: Persists processed events in vector storage for retrieval
    """

    def __init__(self, threshold: float = 0.5, vector_storage: Optional[VectorStorageService] = None ):
        """
        Initialize ingestion service with required components
        """

        self.threshold = threshold

        # Initialize semantic filter for determining IT-relevance of events
        self.semantic_filter = SemanticContentFilter(threshold=threshold)

        # Initialize ranking engine for scoring event importance
        self.ranking_engine = RankingEngine()

        # Initialize storage service for persisting processed events
        self.storage_service = vector_storage or VectorStorageService()

        logger.info("Ingestion service initialized")


    def get_all_events_ranked(self) -> List[EnrichedEvent]:
        """
        Retrieve all stored events in ranked order by importance.
        """
        return self.storage_service.get_all_events_ranked()


    def filter_events(self, events: List[Event]) -> Dict:
        """Apply comprehensive filtering to events including duplicate detection and semantic filtering."""

        # Initialize result structure with default values
        filter_result={
            'relevant_events': [],
            'filtered_events': [],
            'statistics': {
                'total_processed': len(events),
                'duplicates_skipped': 0,
                'relevant_count': 0,
                'filtered_count': 0,
                'error_count': 0
            }
        }

        # Handle empty input case
        if not events:
            return filter_result

        # STEP 1 -  DUPLICATE DETECTION
        # Filter out duplicates based on events hash
        logger.info(f"Checking {len(events)} events for duplicates...")
        new_events = self.storage_service.filter_duplicates(events)
        duplicates_skipped = len(events) - len(new_events)

        if duplicates_skipped > 0:
            logger.info(f"Skipped {duplicates_skipped} duplicates.")
            filter_result['statistics']['duplicates_skipped'] = duplicates_skipped

        # If all events were duplicates, skip semantic filtering
        if not new_events:
            logger.info("All events were duplicates, skipping semantic filtering")
            return filter_result

        # STEP 2 - SEMANTIC FILTERING
        # Apply ML-based relevance filtering (only to new events)
        logger.info(f"🎯 Running semantic filtering on {len(new_events)} new events...")
        relevant_events = []
        filtered_events = []
        errors = 0

        # Process each new event through semantic filtering
        for event in new_events:
            try:
                # Determine if event is relevant to IT operations
                is_relevant = self.semantic_filter.is_it_relevant(event)
                explanation = self.semantic_filter.get_filter_explanation(event)

                # Categorize event based on relevance with explanation
                if is_relevant:
                    relevant_events.append({
                        'event': event,
                        'explanation': explanation
                    })
                else:
                    filtered_events.append({
                        'event': event,
                        'explanation': explanation
                    })

            except Exception as e:
                # Log individual event errors and continue processing batch
                logger.error(f"Error filtering event {event.id}: {e}")
                errors += 1

        # Update result structure with filtering outcomes
        filter_result['relevant_events'] = relevant_events
        filter_result['filtered_events'] = filtered_events
        filter_result['statistics']['relevant_count'] = len(relevant_events)
        filter_result['statistics']['filtered_count'] = len(filtered_events)
        filter_result['statistics']['error_count'] = errors

        return filter_result



    def rank_events(self,  events: List[Event], filter_explanations: List[Dict]) -> List[EnrichedEvent] :
        """Rank events."""
        return self.ranking_engine.rank_events(events, filter_explanations)

    def store_events(self, events: List[Union[Event, EnrichedEvent]]) -> Dict:
        """Store processed events in the vector database."""

        # Handle empty input case
        if not events:
            return {'stored_count': 0, 'total_in_storage': 0}

        try:
            # Capture initial state to calculate delta
            initial_count = self.storage_service.get_index_stats()['total_events']

            # Store events in vector database (includes embedding generation)
            self.storage_service.store_events(events)

            # Capture final state to calculate actual stored count
            final_count = self.storage_service.get_index_stats()['total_events']

            # Calculate actual number of events stored (handles potential duplicates)
            stored_count = final_count - initial_count
            logger.info(f"Stored {stored_count} events in vector database")

            return {
                'stored_count': stored_count,
                'total_in_storage': final_count
            }
        except Exception as e:
            # Provide detailed error information
            logger.error(f"Error storing events: {e}")
            return {'stored_count': 0, 'total_in_storage': 0, 'error': str(e)}

    def update_threshold(self, new_threshold: float) -> None:
        """Update filtering threshold.
           This method allows runtime adjustment of filtering sensitivity without recreating the orchestrator instance"""
        self.threshold = new_threshold
        self.semantic_filter.update_threshold(new_threshold)
        logger.info(f"Updated threshold to {new_threshold}")
