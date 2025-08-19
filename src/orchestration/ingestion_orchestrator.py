import logging
from typing import List, Dict, Optional, Union

from src.aggregation.data_aggregation_service import AggregationService, SourceConfig
from src.filtering.semantic_filtering_engine import SemanticContentFilter
from src.logging_setup import configure_logging
from src.models import Event, EnrichedEvent
from src.ranking.ranking_engine import RankingEngine
from src.storage.storage_service import VectorStorageService

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()


class IngestionOrchestrator:
    """
    High-level orchestrator for the complete ingestion pipeline.

    This class coordinates the entire data ingestion workflow from multiple sources:

    **Pipeline Stages:**
    1. **Fetch**: Collect events from various sources (Reddit, RSS feeds, cloud status pages)
    2. **Filter**: Remove duplicates and apply semantic filtering for IT relevance
    3. **Rank** : Rank events based on their ranking score
    4. **Store**: Persist relevant events in vector storage for retrieval

    Design Pattern :

    This orchestrator follows the "Facade" design pattern, providing a simplified interface
    to coordinate multiple complex subsystems (aggregation, filtering, ranking storage).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 vector_storage: Optional[VectorStorageService] = None):
        """Initialize the ingestion orchestrator with configuration and dependencies."""

        # Pre-configured data sources covering major IT/security information feeds
        self.default_sources = [

            # Reddit communities focused on system administration and outages
            SourceConfig("r/sysadmin", "reddit", {"url": "https://www.reddit.com/r/sysadmin.json"}),
            SourceConfig("r/outages", "reddit", {"url": "https://www.reddit.com/r/outages.json"}),
            SourceConfig("r/cybersecurity", "reddit", {"url": "https://www.reddit.com/r/cybersecurity.json"}),

            # Security-focused RSS feeds from reputable sources
            SourceConfig("ars-security", "rss", {"url": "https://feeds.arstechnica.com/arstechnica/security"}),
            SourceConfig("krebs-security", "rss", {"url": "https://krebsonsecurity.com/feed/"}),

            # Cloud provider status feeds for infrastructure monitoring
            SourceConfig("aws-status", "rss", {"url": "https://status.aws.amazon.com/rss/all.rss"}),
            SourceConfig("azure-status", "rss", {"url": "https://azurestatuscdn.azureedge.net/en-us/status/feed/"}),
        ]

        self.threshold = threshold

        # Initialize semantic filter for determining IT-relevance of events
        self.semantic_filter = SemanticContentFilter(threshold=threshold)

        # Initialize ranking engine for scoring event importance
        self.ranking_engine = RankingEngine()

        # Initialize storage service for persisting processed events
        self.storage_service = vector_storage or VectorStorageService()

        # Initialize storage service for fetching events from sources
        self.aggregation_service = AggregationService(source_configs=self.default_sources)

        logger.info("Ingestion orchestrator initialized")


    async def fetch_events(self) -> List[Event]:
        """Fetch events from configured data sources using the aggregation service"""

        try:
            # Fetch events asynchronously from all configured sources
            events = await self.aggregation_service.fetch_all_events()
            logger.info(f"Fetched {len(events)} events from {len(self.default_sources)} sources")
            return events
        except Exception as e:
            # Log error and return empty list to maintain pipeline stability
            logger.error(f"Error fetching events: {e}")
            return []

    def filter_events(self, events: List[Event]) -> Dict:
        """Apply comprehensive filtering to events including duplicate detection and semantic filtering."""

        # Initialize result structure with default values
        filter_result = {
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

    def rank_events(self, events: List[Event], filter_explanations: List[Dict]) -> List[EnrichedEvent]:
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

    async def run_full_pipeline(self, store_relevant_only: bool = True) -> Dict:
        """
        Run the complete ingestion pipeline from fetch to storage.
        """
        logger.info("Starting full ingestion pipeline")

        try:
            # Step 1 : Fetch events from sources
            events = await self.fetch_events()

            # Handle case where no events were fetched
            if not events:
                return {
                    'success': True,
                    'fetched_count': 0,
                    'duplicates_skipped': 0,
                    'relevant_count': 0,
                    'stored_count': 0,
                    'message': 'No events fetched'
                }

            # Step 2 : Filter events  (includes deduplication + semantic filtering)
            filter_result = self.filter_events(events)

            # Step 3 : Rank events (relevant only)
            # Extract relevant events form filter_result
            relevant_events = [item['event'] for item in filter_result['relevant_events']]
            events_explanations = [item['explanation'] for item in filter_result['relevant_events']]

            enriched_events = self.rank_events(relevant_events, events_explanations)

            # Step 4: Store events into index
            storage_result = self.store_events(enriched_events)

            # Compile results
            result = {
                'success': True,
                'fetched_count': len(events),
                'duplicates_skipped': filter_result['statistics']['duplicates_skipped'],
                'relevant_count': filter_result['statistics']['relevant_count'],
                'filtered_count': filter_result['statistics']['filtered_count'],
                'stored_count': storage_result.get('stored_count', 0),
                'total_in_storage': storage_result.get('total_in_storage', 0),
                'filter_statistics': filter_result['statistics']
            }

            logger.info(f"Pipeline complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {
                'success': False,
                'error': str(e),
                'fetched_count': 0,
                'duplicates_skipped': 0,
                'relevant_count': 0,
                'stored_count': 0
            }

    def get_all_events_ranked(self) -> List[EnrichedEvent]:
        """
        Retrieve all stored events in ranked order by importance.
        """
        return self.storage_service.get_all_events_ranked()

    def update_threshold(self, new_threshold: float) -> None:
        """Update filtering threshold.
           This method allows runtime adjustment of filtering sensitivity without recreating the orchestrator instance"""
        self.threshold = new_threshold
        self.semantic_filter.update_threshold(new_threshold)
        logger.info(f"Updated threshold to {new_threshold}")
