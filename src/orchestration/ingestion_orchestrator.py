import logging
from typing import List, Dict, Optional, Union

from src.aggregation.data_aggregation_service import AggregationService, SourceConfig
from src.ingestion.ingestion_engine import IngestionEngine
from src.logging_setup import configure_logging
from src.models import Event, EnrichedEvent
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
    to coordinate multiple complex subsystems (aggregation, filtering, storage).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 vector_storage: Optional[VectorStorageService] = None):
        """Initialize the ingestion orchestrator with configuration and dependencies."""

        # Pre-configured data sources covering major IT/security information feeds
        self.default_sources = [

            # Reddit communities focused on system administration and outages
            SourceConfig("r/sysadmin", "reddit", {"url": "https://www.reddit.com/r/sysadmin.json?limit=8"}),
            SourceConfig("r/outages", "reddit", {"url": "https://www.reddit.com/r/outages.json?limit=3"}),
            SourceConfig("r/cybersecurity", "reddit", {"url": "https://www.reddit.com/r/cybersecurity.json?limit=4"}),

            # Security-focused RSS feeds from reputable sources
            SourceConfig("ars-security", "rss", {"url": "https://feeds.arstechnica.com/arstechnica/security"}),
            SourceConfig("krebs-security", "rss", {"url": "https://krebsonsecurity.com/feed/"}),

            # Cloud provider status feeds for infrastructure monitoring
            SourceConfig("aws-status", "rss", {"url": "https://status.aws.amazon.com/rss/all.rss"}),
            SourceConfig("azure-status", "rss", {"url": "https://azurestatuscdn.azureedge.net/en-us/status/feed/"}),
        ]

        self.ingestion_engine = IngestionEngine(threshold=threshold, vector_storage=vector_storage)

        self.aggregation_service = AggregationService(source_configs=self.default_sources)

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

        return self.ingestion_engine.filter_events(events)

    def rank_events(self, events: List[Event]) -> List[EnrichedEvent]:
        """Rank events """
        return self.ingestion_engine.rank_events(events)


    def store_events(self, events: List[Union[Event, EnrichedEvent]]) -> Dict:
        """Store processed events in the vector database."""
        return self.ingestion_engine.store_events(events)



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
            enriched_events = self.rank_events(relevant_events)

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
        return self.ingestion_engine.get_all_events_ranked()

    def update_threshold(self, new_threshold: float) -> None:
        """Update filtering threshold"""
        return self.ingestion_engine.update_threshold(new_threshold)
