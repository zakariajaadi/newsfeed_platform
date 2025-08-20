import logging
import os

from src.aggregation.data_aggregation_service import AggregationService
from src.config import get_config
from src.filtering.semantic_filtering_engine import SemanticContentFilter
from src.logging_setup import configure_logging
from src.orchestration.ingestion_orchestrator import IngestionOrchestrator
from src.ranking.ranking_engine import RankingEngine
from src.storage.storage_service import VectorStorageService

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()

class IngestionServiceFactory:
    """Factory class for creating and configuring ingestion-related services.

       This factory provides centralized creation and configuration of ingestion
       components with consistent default settings and shared dependencies.
    """


    @staticmethod
    def create_orchestrator() -> IngestionOrchestrator:
        """Create a configured ingestion orchestrator instance."""

        cfg = get_config()

        # Initialize semantic filter for determining IT-relevance of events
        semantic_filter = SemanticContentFilter(cfg.filtering.threshold,
                                                cfg.filtering.it_reference_phrases)

        # Initialize ranking engine for scoring event importance
        ranking_engine = RankingEngine(cfg.ranking.content_scoring_dict,
                                       cfg.ranking.final_scoring_dict,
                                       cfg.ranking.urgency_multipliers,
                                       cfg.ranking.source_ranking_weights,
                                       cfg.ranking.recency_half_life_hours)

        # Initialize storage service for persisting processed events
        vector_storage = IngestionServiceFactory.create_shared_storage()

        # Initialize aggregation service for fetching events from sources
        # Uses provided sources or defaults to built-in IT-focused sources
        aggregation_service = AggregationService(cfg.fetching.all_sources)

        return IngestionOrchestrator(semantic_filter=semantic_filter,
                                     ranking_engine=ranking_engine,
                                     vector_storage=vector_storage,
                                     aggregation_service=aggregation_service)


    @staticmethod
    def create_shared_storage(data_dir: str = None) -> VectorStorageService:
        """Create a shared vector storage instance with persistent file storage :

        - Ensures data directory exists for file storage
        - Uses consistent file naming for index and metadata
        - Configures automatic saving for data durability
        """
        # Fetch config
        cfg = get_config()
        # Use absolute path relative to project root
        if data_dir is None:
            # Get the project root directory (go up from wherever this file is)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # Go up 3 levels
            data_dir = os.path.join(project_root, "data")

        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Use absolute file paths
        index_path = os.path.join(data_dir, cfg.vector_storage.index_file_name)
        metadata_path = os.path.join(data_dir,cfg.vector_storage.metadata_file_name)
        autosave_every = cfg.vector_storage.autosave_every

        logger.info(f"Using storage paths: {index_path}")

        return VectorStorageService(
            embedding_dim=cfg.vector_storage.embedding_dim,
            index_file_path=index_path,
            metadata_file_path=metadata_path,
            autosave_every=autosave_every
        )

    @staticmethod
    def create_ranking_engine() -> RankingEngine:
        """Create a configured ingestion orchestrator instance."""

        cfg = get_config()

        # Initialize ranking engine for scoring event importance
        ranking_engine = RankingEngine(cfg.ranking.content_scoring_dict,
                                       cfg.ranking.final_scoring_dict,
                                       cfg.ranking.urgency_multipliers,
                                       cfg.ranking.source_ranking_weights,
                                       cfg.ranking.recency_half_life_hours)

        return ranking_engine


