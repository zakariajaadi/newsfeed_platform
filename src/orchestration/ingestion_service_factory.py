import logging
from typing import Optional
import os

from src.logging_setup import configure_logging
from src.storage.storage_service import VectorStorageService
from src.orchestration.scheduler_service import SchedulerService
from src.orchestration.ingestion_orchestrator import IngestionOrchestrator

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()

class IngestionServiceFactory:
    """Factory class for creating and configuring ingestion-related services.

       This factory provides centralized creation and configuration of ingestion
       components with consistent default settings and shared dependencies.
    """


    @staticmethod
    def create_orchestrator(threshold: float = 0.5, vector_storage: Optional[VectorStorageService] = None) -> IngestionOrchestrator:
        """Create a configured ingestion orchestrator instance."""
        return IngestionOrchestrator(threshold, vector_storage)

    @staticmethod
    def create_scheduler(threshold: float = 0.5, vector_storage: Optional[VectorStorageService] = None) -> SchedulerService:
        """Create a configured scheduler service instance."""
        return SchedulerService(threshold, vector_storage)

    @staticmethod
    def create_shared_storage(data_dir: str = None, autosave_every: int = 1) -> VectorStorageService:
        """Create a shared vector storage instance with persistent file storage :

        - Ensures data directory exists for file storage
        - Uses consistent file naming for index and metadata
        - Configures automatic saving for data durability
        """
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
        index_path = os.path.join(data_dir, "vector_index.faiss")
        metadata_path = os.path.join(data_dir, "index_metadata.pkl")

        logger.info(f"Using storage paths: {index_path}")

        return VectorStorageService(
            index_file_path=index_path,
            metadata_file_path=metadata_path,
            autosave_every=autosave_every
        )

