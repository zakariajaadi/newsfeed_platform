import asyncio
import logging
import time
from typing import Dict

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.logging_setup import configure_logging
from src.orchestration.ingestion_service_factory import IngestionServiceFactory

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()

class SchedulerService:
    """
    Manages scheduled and background execution of the ingestion pipeline.

    This service provides automated, periodic execution of the ingestion process

    Architecture :

    This class focuses purely on scheduling orchestration concerns,
    delegating actual ingestion logic to the IngestionOrchestrator
    """

    def __init__(self, interval_minutes: int):
        """Initialize the scheduler service with ingestion orchestrator and job scheduler."""

        # Initialize orchestrator for actual ingestion
        self.orchestrator = IngestionServiceFactory.create_orchestrator()

        # BackgroundScheduler for non-async environments
        self.scheduler = BackgroundScheduler() # AsyncIOScheduler()

        # Scheduler interval
        self.interval = interval_minutes

        # Track scheduler state and execution history
        self.is_running = False
        self.last_run_result = None

    def run_ingestion_cycle(self) -> Dict:
        """Execute a single scheduled ingestion cycle"""
        logger.info("Starting scheduled ingestion cycle")

        try:
            # Execute the async pipeline in synchronous scheduler context
            result = asyncio.run(self.orchestrator.run_full_pipeline())

            # Enhance result with scheduling-specific metadata
            self.last_run_result = {
                **result,
                'timestamp': time.time(),
                'run_type': 'scheduled'
            }

            # Log execution summary for monitoring
            if result.get('success'):
                logger.info(f"Scheduled ingestion completed: "
                            f"fetched={result.get('fetched_count', 0)}, "
                            f"relevant={result.get('relevant_count', 0)}, "
                            f"stored={result.get('stored_count', 0)}")
            else:
                logger.error(f"Scheduled ingestion failed: {result.get('error')}")

            return self.last_run_result

        except Exception as e:
            # Handle scheduler-specific errors
            error_result = {
                'success': False,
                'error': str(e),
                'timestamp': time.time(),
                'run_type': 'scheduled'
            }
            self.last_run_result = error_result
            logger.error(f"Scheduled ingestion exception: {e}")
            return error_result

    def start_scheduler(self) -> None:
        """Start the periodic scheduler with specified interval."""

        # Prevent multiple scheduler instances
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        try:
            # Configure scheduled job
            self.scheduler.add_job(
                self.run_ingestion_cycle, # Function to execute
                trigger=IntervalTrigger(minutes=self.interval), # Interval-based trigger
                id='scheduled_ingestion', # Interval-based trigger
                replace_existing=True, # Allow reconfiguration
                max_instances=1  # Prevent overlapping runs
            )

            # Start the scheduler background thread
            self.scheduler.start()
            self.is_running = True
            logger.info(f"Scheduler started - interval: {self.interval} minutes")

        except Exception as e:
            # Handle scheduler startup failures
            logger.error(f"Failed to start scheduler: {e}")
            self.is_running = False

    def stop_scheduler(self) -> None:
        """Stop the scheduler and clean up background threads."""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)
            self.is_running = False
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

    def get_status(self) -> Dict:
        """Get scheduler status and statistics."""

        # Collect information about active scheduled jobs
        jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger)
                })

        return {
            'is_running': self.is_running,
            'scheduler_running': getattr(self.scheduler, 'running', False),
            'jobs': jobs,
            'last_run_result': self.last_run_result,
            'storage_stats': self.orchestrator.storage_service.get_index_stats()
        }

    def update_threshold(self, new_threshold: float) -> None:
        """Update the filtering threshold through the orchestrator."""
        self.orchestrator.update_threshold(new_threshold)

    async def run_manual_ingestion(self) -> Dict:
        """Execute a manual ingestion cycle outside scheduled operations."""

        logger.info("Starting manual ingestion")

        # Execute pipeline and add manual execution metadata
        result = await self.orchestrator.run_full_pipeline()
        result['run_type'] = 'manual'
        result['timestamp'] = asyncio.get_event_loop().time()
        return result


