import logging
from datetime import datetime, timezone
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, status

from src.logging_setup import configure_logging
from src.models import Event, IngestResponse, EnrichedEvent
from src.orchestration.ingestion_orchestrator import IngestionOrchestrator
from src.orchestration.ingestion_service_factory import IngestionServiceFactory

# Set logging
logger = logging.getLogger(__name__)
configure_logging()

# Initialize FastAPI app
app = FastAPI(
    title="IT Newsfeed Platform",
    description="Real-time IT newsfeed aggregation and filtering platform",
    version="1.0.0"
)

# Initialize engines
ingestion_orchestrator= IngestionOrchestrator(vector_storage=IngestionServiceFactory.create_shared_storage())


@app.post("/ingest", status_code=200, response_model=IngestResponse)
async def ingest_events(events: List[Event]) -> IngestResponse:
    """
    Events ingestion endpoint.
    """
    try:
        logger.info(f"Received {len(events)} events for ingestion")

        # Step 1: Filter events (includes deduplication + semantic filtering)
        filter_result = ingestion_orchestrator.filter_events(events)

        # Step 2: Rank events (relevant only)
        # Extract relevant events from filter_result
        relevant_events = [item['event'] for item in filter_result['relevant_events']]
        events_explanations = [item['explanation'] for item in filter_result['relevant_events']]
        enriched_events = ingestion_orchestrator.rank_events(relevant_events,events_explanations)

        # Step 3: Store events into index
        storage_result = ingestion_orchestrator.store_events(enriched_events)

        # Calculate totals for response
        total_events = len(events)
        stored_events = storage_result.get('stored_count', 0)

        logger.info(f"Ingestion complete: {stored_events}/{total_events} events stored")

        return IngestResponse(
            ingested_count=total_events,
            message=f"Successfully ingested {stored_events} relevant events out of {total_events} total"
        )

    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest events: {str(e)}"
        )


@app.get("/retrieve", response_model=List[Event])
async def retrieve_events() -> List[Event]:
    """
    Events retrieval endpoints.
    """
    try:
        logger.info(f"Retrieving and ranking events")

        # Enriched events sorted by final_score (descending) then by id for deterministic ordering
        ranked_events = ingestion_orchestrator.get_all_events_ranked()

        # Convert EnrichedEvents back to Event models for API response
        api_events = [
            Event.model_validate(enriched.model_dump(include={'id', 'source', 'title', 'body', 'published_at'}))
            for enriched in ranked_events
        ]

        logger.info(f"Returning {len(api_events)} events in ranked order")

        return api_events

    except Exception as e:
        logger.error(f"Error retrieving events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve events: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with system status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc),
        "storage": "Storage Vector DB"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "IT Newsfeed Platform",
        "version": "1.0.0",
        "description": "Real-time IT newsfeed aggregation with intelligent filtering and ranking",
        "endpoints": {
            "POST /ingest": "Filter, Rank, and Store new IT-relevant events",
            "GET /retrieve": "Retrieve ranked IT-relevant events",
            "GET /health": "System health check"
        }
    }

if __name__ == "__main__":
    logger.info("Starting IT Newsfeed Platform...")
    uvicorn.run(
        "src.api.api:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )