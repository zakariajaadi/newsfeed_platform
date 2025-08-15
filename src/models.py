"""
Data models for the IT Newsfeed Platform

This module defines the core data structures that match the API contract
specified in the assessment requirements.
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from typing import Optional, List

class Event(BaseModel):
    """
    Core Event model that matches the API contract specification.

    This model represents both input events (from /ingest) and output events (from /retrieve).
    The fields match exactly what the automated test harness expects.
    """
    id: str = Field(..., description="Unique identifier for the event")
    source: str = Field(..., description="Source of the event (e.g., 'reddit', 'ars-technica')")
    title: str = Field(..., description="Event title")
    body: Optional[str] = Field(None, description="Event body content (optional)")
    published_at: datetime = Field(..., description="Publication timestamp in ISO-8601/RFC 3339 format UTC")

    @field_validator("published_at", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return v

    class Config:
        # Ensure datetime is serialized in ISO format for API compatibility
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }



class EnrichedEvent(Event):
    """
    Extended Event model with processing metadata for internal use
    Inherits all Event fields and adds processing-specific fields
    """
    # Detailed scoring breakdown
    semantic_score: float= Field(..., description="Semantic score")
    urgency_score: float = Field(..., description="Urgency score")
    recency_score: float = Field(..., description="Recency score")
    source_score: float = Field(..., description="Source score")
    final_score: Optional[float] = Field(None, description="Ranking importance score")

    # Filtering metadata
    matched_reference: str = Field(..., description="Matched reference")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    filter_passed: bool = Field(..., description="Whether event passed relevance filtering")

    @field_validator("ingested_at", mode="before")
    @classmethod
    def ensure_utc(cls, v):
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return v

class IngestResponse(BaseModel):
    """
    Response model for the /ingest endpoint.

    Returns acknowledgment as required by the API contract.
    """
    status: str = "success"
    ingested_count: int
    message: str = "Events ingested successfully"


# Test the models work correctly
if __name__ == "__main__":
    # Example of creating an event
    test_event = Event(
        id="test-001",
        source="reddit",
        title="Major Azure outage affecting multiple services",
        body="Users reporting widespread issues with Azure services including authentication failures.",
        published_at=datetime.now()
    )

    print("Test Event Created:")
    print(test_event.model_dump_json(indent=2))

    # Example of ingest response
    response = IngestResponse(
        ingested_count=1,
        message="Successfully processed 1 event"
    )