import hashlib
import logging
import os
import pickle
from typing import List, Union

import faiss
import numpy as np

from src.embedding.embedding_service import EmbeddingService
from src.logging_setup import configure_logging
from src.models import Event, EnrichedEvent
from src.ranking.ranking_engine import RankingEngine

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()



class VectorStorageService:

    def __init__(self, embedding_dim, index_file_path, metadata_file_path, autosave_every):
        self.embedding_dim = embedding_dim         # Embedding dimension
        self.embedding_service = EmbeddingService() # Embedding service
        self.index_file_path = index_file_path # Index file path
        self.metadata_file_path = metadata_file_path # ID-to-event file path
        self.autosave_every = autosave_every  # After how many writes the index should save automatically (0 = manual only on exit)
        self._pending_writes = 0 # Counter of unsaved changes
        self._load_or_create_index() # Index

    def _load_or_create_index(self):
        # If index exists, load it
        if os.path.exists(self.index_file_path) and os.path.exists(self.metadata_file_path):
            self.index = faiss.read_index(self.index_file_path)
            with open(self.metadata_file_path, 'rb') as f:
                metadata = pickle.load(f)
                self.id_to_event = metadata['id_to_event']
                self.next_id = metadata['next_id']
            logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
        # Else, create it
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
            self.id_to_event, self.next_id = {}, 0
            logger.info("Created new index with ID mapping")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)

    def _compute_hash(self, event: Union[Event, EnrichedEvent]) -> str:
        content = f"{event.title} {event.body}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def save_index(self):

        # Save index
        faiss.write_index(self.index, self.index_file_path)

        # Save index metadata
        with open(self.metadata_file_path, 'wb') as f:
            pickle.dump({'id_to_event': self.id_to_event, 'next_id': self.next_id}, f)
        logger.info(f"Index saved with {self.index.ntotal} vectors")

        # Reset _pending_writes (Unsaved changes counter)
        self._pending_writes = 0

    def store_event(self, event: Union[Event, EnrichedEvent]):
        """Store a single event """

        # Converts event text to embedding and normalize it
        text = f"{event.title} {event.body}"
        embedding = self.embedding_service.text_to_embedding(text)
        embedding = self._normalize(np.array([embedding], dtype='float32'))

        # Add event embedding to index with a custom ID
        self.index.add_with_ids(embedding, np.array([self.next_id]))

        # Saves the event object in id_to_event mapping
        event_hash = self._compute_hash(event)
        self.id_to_event[self.next_id] = {
            "hash": event_hash,
            "event": event
        }
        self.next_id += 1

        # Increment unsaved changes counter
        self._pending_writes += 1

        # If autosave threshold reached, save index
        if self.autosave_every and self._pending_writes >= self.autosave_every:
            self.save_index()

    def store_events(self, events: List[Union[Event, EnrichedEvent]]):
        """Store multiple events (assumes duplicates already filtered out)."""

        if not events:
            logger.info("No events to store")
            return

        embeddings, ids = [], []

        # Generate embeddings for the events
        for event in events:
            # Converts event text to embedding
            text = f"{event.title} {event.body}"
            embedding = self.embedding_service.text_to_embedding(text)

            # Add embedding to list
            embeddings.append(embedding)
            ids.append(self.next_id)

            # Save the event in id_to_event mapping
            event_hash = self._compute_hash(event)
            self.id_to_event[self.next_id] = {
                "hash": event_hash,
                "event": event
            }
            self.next_id += 1

        # Normalize embeddings
        embeddings = self._normalize(np.array(embeddings, dtype='float32'))

        # Add embeddings to index with custom IDs
        self.index.add_with_ids(embeddings, np.array(ids))
        self._pending_writes += len(events)

        # If autosave threshold reached, save index
        if self.autosave_every and self._pending_writes >= self.autosave_every:
            self.save_index()

    def search(self, query: str, top_k=5) -> List[Union[Event, EnrichedEvent]]:
        if self.index.ntotal == 0:
            return []

        # Generate embedding for query and normalize it
        query_emb = self.embedding_service.text_to_embedding(query)
        query_emb = self._normalize(np.array([query_emb], dtype='float32'))

        # Search FAISS index for top_k matches.
        top_k = min(top_k, self.index.ntotal)
        _, indices = self.index.search(query_emb, top_k)

        # Look up stored Event objects by ID and returns them
        events = [self.id_to_event[i]['event'] for i in indices[0]
                  if i != -1 and i in self.id_to_event]

        return events

    def get_all_events(self) -> List[Union[Event, EnrichedEvent]]:
        """
        Retrieve all stored events from the vector storage.

        Returns:
            List[Union[Event, EnrichedEvent]]: List of all events stored in the index
        """
        # Extract only events metadata structure
        events = [meta['event'] for meta in self.id_to_event.values()]

        return events

    def get_all_events_ranked(self) -> List[EnrichedEvent]:
        """
        Retrieve all stored events in ranked order by importance.

        It's used by dashboard and API services to present events in priority order.

        The ranking logic:
        1. Primary sort: final_score (descending) - highest importance first
        2. Secondary sort: event.id (ascending) -  When two events have the same score, we sort by ID
        3. Events without scores (filtered out) get score of 0

        This ensures that:
        - Most important events appear first
        - Ordering is consistent and predictable
        - All stored events are included in results

        Returns:
            List[EnrichedEvent]: Events sorted by final_score (highest first)
        """

        # Retrieve all enriched events from storage
        all_enriched_events = self.get_all_events()

        # Handle empty storage case
        if not all_enriched_events:
            return []

        # Since events already have final_score from ingestion we just sort them
        # Sort events by importance score (descending) with secondary sort by ID for determinism
        sorted_enriched_events = sorted(
            all_enriched_events,
            key=lambda x: (-(x.final_score or 0), x.id)
        )

        logger.info(f"Retrieved {len(sorted_enriched_events)} events in ranked order")
        return sorted_enriched_events

    def get_all_events_reranked(self,
                                ranking_engine: RankingEngine,
                                importance_weight: float = 0.7,
                                recency_weight: float = 0.3) -> List[EnrichedEvent]:
        """
        Rerank events dynamically based on user preferences.
        """

        # Retrieve all enriched events from storage
        all_enriched_events = self.get_all_events()

        # Handle empty storage case
        if not all_enriched_events:
            return []

        for enriched_event in all_enriched_events:
            recency_score = ranking_engine.calculate_recency_score(enriched_event.published_at)
            importance_score = enriched_event.importance_score
            new_final_score = importance_weight * importance_score + recency_weight * recency_score
            enriched_event.final_score = new_final_score

        # Since events already have final_score from ingestion we just sort them
        # Sort events by importance score (descending) with secondary sort by ID for determinism
        sorted_enriched_events = sorted(
            all_enriched_events,
            key=lambda x: (-(x.final_score or 0), x.id)
        )

        logger.info(f"Retrieved {len(sorted_enriched_events)} events in ranked order")
        return sorted_enriched_events

    def filter_duplicates(self, events: List[Union[Event, EnrichedEvent]]) -> List[Union[Event, EnrichedEvent]]:
        """Filter out duplicate events before processing. Returns only new events."""
        new_events = []
        duplicate_count = 0

        for event in events:
            event_hash = self._compute_hash(event)

            # Check if this hash already exists
            if any(meta['hash'] == event_hash for meta in self.id_to_event.values()):
                duplicate_count += 1
                continue

            new_events.append(event)

        if duplicate_count > 0:
            logger.info(
                f"Early deduplication: {duplicate_count} duplicates filtered out, {len(new_events)} new events to process")

        return new_events

    def remove_event(self, event_id: int):
        if event_id in self.id_to_event:
            self.index.remove_ids(np.array([event_id]))
            del self.id_to_event[event_id]
            logger.info(f"Removed event with ID {event_id}")

    def clear_index(self):
        self.index.reset()
        self.id_to_event, self.next_id = {}, 0
        for path in (self.index_file_path, self.metadata_file_path):
            if os.path.exists(path):
                os.remove(path)
        logger.info("Index cleared")

    def get_index_stats(self):
        return {
            'total_vectors': self.index.ntotal,
            'total_events': len(self.id_to_event),
            'next_id': self.next_id,
            'embedding_dim': self.embedding_dim
        }

    def __del__(self):
        try:
            if self._pending_writes > 0:
                self.save_index()
        except:
            pass


