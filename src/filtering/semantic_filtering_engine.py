import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embedding.embedding_service import get_embedding_service_instance
from src.models import Event

logger = logging.getLogger(__name__)


class SemanticContentFilter:
    """ Semantic content filter for determining IT-relevance of events.

        This filter uses machine learning embeddings and cosine similarity to determine
        whether incoming events are relevant to IT operations. It works by

        1. Pre-computing embeddings for a set of IT-relevant reference phrases
        2. Computing embeddings for incoming events
        3. Measuring semantic similarity using cosine similarity
        4 . Filtering events based on a configurable threshold

        This approach allows for more nuanced filtering than keyword-based systems,
        as it can detect semantic relationships even when exact keywords don't match.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize the semantic content filter with IT-relevant reference phrases.
        """
        # Initialize embedding service for embeddings generation
        self.embedding_service = get_embedding_service_instance()

        # Comprehensive set of IT-relevant reference phrases
        self.it_reference_phrases = [
            # Infrastructure & Systems
            "server outage", "system failure", "network disruption",
            "database crash", "service downtime", "infrastructure issue",
            "hardware failure", "software malfunction", "system maintenance",

            # Security
            "security breach", "cyber attack", "data breach", "vulnerability",
            "malware detected", "unauthorized access", "security incident",
            "phishing attack", "ransomware", "security alert",

            # Development & Operations
            "critical bug", "emergency patch", "deployment failed",
            "application error", "API outage", "configuration error",
            "performance issue", "memory leak", "disk space full",

            # Monitoring & Alerts
            "system alert", "monitoring alert", "threshold exceeded",
            "service unavailable", "connection timeout", "high latency",

            # Cloud & DevOps
            "cloud service disruption", "container failure", "kubernetes error",
            "CI/CD pipeline failure", "automated backup failed"
        ]

        # Pre-compute embeddings for reference phrases to optimize performance
        logger.info(f"Pre-computing embeddings for {len(self.it_reference_phrases)} reference phrases...")
        self.it_reference_embeddings = self._precompute_reference_embeddings()

        # Set the similarity threshold for determining relevance
        self.threshold = threshold

        logger.info(
            f"SemanticContentFilter initialized: Threshold: {self.threshold}, Reference phrases: {len(self.it_reference_phrases)}")

    def _precompute_reference_embeddings(self) -> np.ndarray:
        """Pre-compute embeddings for all IT-relevant reference phrases."""
        try:
            # Convert each reference phrase to its embedding representation
            embeddings = []
            for phrase in self.it_reference_phrases:
                embedding = self.embedding_service.text_to_embedding(phrase)
                embeddings.append(embedding)

            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error pre-computing reference embeddings: {e}")
            return None

    def is_it_relevant(self, event: Event) -> bool:
        """
        Determine if an event is IT-relevant based on semantic similarity.

        This is the main filtering method that:
        1. Prepares the event text for analysis
        2. Converts event text to an embedding vector
        3. Computes cosine similarity against all pre-computed reference embeddings
        4. Determines relevance based on the highest similarity score

        The method uses cosine similarity, which measures the angle between vectors
        and is effective for determining semantic similarity regardless of text length.
        """
        try:
            # Combine event title and body into a single text for analysis
            event_text = self._prepare_event_text(event)

            # Convert event text to embedding vector
            event_embedding = self.embedding_service.text_to_embedding(event_text)
            event_embedding = np.array([event_embedding])  # Shape: (1, embedding_dim)

            # Compute cosine similarities against pre-computed reference embeddings
            cosine_scores = cosine_similarity(event_embedding, self.it_reference_embeddings)
            max_score = float(cosine_scores.max())

            # Determine relevance based on threshold comparison
            is_relevant = max_score >= self.threshold

            logger.info(f"Event {event.id}: similarity={max_score:.3f}, relevant={is_relevant}")
            return is_relevant

        except Exception as e:
            logger.error(f"Error in semantic filtering for event {event.id}: {str(e)}")
            # Fail safe: consider event relevant if error occurs
            return True

    def get_filter_explanation(self, event: Event) -> dict:
        """
        Get detailed explanation of filtering decision.

        This method provides information about how the filtering
        decision was made, including:
        - The highest similarity score achieved
        - Which reference phrase was the best match
        - Whether the event passed the threshold
        - The final filtering decision

        """
        try:
            # Prepare event text for analysis
            event_text = self._prepare_event_text(event)

            # Get embedding for the event text
            event_embedding = self.embedding_service.text_to_embedding(event_text)
            event_embedding = np.array([event_embedding])

            # Compute similarities using pre-computed reference embeddings
            cosine_scores = cosine_similarity(event_embedding, self.it_reference_embeddings)
            cosine_scores = cosine_scores.flatten()  # Convert to 1D array

            # Find the best matching reference phrase
            max_score = float(cosine_scores.max())
            max_idx = int(cosine_scores.argmax())
            matched_phrase = self.it_reference_phrases[max_idx]

            # Determine relevance based on threshold
            is_relevant = max_score >= self.threshold

            return {
                "event_id": event.id,
                "event_title": event.title,
                "max_similarity_score": round(max_score, 3),
                "matched_reference_phrase": matched_phrase,
                "threshold": self.threshold,
                "is_relevant": is_relevant,
                "decision": "KEEP" if is_relevant else "FILTER_OUT"
            }

        except Exception as e:
            logger.error(f"Error in get_filter_explanation for event {event.id}: {e}")
            return {
                "event_id": event.id,
                "event_title": event.title,
                "max_similarity_score": 0.0,
                "matched_reference_phrase": "error",
                "threshold": self.threshold,
                "is_relevant": True,  # Fail safe
                "decision": "KEEP",
                "error": str(e)
            }

    def _prepare_event_text(self, event: Event) -> str:
        """
        Prepare event text for embedding generation by combining title and body.
        """
        # Extract and clean title and body text
        title = (event.title or "").strip()
        body = (event.body or "").strip()

        # Combine title and body
        if title and body:
            return f"{title} {body}"
        elif title:
            return title
        elif body:
            return body
        else:
            return ""

    def update_threshold(self, threshold: float):
        """
            Update the relevance threshold for dynamic tuning.

            This method allows runtime adjustment of the filtering threshold
            without recreating the entire filter instance.
        """
        self.threshold = threshold

    def get_performance_stats(self) -> dict:
        """Get performance-related statistics and configuration information."""
        return {
            "reference_phrases_count": len(self.it_reference_phrases),
            "precomputed_embeddings": self.it_reference_embeddings is not None,
            "embedding_dimension": self.it_reference_embeddings.shape[
                1] if self.it_reference_embeddings is not None else 0,
            "threshold": self.threshold
        }