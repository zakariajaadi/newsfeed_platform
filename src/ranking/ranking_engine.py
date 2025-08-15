import logging
from datetime import datetime, timezone
from typing import Dict, List

from src.filtering.semantic_filtering_engine import SemanticContentFilter
from src.models import Event, EnrichedEvent
from src.logging_setup import configure_logging

# Set logger
logger = logging.getLogger(__name__)
configure_logging()

class RankingEngine:
    """
    Calculates ranking scores by combining semantic content analysis with recency

    Reuses SemanticContentFilter for semantic similarity analysis - no duplication!
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize ranking engine with semantic content filter"""

        # Reuse the semantic content filter for text analysis
        self.semantic_filter = SemanticContentFilter(threshold=threshold)

        # Additional importance boosters for ranking (beyond basic IT relevance)
        self.urgency_multipliers = {
            'critical': 1.5,
            'urgent': 1.4,
            'emergency': 1.6,
            'breaking': 1.3,
            'major': 1.2,
            'severe': 1.4,
            'immediate': 1.3,
            'alert': 1.2,
        }

        # Source credibility for ranking
        self.source_ranking_weights = {
            'azure-health': 1.8,    # Official status pages get boost
            'aws-status': 1.8,
            'azure': 1.7,
            'security': 1.6,       # Security advisories
            'cert': 1.6,
            'cve': 1.7,
            'devops': 1.3,
            'monitoring': 1.4,
            'reddit': 1.0,         # Community reports (baseline)
            'ars-technica': 1.2,   # Tech journalism
            'github': 1.1,
            'office': 0.8,         # Less critical for IT ops
            'hr': 0.7,
        }

        # SCORING WEIGHTS - How to balance importance vs recency
        self.semantic_weight = 0.5      # 50% semantic similarity to IT content
        self.urgency_weight = 0.2       # 20% urgency keywords
        self.recency_weight = 0.2       # 20% how recent it is
        self.source_weight = 0.1        # 10% source credibility

        # RECENCY DECAY - How quickly old news becomes less important
        self.recency_half_life_hours = 12  # Score halves every 12 hours (faster decay for IT)

    def calculate_semantic_importance_score(self, event: Event) -> Dict[str, float]:
        """
        Calculate importance using semantic similarity + ranking-specific boosts

        Args:
            event: Event to score

        Returns:
            Dict with component scores and final importance
        """

        # Step 1: Get semantic similarity score from content filter
        # This gives us the core IT relevance based on embeddings
        filter_explanation = self.semantic_filter.get_filter_explanation(event)
        semantic_score = filter_explanation['max_similarity_score']

        # Step 2: Apply ranking-specific urgency multipliers
        urgency_score = self._calculate_urgency_score(event)

        # Step 3: Apply source credibility for ranking
        source_score = self.source_ranking_weights.get(event.source.lower(), 1.0)
        # Normalize source score to 0-1 range
        normalized_source_score = min(source_score / 2.0, 1.0)

        # Step 4: Calculate recency score
        recency_score = self.calculate_recency_score(event.published_at)

        # Step 5: Combine all factors with weights
        importance_score = (
            semantic_score * self.semantic_weight +
            urgency_score * self.urgency_weight +
            recency_score * self.recency_weight +
            normalized_source_score * self.source_weight
        )

        component_scores = {
            'semantic_score': semantic_score,
            'urgency_score': urgency_score,
            'recency_score': recency_score,
            'source_score': normalized_source_score,
            'final_importance': importance_score,
            'matched_reference': filter_explanation.get('matched_reference_phrase', ''),
            'is_relevant': filter_explanation.get('is_relevant', False)
        }

        logger.debug(f"Event {event.id} importance: semantic={semantic_score:.3f}, "
                    f"urgency={urgency_score:.3f}, recency={recency_score:.3f}, "
                    f"source={normalized_source_score:.3f}, final={importance_score:.3f}")

        return component_scores

    def _calculate_urgency_score(self, event: Event) -> float:
        """
        Calculate urgency score based on urgency keywords (0.0 to 1.0)

        This is ranking-specific logic (separate from basic IT relevance)
        """
        text = f"{event.title} {event.body or ''}".lower()

        max_urgency = 0.0  # Default (no urgency)

        for urgency_word, multiplier in self.urgency_multipliers.items():
            if urgency_word in text:
                # Convert multiplier to 0-1 score
                urgency_value = (multiplier - 1.0) / 0.6  # Normalize assuming max multiplier ~1.6
                max_urgency = max(max_urgency, urgency_value)

        # Ensure we stay in 0-1 range
        return min(max_urgency, 1.0)

    def calculate_recency_score(self, published_at: datetime) -> float:
        """
        Calculate how recent this event is using exponential decay

        Args:
            published_at: When the event was published

        Returns:
            float: Recency score (1.0 = just published, 0.0 = very old)
        """
        try:

            # Calculate age in hours
            now = datetime.now(timezone.utc)
            age_hours = (now - published_at).total_seconds() / 3600

            # Handle future dates (clock skew)
            if age_hours < 0:
                age_hours = 0

            # Exponential decay: score = 0.5^(age / half_life)
            recency_score = 0.5 ** (age_hours / self.recency_half_life_hours)

            logger.debug(f"Event age: {age_hours:.1f} hours, recency: {recency_score:.3f}")

            return recency_score

        except Exception as e:
            logger.error(f"Error calculating recency: {str(e)}")
            return 0.5  # Default to middle score



    def get_ranking_explanation(self, ranked_event: Dict) -> Dict:
        """
        Get detailed explanation of ranking calculation

        Args:
            ranked_event: Event dict from rank_events()

        Returns:
            Dict with detailed ranking explanation
        """
        event = ranked_event['event']

        # Calculate age for display
        now = datetime.now(timezone.utc)
        age_hours = (now - event.published_at).total_seconds() / 3600

        return {
            'event_id': event.id,
            'title': event.title[:80] + '...' if len(event.title) > 80 else event.title,
            'final_score': round(ranked_event['final_score'], 3),
            'is_relevant': ranked_event['is_relevant'],
            'components': {
                'semantic_score': round(ranked_event['semantic_score'], 3),
                'urgency_score': round(ranked_event['urgency_score'], 3),
                'recency_score': round(ranked_event['recency_score'], 3),
                'source_score': round(ranked_event['source_score'], 3),
            },
            'weights': {
                'semantic_weight': self.semantic_weight,
                'urgency_weight': self.urgency_weight,
                'recency_weight': self.recency_weight,
                'source_weight': self.source_weight,
            },
            'age_hours': round(age_hours, 1),
            'source': event.source,
            'matched_reference': ranked_event['matched_reference'],
            'formula': (f"({ranked_event['semantic_score']:.3f} * {self.semantic_weight}) + "
                       f"({ranked_event['urgency_score']:.3f} * {self.urgency_weight}) + "
                       f"({ranked_event['recency_score']:.3f} * {self.recency_weight}) + "
                       f"({ranked_event['source_score']:.3f} * {self.source_weight}) = "
                       f"{ranked_event['final_score']:.3f}")
        }

    def update_threshold(self, new_threshold: float):
        """Update the semantic similarity threshold"""
        self.semantic_filter.update_threshold(new_threshold)

    def rank_events(self, events: List[Event]) -> List[EnrichedEvent]:
        """
        Score and rank events by semantic importance (highest first)

        Args:
            events: List of Event objects to rank

        Returns:
            List of ranked events with scores and explanations
        """
        scored_events = []

        for event in events:
            # Calculate all component scores
            scores = self.calculate_semantic_importance_score(event)

            # Step 3: Create EnrichedEvent object
            enriched_event = EnrichedEvent(

                # Copy all original
                **event.model_dump(),

                # Add enrichment fields with processing results
                semantic_score = scores['semantic_score'],
                urgency_score = scores['urgency_score'],
                recency_score = scores['recency_score'],
                source_score = scores['source_score'],
                matched_reference = scores['matched_reference'],
                final_score=scores['final_importance'],
                ingested_at=datetime.now(timezone.utc),
                filter_passed=scores['is_relevant']
            )

            scored_events.append(enriched_event)

        # Sort by final score (highest first), with ID as tiebreaker for deterministic ordering
        sorted_events = sorted(
            scored_events,
            key=lambda x: (-x.final_score, x.id)
        )

        logger.info(f"Ranked {len(sorted_events)} events")


        return sorted_events



# Test the semantic ranking engine
if __name__ == "__main__":
    from datetime import timedelta

    print("Testing Semantic Ranking Engine:")
    print("=" * 60)

    # Test events
    now = datetime.now(timezone.utc)

    test_events = [
        Event(
            id="critical-recent",
            source="azure",
            title="Critical outage affecting authentication services",
            body="Authentication service down, affecting user logins",
            published_at=now - timedelta(hours=1)
        ),
        Event(
            id="security-old",
            title="Security vulnerability in database systems",
            body="Suspicious network activity flagged by monitoring systems",
            source="security",
            published_at=now - timedelta(hours=48)
        ),
        Event(
            id="maintenance-recent",
            title="Scheduled maintenance completed",
            body="Regular system maintenance has been completed successfully",
            source="devops",
            published_at=now - timedelta(minutes=30)
        ),
        Event(
            id="coffee-machine",
            title="New coffee machine installed",
            body="Espresso machine installed in kitchen area",
            source="office",
            published_at=now - timedelta(hours=2)
        )
    ]

    ranking_engine = RankingEngine(threshold=0.5)

    # Rank events
    ranked_events = ranking_engine.rank_events(test_events)

    # Show results
    print(f"\n" + "="*60)
    print("RANKING RESULTS:")
    print("="*60)

    for i, ranked_event in enumerate(ranked_events):
        explanation = ranking_engine.get_ranking_explanation(ranked_event)

        print(f"\n#{i+1} - {explanation['event_id']}")
        print(f"Title: {explanation['title']}")
        print(f"Source: {explanation['source']} | Age: {explanation['age_hours']} hours")
        print(f"IT Relevant: {'✅' if explanation['is_relevant'] else '❌'}")
        print(f"Final Score: {explanation['final_score']}")
        print(f"Components:")
        print(f"  • Semantic: {explanation['components']['semantic_score']}")
        print(f"  • Urgency:  {explanation['components']['urgency_score']}")
        print(f"  • Recency:  {explanation['components']['recency_score']}")
        print(f"  • Source:   {explanation['components']['source_score']}")
        if explanation['matched_reference']:
            print(f"Matched: '{explanation['matched_reference']}'")
        print("-" * 60)