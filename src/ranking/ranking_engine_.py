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
    Calculates ranking scores using clean two-stage approach: Importance + Recency

    Stage 1: Calculate pure importance score (how critical/relevant the event is)
    Stage 2: Calculate pure recency score (how fresh the event is)
    Stage 3: Balance importance and recency as specified in assessment

    This approach provides clear separation of concerns and better alignment
    with the assessment's "balancing importance and recency" requirement.
    """

    def __init__(self, threshold: float = 0.5):
        """Initialize ranking engine with semantic content filter"""

        # Reuse the semantic content filter for text analysis
        self.semantic_filter = SemanticContentFilter(threshold=threshold)

        # Importance calculation weights (how to combine importance factors)
        self.importance_weights = {
            'semantic': 0.6,  # 60% - Core IT relevance from semantic similarity
            'urgency': 0.3,  # 30% - Urgency keywords boost
            'source': 0.1  # 10% - Source credibility
        }

        # Final score weights (how to balance importance vs recency)
        self.final_weights = {
            'importance': 0.7,  # 70% - What matters (importance)
            'recency': 0.3  # 30% - Freshness boost (recency)
        }

        # Urgency multipliers for importance calculation
        self.urgency_multipliers = {
            'critical': 1.5,
            'urgent': 1.4,
            'emergency': 1.6,
            'breaking': 1.3,
            'major': 1.2,
            'severe': 1.4,
            'immediate': 1.3,
            'alert': 1.2,
            'outage': 1.4,
            'down': 1.3,
            'breach': 1.5,
            'vulnerability': 1.3,
        }

        # Source credibility for importance calculation
        self.source_credibility = {
            'azure-health': 1.8,  # Official status pages
            'aws-status': 1.8,
            'azure-status': 1.8,
            'krebs-security': 1.6,  # Respected security sources
            'ars-security': 1.5,
            'cert': 1.7,
            'cve': 1.7,
            'security': 1.4,
            'r/sysadmin': 1.2,  # Community reports
            'r/outages': 1.3,
            'r/cybersecurity': 1.2,
            'reddit': 1.0,  # Baseline
            'github': 1.1,
            'default': 1.0
        }

        # Recency decay configuration
        self.recency_half_life_hours = 12  # Score halves every 12 hours

        logger.info("RankingEngine initialized with two-stage importance + recency approach")

    def calculate_importance_score(self, event: Event) -> float:
        """
        Calculate pure importance score - how critical/relevant this event is.

        Combines three factors:
        - Semantic similarity to IT reference phrases (60%)
        - Urgency keywords boost (30%)
        - Source credibility (10%)

        Args:
            event: Event to evaluate

        Returns:
            float: Importance score (0.0 to 1.0+, can exceed 1.0 for very critical events)
        """
        # Factor 1: Semantic similarity (core IT relevance)
        filter_explanation = self.semantic_filter.get_filter_explanation(event)
        semantic_score = filter_explanation['max_similarity_score']

        # Factor 2: Urgency keywords (criticality boost)
        urgency_score = self._calculate_urgency_score(event)

        # Factor 3: Source credibility
        source_score = self._calculate_source_score(event)

        # Combine importance factors with weights
        importance = (
                semantic_score * self.importance_weights['semantic'] +
                urgency_score * self.importance_weights['urgency'] +
                source_score * self.importance_weights['source']
        )

        logger.debug(f"Event {event.id} importance: semantic={semantic_score:.3f}, "
                     f"urgency={urgency_score:.3f}, source={source_score:.3f}, "
                     f"combined_importance={importance:.3f}")

        return importance

    def calculate_recency_score(self, published_at: datetime) -> float:
        """
        Calculate pure recency score - how fresh this event is.

        Uses exponential decay: recent events score close to 1.0,
        older events decay toward 0.0 based on half-life.

        Args:
            published_at: When the event was published

        Returns:
            float: Recency score (1.0 = just published, approaches 0.0 = very old)
        """
        if not published_at:
            return 0.1  # Low score for events without timestamps

        try:
            # Calculate age in hours
            now = datetime.now(timezone.utc)

            # Handle timezone-naive timestamps
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)

            age_hours = (now - published_at).total_seconds() / 3600

            # Handle future dates (clock skew)
            if age_hours < 0:
                age_hours = 0

            # Exponential decay: score = 0.5^(age / half_life)
            recency_score = 0.5 ** (age_hours / self.recency_half_life_hours)

            # Ensure minimum score for very old but potentially relevant events
            recency_score = max(recency_score, 0.01)

            logger.debug(f"Event age: {age_hours:.1f} hours, recency: {recency_score:.3f}")

            return recency_score

        except Exception as e:
            logger.error(f"Error calculating recency: {str(e)}")
            return 0.1  # Default to low score

    def balance_importance_and_recency(self, importance: float, recency: float) -> float:
        """
        Balance importance and recency as specified in assessment.

        Implements "balancing importance and recency" by weighting both factors.
        Importance gets higher weight since it's more critical for IT operations.

        Args:
            importance: Importance score (0.0 to 1.0+)
            recency: Recency score (0.0 to 1.0)

        Returns:
            float: Final balanced score
        """
        final_score = (
                importance * self.final_weights['importance'] +
                recency * self.final_weights['recency']
        )

        logger.debug(f"Balanced score: importance({importance:.3f}) × {self.final_weights['importance']} + "
                     f"recency({recency:.3f}) × {self.final_weights['recency']} = {final_score:.3f}")

        return final_score

    def calculate_semantic_importance_score(self, event: Event) -> Dict[str, float]:
        """
        Calculate final ranking score using clean two-stage approach.

        Stage 1: Calculate pure importance (how critical/relevant)
        Stage 2: Calculate pure recency (how fresh)
        Stage 3: Balance importance and recency

        Args:
            event: Event to score

        Returns:
            Dict with component scores and final ranking score
        """
        # Stage 1: Calculate pure importance score
        importance_score = self.calculate_importance_score(event)

        # Stage 2: Calculate pure recency score
        recency_score = self.calculate_recency_score(event.published_at)

        # Stage 3: Balance importance and recency
        final_score = self.balance_importance_and_recency(importance_score, recency_score)

        # Get additional metadata for transparency
        filter_explanation = self.semantic_filter.get_filter_explanation(event)

        component_scores = {
            'semantic_score': filter_explanation['max_similarity_score'],
            'urgency_score': self._calculate_urgency_score(event),
            'recency_score': recency_score,
            'source_score': self._calculate_source_score(event),
            'importance_score': importance_score,  # Pure importance
            'final_importance': final_score,  # Final balanced score
            'matched_reference': filter_explanation.get('matched_reference_phrase', ''),
            'is_relevant': filter_explanation.get('is_relevant', False)
        }

        logger.debug(f"Event {event.id} final ranking: importance={importance_score:.3f}, "
                     f"recency={recency_score:.3f}, final={final_score:.3f}")

        return component_scores


    def get_ranking_explanation(self, ranked_event: Dict) -> Dict:
        """
        Get detailed explanation of two-stage ranking calculation.

        Args:
            ranked_event: Event dict from rank_events()

        Returns:
            Dict with detailed ranking explanation showing importance + recency balance
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
            'stages': {
                'importance_score': round(ranked_event.get('importance_score', 0), 3),
                'recency_score': round(ranked_event['recency_score'], 3),
            },
            'importance_components': {
                'semantic_score': round(ranked_event['semantic_score'], 3),
                'urgency_score': round(ranked_event['urgency_score'], 3),
                'source_score': round(ranked_event['source_score'], 3),
            },
            'weights': {
                'importance_weight': self.final_weights['importance'],
                'recency_weight': self.final_weights['recency'],
                'importance_breakdown': self.importance_weights,
            },
            'age_hours': round(age_hours, 1),
            'source': event.source,
            'matched_reference': ranked_event['matched_reference'],
            'formula': (
                f"Importance({ranked_event.get('importance_score', 0):.3f}) × {self.final_weights['importance']} + "
                f"Recency({ranked_event['recency_score']:.3f}) × {self.final_weights['recency']} = "
                f"{ranked_event['final_score']:.3f}"),
            'importance_formula': (
                f"Semantic({ranked_event['semantic_score']:.3f}) × {self.importance_weights['semantic']} + "
                f"Urgency({ranked_event['urgency_score']:.3f}) × {self.importance_weights['urgency']} + "
                f"Source({ranked_event['source_score']:.3f}) × {self.importance_weights['source']} = "
                f"{ranked_event.get('importance_score', 0):.3f}")
        }


    def rank_events(self, events: List[Event]) -> List[EnrichedEvent]:
        """
        Score and rank events using two-stage importance + recency approach.

        Args:
            events: List of Event objects to rank

        Returns:
            List of EnrichedEvent objects sorted by final_score (highest first)
        """
        scored_events = []

        for event in events:
            # Calculate scores using clean two-stage approach
            scores = self.calculate_semantic_importance_score(event)

            # Create EnrichedEvent object with all component scores
            enriched_event = EnrichedEvent(
                # Copy all original event fields
                **event.model_dump(),

                # Add component scores for transparency
                semantic_score=scores['semantic_score'],
                urgency_score=scores['urgency_score'],
                recency_score=scores['recency_score'],
                source_score=scores['source_score'],
                matched_reference=scores['matched_reference'],

                # Final balanced score (importance + recency)
                final_score=scores['final_importance'],
                filter_passed=scores['is_relevant'],
                ingested_at=datetime.now(timezone.utc)
            )

            scored_events.append(enriched_event)

        # Sort by final score (highest first), with ID as tiebreaker for deterministic ordering
        sorted_events = sorted(
            scored_events,
            key=lambda x: (-x.final_score, x.id)
        )

        logger.info(f"Ranked {len(sorted_events)} events using importance + recency balance")

        return sorted_events

    def update_threshold(self, new_threshold: float):
        """Update the semantic similarity threshold"""
        self.semantic_filter.update_threshold(new_threshold)

    # --- Helper functions --- #

    def _calculate_urgency_score(self, event: Event) -> float:
        """
        Calculate urgency score based on urgency keywords (0.0 to 1.0).

        This contributes to importance calculation, not recency.
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

    def _calculate_source_score(self, event: Event) -> float:
        """
        Calculate source credibility score (0.0 to 1.0).

        This contributes to importance calculation based on source trustworthiness.
        """
        source_weight = self.source_credibility.get(event.source.lower(),
                                                    self.source_credibility['default'])

        # Normalize to 0-1 range (assuming max credibility ~1.8)
        normalized_score = min(source_weight / 1.8, 1.0)

        return normalized_score


# Test the two-stage ranking engine
if __name__ == "__main__":
    from datetime import timedelta

    print("Testing Two-Stage Ranking Engine (Importance + Recency):")
    print("=" * 65)

    # Test events with different importance and recency characteristics
    now = datetime.now(timezone.utc)

    test_events = [
        Event(
            id="critical-recent",
            source="azure-health",
            title="Critical authentication service emergency outage",
            body="Emergency: Authentication service completely down, immediate action required",
            published_at=now - timedelta(hours=1)  # Recent + Very Important
        ),
        Event(
            id="security-old",
            source="krebs-security",
            title="Major security vulnerability discovered in database systems",
            body="Critical security vulnerability affects millions of installations worldwide",
            published_at=now - timedelta(hours=48)  # Old but Very Important
        ),
        Event(
            id="maintenance-recent",
            source="r/sysadmin",
            title="Scheduled system maintenance completed successfully",
            body="Regular maintenance window completed without issues",
            published_at=now - timedelta(minutes=30)  # Recent but Lower Importance
        ),
        Event(
            id="coffee-machine",
            source="reddit",
            title="New coffee machine installed in office kitchen",
            body="Espresso machine now available for all staff members",
            published_at=now - timedelta(hours=2)  # Recent but Not IT Relevant
        )
    ]

    ranking_engine = RankingEngine(threshold=0.5)

    print("\nProcessing events through two-stage ranking...")
    ranked_events = ranking_engine.rank_events(test_events)

    print(f"\n" + "=" * 65)
    print("RANKING RESULTS (Importance + Recency Balance):")
    print("=" * 65)

    for i, event in enumerate(ranked_events):
        # Convert enriched event back to dict format for explanation
        event_dict = {
            'event': event,
            'final_score': event.final_score,
            'semantic_score': event.semantic_score,
            'urgency_score': event.urgency_score,
            'recency_score': event.recency_score,
            'source_score': event.source_score,
            'matched_reference': event.matched_reference,
            'is_relevant': event.filter_passed,
            'importance_score': event.urgency_score  # For explanation compatibility
        }

        explanation = ranking_engine.get_ranking_explanation(event_dict)

        print(f"\n#{i + 1} - {explanation['event_id']}")
        print(f"Title: {explanation['title']}")
        print(f"Source: {explanation['source']} | Age: {explanation['age_hours']} hours")
        print(f"IT Relevant: {'✅' if explanation['is_relevant'] else '❌'}")
        print(f"Final Score: {explanation['final_score']}")
        print(f"\nTwo-Stage Breakdown:")
        print(f"  Stage 1 - Importance: {explanation['stages']['importance_score']}")
        print(f"    • Semantic: {explanation['importance_components']['semantic_score']}")
        print(f"    • Urgency:  {explanation['importance_components']['urgency_score']}")
        print(f"    • Source:   {explanation['importance_components']['source_score']}")
        print(f"  Stage 2 - Recency: {explanation['stages']['recency_score']}")
        print(f"  Balance Formula: {explanation['formula']}")
        if explanation['matched_reference']:
            print(f"  Matched Reference: '{explanation['matched_reference']}'")
        print("-" * 65)