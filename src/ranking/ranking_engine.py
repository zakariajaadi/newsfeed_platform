import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

from src.models import Event, EnrichedEvent
from src.logging_setup import configure_logging

# Set logger
logger = logging.getLogger(__name__)
configure_logging()


class RankingEngine:
    """
    Calculates ranking scores by combining pre-computed semantic scores with recency
    """

    def __init__(self):
        """Initialize ranking engine"""

        # Importance calculation weights (combine importance factors)
        self.importance_weights = {
            'semantic': 0.6,  # 60% - Core IT relevance from semantic similarity
            'urgency': 0.3,  # 30% - Urgency keywords boost
            'source': 0.1  # 10% - Source credibility
        }

        # Final score weights (balance importance x recency)
        self.final_weights = {
            'importance': 0.7,  # 70% - What matters (importance)
            'recency': 0.3  # 30% - Freshness boost (recency)
        }

        # Additional importance boosters for ranking (beyond basic IT relevance)
        self.urgency_multipliers = {
            'emergency': 1.6,
            'critical': 1.5,
            'urgent': 1.4,
            'severe': 1.4,
            'breaking': 1.3,
            'immediate': 1.3,
            'major': 1.2,
            'alert': 1.2,
        }

        # Source credibility for ranking
        self.source_ranking_weights = {
            'azure-health': 1.8,  # Official status pages get boost
            'aws-status': 1.8,
            'azure': 1.7,
            'cve': 1.7,
            'security': 1.6,  # Security advisories
            'cert': 1.6,
            'monitoring': 1.4,
            'devops': 1.3,
            'ars-technica': 1.2,  # Tech journalism
            'reddit': 1.0,  # Community reports (baseline)
            'github': 1.1,
            'office': 0.8,  # Less critical for IT ops
            'hr': 0.7,
        }

        # RECENCY DECAY - How quickly old news becomes less important
        self.recency_half_life_hours = 12  # Score halves every 12 hours (faster decay for IT)

    def calculate_ranking_score(self, event: Event, filter_explanation: Dict) -> Dict[str, float]:
        """
        Calculate importance using PRE-COMPUTED semantic scores + ranking-specific boosts

        Args:
            event: Event to score
            filter_explanation: Pre-computed semantic analysis results dict containing:
                - 'max_similarity_score': float
                - 'matched_reference_phrase': str (optional)
                - 'is_relevant': bool (optional)
                - Any other fields from semantic filter

        Returns:
            Dict with component scores and final importance
        """
        # Step 1 - calculate importance score
        importance_components = self.calculate_importance_score(event, filter_explanation)
        importance_score = importance_components['importance_score']

        # Step 2 - calculate recency score

        recency_score = self.calculate_recency_score(event.published_at)

        # Step 3 - final score balancing importance x recency

        final_score = (
                importance_score * self.final_weights['importance'] +
                recency_score * self.final_weights['recency']
        )

        component_scores = {
            'semantic_score': importance_components['semantic_score'],
            'urgency_score': importance_components['urgency_score'],
            'recency_score': recency_score,
            'source_score': importance_components['source_score'],
            'importance_score': importance_components['importance_score'],
            'final_score': final_score,
            'matched_reference': filter_explanation['matched_reference_phrase'],
            'is_relevant': filter_explanation['is_relevant']
        }

        logger.debug(f"Balanced score: importance({importance_score:.3f}) × {self.final_weights['importance']} + "
                     f"recency({recency_score:.3f}) × {self.final_weights['recency']} = {final_score:.3f}")

        return component_scores

    def calculate_importance_score(self, event: Event, filter_explanation: Dict) -> Dict[str, float]:
        """
        Calculate importance using PRE-COMPUTED semantic scores + ranking-specific boosts

        Args:
            event: Event to score
            filter_explanation: Pre-computed semantic analysis results dict containing:
                - 'max_similarity_score': float
                - 'matched_reference_phrase': str (optional)
                - 'is_relevant': bool (optional)
                - Any other fields from semantic filter

        Returns:
            Dict with component scores and final importance
        """
        # --- Stage 1 : calculate importance score --- #

        # Factor 1: Semantic similarity (core IT relevance)
        semantic_score = filter_explanation.get('max_similarity_score', 0.0)
        matched_reference = filter_explanation.get('matched_reference_phrase', '')
        is_relevant = filter_explanation.get('is_relevant', True)

        # Factor 2: Urgency keywords (criticality boost)
        urgency_score = self._calculate_urgency_score(event)

        # Factor 3: Source credibility
        source_score = self.source_ranking_weights.get(event.source.lower(), 1.0)

        # Combine importance factors with weights
        importance_score = (
                semantic_score * self.importance_weights['semantic'] +
                urgency_score * self.importance_weights['urgency'] +
                source_score * self.importance_weights['source']
        )

        component_scores = {
            'semantic_score': semantic_score,
            'urgency_score': urgency_score,
            'source_score': source_score,
            'importance_score': importance_score
        }

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

    def get_ranking_explanation(self, enriched_event: EnrichedEvent) -> Dict:
        """
        Get detailed explanation of two-stage ranking calculation.

        Args:
            enriched_event: EnrichedEvent object with computed scores

        Returns:
            Dict with detailed ranking explanation showing importance + recency balance
        """
        # Calculate age for display
        now = datetime.now(timezone.utc)
        age_hours = (now - enriched_event.published_at).total_seconds() / 3600

        # Calculate importance score from components (for explanation)
        importance_score = (
                enriched_event.semantic_score * self.importance_weights['semantic'] +
                enriched_event.urgency_score * self.importance_weights['urgency'] +
                enriched_event.source_score * self.importance_weights['source']
        )

        return {
            'event_id': enriched_event.id,
            'title': enriched_event.title[:80] + '...' if len(enriched_event.title) > 80 else enriched_event.title,
            'final_score': round(enriched_event.final_score, 3),
            'is_relevant': enriched_event.filter_passed,
            'stages': {
                'importance_score': round(importance_score, 3),
                'recency_score': round(enriched_event.recency_score, 3),
            },
            'importance_components': {
                'semantic_score': round(enriched_event.semantic_score, 3),
                'urgency_score': round(enriched_event.urgency_score, 3),
                'source_score': round(enriched_event.source_score, 3),
            },
            'weights': {
                'importance_weight': self.final_weights['importance'],
                'recency_weight': self.final_weights['recency'],
                'importance_breakdown': self.importance_weights,
            },
            'age_hours': round(age_hours, 1),
            'source': enriched_event.source,
            'matched_reference': enriched_event.matched_reference,
            'formula': (
                f"Importance({importance_score:.3f}) × {self.final_weights['importance']} + "
                f"Recency({enriched_event.recency_score:.3f}) × {self.final_weights['recency']} = "
                f"{enriched_event.final_score:.3f}"),
            'importance_formula': (
                f"Semantic({enriched_event.semantic_score:.3f}) × {self.importance_weights['semantic']} + "
                f"Urgency({enriched_event.urgency_score:.3f}) × {self.importance_weights['urgency']} + "
                f"Source({enriched_event.source_score:.3f}) × {self.importance_weights['source']} = "
                f"{importance_score:.3f}")
        }

    def rank_events(self, events: List[Event], filter_explanations: List[Dict]) -> List[EnrichedEvent]:
        """
        Rank events using pre-computed semantic filter explanations (NO SEMANTIC FILTER CALLS)

        Args:
            events: List of Event objects to rank
            filter_explanations: List of filter explanation dicts from semantic filtering stage
                                Each dict should contain semantic analysis results from
                                SemanticContentFilter.get_filter_explanation()

        Returns:
            List of ranked EnrichedEvent objects
        """
        if len(events) != len(filter_explanations):
            raise ValueError(
                f"Events count ({len(events)}) must match filter_explanations count ({len(filter_explanations)})")

        enriched_events = []

        for event, filter_explanation in zip(events, filter_explanations):
            # Calculate importance using pre-computed filter explanation
            scores = self.calculate_ranking_score(event, filter_explanation)

            # Create EnrichedEvent object
            enriched_event = EnrichedEvent(
                # Copy all original event data
                **event.model_dump(),

                # Add enrichment fields with computed scores
                semantic_score=scores['semantic_score'],
                urgency_score=scores['urgency_score'],
                recency_score=scores['recency_score'],
                source_score=scores['source_score'],
                matched_reference=scores['matched_reference'],
                final_score=scores['final_score'],
                ingested_at=datetime.now(timezone.utc),
                filter_passed=scores['is_relevant']
            )

            enriched_events.append(enriched_event)

        # Sort by final score (highest first), with ID as tiebreaker for deterministic ordering
        sorted_events = sorted(
            enriched_events,
            key=lambda x: (-x.final_score, x.id)
        )

        logger.info(f"Ranked {len(sorted_events)} events")

        return sorted_events


# Example usage showing how to integrate with semantic filtering
if __name__ == "__main__":
    from datetime import timedelta
    from src.filtering.semantic_filtering_engine import SemanticContentFilter

    print("Testing Ranking Engine")
    print("=" * 70)

    # Test Events
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

    # STEP 1: Run semantic filtering
    print("Step 1: Computing semantic filter explanations (normally done in filtering stage)...")
    semantic_filter = SemanticContentFilter(threshold=0.4)

    filter_explanations = []
    for event in test_events:
        filter_explanation = semantic_filter.get_filter_explanation(event)
        filter_explanations.append(filter_explanation)

    # STEP 2: Run ranking using pre-computed filter explanations
    print("Step 2: Ranking events using pre-computed filter explanations...")
    ranking_engine = RankingEngine()  # NO semantic filter dependency!

    ranked_events = ranking_engine.rank_events(test_events, filter_explanations)

    # Show results
    print(f"\n" + "=" * 70)
    print("RANKING RESULTS:")
    print("=" * 70)

    for i, enriched_event in enumerate(ranked_events):
        explanation = ranking_engine.get_ranking_explanation(enriched_event)

        print(f"\n#{i + 1} - {explanation['event_id']}")
        print(f"Title: {explanation['title']}")
        print(f"Source: {explanation['source']} | Age: {explanation['age_hours']} hours")
        print(f"IT Relevant: {'✅' if explanation['is_relevant'] else '❌'}")
        print(f"Final Score: {explanation['final_score']}")
        print(f"Stages:")
        print(f"  • Importance: {explanation['stages']['importance_score']}")
        print(f"  • Recency:    {explanation['stages']['recency_score']}")
        print(f"Importance Components:")
        print(f"  • Semantic: {explanation['importance_components']['semantic_score']} (FROM FILTER EXPLANATION)")
        print(f"  • Urgency:  {explanation['importance_components']['urgency_score']}")
        print(f"  • Source:   {explanation['importance_components']['source_score']}")
        if explanation['matched_reference']:
            print(f"Matched: '{explanation['matched_reference']}'")
        print(f"Formula: {explanation['formula']}")
        print("-" * 70)