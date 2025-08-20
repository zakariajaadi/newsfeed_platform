import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict

from omegaconf import OmegaConf

from src.aggregation.data_aggregation_service import SourceConfig
from src.logging_setup import configure_logging

# Set logger
logger = logging.getLogger(__name__)
configure_logging()





@dataclass
class VectorStorageConfig:
    index_file_name: str
    metadata_file_name: str
    embedding_dim: int
    autosave_every: int

@dataclass
class SchedulerConfig:
    interval: int


@dataclass
class FilteringConfig:
    threshold: float
    it_reference_phrases: List[str]


@dataclass
class ContentScoringConfig:
    semantic_weight: float
    urgency_weight: float
    source_weight: float


@dataclass
class FinalScoringConfig:
    importance_weight: float
    recency_weight: float


@dataclass
class RankingConfig:
    content_scoring: ContentScoringConfig
    final_scoring: FinalScoringConfig
    urgency_multipliers: Dict[str, float]
    source_ranking_weights: Dict[str, float]
    recency_half_life_hours: int

    @property
    def content_scoring_dict(self) -> Dict[str, float]:
        return asdict(self.content_scoring)

    @property
    def final_scoring_dict(self) -> Dict[str, float]:
        return asdict(self.final_scoring)

@dataclass
class SourceEntryConfig:
    name: str
    url: str


@dataclass
class SourceEntriesConfig:
    reddit: List[SourceEntryConfig]
    rss: List[SourceEntryConfig]


@dataclass
class FetchingConfig:
    sources: SourceEntriesConfig

    @property
    def all_sources(self) -> List[SourceConfig]:
        """Get all sources as a flat list with type information"""
        sources = []

        # Add Reddit sources with type
        for reddit_source in self.sources.reddit:
            sources.append(SourceConfig(
                name=reddit_source.name,
                plugin_type="reddit",
                config={"url": reddit_source.url}
            ))

        # Add RSS sources with type
        for rss_source in self.sources.rss:
            sources.append(SourceConfig(
                name=rss_source.name,
                plugin_type="rss",
                config={"url": rss_source.url}
            ))

        return sources




@dataclass
class AppConfig:
    vector_storage: VectorStorageConfig
    scheduler: SchedulerConfig
    filtering: FilteringConfig
    ranking: RankingConfig
    fetching: FetchingConfig


def load_config(config_file_name:str="config.yaml"):
    """Load configuration using OmegaConf"""
    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "config" / config_file_name

    # Load config with OmegaConf
    cfg = OmegaConf.load(config_path)

    # Parse nested configurations
    vector_storage = VectorStorageConfig(**cfg.vector_storage)
    scheduler = SchedulerConfig(**cfg.scheduler)
    filtering = FilteringConfig(**cfg.filtering)

    # Parse ranking with nested scoring configs
    content_scoring = ContentScoringConfig(**cfg.ranking.content_scoring)
    final_scoring = FinalScoringConfig(**cfg.ranking.final_scoring)
    ranking = RankingConfig(
        content_scoring=content_scoring,
        final_scoring=final_scoring,
        urgency_multipliers=dict(cfg.ranking.urgency_multipliers),
        source_ranking_weights=dict(cfg.ranking.source_ranking_weights),
        recency_half_life_hours=cfg.ranking.recency_half_life_hours
    )

    # Parse fetching with nested sources
    reddit_sources = [SourceEntryConfig(**source) for source in cfg.fetching.sources.reddit]
    rss_sources = [SourceEntryConfig(**source) for source in cfg.fetching.sources.rss]
    sources = SourceEntriesConfig(reddit=reddit_sources, rss=rss_sources)
    fetching = FetchingConfig(sources=sources)

    # Create main app config
    app_config = AppConfig(
        vector_storage=vector_storage,
        scheduler=scheduler,
        filtering=filtering,
        ranking=ranking,
        fetching=fetching
    )

    return app_config


# Global config instance - load once and reuse
config: AppConfig = None


def get_config() -> AppConfig:
    """Get the global config instance, loading it if necessary"""
    global config
    if config is None:
        config = load_config()
        logger.info(f"Config loaded !")
    return config


# Usage examples:
if __name__ == "__main__":
    # Set environment mode (optional, defaults to "dev")
    # os.environ["ENV_MODE"] = "prod"

    # Load config
    app_config = get_config()

    # Access config values
    print(f"Scheduler interval: {app_config.scheduler.interval}")
    print(f"Filtering threshold: {app_config.filtering.threshold}")
    print(f"Semantic weight: {app_config.ranking.content_scoring.semantic_weight}")
    print(f"Content ranking wieghts dict: {app_config.ranking.content_scoring_dict}")

    # Access sources
    for source in app_config.fetching.sources.reddit:
        print(f"Reddit source: {source.name} -> {source.url}")

    for source in app_config.fetching.sources.rss:
        print(f"RSS source: {source.name} -> {source.url}")