import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any

import aiohttp
import feedparser

from src.logging_setup import configure_logging
from src.models import Event

logger = logging.getLogger(__name__)
configure_logging()

@dataclass
class SourceConfig:
    """Configuration for all sources"""
    name: str
    plugin_type: str
    config: Dict[str, Any]
    enabled: bool = True


class BaseSourcePlugin(ABC):
    """Base class for all source plugins (A Contract all plugins must follow)"""

    def __init__(self, config: SourceConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def fetch_events(self, session: aiohttp.ClientSession) -> List[Event]:
        pass


class RedditSourcePlugin(BaseSourcePlugin):
    """Plugin to fetch posts from Reddit using JSON API"""

    async def fetch_events(self, session: aiohttp.ClientSession) -> List[Event]:

        # Fetch source url
        url = self.config.config['url']

        # Make HTTP request to Reddit's JSON AP
        try:
            async with session.get(url) as resp:
                # Check if request was successful (HTTP 200)
                if resp.status != 200:
                    logger.error(f"Reddit API returned status {resp.status} for {url}")
                    return [] # Return empty list on HTTP error

                # Parse JSON response from Reddit API
                data = await resp.json()
        except Exception as e:
            logger.error(f"Error fetching Reddit data from {url}: {e}")
            return []

        # Initialize list to store converted Event objects
        events = []

        # Navigate Reddit's JSON structure to get post
        posts_data = data.get('data', {}).get('children', [])

        # Process each Reddit post
        for post_data in posts_data:

            # Extract the actual post
            post = post_data['data']

            # Create Event object from Reddit post data
            event = Event(
                id=f"reddit_{post['id']}",
                source=self.name,
                title=post.get('title', ''),

                # Use selftext (text posts) or URL (link posts) as body
                body=post.get('selftext') or post.get('url', ''),

                # Convert Reddit's UTC timestamp to Python datetime
                published_at=datetime.fromtimestamp(post['created_utc'], tz=timezone.utc)
            )
            # Add the converted event to our collection
            events.append(event)

        logger.debug(f"Fetched {len(events)} events from Reddit source: {self.name}")
        return events


class RSSSourcePlugin(BaseSourcePlugin):
    """Plugin to fetch RSS feeds"""

    async def fetch_events(self, session: aiohttp.ClientSession) -> List[Event]:
        # Extract the RSS feed URL from plugin configuration
        url = self.config.config['url']

        # Make asynchronous HTTP request to the RSS feed
        try:
            async with session.get(url) as resp:
                # Verify successful HTTP response (status 200)
                if resp.status != 200:
                    logger.error(f"RSS feed returned status {resp.status} for {url}")
                    return []
                # Get the raw XML/RSS content as text
                feed_content = await resp.text()
                feed = feedparser.parse(feed_content)

        except Exception as e:
            logger.error(f"Error fetching RSS feed from {url}: {e}")
            return []

        # Initialize collection for processed Event objects
        events = []

        # Process each RSS entry/article
        for entry in feed.entries:
            try:
                # Handle missing published_parsed
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                else:
                    # This ensures we always have a valid timestamp for sorting/filtering
                    pub_date = datetime.now(timezone.utc)

                # Create Event object from RSS entry data
                event = Event(
                    id=f"{self.name}_{hash(entry.link)}",
                    source=self.name,
                    title=entry.title,
                    body=getattr(entry, 'summary', ''),
                    published_at=pub_date
                )
                # Add the converted event to our collection
                events.append(event)

            except Exception as e:
                logger.error(f"Error parsing RSS entry: {e}")
                continue

        logger.debug(f"Fetched {len(events)} events from RSS source: {self.name}")
        return events


class PluginRegistry:

    """Registry for source plugins, allows easy extension with new data sources without
    modifying existing code"""

    def __init__(self):
        # Dictionary mapping plugin type identifiers to their implementation classes
        self._plugins = {
            'reddit': RedditSourcePlugin,
            'rss': RSSSourcePlugin
        }

    def get(self, plugin_type: str):
        """ Retrieve a plugin class by its type identifier. """
        # Look up plugin class in registry dictionary
        return self._plugins.get(plugin_type)

    def register_plugin(self, plugin_type: str, plugin_class):
        """Allow registration of custom plugins"""

        # Add or update plugin in the registry dictionary
        self._plugins[plugin_type] = plugin_class

class AggregationService:
    """
    Orchestrates data collection from multiple sources (RSS feeds, Reddit, etc.)
    """

    def __init__(self, source_configs: List[SourceConfig]):
        """ Initialize the aggregation service with configured data sources"""

        # Initialize plugin registry for managing different source types (Reddit, RSS, etc.)
        self.registry = PluginRegistry()

        # Initialize and configure source plugins based on provided configurations
        self.sources = []

        for config in source_configs:
            # Skip sources that are explicitly disabled in configuration
            if not config.enabled:
                logger.info(f"Skipping disabled source: {config.name}")
                continue

            # Look up appropriate plugin class for this source type (reddit, rss, etc.)
            plugin_class = self.registry.get(config.plugin_type)
            if plugin_class:
                # Instantiate the plugin with its specific configuration
                self.sources.append(plugin_class(config))
                logger.info(f"Initialized source: {config.name} ({config.plugin_type})")
            else:
                logger.warning(f"Unknown plugin type: {config.plugin_type} for source: {config.name}")

        logger.info(f"NewsAggregationService initialized with {len(self.sources)} active sources")

    async def fetch_all_events(self) -> List[Event]:
        """
        Concurrently fetch events from all configured and active sources.
        """
        # Early return if no sources are configured
        if not self.sources:
            logger.warning("No active sources configured")
            return []

        # Initialize collection for all fetched events
        all_events = []

        # Create shared HTTP session for all source requests
        async with aiohttp.ClientSession() as session:

            # Create concurrent fetch tasks for all active sources
            fetch_tasks = [source.fetch_events(session) for source in self.sources]
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        # Process results from each source and handle any errors
        for i, result in enumerate(results):

            # Get the source name for logging purposes
            source_name = self.sources[i].name

            # Log error but continue processing other sources
            if isinstance(result, Exception):
                logger.error(f"Error fetching from source '{source_name}': {result}")
            else:
                # Add successfully fetched events to the collection
                all_events.extend(result)
                logger.info(f"Fetched {len(result)} events from source '{source_name}'")

        logger.info(f"Total events fetched: {len(all_events)}")
        return all_events



