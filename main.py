#!/usr/bin/env python3
"""
Main ingestion script - extracted from streamlit app.
"""

import argparse
import asyncio
import logging
import time
import signal
import sys

from src.logging_setup import configure_logging
from src.orchestration.scheduler_service import SchedulerService
from src.orchestration.ingestion_service_factory import IngestionServiceFactory

# Set up logging
logger = logging.getLogger(__name__)
configure_logging()


def run_manual_ingestion(scheduler_service: SchedulerService):
    """Run manual ingestion with progress display."""
    logger.info("🔄 Initializing ingestion pipeline...")

    try:
        # Create a custom orchestrator to get intermediate results
        orchestrator = scheduler_service.orchestrator

        logger.info("📡 Fetching events from sources...")
        events = asyncio.run(orchestrator.fetch_events())
        fetched_count = len(events)

        logger.info(f"🔍 Filtering {fetched_count} events for relevance...")
        filter_result = orchestrator.filter_events(events)
        relevant_count = filter_result['statistics']['relevant_count']

        logger.info(f"💾 Storing {relevant_count} relevant events...")
        relevant_events = [item['event'] for item in filter_result['relevant_events']]
        storage_result = orchestrator.store_events(relevant_events)

        # Compile final result
        result = {
            'success': True,
            'fetched_count': fetched_count,
            'relevant_count': relevant_count,
            'filtered_count': filter_result['statistics']['filtered_count'],
            'stored_count': storage_result.get('stored_count', 0),
            'total_in_storage': storage_result.get('total_in_storage', 0),
            'run_type': 'manual'
        }

        logger.info(
            f" ✅ Ingestion completed ! Fetched: {fetched_count}, Relevant: {relevant_count}, Duplicates: {relevant_count-storage_result.get('stored_count', 0)},  Stored: {storage_result.get('stored_count', 0)}")

        return result

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return None


def show_status(scheduler_service: SchedulerService):
    """Show ingestion status (extracted from streamlit)."""
    status = scheduler_service.get_status()

    print("\n📊 Ingestion Status:")
    if status['is_running']:
        print("   🟢 Scheduler Running")
    else:
        print("   🔴 Scheduler Stopped")

    storage_stats = status.get('storage_stats', {})
    print(f"   Events in Storage: {storage_stats.get('total_events', 0)}")

    # Show last run results (from streamlit)
    last_result = status.get('last_run_result')
    if last_result:
        print("\n📈 Last Run Results:")
        print(f"   Fetched: {last_result.get('fetched_count', 0)}")
        print(f"   Relevant: {last_result.get('relevant_count', 0)}")
        print(f"   Stored: {last_result.get('stored_count', 0)}")
        print(f"   Run Type: {last_result.get('run_type', 'unknown').title()}")

        if not last_result.get('success', True):
            print(f"❌ Error: {last_result.get('error', 'Unknown error')}")
    else:
        print("No previous runs")


def start_scheduler(scheduler_service: SchedulerService, interval: int):
    """Start scheduler (extracted from streamlit)."""
    status = scheduler_service.get_status()

    if status['is_running']:
        print("⚠️ Scheduler is already running")
        return

    print(f"▶️ Starting scheduler with {interval} minute interval...")
    scheduler_service.start_scheduler(interval)
    print("✅ Scheduler started successfully")


def stop_scheduler(scheduler_service: SchedulerService):
    """Stop scheduler (extracted from streamlit)."""
    status = scheduler_service.get_status()

    if not status['is_running']:
        print("ℹ️ Scheduler is not running")
        return

    print("⏹️ Stopping scheduler...")
    scheduler_service.stop_scheduler()
    print("✅ Scheduler stopped successfully")


def run_daemon_mode(scheduler_service: SchedulerService, interval: int):
    """Run scheduler in daemon mode with signal handling."""

    def signal_handler(sig, frame):
        print(f"\n🛑 Received signal {sig}, shutting down...")
        scheduler_service.stop_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"🚀 Starting scheduler daemon (interval: {interval} minutes)")
    scheduler_service.start_scheduler(interval)

    print("📊 Scheduler running in background. Press Ctrl+C to stop.")
    print("🔍 Use 'python main.py status' in another terminal to check status.")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received")
    finally:
        scheduler_service.stop_scheduler()
        print("👋 Goodbye!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="IT Newsfeed Ingestion Manager")

    parser.add_argument('command',
                        choices=['manual', 'start-scheduler', 'stop-scheduler', 'daemon', 'status'],
                        help='Command to run')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Relevance threshold (default: 0.5)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Scheduler interval in minutes (default: 5)')

    args = parser.parse_args()

    # Initialize services
    shared_storage = IngestionServiceFactory.create_shared_storage()
    scheduler_service = IngestionServiceFactory.create_scheduler(threshold=0.5, vector_storage=shared_storage)

    if args.command == 'manual':
        run_manual_ingestion(scheduler_service)

    elif args.command == 'start-scheduler':
        start_scheduler(scheduler_service, args.interval)

    elif args.command == 'stop-scheduler':
        stop_scheduler(scheduler_service)

    elif args.command == 'daemon':
        run_daemon_mode(scheduler_service, args.interval)

    elif args.command == 'status':
        show_status(scheduler_service)


if __name__ == "__main__":
    #main()
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize service
    shared_storage = IngestionServiceFactory.create_shared_storage()
    scheduler_service= IngestionServiceFactory.create_scheduler(threshold=0.5,vector_storage=shared_storage)
    # Test manual ingestion
    #print("Testing manual ingestion...")
    #result = run_manual_ingestion(scheduler_service)

    # Test status
    #print("\nTesting status...")
    #show_status(scheduler_service)


    # Test scheduled ingestion
    print("🧪 Testing scheduled ingestion...")
    print("⏰ Starting scheduler with 1-minute intervals for testing...")

    try:
        # Start the scheduler
        start_scheduler(scheduler_service, 1)  # 1 minute intervals for testing

        print("📊 Scheduler is now running. Waiting for jobs to execute...")
        print("🔍 You should see ingestion logs every minute.")
        print("⌨️  Press Ctrl+C to stop the test\n")

        # Keep the program alive to let scheduled jobs run
        cycle_count = 0
        while True:
            time.sleep(10)  # Check every 10 seconds
            cycle_count += 1

            # Show status every 30 seconds
            if cycle_count % 3 == 0:
                print(f"\n📈 Status check (running for {cycle_count * 10} seconds):")
                show_status(scheduler_service)
                print("─" * 50)

    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        print("⏹️  Stopping scheduler...")
        stop_scheduler(scheduler_service)
        print("✅ Scheduler stopped. Test complete!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("⏹️  Stopping scheduler...")
        stop_scheduler(scheduler_service)



