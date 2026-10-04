"""
Ingestion scheduler for periodic data updates.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class ScheduledTask:
    """A scheduled ingestion task."""
    name: str
    provider_name: str
    interval_hours: float = 24.0
    last_run: Optional[str] = None
    status: str = "PENDING"


class IngestionScheduler:
    """Schedule and manage periodic data ingestion tasks.

    Status: IMPLEMENTED ARCHITECTURE
    Requires configured data providers for actual execution.
    """

    def __init__(self):
        self._tasks: Dict[str, ScheduledTask] = {}
        self._providers: Dict[str, object] = {}

    def register_provider(self, name: str, provider: object):
        """Register a data provider."""
        self._providers[name] = provider

    def add_task(self, name: str, provider_name: str, interval_hours: float = 24.0):
        """Add a scheduled ingestion task."""
        self._tasks[name] = ScheduledTask(
            name=name,
            provider_name=provider_name,
            interval_hours=interval_hours,
        )

    def get_status(self) -> Dict[str, str]:
        """Get status of all tasks."""
        return {
            name: {
                "provider": task.provider_name,
                "interval_hours": task.interval_hours,
                "last_run": task.last_run,
                "status": task.status,
            }
            for name, task in self._tasks.items()
        }

    def run_task(self, name: str, output_dir: str = "cache") -> Optional[str]:
        """Run a specific ingestion task.

        Returns file path on success, None on failure.
        """
        task = self._tasks.get(name)
        if task is None:
            return None

        provider = self._providers.get(task.provider_name)
        if provider is None:
            task.status = "PROVIDER_MISSING"
            return None

        try:
            products = provider.search()
            if products:
                result = provider.download(products[0], output_dir)
                task.last_run = datetime.now().isoformat()
                task.status = "COMPLETED"
                return result
        except Exception as e:
            task.status = f"FAILED: {str(e)[:100]}"

        return None
