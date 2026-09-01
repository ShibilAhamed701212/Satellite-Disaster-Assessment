"""
Real-time and near-real-time data ingestion architecture.
"""

from .base import DataProvider
from .cache import DataCache
from .scheduler import IngestionScheduler

__all__ = ["DataProvider", "DataCache", "IngestionScheduler"]
