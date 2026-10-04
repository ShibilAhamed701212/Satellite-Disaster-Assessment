"""
Base data provider interface for satellite imagery ingestion.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class DataProduct:
    """A downloaded/available data product."""
    product_id: str
    source: str
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "AVAILABLE"


class DataProvider(ABC):
    """Abstract base class for data providers.

    All data sources (Sentinel, weather, MOSDAC, etc.)
    implement this interface.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return provider name."""
        ...

    @abstractmethod
    def get_status(self) -> str:
        """Return operational status.

        Returns one of:
            READY, NOT_CONFIGURED, AUTH_REQUIRED, UNAVAILABLE
        """
        ...

    @abstractmethod
    def search(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs,
    ) -> List[DataProduct]:
        """Search for available data products."""
        ...

    @abstractmethod
    def download(
        self,
        product: DataProduct,
        output_dir: str = "cache",
    ) -> Optional[str]:
        """Download a data product. Returns file path."""
        ...

    @abstractmethod
    def validate(self, file_path: str) -> bool:
        """Validate a downloaded file."""
        ...
