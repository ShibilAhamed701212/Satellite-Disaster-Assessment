"""
Weather data provider.

Status: NOT_CONFIGURED
Requires weather API key.
"""

from typing import List, Optional

from ..base import DataProvider, DataProduct


class WeatherDataProvider(DataProvider):
    """Weather data provider.

    Status: NOT_CONFIGURED - Requires API key.
    """

    def get_name(self) -> str:
        return "Weather"

    def get_status(self) -> str:
        return "NOT_CONFIGURED"

    def search(self, bbox=None, start_date=None, end_date=None, **kwargs) -> List[DataProduct]:
        return []

    def download(self, product, output_dir="cache") -> Optional[str]:
        return None

    def validate(self, file_path: str) -> bool:
        return False
