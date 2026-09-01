"""
Sentinel-2 and Sentinel-1 data provider.

Status: AUTH_REQUIRED
Requires Copernicus Open Access Hub credentials.
"""

from typing import List, Optional

from ..base import DataProvider, DataProduct


class SentinelDataProvider(DataProvider):
    """Sentinel satellite data provider.

    Status: AUTH_REQUIRED - Requires Copernicus credentials.
    """

    def __init__(self, username: str = "", password: str = ""):
        self.username = username
        self.password = password

    def get_name(self) -> str:
        return "Sentinel"

    def get_status(self) -> str:
        if not self.username or not self.password:
            return "AUTH_REQUIRED"
        return "NOT_CONFIGURED"

    def search(self, bbox=None, start_date=None, end_date=None, **kwargs) -> List[DataProduct]:
        status = self.get_status()
        if status != "READY":
            return []
        # Would use sentinelsat or Copernicus Data Space
        return []

    def download(self, product, output_dir="cache") -> Optional[str]:
        return None

    def validate(self, file_path: str) -> bool:
        return False
