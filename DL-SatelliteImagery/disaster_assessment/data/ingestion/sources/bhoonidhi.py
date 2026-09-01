"""
Bhoonidhi (NRSC) data provider.

Status: AUTH_REQUIRED
Bhoonidhi is NRSC's geospatial data portal.
"""

from typing import List, Optional

from ..base import DataProvider, DataProduct


class BhoonidhiDataProvider(DataProvider):
    """Bhoonidhi geospatial data provider.

    Status: AUTH_REQUIRED - Requires NRSC credentials.
    """

    def get_name(self) -> str:
        return "Bhoonidhi"

    def get_status(self) -> str:
        return "AUTH_REQUIRED"

    def search(self, bbox=None, start_date=None, end_date=None, **kwargs) -> List[DataProduct]:
        return []

    def download(self, product, output_dir="cache") -> Optional[str]:
        return None

    def validate(self, file_path: str) -> bool:
        return False
