"""
MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) provider.

Status: NOT_CONFIGURED
MOSDAC is ISRO's satellite data portal.
"""

from typing import List, Optional

from ..base import DataProvider, DataProduct


class MOSDACDataProvider(DataProvider):
    """MOSDAC data provider for ISRO satellite data.

    Status: NOT_CONFIGURED - API access not implemented.
    """

    def get_name(self) -> str:
        return "MOSDAC"

    def get_status(self) -> str:
        return "NOT_CONFIGURED"

    def search(self, bbox=None, start_date=None, end_date=None, **kwargs) -> List[DataProduct]:
        return []

    def download(self, product, output_dir="cache") -> Optional[str]:
        return None

    def validate(self, file_path: str) -> bool:
        return False
