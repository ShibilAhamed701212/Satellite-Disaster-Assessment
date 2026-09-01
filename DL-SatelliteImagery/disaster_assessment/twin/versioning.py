"""
State versioning and history tracking for the Digital Twin.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StateVersion:
    """A versioned snapshot of twin state."""
    version: int
    timestamp: str
    state_summary: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""


class VersionTracker:
    """Track state versions for reproducibility."""

    def __init__(self):
        self._versions: List[StateVersion] = []

    def record(self, version: int, timestamp: str, state_dict: Dict) -> None:
        """Record a new version."""
        import hashlib
        checksum = hashlib.md5(str(state_dict).encode()).hexdigest()[:8]

        self._versions.append(StateVersion(
            version=version,
            timestamp=timestamp,
            state_summary=state_dict,
            checksum=checksum,
        ))

    def get_versions(self) -> List[StateVersion]:
        return list(self._versions)

    def get_latest(self) -> Optional[StateVersion]:
        return self._versions[-1] if self._versions else None

    def get_by_version(self, version: int) -> Optional[StateVersion]:
        for v in self._versions:
            if v.version == version:
                return v
        return None
