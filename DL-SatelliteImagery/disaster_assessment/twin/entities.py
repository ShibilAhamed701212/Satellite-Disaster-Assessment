"""
Entity definitions for the Digital Twin state engine.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DisasterEvent:
    """A disaster event tracked by the twin."""
    event_id: str
    event_type: str  # flood, earthquake, wildfire, cyclone, etc.
    location: Optional[str] = None
    start_time: str = ""
    end_time: Optional[str] = None
    severity: float = 0.0
    status: str = "ACTIVE"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HazardLayer:
    """A hazard analysis layer."""
    layer_id: str
    hazard_type: str  # flood, damage, change, fire
    timestamp: str = ""
    data: Any = None  # numpy array or path
    crs: Optional[str] = None
    resolution: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FloodRegion:
    """A detected flood region."""
    region_id: str
    area_m2: Optional[float] = None
    area_km2: Optional[float] = None
    mean_depth: Optional[float] = None
    max_depth: Optional[float] = None
    timestamp: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DamagedBuildingRegion:
    """A detected building damage region."""
    region_id: str
    damage_level: str = "unknown"  # undamaged, minor, major, destroyed
    building_count: int = 0
    area_m2: Optional[float] = None
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AffectedArea:
    """An affected area with combined hazard assessment."""
    area_id: str
    total_area_km2: float = 0.0
    severity_level: str = "UNKNOWN"
    severity_score: float = 0.0
    hazards: List[str] = field(default_factory=list)
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SatelliteObservation:
    """A satellite observation record."""
    observation_id: str
    sensor: str = ""
    acquisition_time: str = ""
    file_path: Optional[str] = None
    crs: Optional[str] = None
    resolution: Optional[float] = None
    bands: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
