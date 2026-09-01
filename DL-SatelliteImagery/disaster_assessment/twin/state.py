"""
Digital Twin state management.

Maintains evolving disaster state based on real observations and analysis outputs.

Status: IMPLEMENTED ARCHITECTURE
Requires continuous data ingestion for live operation.
Current mode: STATIC DATA MODE (processes on-demand analysis results)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .entities import (
    DisasterEvent,
    HazardLayer,
    FloodRegion,
    DamagedBuildingRegion,
    AffectedArea,
    SatelliteObservation,
)


@dataclass
class TwinState:
    """Current state of the digital twin."""
    timestamp: str = ""
    events: List[DisasterEvent] = field(default_factory=list)
    hazard_layers: List[HazardLayer] = field(default_factory=list)
    flood_regions: List[FloodRegion] = field(default_factory=list)
    damaged_buildings: List[DamagedBuildingRegion] = field(default_factory=list)
    affected_areas: List[AffectedArea] = field(default_factory=list)
    observations: List[SatelliteObservation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "events_count": len(self.events),
            "hazard_layers_count": len(self.hazard_layers),
            "flood_regions_count": len(self.flood_regions),
            "damaged_buildings_count": len(self.damaged_buildings),
            "affected_areas_count": len(self.affected_areas),
            "observations_count": len(self.observations),
            "metadata": self.metadata,
        }


class DisasterTwin:
    """Stateful disaster intelligence engine.

    Maintains:
    - Current state from latest observations
    - Historical state versions
    - Provenance tracking

    Mode: STATIC DATA MODE by default.
    Requires real-time ingestion for LIVE_DATA_CONNECTED mode.
    """

    def __init__(self):
        self._current_state = TwinState(
            timestamp=datetime.now().isoformat(),
        )
        self._history: List[TwinState] = []
        self._version = 0

    def update_from_analysis(self, report: Any) -> None:
        """Update twin state from a DisasterReport.

        This is the main entry point for incorporating analysis results.
        """
        self._version += 1
        state = self._current_state

        if hasattr(report, "flood_result") and report.flood_result is not None:
            fr = report.flood_result
            state.flood_regions.append(FloodRegion(
                region_id=f"flood_{self._version}",
                area_m2=getattr(fr, "flood_area_m2", None),
                area_km2=getattr(fr, "flood_area_km2", None),
                timestamp=datetime.now().isoformat(),
            ))

        if hasattr(report, "severity") and report.severity is not None:
            state.affected_areas.append(AffectedArea(
                area_id=f"affected_{self._version}",
                severity_score=report.severity.total_score,
                severity_level=report.severity.severity_level,
                timestamp=datetime.now().isoformat(),
            ))

        if hasattr(report, "maps") and "change_map" in report.maps:
            state.hazard_layers.append(HazardLayer(
                layer_id=f"change_{self._version}",
                hazard_type="change",
                timestamp=datetime.now().isoformat(),
            ))

        if hasattr(report, "maps") and "flood_map" in report.maps:
            state.hazard_layers.append(HazardLayer(
                layer_id=f"flood_{self._version}",
                hazard_type="flood",
                timestamp=datetime.now().isoformat(),
            ))

        state.timestamp = datetime.now().isoformat()
        state.metadata["last_analysis_version"] = self._version

    def add_observation(self, observation: SatelliteObservation) -> None:
        """Record a satellite observation."""
        self._current_state.observations.append(observation)

    def add_event(self, event: DisasterEvent) -> None:
        """Register a disaster event."""
        self._current_state.events.append(event)

    def current_state(self) -> TwinState:
        """Get current twin state."""
        return self._current_state

    def get_history(self) -> List[TwinState]:
        """Get historical states."""
        return list(self._history)

    @property
    def version(self) -> int:
        return self._version

    @property
    def mode(self) -> str:
        """Return current operational mode."""
        if self._current_state.observations:
            return "STATIC_DATA_MODE"
        return "IMPLEMENTED_ARCHITECTURE"

    def snapshot(self) -> None:
        """Save current state as a historical snapshot."""
        import copy
        self._history.append(copy.deepcopy(self._current_state))

    def summary(self) -> str:
        """Human-readable summary of twin state."""
        s = self._current_state
        lines = [
            f"Disaster Digital Twin (v{self._version})",
            f"  Mode: {self.mode}",
            f"  Events: {len(s.events)}",
            f"  Hazard Layers: {len(s.hazard_layers)}",
            f"  Flood Regions: {len(s.flood_regions)}",
            f"  Affected Areas: {len(s.affected_areas)}",
            f"  Observations: {len(s.observations)}",
            f"  Last Update: {s.timestamp}",
        ]
        return "\n".join(lines)
