"""
Tests for the Digital Twin state engine.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.twin.state import DisasterTwin, TwinState
from disaster_assessment.twin.entities import (
    DisasterEvent, HazardLayer, FloodRegion,
    DamagedBuildingRegion, AffectedArea, SatelliteObservation,
)
from disaster_assessment.twin.versioning import VersionTracker


class TestDisasterTwin:
    def test_initial_state(self):
        twin = DisasterTwin()
        state = twin.current_state()
        assert state.events == []
        assert state.flood_regions == []
        assert twin.version == 0

    def test_update_from_analysis(self):
        twin = DisasterTwin()
        # Create mock report
        class MockFlood:
            flood_area_m2 = 1000.0
            flood_area_km2 = 0.001

        class MockSeverity:
            total_score = 75.0
            severity_level = "HIGH"

        class MockReport:
            flood_result = MockFlood()
            severity = MockSeverity()
            maps = {"change_map": "data", "flood_map": "data"}

        twin.update_from_analysis(MockReport())
        assert twin.version == 1
        state = twin.current_state()
        assert len(state.flood_regions) == 1
        assert len(state.affected_areas) == 1
        assert state.flood_regions[0].area_m2 == 1000.0

    def test_add_event(self):
        twin = DisasterTwin()
        event = DisasterEvent(
            event_id="evt_001",
            event_type="flood",
            location="Mumbai",
        )
        twin.add_event(event)
        assert len(twin.current_state().events) == 1

    def test_add_observation(self):
        twin = DisasterTwin()
        obs = SatelliteObservation(
            observation_id="obs_001",
            sensor="Sentinel-2",
        )
        twin.add_observation(obs)
        assert len(twin.current_state().observations) == 1

    def test_mode(self):
        twin = DisasterTwin()
        assert twin.mode == "IMPLEMENTED_ARCHITECTURE"
        twin.add_observation(SatelliteObservation(observation_id="o1", sensor="S2"))
        assert twin.mode == "STATIC_DATA_MODE"

    def test_snapshot(self):
        twin = DisasterTwin()
        twin.update_from_analysis(type('Report', (), {
            'flood_result': type('FR', (), {'flood_area_m2': 100, 'flood_area_km2': 0.0001})(),
            'severity': type('S', (), {'total_score': 50, 'severity_level': 'MODERATE'})(),
            'maps': {'change_map': True},
        })())
        twin.snapshot()
        assert len(twin.get_history()) == 1

    def test_summary(self):
        twin = DisasterTwin()
        summary = twin.summary()
        assert "Disaster Digital Twin" in summary


class TestVersionTracker:
    def test_record_version(self):
        tracker = VersionTracker()
        tracker.record(1, "2024-01-01", {"events": 1})
        versions = tracker.get_versions()
        assert len(versions) == 1
        assert versions[0].version == 1

    def test_get_latest(self):
        tracker = VersionTracker()
        tracker.record(1, "2024-01-01", {})
        tracker.record(2, "2024-01-02", {})
        latest = tracker.get_latest()
        assert latest.version == 2

    def test_get_by_version(self):
        tracker = VersionTracker()
        tracker.record(1, "2024-01-01", {})
        tracker.record(2, "2024-01-02", {})
        v = tracker.get_by_version(2)
        assert v is not None
        assert v.version == 2

    def test_checksum(self):
        tracker = VersionTracker()
        tracker.record(1, "2024-01-01", {"key": "value"})
        versions = tracker.get_versions()
        assert len(versions[0].checksum) == 8
