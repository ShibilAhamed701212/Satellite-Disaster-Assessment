"""
Tests for the feature status system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from disaster_assessment.feature_status import (
    Status, FeatureReport, FeatureStatusRegistry,
    check_feature_status,
)


class TestStatus:
    def test_status_values(self):
        assert Status.READY.value == "READY"
        assert Status.UNAVAILABLE.value == "UNAVAILABLE"
        assert Status.EXPERIMENTAL.value == "EXPERIMENTAL"

    def test_status_count(self):
        assert len(Status) >= 15


class TestFeatureReport:
    def test_to_dict(self):
        report = FeatureReport(feature="test", status=Status.READY, reason="ok")
        d = report.to_dict()
        assert d["feature"] == "test"
        assert d["status"] == "READY"

    def test_str(self):
        report = FeatureReport(feature="test", status=Status.READY, reason="works")
        s = str(report)
        assert "READY" in s


class TestFeatureStatusRegistry:
    def test_register(self):
        registry = FeatureStatusRegistry()
        report = registry.register("test_feature", Status.READY, "works")
        assert report.feature == "test_feature"

    def test_get(self):
        registry = FeatureStatusRegistry()
        registry.register("feat1", Status.READY)
        registry.register("feat2", Status.UNAVAILABLE)

        assert registry.get("feat1") is not None
        assert registry.get("nonexistent") is None

    def test_get_all(self):
        registry = FeatureStatusRegistry()
        registry.register("a", Status.READY)
        registry.register("b", Status.READY)
        all_features = registry.get_all()
        assert len(all_features) == 2

    def test_get_summary(self):
        registry = FeatureStatusRegistry()
        registry.register("a", Status.READY)
        registry.register("b", Status.UNAVAILABLE)
        summary = registry.get_summary()
        assert summary["a"] == "READY"
        assert summary["b"] == "UNAVAILABLE"

    def test_summary_table(self):
        registry = FeatureStatusRegistry()
        registry.register("feature_a", Status.READY, "working")
        registry.register("feature_b", Status.UNAVAILABLE, "missing")
        table = registry.summary_table()
        assert "FEATURE STATUS REPORT" in table
        assert "feature_a" in table

    def test_check_requirements(self):
        registry = FeatureStatusRegistry()
        registry.register("core", Status.READY)
        registry.register("optional", Status.UNAVAILABLE)

        missing = registry.check_requirements(["core", "optional", "missing"])
        assert len(missing) == 2


class TestCheckFeatureStatus:
    def test_check_all_features(self):
        registry = check_feature_status()
        assert len(registry.get_all()) > 0
        # Core features should be registered
        assert registry.get("land_cover_segmentation") is not None
        assert registry.get("change_detection") is not None
        assert registry.get("flood_detection") is not None

    def test_feature_status_values(self):
        registry = check_feature_status()
        summary = registry.get_summary()
        for feature, status in summary.items():
            assert status in [s.value for s in Status]
