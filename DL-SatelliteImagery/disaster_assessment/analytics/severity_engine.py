"""
Configurable rule-based disaster severity scoring engine.

Computes a normalized severity score (0-100) from:
    - Flood percentage
    - Building damage score
    - Changed area percentage
    - Vegetation loss percentage

Weights and thresholds are loaded from a YAML config file.
All weights, thresholds, and labels are fully configurable.

The engine explains HOW the score was calculated.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import yaml


# Default config file path (relative to this module)
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "severity_config.yaml",
)


@dataclass
class SeverityResult:
    """Structured severity assessment result."""

    total_score: float = 0.0
    severity_level: str = "LOW"
    component_scores: Dict[str, Dict] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    thresholds_used: Dict[str, int] = field(default_factory=dict)
    explanation: str = ""

    def summary(self) -> str:
        lines = [
            "Disaster Severity Assessment:",
            f"  Score: {self.total_score:.1f} / 100",
            f"  Level: {self.severity_level}",
            "",
            "  Component Breakdown:",
        ]
        for name, info in self.component_scores.items():
            raw = info.get("raw_value", 0)
            normalized = info.get("normalized", 0)
            weight = info.get("weight", 0)
            contribution = info.get("contribution", 0)
            lines.append(
                f"    {name:<20}: raw={raw:.1f}%  "
                f"normalized={normalized:.1f}  "
                f"weight={weight:.2f}  "
                f"contribution={contribution:.1f}"
            )
        lines.extend([
            "",
            f"  {self.explanation}",
        ])
        return "\n".join(lines)


class SeverityEngine:
    """
    Configurable rule-based disaster severity scorer.

    Default formula:
        severity_score = (
            flood_weight * flood_score
          + building_weight * building_damage_score
          + change_weight * changed_area_score
          + vegetation_weight * vegetation_loss_score
        )

    All components are normalized to 0-100 before weighting.
    The final score is clamped to 0-100.

    Severity levels:
        0-20:   LOW
        21-40:  MODERATE
        41-70:  HIGH
        71-100: CRITICAL
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to severity_config.yaml. Uses default if None.
        """
        self.config = self._load_config(config_path or DEFAULT_CONFIG_PATH)
        self.weights = self.config.get("weights", {})
        self.thresholds = self.config.get("thresholds", {})
        self.severity_labels = self.config.get("severity_labels", {
            "low": "LOW",
            "moderate": "MODERATE",
            "high": "HIGH",
            "critical": "CRITICAL",
        })

    def _load_config(self, config_path: str) -> dict:
        """Load and validate severity configuration."""
        if not os.path.isfile(config_path):
            print(
                f"[SeverityEngine] Config not found at {config_path}. "
                f"Using default weights."
            )
            return self._default_config()

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            return self._default_config()

        # Validate weights sum to ~1.0
        weights = config.get("weights", {})
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            print(
                f"[SeverityEngine] Warning: weights sum to {weight_sum:.2f}, "
                f"expected ~1.0. Normalizing."
            )
            for k in weights:
                weights[k] /= weight_sum

        return config

    @staticmethod
    def _default_config() -> dict:
        """Return default severity configuration."""
        return {
            "weights": {
                "flood": 0.40,
                "building_damage": 0.35,
                "change": 0.15,
                "vegetation_loss": 0.10,
            },
            "thresholds": {
                "low": 20,
                "moderate": 40,
                "high": 70,
                "critical": 100,
            },
            "severity_labels": {
                "low": "LOW",
                "moderate": "MODERATE",
                "high": "HIGH",
                "critical": "CRITICAL",
            },
        }

    def calculate_severity(
        self,
        flood_percentage: float = 0.0,
        building_damage_score: float = 0.0,
        changed_area_percentage: float = 0.0,
        vegetation_loss_percentage: float = 0.0,
    ) -> SeverityResult:
        """
        Calculate disaster severity score.

        All input values should be in the range 0-100.

        Args:
            flood_percentage: Percentage of area detected as flooded (0-100).
            building_damage_score: Building damage score (0-100).
            changed_area_percentage: Total changed area percentage (0-100).
            vegetation_loss_percentage: Vegetation loss percentage (0-100).

        Returns:
            SeverityResult with score, level, and component breakdown.
        """
        result = SeverityResult()
        result.weights_used = dict(self.weights)
        result.thresholds_used = dict(self.thresholds)

        # Normalize all inputs to 0-100 range (clamp)
        components = {
            "flood": min(max(flood_percentage, 0), 100),
            "building_damage": min(max(building_damage_score, 0), 100),
            "change": min(max(changed_area_percentage, 0), 100),
            "vegetation_loss": min(max(vegetation_loss_percentage, 0), 100),
        }

        # Calculate weighted score
        total_score = 0.0
        for name, raw_value in components.items():
            weight = self.weights.get(name, 0.0)
            contribution = raw_value * weight
            total_score += contribution

            result.component_scores[name] = {
                "raw_value": raw_value,
                "normalized": raw_value,
                "weight": weight,
                "contribution": contribution,
            }

        result.total_score = min(max(total_score, 0), 100)

        # Determine severity level
        result.severity_level = self._get_severity_level(result.total_score)

        # Build explanation
        result.explanation = self._build_explanation(result)

        return result

    def _get_severity_level(self, score: float) -> str:
        """Map score to severity level using configured thresholds."""
        if score <= self.thresholds.get("low", 20):
            return self.severity_labels.get("low", "LOW")
        elif score <= self.thresholds.get("moderate", 40):
            return self.severity_labels.get("moderate", "MODERATE")
        elif score <= self.thresholds.get("high", 70):
            return self.severity_labels.get("high", "HIGH")
        else:
            return self.severity_labels.get("critical", "CRITICAL")

    @staticmethod
    def _build_explanation(result: SeverityResult) -> str:
        """Build a human-readable explanation of the score."""
        parts = []
        for name, info in result.component_scores.items():
            contribution = info["contribution"]
            weight = info["weight"]
            raw = info["raw_value"]
            if contribution > 0:
                parts.append(
                    f"{name}({raw:.1f}% × {weight:.2f} = {contribution:.1f})"
                )
        formula = " + ".join(parts) if parts else "no contributing factors"
        return f"Score = {formula} = {result.total_score:.1f}"
