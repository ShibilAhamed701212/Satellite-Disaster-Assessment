"""
Unified Disaster Assessment Pipeline (Phase 1 & Phase 2).

Orchestrates the full analysis workflow:
    1. Validate input images
    2. Align/resize image pairs
    3. Validate models before inference (no random-weight output)
    4. Run land-cover segmentation (trained model or clearly-labeled heuristic)
    5. Run change detection (validated Siamese U-Net)
    6. Run flood detection (validated Flood U-Net) + Georeferenced area calculation
    7. Run building damage assessment (Validated model OR Estimated structural change)
    8. Calculate comprehensive land change & severity metrics
    9. Generate rich multi-layer visualizations & heatmaps
    10. Return structured DisasterReport

Optimized for GTX 1650 (4GB VRAM):
    - Lazy model loading
    - torch.no_grad() inference
    - Mixed precision on CUDA
    - Automatic memory cleanup

Model safety:
    - Every model is validated before inference
    - Random/degenerate weights are detected and refused
    - Heuristic fallback is clearly labeled
    - Reports distinguish trained vs heuristic vs unavailable
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from ..preprocessing.image_validation import ImageValidator, ValidationReport
from ..preprocessing.image_alignment import ImageAligner, AlignmentReport
from ..models.siamese_unet import SiameseUNet, create_siamese_unet
from ..models.flood_unet import FloodUNet, create_flood_unet
from ..models.base_unet import get_device
from ..models.validation import ModelRegistry, ModelInferenceMode, ModelStatus, ValidationResult
from ..analytics.area_calculator import AreaCalculator
from ..analytics.geospatial_area import GeospatialAreaCalculator, GeospatialFloodResult
from ..analytics.damage_metrics import DamageMetrics, BuildingDamageResult
from ..analytics.land_change_metrics import LandChangeMetrics, LandChangeResult
from ..analytics.severity_engine import SeverityEngine, SeverityResult
from ..inference.building_damage_inference import (
    BuildingDamageInference,
    DamageAssessmentOutput,
    TRAINED_DAMAGE_COLORS,
)
from ..visualization.overlays import OverlayRenderer, LANDCOVER_COLORS, LANDCOVER_LABELS
from ..visualization.heatmap import HeatmapGenerator
from ..visualization.comparison import ComparisonRenderer


@dataclass
class DisasterReport:
    """Structured output from the disaster analysis pipeline.

    Every component records:
    - method: How the result was produced
    - model: Which model/heuristic was used
    - mode: trained, heuristic, unavailable
    - availability: Whether the feature is actually operational
    """

    # Validation
    pre_validation: Optional[ValidationReport] = None
    post_validation: Optional[ValidationReport] = None
    alignment: Optional[AlignmentReport] = None

    # Change & Flood Metrics
    change_percentage: float = 0.0
    change_mode: str = "unavailable"  # trained, unavailable, heuristic
    flood_percentage: float = 0.0
    flood_mode: str = "unavailable"  # trained, unavailable
    flood_result: Optional[GeospatialFloodResult] = None
    flood_area_m2: Optional[float] = None
    flood_area_km2: Optional[float] = None
    area_status: str = "Geospatial resolution unavailable"

    # Building Damage (Phase 1 & Phase 2 unified)
    building_damage: Optional[Union[BuildingDamageResult, DamageAssessmentOutput]] = None
    damage_output: Optional[DamageAssessmentOutput] = None
    damage_mode: str = "estimated"

    # Land Cover
    landcover_mode: str = "heuristic"  # trained, heuristic, unavailable

    # Land Change & Severity
    land_change: Optional[LandChangeResult] = None
    severity: Optional[SeverityResult] = None
    vegetation_loss_percentage: float = 0.0
    water_expansion_percentage: float = 0.0

    # Maps (numpy arrays)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)

    # Model validation results
    model_validations: Dict[str, dict] = field(default_factory=dict)

    # Warnings and info
    warnings: List[str] = field(default_factory=list)
    measurement_basis: str = "pixel-based estimate"

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary (without map arrays)."""
        result = {
            "change_percentage": round(self.change_percentage, 2),
            "change_mode": self.change_mode,
            "vegetation_loss_percentage": round(self.vegetation_loss_percentage, 2),
            "water_expansion_percentage": round(self.water_expansion_percentage, 2),
            "measurement_basis": self.measurement_basis,
            "area_status": self.area_status,
            "landcover_mode": self.landcover_mode,
            "flood_mode": self.flood_mode,
            "warnings": self.warnings,
            "model_validations": self.model_validations,
        }

        # Flood structure
        if self.flood_result is not None:
            result["flood"] = self.flood_result.to_dict()
            result["flood_percentage"] = round(self.flood_result.flood_percentage, 2)
            result["flood_area_m2"] = self.flood_result.flood_area_m2
            result["flood_area_km2"] = self.flood_result.flood_area_km2
        else:
            result["flood_percentage"] = round(self.flood_percentage, 2)

        # Building damage structure
        if self.damage_output is not None:
            result["building_damage"] = self.damage_output.to_dict()
        elif self.building_damage is not None and hasattr(self.building_damage, "to_dict"):
            result["building_damage"] = self.building_damage.to_dict()
        elif self.building_damage is not None:
            result["building_damage"] = {
                "damage_mode": "estimated",
                "possible_damage_percentage": round(
                    getattr(self.building_damage, "possible_damage_percentage", 0.0), 2
                ),
                "severe_damage_percentage": round(
                    getattr(self.building_damage, "severe_damage_percentage", 0.0), 2
                ),
                "overall_damage_score": round(
                    getattr(self.building_damage, "overall_damage_score", 0.0), 2
                ),
            }

        if self.severity is not None:
            result["severity_score"] = round(self.severity.total_score, 2)
            result["severity_level"] = self.severity.severity_level
        if self.land_change is not None:
            result["total_changed_percentage"] = round(
                self.land_change.total_changed_percentage, 2
            )
        return result

    def summary(self) -> str:
        """Generate a human-readable summary of the full analysis."""
        lines = [
            "=" * 60,
            "  SATELLITE DISASTER ASSESSMENT REPORT",
            "=" * 60,
            "",
        ]

        if self.severity is not None:
            lines.extend([
                f"  Severity Score: {self.severity.total_score:.1f}/100",
                f"  Severity Level: {self.severity.severity_level}",
                "",
            ])

        lines.extend([
            f"  Changed Area:      {self.change_percentage:.2f}% [{self.change_mode}]",
            f"  Flooded Area:      {self.flood_percentage:.2f}% [{self.flood_mode}]",
        ])

        if self.flood_area_m2 is not None:
            lines.append(f"    * Metric Area:   {self.flood_area_m2:,.2f} m2 ({self.flood_area_km2:,.6f} km2)")

        lines.extend([
            f"  Vegetation Loss:   {self.vegetation_loss_percentage:.2f}%",
            f"  Water Expansion:   {self.water_expansion_percentage:.2f}%",
        ])

        # Building damage summary
        if self.damage_output is not None:
            lines.extend([
                f"  Building Damage ({self.damage_output.damage_mode.upper()} MODE): {self.damage_output.overall_damage_score:.1f}/100",
            ])
            if self.damage_output.damage_mode == "trained":
                lines.extend([
                    f"    * Undamaged:    {self.damage_output.undamaged_percentage:.1f}%",
                    f"    * Minor Damage: {self.damage_output.minor_damage_percentage:.1f}%",
                    f"    * Major Damage: {self.damage_output.major_damage_percentage:.1f}%",
                    f"    * Destroyed:    {self.damage_output.destroyed_percentage:.1f}%",
                ])
            elif self.damage_output.estimated_details is not None:
                lines.extend([
                    f"    * Possible:     {self.damage_output.estimated_details.possible_damage_percentage:.1f}%",
                    f"    * Severe:       {self.damage_output.estimated_details.severe_damage_percentage:.1f}%",
                ])
        elif self.building_damage is not None:
            lines.extend([
                f"  Building Damage:   {self.building_damage.overall_damage_score:.1f}/100",
                f"    * Possible:      {self.building_damage.possible_damage_percentage:.1f}%",
                f"    * Severe:        {self.building_damage.severe_damage_percentage:.1f}%",
            ])

        lines.extend([
            "",
            f"  Land Cover Mode: {self.landcover_mode}",
            f"  Measurement: {self.measurement_basis}",
            f"  Area Status: {self.area_status}",
        ])

        # Model status summary
        if self.model_validations:
            lines.append("")
            lines.append("  Model Status:")
            for name, val in self.model_validations.items():
                status_str = val.get("status", "UNKNOWN")
                mode_str = val.get("inference_mode", "unknown")
                lines.append(f"    {name}: {status_str} ({mode_str})")

        if self.warnings:
            lines.append("")
            lines.append("  Warnings:")
            for w in self.warnings:
                lines.append(f"    ! {w}")

        lines.append("=" * 60)
        return "\n".join(lines)


class DisasterAnalyzer:
    """
    Main orchestrator for satellite disaster assessment (Phase 1 & Phase 2).

    Key safety guarantees:
    - Models are validated before inference
    - Random/degenerate weights are refused
    - Heuristic fallbacks are clearly labeled
    - Results always indicate their provenance
    """

    def __init__(
        self,
        change_model_weights: Optional[str] = None,
        flood_model_weights: Optional[str] = None,
        trained_damage_weights: Optional[str] = None,
        landcover_model_path: Optional[str] = None,
        gsd_meters: Optional[float] = None,
        target_size: Tuple[int, int] = (256, 256),
        device: Optional[torch.device] = None,
        enable_feature_alignment: bool = False,
        severity_config_path: Optional[str] = None,
        config=None,
    ):
        self.device = device or get_device()
        self.target_size = target_size
        self.gsd_meters = gsd_meters
        self.config = config

        # Weight paths for lazy loading
        self._change_weights = change_model_weights
        self._flood_weights = flood_model_weights
        self._trained_damage_weights = trained_damage_weights
        self._landcover_path = landcover_model_path

        # Core components
        self.validator = ImageValidator()
        self.aligner = ImageAligner(
            target_size=target_size,
            enable_feature_alignment=enable_feature_alignment,
        )
        self.area_calculator = AreaCalculator(gsd_meters=gsd_meters)
        self.damage_metrics = DamageMetrics()
        self.land_change = LandChangeMetrics()
        self.severity_engine = SeverityEngine(config_path=severity_config_path)
        self.overlay_renderer = OverlayRenderer()
        self.heatmap_generator = HeatmapGenerator()
        self.comparison_renderer = ComparisonRenderer()

        # Phase 2 components
        self.geospatial_calculator = GeospatialAreaCalculator()
        self.damage_inference = BuildingDamageInference(
            weights_path=trained_damage_weights,
            device=self.device,
        )

        # Model registry for validation
        quality_config = {}
        if config and hasattr(config, 'get_quality_config'):
            quality_config = config.get_quality_config()
        self.model_registry = ModelRegistry(quality_config=quality_config)

        # Lazy-loaded models
        self._change_model: Optional[SiameseUNet] = None
        self._change_validation: Optional[ValidationResult] = None
        self._flood_model: Optional[FloodUNet] = None
        self._flood_validation: Optional[ValidationResult] = None
        self._landcover_model = None

    def validate_models(self) -> Dict[str, dict]:
        """Validate all models and return their status."""
        # Validate change detection
        change_result = self.model_registry.register(
            "change_detection",
            SiameseUNet,
            weights_path=self._change_weights,
            device=self.device,
        )

        # Validate flood detection
        flood_result = self.model_registry.register(
            "flood_detection",
            FloodUNet,
            weights_path=self._flood_weights,
            device=self.device,
        )

        # Validate building damage
        damage_result = self.damage_inference.validate()

        return {
            "change_detection": change_result.to_dict(),
            "flood_detection": flood_result.to_dict(),
            "building_damage": damage_result,
        }

    def _load_change_model(self) -> bool:
        """Load change detection model if validated."""
        if self._change_model is not None:
            return True

        if self._change_validation is None:
            self._change_validation = self.model_registry.register(
                "change_detection",
                SiameseUNet,
                weights_path=self._change_weights,
                device=self.device,
            )

        if not self._change_validation.is_usable:
            return False

        try:
            self._change_model = create_siamese_unet(
                weights_path=self._change_weights,
                device=self.device,
            )
            return True
        except Exception as e:
            print(f"[DisasterAnalyzer] Failed to load change detection model: {e}")
            return False

    def _load_flood_model(self) -> bool:
        """Load flood detection model if validated."""
        if self._flood_model is not None:
            return True

        if self._flood_validation is None:
            self._flood_validation = self.model_registry.register(
                "flood_detection",
                FloodUNet,
                weights_path=self._flood_weights,
                device=self.device,
            )

        if not self._flood_validation.is_usable:
            return False

        try:
            self._flood_model = create_flood_unet(
                weights_path=self._flood_weights,
                device=self.device,
            )
            return True
        except Exception as e:
            print(f"[DisasterAnalyzer] Failed to load flood detection model: {e}")
            return False

    def _load_landcover_model(self):
        if self._landcover_model is None and self._landcover_path is not None:
            try:
                import segmentation_models as sm
                from keras import backend as K
                from keras.models import load_model

                def jaccard_coef(y_true, y_pred):
                    y_true_flatten = K.flatten(y_true)
                    y_pred_flatten = K.flatten(y_pred)
                    intersection = K.sum(y_true_flatten * y_pred_flatten)
                    return (intersection + 1.0) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)

                weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
                dice_loss = sm.losses.DiceLoss(class_weights=weights)
                focal_loss = sm.losses.CategoricalFocalLoss()
                total_loss = dice_loss + (1 * focal_loss)

                self._landcover_model = load_model(
                    self._landcover_path,
                    custom_objects={
                        "dice_loss_plus_1focal_loss": total_loss,
                        "jaccard_coef": jaccard_coef,
                    },
                )
            except Exception as e:
                self._landcover_model = None

    def _run_landcover_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Run land-cover segmentation.

        Returns:
            Tuple of (segmentation_mask, mode_used)
            mode_used is one of: "trained", "heuristic", "unavailable"
        """
        self._load_landcover_model()
        if self._landcover_model is not None:
            try:
                h, w = image.shape[:2]
                img_resized = np.array(__import__("PIL").Image.fromarray(image).resize((256, 256)))
                img_input = np.expand_dims(img_resized.astype(np.float32) / 255.0, 0)
                prediction = self._landcover_model.predict(img_input, verbose=0)
                seg_map = np.argmax(prediction, axis=3)[0]
                if seg_map.shape != (h, w):
                    seg_map = np.array(
                        __import__("PIL").Image.fromarray(seg_map.astype(np.uint8)).resize(
                            (w, h), __import__("PIL").Image.NEAREST
                        )
                    )
                return seg_map, "trained"
            except Exception:
                pass

        # Heuristic fallback — clearly labeled
        result = self._simple_landcover_estimate(image)
        return result, "heuristic"

    @staticmethod
    def _simple_landcover_estimate(image: np.ndarray) -> np.ndarray:
        """
        HSV color-space heuristic for land-cover estimation.

        DISCLAIMER: This is NOT a trained segmentation model.
        It uses color-space thresholds and should not be presented as
        AI-based segmentation. Results are approximate at best.
        """
        import cv2

        h, w = image.shape[:2]
        seg_map = np.full((h, w), 5, dtype=np.uint8)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        water_mask = (hue >= 90) & (hue <= 130) & (sat > 50)
        seg_map[water_mask] = 0

        veg_mask = (hue >= 30) & (hue <= 85) & (sat > 40) & (val > 40)
        seg_map[veg_mask] = 4

        building_mask = (val < 80) & (sat < 60)
        seg_map[building_mask] = 3

        road_mask = (sat < 40) & (val >= 80) & (val <= 180)
        seg_map[road_mask] = 2

        land_mask = seg_map == 5
        land_hint = (hue >= 10) & (hue <= 30) & (sat > 30)
        seg_map[land_mask & land_hint] = 1

        return seg_map

    def _numpy_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        img_float = image.astype(np.float32) / 255.0
        return torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def analyze(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        damage_mode: str = "auto",
        meters_per_pixel: Optional[float] = None,
        geotiff_path: Optional[str] = None,
    ) -> DisasterReport:
        """
        Run the complete disaster assessment pipeline.

        Safety guarantees:
        - No random-weight inference
        - Heuristic results clearly labeled
        - Model validation before inference
        - UNAVAILABLE results when models missing

        Args:
            pre_image: Pre-disaster image (H, W, 3) uint8 RGB.
            post_image: Post-disaster image (H, W, 3) uint8 RGB.
            damage_mode: "auto", "trained", or "estimated".
            meters_per_pixel: Optional ground resolution in meters per pixel.
            geotiff_path: Optional path to GeoTIFF file for automated metadata extraction.

        Returns:
            DisasterReport with full analytics, metrics, and visualization overlays.
        """
        report = DisasterReport()
        effective_gsd = meters_per_pixel if meters_per_pixel is not None else self.gsd_meters

        # Validate all models upfront
        report.model_validations = self.validate_models()

        # Step 1: Validate images
        report.pre_validation = self.validator.validate_array(pre_image)
        report.post_validation = self.validator.validate_array(post_image)

        if not report.pre_validation.is_valid:
            report.warnings.append(f"Pre-image validation failed: {report.pre_validation.errors}")
            return report
        if not report.post_validation.is_valid:
            report.warnings.append(f"Post-image validation failed: {report.post_validation.errors}")
            return report

        pre_image = ImageValidator.normalize_image(pre_image)
        post_image = ImageValidator.normalize_image(post_image)

        # Step 2: Align images
        pre_aligned, post_aligned, alignment_report = self.aligner.align(pre_image, post_image)
        report.alignment = alignment_report
        for w in alignment_report.warnings:
            report.warnings.append(f"Alignment: {w}")

        # Step 3: Land-cover segmentation (with honest mode label)
        pre_segmentation, lc_mode = self._run_landcover_segmentation(pre_aligned)
        post_segmentation, _ = self._run_landcover_segmentation(post_aligned)
        report.landcover_mode = lc_mode
        if lc_mode == "heuristic":
            report.warnings.append(
                "Land-cover uses HSV color heuristic, not a trained segmentation model. "
                "Results are approximate."
            )

        # Prepare tensors (needed if either model is available)
        pre_tensor = self._numpy_to_tensor(pre_aligned)
        post_tensor = self._numpy_to_tensor(post_aligned)

        # Step 4: Change detection
        change_mask_tensor = None
        if self._load_change_model():
            report.change_mode = "trained"

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        change_mask_tensor = self._change_model.predict(pre_tensor, post_tensor)
                else:
                    change_mask_tensor = self._change_model.predict(pre_tensor, post_tensor)

            change_mask = change_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
            change_area = self.area_calculator.total_changed_area(change_mask)
            report.change_percentage = change_area.percentage
        else:
            report.change_mode = "unavailable"
            change_mask = np.zeros(
                (pre_aligned.shape[0], pre_aligned.shape[1]), dtype=np.uint8
            )
            report.warnings.append(
                "Change detection unavailable: no validated trained checkpoint. "
                "Change percentage set to 0.0%."
            )

        # Step 5: Flood detection
        flood_mask_tensor = None
        if self._load_flood_model():
            report.flood_mode = "trained"

            with torch.no_grad():
                if self.device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        flood_mask_tensor = self._flood_model.predict(post_tensor)
                else:
                    flood_mask_tensor = self._flood_model.predict(post_tensor)

            flood_mask = flood_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
        else:
            report.flood_mode = "unavailable"
            flood_mask = np.zeros(
                (pre_aligned.shape[0], pre_aligned.shape[1]), dtype=np.uint8
            )
            report.warnings.append(
                "Flood detection unavailable: no validated trained checkpoint. "
                "Flood percentage set to 0.0%."
            )

        # Calculate geospatial flood area
        flood_geo = self.geospatial_calculator.calculate_flood_area(
            flood_mask=flood_mask,
            geotiff_path=geotiff_path,
            meters_per_pixel=effective_gsd,
        )
        report.flood_result = flood_geo
        report.flood_percentage = flood_geo.flood_percentage
        report.flood_area_m2 = flood_geo.flood_area_m2
        report.flood_area_km2 = flood_geo.flood_area_km2
        report.area_status = flood_geo.area_status

        if flood_geo.flood_area_m2 is not None:
            report.measurement_basis = (
                f"geospatially calculated ({flood_geo.georeference.source_type})"
            )
        else:
            report.measurement_basis = "pixel-based estimate (no geographic resolution provided)"

        # Step 6: Building Damage Assessment
        damage_out = self.damage_inference.assess(
            pre_image=pre_aligned,
            post_image=post_aligned,
            pre_segmentation=pre_segmentation,
            post_segmentation=post_segmentation,
            mode=damage_mode,
        )
        report.damage_output = damage_out
        report.damage_mode = damage_out.damage_mode
        report.building_damage = damage_out if damage_out.damage_mode == "trained" else (damage_out.estimated_details or damage_out)
        if damage_out.warning:
            report.warnings.append(damage_out.warning)

        # Step 7: Land Change Metrics
        report.land_change = self.land_change.compute(pre_segmentation, post_segmentation)
        report.vegetation_loss_percentage = report.land_change.vegetation_loss_percentage
        report.water_expansion_percentage = report.land_change.water_expansion_percentage

        # Step 8: Severity Scoring
        # When models are unavailable, severity should reflect that
        damage_score = damage_out.overall_damage_score
        change_for_severity = report.change_percentage if report.change_mode == "trained" else 0.0
        flood_for_severity = report.flood_percentage if report.flood_mode == "trained" else 0.0

        report.severity = self.severity_engine.calculate_severity(
            flood_percentage=flood_for_severity,
            building_damage_score=damage_score,
            changed_area_percentage=change_for_severity,
            vegetation_loss_percentage=report.vegetation_loss_percentage,
        )

        # Add severity warnings if components were unavailable
        unavailable_components = []
        if report.change_mode == "unavailable":
            unavailable_components.append("change_detection")
        if report.flood_mode == "unavailable":
            unavailable_components.append("flood_detection")
        if unavailable_components:
            report.warnings.append(
                f"Severity score excludes unavailable components: {', '.join(unavailable_components)}. "
                f"Actual severity may be higher."
            )

        # Step 9: Visualizations
        report.maps["pre_image"] = pre_aligned
        report.maps["post_image"] = post_aligned
        report.maps["pre_landcover"] = self.overlay_renderer.landcover_colorized(pre_segmentation)
        report.maps["post_landcover"] = self.overlay_renderer.landcover_colorized(post_segmentation)
        report.maps["change_map"] = self.overlay_renderer.change_overlay(post_aligned, change_mask)
        report.maps["flood_map"] = self.overlay_renderer.flood_overlay(post_aligned, flood_mask)

        if damage_out.damage_map is not None:
            if damage_out.damage_mode == "trained":
                report.maps["building_damage"] = self.overlay_renderer.trained_damage_overlay(
                    post_aligned, damage_out.damage_map
                )
            else:
                report.maps["building_damage"] = self.overlay_renderer.damage_overlay(
                    post_aligned, damage_out.damage_map
                )

        report.maps["vegetation_loss"] = self.overlay_renderer.vegetation_loss_overlay(
            post_aligned, pre_segmentation, post_segmentation
        )

        veg_loss_mask = ((pre_segmentation == 4) & (post_segmentation != 4)).astype(np.uint8)
        report.maps["disaster_heatmap"] = self.heatmap_generator.overlay_heatmap(
            post_aligned,
            self.heatmap_generator.combined_disaster_heatmap(
                change_mask=change_mask,
                flood_mask=flood_mask,
                damage_map=damage_out.damage_map,
                vegetation_loss_mask=veg_loss_mask,
            ),
        )

        # Cleanup GPU cache
        del pre_tensor, post_tensor
        if change_mask_tensor is not None:
            del change_mask_tensor
        if flood_mask_tensor is not None:
            del flood_mask_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return report

    def analyze_single(self, image: np.ndarray) -> Dict[str, Any]:
        validation = self.validator.validate_array(image)
        if not validation.is_valid:
            raise ValueError(f"Image validation failed: {validation.errors}")

        image = ImageValidator.normalize_image(image)
        h, w = image.shape[:2]
        target_h, target_w = self.target_size

        if h != target_h or w != target_w:
            import cv2
            image_resized = cv2.resize(image, (target_w, target_h))
        else:
            image_resized = image

        segmentation, lc_mode = self._run_landcover_segmentation(image_resized)
        overlay = self.overlay_renderer.landcover_overlay(image_resized, segmentation)
        colorized = self.overlay_renderer.landcover_colorized(segmentation)
        legend = self.overlay_renderer.create_legend(LANDCOVER_LABELS, LANDCOVER_COLORS)

        # Calculate class distribution statistics
        total_pixels = segmentation.size
        class_counts = {}
        class_percentages = {}
        for cls_id, cls_name in LANDCOVER_LABELS.items():
            count = int((segmentation == cls_id).sum())
            pct = round((count / max(total_pixels, 1)) * 100.0, 2)
            class_counts[cls_name] = count
            class_percentages[cls_name] = pct

        mode_label = "TRAINED MODEL" if lc_mode == "trained" else "HSV COLOR HEURISTIC (not AI)"
        summary_lines = [
            "=" * 50,
            "  LAND-COVER SEGMENTATION REPORT (6 CLASSES)",
            "=" * 50,
            f"  Total Pixels: {total_pixels:,} ({target_w}x{target_h})",
            f"  Method: {mode_label}",
            "",
            "  Class Distribution:",
        ]
        for cls_id, cls_name in LANDCOVER_LABELS.items():
            cnt = class_counts[cls_name]
            pct = class_percentages[cls_name]
            summary_lines.append(f"    * {cls_name:<12}: {pct:>6.2f}% ({cnt:,} px)")
        summary_lines.append("=" * 50)

        return {
            "original": image_resized,
            "segmentation": colorized,
            "raw_mask": segmentation,
            "overlay": overlay,
            "legend": legend,
            "total_pixels": total_pixels,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
            "mode": lc_mode,
            "summary": "\n".join(summary_lines),
        }
