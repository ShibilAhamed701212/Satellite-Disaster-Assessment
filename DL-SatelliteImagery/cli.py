"""
CLI for Satellite Disaster Assessment System.

Commands:
    analyze           Run disaster analysis on image pairs
    validate-models   Check model validation status
    feature-status    Show feature availability
    help              Show help

Usage:
    python cli.py analyze --pre pre.tif --post post.tif --output results/
    python cli.py validate-models
    python cli.py feature-status
"""

import argparse
import json
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_analyze(args):
    """Run disaster analysis on image pairs."""
    import numpy as np
    from PIL import Image

    from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer

    print("=" * 60)
    print("  DISASTER ASSESSMENT PIPELINE")
    print("=" * 60)

    # Load images
    if not os.path.isfile(args.pre):
        print(f"Error: Pre-image not found: {args.pre}")
        return 1
    if not os.path.isfile(args.post):
        print(f"Error: Post-image not found: {args.post}")
        return 1

    print(f"  Pre-image:  {args.pre}")
    print(f"  Post-image: {args.post}")

    pre_image = np.array(Image.open(args.pre).convert("RGB"))
    post_image = np.array(Image.open(args.post).convert("RGB"))

    print(f"  Pre size:   {pre_image.shape}")
    print(f"  Post size:  {post_image.shape}")

    # Initialize analyzer
    weights_dir = os.path.join(os.path.dirname(__file__), "disaster_assessment", "weights")
    change_w = os.path.join(weights_dir, "siamese_unet.pth")
    flood_w = os.path.join(weights_dir, "flood_unet.pth")
    damage_w = os.path.join(weights_dir, "building_damage", "best_damage_model.pth")

    analyzer = DisasterAnalyzer(
        change_model_weights=change_w if os.path.isfile(change_w) else None,
        flood_model_weights=flood_w if os.path.isfile(flood_w) else None,
        trained_damage_weights=damage_w if os.path.isfile(damage_w) else None,
        landcover_model_path=None,
        gsd_meters=args.gsd if args.gsd and args.gsd > 0 else None,
    )

    # Run analysis
    print("\nRunning analysis...")
    report = analyzer.analyze(
        pre_image=pre_image,
        post_image=post_image,
        damage_mode=args.damage_mode,
        meters_per_pixel=args.gsd if args.gsd and args.gsd > 0 else None,
        geotiff_path=args.geotiff,
    )

    # Print report
    print(report.summary())

    # Save outputs
    if args.output:
        os.makedirs(args.output, exist_ok=True)

        # Save report JSON
        report_dict = report.to_dict()
        json_path = os.path.join(args.output, "report.json")
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"\nReport saved to: {json_path}")

        # Save visualization maps
        for name, map_array in report.maps.items():
            map_path = os.path.join(args.output, f"{name}.png")
            Image.fromarray(map_array).save(map_path)
            print(f"  Saved: {name}.png")

        # Save summary
        summary_path = os.path.join(args.output, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(report.summary())
        print("  Saved: summary.txt")

    print("\nAnalysis complete.")
    return 0


def cmd_validate_models(args):
    """Check model validation status."""
    from disaster_assessment.models.siamese_unet import SiameseUNet
    from disaster_assessment.models.flood_unet import FloodUNet
    from disaster_assessment.models.building_damage_model import BuildingDamageUNet
    from disaster_assessment.models.validation import ModelRegistry

    weights_dir = os.path.join(os.path.dirname(__file__), "disaster_assessment", "weights")
    change_w = os.path.join(weights_dir, "siamese_unet.pth")
    flood_w = os.path.join(weights_dir, "flood_unet.pth")
    damage_w = os.path.join(weights_dir, "building_damage", "best_damage_model.pth")

    # Quality thresholds from config
    quality_config = {
        "change_detection": {"minimum_epoch": 5, "minimum_iou": 0.4},
        "flood_detection": {"minimum_epoch": 5, "minimum_iou": 0.4},
        "building_damage": {"minimum_epoch": 5, "minimum_iou": 0.3},
    }

    registry = ModelRegistry(quality_config=quality_config)

    print("Validating models...")
    print()

    # Validate each model
    registry.register(
        "change_detection",
        SiameseUNet,
        weights_path=change_w if os.path.isfile(change_w) else None,
    )

    registry.register(
        "flood_detection",
        FloodUNet,
        weights_path=flood_w if os.path.isfile(flood_w) else None,
    )

    registry.register(
        "building_damage",
        BuildingDamageUNet,
        weights_path=damage_w if os.path.isfile(damage_w) else None,
    )

    print(registry.summary())

    if args.json:
        result = {}
        for name in ["change_detection", "flood_detection", "building_damage"]:
            v = registry.get_validation(name)
            result[name] = v.to_dict() if v else {"status": "NOT_REGISTERED"}
        print("\nJSON output:")
        print(json.dumps(result, indent=2))


def cmd_feature_status(args):
    """Show feature availability status."""
    from disaster_assessment.feature_status import check_feature_status

    registry = check_feature_status()
    print(registry.summary_table())

    if args.json:
        features = {}
        for f in registry.get_all():
            features[f.feature] = f.to_dict()
        print("\nJSON output:")
        print(json.dumps(features, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Satellite Disaster Assessment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py analyze --pre pre.tif --post post.tif --output results/
  python cli.py analyze --pre pre.jpg --post post.jpg --gsd 0.5
  python cli.py validate-models
  python cli.py validate-models --json
  python cli.py feature-status
  python cli.py feature-status --json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run disaster analysis")
    analyze_parser.add_argument("--pre", required=True, help="Path to pre-disaster image")
    analyze_parser.add_argument("--post", required=True, help="Path to post-disaster image")
    analyze_parser.add_argument("--output", "-o", help="Output directory for results")
    analyze_parser.add_argument("--gsd", type=float, help="Ground sample distance (meters/pixel)")
    analyze_parser.add_argument("--geotiff", help="Path to GeoTIFF file")
    analyze_parser.add_argument(
        "--damage-mode",
        choices=["auto", "trained", "estimated"],
        default="auto",
        help="Damage assessment mode (default: auto)",
    )

    # validate-models command
    validate_parser = subparsers.add_parser("validate-models", help="Check model validation status")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # feature-status command
    status_parser = subparsers.add_parser("feature-status", help="Show feature availability")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "validate-models":
        return cmd_validate_models(args)
    elif args.command == "feature-status":
        return cmd_feature_status(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
