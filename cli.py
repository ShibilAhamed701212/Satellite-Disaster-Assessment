#!/usr/bin/env python3
"""
Command-Line Interface (CLI) for Satellite Disaster Assessment.

Usage examples:
    # Run disaster assessment on pre/post image pair:
    python cli.py --pre sample_images/pre_disaster.png --post sample_images/post_disaster.png --gsd 0.5

    # Run single-image land-cover segmentation:
    python cli.py --single sample_images/pre_disaster.png

    # Output structured JSON report:
    python cli.py --pre pre.png --post post.png --json-out report.json
"""

import argparse
import json
import os
import sys

# Configure UTF-8 stdout/stderr for Windows console compatibility
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        _reconf = getattr(_stream, "reconfigure", None)
        if callable(_reconf):
            try:
                _reconf(encoding="utf-8")
            except Exception:
                pass

# Ensure DL-SatelliteImagery is in python path
_dl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL-SatelliteImagery")
if _dl_path not in sys.path:
    sys.path.insert(0, _dl_path)

import numpy as np
from PIL import Image

try:
    from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer
except ImportError:
    import importlib
    _mod = importlib.import_module("disaster_assessment.pipeline.disaster_analyzer")
    DisasterAnalyzer = getattr(_mod, "DisasterAnalyzer")
except OSError as e:
    if "DLL" in str(e) or "c10" in str(e) or "1114" in str(e):
        print(f"\n[Environment Error] PyTorch dynamic library failed to load in current Python environment.")
        print(f"Details: {e}")
        print("\nPlease run this script using the project's dedicated virtual environment:")
        print("  PowerShell (Windows):  .\\.venv\\Scripts\\python.exe cli.py [options]")
        print("  Or activate venv:      .\\.venv\\Scripts\\Activate.ps1; python cli.py [options]\n")
        sys.exit(1)
    raise


def main():
    parser = argparse.ArgumentParser(
        description="🛰️ Satellite Disaster Assessment & Remote Sensing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--pre", type=str, help="Path to PRE-disaster satellite image (PNG/JPG/TIF)")
    parser.add_argument("--post", type=str, help="Path to POST-disaster satellite image (PNG/JPG/TIF)")
    parser.add_argument("--single", type=str, help="Path to single image for land-cover segmentation")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "trained", "estimated"], help="Damage assessment mode (default: auto)")
    parser.add_argument("--gsd", type=float, default=None, help="Ground Sample Distance in meters/pixel (e.g., 0.5 for WorldView-3, 250.0 for MODIS)")
    parser.add_argument("--geotiff", type=str, default=None, help="Optional path to GeoTIFF raster file for automatic CRS/GSD extraction")
    parser.add_argument("--json-out", type=str, default=None, help="Path to save output report as structured JSON")
    parser.add_argument("--save-vis", type=str, default=None, help="Path to save colorized visualization overlay image (PNG/JPG)")

    args = parser.parse_args()

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DL-SatelliteImagery", "disaster_assessment", "weights")
    damage_weights = os.path.join(weights_dir, "building_damage", "best_damage_model.pth")
    change_weights = os.path.join(weights_dir, "siamese_unet.pth")
    flood_weights = os.path.join(weights_dir, "flood_unet.pth")

    analyzer = DisasterAnalyzer(
        trained_damage_weights=damage_weights if os.path.isfile(damage_weights) else None,
        change_model_weights=change_weights if os.path.isfile(change_weights) else None,
        flood_model_weights=flood_weights if os.path.isfile(flood_weights) else None,
    )

    # 1. Single-image land-cover segmentation mode
    if args.single:
        if not os.path.isfile(args.single):
            print(f"Error: File not found: {args.single}")
            sys.exit(1)
        print(f"[CLI] Running Land-Cover Segmentation on: {args.single}")
        img = np.array(Image.open(args.single).convert("RGB"))
        results = analyzer.analyze_single(img)
        print("\n" + results.get("summary", "✓ Segmentation complete."))

        if args.save_vis:
            out_img = Image.fromarray(results["overlay"])
            out_img.save(args.save_vis)
            print(f"\n[CLI] Visualization overlay saved to: {args.save_vis}")

        if args.json_out:
            report_data = {
                "image_path": args.single,
                "total_pixels": results.get("total_pixels", 0),
                "class_percentages": results.get("class_percentages", {}),
                "class_counts": results.get("class_counts", {}),
            }
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)
            print(f"[CLI] JSON report saved to: {args.json_out}")
        return

    # 2. Pre/Post disaster pair analysis mode
    if args.pre or args.post:
        if not args.pre or not args.post:
            print("Error: Both --pre and --post images are required for disaster assessment.")
            print("       For single image land-cover segmentation, use --single <image> instead.")
            sys.exit(1)

        if not os.path.isfile(args.pre):
            print(f"Error: PRE file not found: {args.pre}")
            sys.exit(1)
        if not os.path.isfile(args.post):
            print(f"Error: POST file not found: {args.post}")
            sys.exit(1)

        print(f"[CLI] Loading PRE image:  {args.pre}")
        print(f"[CLI] Loading POST image: {args.post}")
        pre_img = np.array(Image.open(args.pre).convert("RGB"))
        post_img = np.array(Image.open(args.post).convert("RGB"))

        print(f"[CLI] Running Disaster Assessment (Mode: {args.mode}, GSD: {args.gsd} m/px)...")
        report = analyzer.analyze(
            pre_image=pre_img,
            post_image=post_img,
            damage_mode=args.mode,
            meters_per_pixel=args.gsd,
            geotiff_path=args.geotiff,
        )

        print("\n" + report.summary())

        if args.save_vis:
            # Prefer damage overlay, fallback to heatmap or change map
            vis_map = report.maps.get("building_damage")
            if vis_map is None:
                vis_map = report.maps.get("disaster_heatmap")
            if vis_map is None:
                vis_map = report.maps.get("change_map")

            if vis_map is not None:
                out_img = Image.fromarray(vis_map)
                out_img.save(args.save_vis)
                print(f"\n[CLI] Visualization overlay saved to: {args.save_vis}")

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"[CLI] JSON report saved to: {args.json_out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
