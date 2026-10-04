"""
Unified Gradio Interface for Satellite Disaster Assessment (Phase 1 & Phase 2).

Extends the existing Gradio application with:
    1. MODE 1 — Single Image Land-Cover Analysis (6-class segmentation)
    2. MODE 2 — Full Disaster Assessment:
       - Damage Assessment Modes: [Auto, Trained (4-Class), Estimated]
       - Georeferenced Flood Area Calculation (GeoTIFF / meters_per_pixel)
       - Visualizations: Change Map, Flood Map, Building Damage, Heatmap
       - Honest model status reporting
    3. MODE 3 — Quick Change Detection
    4. Feature Status Panel

Launch:
    cd DL-SatelliteImagery
    python disaster_gradio_app.py
"""

import os
import sys

import gradio as gr
import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from disaster_assessment.pipeline.disaster_analyzer import DisasterAnalyzer

# ============================================================
# Initialize Analyzer with available model weights
# ============================================================
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "disaster_assessment", "weights")
CHANGE_WEIGHTS = os.path.join(WEIGHTS_DIR, "siamese_unet.pth")
FLOOD_WEIGHTS = os.path.join(WEIGHTS_DIR, "flood_unet.pth")
DAMAGE_WEIGHTS = os.path.join(WEIGHTS_DIR, "building_damage", "best_damage_model.pth")

analyzer = DisasterAnalyzer(
    change_model_weights=CHANGE_WEIGHTS if os.path.isfile(CHANGE_WEIGHTS) else None,
    flood_model_weights=FLOOD_WEIGHTS if os.path.isfile(FLOOD_WEIGHTS) else None,
    trained_damage_weights=DAMAGE_WEIGHTS if os.path.isfile(DAMAGE_WEIGHTS) else None,
    landcover_model_path=None,
)


# Sample imagery lives at the repository root, one level above this file.
SAMPLE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_images"))


def _upload_path(uploaded):
    """Return the local path of a gr.File upload.

    Gradio 4+ passes a filepath string; Gradio 3 passed a tempfile wrapper
    with a ``.name`` attribute.
    """
    if uploaded is None:
        return None
    if isinstance(uploaded, (str, os.PathLike)):
        return os.fspath(uploaded)
    return getattr(uploaded, "name", None)


# ============================================================
# Feature Status Function
# ============================================================
def get_feature_status():
    """Get honest feature status for display."""
    validations = analyzer.validate_models()

    lines = [
        "=" * 60,
        "  MODEL & FEATURE STATUS REPORT",
        "=" * 60,
        "",
    ]

    # Land cover
    lc_mode = "HSV COLOR HEURISTIC (not a trained model)"
    lines.append("  [HEURISTIC] Land Cover Segmentation")
    lines.append(f"      Mode: {lc_mode}")
    lines.append("      Note: Results are approximate. Not AI-based.")
    lines.append("")

    # Change detection
    cd_val = validations.get("change_detection", {})
    cd_status = cd_val.get("status", "UNKNOWN")
    cd_mode = cd_val.get("inference_mode", "unknown")
    lines.append(f"  [{'OK' if cd_mode == 'trained' else '--'}] Change Detection")
    lines.append(f"      Status: {cd_status}")
    lines.append(f"      Mode: {cd_mode}")
    if cd_mode != "trained":
        lines.append("      Note: No validated trained checkpoint. Output is UNAVAILABLE.")
    lines.append("")

    # Flood detection
    fd_val = validations.get("flood_detection", {})
    fd_status = fd_val.get("status", "UNKNOWN")
    fd_mode = fd_val.get("inference_mode", "unknown")
    lines.append(f"  [{'OK' if fd_mode == 'trained' else '--'}] Flood Detection")
    lines.append(f"      Status: {fd_status}")
    lines.append(f"      Mode: {fd_mode}")
    if fd_mode != "trained":
        lines.append("      Note: No validated trained checkpoint. Output is UNAVAILABLE.")
    lines.append("")

    # Building damage
    bd_val = validations.get("building_damage", {})
    bd_status = bd_val.get("status", "UNKNOWN")
    bd_mode = bd_val.get("inference_mode", "unknown")
    lines.append(f"  [{'OK' if bd_mode == 'trained' else 'EST'}] Building Damage")
    lines.append(f"      Status: {bd_status}")
    lines.append(f"      Mode: {bd_mode}")
    if bd_mode != "trained":
        lines.append("      Fallback: Structural change heuristic (estimated)")
    else:
        meta = bd_val.get("metadata", {})
        if meta.get("epoch", -1) >= 0:
            lines.append(f"      Epoch: {meta['epoch']}")
        if meta.get("metrics"):
            lines.append(f"      Metrics: {meta['metrics']}")
    lines.append("")

    # Additional features
    lines.extend([
        "  [--] GeoTIFF Tiling: Architecture ready, not connected to pipeline",
        "  [--] Multispectral Indices: Architecture ready, not connected to pipeline",
        "  [--] SAR Processing: Architecture ready, requires real SAR data",
        "  [--] DEM Analysis: Architecture ready, requires DEM input",
        "  [--] GIS Vector Export: Architecture ready, not connected to pipeline",
        "  [--] Digital Twin: In-memory state, no persistence",
        "  [--] RAG Copilot: Architecture ready, requires LLM provider",
        "  [--] Data Ingestion: Architecture ready, requires API credentials",
        "",
        "=" * 60,
        "  IMPORTANT: Change Detection and Flood Detection outputs",
        "  are UNAVAILABLE in this configuration. The system uses",
        "  heuristic estimates where applicable. No AI models are",
        "  currently producing meaningful predictions for these tasks.",
        "=" * 60,
    ])

    return "\n".join(lines)


# ============================================================
# Mode 1: Single Image Land-Cover Segmentation
# ============================================================
def analyze_single_image(image):
    if image is None:
        return None, None, "Please upload an image."

    try:
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        results = analyzer.analyze_single(image)
        mode = results.get("mode", "unknown")
        mode_label = "TRAINED MODEL" if mode == "trained" else "HSV HEURISTIC (not AI)"

        info_text = results.get("summary", "Land-Cover Segmentation Complete")
        info_text += f"\n\nMethod: {mode_label}"
        if mode == "heuristic":
            info_text += "\nNote: This is a color-space heuristic, not a trained neural network."

        return results["overlay"], results["segmentation"], info_text

    except Exception as e:
        return None, None, f"Error: {str(e)}"


# ============================================================
# Mode 2: Full Disaster Assessment (Phase 2 Enhanced)
# ============================================================
def analyze_disaster(
    pre_image,
    post_image,
    damage_mode_choice: str = "Auto",
    meters_per_pixel_val: float = 0.0,
    geotiff_file=None,
):
    if pre_image is None or post_image is None:
        empty = np.zeros((256, 256, 3), dtype=np.uint8)
        return (
            empty, empty, empty, empty,
            empty, empty, empty, empty,
            "Please upload both PRE and POST disaster images."
        )

    try:
        if not isinstance(pre_image, np.ndarray):
            pre_image = np.array(pre_image)
        if not isinstance(post_image, np.ndarray):
            post_image = np.array(post_image)

        # Parse ground resolution input
        gsd = float(meters_per_pixel_val) if meters_per_pixel_val and float(meters_per_pixel_val) > 0 else None

        # Parse GeoTIFF path if provided
        geotiff_path = _upload_path(geotiff_file)

        mode_str = damage_mode_choice.lower()

        report = analyzer.analyze(
            pre_image=pre_image,
            post_image=post_image,
            damage_mode=mode_str,
            meters_per_pixel=gsd,
            geotiff_path=geotiff_path,
        )

        maps = report.maps
        pre_lc = maps.get("pre_landcover", np.zeros((256, 256, 3), dtype=np.uint8))
        post_lc = maps.get("post_landcover", np.zeros((256, 256, 3), dtype=np.uint8))
        change_map = maps.get("change_map", np.zeros((256, 256, 3), dtype=np.uint8))
        flood_map = maps.get("flood_map", np.zeros((256, 256, 3), dtype=np.uint8))
        damage_map = maps.get("building_damage", np.zeros((256, 256, 3), dtype=np.uint8))
        heatmap = maps.get("disaster_heatmap", np.zeros((256, 256, 3), dtype=np.uint8))

        # Build detailed formatted metrics string with honest labels
        severity_level = report.severity.severity_level if report.severity else "N/A"
        severity_score = report.severity.total_score if report.severity else 0

        severity_icon = {
            "LOW": "[LOW]",
            "MODERATE": "[MOD]",
            "HIGH": "[HIGH]",
            "CRITICAL": "[CRIT]",
        }.get(severity_level, "[?]")

        # Land cover mode
        lc_mode_label = "TRAINED MODEL" if report.landcover_mode == "trained" else "HSV HEURISTIC"

        # Building damage text
        dmg_mode_label = "TRAINED NEURAL MODEL (4 Classes)" if report.damage_mode == "trained" else "ESTIMATED STRUCTURAL CHANGE"
        dmg_text_lines = [f"  Building Damage Source: [{dmg_mode_label}]"]

        if report.damage_output is not None and report.damage_mode == "trained":
            dmg_text_lines.extend([
                f"    * Undamaged:       {report.damage_output.undamaged_percentage:>6.2f}%",
                f"    * Minor Damage:    {report.damage_output.minor_damage_percentage:>6.2f}%",
                f"    * Major Damage:    {report.damage_output.major_damage_percentage:>6.2f}%",
                f"    * Destroyed:       {report.damage_output.destroyed_percentage:>6.2f}%",
                f"    * Score:           {report.damage_output.overall_damage_score:.1f}/100",
            ])
        elif report.damage_output is not None and report.damage_output.estimated_details is not None:
            est = report.damage_output.estimated_details
            dmg_text_lines.extend([
                f"    * Undamaged:       {est.undamaged_percentage:>6.2f}%",
                f"    * Possible Change: {est.possible_damage_percentage:>6.2f}%",
                f"    * Severe Change:   {est.severe_damage_percentage:>6.2f}%",
                f"    * Score:           {est.overall_damage_score:.1f}/100",
            ])

        # Flood area text with honest mode
        flood_mode_label = report.flood_mode.upper()
        change_mode_label = report.change_mode.upper()

        flood_text_lines = [
            f"  Flood Assessment [{flood_mode_label}]:",
            f"    * Flooded Area:    {report.flood_percentage:.2f}%",
        ]
        if report.flood_area_m2 is not None:
            flood_text_lines.extend([
                f"    * Metric Area:     {report.flood_area_m2:,.2f} m2",
                f"    * Metric Area:     {report.flood_area_km2:,.6f} km2",
            ])
        flood_text_lines.append(f"    * Status:          {report.area_status}")

        metrics_text = (
            f"{'=' * 60}\n"
            f"  SATELLITE DISASTER ASSESSMENT REPORT\n"
            f"{'=' * 60}\n\n"
            f"  {severity_icon} Overall Severity: {severity_score:.1f}/100 -- {severity_level}\n\n"
            f"  Land Cover Method: {lc_mode_label}\n"
            f"  Change Detection: [{change_mode_label}]\n"
            + "\n".join(dmg_text_lines)
            + "\n\n"
            + "\n".join(flood_text_lines)
            + "\n\n"
            f"  Additional Land Cover Dynamics:\n"
            f"    * Total Changed Area:  {report.change_percentage:.2f}%\n"
            f"    * Vegetation Loss:     {report.vegetation_loss_percentage:.2f}%\n"
            f"    * Water Expansion:     {report.water_expansion_percentage:.2f}%\n\n"
            f"  Measurement: {report.measurement_basis}\n"
        )

        if report.warnings:
            metrics_text += "\n  System Notes & Warnings:\n"
            for w in report.warnings:
                metrics_text += f"    * {w}\n"

        return (
            pre_lc, post_lc,
            change_map, flood_map,
            damage_map, heatmap,
            maps.get("vegetation_loss", np.zeros((256, 256, 3), dtype=np.uint8)),
            maps.get("post_image", np.zeros((256, 256, 3), dtype=np.uint8)),
            metrics_text,
        )

    except Exception as e:
        import traceback
        empty = np.zeros((256, 256, 3), dtype=np.uint8)
        return (
            empty, empty, empty, empty,
            empty, empty, empty, empty,
            f"Error during analysis: {str(e)}\n{traceback.format_exc()}"
        )


# ============================================================
# Mode 3: Quick Change Detection
# ============================================================
def quick_change_detection(pre_image, post_image):
    if pre_image is None or post_image is None:
        return None, "Please upload both images."

    try:
        import torch
        from disaster_assessment.preprocessing.image_alignment import ImageAligner
        from disaster_assessment.preprocessing.image_validation import ImageValidator

        if not isinstance(pre_image, np.ndarray):
            pre_image = np.array(pre_image)
        if not isinstance(post_image, np.ndarray):
            post_image = np.array(post_image)

        pre_image = ImageValidator.normalize_image(pre_image)
        post_image = ImageValidator.normalize_image(post_image)

        aligner = ImageAligner(target_size=(256, 256))
        pre_aligned, post_aligned, _ = aligner.align(pre_image, post_image)

        # Validate model first
        if not analyzer._load_change_model():
            overlay = analyzer.overlay_renderer.change_overlay(
                post_aligned, np.zeros((256, 256), dtype=np.uint8)
            )
            info = (
                "Change detection is UNAVAILABLE.\n\n"
                "No validated trained checkpoint is configured.\n"
                "Train the SiameseUNet model or provide trained weights."
            )
            return overlay, info

        analyzer._load_change_model()
        pre_tensor = analyzer._numpy_to_tensor(pre_aligned)
        post_tensor = analyzer._numpy_to_tensor(post_aligned)

        with torch.no_grad():
            change_probs = analyzer._change_model(pre_tensor, post_tensor)
            change_mask = (change_probs >= 0.5).float()

        change_np = change_mask.squeeze().cpu().numpy().astype(np.uint8)
        overlay = analyzer.overlay_renderer.change_overlay(post_aligned, change_np)

        changed_pct = (change_np.sum() / change_np.size) * 100
        info = (
            f"Change Detection Complete [TRAINED MODEL]\n\n"
            f"Changed pixels: {change_np.sum():,} / {change_np.size:,}\n"
            f"Changed area:   {changed_pct:.2f}%\n\n"
            f"Red regions indicate detected surface changes."
        )

        return overlay, info

    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================
# Build Gradio UI
# ============================================================
def create_app():
    with gr.Blocks(title="Satellite Disaster Assessment System") as app:
        gr.Markdown(
            """
            # Satellite Disaster Assessment & Analysis System

            **Important:** Some features use heuristic estimation, not trained AI models.
            See the Feature Status tab for honest capability reporting.

            ---
            """
        )

        with gr.Tabs():
            # ==================== TAB 1 ====================
            with gr.TabItem("Land-Cover Segmentation"):
                gr.Markdown(
                    "### Single Image Land-Cover Analysis\n"
                    "Upload a satellite image to segment it into 6 land-cover classes: "
                    "Water, Land, Road, Building, Vegetation, Unlabeled.\n\n"
                    "**Note:** Uses HSV color-space heuristic. Not a trained neural network."
                )

                with gr.Row():
                    with gr.Column():
                        single_input = gr.Image(label="Upload Satellite Image", type="numpy")
                        single_btn = gr.Button("Analyze Land Cover", variant="primary")

                    with gr.Column():
                        single_overlay = gr.Image(label="Segmentation Overlay")
                        single_colorized = gr.Image(label="Land Cover Map")
                        single_info = gr.Textbox(label="Analysis Info", lines=15, interactive=False)

                single_btn.click(
                    fn=analyze_single_image,
                    inputs=[single_input],
                    outputs=[single_overlay, single_colorized, single_info],
                )

                # Tab 1 Examples
                sample_root = SAMPLE_ROOT
                lc_dir = os.path.join(sample_root, "landcover_tiles")
                tab1_samples = [
                    [os.path.join(lc_dir, "satellite_urban_city.jpg")],
                    [os.path.join(lc_dir, "satellite_water_coastal.jpg")],
                    [os.path.join(lc_dir, "satellite_forest_canopy.jpg")],
                    [os.path.join(lc_dir, "satellite_agriculture_farmland.jpg")],
                ]
                tab1_valid_samples = [s for s in tab1_samples if os.path.isfile(s[0])]
                if tab1_valid_samples:
                    gr.Examples(
                        examples=tab1_valid_samples,
                        inputs=[single_input],
                        label="Sample Land-Cover Scenes",
                    )

            # ==================== TAB 2 ====================
            with gr.TabItem("Disaster Assessment"):
                gr.Markdown(
                    "### Comprehensive Multi-Hazard Disaster Analysis\n"
                    "Upload PRE and POST disaster satellite images to evaluate building damage, "
                    "flood spread, and severity.\n\n"
                    "**Warning:** Change detection and flood detection models are currently "
                    "UNAVAILABLE (no validated trained weights). Results for these components "
                    "will be zero."
                )

                with gr.Row():
                    pre_input = gr.Image(label="PRE-Disaster Image", type="numpy")
                    post_input = gr.Image(label="POST-Disaster Image", type="numpy")

                with gr.Accordion("Advanced Geospatial & Damage Configuration", open=True):
                    with gr.Row():
                        damage_mode_ui = gr.Radio(
                            choices=["Auto", "Trained", "Estimated"],
                            value="Auto",
                            label="Damage Assessment Mode",
                            info="Auto: Uses trained model if validated; otherwise falls back to estimated.",
                        )
                        gsd_ui = gr.Number(
                            label="Ground Resolution (Meters/Pixel, optional)",
                            value=None,
                            info="e.g., 0.5 for WorldView-3, 250.0 for MODIS.",
                        )
                        geotiff_ui = gr.File(
                            label="Optional GeoTIFF Raster File",
                            file_types=[".tif", ".tiff"],
                        )

                disaster_btn = gr.Button("ANALYZE DISASTER", variant="primary", size="lg")

                gr.Markdown("#### Visual Assessment Outputs")

                with gr.Row():
                    pre_lc_out = gr.Image(label="PRE Land Cover")
                    post_lc_out = gr.Image(label="POST Land Cover")
                    change_out = gr.Image(label="Change Detection Map")
                    flood_out = gr.Image(label="Flood Inundation Map")

                with gr.Row():
                    damage_out = gr.Image(label="Building Damage Overlay")
                    heatmap_out = gr.Image(label="Disaster Heatmap")
                    veg_loss_out = gr.Image(label="Vegetation Loss Overlay")
                    post_ref_out = gr.Image(label="POST Reference Image")

                metrics_out = gr.Textbox(
                    label="Disaster Metrics, Geospatial Area & Severity",
                    lines=30,
                    interactive=False,
                )

                disaster_btn.click(
                    fn=analyze_disaster,
                    inputs=[pre_input, post_input, damage_mode_ui, gsd_ui, geotiff_ui],
                    outputs=[
                        pre_lc_out, post_lc_out,
                        change_out, flood_out,
                        damage_out, heatmap_out,
                        veg_loss_out, post_ref_out,
                        metrics_out,
                    ],
                )

                # Tab 2 Examples
                dp_dir = os.path.join(sample_root, "disaster_pairs")
                tab2_samples = [
                    [
                        os.path.join(sample_root, "pre_disaster.png"),
                        os.path.join(sample_root, "post_disaster.png"),
                        "Auto",
                        250.0,
                    ],
                    [
                        os.path.join(dp_dir, "bushfire_pre.jpg"),
                        os.path.join(dp_dir, "bushfire_post.jpg"),
                        "Auto",
                        250.0,
                    ],
                    [
                        os.path.join(dp_dir, "hurricane_pre.jpg"),
                        os.path.join(dp_dir, "hurricane_post.jpg"),
                        "Auto",
                        250.0,
                    ],
                ]
                tab2_valid_samples = [s for s in tab2_samples if os.path.isfile(s[0]) and os.path.isfile(s[1])]
                if tab2_valid_samples:
                    gr.Examples(
                        examples=tab2_valid_samples,
                        inputs=[pre_input, post_input, damage_mode_ui, gsd_ui],
                        label="Sample Disaster Events",
                    )

            # ==================== TAB 3 ====================
            with gr.TabItem("Change Detection"):
                gr.Markdown(
                    "### Quick Surface Change Detection\n"
                    "Upload two satellite images to detect changes between them.\n\n"
                    "**Status:** Requires validated trained SiameseUNet weights. "
                    "Currently UNAVAILABLE in this configuration."
                )

                with gr.Row():
                    cd_pre = gr.Image(label="Image 1 (Before)", type="numpy")
                    cd_post = gr.Image(label="Image 2 (After)", type="numpy")

                cd_btn = gr.Button("Detect Changes", variant="primary")

                with gr.Row():
                    cd_result = gr.Image(label="Change Detection Result")
                    cd_info = gr.Textbox(label="Detection Info", lines=10, interactive=False)

                cd_btn.click(
                    fn=quick_change_detection,
                    inputs=[cd_pre, cd_post],
                    outputs=[cd_result, cd_info],
                )

                # Tab 3 Examples
                tab3_samples = [
                    [
                        os.path.join(sample_root, "pre_disaster.png"),
                        os.path.join(sample_root, "post_disaster.png"),
                    ],
                    [
                        os.path.join(dp_dir, "deforestation_pre.jpg"),
                        os.path.join(dp_dir, "deforestation_post.jpg"),
                    ],
                ]
                tab3_valid_samples = [s for s in tab3_samples if os.path.isfile(s[0]) and os.path.isfile(s[1])]
                if tab3_valid_samples:
                    gr.Examples(
                        examples=tab3_valid_samples,
                        inputs=[cd_pre, cd_post],
                        label="Sample Change Detection Pairs",
                    )

            # ==================== TAB 4 ====================
            with gr.TabItem("Feature Status"):
                gr.Markdown(
                    "### Model & Feature Status\n"
                    "This tab shows the actual operational status of every component."
                )

                status_btn = gr.Button("Refresh Status", variant="secondary")
                status_output = gr.Textbox(
                    label="Feature Status Report",
                    lines=40,
                    interactive=False,
                )

                status_btn.click(
                    fn=get_feature_status,
                    inputs=[],
                    outputs=[status_output],
                )

                # Auto-load status on tab visit
                app.load(
                    fn=get_feature_status,
                    inputs=[],
                    outputs=[status_output],
                )

        gr.Markdown(
            """
            ---
            *Architecture: PyTorch Siamese U-Net + BuildingDamageUNet (4-Class) + Geospatial Raster Engine.*
            *Note: Some features use heuristic estimation, not trained AI models.*
            """
        )

    return app


if __name__ == "__main__":
    app = create_app()
    import socket

    def get_open_port(default_port=7860):
        for port in range(default_port, default_port + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', port)) != 0:
                    return port
        return None

    selected_port = get_open_port(7860) or 7860
    print(f"Starting Gradio on port {selected_port}...")
    app.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=selected_port,
        share=False,
        show_error=True,
    )
