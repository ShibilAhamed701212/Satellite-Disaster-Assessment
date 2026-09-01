"""
Foundation model backbone adapters.

Status: IMPLEMENTED ARCHITECTURE
No foundation model weights are downloaded automatically.
Requires explicit setup and weight download.
"""

from typing import List, Optional

from .registry import register_model


class SegFormerAdapter:
    """Adapter for SegFormer-style transformers.

    Status: IMPLEMENTED ARCHITECTURE
    Requires: timm, segformer pretrained weights
    """
    name = "segformer"
    channels = [64, 128, 256, 512]
    params = "~27M (MiT-B2)"
    available = False
    description = "SegFormer MiT-B2 encoder (requires timm + pretrained weights)"
    download_url = None  # Not auto-downloaded


class SwinBackboneAdapter:
    """Adapter for Swin Transformer backbone.

    Status: IMPLEMENTED ARCHITECTURE
    Requires: timm, swin pretrained weights
    """
    name = "swin"
    channels = [96, 192, 384, 768]
    params = "~28M (Swin-T)"
    available = False
    description = "Swin Transformer Tiny encoder (requires timm + pretrained weights)"
    download_url = None


class PrithviAdapter:
    """Adapter for NASA/IBM Prithvi EO foundation model.

    Status: IMPLEMENTED ARCHITECTURE
    Requires: Specialized weights from HuggingFace
    """
    name = "prithvi"
    channels = [128, 256, 512, 1024]
    params = "~100M"
    available = False
    description = "NASA/IBM Prithvi EO foundation model (requires HuggingFace access)"
    download_url = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M"


# Register them (they'll report unavailable)
@register_model("segformer")
class SegFormerRegistration:
    available = False
    description = SegFormerAdapter.description

@register_model("swin")
class SwinRegistration:
    available = False
    description = SwinBackboneAdapter.description

@register_model("prithvi")
class PrithviRegistration:
    available = False
    description = PrithviAdapter.description
