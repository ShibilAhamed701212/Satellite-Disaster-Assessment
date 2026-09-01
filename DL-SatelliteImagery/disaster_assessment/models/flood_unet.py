"""
Lightweight U-Net for binary flood segmentation.

Architecture:
    - Lightweight CNN encoder (16→32→64→128→256)
    - U-Net decoder with skip connections
    - Binary output: 0 = Non-flooded, 1 = Flooded

Input:  256×256×3 satellite image
Output: 256×256×1 binary flood mask

Total parameters: ~1.9M
Peak VRAM (inference, fp32): ~8MB
"""

import torch
import torch.nn as nn

from .base_unet import LightweightEncoder, LightweightDecoder, get_device, count_parameters


class FloodUNet(nn.Module):
    """
    Standard lightweight U-Net for binary flood segmentation.

    Single-image input, binary flood mask output.
    Independent from the existing land-cover U-Net — does not share weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = LightweightEncoder(in_channels=in_channels, dropout=dropout)
        encoder_channels = self.encoder.channels  # [16, 32, 64, 128, 256]

        self.decoder = LightweightDecoder(
            out_channels=1,
            encoder_channels=encoder_channels,
            dropout=dropout,
            skip_channel_multiplier=1,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Satellite image tensor [B, 3, H, W] (normalized 0-1).

        Returns:
            Flood probability map [B, 1, H, W] with sigmoid activation.
        """
        bottleneck, skip_connections = self.encoder(image)
        logits = self.decoder(bottleneck, skip_connections)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Run inference and return a binary flood mask.

        Args:
            image:     [B, 3, H, W] normalized tensor.
            threshold: Probability threshold for binary classification.

        Returns:
            Binary mask [B, 1, H, W] with values 0 or 1.
        """
        self.eval()
        probs = self.forward(image)
        return (probs >= threshold).float()


def create_flood_unet(
    in_channels: int = 3,
    dropout: float = 0.2,
    weights_path: str = None,
    device: torch.device = None,
) -> FloodUNet:
    """
    Factory function to create and optionally load weights for the Flood U-Net.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        dropout: Dropout probability.
        weights_path: Optional path to pre-trained weights (.pth).
        device: Target device. Auto-detected if None.

    Returns:
        FloodUNet model on the specified device.
    """
    if device is None:
        device = get_device()

    model = FloodUNet(in_channels=in_channels, dropout=dropout)

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"[FloodUNet] Loaded weights from: {weights_path}")
    else:
        print(
            "[FloodUNet] No weights loaded — model initialized with random weights.\n"
            "  To use for real predictions, train the model first or provide weights_path."
        )

    model = model.to(device)
    param_count = count_parameters(model)
    print(f"[FloodUNet] Parameters: {param_count:,} | Device: {device}")

    return model
