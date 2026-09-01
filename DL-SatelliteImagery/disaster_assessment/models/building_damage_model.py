"""
Trained Building Damage Assessment Model.

Architecture:
    - Siamese Dual-Encoder Lightweight U-Net
    - Processes PRE and POST disaster images simultaneously
    - Multi-scale feature fusion at all encoder levels:
      [pre_feat, post_feat, |pre_feat - post_feat|]
    - 4-class segmentation output:
      0 = Undamaged
      1 = Minor Damage
      2 = Major Damage
      3 = Destroyed

Total parameters: ~3.8M
VRAM Footprint: ~15MB (fp32) / ~7.5MB (fp16)
Optimized for NVIDIA GTX 1650 4GB VRAM.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_unet import ConvBlock, LightweightEncoder, count_parameters, get_device

# 4 Discrete Building Damage Classes
DAMAGE_CLASSES: Dict[int, str] = {
    0: "Undamaged",
    1: "Minor Damage",
    2: "Major Damage",
    3: "Destroyed",
}

NUM_DAMAGE_CLASSES = len(DAMAGE_CLASSES)


class FusedSkipDecoder(nn.Module):
    """
    U-Net Decoder designed to ingest 3x concatenated skip features
    (pre + post + absolute_difference).
    """

    def __init__(
        self,
        num_classes: int = 4,
        encoder_channels: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256]

        self.channels = encoder_channels

        # Bottleneck fusion: in = 256*3, out = 256
        self.bottleneck_fusion = ConvBlock(self.channels[4] * 3, self.channels[4], dropout)

        # Upsampling path with 3x skip connections
        self.up4 = nn.ConvTranspose2d(self.channels[4], self.channels[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(self.channels[3] + (self.channels[3] * 3), self.channels[3], dropout)

        self.up3 = nn.ConvTranspose2d(self.channels[3], self.channels[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(self.channels[2] + (self.channels[2] * 3), self.channels[2], dropout)

        self.up2 = nn.ConvTranspose2d(self.channels[2], self.channels[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(self.channels[1] + (self.channels[1] * 3), self.channels[1], dropout)

        self.up1 = nn.ConvTranspose2d(self.channels[1], self.channels[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(self.channels[0] + (self.channels[0] * 3), self.channels[0], dropout)

        self.final_conv = nn.Conv2d(self.channels[0], num_classes, kernel_size=1)

    def forward(
        self,
        fused_bottleneck: torch.Tensor,
        fused_skips: List[torch.Tensor],
    ) -> torch.Tensor:
        s1, s2, s3, s4 = fused_skips

        bn = self.bottleneck_fusion(fused_bottleneck)

        d4 = self.up4(bn)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


class BuildingDamageUNet(nn.Module):
    """
    Siamese 4-Class Building Damage Segmentation Network.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Shared Siamese encoder for pre/post imagery
        self.encoder = LightweightEncoder(in_channels=in_channels, dropout=dropout)

        # Multi-scale fused decoder
        self.decoder = FusedSkipDecoder(
            num_classes=num_classes,
            encoder_channels=self.encoder.channels,
            dropout=dropout,
        )

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pre_image:  [B, 3, H, W] normalized pre-disaster tensor.
            post_image: [B, 3, H, W] normalized post-disaster tensor.

        Returns:
            Logits tensor [B, 4, H, W] for the 4 damage classes.
        """
        pre_bn, pre_skips = self.encoder(pre_image)
        post_bn, post_skips = self.encoder(post_image)

        # Fuse skip connections: [pre, post, |pre - post|]
        fused_skips = [
            torch.cat([s_pre, s_post, torch.abs(s_pre - s_post)], dim=1)
            for s_pre, s_post in zip(pre_skips, post_skips)
        ]

        # Fuse bottleneck
        fused_bn = torch.cat([pre_bn, post_bn, torch.abs(pre_bn - post_bn)], dim=1)

        # Decode to 4-class logits
        logits = self.decoder(fused_bn, fused_skips)
        return logits

    @torch.no_grad()
    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference and return class indices (0-3) and softmax probabilities.

        Returns:
            class_map: [B, H, W] integer tensor (values 0..3)
            probs:     [B, 4, H, W] softmax probability tensor
        """
        self.eval()
        logits = self.forward(pre_image, post_image)
        probs = F.softmax(logits, dim=1)
        class_map = torch.argmax(probs, dim=1)
        return class_map, probs


def create_building_damage_model(
    in_channels: int = 3,
    num_classes: int = 4,
    dropout: float = 0.2,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> BuildingDamageUNet:
    """
    Factory function to instantiate and optionally load weights for BuildingDamageUNet.
    """
    if device is None:
        device = get_device()

    model = BuildingDamageUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout=dropout,
    )

    if weights_path is not None:
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception:
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        # Handle state_dict wrapped in a checkpoint dictionary
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"[BuildingDamageUNet] Loaded trained weights from: {weights_path}")
    else:
        print(
            "[BuildingDamageUNet] No weights loaded — model initialized with random weights.\n"
            "  Use train_building_damage.py to train on xBD or custom damage datasets."
        )

    model = model.to(device)
    param_count = count_parameters(model)
    print(f"[BuildingDamageUNet] Parameters: {param_count:,} | Device: {device}")

    return model
