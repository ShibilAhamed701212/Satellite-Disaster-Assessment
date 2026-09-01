"""
SAR + Optical Fusion U-Net for multi-modal disaster segmentation.

Architecture:
    - Optical encoder (processes RGB/multispectral imagery)
    - SAR encoder (processes VV/VH backscatter)
    - Feature-level fusion at multiple scales
    - Decoder produces segmentation/change mask

Status: IMPLEMENTED ARCHITECTURE
Requires trained weights for production inference.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_unet import ConvBlock, LightweightEncoder, count_parameters, get_device


class SAROpticalFusionUNet(nn.Module):
    """Multi-modal SAR + Optical fusion U-Net.

    Supports two fusion strategies:
    1. Early fusion: concatenate SAR and optical at input level
    2. Feature fusion: process separately, fuse at each encoder level

    For GTX 1650 (4GB VRAM), uses lightweight encoder channels.
    """

    def __init__(
        self,
        optical_channels: int = 3,
        sar_channels: int = 2,  # VV + VH
        num_classes: int = 1,
        fusion_type: str = "feature",  # "early" or "feature"
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        if fusion_type == "early":
            # Concatenate optical + SAR at input
            total_channels = optical_channels + sar_channels
            self.optical_encoder = LightweightEncoder(in_channels=total_channels, dropout=dropout)
            self.sar_encoder = None
            encoder_channels = self.optical_encoder.channels
        else:
            # Separate encoders
            self.optical_encoder = LightweightEncoder(in_channels=optical_channels, dropout=dropout)
            self.sar_encoder = LightweightEncoder(in_channels=sar_channels, dropout=dropout)
            encoder_channels = self.optical_encoder.channels

            # Feature fusion layers at each scale
            self.fusion_layers = nn.ModuleList([
                nn.Conv2d(ch * 2, ch, kernel_size=1) for ch in encoder_channels
            ])

        # Decoder
        self.decoder = _FusionDecoder(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            optical: [B, C_opt, H, W] optical image.
            sar: [B, C_sar, H, W] SAR data (VV/VH).

        Returns:
            [B, num_classes, H, W] logits.
        """
        if self.fusion_type == "early":
            # Concatenate inputs
            fused_input = torch.cat([optical, sar], dim=1)
            bottleneck, skips = self.optical_encoder(fused_input)
            return self.decoder(bottleneck, skips)

        else:
            # Separate encoders
            opt_bn, opt_skips = self.optical_encoder(optical)
            sar_bn, sar_skips = self.sar_encoder(sar)

            # Fuse features at each level
            fused_skips = []
            for i, (opt_s, sar_s) in enumerate(zip(opt_skips, sar_skips)):
                cat = torch.cat([opt_s, sar_s], dim=1)
                fused_skips.append(self.fusion_layers[i](cat))

            # Fuse bottleneck
            fused_bn = torch.cat([opt_bn, sar_bn], dim=1)
            fused_bn = self.fusion_layers[-1](fused_bn) if len(self.fusion_layers) > len(opt_skips) else fused_bn

            return self.decoder(fused_bn, fused_skips)

    @torch.no_grad()
    def predict(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run inference and return binary mask."""
        self.eval()
        logits = self.forward(optical, sar)
        if self.num_classes == 1:
            return (torch.sigmoid(logits) >= threshold).float()
        else:
            return torch.argmax(torch.softmax(logits, dim=1), dim=1).float()


class _FusionDecoder(nn.Module):
    """Shared decoder for fusion models."""

    def __init__(
        self,
        encoder_channels: List[int],
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        ch = encoder_channels

        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(ch[3] + ch[3], ch[3], dropout)

        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(ch[2] + ch[2], ch[2], dropout)

        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(ch[1] + ch[1], ch[1], dropout)

        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(ch[0] + ch[0], ch[0], dropout)

        self.final_conv = nn.Conv2d(ch[0], num_classes, kernel_size=1)

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3, s4 = skips

        d4 = self.up4(bottleneck)
        d4 = self._pad_cat(d4, s4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_cat(d3, s3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_cat(d2, s2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_cat(d1, s1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    @staticmethod
    def _pad_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return torch.cat([x, skip], dim=1)
