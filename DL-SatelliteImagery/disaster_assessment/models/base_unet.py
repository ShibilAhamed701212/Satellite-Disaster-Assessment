"""
Shared lightweight U-Net encoder-decoder building blocks.

Matches the existing project's channel widths (16→32→64→128→256)
and 256×256 patch size to maintain architectural consistency.
Implemented in PyTorch for the disaster assessment extension.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv2d → BN → ReLU → Conv2d → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LightweightEncoder(nn.Module):
    """
    Lightweight CNN encoder matching existing U-Net channel structure.

    Channel widths: 16 → 32 → 64 → 128 → 256 (5 levels)
    Each level: ConvBlock → MaxPool2d(2×2)

    Parameters: ~590K
    """

    def __init__(self, in_channels: int = 3, dropout: float = 0.2):
        super().__init__()
        self.channels = [16, 32, 64, 128, 256]

        self.enc1 = ConvBlock(in_channels, self.channels[0], dropout)
        self.enc2 = ConvBlock(self.channels[0], self.channels[1], dropout)
        self.enc3 = ConvBlock(self.channels[1], self.channels[2], dropout)
        self.enc4 = ConvBlock(self.channels[2], self.channels[3], dropout)
        self.bottleneck = ConvBlock(self.channels[3], self.channels[4], dropout)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            bottleneck_features: Tensor at lowest resolution
            skip_connections: List of [enc1, enc2, enc3, enc4] feature maps
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.bottleneck(self.pool(e4))

        return bn, [e1, e2, e3, e4]


class LightweightDecoder(nn.Module):
    """
    Lightweight U-Net decoder with skip connections.

    Channel widths: 256 → 128 → 64 → 32 → 16 (mirrors encoder)
    Each level: ConvTranspose2d(2×2) → Cat(skip) → ConvBlock

    Parameters: ~590K
    """

    def __init__(
        self,
        out_channels: int = 1,
        encoder_channels: list = None,
        dropout: float = 0.2,
        skip_channel_multiplier: int = 1,
    ):
        """
        Args:
            out_channels: Number of output mask channels.
            encoder_channels: Channel list [16, 32, 64, 128, 256].
            dropout: Dropout probability.
            skip_channel_multiplier: Multiplier for skip connection channels.
                Set to 1 for standard U-Net, 2 for Siamese (concatenated skips).
        """
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [16, 32, 64, 128, 256]

        self.channels = encoder_channels
        skip_mult = skip_channel_multiplier

        # Upsampling path
        self.up4 = nn.ConvTranspose2d(
            self.channels[4], self.channels[3], kernel_size=2, stride=2
        )
        self.dec4 = ConvBlock(
            self.channels[3] + self.channels[3] * skip_mult, self.channels[3], dropout
        )

        self.up3 = nn.ConvTranspose2d(
            self.channels[3], self.channels[2], kernel_size=2, stride=2
        )
        self.dec3 = ConvBlock(
            self.channels[2] + self.channels[2] * skip_mult, self.channels[2], dropout
        )

        self.up2 = nn.ConvTranspose2d(
            self.channels[2], self.channels[1], kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(
            self.channels[1] + self.channels[1] * skip_mult, self.channels[1], dropout
        )

        self.up1 = nn.ConvTranspose2d(
            self.channels[1], self.channels[0], kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(
            self.channels[0] + self.channels[0] * skip_mult, self.channels[0], dropout
        )

        self.final_conv = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)

    def forward(
        self, bottleneck: torch.Tensor, skip_connections: list
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: Bottleneck feature map from encoder.
            skip_connections: List of [s1, s2, s3, s4] skip feature maps
                (for Siamese, these may be difference/concatenated features).

        Returns:
            Output mask tensor (logits, no activation applied).
        """
        s1, s2, s3, s4 = skip_connections

        d4 = self.up4(bottleneck)
        d4 = self._pad_and_cat(d4, s4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, s3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, s2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, s1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle potential size mismatches between upsampled and skip features."""
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        if diff_h != 0 or diff_w != 0:
            upsampled = F.pad(
                upsampled,
                [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
            )
        return torch.cat([upsampled, skip], dim=1)


def get_device() -> torch.device:
    """Auto-detect the best available device (CUDA → CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
