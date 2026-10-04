"""
Siamese U-Net for pre/post-disaster change detection.

Architecture:
    - Shared lightweight encoder processes both pre and post images
    - Feature difference computed at each encoder level
    - U-Net decoder with skip connections from difference features
    - Binary output: 0 = No Change, 1 = Changed

Input:  Two 256×256×3 satellite images (pre-disaster, post-disaster)
Output: 256×256×1 binary change mask

Total parameters: ~3.5M
Peak VRAM (inference, fp32): ~14MB
"""

import torch
import torch.nn as nn

from .base_unet import LightweightEncoder, LightweightDecoder, ConvBlock, get_device, count_parameters


class SiameseUNet(nn.Module):
    """
    Siamese U-Net for binary change detection between pre/post image pairs.

    The shared encoder extracts features from both images independently.
    At each encoder level, the absolute difference of features is computed.
    The decoder reconstructs the binary change mask from these difference features.
    """

    def __init__(
        self,
        in_channels: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared encoder — same weights process both pre and post images
        self.encoder = LightweightEncoder(in_channels=in_channels, dropout=dropout)
        encoder_channels = self.encoder.channels  # [16, 32, 64, 128, 256]

        # Bottleneck difference processing
        # After computing |pre_bn - post_bn|, we process it through a ConvBlock
        self.bottleneck_diff = ConvBlock(
            encoder_channels[4], encoder_channels[4], dropout
        )

        # Decoder — takes difference features as skip connections
        # skip_channel_multiplier=1 because we compute element-wise difference
        # (same channel count as a single encoder output)
        self.decoder = LightweightDecoder(
            out_channels=1,
            encoder_channels=encoder_channels,
            dropout=dropout,
            skip_channel_multiplier=1,
        )

    def forward(
        self, pre_image: torch.Tensor, post_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pre_image:  Pre-disaster image tensor [B, 3, H, W] (normalized 0-1).
            post_image: Post-disaster image tensor [B, 3, H, W] (normalized 0-1).

        Returns:
            Change probability map [B, 1, H, W] with sigmoid activation.
        """
        # Shared encoder forward pass
        pre_bn, pre_skips = self.encoder(pre_image)   # pre_skips = [e1, e2, e3, e4]
        post_bn, post_skips = self.encoder(post_image)

        # Compute absolute difference at each level
        diff_skips = [
            torch.abs(pre_s - post_s)
            for pre_s, post_s in zip(pre_skips, post_skips)
        ]

        # Bottleneck difference
        bn_diff = torch.abs(pre_bn - post_bn)
        bn_diff = self.bottleneck_diff(bn_diff)

        # Decode from difference features
        logits = self.decoder(bn_diff, diff_skips)

        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Run inference and return a binary change mask.

        Args:
            pre_image:  [B, 3, H, W] normalized tensor.
            post_image: [B, 3, H, W] normalized tensor.
            threshold:  Probability threshold for binary classification.

        Returns:
            Binary mask [B, 1, H, W] with values 0 or 1.
        """
        self.eval()
        probs = self.forward(pre_image, post_image)
        return (probs >= threshold).float()


def create_siamese_unet(
    in_channels: int = 3,
    dropout: float = 0.2,
    weights_path: str = None,
    device: torch.device = None,
) -> SiameseUNet:
    """
    Factory function to create and optionally load weights for the Siamese U-Net.

    Args:
        in_channels: Number of input image channels (default 3 for RGB).
        dropout: Dropout probability.
        weights_path: Optional path to pre-trained weights (.pth).
        device: Target device. Auto-detected if None.

    Returns:
        SiameseUNet model on the specified device.
    """
    if device is None:
        device = get_device()

    model = SiameseUNet(in_channels=in_channels, dropout=dropout)

    if weights_path is not None:
        # weights_only: never unpickle arbitrary objects from a checkpoint file.
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"[SiameseUNet] Loaded weights from: {weights_path}")
    else:
        print(
            "[SiameseUNet] No weights loaded — model initialized with random weights.\n"
            "  To use for real predictions, train the model first or provide weights_path."
        )

    model = model.to(device)
    param_count = count_parameters(model)
    print(f"[SiameseUNet] Parameters: {param_count:,} | Device: {device}")

    return model
