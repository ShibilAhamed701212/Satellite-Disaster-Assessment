"""
Feature-level fusion modules for multi-modal networks.
"""

from typing import List

import torch
import torch.nn as nn


class EarlyFusion(nn.Module):
    """Concatenate multi-modal inputs along channel dimension.

    Simplest fusion: stack all modalities and let the network learn.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Concatenate along channel dimension (dim=1).

        Args:
            inputs: List of [B, C_i, H, W] tensors.

        Returns:
            [B, sum(C_i), H, W] tensor.
        """
        return torch.cat(inputs, dim=1)


class FeatureLevelFusion(nn.Module):
    """Fuse features from multiple encoders at each scale level.

    Options: concatenation, addition, learned gating.
    """

    def __init__(self, in_channels: List[int], fusion_method: str = "concat"):
        super().__init__()
        self.fusion_method = fusion_method

        if fusion_method == "gate":
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch, kernel_size=1),
                    nn.Sigmoid(),
                )
                for ch in in_channels
            ])

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple sources.

        Args:
            features_list: List of [B, C, H, W] tensors with same C.

        Returns:
            Fused [B, C, H, W] tensor.
        """
        if self.fusion_method == "concat":
            # Channel-wise concatenation + 1x1 conv
            return torch.cat(features_list, dim=1)

        elif self.fusion_method == "add":
            return sum(features_list)

        elif self.fusion_method == "gate":
            if len(features_list) < 2:
                return features_list[0]
            cat = torch.cat(features_list, dim=1)
            gate = self.gates[0](cat)
            return gate * features_list[0] + (1 - gate) * features_list[1]

        return features_list[0]
