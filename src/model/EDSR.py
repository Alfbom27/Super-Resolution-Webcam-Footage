# Implementation from https://github.com/Coloquinte/torchSR/blob/main/torchsr/models/edsr.py
# Based on: Enhanced Deep Residual Networks for Single Image Super-Resolution
# https://arxiv.org/abs/1707.02921
# Modified using MobileNetv2 and MobileNetv3-style Inverted Residual Bottleneck blocks.

import math
import torch
import torch.nn as nn

from model.MobileNetV2 import InverseResidualBlock as IRBv2
from model.MobileNetV3 import InverseResidualBlock as IRBv3


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class StackedInverseResidualBlock(nn.Module):
    """Stack of inverse residual blocks (MobileNetV2 style)."""
    def __init__(self, n_feats: int, expansion_factor: int = 6, repeats: int = 1):
        super().__init__()
        self.blocks = nn.Sequential(
            *[IRBv2(n_feats, n_feats, expansion_factor) for _ in range(repeats)]
        )

    def forward(self, x):
        return self.blocks(x)


class StackedInverseResidualBlockV3(nn.Module):
    """Stack of inverse residual blocks (MobileNetV3 style with squeeze-excitation)."""
    def __init__(self, n_feats: int, expansion_factor: int = 6, repeats: int = 1,
                 squeeze_excitation: bool = True, kernel_size: int = 3):
        super().__init__()
        self.blocks = nn.Sequential(
            *[IRBv3(n_feats, n_feats, kernel_size=kernel_size,
                    expansion_size=n_feats * expansion_factor,
                    squeeze_exitation=squeeze_excitation)
              for _ in range(repeats)]
        )

    def forward(self, x):
        return self.blocks(x)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(
        self,
        n_resblocks,
        n_feats,
        scale,
        res_scale,
        map_location=None,
        use_inverse_residual=False,
        expansion_factor=6,
        irb_repeats=1,
        irb_version="v2",
        squeeze_excitation=True,
    ):
        """
        Args:
            n_resblocks: Number of residual blocks
            n_feats: Number of feature channels
            scale: Upscaling factor (2, 3, or 4)
            res_scale: Residual scaling factor (only used for standard ResBlock)
            map_location: Device for loading pretrained weights
            use_inverse_residual: If True, use MobileNet-style inverse residual blocks
            expansion_factor: Expansion factor for inverse residual blocks (default: 6)
            irb_repeats: Number of inverse residual blocks to stack per block (default: 1)
            irb_version: "v2" for MobileNetV2 IRB, "v3" for MobileNetV3 IRB with SE (default: "v2")
            squeeze_excitation: Enable squeeze-excitation in V3 blocks (default: True)
        """
        super(EDSR, self).__init__()
        self.scale = scale

        kernel_size = 3
        n_colors = 3
        rgb_range = 255
        conv = default_conv
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        if use_inverse_residual:
            if irb_version == "v3":
                m_body = [
                    StackedInverseResidualBlockV3(n_feats, expansion_factor, irb_repeats,
                                                   squeeze_excitation=squeeze_excitation)
                    for _ in range(n_resblocks)
                ]
            else:  # v2 (default)
                m_body = [
                    StackedInverseResidualBlock(n_feats, expansion_factor, irb_repeats)
                    for _ in range(n_resblocks)
                ]
        else:
            m_body = [
                ResBlock(conv, n_feats, kernel_size, act=act, res_scale=res_scale)
                for _ in range(n_resblocks)
            ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x, scale=None):
        if scale is not None and scale != self.scale:
            raise ValueError(f"Network scale is {self.scale}, not {scale}")
        x = self.sub_mean(255 * x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x) / 255

        return x

    def load_pretrained(self, weights_path, map_location=None):
        if map_location is None:
            map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(weights_path, map_location=map_location)
        self.load_state_dict(state_dict)


def edsr_r16f64(scale):
    return EDSR(16, 64, scale, 1.0)


def edsr_r32f256(scale):
    return EDSR(32, 256, scale, 0.1)


def edsr_baseline(scale):
    return edsr_r16f64(scale)


def edsr(scale):
    return edsr_r32f256(scale)


# Inverse residual block variants
def edsr_r16f64_irb(scale, expansion_factor=6, irb_repeats=1, irb_version="v2", squeeze_excitation=True):
    """EDSR baseline with inverse residual blocks.

    Args:
        scale: Upscaling factor (2, 3, or 4)
        expansion_factor: Expansion factor for IRB (default: 6)
        irb_repeats: Number of IRBs to stack per block (default: 1)
        irb_version: "v2" for MobileNetV2, "v3" for MobileNetV3 with SE (default: "v2")
        squeeze_excitation: Enable SE in V3 blocks (default: True)
    """
    return EDSR(16, 64, scale, 1.0, use_inverse_residual=True,
                expansion_factor=expansion_factor, irb_repeats=irb_repeats,
                irb_version=irb_version, squeeze_excitation=squeeze_excitation)


def edsr_r32f256_irb(scale, expansion_factor=6, irb_repeats=1, irb_version="v2", squeeze_excitation=True):
    """EDSR full with inverse residual blocks.

    Args:
        scale: Upscaling factor (2, 3, or 4)
        expansion_factor: Expansion factor for IRB (default: 6)
        irb_repeats: Number of IRBs to stack per block (default: 1)
        irb_version: "v2" for MobileNetV2, "v3" for MobileNetV3 with SE (default: "v2")
        squeeze_excitation: Enable SE in V3 blocks (default: True)
    """
    return EDSR(32, 256, scale, 0.1, use_inverse_residual=True,
                expansion_factor=expansion_factor, irb_repeats=irb_repeats,
                irb_version=irb_version, squeeze_excitation=squeeze_excitation)
