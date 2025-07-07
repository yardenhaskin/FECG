from server.HelpFunctions import *  # for docker

# from HelpFunctions import *  # for local testing
from torch import nn


class ResNetBasicBlockEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="leaky_relu",
        expansion=1,
        downsampling=1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.activation, self.expansion, self.downsampling = (
            activation,
            expansion,
            downsampling,
        )

        self.blocks = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                stride=self.downsampling,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv1d(
                self.out_channels,
                self.expanded_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    self.expanded_channels,
                    kernel_size=3,
                    stride=self.downsampling,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expanded_channels),
            )
            if self.should_apply_shortcut
            else None
        )

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion


class ResNetBasicBlockDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="leaky_relu",
        expansion=1,
        upsampling=1,
        output_padding=0,
        *args,
        **kwargs,
    ):
        super().__init__()
        (
            self.in_channels,
            self.out_channels,
            self.activation,
            self.expansion,
            self.upsampling,
            self.output_padding,
        ) = (
            in_channels,
            out_channels,
            activation,
            expansion,
            upsampling,
            output_padding,
        )
        self.blocks = nn.Sequential(
            nn.ConvTranspose1d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                stride=self.upsampling,
                output_padding=self.output_padding,
            ),
            nn.BatchNorm1d(out_channels),
            activation_func(activation),
            nn.ConvTranspose1d(
                self.out_channels,
                self.expanded_channels,
                kernel_size=3,
                padding=0 if upsampling > 1 else 1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.ConvTranspose1d(
                    self.in_channels,
                    self.expanded_channels,
                    kernel_size=3,
                    stride=self.upsampling,
                    output_padding=self.output_padding,
                    padding=0 if upsampling > 1 else 1,
                    bias=False,
                ),
                nn.BatchNorm1d(self.expanded_channels),
            )
            if self.should_apply_shortcut
            else None
        )

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
