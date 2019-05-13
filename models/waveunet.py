from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as functional


def crop_and_concat(x1, x2):
    # input is [Channels, Width], crop and concat functionality
    diff = x2.size()[2] - x1.size()[2]
    if diff > 0:
        x2 = x2[:, :, math.floor(diff/2): -(diff - math.floor(diff/2))]
    x = torch.cat((x2, x1), dim=1)
    return x


class DownsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=0):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.features = None

    def forward(self, x):
        x = self.conv(x)
        self.features = x
        x = nn.functional.interpolate(x, scale_factor=0.5, mode='linear')
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=0):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = nn.functional.interpolate(x1, scale_factor=2, mode='linear')
        x = crop_and_concat(x1, x2)
        x = self.conv(x)
        return x


class WaveUNet(nn.Module):

    def __init__(self, n_sources, n_blocks=12,
                 n_filters=24, filter_size=15, merge_filter_size=5,
                 conditioning=None, context=True):
        super(WaveUNet, self).__init__()
        self.n_sources = n_sources
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.merge_filter_size = merge_filter_size
        self.conditioning = conditioning    # None, 'multi', 'concat'
        self.context = context

        if self.context:
            self.encoding_padding = 0
            self.decoding_padding = 0
        else:
            self.encoding_padding = self.filter_size // 2
            self.decoding_padding = self.merge_filter_size // 2

        # number of input/output channels for every layer in encoder and decoder
        channels = [1] + [(i + 1) * n_filters for i in range(n_blocks+1)]

        self.encoder = self._make_encoder(channels)
        self.bottleneck = nn.Sequential(nn.Conv1d(channels[-2], channels[-1], self.filter_size,
                                                  padding=self.encoding_padding, dilation=1),
                                        nn.BatchNorm1d(channels[-1]),
                                        nn.LeakyReLU(inplace=True)
                                        )
        if self.conditioning:
            self.scale_conditioning = nn.Linear(n_sources, channels[-1])
        self.decoder = self._make_decoder(channels[::-1])
        self.output = self.output_layer()

    def forward(self, x, labels=None):
        original_audio = x

        for i_block in range(self.n_blocks):
            x = self.encoder[i_block](x)

        x = self.bottleneck(x)

        if self.conditioning:
            # Apply multiplicative conditioning
            scaled_labels = self.scale_conditioning(labels)
            x = torch.mul(x, scaled_labels.view(scaled_labels.shape + (1, )))

        for i_block in range(self.n_blocks):
            x = self.decoder[i_block](x, self.encoder[-i_block-1].features)

        x = crop_and_concat(x, original_audio)

        outputs = list()
        for i in range(self.n_sources):
            outputs.append(self.output(x))
        return torch.stack(outputs, dim=1)

    def _make_encoder(self, channels):
        layers = list()
        for i in range(self.n_blocks):
            layers.append(DownsampleBlock(channels[i], channels[i+1],
                                          kernel_size=self.filter_size,
                                          padding=self.encoding_padding
                                          ))
        return nn.Sequential(*layers)

    def _make_decoder(self, channels):
        layers = list()
        for i in range(self.n_blocks):
            layers.append(UpsampleBlock(channels[i]+channels[i+1], channels[i+1],
                                        kernel_size=self.merge_filter_size,
                                        padding=self.decoding_padding,
                                        ))

        return nn.Sequential(*layers)

    def output_layer(self):
        # return an output for one source individually
        return nn.Sequential(
                nn.Conv1d(self.n_filters + 1, 1, 1),
                nn.Tanh())
