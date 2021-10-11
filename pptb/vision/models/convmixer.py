from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.nn as nn

__all__ = []


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Layer):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000, with_pool=True):
        super(ConvMixer, self).__init__()

        self.num_classes = num_classes
        self.with_pool = with_pool

        self.embedding_layer = nn.Sequential(
            nn.Conv2D(3, dim, kernel_size=patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2D(dim)
        )
        self.mixer_layer = nn.LayerList(
            [
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2D(dim, dim, kernel_size, groups=dim, padding="same"), nn.GELU(), nn.BatchNorm2D(dim)
                        )
                    ),
                    nn.Conv2D(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2D(dim),
                )
                for _ in range(depth)
            ]
        )

        if with_pool:
            self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.classifer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, num_classes),
            )

    def forword(self, x):
        x = self.embedding_layer(x)
        x = self.mixer_layer(x)
        if self.with_pool:
            x = self.avg_pool(x)
        if self.num_classes > 0:
            x = self.classifer(x)
        return x


def _convmixer(arch, dim, depth, kernel_size=9, patch_size=7, pretrained=False, **kwargs):
    return ConvMixer(dim, depth, kernel_size=kernel_size, patch_size=patch_size, **kwargs)


def convmixer_1024_20(pretrained=False, **kwargs):
    return _convmixer(
        "convmixer_1024_20",
        1024,
        20,
        kernel_size=9,
        patch_size=14,
        num_classes=1000,
        pretrained=pretrained,
        **kwargs,
    )


def convmixer_1536_20(pretrained=False, **kwargs):
    return _convmixer(
        "convmixer_1536_20",
        1536,
        20,
        kernel_size=9,
        patch_size=7,
        num_classes=1000,
        pretrained=pretrained,
        **kwargs,
    )
