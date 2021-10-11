from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

__all__ = []

model_urls = {
    "convmixer_768_32": (
        "https://github.com/cattidea/paddle-toolbox/releases/download/convmixer-pretrained-v1/convmixer_768_32_ks7_p7_relu.pdparams",
        "e0812272ad8b994aa169cfd27d93f626",
    ),
    "convmixer_1024_20_ks9_p14": (
        "https://github.com/cattidea/paddle-toolbox/releases/download/convmixer-pretrained-v1/convmixer_1024_20_ks9_p14.pdparams",
        "53462088cb1e55f7a43bc0b9adb9e683",
    ),
    "convmixer_1536_20": (
        "https://github.com/cattidea/paddle-toolbox/releases/download/convmixer-pretrained-v1/convmixer_1536_20_ks9_p7.pdparams",
        "0442b7e98b873a707de5a8d5419db26d",
    ),
}


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Layer):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, activation=nn.GELU, num_classes=1000, with_pool=True):
        super(ConvMixer, self).__init__()

        self.num_classes = num_classes
        self.with_pool = with_pool

        self.stem = nn.Sequential(
            nn.Conv2D(3, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2D(dim),
        )
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2D(dim, dim, kernel_size, groups=dim, padding="same"),
                            activation(),
                            nn.BatchNorm2D(dim),
                        )
                    ),
                    nn.Conv2D(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2D(dim),
                )
                for _ in range(depth)
            ]
        )

        if with_pool:
            self.pooling = nn.AdaptiveAvgPool2D((1, 1))

        if num_classes > 0:
            self.classifer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dim, num_classes),
            )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        if self.with_pool:
            x = self.pooling(x)
        if self.num_classes > 0:
            x = self.classifer(x)
        return x


def _convmixer(arch, dim, depth, kernel_size=9, patch_size=7, activation=nn.GELU, pretrained=False, **kwargs):
    model = ConvMixer(
        dim,
        depth,
        kernel_size=kernel_size,
        patch_size=patch_size,
        activation=activation,
        **kwargs,
    )
    if pretrained:
        assert (
            arch in model_urls
        ), "{} model do not have a pretrained model now, you should set pretrained=False".format(arch)
        weight_path = get_weights_path_from_url(model_urls[arch][0], model_urls[arch][1])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def convmixer_768_32(pretrained=False, **kwargs):
    return _convmixer(
        "convmixer_768_32",
        768,
        32,
        kernel_size=7,
        patch_size=7,
        activation=nn.ReLU,
        pretrained=pretrained,
        **kwargs,
    )


def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs):
    return _convmixer(
        "convmixer_1024_20",
        1024,
        20,
        kernel_size=9,
        patch_size=14,
        activation=nn.GELU,
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
        activation=nn.GELU,
        pretrained=pretrained,
        **kwargs,
    )
