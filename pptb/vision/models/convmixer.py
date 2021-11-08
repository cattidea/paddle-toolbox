# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from paddle.utils.download import get_weights_path_from_url

from ...nn import Residual

__all__ = []

model_urls = {
    "convmixer_768_32": (
        "https://bj.bcebos.com/v1/ai-studio-online/3c594317e9cd464c92cefffc8805709af7c027fee65a448e813e02662281f439",
        "e0812272ad8b994aa169cfd27d93f626",
    ),
    "convmixer_1024_20_ks9_p14": (
        "https://bj.bcebos.com/v1/ai-studio-online/a645ddcf3c5245218f4caf22bdd41e982f3c34c5d9ff4cada80d3414b1a252c3",
        "53462088cb1e55f7a43bc0b9adb9e683",
    ),
    "convmixer_1536_20": (
        "https://bj.bcebos.com/v1/ai-studio-online/51af9ab6b0fb4627b94433b8b48dc74f8657544e192a472c85d5c92445853ce1",
        "0442b7e98b873a707de5a8d5419db26d",
    ),
}


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
        "convmixer_1024_20_ks9_p14",
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
