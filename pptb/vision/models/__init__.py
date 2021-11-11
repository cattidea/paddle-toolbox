# GoogLeNet
from .googlenet import GoogLeNet
from .googlenet import googlenet

# InceptionV3
from .inceptionv3 import InceptionV3
from .inceptionv3 import inception_v3

# ResNeXt
from .resnext import ResNeXt
from .resnext import resnext50_32x4d
from .resnext import resnext50_64x4d
from .resnext import resnext101_32x4d
from .resnext import resnext101_64x4d
from .resnext import resnext152_32x4d
from .resnext import resnext152_64x4d

# ShuffleNetV2
from .shufflenetv2 import ShuffleNetV2
from .shufflenetv2 import shufflenet_v2_x0_25
from .shufflenetv2 import shufflenet_v2_x0_33
from .shufflenetv2 import shufflenet_v2_x0_5
from .shufflenetv2 import shufflenet_v2_x1_0
from .shufflenetv2 import shufflenet_v2_x1_5
from .shufflenetv2 import shufflenet_v2_x2_0
from .shufflenetv2 import shufflenet_v2_swish

# ConvMixer
from .convmixer import ConvMixer
from .convmixer import convmixer_768_32
from .convmixer import convmixer_1024_20_ks9_p14
from .convmixer import convmixer_1536_20

__all__ = [
    "GoogLeNet",
    "googlenet",
    "InceptionV3",
    "inception_v3",
    "ResNeXt",
    "resnext50_32x4d",
    "resnext50_64x4d",
    "resnext101_32x4d",
    "resnext101_64x4d",
    "resnext152_32x4d",
    "resnext152_64x4d",
    "ShuffleNetV2",
    "shufflenet_v2_x0_25",
    "shufflenet_v2_x0_33",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "shufflenet_v2_swish",
    "ConvMixer",
    "convmixer_768_32",
    "convmixer_1024_20_ks9_p14",
    "convmixer_1536_20",
]
