# Paddle Toolbox [Early WIP]

一些方便的小工具，参考 Paddle 的 API 设计以及 Torch Toolbox API 设计

:warning: 目前正处于早期设计阶段，大多数功能的开发方案尚处于草案阶段～

## 安装

### 使用 pip 安装

注意：Python 需至少 3.7.0 版本，PaddlePaddle 需至少 2.3.0 版本（会跟随 paddle 版本变化）

```bash
pip install pptb==0.2.0
```

由于仍处于早期开发设计阶段，API 较为不稳定，安装时请**一定要指定版本号**

### 直接从 GitHub 拉取最新代码

这里以 AI Studio 为例

```bash
git clone https://github.com/cattidea/paddle-toolbox.git work/paddle-toolbox/
# 如果下载太慢导致出错请使用下面的命令
# git clone https://hub.fastgit.org/cattidea/paddle-toolbox.git work/paddle-toolbox/
```

之后在你的 Notebook 或者 Python 文件中加入以下代码

```python
import sys

sys.path.append('/home/aistudio/work/paddle-toolbox/')
```

## 已支持的工具

### LabelSmoothingLoss

```python
import paddle
from pptb.nn import LabelSmoothingLoss, LabelSmoothingCrossEntropyLoss

label_smooth_epision = 0.1

loss_function = paddle.nn.CrossEntropyLoss()
# 如果需要标签平滑后 Loss，将下面这行替换成后面那一行即可
loss_function = LabelSmoothingLoss(
   paddle.nn.CrossEntropyLoss(soft_label=True),
   label_smooth_epision
)
# 由于 CrossEntropyLoss 的 LabelSmoothing 比较常用，因此也可以使用下面这个别名
loss_function = LabelSmoothingCrossEntropyLoss(label_smooth_epision)
```

### CosineWarmup

```python
import paddle
from pptb.optimizer.lr import CosineWarmup

# ...

train_batch_size = 32
learning_rate = 3e-4
step_each_epoch = len(train_set) // train_batch_size
num_epochs = 40
warmup_epochs = 3

lr_scheduler = CosineWarmup(
    learning_rate,
    total_steps = num_epochs * step_each_epoch,
    warmup_steps = warmup_epochs * step_each_epoch,
    warmup_start_lr = 0.0,
    cosine_end_lr = 0.0,
    last_epoch = -1
)

```

### Mixup && Cutmix

#### Mixup

```python
import paddle
from pptb.tools import mixup_data, mixup_criterion, mixup_metric

# ...

use_mixup = True
mixup_alpha = 0.2

for X_batch, y_batch in train_loader():
   # 使用 mixup 与不使用 mixup 代码的前向传播部分代码差异对比
   if use_mixup:
      X_batch_mixed, y_batch_a, y_batch_b, lam = mixup_data(X_batch, y_batch, mixup_alpha)
      predicts = model(X_batch_mixed)
      loss = mixup_criterion(loss_function, predicts, y_batch_a, y_batch_b, lam)
      acc = mixup_metric(paddle.metric.accuracy, predicts, y_batch_a, y_batch_b, lam)
   else:
      predicts = model(X_batch)
      loss = loss_function(predicts, y_batch)
      acc = paddle.metric.accuracy(predicts, y_batch)

   # ...
```

除了用于处理 paddle 里 `Tensor` 的 `mixup_data`，还可以使用 `mixup_data_numpy` 处理 numpy 的 ndarray。

#### Cutmix

和 Mixup 一样，只需要将 `mixup_data` 换为 `cutmix_data` 即可，与 `mixup_data` 不同的是，`cutmix_data` 还接收一个额外参数 `axes` 用于控制需要 mix 的是哪几根 axis，默认 `axes = [2, 3]`，也即 `NCHW` 格式图片数据对应的 `H` 与 `W` 两根 axis。

#### MixingDataController

用于方便管理使用 Mixup 和 Cutmix

```python
import paddle
from pptb.tools import MixingDataController

# ...

mixing_data_controller = MixingDataController(
   mixup_prob=0.3,
   cutmix_prob=0.3,
   mixup_alpha=0.2,
   cutmix_alpha=0.2,
   cutmix_axes=[2, 3],
   loss_function=paddle.nn.CrossEntropyLoss(),
   metric_function=paddle.metric.accuracy,
)

for X_batch, y_batch in train_loader():
   X_batch_mixed, y_batch_a, y_batch_b, lam = mixing_data_controller.mix(X_batch, y_batch, is_numpy=False)
   predicts = model(X_batch_mixed)
   loss = mixing_data_controller.loss(predicts, y_batch_a, y_batch_b, lam)
   acc = mixing_data_controller.metric(predicts, y_batch_a, y_batch_b, lam)

   # ...
```

### Vision models

提供更加丰富的 backbone，所有模型均会提供预训练权重

合入 paddle 主线的模型会在新版本发布时移除，避免 API 不同步导致的问题

已支持一些 PaddleClas 下的预训练模型，以及比较新的 ConvMixer

-  GoogLeNet（已并入 paddle 主线且已移除，请直接使用 paddle.vision.models.GoogLeNet）
-  Incetpionv3（已并入 paddle 主线且已移除，请直接使用 paddle.vision.models.InceptionV3）
-  ResNeXt（已并入 paddle 主线且已移除，请直接使用 paddle.vision.models.ResNet）
-  ShuffleNetV2（已并入 paddle 主线且已移除，请直接使用 paddle.vision.models.ShuffleNetV2）
-  MobileNetV3（已并入 paddle 主线且已移除，请直接使用 paddle.vision.models.MobileNetV3Large 和 paddle.vision.models.MobileNetV3Small）
-  ConvMixer（预训练权重转自 PyTorch）

```python
import paddle
import pptb.vision.models as ppmodels

model = ppmodels.convmixer_768_32(pretrained=True)
```

PS: 如果这些模型无法满足你的需求的话，可以试试囊括了很多比较新的模型的 [ppim](https://github.com/AgentMaker/Paddle-Image-Models)~

#### ConvMixer

| Model Name                | Kernel Size | Patch Size | Top-1                                                 | Top-5  |
| ------------------------- | ----------- | ---------- | ----------------------------------------------------- | ------ |
| convmixer_768_32          | 7           | 7          | 0.7974<span style="color:green;"><sub>(-0.0042)</sub> | 0.9486 |
| convmixer_1024_20_ks9_p14 | 9           | 14         | 0.7681<span style="color:green;"><sub>(-0.0013)</sub> | 0.9335 |
| convmixer_1536_20         | 9           | 7          | 0.8083<sub><span style="color:green;">(-0.0054)</sub> | 0.9557 |

### TODO List

一些近期想做的功能

-  [x] Cutmix
-  [ ] Activation、Mish
-  [ ] RandomErasing
-  [ ] AutoAugment、RandAugment
-  [ ] Transform Layer（使用 Layer 实现某些 Transform）
-  [ ] 更多 vision models
   -  [ ] Xception
   -  [ ] Swin Transformer
   -  [ ] CvT
-  [ ] 完整的单元测试

## References

-  [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
-  [Torch Toolbox](https://github.com/PistonY/torch-toolbox)
-  [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
