# Paddle Toolbox [WIP]

一些方便的小工具，参考 Paddle 的 API 设计以及 Torch Toolbox API 设计

## 安装

### 使用 pip 安装

```bash
pip install pptb
```

由于仍处于开发阶段，API 较为不稳定，建议安装时指定版本号

```bash
pip install pptb==0.1.4
```

### 直接从 GitHub 拉取最新代码

这里以 AiStudio 为例

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
from pptb.nn import LabelSmoothingLoss

num_classes = 40
label_smooth_epision = 0.1

# 如果需要标签平滑后 Loss，将下面这行替换成后面那一行即可
# loss_function = paddle.nn.CrossEntropyLoss()
loss_function = LabelSmoothingLoss(paddle.nn.CrossEntropyLoss(soft_label=True), num_classes, label_smooth_epision)
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

### Mixup

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

### Vision models

提供更加丰富的 backbone，所有模型均会提供预训练权重

已支持一些 PaddleClas 下的预训练模型，以及比较新的 ConvMixer

-  GoogLeNet
-  Incetpionv3（已并入 paddle 主线）
-  ResNeXt（已并入 paddle 主线）
-  ShuffleNetV2
-  ConvMixer（预训练权重转自 PyTorch）
-  DenseNet (未完整支持)

```python
import paddle
import pptb.vision.models as ppmodels

model = ppmodels.resnext50_32x4d(pretrained=True)
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

-  [ ] Cutout
-  [ ] Activation、Mish
-  [ ] ~~Lookahead (paddle.incubate.LookAhead 已经有了)~~
-  [ ] 更多 vision models
   -  [ ] MobileNetV3
   -  [ ] Xception
   -  [ ] Swin Transformer
   -  [ ] DenseNet（完整支持）
-  [ ] 完整的单元测试

## References

-  [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
-  [Torch Toolbox](https://github.com/PistonY/torch-toolbox)
