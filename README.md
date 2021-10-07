# Paddle Toolbox [WIP]

一些方便的小工具，参考 Paddle 的 API 设计以及 Torch Toolbox API 设计

## 安装

### 使用 pip 安装

```bash
pip install pptb
```

由于仍处于开发阶段，API 较为不稳定，建议安装时指定版本号

```bash
pip install pptb==0.1.3
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
warmup_epoch = 3

lr_scheduler = CosineWarmup(
    learning_rate,
    step_each_epoch = step_each_epoch,
    num_epochs = num_epochs,
    warmup_epoch = warmup_epoch,
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

### 各种模型

已支持一些 PaddleClas 下的预训练模型

日后如果合入 paddle，这些模型会删除

-  GoogLeNet
-  Incetpionv3
-  ResNeXt
-  ShuffleNetV2
-  DenseNet (未完整支持)

```python
import paddle
import pptb.vision.models as ppmodels

model = ppmodels.resnext50_32x4d(pretrained=True)
```

PS: 如果这些模型无法满足你的需求的话，可以试试囊括了很多比较新的模型的 [ppim](https://github.com/AgentMaker/Paddle-Image-Models)~

## References

-  [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
-  [Torch Toolbox](https://github.com/PistonY/torch-toolbox)
