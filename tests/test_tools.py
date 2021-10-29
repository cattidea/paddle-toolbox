import numpy as np
import paddle
import pytest
from typing import Sequence

from pptb.tools import mixup_criterion, mixup_data, mixup_metric
from pptb.tools import cutmix_criterion, cutmix_data, cutmix_metric
from pptb.tools import MixingDataController
from pptb.vision.models import resnext50_32x4d


@pytest.mark.parametrize("batch_size", [1, 10])
def test_mixup(batch_size: int):
    fake_inputs = paddle.to_tensor(np.array(np.random.random((batch_size, 3, 224, 224)), dtype=np.float32))
    fake_labels = paddle.to_tensor(np.array(np.random.random((batch_size, 1)), dtype=np.int64))
    model = resnext50_32x4d()
    mixup_alpha = 0.2
    loss_function = paddle.nn.CrossEntropyLoss()

    X_batch_mixed, y_batch_a, y_batch_b, lam = mixup_data(fake_inputs, fake_labels, mixup_alpha)
    predicts = model(X_batch_mixed)
    loss = mixup_criterion(loss_function, predicts, y_batch_a, y_batch_b, lam)
    acc = mixup_metric(paddle.metric.accuracy, predicts, y_batch_a, y_batch_b, lam)

    assert X_batch_mixed.shape == fake_inputs.shape
    assert y_batch_a.shape == y_batch_b.shape == fake_labels.shape


@pytest.mark.parametrize(
    "batch_size, data_shape, mix_axes",
    [
        (1, [3, 224, 224], [2, 3]),
        (10, [3, 224, 224], [2, 3]),
        (10, [1000, 25], [1]),
        (10, [1000, 50, 100], [1, 2, 3]),
    ],
)
def test_cutmix(batch_size: int, data_shape: Sequence[int], mix_axes: Sequence[int]):
    fake_inputs = paddle.to_tensor(np.array(np.random.random((batch_size, *data_shape)), dtype=np.float32))
    fake_labels = paddle.to_tensor(np.array(np.random.random((batch_size, 1)), dtype=np.int64))
    model = resnext50_32x4d()
    cutmix_alpha = 0.2
    loss_function = paddle.nn.CrossEntropyLoss()

    X_batch_mixed, y_batch_a, y_batch_b, lam = cutmix_data(fake_inputs, fake_labels, cutmix_alpha, axes=mix_axes)
    predicts = model(X_batch_mixed)
    loss = cutmix_criterion(loss_function, predicts, y_batch_a, y_batch_b, lam)
    acc = cutmix_metric(paddle.metric.accuracy, predicts, y_batch_a, y_batch_b, lam)

    assert X_batch_mixed.shape == fake_inputs.shape
    assert y_batch_a.shape == y_batch_b.shape == fake_labels.shape


@pytest.mark.parametrize("is_numpy", [True, False])
def test_mixing_data_controller(is_numpy: bool):
    mixing_data_controller = MixingDataController(
        mixup=True,
        cutmix=True,
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        mixup_prob=0.2,
        cutmix_prob=0.2,
        cutmix_axes=[2, 3],
        loss_function=paddle.nn.CrossEntropyLoss(),
        metric_function=paddle.metric.accuracy,
    )
    batch_size = 16
    model = resnext50_32x4d()

    fake_inputs = np.array(np.random.random((batch_size, 3, 224, 224)), dtype=np.float32)
    fake_labels = np.array(np.random.random((batch_size, 1)), dtype=np.int64)

    if not is_numpy:
        fake_inputs = paddle.to_tensor(fake_inputs)
        fake_labels = paddle.to_tensor(fake_labels)

    X_batch_mixed, y_batch_a, y_batch_b, lam = mixing_data_controller.mix(fake_inputs, fake_labels, is_numpy=is_numpy)

    if is_numpy:
        X_batch_mixed = paddle.to_tensor(X_batch_mixed)
        y_batch_a = paddle.to_tensor(y_batch_a)
        y_batch_b = paddle.to_tensor(y_batch_b)

    predicts = model(X_batch_mixed)
    loss = mixing_data_controller.loss(predicts, y_batch_a, y_batch_b, lam)
    acc = mixing_data_controller.metric(predicts, y_batch_a, y_batch_b, lam)

    assert X_batch_mixed.shape == fake_inputs.shape
    assert y_batch_a.shape == y_batch_b.shape == fake_labels.shape
