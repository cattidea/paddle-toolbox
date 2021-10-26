import numpy as np
import paddle
import pytest

from pptb.tools import mixup_criterion, mixup_data, mixup_metric
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
