import numpy as np
import paddle

from pptb.nn import DropPath, LabelSmoothingLoss


def test_label_smoothing_loss():
    batch_size = 5
    num_classes = 24
    label_smooth_epision = 0.1
    loss_function = LabelSmoothingLoss(paddle.nn.CrossEntropyLoss(soft_label=True), label_smooth_epision)
    predicts = paddle.uniform([batch_size, num_classes], dtype=paddle.float32)
    labels_data = np.random.randint(0, num_classes, size=(batch_size)).astype(np.int64)

    labels = paddle.to_tensor(labels_data)
    output = loss_function(predicts, labels)


def test_drop_path():
    data_shape = (3, 224, 224)
    drop_prob = np.random.random()
    layer = DropPath(drop_prob)
    input = paddle.uniform(data_shape, dtype=paddle.float32)

    output = layer(input)

    assert input.shape == output.shape
