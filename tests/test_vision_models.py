import numpy as np
import paddle
import pytest

from pptb.vision.models import (
    convmixer_768_32,
    convmixer_1024_20_ks9_p14,
    convmixer_1536_20,
)


def model_forwards(model, data_shape=(3, 224, 224), batch_size=3):
    model.eval()
    input = np.array(np.random.random((batch_size, *data_shape)), dtype=np.float32)
    with paddle.no_grad():
        return model(paddle.to_tensor(input))


def test_convmixer_768_32():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = convmixer_768_32(pretrained=True, num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_convmixer_1024_20_ks9_p14():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = convmixer_1024_20_ks9_p14(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_convmixer_1536_20():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = convmixer_1536_20(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]
