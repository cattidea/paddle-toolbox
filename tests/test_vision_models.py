import pytest
import numpy as np
import paddle

from pptb.vision.models import googlenet, inception_v3
from pptb.vision.models import (
    resnext50_32x4d,
    resnext50_64x4d,
    resnext101_32x4d,
    resnext101_64x4d,
    resnext152_32x4d,
    resnext152_64x4d,
)
from pptb.vision.models import (
    shufflenet_v2_x0_25,
    shufflenet_v2_x0_33,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
    shufflenet_v2_swish,
)
from pptb.vision.models import (
    convmixer_768_32,
    convmixer_1024_20_ks9_p14,
    convmixer_1536_20,
)


def model_forwards(model, data_shape=(3, 224, 224), batch_size=3):
    input = np.array(np.random.random((batch_size, *data_shape)), dtype=np.float32)
    return model(paddle.to_tensor(input))


def test_googlenet():
    batch_size = 5
    num_classes = 50
    model = googlenet(pretrained=True, num_classes=num_classes)
    out, out1, out2 = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert out.shape == [batch_size, num_classes]
    assert out1.shape == [batch_size, num_classes]
    assert out2.shape == [batch_size, num_classes]


def test_inception_v3():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = inception_v3(pretrained=True, num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 299, 299), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


def test_resnext50_32x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext50_32x4d(pretrained=True, num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_resnext50_64x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext50_64x4d(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_resnext101_32x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext101_32x4d(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_resnext101_64x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext101_64x4d(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_resnext152_32x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext152_32x4d(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_resnext152_64x4d():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = resnext152_64x4d(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


def test_shufflenet_v2_x0_25():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x0_25(pretrained=True, num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_x0_33():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x0_33(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_x0_5():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x0_5(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_x1_0():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x1_0(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_x1_5():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x1_5(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_x2_0():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_x2_0(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


@pytest.mark.ci_skip
def test_shufflenet_v2_swish():
    batch_size = 2 ** np.random.randint(3, 5)
    num_classes = np.random.randint(10, 1000)
    model = shufflenet_v2_swish(num_classes=num_classes)
    output = model_forwards(model, data_shape=(3, 224, 224), batch_size=batch_size)
    assert output.shape == [batch_size, num_classes]


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
