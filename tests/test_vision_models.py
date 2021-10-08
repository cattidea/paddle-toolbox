import numpy as np
import paddle

from pptb.vision.models import googlenet, inception_v3


def test_googlenet():
    batch_size = 5
    img_channel = 3
    img_height = 224
    img_width = 224
    num_classes = 50
    input = np.array(np.random.random((batch_size, img_channel, img_height, img_width)), dtype=np.float32)
    model = googlenet(pretrained=True, num_classes=num_classes)
    out, out1, out2 = model(paddle.to_tensor(input))
    assert out.shape == [batch_size, num_classes]
    assert out1.shape == [batch_size, num_classes]
    assert out2.shape == [batch_size, num_classes]


def test_inception_v3():
    batch_size = 5
    img_channel = 3
    img_height = 299
    img_width = 299
    num_classes = 50
    input = np.array(np.random.random((batch_size, img_channel, img_height, img_width)), dtype=np.float32)
    model = inception_v3(pretrained=True, num_classes=num_classes)
    output = model(paddle.to_tensor(input))
    assert output.shape == [batch_size, num_classes]
