import paddle.nn as nn


class Identity(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Residual(nn.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
