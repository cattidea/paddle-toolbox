import paddle
import paddle.nn.functional as F


def label_smooth_loss(predicts, labels, loss_func, ls_eps=0.1):
    """平滑标签 loss
    @ref: https://github.com/PaddlePaddle/PaddleVideo/blob/29039894d65d3660e9fb2a08a7c9374b97bfb10f/paddlevideo/modeling/heads/base.py#L115-L120
    """
    _, num_classes = predicts.shape
    labels = F.one_hot(labels, num_classes)
    labels = F.label_smooth(labels, epsilon=ls_eps)
    labels = paddle.squeeze(labels, axis=1)
    loss = loss_func(predicts, labels)
    return loss


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    @ref: https://github.com/rwightman/pytorch-image-models/blob/947e1df3ef495b47f1e537bbaca91e1cc1850275/timm/models/layers/drop.py#L140-L168
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output
