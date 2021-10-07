import paddle
import paddle.nn.functional as F


def label_smooth_loss(predicts, labels, loss_func, num_classes, ls_eps=0.1):
    """平滑标签 loss
    @ref: https://github.com/PaddlePaddle/PaddleVideo/blob/29039894d65d3660e9fb2a08a7c9374b97bfb10f/paddlevideo/modeling/heads/base.py#L115-L120
    """
    labels = F.one_hot(labels, num_classes)
    labels = F.label_smooth(labels, epsilon=ls_eps)
    labels = paddle.squeeze(labels, axis=1)
    loss = loss_func(predicts, labels)
    return loss
