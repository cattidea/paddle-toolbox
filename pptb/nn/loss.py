import paddle.nn as nn

from .functional import label_smooth_loss


class LabelSmoothingLoss(nn.Layer):
    def __init__(self, loss_func, ls_eps=0.1):
        super().__init__()
        assert 0 <= ls_eps < 1.0
        self.loss_func = loss_func
        self.ls_eps = ls_eps

    def forward(self, predicts, labels):
        return label_smooth_loss(predicts, labels, self.loss_func, self.ls_eps)


class LabelSmoothingCrossEntropyLoss(LabelSmoothingLoss):
    def __init__(self, ls_eps=0.1):
        loss_func = nn.CrossEntropyLoss(soft_label=True)
        super().__init__(loss_func=loss_func, ls_eps=ls_eps)
