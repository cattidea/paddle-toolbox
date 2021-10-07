import paddle
from .functional import label_smooth_loss


class LabelSmoothingLoss(paddle.nn.Layer):
    def __init__(self, loss_func, num_classes, ls_eps=0.1):
        super().__init__()
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.ls_eps = ls_eps

    def forward(self, predicts, labels):
        return label_smooth_loss(predicts, labels, self.loss_func, self.num_classes, self.ls_eps)
