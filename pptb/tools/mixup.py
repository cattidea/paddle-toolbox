import numpy as np
import paddle


def mixup_data(data, labels, alpha=0.2):
    # @refs: https://www.zhihu.com/question/308572298
    lam = np.random.beta(alpha, alpha)
    index = paddle.randperm(len(data))
    data_mixed = lam * data + (1 - lam) * data[index]

    # see https://github.com/PaddlePaddle/Paddle/issues/35891
    if len(labels) == 1:
        labels_a, labels_b = labels, labels[index].unsqueeze(0)
    else:
        labels_a, labels_b = labels, labels[index]
    return data_mixed, labels_a, labels_b, lam


def mixup_criterion(loss_function, predicts, labels_a, labels_b, lam):
    return lam * loss_function(predicts, labels_a) + (1 - lam) * loss_function(predicts, labels_b)


def mixup_metric(metric_function, predicts, labels_a, labels_b, lam):
    return lam * metric_function(predicts, labels_a) + (1 - lam) * metric_function(predicts, labels_b)
