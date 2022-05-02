import collections
from typing import Tuple

import numpy as np
import paddle


def _cutmix_on_one_axis(size: int, ratio: float) -> Tuple[int, int]:
    cut_size = int(size * ratio)

    cut_center = np.random.randint(size)
    lower_bound = np.clip(cut_center - cut_size // 2, 0, size)
    upper_bound = np.clip(cut_center + cut_size // 2, 0, size)
    return lower_bound, upper_bound


def _index_tensor_by_1dtensor(tensor: paddle.Tensor, indices: paddle.Tensor) -> paddle.Tensor:
    # see https://github.com/PaddlePaddle/Paddle/issues/35891
    original_ndim = tensor.ndim
    tensor = tensor[indices]
    if indices.shape[0] == 1:
        tensor = tensor.unsqueeze(0)
    assert tensor.ndim == original_ndim
    return tensor


def mixup_data_numpy(data, labels, alpha=0.2):
    original_dtype = data.dtype
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(data.shape[0])
    data_mixed = lam * data + (1 - lam) * data[index]

    labels_a, labels_b = labels, labels[index]
    return data_mixed.astype(original_dtype), labels_a, labels_b, lam


def mixup_data(data, labels, alpha=0.2):
    # @refs: https://www.zhihu.com/question/308572298
    original_dtype = data.dtype
    lam = np.random.beta(alpha, alpha)
    index = paddle.randperm(len(data))
    data_mixed = lam * data + (1 - lam) * data[index]

    labels_a, labels_b = labels, _index_tensor_by_1dtensor(labels, index)
    return data_mixed.astype(original_dtype), labels_a, labels_b, lam


def cutmix_data_numpy(data, labels, alpha=0.2, axes=[2, 3]):
    assert 0 not in axes, "batch 轴不应进行 cutmix"

    lam = np.random.beta(alpha, alpha)

    indices_0 = np.random.permutation(data.shape[0])
    labels_a, labels_b = labels, labels[indices_0]

    cut_ratio = np.power(1 - lam, 1 / len(axes))
    bounds = [_cutmix_on_one_axis(data.shape[axis], cut_ratio) for axis in axes]
    indices_cut_area = tuple(slice(*bounds[axes.index(i)]) if i in axes else slice(None) for i in range(1, data.ndim))

    data_mixed = data.copy()
    data_mixed[(slice(None), *indices_cut_area)] = data[(indices_0, *indices_cut_area)]

    return data_mixed, labels_a, labels_b, lam


def cutmix_data(data, labels, alpha=0.2, axes=[2, 3]):
    assert 0 not in axes, "batch 轴不应进行 cutmix"

    lam = np.random.beta(alpha, alpha)

    indices_0 = paddle.randperm(data.shape[0])
    labels_a, labels_b = labels, _index_tensor_by_1dtensor(labels, indices_0)

    cut_ratio = np.power(1 - lam, 1 / len(axes))
    bounds = [_cutmix_on_one_axis(data.shape[axis], cut_ratio) for axis in axes]
    # see https://github.com/PaddlePaddle/Paddle/issues/36223
    for bound in bounds:
        if bound[0] == bound[1]:
            return data.detach(), labels_a, labels_b, lam
    indices_cut_area = tuple(slice(*bounds[axes.index(i)]) if i in axes else slice(None) for i in range(1, data.ndim))

    data_mixed = data.detach()
    data_shuffled = _index_tensor_by_1dtensor(data.detach(), indices_0)
    data_mixed[(slice(None), *indices_cut_area)] = data_shuffled[(slice(None), *indices_cut_area)]

    return data_mixed, labels_a, labels_b, lam


def mixup_criterion(loss_function, predicts, labels_a, labels_b, lam):
    return lam * loss_function(predicts, labels_a) + (1 - lam) * loss_function(predicts, labels_b)


def mixup_metric(metric_function, predicts, labels_a, labels_b, lam):
    return lam * metric_function(predicts, labels_a) + (1 - lam) * metric_function(predicts, labels_b)


cutmix_criterion = mixup_criterion
cutmix_metric = mixup_metric


class MixingDataController:
    def __init__(
        self,
        *,
        mixup_prob=0.3,
        cutmix_prob=0.3,
        mixup_alpha=0.2,
        cutmix_alpha=0.2,
        cutmix_axes=[2, 3],
        loss_function=paddle.nn.CrossEntropyLoss(),
        metric_function=paddle.metric.accuracy,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.cutmix_axes = cutmix_axes
        self.loss_function = loss_function
        self.metric_function = metric_function

        assert self.mixup_prob >= 0
        assert self.cutmix_prob >= 0
        assert np.sum([self.mixup_prob, self.cutmix_prob]) <= 1, "mixup_prob + cutmix_prob > 1"

    def mix(self, data, labels, is_numpy=False):
        mix_mode = np.random.choice(
            ["mixup", "cutmix", "no_mix"], p=[self.mixup_prob, self.cutmix_prob, 1 - self.mixup_prob - self.cutmix_prob]
        )
        labels_a, labels_b = labels, labels
        if mix_mode == "mixup":
            if is_numpy:
                data, labels_a, labels_b, lam = mixup_data_numpy(data, labels, alpha=self.mixup_alpha)
            else:
                data, labels_a, labels_b, lam = mixup_data(data, labels, alpha=self.mixup_alpha)
        elif mix_mode == "cutmix":
            if is_numpy:
                data, labels_a, labels_b, lam = cutmix_data_numpy(
                    data, labels, alpha=self.cutmix_alpha, axes=self.cutmix_axes
                )
            else:
                data, labels_a, labels_b, lam = cutmix_data(
                    data, labels, alpha=self.cutmix_alpha, axes=self.cutmix_axes
                )
        else:
            labels_a, labels_b, lam = labels, labels, -1

        return data, labels_a, labels_b, lam

    def loss(self, predicts, labels_a, labels_b, lam):
        if lam == -1:
            return self.loss_function(predicts, labels_a)
        return mixup_criterion(self.loss_function, predicts, labels_a, labels_b, lam)

    def metric(self, predicts, labels_a, labels_b, lam):
        # 不进行 mix
        if lam == -1:
            # 如果 metric 是一系列函数
            if isinstance(self.metric_function, collections.abc.Sequence):
                return [metric_function(predicts, labels_a) for metric_function in self.metric_function]
            return self.metric_function(predicts, labels_a)
        else:
            if isinstance(self.metric_function, collections.abc.Sequence):
                return [
                    mixup_metric(metric_function, predicts, labels_a, labels_b, lam)
                    for metric_function in self.metric_function
                ]
            return mixup_metric(self.metric_function, predicts, labels_a, labels_b, lam)
