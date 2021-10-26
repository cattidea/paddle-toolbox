import numpy as np

from pptb.optimizer.lr import CosineWarmup


def test_cosine_warmup():
    max_lr = 1
    warmup_start_lr = max_lr * np.random.random() * 0.8
    cosine_end_lr = max_lr * np.random.random() * 0.2
    total_steps = 100
    warmup_steps = int(total_steps * np.random.random() * 0.8)

    lr_scheduler = CosineWarmup(
        max_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        cosine_end_lr=cosine_end_lr,
        last_epoch=-1,
    )

    lr_history = []
    lr_history.append(lr_scheduler())
    for i in range(total_steps):
        lr_scheduler.step()
        lr_history.append(lr_scheduler())
    assert lr_history[warmup_steps] == max_lr
    assert lr_history[0] == warmup_start_lr
    assert lr_history[-1] == cosine_end_lr
