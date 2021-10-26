import numpy as np

from pptb.optimizer.lr import CosineWarmup


def test_cosine_warmup():
    start_lr = 1.0
    max_steps = 100
    warmup_steps = int(max_steps * np.random.random() * 0.8)

    lr_scheduler = CosineWarmup(
        start_lr, T_max=max_steps, warmup_steps=warmup_steps, warmup_start_lr=0.0, last_epoch=-1
    )

    lr_history = []
    for _ in range(max_steps):
        lr_scheduler.step()
        lr_history.append(lr_scheduler())
    assert lr_history[warmup_steps - 1] == start_lr
