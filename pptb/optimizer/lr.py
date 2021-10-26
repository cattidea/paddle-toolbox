import paddle


class CosineWarmup(paddle.optimizer.lr.LinearWarmup):
    """
    @refs: https://github.com/PaddlePaddle/PaddleClas/blob/a6d927a122387642d04bb0ebb5785e1d7c74f78f/ppcls/optimizer/learning_rate.py#L66-L102
    """

    def __init__(
        self,
        learning_rate,
        total_steps,
        warmup_steps=1,
        warmup_start_lr=0.0,
        cosine_end_lr=0.0,
        last_epoch=-1,
        verbose=False,
    ):
        lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=learning_rate,
            T_max=total_steps - warmup_steps,
            eta_min=cosine_end_lr,
            last_epoch=last_epoch,
            verbose=verbose,
        )
        super().__init__(
            learning_rate=lr_scheduler,
            warmup_steps=warmup_steps,
            start_lr=warmup_start_lr,
            end_lr=learning_rate,
            last_epoch=last_epoch,
            verbose=verbose,
        )
