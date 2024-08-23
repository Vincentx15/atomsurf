from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, LinearLR, CosineAnnealingLR, SequentialLR, \
    LambdaLR


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) /
                        (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]


def get_lr_scheduler(scheduler, optimizer, **kwargs):
    warmup_epochs = kwargs['warmup_epochs'] if 'warmup_epochs' in kwargs else 0

    if scheduler == 'PolynomialLRWithWarmup':
        total_epochs = kwargs['total_epochs']
        decay_scheduler = PolynomialLR(optimizer,
                                       total_iters=total_epochs - warmup_epochs,
                                       power=1)
    elif scheduler == 'CosineAnnealingLRWithWarmup':
        total_epochs = kwargs['total_epochs']
        eta_min = kwargs['eta_min'] if 'eta_min' in kwargs else 1e-8
        decay_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=total_epochs - warmup_epochs,
                                            eta_min=eta_min)
    elif scheduler == 'constant':
        lambda1 = lambda epoch: 1.0  # noqa
        decay_scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    elif scheduler == 'ReduceLROnPlateau':
        decay_scheduler = ReduceLROnPlateau(optimizer,
                                            patience=kwargs['patience'],
                                            factor=kwargs['factor'],
                                            mode='max')
    else:
        raise NotImplementedError

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(optimizer,
                                    start_factor=1E-3,
                                    total_iters=warmup_epochs)
        return SequentialLR(optimizer,
                            schedulers=[warmup_scheduler, decay_scheduler],
                            milestones=[warmup_epochs])
    else:
        return decay_scheduler
