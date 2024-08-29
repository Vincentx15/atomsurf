import torch
import pytorch_lightning as pl

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
    total_epochs = kwargs['num_epochs']

    if scheduler == 'PolynomialLR':
        decay_scheduler = PolynomialLR(optimizer,
                                       total_iters=total_epochs - warmup_epochs,
                                       power=1)
    elif scheduler == 'CosineAnnealingLR':
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


class AtomPLModule(pl.LightningModule):
    """
    A generic PL module to subclass
    """

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_res = list()
        self.val_res = list()
        self.test_res = list()

    def get_metrics(self, logits, labels, prefix):
        # Do something like: self.log_dict({f"accuracy_balanced/{prefix}": 0, }, on_epoch=True, batch_size=len(logits))
        pass

    def step(self, batch):
        raise NotImplementedError("Each subclass of AtomPLModule must implement the `step` method")

    def on_after_backward(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print('Detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits.detach().cpu(), labels.detach().cpu()))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.val_res.append((logits.detach().cpu(), labels.detach().cpu()))

    def test_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.test_res.append((logits.detach().cpu(), labels.detach().cpu()))

    def on_train_epoch_end(self) -> None:
        if len(self.train_res) > 0:
            logits, labels = [list(i) for i in zip(*self.train_res)]
            self.get_metrics(logits, labels, 'train')
            self.train_res = list()

    def on_validation_epoch_end(self) -> None:
        if len(self.train_res) > 0:
            logits, labels = [list(i) for i in zip(*self.val_res)]
            self.get_metrics(logits, labels, 'val')
            self.val_res = list()

    def on_test_epoch_end(self) -> None:
        if len(self.train_res) > 0:
            logits, labels = [list(i) for i in zip(*self.test_res)]
            self.get_metrics(logits, labels, 'test')
            self.test_res = list()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        batch = batch.to(device)
        return batch

    def configure_optimizers(self):
        opt_params = self.hparams.cfg.optimizer
        sched_params = self.hparams.cfg.scheduler
        num_epochs = self.hparams.cfg.epochs
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)

        # Set up the scheduler.
        # For schedulers others than ReduceLROnPlateau, we don't need to monitor any metrics, so we don't need them
        scheduler_obj = get_lr_scheduler(scheduler=sched_params.name,
                                         optimizer=optimizer,
                                         num_epochs=num_epochs,
                                         **sched_params)
        needs_monitor = sched_params.name == 'ReduceLROnPlateau'
        monitor = None if not needs_monitor else self.hparams.cfg.train.to_monitor

        # Update scheduler params
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': monitor,
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": needs_monitor,
                     'name': "epoch/lr"}

        # return optimizer and scheduler
        return [optimizer], [scheduler]
