import os
import sys

import torch

# project
from atomsurf.tasks.lba.model import LBANet
from atomsurf.utils.learning_utils import AtomPLModule
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import scipy

class LBAModule(AtomPLModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.model = LBANet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    def step(self, batch):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None
        labels = batch.g_ligand.y
        outputs = self(batch)
        if outputs==None:
            return None, None, None
        loss = self.criterion(outputs, labels)
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
        return loss, outputs, labels
    
    def metrics_reg(self,targets,predicts):
        mae = metrics.mean_absolute_error(y_true=targets,y_pred=predicts)
        rmse = metrics.mean_squared_error(y_true=targets,y_pred=predicts,squared=False)
        r = scipy.stats.mstats.pearsonr(targets, predicts)[0]

        x = [ [item] for item in predicts]
        lr = LinearRegression()
        lr.fit(X=x,y=targets)
        y_ = lr.predict(x)
        sd = (((targets - y_) ** 2).sum() / (len(targets) - 1)) ** 0.5

        return mae,rmse,r,sd

    def get_metrics(self, logits, labels, prefix):
        logits, labels = torch.cat(logits, dim=0), torch.cat(labels, dim=0)
        mse,rmse,r,sd = self.metrics_reg(logits, labels )
        print('mse,rmse,r,sd',mse,rmse,r,sd,prefix)
        self.log_dict({f"mse/{prefix}": mse,
                       f"rmse/{prefix}": rmse,
                       f"r/{prefix}": r,
                       f"sd/{prefix}": sd,
                       }, on_epoch=True, batch_size=len(logits))
