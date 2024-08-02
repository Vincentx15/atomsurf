import os
import re

import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor
import pytorch_lightning as pl

from atomsurf.protein.surfaces import SurfaceObject, SurfaceBatch
from atomsurf.protein.residue_graph import ResidueGraph, RGraphBatch
from atomsurf.protein.atom_graph import AtomGraph, AGraphBatch

from torch.optim.lr_scheduler import _LRScheduler, LinearLR, CosineAnnealingLR, SequentialLR, LambdaLR


class GaussianDistance(object):
    def __init__(self, start, stop, num_centers):
        self.filters = torch.linspace(start, stop, num_centers)
        self.var = (stop - start) / (num_centers - 1)

    def __call__(self, d):
        """
        :param d: shape (n,1)
        :return: shape (n,len(filters))
        """
        return torch.exp(-(((d - self.filters) / self.var) ** 2))


class SurfaceLoader:
    """
    This class is used to go load surfaces saved in a .pt file
    Based on the config it's given, it will load the corresponding features, optionnally expand them and
    populate a .x key with them.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.feature_expander = None
        if "gdf_expand" in config and config.gdf_expand:
            self.gauss_curv_gdf = GaussianDistance(start=-0.1, stop=0.1, num_centers=16)
            self.mean_curv_gdf = GaussianDistance(start=-0.5, stop=0.5, num_centers=16)
            self.feature_expander = {'geom_feats': self.gdf_expand}

    def gdf_expand(self, geom_features):
        gauss_curvs, mean_curvs, others = torch.split(geom_features, [1, 1, 20], dim=-1)
        gauss_curvs_gdf = self.gauss_curv_gdf(gauss_curvs)
        mean_curvs_gdf = self.mean_curv_gdf(mean_curvs)
        return torch.cat([gauss_curvs_gdf, mean_curvs_gdf, others], dim=-1)

    def load(self, surface_name):
        if not self.config.use_surfaces:
            return Data()
        try:
            # Timing for just one system: Loading: 1,297,731ns Expanding: 556,107ns, so approx 3/4 of time is loading
            surface = torch.load(os.path.join(self.data_dir, f"{surface_name}.pt"))
            with torch.no_grad():
                surface.expand_features(remove_feats=True,
                                        feature_keys=self.config.feat_keys,
                                        oh_keys=self.config.oh_keys,
                                        feature_expander=self.feature_expander)
                if torch.isnan(surface.x).any() or torch.isnan(surface.verts).any():
                    return None
            return surface
        except Exception:
            return None


class GraphLoader:
    """
    This class is used to go load graphs saved in a .pt file
    It is similar to the Surface one, but can also be extended with ESM or ProNet features
    Based on the config it's given, it will load the corresponding features and populate a .x key with them.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.esm_dir = config.esm_dir
        self.use_esm = config.use_esm
        self.feature_expander = None

    def load(self, graph_name):
        if not self.config.use_graphs:
            return Data()
        try:
            graph = torch.load(os.path.join(self.data_dir, f"{graph_name}.pt"))
            feature_keys = self.config.feat_keys
            if self.use_esm:
                esm_feats_path = os.path.join(self.esm_dir, f"{graph_name}_esm.pt")
                esm_feats = torch.load(esm_feats_path)
                graph.features.add_named_features('esm_feats', esm_feats)
                if feature_keys != 'all':
                    feature_keys.append('esm_feats')
            graph.expand_features(remove_feats=True,
                                  feature_keys=feature_keys,
                                  oh_keys=self.config.oh_keys,
                                  feature_expander=self.feature_expander)
            if torch.isnan(graph.x).any() or torch.isnan(graph.node_pos).any():
                return None
        except Exception:
            return None
        return graph


def update_model_input_dim(cfg, dataset_temp):
    # Useful to create a Model of the right input dims
    try:
        from omegaconf import open_dict
        for i, example in enumerate(dataset_temp):
            if example is not None:
                with open_dict(cfg):
                    feat_encoder_kwargs = cfg.encoder.blocks[0].kwargs
                    feat_encoder_kwargs['graph_feat_dim'] = example.graph.x.shape[1]
                    feat_encoder_kwargs['surface_feat_dim'] = example.surface.x.shape[1]
                break
            if i > 50:
                raise Exception('All data returned by Dataloader is None')
    except Exception as e:
        raise Exception('Could not update model input dims because of error: ', e)


class AtomBatch(Data):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_times = []

    @staticmethod
    def batch_keys(batch, key):
        item = batch[key][0]
        if isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        elif bool(re.search('(locs_left|locs_right|neg_stack|pos_stack)', key)):
            batch[key] = batch[key]
        elif key == 'labels_pip':
            batch[key] = torch.cat(batch[key])
        elif torch.is_tensor(item):
            try:
                # If they are all the same size
                batch[key] = torch.stack(batch[key])
            except Exception:
                batch[key] = batch[key]
        elif isinstance(item, SurfaceObject):
            batch[key] = SurfaceBatch.batch_from_data_list(batch[key])
        elif isinstance(item, ResidueGraph):
            batch[key] = RGraphBatch.batch_from_data_list(batch[key])
        elif isinstance(item, AtomGraph):
            batch[key] = AGraphBatch.batch_from_data_list(batch[key])
        elif isinstance(item, Data):
            batch[key] = Batch.from_data_list(batch[key])
            batch[key] = batch[key] if batch[key].num_graphs > 0 else None
        elif isinstance(item, list):
            batch[key] = batch[key]
        elif isinstance(item, str):
            batch[key] = batch[key]
        elif isinstance(item, SparseTensor):
            batch[key] = batch[key]
        else:
            raise ValueError(f"Unsupported attribute type: {type(item)}, item : {item}, key : {key}")

    # @classmethod
    def from_data_list(self, data_list):
        import time

        t0 = time.perf_counter()
        # Filter out None
        data_list = [x for x in data_list if x is not None]

        batch = AtomBatch()
        if len(data_list) == 0:
            batch.num_graphs = 0
            return batch

        # Get all keys
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        # Create a data containing lists of items for every key
        for key in keys:
            batch[key] = []
        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        del batch.all_times
        # Organize the keys together
        for key in batch.keys:
            AtomBatch.batch_keys(batch, key)
        batch = batch.contiguous()
        batch.num_graphs = len(data_list)
        a = time.perf_counter() - t0
        # print("Time batching:", a)
        self.all_times.append(a)
        import numpy as np
        if not len(self.all_times) % 100:
            print("Whole time batching", np.sum(self.all_times))
        return batch


class AtomPLModule(pl.LightningModule):
    """
    A generic PL module to subclass
    """

    def __init__(self, cfg) -> None:
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
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits, labels))
        return loss

    def validation_step(self, batch, batch_idx: int):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.val_res.append((logits, labels))

    def test_step(self, batch, batch_idx: int):
        if batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            return None
        self.log_dict({"loss/test": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.test_res.append((logits, labels))

    def on_train_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.train_res)]
        self.get_metrics(logits, labels, 'train')
        self.train_res = list()
        pass

    def on_validation_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.val_res)]
        self.get_metrics(logits, labels, 'val')
        self.val_res = list()
        pass

    def on_test_epoch_end(self) -> None:
        logits, labels = [list(i) for i in zip(*self.test_res)]
        self.get_metrics(logits, labels, 'test')
        self.test_res = list()
        pass

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch = batch.to(device)
        return batch

    def configure_optimizers(self):
        opt_params = self.hparams.cfg.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        # scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                            patience=opt_params.patience,
        #                                                            factor=opt_params.factor,
        #                                                            mode='max')
        scheduler_obj = get_lr_scheduler(scheduler=self.hparams.cfg.lr_scheduler,
                                         optimizer=optimizer,
                                         warmup_epochs=self.hparams.cfg.warmup_epochs,
                                         total_epochs=self.hparams.cfg.epochs,
                                         eta_min=self.hparams.cfg.lr_eta_min)
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': self.hparams.cfg.train.to_monitor,
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]


def get_lr_scheduler(scheduler, optimizer, warmup_epochs, total_epochs, eta_min=1E-8):
    warmup_scheduler = LinearLR(optimizer,
                                start_factor=1E-3,
                                total_iters=warmup_epochs)

    if scheduler == 'PolynomialLRWithWarmup':
        decay_scheduler = PolynomialLR(optimizer,
                                       total_iters=total_epochs - warmup_epochs,
                                       power=1)
    elif scheduler == 'CosineAnnealingLRWithWarmup':
        decay_scheduler = CosineAnnealingLR(optimizer,
                                            T_max=total_epochs - warmup_epochs,
                                            eta_min=eta_min)
    elif scheduler == 'constant':
        lambda1 = lambda epoch: 1.0  # noqa
        decay_scheduler = LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        raise NotImplementedError

    return SequentialLR(optimizer,
                        schedulers=[warmup_scheduler, decay_scheduler],
                        milestones=[warmup_epochs])


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
