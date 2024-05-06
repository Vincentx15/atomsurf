import os
import re
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_sparse import SparseTensor
import pytorch_lightning as pl

from atomsurf.protein.surfaces import SurfaceObject
from atomsurf.protein.residue_graph import ResidueGraph
from atomsurf.protein.atom_graph import AtomGraph


class SurfaceLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir

    def load(self, pocket_name):
        if not self.config.use_surfaces:
            return Data()
        try:
            surface = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
            surface.expand_features(remove_feats=True, feature_keys=self.config.feat_keys, oh_keys=self.config.oh_keys)
            return surface
        except Exception as e:
            return None


class GraphLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = config.data_dir
        self.esm_dir = config.esm_dir
        self.use_esm = config.use_esm

    def load(self, pocket_name):
        if not self.config.use_graphs:
            return Data()
        try:
            graph = torch.load(os.path.join(self.data_dir, f"{pocket_name}.pt"))
            feature_keys = self.config.feat_keys
            if self.use_esm:
                esm_feats_path = os.path.join(self.esm_dir, f"{pocket_name}_esm.pt")
                esm_feats = torch.load(esm_feats_path)
                graph.features.add_named_features('esm_feats', esm_feats)
                if feature_keys != 'all':
                    feature_keys.append('esm_feats')
            graph.expand_features(remove_feats=True, feature_keys=feature_keys, oh_keys=self.config.oh_keys)
        except Exception as e:
            return None
        return graph


def update_model_input_dim(cfg, dataset_temp):
    # Useful to create a Model of the right input dims
    try:
        from omegaconf import open_dict
        for example in dataset_temp:
            if example is not None:
                with open_dict(cfg):
                    feat_encoder_kwargs = cfg.encoder.blocks[0].kwargs
                    feat_encoder_kwargs['graph_feat_dim'] = example.graph.x.shape[1]
                    feat_encoder_kwargs['surface_feat_dim'] = example.surface.x.shape[1]
                break
    except Exception as e:
        print('Could not update model input dims because of error: ', e)


class AtomBatch(Data):
    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch
        self.__data_class__ = Data

    @staticmethod
    def from_data_list(data_list):
        # Filter out None
        data_list = [x for x in data_list if x is not None]
        if len(data_list) == 0:
            return None
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        batch = AtomBatch()
        batch.__data_class__ = data_list[0].__class__

        for key in keys:
            batch[key] = []

        for _, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        for key in batch.keys:
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
                except:
                    batch[key] = batch[key]
            elif isinstance(item, SurfaceObject):
                batch[key] = SurfaceObject.batch_from_data_list(batch[key])
            elif isinstance(item, ResidueGraph):
                batch[key] = ResidueGraph.batch_from_data_list(batch[key])
            elif isinstance(item, AtomGraph):
                batch[key] = AtomGraph.batch_from_data_list(batch[key])
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

        batch = batch.contiguous()
        return batch

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


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
        pass
        # self.log_dict({f"accuracy_balanced/{prefix}": 0, }, on_epoch=True, batch_size=len(logits))

    def step(self, batch):

        if batch is None:
            return None, None, None
        labels = batch.label
        # return None, None, None
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        if torch.isnan(loss).any():
            print('Nan loss')
            return None, None, None
        return loss, outputs, labels

    def on_after_backward(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'Detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        if loss is None:
            return None
        self.log_dict({"loss/train": loss.item()},
                      on_step=True, on_epoch=True, prog_bar=False, batch_size=len(logits))
        self.train_res.append((logits, labels))
        return loss

    def validation_step(self, batch, batch_idx: int):
        self.model.train()
        loss, logits, labels = self.step(batch)
        if loss is None or logits.isnan().any() or labels.isnan().any():
            print("validation step skipped!")
            return None
        self.log_dict({"loss/val": loss.item()},
                      on_step=False, on_epoch=True, prog_bar=True, batch_size=len(logits))
        self.val_res.append((logits, labels))

    def test_step(self, batch, batch_idx: int):
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

    def configure_optimizers(self):
        opt_params = self.hparams.cfg.optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=opt_params.lr)
        scheduler_obj = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   patience=opt_params.patience,
                                                                   factor=opt_params.factor,
                                                                   mode='max')
        scheduler = {'scheduler': scheduler_obj,
                     'monitor': self.hparams.cfg.train.to_monitor,
                     'interval': "epoch",
                     'frequency': 1,
                     "strict": True,
                     'name': "epoch/lr"}
        # return optimizer
        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if batch is None:
            return None
        batch = batch.to(device)
        return batch
