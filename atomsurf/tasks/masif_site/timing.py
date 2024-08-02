import os
import sys

from pathlib import Path
import hydra
import time
import torch

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from atomsurf.tasks.masif_site.data_loader import MasifSiteDataModule
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.tasks.masif_site.pl_model import masif_site_loss


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    import numpy as np
    np.random.seed(42)
    # cfg.cfg_surface.data_dir = "../../../data/masif_site/surfaces_1.0_False"
    cfg.cfg_surface.data_dir = "../../../data/masif_site/surfaces_0.1_False"
    # cfg.cfg_graph.data_dir = "../../../data/masif_site/agraph"
    cfg.cfg_graph.data_dir = "../../../data/masif_site/rgraph"
    cfg.cfg_graph.esm_dir = '../../../data/masif_site/01-benchmark_esm_embs'
    cfg.cfg_graph.use_esm = True
    cfg.loader.num_workers = 2
    cfg.loader.batch_size = 4
    cfg.loader.shuffle = False

    # init datamodule
    datamodule = MasifSiteDataModule(cfg)
    loader = datamodule.train_dataloader()
    # init model
    model = MasifSiteNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    times_model = []
    t0 = time.perf_counter()
    for i, batch in enumerate(loader):
        t_tot = time.perf_counter()

        batch = batch.to(device)
        labels = torch.concatenate(batch.label)
        out_surface_batch = model(batch)
        outputs = out_surface_batch.x.flatten()
        loss, outputs_concat, labels_concat = masif_site_loss(outputs, labels)
        loss.backward()
        time_model = time.perf_counter() - t_tot
        print("Time total model", i, time_model)
        print()
        if i > 400:
            break
    print('total : ', time.time() - t0)


main()
