# import os
# import sys
#
# import numpy as np
# import time
# import torch
# from torch_geometric.data import Data
#
# if __name__ == '__main__':
#     script_dir = os.path.dirname(os.path.realpath(__file__))
#     sys.path.append(os.path.join(script_dir, '..', '..', '..'))
#
# from atomsurf.utils.data_utils import SurfaceLoader, GraphLoader, AtomBatch, update_model_input_dim
# from atomsurf.tasks.masif_site.data_loader import MasifSiteDataset, MasifSiteDataModule
#
# ###################### GET DATA
#
# script_dir = os.path.dirname(os.path.realpath(__file__))
# masif_site_data_dir = os.path.join(script_dir, '..', '..', '..', 'data', 'masif_site')
#
# # SURFACE
# cfg_surface = Data()
# cfg_surface.use_surfaces = True
# cfg_surface.feat_keys = 'all'
# cfg_surface.oh_keys = 'all'
# cfg_surface.gdf_expand = True
# # cfg_surface.data_dir = os.path.join(masif_site_data_dir, 'surfaces_1.0_False')
# cfg_surface.data_dir = os.path.join(masif_site_data_dir, 'surfaces_0.1_False')
# surface_loader = SurfaceLoader(cfg_surface)
#
# # GRAPHS
# cfg_graph = Data()
# cfg_graph.use_graphs = True
# cfg_graph.feat_keys = 'all'
# cfg_graph.oh_keys = 'all'
# cfg_graph.esm_dir = os.path.join(masif_site_data_dir, '01-benchmark_esm_embs')
# cfg_graph.use_esm = True
# cfg_graph.data_dir = os.path.join(masif_site_data_dir, 'rgraph')
# # cfg_graph.data_dir = os.path.join(masif_site_data_dir, 'agraph')
# graph_loader = GraphLoader(cfg_graph)
#
# test_systems_list = os.path.join(masif_site_data_dir, 'test_list.txt')
# test_sys = [name.strip() for name in open(test_systems_list, 'r').readlines()]
# dataset = MasifSiteDataset(test_sys, surface_loader, graph_loader)
# a = dataset[0]
#
# loader_cfg = Data(num_workers=0, batch_size=4, pin_memory=False, prefetch_factor=2, shuffle=False)
# simili_cfg = Data(cfg_surface=cfg_surface, cfg_graph=cfg_graph, loader=loader_cfg)
# datamodule = MasifSiteDataModule(cfg=simili_cfg)
# loader = datamodule.train_dataloader()
#


# std
import sys
from pathlib import Path
# 3p
import hydra
import torch
import pytorch_lightning as pl

# project
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from pl_model import MasifSiteModule
from data_loader import MasifSiteDataModule
from atomsurf.tasks.masif_site.model import MasifSiteNet
from atomsurf.tasks.masif_site.pl_model import masif_site_loss
import time


@hydra.main(config_path="conf", config_name="config")
def main(cfg=None):
    import numpy as np
    np.random.seed(42)
    cfg.cfg_surface.data_dir = "../../../data/masif_site/surfaces_1.0_False"
    # cfg.cfg_surface.data_dir = "../../../data/masif_site/surfaces_0.1_False"
    # cfg.cfg_graph.data_dir = "../../../data/masif_site/agraph"
    cfg.cfg_graph.data_dir = "../../../data/masif_site/rgraph"
    cfg.cfg_graph.esm_dir = '../../../data/masif_site/01-benchmark_esm_embs'
    cfg.cfg_graph.use_esm = False
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

    t0 = None
    times_model = []
    for i, batch in enumerate(loader):
        # There is a loader creation time
        if t0 is None:
            t0 = time.perf_counter()
        batch = batch.to(device)
        a = batch.pocket
        t_tot = time.perf_counter()
        t1 = time.perf_counter()
        labels = torch.concatenate(batch.label)
        out_surface_batch = model(batch)
        outputs = out_surface_batch.x.flatten()
        torch.cuda.synchronize()
        time_model = time.perf_counter() - t1
        print("Time fwd", i, time_model)

        t1 = time.perf_counter()
        loss, outputs_concat, labels_concat = masif_site_loss(outputs, labels)
        torch.cuda.synchronize()
        time_model = time.perf_counter() - t1
        print("Time loss", i, time_model)

        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        time_model = time.perf_counter() - t1
        print("Time bwd", i, time_model)
        time_model = time.perf_counter() - t_tot
        print("Time total model", i, time_model)

        time_elapsed = time.perf_counter() - t0
        mean_time_elapsed = time_elapsed / (i + 1)
        print("Mean time elapsed", mean_time_elapsed)
        print()

        # times_model.append(time_model)
        # if not i % 10:
        #     print(f"Done {i}/{len(loader)}")
        #     print("Time model", np.sum(times_model))

        if i > 400:
            break
    print('total : ', time.time() - t0)


main()
