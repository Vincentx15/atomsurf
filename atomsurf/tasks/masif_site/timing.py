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

# Input Encoder    0.0005778130002909165
# DiffusionNet     0.0074715529999593855
# Graph encoding   0.0014750000000276486
# Mess passing     0.004244882999955735
# DiffusionNet     0.0020042539999849396
# Graph encoding   0.0006412389998331491
# Mess passing     0.0012595930002134992
# DiffusionNet     0.002089285999772983
# Graph encoding   0.0006220519999260432
# Mess passing     0.001272218000394787
# DiffusionNet     0.0019242710000071384
# Graph encoding   0.0006056809997971868
# Mess passing     0.0023985530001482402
# Time fwd 4 0.02717120999977851
# Time loss 4 0.0005937789997005893
# Time bwd 4 0.03451869299988175
# Time total model 4 0.062322944999777974
# => Bottlenecks are creating the lists for first diffnet and MP (can be addressed to gain 0.008 ~ 1/3rd of fwd)
# => Without cuda blocking, this is approximately the same throughput as data loading (can be made a bit faster too)

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
