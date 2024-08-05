import os
import errno

import time
from torch.utils.data import DataLoader


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def do_all(dataset, num_workers=4, prefetch_factor=100):
    """
    Given a pytorch dataset, uses pytorch multiprocessing system for easy parallelization.
    :param dataset:
    :param num_workers:
    :param prefetch_factor:
    :return:
    """
    prefetch_factor = prefetch_factor if num_workers > 0 else 2
    dataloader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=1,
                            prefetch_factor=prefetch_factor,
                            collate_fn=lambda x: x[0])
    total_success = 0
    t0 = time.time()
    for i, success in enumerate(dataloader):
        # if i > 5: break
        total_success += int(success)
        if not i % 5:
            print(f'Processed {i + 1}/{len(dataloader)}, in {time.time() - t0:.3f}s, with {total_success} successes')
