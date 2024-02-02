from multiprocessing.pool import Pool
import numpy as np
from tqdm import tqdm
from time import time
import os
from cProfile import Profile
from pstats import SortKey, Stats

from itertools import starmap


# rng = np.random.default_rng(seed=1)

def task(n, dim):
    rng = np.random.default_rng(seed=1)
    out = np.identity(dim)
    for _ in range(n):
        out = out @ rng.standard_normal(size=(dim, dim)) / np.linalg.norm(out)
        # print(out)
    return np.sum(out)

def diag(dim):
    rng = np.random.default_rng(seed=1)
    x = rng.standard_normal(size=(dim,dim))
    result, _ = np.linalg.eig(x)
    return result[0]


if __name__=="__main__":
    # print(result)

    # n_procs = os.cpu_count() - 1
    n_procs = 16

    with Profile() as profile:
        t0 = time()

        do_parallel = True
        if do_parallel:
            with Pool(processes=n_procs) as pool:
                # result = pool.starmap_async(task, tqdm([(100, 100)]*n_procs, leave=True)).get()
                result = pool.starmap_async(diag, tqdm([(1000,)]*n_procs, leave=True)).get()
        else:
            result = list(starmap(task, [(10000, 100)]*n_procs))
            


        print('time =', time() - t0)
        print(np.mean(result))
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
