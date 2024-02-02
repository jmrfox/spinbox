from multiprocessing.pool import Pool
import numpy as np
from tqdm import tqdm
from time import time
import os
from cProfile import Profile
from pstats import SortKey, Stats


rng = np.random.default_rng()


def task(n, dim):
    out = np.identity(dim)
    for _ in range(n):
        out = out @ rng.standard_normal(size=(dim, dim)) / np.sqrt(dim)
        # print(out)
    return np.sum(out)

def diag(dim):
    x = rng.standard_normal(size=(dim,dim))
    result, _ = np.linalg.eig(x)
    return result[0]


if __name__=="__main__":
    # print(result)

    # n_procs = os.cpu_count() - 1
    n_procs = 64

    with Profile() as profile:
        t0 = time()

        with Pool(processes=n_procs) as pool:
            # result = pool.starmap_async(task, tqdm([(100, 100)]*n_procs, leave=True)).get()
            result = pool.starmap_async(diag, tqdm([(1000,)]*n_procs, leave=True)).get()
        
        print('time =', time() - t0)
        print(np.mean(result))
        Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
