from multiprocessing.pool import Pool, ThreadPool
import numpy as np
from tqdm import tqdm
from time import time
import os
from cProfile import Profile
from pstats import SortKey, Stats

from itertools import starmap


def diag(dim):
    rng = np.random.default_rng(seed=1)
    x = rng.standard_normal(size=(dim,dim))
    result, _ = np.linalg.eig(x)
    return result[0]

def matmuls(n, dim):
    rng = np.random.default_rng(seed=1)
    out = np.identity(dim)
    for _ in range(n):
        out = out @ rng.standard_normal(size=(dim, dim)) / np.linalg.norm(out)
        # print(out)
    return np.mean(out)

def factors(n):
    from sympy import factorint
    rng = np.random.default_rng(seed=1)
    k = 10
    out = []
    for _ in range(n):
        x = int( rng.uniform(10**k, 10**(k+1)) )
        res = factorint(x)
        out.append(sum(list(res)))
    return np.mean(out)

do_parallel = True
n_procs = 4
n_jobs = 100

def task(i):
    # return matmuls(100,100)
    return factors(1)


inputs = range(n_jobs)

def main():
    with Profile() as profile:
        t0 = time()
        if do_parallel:
            with ThreadPool(processes=n_procs) as pool:
                result = pool.map(task, tqdm(inputs) )
        else:
            result = list(map(task, tqdm(inputs) ) )

        print('total time =', time() - t0)
        print(np.mean(result))
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()



if __name__=="__main__":
    main()