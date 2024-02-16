import matplotlib.pyplot as plt
import os
from quap import *
import itertools 

#from icecream import ic
# import matplotlib
# matplotlib.use('Agg', force=True)

os.environ["OMP_NUM_THREADS"] = 1


dt = 0.01
n_samples = 1000
# n_procs = os.cpu_count() - 1
n_procs = 8
run_tag = '_test'  #start with a _
global_seed = 17

n_particles = 2
pairs_ij = list(itertools.combinations(range(n_particles), 2))

def make_test_states(rng=None, manybody=False):
    """returns one body basis spin-isospin states for testing"""
    bra = OneBodyBasisSpinIsospinState(n_particles, 'bra', np.zeros(shape=(n_particles, 1, 4))).randomize(100)
    ket = OneBodyBasisSpinIsospinState(n_particles, 'ket', np.zeros(shape=(n_particles, 4, 1))).randomize(101)
    if manybody:
        bra = bra.to_many_body_state()
        ket = ket.to_many_body_state()
    return bra, ket

def make_potential(shape, scale=10.0, rng=None):
    if rng is not None:
        out = scale * rng.standard_normal(size=shape)
    else:
        out = scale * np.ones(shape=shape)
    return out

def make_asig(scale=10.0, rng=None):
    v =  make_potential((3, n_particles, 3, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[:, i, :, i] = 0
    return v 


def make_asigtau(scale=10.0, rng=None):
    v = make_potential((3, n_particles, 3, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[:, i, :, i] = 0
    return v 

def make_atau(scale=10.0, rng=None):
    v =  make_potential((n_particles, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[i, i] = 0
    return v 

def make_vcoul(scale=10.0, rng=None):
    v =  make_potential((n_particles, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[i, i] = 0
    return v 

def make_bls(scale=0.1, rng=None):
    v =  make_potential((3, n_particles, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[:, i, i] = 0
    return v 

def make_all_potentials(scale=10.0, rng=None):
    out = {}

    mode = 'ls_test'
    if mode=='normal':
        out['asig'] = make_asig(scale=scale, rng=rng)
        out['asigtau'] = make_asigtau(scale=scale, rng=rng)
        out['atau'] = make_atau(scale=scale, rng=rng)
        out['vcoul'] = make_vcoul(scale=scale, rng=rng)
        out['bls'] = make_bls(scale=scale, rng=rng)
        out['gls'] = np.sum(out['bls'], axis = 2) 
    elif mode=='ls_test':
        print("make_all_potentials IS IN TEST MODE!!!!")
        out['asig'] = make_asig(scale=0, rng=rng)
        out['asigtau'] = make_asigtau(scale=0, rng=rng)
        out['atau'] = make_atau(scale=0, rng=rng)
        out['vcoul'] = make_vcoul(scale=0, rng=rng)
        out['bls'] = make_bls(scale=scale, rng=rng)
        out['gls'] = np.sum(out['bls'], axis = 2) 
    return out


def plot_samples(X, filename, title, bins='auto', range=None):
    plt.figure(figsize=(7, 5))
    n = len(X)
    Xre = np.real(X)
    Xim = np.imag(X)
    mre, sre = np.mean(Xre), np.std(Xre)
    mim, sim = np.mean(Xim), np.std(Xim)
    plt.hist(Xre, label='Re', alpha=0.5, bins=bins, range=range, color='red')
    plt.hist(Xim, label='Im', alpha=0.5, bins=bins, range=range, color='blue')
    title += "\n" + rf"Re : $\mu$ = {mre:.6f}, $\sigma$ = {sre:.6f}, $\epsilon$ = {sre/np.sqrt(n):.6f}"
    title += "\n" + rf"Im : $\mu$ = {mim:.6f}, $\sigma$ = {sim:.6f}, $\epsilon$ = {sim/np.sqrt(n):.6f}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
