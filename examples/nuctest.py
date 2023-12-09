import matplotlib.pyplot as plt
import os
from quap import *

#from icecream import ic
# import matplotlib
# matplotlib.use('Agg', force=True)

dt = 0.001
n_samples = 10_000
n_procs = os.cpu_count() - 2 
run_tag = '_test'  #start with a _
global_seed = 17

def make_test_states(rng=None):
    """returns one body basis spin-isospin states for testing"""
    bra, ket = random_spinisospin_bra_ket(2, bra_seed=global_seed, ket_seed=global_seed+1)
    return bra, ket

def make_potential(size, scale=10.0, rng=None):
    if rng is not None:
        out = scale * rng.standard_normal(size=size)
    else:
        out = scale * np.ones(shape=size)
    return out

def make_asig(scale=10.0, rng=None):
    return make_potential((3, 3), scale=scale, rng=rng)

def make_asigtau(scale=10.0, rng=None):
    return make_potential((3, 3), scale=scale, rng=rng)

def make_atau(scale=10.0, rng=None):
    return make_potential((), scale=scale, rng=rng)

def make_vcoul(scale=10.0, rng=None):
    return make_potential((), scale=scale, rng=rng)

def make_bls(scale=10.0, rng=None):
    return make_potential((3), scale=scale, rng=rng)

def make_all_potentials(scale=10.0, rng=None):
    out = {}
    option = 'all'

    if option=='all':
        out['asig'] = make_asig(scale=scale, rng=rng)
        out['asigtau'] = make_asigtau(scale=scale, rng=rng)
        out['atau'] = make_atau(scale=scale, rng=rng)
        out['vcoul'] = make_vcoul(scale=scale, rng=rng)
        out['bls'] = make_bls(scale=scale, rng=rng)
    elif option=='coul':
        out['asig'] = make_asig(scale=0., rng=rng)
        out['asigtau'] = make_asigtau(scale=0., rng=rng)
        out['atau'] = make_atau(scale=0., rng=rng)
        out['vcoul'] = make_vcoul(scale=scale, rng=rng)
        out['bls'] = make_bls(scale=0., rng=rng)
    elif option=='test':
        out['asig'] = make_asig(scale=scale, rng=rng)
        out['asigtau'] = make_asigtau(scale=0., rng=rng)
        out['atau'] = make_atau(scale=0., rng=rng)
        out['vcoul'] = make_vcoul(scale=0., rng=rng)
        out['bls'] = make_bls(scale=0., rng=rng)
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
