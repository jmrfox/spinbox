import matplotlib.pyplot as plt
import os
from spinbox import *
import itertools 

#from icecream import ic
# import matplotlib
# matplotlib.use('Agg', force=True)

os.environ["OMP_NUM_THREADS"] = "1"

dt = 0.001
n_samples = 1000
n_procs = os.cpu_count() 
# n_procs = 8
run_tag = '_test'  #start with a _
global_seed = 17

n_particles = 2
pairs_ij = interaction_indices(n_particles)

def make_test_states(manybody=False):
    """returns one body basis spin-isospin states for testing"""
    bra = AFDMCSpinIsospinState(n_particles, 'bra', np.zeros(shape=(n_particles, 1, 4))).randomize(100)
    ket = AFDMCSpinIsospinState(n_particles, 'ket', np.zeros(shape=(n_particles, 4, 1))).randomize(101)
    # ket = bra.copy().dagger()
    if manybody:
        bra = bra.to_manybody_basis()
        ket = ket.to_manybody_basis()
    return bra, ket

def make_potential(shape, scale=1.0, rng=None):
    if rng is not None:
        out = scale * rng.standard_normal(size=shape)
    else:
        out = scale * np.ones(shape=shape)
    return out

def make_asig(scale=1.0, rng=None):
    v =  make_potential((3, n_particles, 3, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[:, i, :, i] = 0
    return v 


def make_asigtau(scale=1.0, rng=None):
    v = make_potential((3, n_particles, 3, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[:, i, :, i] = 0
    return v 

def make_atau(scale=1.0, rng=None):
    v =  make_potential((n_particles, n_particles), scale=scale, rng=rng)
    for i in range(n_particles):
        v[i, i] = 0
    return v 

def make_vcoul(scale=1.0, rng=None):
    v =  make_potential((n_particles, n_particles), scale=0.1*scale, rng=rng)
    for i in range(n_particles):
        v[i, i] = 0
    return v 

def make_bls(scale=1.0, rng=None):
    v =  make_potential((3, n_particles, n_particles), scale=0.1*scale, rng=rng)
    for i in range(n_particles):
        v[:, i, i] = 0
    return v 

def make_all_potentials(scale=1.0, rng=None, mode='normal'):
    out = {}
    if mode=='normal':
        out['asig'] = make_asig(scale=scale, rng=rng)
        out['asigtau'] = make_asigtau(scale=scale, rng=rng)
        out['atau'] = make_atau(scale=scale, rng=rng)
        out['vcoul'] = make_vcoul(scale=scale, rng=rng)
        out['bls'] = make_bls(scale=scale, rng=rng)
        out['gls'] = np.sum(out['bls'], axis = 2)
    elif mode=='test':
        print("make_all_potentials IS IN TEST MODE!!!!")
        out['asig'] = make_asig(scale=0, rng=rng)
        out['asigtau'] = make_asigtau(scale=0, rng=rng)
        out['atau'] = make_atau(scale=0, rng=rng)
        out['vcoul'] = make_vcoul(scale=0., rng=rng)
        out['bls'] = make_bls(scale=0., rng=rng)
        out['gls'] = np.sum(out['bls'], axis = 2) 
    return out




def load_h2(manybody=False, data_dir = './data/h2/'):
    # data_dir = './data/h2/'
    c_i = read_from_file(data_dir+'fort.770', complex=True, shape=(2,4,1))
    c_f = read_from_file(data_dir+'fort.775', complex=True, shape=(2,4,1))
    ket = AFDMCSpinIsospinState(2, 'ket', c_i) 
    ket_f = AFDMCSpinIsospinState(2, 'ket', c_f) 
    if manybody:
        ket = ket.to_manybody_basis()
        ket_f = ket_f.to_manybody_basis()
    asig = read_from_file(data_dir+'fort.7701', shape=(3,2,3,2))
    asigtau = read_from_file(data_dir+'fort.7702', shape=(3,2,3,2))
    atau = read_from_file(data_dir+'fort.7703', shape=(2,2))
    vcoul = read_from_file(data_dir+'fort.7704', shape=(2,2))
    gls = read_from_file(data_dir+'fort.7705', shape=(3,2))
    asigls = read_from_file(data_dir+'fort.7706', shape=(3,2,3,2))

    pot_dict={}
    pot_dict['asig'] = asig
    pot_dict['asigtau'] = asigtau
    pot_dict['atau'] = atau
    pot_dict['vcoul'] = vcoul
    pot_dict['gls'] = gls
    pot_dict['asigls'] = asigls
    # return ket, asig, asigtau, atau, vcoul, gls, asigls
    return ket, pot_dict, ket_f