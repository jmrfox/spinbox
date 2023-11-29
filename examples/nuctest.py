import matplotlib.pyplot as plt
import sys, os

sys.path.append('/home/anl.gov/jfox/Projects/quap/src/')
from quap import *

#from icecream import ic
# import matplotlib
# matplotlib.use('Agg', force=True)

dt = 0.001
n_samples = 50000
n_procs = os.cpu_count() - 2 
run_tag = '_CoulOnly'  #start with a _

def make_test_states():
    """returns one body basis spin-isospin states for testing"""
    bra, ket = random_spinisospin_bra_ket(2)
    return bra, ket


# def make_A_matrices(random=False):
#     if random:
#         def random_A_matrices():
#             """this makes the same random matrices every time
#             for 2-particle systems, we only need the i=1, j=2 slice of the matrices
#             so, Asig[a,b] , Asigtau[a,b,c] , Atau[c] for a,b,c = 1,2,3
#             """
#             rng = np.random.default_rng(2023)
#             spread = 10
#             Asig = spread * rng.standard_normal(size=(3, 3))
#             Asigtau = spread * rng.standard_normal(size=(3, 3, 3))
#             Atau = spread * rng.standard_normal(size=3)
#             return Asig, Asigtau, Atau

#         return random_A_matrices()
#     else:
#         a = 1.0
#         Asig = a * np.ones((3, 3))
#         Asigtau = a * np.ones((3, 3, 3))
#         Atau = a * np.ones(3)
#         return Asig, Asigtau, Atau

def make_asig(scale=1.0, random=False):
    if random:
        rng = np.random.default_rng(1001)
        a = scale * rng.standard_normal(size=(3, 3))
    else:
        a = scale * np.ones(shape=(3, 3))
    return a

def make_asigtau(scale=1.0, random=False):
    if random:
        rng = np.random.default_rng(1002)
        a = scale * rng.standard_normal(size=(3, 3, 3))
    else:
        a = scale * np.ones(shape=(3, 3, 3))
    return a

def make_atau(scale=1.0, random=False):
    if random:
        rng = np.random.default_rng(1003)
        a = scale * rng.standard_normal(size=(3))
    else:
        a = scale * np.ones(shape=(3))
    return a

def make_vcoul(scale=1.0, random=False):
    if random:
        rng = np.random.default_rng(1004)
        a = scale * rng.standard_normal()
    else:
        a = scale 
    return a

def make_bls(scale=1.0, random=False):
    if random:
        rng = np.random.default_rng(1005)
        b = scale * rng.standard_normal(size=(3))
    else:
        b = scale * np.ones(shape=(3))
    return b


def make_potentials(scale=10.0, random=True):
    out = {}
    option = 'coul'
    if option=='all':
        out['asig'] = make_asig(scale=scale, random=random)
        out['asigtau'] = make_asigtau(scale=scale, random=random)
        out['atau'] = make_atau(scale=scale, random=random)
        out['vcoul'] = make_vcoul(scale=scale, random=random)
        out['bls'] = make_bls(scale=scale, random=random)
    elif option=='coul':
        out['asig'] = make_asig(scale=0., random=random)
        out['asigtau'] = make_asigtau(scale=0., random=random)
        out['atau'] = make_atau(scale=0., random=random)
        out['vcoul'] = make_vcoul(scale=1000., random=random)
        out['bls'] = make_bls(scale=0., random=random)
    return out



def plot_samples(X, range, filename, title):
    plt.figure(figsize=(5, 3))
    plt.hist(np.real(X), label='Re', alpha=0.5, bins=30, range=range, color='red')
    plt.hist(np.imag(X), label='Im', alpha=0.5, bins=30, range=range, color='blue')
    plt.title(title + f"\n<G> = {np.mean(X):.6f}  +/-  {np.std(X)/np.sqrt(len(X)):.6f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
