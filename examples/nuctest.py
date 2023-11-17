import matplotlib.pyplot as plt

from quap import *
from icecream import ic
# import matplotlib
# matplotlib.use('Agg', force=True)
import os

dt = 0.0002
n_samples = 400_000
n_procs = os.cpu_count() - 2
run_tag = '_vc1000'

def make_test_states():
    """returns one body basis spin-isospin states for testing"""
    bra, ket = random_spinisospin_bra_ket(2)
    return bra, ket


def make_A_matrices(random=False):
    if random:
        def random_A_matrices():
            """this makes the same random matrices every time
            for 2-particle systems, we only need the i=1, j=2 slice of the matrices
            so, Asig[a,b] , Asigtau[a,b,c] , Atau[c] for a,b,c = 1,2,3
            """
            rng = np.random.default_rng(2023)
            spread = 10
            Asig = spread * rng.standard_normal(size=(3, 3))
            Asigtau = spread * rng.standard_normal(size=(3, 3, 3))
            Atau = spread * rng.standard_normal(size=3)
            return Asig, Asigtau, Atau

        return random_A_matrices()
    else:
        a = 1.0
        Asig = a * np.ones((3, 3))
        Asigtau = a * np.ones((3, 3, 3))
        Atau = a * np.ones(3)
        return Asig, Asigtau, Atau


def make_potentials(random=False):
    Asig, Asigtau, Atau = make_A_matrices(random=True)
    Vcoul = 1000.0
    return Asig, Asigtau, Atau, Vcoul


def plot_samples(X, range, filename, title):
    plt.figure(figsize=(5, 3))
    plt.hist(np.real(X), label='Re', alpha=0.5, bins=30, range=range, color='red')
    plt.hist(np.imag(X), label='Im', alpha=0.5, bins=30, range=range, color='blue')
    plt.title(title + f"\n<G> = {np.mean(X):.6f}  +/-  {np.std(X)/np.sqrt(len(X)):.6f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
