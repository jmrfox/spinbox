import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from spinbox import HilbertOperator, ProductOperator

def chistogram(X, filename, title, bins='fd', range=None):
    n = len(X)
    Xre = np.real(X)
    Xim = np.imag(X)
    mre, sre = np.mean(Xre), np.std(Xre)
    mim, sim = np.mean(Xim), np.std(Xim)
    fig, axs = plt.subplots(2, figsize=(5,8))
    fig.suptitle(title)
    axs[0].hist(Xre, label='Re', alpha=0.5, bins=bins, range=range, color='red')
    axs[1].hist(Xim, label='Im', alpha=0.5, bins=bins, range=range, color='blue')
    axs[0].set_title(rf"Re : $\mu$ = {mre:.6f}, $\sigma$ = {sre:.6f}, $\epsilon$ = {sre/np.sqrt(n):.6f}")
    axs[1].set_title(rf"Im : $\mu$ = {mim:.6f}, $\sigma$ = {sim:.6f}, $\epsilon$ = {sim/np.sqrt(n):.6f}")
    plt.tight_layout()
    plt.savefig(filename)

def spinor2(state='up', ketwise=True, seed=None):
    """returns spin coefficients, numpy array"""
    assert state in ['up', 'down', 'random', 'max']
    if state == 'up':
        sp = np.array([1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1], dtype=complex)
    elif state == 'random':
        rng = default_rng(seed=seed)
        sp = rng.uniform(-1, 1, 2) + 1j * rng.uniform(-1, 1, 2)
    elif state == 'max':
        sp = np.array([1, 1], dtype=complex)
    sp = sp / np.linalg.norm(sp)
    sp = sp.reshape((2,1))
    if not ketwise:
        sp = sp.T
    return sp

def spinor4(state='up', ketwise=True, seed=None):
    """returns spin-isospin coefficients, numpy array"""
    assert state in ['up', 'down', 'random', 'max']
    if state == 'up':
        sp = np.array([1, 0, 1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1, 0, 1], dtype=complex)
    elif state == 'random':
        rng = default_rng(seed=seed)
        sp = rng.uniform(-1, 1, 4) + 1j * rng.uniform(-1, 1, 4)
    elif state == 'max':
        sp = np.array([1, 1, 1, 1], dtype=complex)
    sp = sp / np.linalg.norm(sp)
    sp = sp.reshape((4,1))
    if not ketwise:
        sp = sp.T
    return sp


def nuclear_sigma_operators_hilbert(n_particles):
    """Pauli sigma operators in Hilbert space
    
    usage: 
    sigma = nuclear_sigma_operators_hilbert(n_particles) 
    sigma[particle_index][dimension_index]
    """
    return [[HilbertOperator(n_particles, isospin=True).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]

def nuclear_tau_operators_hilbert(n_particles):
    """Pauli tau operators in Hilbert space
    
    usage: 
    tau = nuclear_tau_operators_hilbert(n_particles) 
    tau[particle_index][dimension_index]
    """
    return[[HilbertOperator(n_particles, isospin=True).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]

def nuclear_sigma_matrices_product(n_particles):
    return

def nuclear_tau_matrices_product(n_particles):
    return
