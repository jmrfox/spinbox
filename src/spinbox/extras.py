import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


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


def sigma_tau_operators_hilbert(n_particles):
    """Pauli sigma and tau operators in Hilbert space
    
    usage: 
    sigma, tau = sigma_tau_operators_hilbert(n_particles) 
    sigma[particle_index][dimension_index]
    """
    from .core import HilbertOperator
    sigma = [[HilbertOperator(n_particles, isospin=True).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    tau = [[HilbertOperator(n_particles, isospin=True).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
    return sigma, tau

def sigma_tau_matrices_product(n_particles):
    """Pauli sigma and tau matrices in product basis
    
    usage: 
    sigma, tau = sigma_tau_matrices_product(n_particles) 
    sigma[particle_index][dimension_index]

    note that I am not using the ProductOperator class here. This is done for memory efficiency.
    In the case of Hilbert space calculations, it makes sense to compute the operator matrices beforehand and store them.
    In the tensor-product basis, this would result in most of our memory being taken up by identity matrices.
    """
    from .core import repeated_kronecker_product, pauli
    sigma = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
    tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
    return sigma, tau
