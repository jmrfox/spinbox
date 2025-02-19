import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng


def chistogram(X, filename=None, title=None, bins='fd', range=None):
    """Complex histogram

    :param X: Set of complex numbers
    :type X: iterable 
    :param filename: filename of plot, including suffix (.pdf)
    :type filename: str
    :param title: Plot title
    :type title: str
    :param bins: binning algorithm (see matplotlib.pyplot.hist), defaults to 'fd' (Freedman-Diaconis)
    :type bins: str, optional
    :param range: fixed range to plot, defaults to None
    :type range: tuple, optional
    """    
    n = len(X)
    Xre = np.real(X)
    Xim = np.imag(X)
    mre, sre = np.mean(Xre), np.std(Xre)
    mim, sim = np.mean(Xim), np.std(Xim)
    fig, axs = plt.subplots(2, figsize=(5,8))
    if title:
        fig.suptitle(title)
    axs[0].hist(Xre, label='Re', alpha=0.5, bins=bins, range=range, color='red')
    axs[1].hist(Xim, label='Im', alpha=0.5, bins=bins, range=range, color='blue')
    axs[0].set_title(rf"Re : $\mu$ = {mre:.6f}, $\sigma$ = {sre:.6f}, $\epsilon$ = {sre/np.sqrt(n):.6f}")
    axs[1].set_title(rf"Im : $\mu$ = {mim:.6f}, $\sigma$ = {sim:.6f}, $\epsilon$ = {sim/np.sqrt(n):.6f}")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def pmat(x, heatmap=False, lims=None, print_zeros=False):
    """Print or plot a complex-valued matrix

    :param x: matrix to be plotted
    :type x: numpy.ndarray
    :param heatmap: True if plotting a heatmap, defaults to False
    :type heatmap: bool, optional
    :param lims: if heatmap, limits for colorbar, defaults to None
    :type lims: tuple, optional
    :param print_zeros: True if printing a part if it is all zeros, defaults to False
    :type print_zeros: bool, optional
    """    
    n, m = x.shape
    re = np.real(x)
    im = np.imag(x)
    if (re != np.zeros_like(re)).any() and not print_zeros:
        print('Real part:')
        for i in range(n):
            print([float(f'{re[i, j]:8.8}') for j in range(m)])
    if (im != np.zeros_like(im)).any() and not print_zeros:
        print('Imaginary part:')
        for i in range(n):
            print([float(f'{im[i, j]:8.8}') for j in range(m)])
    if heatmap:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        if lims is None:
            ax1.imshow(re)
            ax2.imshow(im)
        else:
            ax1.imshow(re, vmin=lims[0], vmax=lims[1])
            ax2.imshow(im, vmin=lims[0], vmax=lims[1])
        plt.show()
        
def spinor2(state='up', ketwise=True, seed=None):
    """Convenience function for making 2-dimensional spin state vectors

    :param state: can be one of ['up', 'down', 'random', 'max'], defaults to 'up'.
    :type state: str, optional
    :param ketwise: True for column vector, False for row vector, defaults to True
    :type ketwise: bool, optional
    :param seed: rng seed, defaults to None
    :type seed: int, optional
    :return: your vector
    :rtype: numpy.ndarray
    """    
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
    """Convenience function for making 4-dimensional spin-isospin state vectors

    :param state: can be one of ['up', 'down', 'random', 'max'], defaults to 'up'.
    :type state: str, optional
    :param ketwise: True for column vector, False for row vector, defaults to True
    :type ketwise: bool, optional
    :param seed: rng seed, defaults to None
    :type seed: int, optional
    :return: your vector
    :rtype: numpy.ndarray
    """   
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
    
    :param n_particles: number of particles
    :type n_particles: int
    :return: (sigma operators, tau operators)
    :rtype: tuple of lists of lists
    
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
    
    :param n_particles: number of particles
    :type n_particles: int
    :return: (sigma matrices, tau matrices)
    :rtype: tuple of lists
    
    usage: 
    sigma, tau = sigma_tau_matrices_product(n_particles) 
    sigma[dimension_index]

    note that I am not using the ProductOperator class here. This is done for memory efficiency.
    In the case of Hilbert space calculations, it makes sense to compute the operator matrices beforehand and store them.
    In the tensor-product basis, this would result in most of our memory being taken up by identity matrices.
    """
    from .core import kronecker_product, pauli
    sigma = [kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
    tau = [kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
    return sigma, tau
