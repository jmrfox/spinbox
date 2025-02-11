# spinbox
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.4'

# imports
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng

from scipy.linalg import expm
from functools import reduce


# functions
# redefine basic fxns to be complex (maybe unnecessary, but better safe than sorry)
# numpy.sqrt will complain if you give it a negative number, so i'm not taking any chances


def csqrt(x):
    """Complex square root
    """
    return np.sqrt(x, dtype=complex)

def ccos(x):
    """Complex cosine
    """
    return np.cos(x, dtype=complex)

def csin(x):
    """Complex sine
    """
    return np.sin(x, dtype=complex)

def cexp(x):
    """Complex exponential
    """
    return np.exp(x, dtype=complex)


def ccosh(x):
    """Complex hyp. cosine
    """
    return np.cosh(x, dtype=complex)

def csinh(x):
    """Complex hyp. sine
    """
    return np.sinh(x, dtype=complex)

def ctanh(x):
    """Complex hyp. tangent
    """
    return np.tanh(x, dtype=complex)

def carctanh(x):
    """Complex incerse hyp. tangent
    """
    return np.arctanh(x, dtype=complex)


def read_coeffs(filename):
    """Reads complex spin coefficients from a file 

    Args:
        filename (str): name of file to load

    Returns:
        numpy.array 
    """
    def r2c(x):
        y = float(x[0]) + 1j * float(x[1])
        return y

    c = np.loadtxt(filename)
    sp = np.array([r2c(x) for x in c], dtype=complex)
    return sp


def prod(l: list):
    """Multiplies (*) elements of list, right to left
    """
    lrev = l[::-1]
    out = lrev[0]
    for x in lrev[1:]:
        out = x * out
    return out


def pauli(arg):
    """returns pauli matrix numpy array"""
    if arg in [0, 'x']:
        out = np.array([[0, 1], [1, 0]], dtype=complex)
    elif arg in [1, 'y']:
        out = np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif arg in [2, 'z']:
        out = np.array([[1, 0], [0, -1]], dtype=complex)
    elif arg in ['list']:
        out = [np.array([[0, 1], [1, 0]], dtype=complex),
               np.array([[0, -1j], [1j, 0]], dtype=complex),
               np.array([[1, 0], [0, -1]], dtype=complex)]
    else:
        raise ValueError(f'No option: {arg}')
    return out


def spinor2(state, orientation, seed=None):
    """returns spin coefficients, numpy array"""
    assert state in ['up', 'down', 'random', 'max']
    assert orientation in ['bra', 'ket']
    if state == 'up':
        sp = np.array([1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1], dtype=complex)
    elif state == 'random':
        rng = default_rng(seed=seed)
        sp = rng.uniform(-1, 1, 2) + 1j * rng.uniform(-1, 1, 2)
        sp = sp / np.linalg.norm(sp)
    elif state == 'max':
        sp = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    if orientation == 'ket':
        return sp.reshape((2, 1))
    elif orientation == 'bra':
        return sp.reshape((1, 2))


def spinor4(state, orientation, seed=None):
    """returns spin-isospin coefficients, numpy array"""
    assert state in ['up', 'down', 'random', 'max']
    assert orientation in ['bra', 'ket']
    if state == 'up':
        sp = np.array([1, 0, 1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1, 0, 1], dtype=complex)
    elif state == 'random':
        rng = default_rng(seed=seed)
        sp = rng.uniform(-1, 1, 4) + 1j * rng.uniform(-1, 1, 4)
        sp = sp / np.linalg.norm(sp)
    elif state == 'max':
        sp = np.array([1, 1, 1, 1], dtype=complex)
    if orientation == 'ket':
        return sp.reshape((4, 1)) / np.linalg.norm(sp)
    elif orientation == 'bra':
        return sp.reshape((1, 4)) / np.linalg.norm(sp)


def random_spin_bra_ket(num_particles, bra_seed=None, ket_seed=None):
    sp_bra, sp_ket = [], []
    for i in range(num_particles):
        sp_bra.append(spinor2('random', 'bra', seed=bra_seed + i))
        sp_ket.append(spinor2('random', 'ket', seed=ket_seed + i))
    coeffs_bra = np.concatenate(sp_bra, axis=1)
    coeffs_ket = np.concatenate(sp_ket, axis=0)
    bra = OneBodyBasisSpinState(num_particles, 'bra', coeffs_bra)
    ket = OneBodyBasisSpinState(num_particles, 'ket', coeffs_ket)
    return bra, ket


def random_spinisospin_bra_ket(num_particles, bra_seed=None, ket_seed=None):
    sp_bra, sp_ket = [], []
    for i in range(num_particles):
        sp_bra.append(spinor4('random', 'bra', seed=bra_seed + i))
        sp_ket.append(spinor4('random', 'ket', seed=ket_seed + i))
    coeffs_bra = np.concatenate(sp_bra, axis=1)
    coeffs_ket = np.concatenate(sp_ket, axis=0)
    bra = OneBodyBasisSpinIsospinState(num_particles, 'bra', coeffs_bra)
    ket = OneBodyBasisSpinIsospinState(num_particles, 'ket', coeffs_ket)
    return bra, ket

 
def repeated_kronecker_product(matrices: list):
    """
    returns the tensor/kronecker product of a list of arrays
    :param matrices:
    :return:
    """
    return np.array(reduce(np.kron, matrices))


def pmat(x, heatmap=False, lims=None, print_zeros=False):
    """print and/or plot a complex matrix
    heatmat: plot a heatmap
    lims: if plotting, limits on colorbar
    print_zeros: whether to print Re/Im parts if all zero"""
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


# classes

class State:
    """
    base class for quantum many-body states
    should not be instantiated
    """

    def __init__(self, num_particles: int, orientation: str):
        assert isinstance(num_particles, int)
        self.num_particles = num_particles
        assert orientation in ['bra', 'ket']
        self.orientation = orientation


class OneBodyBasisSpinState_DEPRECATED(State):
    """
    an array of single particle spin-1/2 spinors
    
    num_particles: number of single particle states
    coefficients: list or array of numbers
    orientation: 'bra' or 'ket'
    """

    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(num_particles, orientation)
        self.dim = self.num_particles * 2
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = OneBodyBasisSpinOperator

    def __add__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients.copy())

    def to_list(self):
        if self.orientation == 'ket':
            return [self.coefficients[i * 2:(i + 1) * 2, 0] for i in range(self.num_particles)]
        elif self.orientation == 'bra':
            return [self.coefficients[0, i * 2:(i + 1) * 2] for i in range(self.num_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, OneBodyBasisSpinState):
                assert other.orientation == 'ket'
                c0 = self.to_list()
                c1 = other.to_list()
                out = np.prod([np.dot(c0[i], c1[i]) for i in range(self.num_particles)])
            elif isinstance(other, OneBodyBasisSpinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, OneBodyBasisSpinState) and other.orientation == 'bra'
            out = OneBodyBasisSpinOperator(num_particles=self.num_particles)
            for i in range(self.num_particles):
                idx_i = i * 2
                idx_f = (i + 1) * 2
                out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(self.coefficients[idx_i:idx_f, 0:1],
                                                                 other.coefficients[0:1, idx_i:idx_f], dtype=complex)
            return out

    def dagger(self):
        if self.orientation == 'bra':
            out = OneBodyBasisSpinState(self.num_particles, 'ket', self.coefficients.conj().T)
            return out
        elif self.orientation == 'ket':
            out = OneBodyBasisSpinState(self.num_particles, 'bra', self.coefficients.conj().T)
            return out

    def __str__(self):
        out = f"{self.__class__.__name__} {self.orientation} of {self.num_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_many_body_state(self):
        """project the NxA TP state into the full N^A MB basis"""
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(2 ** self.num_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, 2 ** self.num_particles)
        return ManyBodyBasisSpinState(self.num_particles, self.orientation, sp_mb)

    def normalize(self):
        out = self.copy()
        if self.orientation == 'ket':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[i:i + 2, 0])
                out.coefficients[i:i + 2, 0] /= n
        if self.orientation == 'bra':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[0, i:i + 2])
                out.coefficients[0, i:i + 2] /= n
        return out

    def scalar_mult(self, particle_index, b):
        out = self.copy()
        if self.orientation == 'ket':
            out.coefficients[particle_index * 2:(particle_index + 1) * 2, 0] *= b
        elif self.orientation == 'bra':
            out.coefficients[0, particle_index * 2:(particle_index + 1) * 2] *= b
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        c = np.array(b, dtype=complex) ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy().spread_scalar_mult(other)
            return out
        else:
            raise ValueError('Problem in rmul')


class OneBodyBasisSpinState(State):
    """
    an array of single particle spin-1/2 spinors
    
    num_particles: number of single particle states
    coefficients: list or array of numbers
    orientation: 'bra' or 'ket'
    """

    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(num_particles, orientation)
        self.dim = self.num_particles * 2
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = OneBodyBasisSpinOperator

    def __add__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients.copy())

    def to_list(self):
        if self.orientation == 'ket':
            return [self.coefficients[i * 2:(i + 1) * 2, 0] for i in range(self.num_particles)]
        elif self.orientation == 'bra':
            return [self.coefficients[0, i * 2:(i + 1) * 2] for i in range(self.num_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, OneBodyBasisSpinState):
                assert other.orientation == 'ket'
                c0 = self.to_list()
                c1 = other.to_list()
                out = np.prod([np.dot(c0[i], c1[i]) for i in range(self.num_particles)])
            elif isinstance(other, OneBodyBasisSpinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, OneBodyBasisSpinState) and other.orientation == 'bra'
            out = OneBodyBasisSpinOperator(num_particles=self.num_particles)
            for i in range(self.num_particles):
                idx_i = i * 2
                idx_f = (i + 1) * 2
                out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(self.coefficients[idx_i:idx_f, 0:1],
                                                                 other.coefficients[0:1, idx_i:idx_f], dtype=complex)
            return out

    def dagger(self):
        if self.orientation == 'bra':
            out = OneBodyBasisSpinState(self.num_particles, 'ket', self.coefficients.conj().T)
            return out
        elif self.orientation == 'ket':
            out = OneBodyBasisSpinState(self.num_particles, 'bra', self.coefficients.conj().T)
            return out

    def __str__(self):
        out = f"{self.__class__.__name__} {self.orientation} of {self.num_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_many_body_state(self):
        """project the NxA TP state into the full N^A MB basis"""
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(2 ** self.num_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, 2 ** self.num_particles)
        return ManyBodyBasisSpinState(self.num_particles, self.orientation, sp_mb)

    def normalize(self):
        out = self.copy()
        if self.orientation == 'ket':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[i:i + 2, 0])
                out.coefficients[i:i + 2, 0] /= n
        if self.orientation == 'bra':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[0, i:i + 2])
                out.coefficients[0, i:i + 2] /= n
        return out

    def scalar_mult(self, particle_index, b):
        out = self.copy()
        if self.orientation == 'ket':
            out.coefficients[particle_index * 2:(particle_index + 1) * 2, 0] *= b
        elif self.orientation == 'bra':
            out.coefficients[0, particle_index * 2:(particle_index + 1) * 2] *= b
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        c = np.array(b, dtype=complex) ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy().spread_scalar_mult(other)
            return out
        else:
            raise ValueError('Problem in rmul')


class OneBodyBasisSpinIsospinState(State):
    """
    an array of single particle spinors with spin-1/2 and isospin-1/2

    e.g. one-body spinor = (p_up p_dn n_up n_dn)^T

    num_particles: number of single particle states
    coefficients: numpy.ndarray of length A*4, kets are columns, bras are rows
    orientation: 'bra' or 'ket'
    """

    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(num_particles, orientation)
        self.dim = self.num_particles * 4
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = OneBodyBasisSpinIsospinOperator

    def __add__(self, other):
        return OneBodyBasisSpinIsospinState(self.num_particles, self.orientation,
                                            self.coefficients + other.coefficients)

    def __sub__(self, other):
        return OneBodyBasisSpinIsospinState(self.num_particles, self.orientation,
                                            self.coefficients - other.coefficients)

    def copy(self):
        return OneBodyBasisSpinIsospinState(self.num_particles, self.orientation, self.coefficients.copy())

    def to_list(self):
        if self.orientation == 'ket':
            return [self.coefficients[i * 4:(i + 1) * 4, 0] for i in
                    range(self.num_particles)]
        elif self.orientation == 'bra':
            return [self.coefficients[0, i * 4:(i + 1) * 4] for i in
                    range(self.num_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or a operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, OneBodyBasisSpinIsospinState):
                assert other.orientation == 'ket'
                c0 = self.to_list()
                c1 = other.to_list()
                return np.prod(
                    [np.dot(c0[i], c1[i]) for i in range(self.num_particles)])
            elif isinstance(other, OneBodyBasisSpinIsospinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
                return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, OneBodyBasisSpinIsospinState) and other.orientation == 'bra'
            out = OneBodyBasisSpinIsospinOperator(num_particles=self.num_particles)
            for i in range(self.num_particles):
                idx_i = i * 4
                idx_f = (i + 1) * 4
                out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(self.coefficients[idx_i:idx_f, 0:1],
                                                                 other.coefficients[0:1, idx_i:idx_f], dtype=complex)
            return out
        else:
            raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')

    def dagger(self):
        if self.orientation == 'bra':
            out = OneBodyBasisSpinIsospinState(self.num_particles, 'ket', self.coefficients.conj().T)
            return out
        elif self.orientation == 'ket':
            out = OneBodyBasisSpinIsospinState(self.num_particles, 'bra', self.coefficients.conj().T)
            return out

    def __str__(self):
        out = f"{self.__class__.__name__} {self.orientation} of {self.num_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_many_body_state(self):
        """project the NxA TP state into the full N^A MB basis"""
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(4 ** self.num_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, 4 ** self.num_particles)
        return ManyBodyBasisSpinIsospinState(self.num_particles, self.orientation, sp_mb)

    def normalize(self):
        out = self.copy()
        if self.orientation == 'ket':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[i:i + 4, 0])
                out.coefficients[i:i + 4, 0] /= n
        if self.orientation == 'bra':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[0, i:i + 4])
                out.coefficients[0, i:i + 4] /= n
        return out

    def scalar_mult(self, particle_index, b):
        out = self.copy()
        if self.orientation == 'ket':
            out.coefficients[particle_index * 4:(particle_index + 1) * 4, 0] *= b
        elif self.orientation == 'bra':
            out.coefficients[0, particle_index * 4:(particle_index + 1) * 4] *= b
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        c = b ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy().spread_scalar_mult(other)
            return out
        else:
            raise ValueError(f'Problem in rmul: got a {type(other)}')


class ManyBodyBasisSpinState(State):
    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(num_particles, orientation)

        self.dim = 2 ** self.num_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = ManyBodyBasisSpinOperator

    def __add__(self, other):
        return ManyBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return ManyBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return ManyBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients.copy())

    def __mul__(self, other):
        if self.orientation == 'bra':  # inner product
            if isinstance(other, ManyBodyBasisSpinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients.flatten(), other.coefficients.flatten())
            elif isinstance(other, ManyBodyBasisSpinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            c = np.outer(self.coefficients.flatten(), other.coefficients.flatten())
            out = ManyBodyBasisSpinOperator(self.num_particles)
            out.matrix = c
            return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.coefficients *= other
        else:
            raise TypeError(f'Not supported: {type(other)} * {self.__class__.__name__}')
        return out

    def dagger(self):
        if self.orientation == 'bra':
            out = ManyBodyBasisSpinState(self.num_particles, 'ket', self.coefficients.conj().T)
            return out
        elif self.orientation == 'ket':
            out = ManyBodyBasisSpinState(self.num_particles, 'bra', self.coefficients.conj().T)
            return out

    def __str__(self):
        out = [f'{self.__class__.__name__} {self.orientation} of {self.num_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)


class ManyBodyBasisSpinIsospinState(State):
    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(num_particles, orientation)

        self.dim = 4 ** self.num_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = ManyBodyBasisSpinIsospinOperator

    def __add__(self, other):
        return ManyBodyBasisSpinIsospinState(self.num_particles, self.orientation,
                                             self.coefficients + other.coefficients)

    def __sub__(self, other):
        return ManyBodyBasisSpinIsospinState(self.num_particles, self.orientation,
                                             self.coefficients - other.coefficients)

    def copy(self):
        return ManyBodyBasisSpinIsospinState(self.num_particles, self.orientation, self.coefficients.copy())

    def __mul__(self, other):
        if self.orientation == 'bra':  # inner product
            if isinstance(other, ManyBodyBasisSpinIsospinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients.flatten(), other.coefficients.flatten())
            elif isinstance(other, ManyBodyBasisSpinIsospinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            c = np.outer(self.coefficients.flatten(), other.coefficients.flatten())
            out = ManyBodyBasisSpinIsospinOperator(self.num_particles)
            out.matrix = c
            return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.coefficients *= other
        else:
            raise TypeError(f'Not supported: {type(other)} * {self.__class__.__name__}')
        return out

    def dagger(self):
        if self.orientation == 'bra':
            out = ManyBodyBasisSpinIsospinState(self.num_particles, 'ket', self.coefficients.conj().T)
            return out
        elif self.orientation == 'ket':
            out = ManyBodyBasisSpinIsospinState(self.num_particles, 'bra', self.coefficients.conj().T)
            return out

    def __str__(self):
        out = [f'{self.__class__.__name__} {self.orientation} of {self.num_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)


class SpinOperator:
    """
    base class for spin operators
    do not instantiate
    """

    def __init__(self, num_particles: int):
        assert isinstance(num_particles, int)
        self.num_particles = num_particles


class OneBodyBasisSpinOperator_DEPRECATED(SpinOperator):
    def __init__(self, num_particles: int):
        super().__init__(num_particles)
        self.dim = num_particles * 2
        self.matrix = np.identity(self.dim, dtype='complex')
        self.friendly_state = OneBodyBasisSpinState

    def __add__(self, other):
        assert isinstance(other, OneBodyBasisSpinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, OneBodyBasisSpinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = OneBodyBasisSpinOperator(self.num_particles)
        out.matrix = self.matrix.copy()
        return out

    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, OneBodyBasisSpinOperator):
            out = other.copy()
            out.matrix = np.matmul(self.matrix, out.matrix, dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or a {self.__class__.__name__}')

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (2, 2)
        idx_i = particle_index * 2
        idx_f = (particle_index + 1) * 2
        out = self.copy()
        out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(matrix, out.matrix[idx_i:idx_f, idx_i:idx_f], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return out

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=b * np.identity(2))
        return out

    def spread_scalar_mult(self, b):
        """scalar multiplication, but 'spread' over all particles
        e.g. for N-particle system, each spinor multiplied by b^(1/N)"""
        assert np.isscalar(b)
        c = np.array(b, dtype=complex) ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            # out = self.copy().spread_scalar_mult(other)
            # return out
            raise ValueError('Cannot multiply this way! Use .scalar_mult() instead!')
        else:
            raise ValueError(f'rmul not supported for {other.__class__.__name__} * {self.__class__.__name__}')

    def exchange(self, particle_a: int, particle_b: int):
        out = self.copy()
        idx_ai = particle_a * 2
        idx_af = (particle_a + 1) * 2
        idx_bi = particle_b * 2
        idx_bf = (particle_b + 1) * 2
        temp = out.matrix.copy()
        out.matrix[idx_ai:idx_af, idx_ai:idx_af] = temp[idx_ai:idx_af, idx_bi:idx_bf]
        out.matrix[idx_ai:idx_af, idx_bi:idx_bf] = temp[idx_ai:idx_af, idx_ai:idx_af]
        out.matrix[idx_bi:idx_bf, idx_bi:idx_bf] = temp[idx_bi:idx_bf, idx_ai:idx_af]
        out.matrix[idx_bi:idx_bf, idx_ai:idx_af] = temp[idx_bi:idx_bf, idx_bi:idx_bf]
        return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

    def dagger(self):
        out = self.copy()
        out.matrix = self.matrix.conj().T
        return out

class OneBodyBasisSpinOperator(SpinOperator):
    def __init__(self, num_particles: int):
        super().__init__(num_particles)
        self.op_stack = np.zeros(shape=(num_particles, 2, 2))
        self.friendly_state = OneBodyBasisSpinState

    def __add__(self, other):
        raise SyntaxError('You should not be adding a OBBSOperator')
    
    def __sub__(self, other):
        raise SyntaxError('You should not be subtracting a OBBSOperator')

    def copy(self):
        out = OneBodyBasisSpinOperator(self.num_particles)
        for i in range(self.num_particles):
            out.op_stack[i:i+1, :, :] = self.op_stack[i:i+1, :, :]
        return out

    def build_matrix(self):
        dim = 2 * self.num_particles
        out = np.zeros(dim,dim)
        for i in range(self.num_particles):
            out[i*2:(i+1)*2 , i*2:(i+1)*2 ] = self.op_stack[i:i+1, :, :]
        return out
    
    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, OneBodyBasisSpinOperator):
            out = other.copy()
            out.matrix = np.matmul(self.matrix, out.matrix, dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or a {self.__class__.__name__}')

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (2, 2)
        idx_i = particle_index * 2
        idx_f = (particle_index + 1) * 2
        out = self.copy()
        out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(matrix, out.matrix[idx_i:idx_f, idx_i:idx_f], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return out

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=b * np.identity(2))
        return out

    def spread_scalar_mult(self, b):
        """scalar multiplication, but 'spread' over all particles
        e.g. for N-particle system, each spinor multiplied by b^(1/N)"""
        assert np.isscalar(b)
        c = np.array(b, dtype=complex) ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            # out = self.copy().spread_scalar_mult(other)
            # return out
            raise ValueError('Cannot multiply this way! Use .scalar_mult() instead!')
        else:
            raise ValueError(f'rmul not supported for {other.__class__.__name__} * {self.__class__.__name__}')

    def exchange(self, particle_a: int, particle_b: int):
        out = self.copy()
        idx_ai = particle_a * 2
        idx_af = (particle_a + 1) * 2
        idx_bi = particle_b * 2
        idx_bf = (particle_b + 1) * 2
        temp = out.matrix.copy()
        out.matrix[idx_ai:idx_af, idx_ai:idx_af] = temp[idx_ai:idx_af, idx_bi:idx_bf]
        out.matrix[idx_ai:idx_af, idx_bi:idx_bf] = temp[idx_ai:idx_af, idx_ai:idx_af]
        out.matrix[idx_bi:idx_bf, idx_bi:idx_bf] = temp[idx_bi:idx_bf, idx_ai:idx_af]
        out.matrix[idx_bi:idx_bf, idx_ai:idx_af] = temp[idx_bi:idx_bf, idx_bi:idx_bf]
        return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

    def dagger(self):
        out = self.copy()
        out.matrix = self.matrix.conj().T
        return out

class OneBodyBasisSpinIsospinOperator(SpinOperator):
    def __init__(self, num_particles: int):
        super().__init__(num_particles)
        self.dim = num_particles * 4
        self.matrix = np.identity(self.dim, dtype=complex)
        self.friendly_state = OneBodyBasisSpinIsospinState

    def __add__(self, other):
        assert isinstance(other, OneBodyBasisSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, OneBodyBasisSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = OneBodyBasisSpinIsospinOperator(self.num_particles)
        out.matrix = self.matrix.copy()
        return out

    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinIsospinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, OneBodyBasisSpinIsospinOperator):
            out = other.copy()
            out.matrix = np.matmul(self.matrix, out.matrix, dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or {self.__class__.__name__}')

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_one_body_operator(self, particle_index: int, isospin_matrix: np.ndarray, spin_matrix: np.ndarray):
        assert isospin_matrix.shape == (2, 2)
        assert spin_matrix.shape == (2, 2)
        onebody_matrix = repeated_kronecker_product([isospin_matrix, spin_matrix])
        idx_i = particle_index * 4
        idx_f = (particle_index + 1) * 4
        out = self.copy()
        out.matrix[idx_i:idx_f, idx_i:idx_f] = np.matmul(onebody_matrix, out.matrix[idx_i:idx_f, idx_i:idx_f],
                                                         dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))
        return out

    def tau(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))
        return out

    # def scalar_mult(self, particle_index, b):
    #     assert np.isscalar(b)
    #     out = self.copy()
    #     out = out.apply_one_body_operator(particle_index=particle_index, isospin_matrix=b * np.identity(2),spin_matrix=np.identity(2, dtype=complex))
    #     return out
    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        idx_i = particle_index * 4
        idx_f = (particle_index + 1) * 4
        out.matrix[idx_i:idx_f, idx_i:idx_f] = b * out.matrix[idx_i:idx_f, idx_i:idx_f]
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        c = np.array(b, dtype=complex) ** (1 / self.num_particles)
        out = self.copy()
        for i in range(self.num_particles):
            out = out.scalar_mult(i, c)
        return out

    def __rmul__(self, other):
        if np.isscalar(other):
            # out = self.copy().spread_scalar_mult(other)
            # return out
            raise ValueError('Cannot multiply this way! Use .scalar_mult() instead!')
        else:
            raise ValueError(f'rmul not supported for {other.__class__.__name__} * {self.__class__.__name__}')

    def exchange(self, particle_a: int, particle_b: int):
        broken = True
        if broken:
            raise ValueError('Exchange for the spin-isospin basis is not set up!')
        else:
            out = self.copy()
            idx_ai = particle_a * 4
            idx_af = (particle_a + 1) * 4
            idx_bi = particle_b * 4
            idx_bf = (particle_b + 1) * 4
            temp = out.matrix.copy()
            out.matrix[idx_ai:idx_af, idx_ai:idx_af] = temp[idx_ai:idx_af, idx_bi:idx_bf]
            out.matrix[idx_ai:idx_af, idx_bi:idx_bf] = temp[idx_ai:idx_af, idx_ai:idx_af]
            out.matrix[idx_bi:idx_bf, idx_bi:idx_bf] = temp[idx_bi:idx_bf, idx_ai:idx_af]
            out.matrix[idx_bi:idx_bf, idx_ai:idx_af] = temp[idx_bi:idx_bf, idx_bi:idx_bf]
            return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

    def dagger(self):
        out = self.copy()
        out.matrix = self.matrix.conj().T
        return out


class ManyBodyBasisSpinOperator(SpinOperator):
    def __init__(self, num_particles: int):
        super().__init__(num_particles)
        self.matrix = np.identity(2 ** num_particles, dtype=complex)
        self.friendly_state = ManyBodyBasisSpinState

    def __add__(self, other):
        assert isinstance(other, ManyBodyBasisSpinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, ManyBodyBasisSpinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = ManyBodyBasisSpinOperator(self.num_particles)
        out.matrix = self.matrix.copy()
        return out

    def __mul__(self, other):
        if isinstance(other, ManyBodyBasisSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, ManyBodyBasisSpinOperator):
            out = other.copy()
            out.matrix = np.matmul(self.matrix, out.matrix, dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or {self.__class__.__name__}')

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.matrix = other * out.matrix
        else:
            raise ValueError(f'rmul not set up for {self.__class__.__name__}')
        return out

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert type(matrix) == np.ndarray and matrix.shape == (2, 2)
        obo = [np.identity(2, dtype=complex) for _ in range(self.num_particles)]
        obo[particle_index] = matrix
        obo = repeated_kronecker_product(obo)
        out = self.copy()
        out.matrix = np.matmul(obo, out.matrix, dtype=complex)
        return out

    def sigma(self, particle_index: int, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return out

    def scalar_mult(self, particle_index: int, b):
        """
        multiply one particle by a scalar.
        To multiply the whole state by a scalar, just do b * ManyBodySpinState.
        :param particle_index:
        :param b:
        :return:
        """
        assert np.isscalar(b)
        out = b * self.copy()
        return out

    def exchange(self, particle_a: int, particle_b: int):
        P_1 = ManyBodyBasisSpinOperator(num_particles=self.num_particles)
        P_x = P_1.copy().sigma(particle_a, 'x').sigma(particle_b, 'x')
        P_y = P_1.copy().sigma(particle_a, 'y').sigma(particle_b, 'y')
        P_z = P_1.copy().sigma(particle_a, 'z').sigma(particle_b, 'z')
        P = (P_x + P_y + P_z + P_1)
        out = 0.5 * P * self.copy()
        return out

    def exponentiate(self):
        out = self.copy()
        out.matrix = expm(out.matrix)
        return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

    def dagger(self):
        out = self.copy()
        out.matrix = self.matrix.conj().T
        return out


class ManyBodyBasisSpinIsospinOperator(SpinOperator):
    def __init__(self, num_particles: int):
        super().__init__(num_particles)
        self.matrix = np.identity(4 ** num_particles, dtype=complex)
        self.friendly_state = ManyBodyBasisSpinIsospinState

    def __add__(self, other):
        assert isinstance(other, ManyBodyBasisSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, ManyBodyBasisSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = ManyBodyBasisSpinIsospinOperator(self.num_particles)
        out.matrix = self.matrix.copy()
        return out
    
    def transpose(self):
        out = self.copy()
        out.matrix = self.matrix.T
        return out

    def __mul__(self, other):
        if isinstance(other, ManyBodyBasisSpinIsospinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, ManyBodyBasisSpinIsospinOperator):
            out = other.copy()
            out.matrix = np.matmul(self.matrix, out.matrix, dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or {self.__class__.__name__}')

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.matrix = other * out.matrix
        else:
            raise ValueError(f'rmul not set up for {self.__class__.__name__}')
        return out

    def __str__(self):
        out = f"{self.__class__.__name__}"
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_one_body_operator(self, particle_index: int, isospin_matrix: np.ndarray, spin_matrix: np.ndarray):
        assert type(isospin_matrix) == np.ndarray and isospin_matrix.shape == (2, 2)
        assert type(spin_matrix) == np.ndarray and spin_matrix.shape == (2, 2)
        obo = [np.identity(4, dtype=complex) for _ in range(self.num_particles)]
        obo[particle_index] = repeated_kronecker_product([isospin_matrix, spin_matrix])
        obo = repeated_kronecker_product(obo)
        out = self.copy()
        out.matrix = np.matmul(obo, self.matrix, dtype=complex)
        return out

    def sigma(self, particle_index: int, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))
        return out

    def tau(self, particle_index: int, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))
        return out

    def scalar_mult(self, particle_index: int, b):
        """
        multiply one particle by a scalar.
        To multiply the whole state by a scalar, just do b * ManyBodySpinState.
        :param particle_index:
        :param b:
        :return:
        """
        assert np.isscalar(b)
        out = self.copy()
        return b * out

    def exchange(self, particle_a: int, particle_b: int):
        P_1 = ManyBodyBasisSpinIsospinOperator(num_particles=self.num_particles)
        P_x = P_1.copy().sigma(particle_a, 'x').sigma(particle_b, 'x')
        P_y = P_1.copy().sigma(particle_a, 'y').sigma(particle_b, 'y')
        P_z = P_1.copy().sigma(particle_a, 'z').sigma(particle_b, 'z')
        P = (P_x + P_y + P_z + P_1)
        out = 0.5 * P * self.copy()
        return out

    def exponentiate(self):
        out = self.copy()
        out.matrix = expm(out.matrix)
        return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

    def dagger(self):
        out = self.copy()
        out.matrix = self.matrix.conj().T
        return out



# class Optimizer():

