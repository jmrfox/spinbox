# quap
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.1.0'

# imports
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng

from scipy.linalg import expm
from functools import reduce

# from dataclasses import dataclass

import itertools
from multiprocessing.pool import Pool
from tqdm import tqdm

# functions
# redefine basic fxns to be complex (maybe unnecessary, but better safe than sorry)
# numpy.sqrt will raise warning (NOT error) if you give it a negative number, so i'm not taking any chances
# these nuse numpy instead of math to be safe with ndarrays

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


def interaction_indices(n, m = 2):
    """ returns a list of all possible m-plets of n objects (labelled 0 to n-1)
    default: m=2, giving all possible pairs
    """
    return list(itertools.combinations(range(n), m))


def read_from_file(filename, complex=False, shape=None, order='F'):
    """Reads numbers from a file 

    Args:
        filename (str): name of file to load

    Returns:
        numpy.array 
    """
    def tuple_to_complex(x):
        y = float(x[0]) + 1j * float(x[1])
        return y

    c = np.loadtxt(filename)
    if complex:
        sp = np.array([tuple_to_complex(x) for x in c], dtype='complex')
    else:
        sp = np.array(c)

    if shape is not None:
        sp = sp.reshape(shape, order=order)

    return sp


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



def repeated_kronecker_product(matrices: list):
    """
    returns the tensor/kronecker product of a list of arrays
    :param matrices:
    :return:
    """
    return np.array(reduce(np.kron, matrices), dtype=complex)


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


def scalar(x: np.ndarray):
    """turns a 1-element array  x into an actual scalar
    """
    # assert type(x) in [list, np.ndarray]
    assert list(x.shape) == [1 for _ in range(len(x.shape))]
    out = x.flatten()[0]
    return out


# SHARED BASE CLASSES

class State:
    """
    base class for quantum many-body states
    should not be instantiated
    """

    def __init__(self, n_particles: int, orientation: str):
        assert isinstance(n_particles, int)
        self.n_particles = n_particles
        assert orientation in ['bra', 'ket']
        self.orientation = orientation

class Operator:
    """
    base class for spin operators
    do not instantiate
    """

    def __init__(self, n_particles: int):
        assert isinstance(n_particles, int)
        self.n_particles = n_particles

# MANY-BODY BASIS CLASSES

class GFMCSpinState(State):
    def __init__(self, n_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(n_particles, orientation)
        self.dim = 2 ** self.n_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError("Inconsistent initialization of state vector. \n\
                             Did you get the shape right?")
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = GFMCSpinOperator

    def copy(self):
        return GFMCSpinState(self.n_particles, self.orientation, self.coefficients.copy())
    
    def __add__(self, other):
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other):
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def __mul__(self, other):
        if self.orientation == 'bra':  # inner product
            if isinstance(other, GFMCSpinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients, other.coefficients)
            elif isinstance(other, GFMCSpinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            out = GFMCSpinOperator(self.n_particles)
            out.matrix = np.matmul(self.coefficients, other.coefficients)
            return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.coefficients *= other
        else:
            raise TypeError(f'Not supported: {type(other)} * {self.__class__.__name__}')
        return out

    def dagger(self):
        out = self.copy()
        if self.orientation == 'bra':
            new_orientation = 'ket'
        elif self.orientation == 'ket':
            new_orientation = 'bra'
        out = GFMCSpinState(self.n_particles, new_orientation, self.coefficients.conj().T)
        return out
        
    def __str__(self):
        out = [f'{self.__class__.__name__} {self.orientation} of {self.n_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)

    def randomize(self, seed):
        out = self.copy()
        rng = np.random.default_rng(seed=seed)
        out.coefficients = rng.standard_normal(size=self.coefficients.shape)
        out.coefficients /= np.linalg.norm(out.coefficients)
        return out
    

class GFMCSpinIsospinState(State):
    def __init__(self, n_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(n_particles, orientation)
        self.dim = 4 ** self.n_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = GFMCSpinIsospinOperator
    
    def copy(self):
        return GFMCSpinIsospinState(self.n_particles, self.orientation, self.coefficients.copy())
    
    def __add__(self, other):
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other):
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def __mul__(self, other):
        if self.orientation == 'bra':  # inner product
            if isinstance(other, GFMCSpinIsospinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients, other.coefficients)
            elif isinstance(other, GFMCSpinIsospinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            out = GFMCSpinIsospinOperator(self.n_particles)
            out.matrix = np.matmul(self.coefficients, other.coefficients)
            return out

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.coefficients *= other
        else:
            raise TypeError(f'Not supported: {type(other)} * {self.__class__.__name__}')
        return out

    def dagger(self):
        out = self.copy()
        if self.orientation == 'bra':
            new_orientation = 'ket'
        elif self.orientation == 'ket':
            new_orientation = 'bra'
        out = GFMCSpinIsospinState(self.n_particles, new_orientation, self.coefficients.conj().T)
        return out
        
    def __str__(self):
        out = [f'{self.__class__.__name__} {self.orientation} of {self.n_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)
    
    def randomize(self, seed):
        out = self.copy()
        rng = np.random.default_rng(seed=seed)
        out.coefficients = rng.standard_normal(size=self.coefficients.shape)
        out.coefficients /= np.linalg.norm(out.coefficients)
        return out


class GFMCSpinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.matrix = np.identity(2 ** n_particles, dtype=complex)
        self.friendly_state = GFMCSpinState

    def copy(self):
        out = GFMCSpinOperator(self.n_particles)
        out.matrix = self.matrix.copy()
        return out
    
    def __add__(self, other):
        assert isinstance(other, GFMCSpinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, GFMCSpinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def __mul__(self, other):
        if isinstance(other, GFMCSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, GFMCSpinOperator):
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
        assert type(matrix) == np.ndarray and matrix.shape == (2,2)
        obo = [np.identity(2, dtype=complex) for _ in range(self.n_particles)]
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
        P_1 = GFMCSpinOperator(n_particles=self.n_particles)
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


class GFMCSpinIsospinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.matrix = np.identity(4 ** n_particles, dtype=complex)
        self.friendly_state = GFMCSpinIsospinState

    def copy(self):
        out = GFMCSpinIsospinOperator(self.n_particles)
        out.matrix = self.matrix.copy()
        return out
    
    def __add__(self, other):
        assert isinstance(other, GFMCSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, GFMCSpinIsospinOperator)
        out = self.copy()
        out.matrix = self.matrix - other.matrix
        return out

    def __mul__(self, other):
        if isinstance(other, GFMCSpinIsospinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = np.matmul(self.matrix, out.coefficients, dtype=complex)
            return out
        elif isinstance(other, GFMCSpinIsospinOperator):
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

    def apply_one_body_operator(self, particle_index: int, isospin_matrix: np.ndarray, spin_matrix: np.ndarray):
        assert type(isospin_matrix) == np.ndarray and isospin_matrix.shape == (2, 2)
        assert type(spin_matrix) == np.ndarray and spin_matrix.shape == (2, 2)
        obo = [np.identity(4, dtype=complex) for _ in range(self.n_particles)]
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
        assert np.isscalar(b)
        out = b * self.copy()
        return out

    def exchange(self, particle_a: int, particle_b: int):
        P_1 = GFMCSpinIsospinOperator(n_particles=self.n_particles)
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




# ONE-BODY BASIS CLASSES
        

class AFDMCSpinState(State):
    def __init__(self, n_particles: int, orientation: str, coefficients: np.ndarray):
        """an array of single particle spinors

        Orientation must be consistent with array shape!
        The shape of a bra is (A, 2, 1)
        The shape of a ket is (A, 1, 2)

        Args:
            n_particles (int): number of single particle states
            orientation (str): 'bra' or 'ket'
            coefficients (np.ndarray): array of complex numbers

        Raises:
            ValueError: _description_
        """
        super().__init__(n_particles, orientation)
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (n_particles, 2, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (n_particles, 1, 2)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            ValueError("Inconsistent initialization of state vector. \n\
                             Did you get the shape right?")
        else:
            self.sp_stack = coefficients.astype(complex)
        self.friendly_operator = AFDMCSpinOperator
        

    def __add__(self, other):
        raise SyntaxError('You should probably not be adding a AFDMC states')
    
    def __sub__(self, other):
        raise SyntaxError('You should probably not be subtracting a AFDMC states')

    def copy(self):
        return AFDMCSpinState(self.n_particles, self.orientation, self.sp_stack.copy())

    def to_list(self):
        return [self.sp_stack[i] for i in range(self.n_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, AFDMCSpinState):
                assert other.orientation == 'ket'
                out = np.prod([np.dot(self.sp_stack[i], other.sp_stack[i]) for i in range(self.n_particles)])
            elif isinstance(other, AFDMCSpinOperator):
                out = self.copy()
                for i in range(self.n_particles):
                    out.sp_stack[i] = np.matmul(self.sp_stack[i], other.op_stack[i], dtype=complex)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, AFDMCSpinState) and other.orientation == 'bra'
            out = AFDMCSpinOperator(n_particles=self.n_particles)
            for i in range(self.n_particles):
                out.op_stack[i] = np.matmul(self.sp_stack[i], other.sp_stack[i], dtype=complex)
            return out

    def dagger(self):
        out = self.copy()
        if self.orientation=='ket':
            out.orientation = 'bra'
        elif self.orientation=='bra':
            out.orientation = 'ket'
        out.sp_stack = np.transpose(self.sp_stack, axes=(0,2,1)).conj()
        return out

    def __str__(self):
        out = f"{self.__class__.__name__} {self.orientation} of {self.n_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_manybody_basis(self):
        """project the NxA TP state into the full N^A MB basis"""
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(2 ** self.n_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, 2 ** self.n_particles)
        return GFMCSpinState(self.n_particles, self.orientation, sp_mb)

    def normalize(self):
        out = self.copy()
        for i in range(self.n_particles):
            n = np.linalg.norm(self.sp_stack[i])
            out.sp_stack[i] /= n
        return out

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        out.sp_stack[particle_index] *= b
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        out = self.copy()
        out.sp_stack *= b ** (1 / self.n_particles)
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (2, 2)
        out = self.copy()
        out.sp_stack[particle_index] = np.matmul(matrix, out.sp_stack[particle_index], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return out
    
    def randomize(self, seed):
        out = self.copy()
        rng = np.random.default_rng(seed=seed)
        out.sp_stack = rng.standard_normal(size=self.sp_stack.shape)
        for i in range(self.n_particles):
            out.sp_stack[i] /= np.linalg.norm(out.sp_stack[i])
        return out
    

class AFDMCSpinIsospinState(State):
    def __init__(self, n_particles: int, orientation: str, coefficients: np.ndarray):
        """an array of single particle spinors

        Orientation must be consistent with array shape!
        The shape of a bra is (A, 4, 1)
        The shape of a ket is (A, 1, 4)

        Args:
            n_particles (int): number of single particle states
            orientation (str): 'bra' or 'ket'
            coefficients (np.ndarray): array of complex numbers

        Raises:
            ValueError: _description_
        """
        super().__init__(n_particles, orientation)
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (n_particles, 4, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (n_particles, 1, 4)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError("Inconsistent initialization of state vector. \n\
                             Did you get the shape right?")
        else:
            self.sp_stack = coefficients.astype(complex)
        self.friendly_operator = AFDMCSpinIsospinOperator
        

    def __add__(self, other):
        raise SyntaxError('You should probably not be adding a AFDMC states')
    
    def __sub__(self, other):
        raise SyntaxError('You should probably not be subtracting a AFDMC states')

    def copy(self):
        return AFDMCSpinIsospinState(self.n_particles, self.orientation, self.sp_stack.copy())

    def to_list(self):
        return [self.sp_stack[i] for i in range(self.n_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, AFDMCSpinIsospinState):
                assert other.orientation == 'ket'
                out = np.prod([np.dot(self.sp_stack[i], other.sp_stack[i]) for i in range(self.n_particles)])
            elif isinstance(other, AFDMCSpinIsospinOperator):
                out = self.copy()
                for i in range(self.n_particles):
                    out.sp_stack[i] = np.matmul(self.sp_stack[i], other.op_stack[i], dtype=complex)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, AFDMCSpinIsospinState) and other.orientation == 'bra'
            out = AFDMCSpinIsospinOperator(n_particles=self.n_particles)
            for i in range(self.n_particles):
                out.op_stack[i] = np.matmul(self.sp_stack[i], other.sp_stack[i], dtype=complex)
            return out

    def dagger(self):
        out = self.copy()
        if self.orientation=='ket':
            out.orientation = 'bra'
        elif self.orientation=='bra':
            out.orientation = 'ket'
        out.sp_stack = np.transpose(self.sp_stack, axes=(0,2,1)).conj()
        return out

    def __str__(self):
        out = f"{self.__class__.__name__} {self.orientation} of {self.n_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_manybody_basis(self):
        """project the NxA TP state into the full N^A MB basis"""
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(4 ** self.n_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, 4 ** self.n_particles)
        return GFMCSpinIsospinState(self.n_particles, self.orientation, sp_mb)

    def normalize(self):
        out = self.copy()
        for i in range(self.n_particles):
            n = np.linalg.norm(self.sp_stack[i])
            out.sp_stack[i] /= n
        return out

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        out.sp_stack[particle_index] *= b
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        out = self.copy()
        out.sp_stack *= b ** (1 / self.n_particles)
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (4, 4)
        out = self.copy()
        out.sp_stack[particle_index] = np.matmul(matrix, out.sp_stack[particle_index], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return out

    def apply_one_body_operator(self, particle_index: int, isospin_matrix: np.ndarray, spin_matrix: np.ndarray):
        assert isospin_matrix.shape == (2, 2)
        assert spin_matrix.shape == (2, 2)
        onebody_matrix = repeated_kronecker_product([isospin_matrix, spin_matrix])
        out = self.copy()
        out.sp_stack[particle_index] = np.matmul(onebody_matrix, out.sp_stack[particle_index], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index,
                                          isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))
        return out

    def tau(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index,
                                          isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))
        return out

    def randomize(self, seed):
        out = self.copy()
        rng = np.random.default_rng(seed=seed)
        out.sp_stack = rng.standard_normal(size=self.sp_stack.shape)
        for i in range(self.n_particles):
            out.sp_stack[i] /= np.linalg.norm(out.sp_stack[i])
        return out


class AFDMCSpinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.op_stack = np.stack(self.n_particles*[np.identity(2)], dtype=complex)    
        self.friendly_state = AFDMCSpinState

    def __add__(self, other):
        raise SyntaxError('You should probably not be adding AFDMC operators')
    
    def __sub__(self, other):
        raise SyntaxError('You should probably not be subtracting AFDMC operators')

    def copy(self):
        out = AFDMCSpinOperator(self.n_particles)
        for i in range(self.n_particles):
            out.op_stack[i] = self.op_stack[i]
        return out

    def to_list(self):
        return [self.op_stack[i] for i in range(self.n_particles)]
    
    def __mul__(self, other):
        if isinstance(other, AFDMCSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            for i in range(self.n_particles):
                out.sp_stack[i] = np.matmul(self.op_stack[i], out.sp_stack[i], dtype=complex)
            return out
        elif isinstance(other, AFDMCSpinOperator):
            out = other.copy()
            for i in range(self.n_particles):
                out.sp_stack[i] = np.matmul(self.op_stack[i], out.op_stack[i], dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or a {self.__class__.__name__}')


    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        for i, op in enumerate(self.to_list()):
            re = str(np.real(op))
            im = str(np.imag(op))
            out += f"Op {i} Re:\n" + re + f"\nOp {i} Im:\n" + im + "\n"
        return out

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (2, 2)
        out = self.copy()
        out.op_stack[particle_index] = np.matmul(matrix, out.op_stack[particle_index], dtype=complex)
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
        assert np.isscalar(b)
        c = b ** (1 / self.n_particles)
        out = self.copy()
        for i in range(self.n_particles):
            out = out.scalar_mult(i, c)
        return out

    def zeros(self):
        out = self.copy()
        out.op_stack *= 0.0
        return out

    def dagger(self):
        out = self.copy()
        out.op_stack = np.transpose(self.op_stack, axes=(0,2,1)).conj()
        return out
    

class AFDMCSpinIsospinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.op_stack = np.stack(self.n_particles*[np.identity(4)], dtype=complex)    
        self.friendly_state = AFDMCSpinIsospinState

    def __add__(self, other):
        raise SyntaxError('You should probably not be adding AFDMC operators')
    
    def __sub__(self, other):
        raise SyntaxError('You should probably not be subtracting AFDMC operators')

    def copy(self):
        out = AFDMCSpinIsospinOperator(self.n_particles)
        for i in range(self.n_particles):
            out.op_stack[i] = self.op_stack[i]
        return out

    def to_list(self):
        return [self.op_stack[i] for i in range(self.n_particles)]
    
    def __mul__(self, other):
        if isinstance(other, AFDMCSpinIsospinState):
            assert other.orientation == 'ket'
            out = other.copy()
            for i in range(self.n_particles):
                out.sp_stack[i] = np.matmul(self.op_stack[i], out.sp_stack[i], dtype=complex)
            return out
        elif isinstance(other, AFDMCSpinIsospinOperator):
            out = other.copy()
            for i in range(self.n_particles):
                out.op_stack[i] = np.matmul(self.op_stack[i], out.op_stack[i], dtype=complex)
            return out
        else:
            raise ValueError(
                f'{self.__class__.__name__} must multiply a {self.friendly_state.__name__}, or a {self.__class__.__name__}')


    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        for i, op in enumerate(self.to_list()):
            re = str(np.real(op))
            im = str(np.imag(op))
            out += f"Op {i} Re:\n" + re + f"\nOp {i} Im:\n" + im + "\n"
        return out

    def apply_one_body_operator(self, particle_index: int, isospin_matrix: np.ndarray, spin_matrix: np.ndarray):
        assert isospin_matrix.shape == (2, 2)
        assert spin_matrix.shape == (2, 2)
        onebody_matrix = repeated_kronecker_product([isospin_matrix, spin_matrix])
        out = self.copy()
        out.op_stack[particle_index] = np.matmul(onebody_matrix, out.op_stack[particle_index], dtype=complex)
        return out

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index,
                                          isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))
        return out

    def tau(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index,
                                          isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))
        return out

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        out = self.copy()
        out = out.apply_one_body_operator(particle_index=particle_index,
                                          isospin_matrix = b * np.identity(2),
                                          spin_matrix = np.identity(2))
        return out

    def spread_scalar_mult(self, b):
        assert np.isscalar(b)
        c = b ** (1 / self.n_particles)
        out = self.copy()
        for i in range(self.n_particles):
            out = out.scalar_mult(i, c)
        return out

    def zeros(self):
        out = self.copy()
        out.op_stack *= 0.0
        return out

    def dagger(self):
        out = self.copy()
        out.op_stack = np.transpose(self.op_stack, axes=(0,2,1)).conj()
        return out        
    
    
            

# COUPLINGS / POTENTIALS

class Coupling:
    def __init__(self, n_particles, shape, file=None):
        self.n_particles = n_particles
        self.shape = shape
        self.coefficients = np.zeros(shape=self.shape)
        if file is not None:
            self.read(file)

    def copy(self):
        out = self.__class__(self.n_particles, self.shape)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out

    def read(self, filename):
        if self.shape is None:
            raise ValueError("Must define self.shape before reading from file.")
        self.coefficients = read_from_file(filename, shape=self.shape)

    def __getitem__(self, key):
        return self.coefficients[key]

    def __str__(self):
        return str(self.coefficients)
    
        
class SigmaCoupling(Coupling):
    """container class for couplings A^sigma (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for a in range(3):
                    for b in range(3):
                        assert self.coefficients[a,i,b,j]==self.coefficients[a,j,b,i]
    

class SigmaTauCoupling(Coupling):
    """container class for couplings A ^ sigma tau (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()
        
    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for a in range(3):
                    for b in range(3):
                        assert self.coefficients[a,i,b,j]==self.coefficients[a,j,b,i]
        

class TauCoupling(Coupling):
    """container class for couplings A^tau (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (n_particles, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                assert self.coefficients[i,j]==self.coefficients[j,i]


class CoulombCoupling(Coupling):
    """container class for couplings V^coul (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (n_particles, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                assert self.coefficients[i,j]==self.coefficients[j,i]


class SpinOrbitCoupling(Coupling):
    """container class for couplings g_LS (a,i)
    for i = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (3, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        # gLS is a vector. no symmetry to validate.   


class ThreeBodyCoupling(Coupling):
    """container class for couplings A(a,i,b,j,c,k)
    for i, j, k = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None, validate=True):
        shape = (3, n_particles, 3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if validate:
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,i,b,k,c,j]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,j,b,i,c,k]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,j,b,k,c,i]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,k,b,i,c,j]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,k,b,j,c,i]


class ArgonnePotential:
    def __init__(self, n_particles):
        self.n_particles = n_particles
        self.sigma = SigmaCoupling(n_particles)
        self.sigmatau = SigmaTauCoupling(n_particles)
        self.tau = TauCoupling(n_particles)
        self.coulomb = CoulombCoupling(n_particles)
        self.spinorbit = SpinOrbitCoupling(n_particles)
    
    def read_sigma(self, filename):
        self.sigma.read(filename)

    def read_sigmatau(self, filename):
        self.sigmatau.read(filename)

    def read_tau(self, filename):
        self.tau.read(filename)

    def read_coulomb(self, filename):
        self.coulomb.read(filename)

    def read_spinorbit(self, filename):
        self.spinorbit.read(filename)



# PROPAGATOR CLASSES

class GFMCPropagatorHS():
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, include_prefactor=True, mix=True, seed=0):
        self.n_particles = n_particles
        self.dt = dt
        self.include_prefactor = include_prefactor
        self.mix = mix
        
        self._ident = GFMCSpinIsospinOperator(self.n_particles)
        self._sig_op = [[self._ident.sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[self._ident.tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._onebody_idx = np.arange(self.n_particles)
        self._pair_idx = np.array(interaction_indices(self.n_particles))
        self._aa = np.arange(3)
        self._bb = np.arange(3)
        self._cc = np.arange(3)
        self._rng = np.random.default_rng(seed=seed)

    def _shuf(self, x: np.ndarray):
        self._rng.shuffle(x)
        
    def apply_one_sample(self, ket, k, x, operator_i, operator_j):
        assert type(ket)==GFMCSpinIsospinState
        assert np.isscalar(k)
        assert np.isscalar(x)
        arg = csqrt(-k)*x
        if self.include_prefactor:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        gi = ccosh(arg) * self._ident + csinh(arg) * operator_i
        gj = ccosh(arg) * self._ident + csinh(arg) * operator_j
        return prefactor * gi * gj * ket
    
    def apply_sigma(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        return ket_prop
    
    def apply_sigmatau(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential,  aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 27*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._cc)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    for c in self._cc:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a]*self._tau_op[i][c], self._sig_op[j][b]*self._tau_op[j][c])
                        idx += 1
        return ket_prop
    
    def apply_tau(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 3*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][a], self._tau_op[j][a])
        return ket_prop

    def apply_coulomb(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: list, mix=True):
        ket_prop = ket.copy()
        assert len(aux) >= len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
                one_body_i = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[i][2]
                one_body_j = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[j][2]
                ket_prop = one_body_i.exponentiate() * one_body_j.exponentiate() * ket_prop
                k = 0.125 * self.dt * potential.coulomb[i,j]
                ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][2], self._tau_op[j][2])
                idx += 1
        return ket_prop
    
    def apply_spinorbit(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: list, mix=True):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._onebody_idx)
        for i in self._onebody_idx:
            for a in self._aa:
                one_body = - 1.j * potential.spinorbit[a,i] * self._sig_op[i][a]
                ket_prop = one_body.exponentiate() * ket_prop
        if self.mix:
            self._shuf(self._aa); self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        ket_prop = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * ket_prop
        return ket_prop
    

class GFMCPropagatorRBM():
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, include_prefactor=True, mix=True, seed=0):
        self.n_particles = n_particles
        self.dt = dt
        self.include_prefactor = include_prefactor
        self.mix = mix

        self._ident = GFMCSpinIsospinOperator(self.n_particles)
        self._sig_op = [[self._ident.sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[self._ident.tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._onebody_idx = np.arange(self.n_particles)
        self._pair_idx = np.array(interaction_indices(self.n_particles))
        self._aa = np.arange(3)
        self._bb = np.arange(3)
        self._cc = np.arange(3)
        self._rng = np.random.default_rng(seed=seed)
        
    def _shuf(self, x: np.ndarray):
        self._rng.shuffle(x)

    def apply_one_sample(self, ket, k, h, operator_i, operator_j):
        assert type(ket)==GFMCSpinIsospinState
        assert np.isscalar(k)
        assert np.isscalar(h)
        if self.include_prefactor:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        sgn = k/abs(k)
        gi = ccosh(arg) * self._ident + csinh(arg) * operator_i
        gj = ccosh(arg) * self._ident - sgn*csinh(arg) * operator_j
        return prefactor * gi * gj * ket


    def apply_sigma(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        return ket_prop
    
    def apply_sigmatau(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 27*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._cc)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    for c in self._cc:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a]*self._tau_op[i][c], self._sig_op[j][b]*self._tau_op[j][c])
                        idx += 1
        return ket_prop
    
    def apply_tau(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 3*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][a], self._tau_op[j][a])
                    idx += 1
        return ket_prop

    def apply_coulomb(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
                one_body_i = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[i][2]
                one_body_j = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[j][2]
                ket_prop = one_body_i.exponentiate() * one_body_j.exponentiate() * ket_prop
                k = 0.125 * self.dt * potential.coulomb[i,j]
                ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][2], self._tau_op[j][2])
                idx += 1
        return ket_prop
    
    def apply_spinorbit(self, ket: GFMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._onebody_idx)
        for i in self._onebody_idx:
            for a in self._aa:
                one_body = - 1.j * potential.spinorbit[a,i] * self._sig_op[i][a]
                ket_prop = one_body.exponentiate() * ket_prop
        if self.mix:
            self._shuf(self._aa); self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        ket_prop = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * ket_prop
        return ket_prop


class AFDMCPropagatorHS():
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, include_prefactor=True, mix=True, seed=0):
        self.n_particles = n_particles
        self.dt = dt
        self.include_prefactor = include_prefactor
        self.mix = mix
        
        self._ident = np.identity(4)
        self._sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        self._tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
        self._onebody_idx = np.arange(self.n_particles)
        self._pair_idx = np.array(interaction_indices(self.n_particles))
        self._aa = np.arange(3)
        self._bb = np.arange(3)
        self._cc = np.arange(3)
        self._rng = np.random.default_rng(seed=seed)

    def _shuf(self, x: np.ndarray):
        self._rng.shuffle(x)
        
    def apply_onebody(k, i, opi):
        """exp (- k opi)"""
        out = AFDMCSpinIsospinOperator(nt.n_particles)
        out.op_stack[i] = ccosh(k) * ident - csinh(k) * opi
        return out

    def apply_twobody_sample(self, ket, k, x, operator_i, operator_j):
        """exp ( sqrt( -kx ) opi opj)"""
        assert type(ket)==AFDMCSpinIsospinState
        assert np.isscalar(k)
        assert np.isscalar(x)
        arg = csqrt(-k)*x
        if self.include_prefactor:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        out = AFDMCSpinIsospinOperator(self.n_particles)
        out.op_stack[i] = ccosh(arg) * self._ident + csinh(arg) * operator_i
        out.op_stack[j] = ccosh(arg) * self._ident + csinh(arg) * operator_j
        out.op_stack[i] *= csqrt(prefactor)
        out.op_stack[j] *= csqrt(prefactor)
        return out
    
    def apply_sigma(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        return ket_prop
    
    def apply_sigmatau(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential,  aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 27*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._cc)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    for c in self._cc:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a]*self._tau_op[i][c], self._sig_op[j][b]*self._tau_op[j][c])
                        idx += 1
        return ket_prop
    
    def apply_tau(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 3*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][a], self._tau_op[j][a])
        return ket_prop

    def apply_coulomb(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: list, mix=True):
        ket_prop = ket.copy()
        assert len(aux) >= len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
                one_body_i = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[i][2]
                one_body_j = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[j][2]
                ket_prop = one_body_i * one_body_j.exponentiate() * ket_prop
                k = 0.125 * self.dt * potential.coulomb[i,j]
                ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][2], self._tau_op[j][2])
                idx += 1
        return ket_prop
    
    def apply_spinorbit(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: list, mix=True):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._onebody_idx)
        for i in self._onebody_idx:
            for a in self._aa:
                one_body = - 1.j * potential.spinorbit[a,i] * self._sig_op[i][a]
                ket_prop = one_body.exponentiate() * ket_prop
        if self.mix:
            self._shuf(self._aa); self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        ket_prop = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * ket_prop
        return ket_prop
    

class AFDMCPropagatorRBM():
    """ exp( - k op_i op_j )
    seed determines mixing
    """
    def __init__(self, n_particles, dt, include_prefactor=True, mix=True, seed=0):
        self.n_particles = n_particles
        self.dt = dt
        self.include_prefactor = include_prefactor
        self.mix = mix

        self._ident = AFDMCSpinIsospinOperator(self.n_particles)
        self._sig_op = [[self._ident.sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[self._ident.tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._onebody_idx = np.arange(self.n_particles)
        self._pair_idx = np.array(interaction_indices(self.n_particles))
        self._aa = np.arange(3)
        self._bb = np.arange(3)
        self._cc = np.arange(3)
        self._rng = np.random.default_rng(seed=seed)
        
    def _shuf(self, x: np.ndarray):
        self._rng.shuffle(x)

    def apply_one_sample(self, ket, k, h, operator_i, operator_j):
        assert type(ket)==AFDMCSpinIsospinState
        assert np.isscalar(k)
        assert np.isscalar(h)
        if self.include_prefactor:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        sgn = k/abs(k)
        gi = ccosh(arg) * self._ident + csinh(arg) * operator_i
        gj = ccosh(arg) * self._ident - sgn*csinh(arg) * operator_j
        gi = gi.scalar_mult(i, csqrt(prefactor))
        gj = gj.scalar_mult(j, csqrt(prefactor))
        return gi * gj * ket


    def apply_sigma(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: list):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        return ket_prop
    
    def apply_sigmatau(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 27*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._bb)
            self._shuf(self._cc)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    for c in self._cc:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a]*self._tau_op[i][c], self._sig_op[j][b]*self._tau_op[j][c])
                        idx += 1
        return ket_prop
    
    def apply_tau(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 3*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][a], self._tau_op[j][a])
                    idx += 1
        return ket_prop

    def apply_coulomb(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
                one_body_i = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[i][2]
                one_body_j = - 0.125 * potential.coulomb[i,j] * self.dt * self._tau_op[j][2]
                ket_prop = one_body_i.exponentiate() * one_body_j.exponentiate() * ket_prop
                k = 0.125 * self.dt * potential.coulomb[i,j]
                ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._tau_op[i][2], self._tau_op[j][2])
                idx += 1
        return ket_prop
    
    def apply_spinorbit(self, ket: AFDMCSpinIsospinState, potential: ArgonnePotential, aux: np.ndarray):
        ket_prop = ket.copy()
        assert len(aux) >= 9*len(self._pair_idx)
        idx = 0
        if self.mix:
            self._shuf(self._aa)
            self._shuf(self._onebody_idx)
        for i in self._onebody_idx:
            for a in self._aa:
                one_body = - 1.j * potential.spinorbit[a,i] * self._sig_op[i][a]
                ket_prop = one_body.exponentiate() * ket_prop
        if self.mix:
            self._shuf(self._aa); self._shuf(self._bb)
            self._shuf(self._pair_idx)
        for i,j in self._pair_idx:
            for a in self._aa:
                for b in self._bb:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    ket_prop = self.apply_one_sample(ket_prop, k, aux[idx], self._sig_op[i][a], self._sig_op[j][b])
                    idx += 1
        ket_prop = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * ket_prop
        return ket_prop
    



########### 

class Integrator():
    def __init__(self, potential: ArgonnePotential, propagator, mix=True):
        if type(propagator) in [GFMCPropagatorHS, AFDMCPropagatorHS]:
            self.method = 'HS'
        elif type(propagator) in [GFMCPropagatorRBM, AFDMCPropagatorRBM]:
            self.method = 'RBM'
        self.n_particles = potential.n_particles
        self.potential = potential
        self.propagator = propagator
        self.mix = mix
        self.controls = {"sigma":False, "sigmatau":False, "tau":False, "coulomb":False, "spinorbit":False, "balanced": True}
        self.is_ready = False

    def setup(self, n_samples: int, seed=1729):
        self.n_samples = n_samples
        n_aux = 0
        if self.controls['sigma']:
            n_aux += 9
        if self.controls['sigmatau']:
            n_aux += 27
        if self.controls['tau']:
            n_aux += 3
        if self.controls['coulomb']:
            n_aux += 1
        if self.controls['spinorbit']:
            n_aux += 9
        n_aux = n_aux * len(self.propagator._pair_idx)
        
        rng = np.random.default_rng(seed=seed)
        if self.method=='HS':
            self.aux_fields = rng.standard_normal(size=(n_samples,n_aux))
            if self.controls["balanced"]:
                self.aux_fields = np.concatenate([self.aux_fields, -self.aux_fields], axis=0)
        elif self.method=='RBM':
            self.aux_fields = rng.integers(0,2,size=(n_samples,n_aux))
            if self.controls["balanced"]:
                self.aux_fields = np.concatenate([self.aux_fields, np.ones_like(self.aux_fields) - self.aux_fields], axis=0)
        self.is_ready = True

    def bracket(self, bra, ket, aux_fields):
        ket_prop = ket.copy()
        i=0
        if self.controls['sigma']:
            ket_prop = self.propagator.apply_sigma(ket_prop, self.potential, aux_fields[i:i+9], )
            i+=9
        if self.controls['sigmatau']:
            ket_prop = self.propagator.apply_sigmatau(ket_prop, self.potential, aux_fields[i:i+27])
            i+=27
        if self.controls['tau']:
            ket_prop = self.propagator.apply_tau(ket_prop, self.potential, aux_fields[i:i+3])
            i+=3
        if self.controls['coulomb']:
            ket_prop = self.propagator.apply_coulomb(ket_prop, self.potential, aux_fields[i:i+1])
            i+=1
        if self.controls['spinorbit']:
            ket_prop = self.propagator.apply_spinorbit(ket_prop, self.potential, aux_fields[i:i+9])
            i+=9
        return bra * ket_prop

    def run(self, bra: State, ket: State, parallel=True, n_processes=None):
        if not self.is_ready:
            raise ValueError("Integrator is not ready. Did you run .setup() ?")
        assert (bra.orientation=='bra') and (ket.orientation=='ket')
        if parallel:
            with Pool(processes=n_processes) as pool:
                b_array = pool.starmap_async(self.bracket, tqdm([(bra.copy(), ket.copy(), aux) for aux in self.aux_fields], leave=True)).get()
        else:
            b_array = list(itertools.starmap(self.bracket, tqdm([(bra.copy(), ket.copy(), aux) for aux in self.aux_fields])))
        b_array = np.array(b_array).flatten()
        return b_array
            