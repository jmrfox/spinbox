# quap
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.5'

# imports
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng

from scipy.linalg import expm
from functools import reduce

from dataclasses import dataclass

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
        sp = np.array([tuple_to_complex(x) for x in c], dtype=complex)
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
    """turns a 1-element array  x into a scalar
    is there a better way to do this?
    """
    assert type(x) == np.ndarray
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

class ManyBodyBasisSpinState(State):
    def __init__(self, n_particles: int, orientation: str, coefficients: np.ndarray):
        super().__init__(n_particles, orientation)
        self.dim = 2 ** self.n_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')
        self.friendly_operator = ManyBodyBasisSpinOperator

    def copy(self):
        return ManyBodyBasisSpinState(self.n_particles, self.orientation, self.coefficients.copy())
    
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
            if isinstance(other, ManyBodyBasisSpinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients, other.coefficients)
            elif isinstance(other, ManyBodyBasisSpinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            out = ManyBodyBasisSpinOperator(self.n_particles)
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
        out = ManyBodyBasisSpinState(self.n_particles, new_orientation, self.coefficients.conj().T)
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
    

class ManyBodyBasisSpinIsospinState(State):
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
        self.friendly_operator = ManyBodyBasisSpinIsospinOperator
    
    def copy(self):
        return ManyBodyBasisSpinIsospinState(self.n_particles, self.orientation, self.coefficients.copy())
    
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
            if isinstance(other, ManyBodyBasisSpinIsospinState):
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients, other.coefficients)
            elif isinstance(other, ManyBodyBasisSpinIsospinOperator):
                out = self.copy()
                out.coefficients = np.matmul(self.coefficients, other.matrix)
            else:
                raise TypeError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert other.orientation == 'bra'
            out = ManyBodyBasisSpinIsospinOperator(self.n_particles)
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
        out = ManyBodyBasisSpinIsospinState(self.n_particles, new_orientation, self.coefficients.conj().T)
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


class ManyBodyBasisSpinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.matrix = np.identity(2 ** n_particles, dtype=complex)
        self.friendly_state = ManyBodyBasisSpinState

    def copy(self):
        out = ManyBodyBasisSpinOperator(self.n_particles)
        out.matrix = self.matrix.copy()
        return out
    
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
        P_1 = ManyBodyBasisSpinOperator(n_particles=self.n_particles)
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


class ManyBodyBasisSpinIsospinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.matrix = np.identity(4 ** n_particles, dtype=complex)
        self.friendly_state = ManyBodyBasisSpinIsospinState

    def copy(self):
        out = ManyBodyBasisSpinIsospinOperator(self.n_particles)
        out.matrix = self.matrix.copy()
        return out
    
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
        P_1 = ManyBodyBasisSpinIsospinOperator(n_particles=self.n_particles)
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
        

class OneBodyBasisSpinState(State):
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
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.sp_stack = coefficients.astype(complex)
        self.friendly_operator = OneBodyBasisSpinOperator
        

    def __add__(self, other):
        raise SyntaxError('You should not be adding a OBB states')
    
    def __sub__(self, other):
        raise SyntaxError('You should not be subtracting a OBB states')

    def copy(self):
        return OneBodyBasisSpinState(self.n_particles, self.orientation, self.sp_stack.copy())

    def to_list(self):
        return [self.sp_stack[i] for i in range(self.n_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, OneBodyBasisSpinState):
                assert other.orientation == 'ket'
                out = np.prod([np.dot(self.sp_stack[i], other.sp_stack[i]) for i in range(self.n_particles)])
            elif isinstance(other, OneBodyBasisSpinOperator):
                out = self.copy()
                for i in range(self.n_particles):
                    out.sp_stack[i] = np.matmul(self.sp_stack[i], other.op_stack[i], dtype=complex)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, OneBodyBasisSpinState) and other.orientation == 'bra'
            out = OneBodyBasisSpinOperator(n_particles=self.n_particles)
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
        return ManyBodyBasisSpinState(self.n_particles, self.orientation, sp_mb)

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
    

class OneBodyBasisSpinIsospinState(State):
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
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.sp_stack = coefficients.astype(complex)
        self.friendly_operator = OneBodyBasisSpinIsospinOperator
        

    def __add__(self, other):
        raise SyntaxError('You should not be adding a OBB states')
    
    def __sub__(self, other):
        raise SyntaxError('You should not be subtracting a OBB states')

    def copy(self):
        return OneBodyBasisSpinIsospinState(self.n_particles, self.orientation, self.sp_stack.copy())

    def to_list(self):
        return [self.sp_stack[i] for i in range(self.n_particles)]

    def __mul__(self, other):
        """
        bra can multiply a ket or an operator, ket can only multiply a bra
        """
        if self.orientation == 'bra':  # inner product
            if isinstance(other, OneBodyBasisSpinIsospinState):
                assert other.orientation == 'ket'
                out = np.prod([np.dot(self.sp_stack[i], other.sp_stack[i]) for i in range(self.n_particles)])
            elif isinstance(other, OneBodyBasisSpinIsospinOperator):
                out = self.copy()
                for i in range(self.n_particles):
                    out.sp_stack[i] = np.matmul(self.sp_stack[i], other.op_stack[i], dtype=complex)
            else:
                raise ValueError(f'{self.__class__.__name__} * {other.__class__.__name__}, invalid multiplication')
            return out
        elif self.orientation == 'ket':  # outer product
            assert isinstance(other, OneBodyBasisSpinIsospinState) and other.orientation == 'bra'
            out = OneBodyBasisSpinIsospinOperator(n_particles=self.n_particles)
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
        return ManyBodyBasisSpinIsospinState(self.n_particles, self.orientation, sp_mb)

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


class OneBodyBasisSpinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.op_stack = np.stack(self.n_particles*[np.identity(2)], dtype=complex)    
        self.friendly_state = OneBodyBasisSpinState

    def __add__(self, other):
        raise SyntaxError('You should not be adding OBB operators')
    
    def __sub__(self, other):
        raise SyntaxError('You should not be subtracting OBB operators')

    def copy(self):
        out = OneBodyBasisSpinOperator(self.n_particles)
        for i in range(self.n_particles):
            out.op_stack[i] = self.op_stack[i]
        return out

    def to_list(self):
        return [self.op_stack[i] for i in range(self.n_particles)]
    
    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            for i in range(self.n_particles):
                out.sp_stack[i] = np.matmul(self.op_stack[i], out.sp_stack[i], dtype=complex)
            return out
        elif isinstance(other, OneBodyBasisSpinOperator):
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
    

class OneBodyBasisSpinIsospinOperator(Operator):
    def __init__(self, n_particles: int):
        super().__init__(n_particles)
        self.op_stack = np.stack(self.n_particles*[np.identity(4)], dtype=complex)    
        self.friendly_state = OneBodyBasisSpinIsospinState

    def __add__(self, other):
        raise SyntaxError('You should not be adding OBB operators')
    
    def __sub__(self, other):
        raise SyntaxError('You should not be subtracting OBB operators')

    def copy(self):
        out = OneBodyBasisSpinIsospinOperator(self.n_particles)
        for i in range(self.n_particles):
            out.op_stack[i] = self.op_stack[i]
        return out

    def to_list(self):
        return [self.op_stack[i] for i in range(self.n_particles)]
    
    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinIsospinState):
            assert other.orientation == 'ket'
            out = other.copy()
            for i in range(self.n_particles):
                out.sp_stack[i] = np.matmul(self.op_stack[i], out.sp_stack[i], dtype=complex)
            return out
        elif isinstance(other, OneBodyBasisSpinIsospinOperator):
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
    
    
            

# POTENTIAL CLASSES

class SigmaCoupling:
    """container class for couplings A^sigma (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, coefficients:np.ndarray, validate=True):
        self.shape = (3, n_particles, 3, n_particles)
        if validate:
            assert coefficients.shape()==self.shape
            for i in range(n_particles):
                for j in range(n_particles):
                    for a in range(3):
                        for b in range(3):
                            assert coefficients[a,i,b,j]==coefficients[b,j,a,i]
        
        self.n_particles = n_particles
        self.coefficients = coefficients
    
    def copy(self):
        out = SpinOrbitCoupling(self.n_particles, self.coefficients)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out

    def read(self, filename):
        out = self.copy()
        out.coefficients = read_from_file(filename, shape=self.shape)
        return out

class SigmaTauCoupling:
    """container class for couplings A ^ sigma tau (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, coefficients:np.ndarray, validate=True):
        self.shape = (3, n_particles, 3, n_particles)
        if validate:
            assert coefficients.shape()==self.shape
            for i in range(n_particles):
                for j in range(n_particles):
                    for a in range(3):
                        for b in range(3):
                            assert coefficients[a,i,b,j]==coefficients[b,j,a,i]
        self.n_particles = n_particles
        self.coefficients = coefficients
    
    def copy(self):
        out = SpinOrbitCoupling(self.n_particles, self.coefficients)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out

    def read(self, filename):
        out = self.copy()
        out.coefficients = read_from_file(filename, shape=self.shape)
        return out

class TauCoupling:
    """container class for couplings A^tau (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, coefficients:np.ndarray, validate=True):
        self.shape = (3, n_particles, 3, n_particles)
        if validate:
            assert coefficients.shape()==self.shape
            for i in range(n_particles):
                for j in range(n_particles):
                    assert coefficients[i,j]==coefficients[j,i]
        self.n_particles = n_particles
        self.coefficients = coefficients
    
    def copy(self):
        out = SpinOrbitCoupling(self.n_particles, self.coefficients)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out

    def read(self, filename):
        out = self.copy()
        out.coefficients = read_from_file(filename, shape=self.shape)
        return out

class CoulombCoupling:
    """container class for couplings V^coul (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, coefficients:np.ndarray, validate=True):
        self.shape = (3, n_particles, 3, n_particles)
        if validate:
            assert coefficients.shape()==self.shape
            for i in range(n_particles):
                for j in range(n_particles):
                    assert coefficients[i,j]==coefficients[j,i]
        self.n_particles = n_particles
        self.coefficients = coefficients
    
    def copy(self):
        out = SpinOrbitCoupling(self.n_particles, self.coefficients)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out

    def read(self, filename):
        out = self.copy()
        out.coefficients = read_from_file(filename, shape=self.shape)
        return out


class SpinOrbitCoupling:
    """container class for couplings g_LS (a,i,j)
    for i, j = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, coefficients:np.ndarray, validate=True):
        self.shape = (3, n_particles, 3, n_particles)
        if validate:
            assert coefficients.shape()==self.shape
            for i in range(n_particles):
                for j in range(n_particles):
                    for a in range(3):
                        assert coefficients[a,i,j]==coefficients[a,j,i]
        self.n_particles = n_particles
        self.coefficients = coefficients
    
    def copy(self):
        out = SpinOrbitCoupling(self.n_particles, self.coefficients)
        return out

    def __mult__(self, other):
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out
    
    def read(self, filename):
        out = self.copy()
        out.coefficients = read_from_file(filename, shape=self.shape)
        return out


class ArgonnePotential:
    def __init__(self, n_particles):
        self.sigma = SigmaCoupling(n_particles, coefficients=np.zeros(shape=(3,n_particles,3,n_particles)))
        self.sigmatau = SigmaTauCoupling(n_particles, coefficients=np.zeros(shape=(3,n_particles,3,n_particles)))
        self.tau = TauCoupling(n_particles, coefficients=np.zeros(shape=(n_particles,n_particles)))
        self.coulomb = CoulombCoupling(n_particles, coefficients=np.zeros(shape=(n_particles,n_particles)))
        self.spinorbit = SpinOrbitCoupling(n_particles, coefficients=np.zeros(shape=(3,n_particles,n_particles)))
    
    def read_sigma(self, filename):
        self.sigma = self.sigma.read(filename)

    def read_sigmatau(self, filename):
        self.sigmatau = self.sigmatau.read(filename)

    def read_tau(self, filename):
        self.tau = self.tau.read(filename)

    def read_coulomb(self, filename):
        self.coulomb = self.coulomb.read(filename)

    def read_spinorbit(self, filename):
        self.spinorbit = self.spinorbit.read(filename)


    

# PROPAGATOR CLASSES

class Propagator():
    def __init__(self) -> None:
        pass