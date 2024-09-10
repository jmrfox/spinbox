# spinbox
# tools for many-body spin systems in a Monte Carlo context
# jordan m r fox 2024

__version__ = '0.1.0'

import sys
import numpy as np
np.set_printoptions(linewidth=200, threshold=sys.maxsize)
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from functools import reduce
# from dataclasses import dataclass
import itertools
from multiprocessing.pool import Pool
from tqdm import tqdm

## i tried to put in numba at one point but couldn't get it working right :/
# from numba import int8, int32, float64, boolean
# from numba import types, typed
# from numba.experimental import jitclass


###

# functions
# redefine basic fxns to be complex (maybe unnecessary, but better safe than sorry)
# numpy.sqrt will raise warning (NOT an error) if you give it a negative number, so i'm not taking any chances
# these use numpy instead of math to be safe with ndarrays

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

def cabs(x):
    """Weird complex absolute value function needed for the RBM"""
    return np.abs(np.real(x)) + 1j*np.abs(np.imag(x))
    

def interaction_indices(n: int, m = 2) -> list:
    """ returns a list of all possible m-plets of n objects (labelled 0 to n-1)
    default: m=2, giving all possible pairs
    for m=1, returns a range(0, n-1)
    :param n: number of items
    :type n: int
    :param m: size of tuplet, defaults to 2
    :type m: int, optional
    :return: list of possible m-plets of n items
    :rtype: list
    """    # """
    if m==1:
        return np.arange(n)
    else:
        return np.array(list(itertools.combinations(range(n), m)))


def read_from_file(filename: str, complex=False, shape=None, order='F') -> np.ndarray:
    """Read numbers from a text file

    :param filename: input file name
    :type filename: str
    :param complex: complex entries, defaults to False
    :type complex: bool, optional
    :param shape: shape of output array, defaults to None
    :type shape: tuple, optional
    :param order: 'F' for columns first, otherwise use 'C', defaults to 'F'
    :type order: str, optional
    :return: Numpy array
    :rtype: numpy.ndarray
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


def pauli(arg) -> np.ndarray:
    """Pauli matrix x, y, z, or a list of all three

    :param arg: 0 or 'x' for Pauli x, 1 or 'y' for Pauli y, 2 or 'z' for Pauli z, 'list' for a list of x, y ,z 
    :type arg: int or str
    :raises ValueError: option not found
    :return: Pauli matrix or list
    :rtype: np.ndarray
    """    
    if arg in [0, 'x']:
        out = np.array([[0, 1], [1, 0]], dtype=complex)
    elif arg in [1, 'y']:
        out = np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif arg in [2, 'z']:
        out = np.array([[1, 0], [0, -1]], dtype=complex)
    elif arg in [3, 'i']:
        out = np.array([[1, 0], [0, 1]], dtype=complex)
    elif arg in ['list']:
        out = [np.array([[0, 1], [1, 0]], dtype=complex),
               np.array([[0, -1j], [1j, 0]], dtype=complex),
               np.array([[1, 0], [0, -1]], dtype=complex)]
    else:
        raise ValueError(f'No option: {arg}')
    return out


def rlprod(factors:list):
    """Multiplies a list of factors right to left

    Args:
        factors (list): things to be multiplied

    """    
    out = factors[-1]
    for x in reversed(factors[:-1]):
        out = x * out
    return out


def kronecker_product(matrices: list) -> np.ndarray:
    """
    returns the kronecker (i.e. tensor) product of a list of arrays
    :param matrices: list of matrix factors
    :type matrices: list
    :return: Kronecker product of input list
    "rtype: np.ndarray
    """
    return np.array(reduce(np.kron, matrices), dtype=complex)



# Hilbert BASIS CLASSES

# numbaspec_hilbertstate = [
#     ('n_particles', int32),
#     ('isospin', boolean),
#     ('n_basis', int8),
#     ('dimension', int32),
#     ('ketwise', boolean),
#     ('coefficients', float64[:,:])
# ]
# @jitclass(numbaspec_hilbertstate)
class HilbertState:
    """A spin state in the "Hilbert" basis, a linear combination of tensor product states.
    
    States must be defined with a number of particles. 
    If ``isospin`` is False, then the one-body basis is only spin up/down. If True, then it is (spin up/down x isospin up/down).
    ``ketwise`` detemines if it is a bra or a ket.
     
    """    
    def __init__(self, n_particles: int, coefficients=None, ketwise=True, isospin=True):
        """Instances a new ``HilbertState``

        :param n_particles: number of particles
        :type n_particles: int
        :param coefficients: an optional array of coefficients, defaults to None
        :type coefficients: numpy.ndarray, optional
        :param ketwise: True for column vector, False for row vector, defaults to True
        :type ketwise: bool, optional
        :param isospin: True for spin-isospin state, False for spin only, defaults to True
        :type isospin: bool, optional
        :raises ValueError: inconsistency in chosen options
        """        
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.dimension = self.n_basis ** self.n_particles
        self.ketwise = ketwise
        
        if coefficients is None:
            if ketwise:
                self.coefficients = np.zeros(shape=(self.dimension, 1))
            else:
                self.coefficients = np.zeros(shape=(1, self.dimension))
        else: 
            # assert type(coefficients)==np.ndarray
            ket_condition = (coefficients.shape == (self.dimension, 1)) and ketwise
            bra_condition = (coefficients.shape == (1, self.dimension)) and not ketwise
            if not ket_condition and not bra_condition:
                raise ValueError("Inconsistent initialization of state vector. \n\
                                Did you get the shape right?")
            else:
                self.coefficients = coefficients.astype('complex')

    def copy(self):
        """Copies the ``HilbertState``.

        :return: a new instance of ``HilbertState`` with all the same properties.
        :rtype: HilbertState
        """        
        return HilbertState(n_particles=self.n_particles, coefficients=self.coefficients.copy(), ketwise=self.ketwise, isospin=self.isospin)
    
    def __add__(self, other: 'HilbertState') -> 'HilbertState':
        """Sums two states. Orientations must be the same.

        :param other: Other ``HilbertState`` to be added.
        :type other: HilbertState
        :return: A new ``HilbertState`` with coefficients given by self + other
        :rtype: HilbertState
        """
        assert isinstance(other, HilbertState)
        assert self.ketwise == other.ketwise
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other: 'HilbertState') -> 'HilbertState':
        """Subtracts one ``HilbertState`` from another. Orientations must be the same.

        :param other: ``HilbertState`` to be subtracted.
        :type other: HilbertState
        :return: A new ``HilbertState`` with coefficients given by self - other
        :rtype: HilbertState
        """        
        assert isinstance(other, HilbertState)
        assert self.ketwise == other.ketwise
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def scale(self, other: complex) -> 'HilbertState':
        """Scalar multiple of a ``HilbertState``.

        :param other: Scalar number to multiply by.
        :type other: complex
        :return: other * self
        :rtype: HilbertState
        """        
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients *= other
        return out

    def inner(self, other: 'HilbertState') -> complex:
        """Inner product of two HilbertState instances. Orientations must be correct.

        :param other: The ket of the inner product.
        :type other: HilbertState
        :return: inner product of self (bra) with other (ket)
        :rtype: complex
        """        
        assert isinstance(other, HilbertState)
        assert not self.ketwise and other.ketwise
        return np.dot(self.coefficients, other.coefficients)
        
    def outer(self, other: 'HilbertState') -> 'HilbertOperator':
        """Outer product of two HilbertState instances, producting a HilbertOperator instance. Orientations must be correct.

        :param other: bra part of the outer product
        :type other: HilbertState
        :return: Outer product of self (ket) with other (bra)
        :rtype: HilbertOperator
        """        
        assert isinstance(other, HilbertState)
        assert self.ketwise and not other.ketwise
        out = HilbertOperator(n_particles=self.n_particles, isospin=self.isospin)
        out.coefficients = np.matmul(self.coefficients, other.coefficients, dtype='complex')
        return out 
    
    def multiply_operator(self, other: 'HilbertOperator') -> 'HilbertState':
        """Multiplies a (bra) ``HilbertState`` on a ``HilbertOperator``.

        :param other: The operator.
        :type other: HilbertOperator
        :return: < self| O(other) 
        :rtype: HilbertState
        """        
        assert isinstance(other, HilbertOperator)
        assert not self.ketwise
        out = self.copy()
        out.coefficients = np.matmul(self.coefficients, other.coefficients, dtype='complex') 
        return out
    
    def dagger(self) -> 'HilbertState':
        """Hermitian conjugate.

        :return: The dual ``HilbertState``
        :rtype: HilbertState
        """        
        out = self.copy()
        out.coefficients = self.coefficients.conj().T
        out.ketwise = not self.ketwise
        return out
        
    def __str__(self):
        orient = 'ket'
        if not self.ketwise:
            orient = 'bra'
        out = [f'{self.__class__.__name__} {orient} of {self.n_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)

    def randomize(self, seed:int=None) -> 'HilbertState':
        """Randomize coefficients.

        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :return: A copy of the ``HilbertState`` with random complex coefficients, normalized.
        :rtype: HilbertState
        """        
        rng = np.random.default_rng(seed=seed)
        out = self.copy()
        out.coefficients = rng.standard_normal(size=out.coefficients.shape) + 1.j*rng.standard_normal(size=out.coefficients.shape)
        out.coefficients /= np.linalg.norm(out.coefficients)
        return out
    
    def zero(self) -> 'HilbertState':
        """Set all coefficients to zero.

        :return: A copy of ``HilbertState`` with all coefficients set to zero.
        :rtype: HilbertState
        """        
        out = self.copy()
        out.coefficients = np.zeros_like(self.coefficients)
        return out
    
    def density(self) -> 'HilbertOperator':
        """Calculate the density matrix for this state as an operator.

        :return: the density matrix
        :rtype: HilbertOperator
        """
        if self.ketwise:
            dens = self.outer(self.copy().dagger())
        else:
            dens = self.dagger().outer(self.copy())
        return dens
        
    def entropy(self) -> complex:
        """Von Neumann entropy, a measure of entanglement.

        :return: VN entropy of the ``HilbertState``
        :rtype: complex
        """
        dens = self.density().coefficients
        log_dens = logm(dens)
        return - np.trace(dens @ log_dens)
     
    def generate_basis_states(self) -> list:
        """Makes a list of corresponding basis vectors.

        :return: A list of tensor product states that span the Hilbert space.
        :rtype: list
        """        
        coeffs_list = list(np.eye(self.n_basis**self.n_particles))
        out = [HilbertState(self.n_particles, coefficients=c, ketwise=True, isospsin=self.isospin) for c in coeffs_list]
        return out
    
    def nearby_product_state(self, seed:int=None, maxiter=100):
        """Finds a ``ProductState`` that has a large overlap with the ``HilbertState``.

        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :param maxiter: maximum iterations to do in optimization, defaults to 100
        :type maxiter: int, optional
        :return: a tuple: (fitted ``ProductState``, optimization result)
        :rtype: (ProductState, scipy.OptimizeResult)
        """        
        from scipy.optimize import minimize, NonlinearConstraint
        fit = ProductState(self.n_particles, isospin=self.isospin).randomize(seed)
        shape = fit.coefficients.shape
        n_coeffs = len(fit.coefficients.flatten())
        n_params = 2*n_coeffs
        def x_to_coef(x):
            return x[:n_params//2].reshape(shape) + 1.j*x[n_params//2:].reshape(shape)   
        def loss(x):
            fit.coefficients = x_to_coef(x)
            overlap = self.dagger().inner(fit.to_full_basis())
            return (1 - np.real(overlap))**2 + np.imag(overlap)**2
        def norm(x):
            fit.coefficients = x_to_coef(x)
            return fit.dagger().inner(fit)
        start = np.concatenate([np.real(fit.coefficients).flatten() , np.imag(fit.coefficients).flatten()])
        print(start)
        normalize = NonlinearConstraint(norm,1.0,1.0)
        result = minimize(loss, x0=start, constraints=normalize, options={'maxiter':maxiter,'disp':True},tol=10**-15)
        fit.coefficients = x_to_coef(result.x)
        return fit, result
    
    def nearest_product_state(self, seeds: list[int], maxiter=100) -> 'ProductState':
        """ Does ``self.nearby_product_state`` for a list of seeds and returns the result maximizing overlap
        
        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :param maxiter: maximum iterations to do in optimization, defaults to 100
        :type maxiter: int, optional
        :return: fitted ``ProductState``
        :rtype: ProductState
        """
        overlap = 0.
        for seed in seeds:
            this_fit, _ = self.nearby_product_state(seed, maxiter=maxiter)
            this_overlap = this_fit.to_full_basis().dagger().inner(self)
            if this_overlap>overlap:
                overlap = this_overlap
                fit = this_fit
        return fit
    
    def __mul__(self, other):
        """Defines the ``*`` multiplication operator to be used in place of ``.inner`` , ``.outer``, ``.multiply_operator``, and ``.scale``.
        """        
        if np.isscalar(other): # |s> * scalar
            return self.copy().scale(other)
        elif isinstance(other, HilbertState):
            if self.ketwise and not other.ketwise: # |s><s'|
                return self.outer(other)
            elif not self.ketwise and other.ketwise: # <s|s'>
                return self.inner(other)
            else:
                raise ValueError("Improper multiplication.")
        elif isinstance(other, HilbertOperator):
            if not self.ketwise:
                return self.multiply_operator(other)
        else:
            raise NotImplementedError("Unsupported multiply.")

    def attach_coordinates(self, coordinates: np.ndarray) -> None:
        """Adds a new ``.coordinates`` attribute to the ``HilbertState``

        :param coordinates: A Numpy array with shape ``(n_particles , 3)`` (e.g. x, y, z)
        :type coordinates: np.ndarray
        """        
        assert coordinates.shape == (self.n_particles, 3)
        self.coordinates = coordinates
        
    def exchange(self, i:int, j:int) -> 'HilbertState':
        """Exchanges two particles via the 2P-1= dot product rule

        :param i: Index of particle one
        :type i: int
        :param j: Index of particle two
        :type j: int
        :return: A copy of the state with particles i and j swapped
        :rtype: HilbertState
        """        
        P_1 = HilbertOperator(n_particles=self.n_particles)
        P_x = P_1.copy().sigma(i, 'x').sigma(j, 'x')
        P_y = P_1.copy().sigma(i, 'y').sigma(j, 'y')
        P_z = P_1.copy().sigma(i, 'z').sigma(j, 'z')
        P = (P_x + P_y + P_z + P_1)
        out = 0.5 * P * self.copy()
        if self.isospin:
            P_1 = HilbertOperator(n_particles=self.n_particles)
            P_x = P_1.copy().tau(i, 'x').tau(j, 'x')
            P_y = P_1.copy().tau(i, 'y').tau(j, 'y')
            P_z = P_1.copy().tau(i, 'z').tau(j, 'z')
            P = (P_x + P_y + P_z + P_1)
            out = 0.5 * P * out
        return out

    
# numbaspec_hilbertoperator = [
#     ('n_particles', int32),
#     ('isospin', boolean),
#     ('n_basis', int8),
#     ('dimension', int32),
#     ('ketwise', boolean),
#     ('coefficients', float64[:,:])
# ]
# @jitclass(numbaspec_hilbertoperator)
class HilbertOperator:
    """An operator in the "Hilbert basis.
    """
    def __init__(self, n_particles: int, isospin=True):
        """Instantiates a ``HilbertOperator``.

        :param n_particles: Number of particles
        :type n_particles: int
        :param isospin: True for spin-isospin states, False for spin only, defaults to True
        :type isospin: bool, optional
        """
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.dimension = self.n_basis ** self.n_particles
        self.coefficients = np.identity(self.dimension, dtype=complex)

    def copy(self) -> 'HilbertOperator':
        """Copies the ``HilbertOperator``.

        :return: a new instance of ``HilbertOperator`` with all the same properties as self.
        :rtype: HilbertOperator
        """  
        out = HilbertOperator(n_particles=self.n_particles, isospin=self.isospin)
        out.coefficients = self.coefficients.copy()
        return out
    
    def __add__(self, other: 'HilbertOperator') -> 'HilbertOperator':
        """Sums operator matrices.

        :param other: Other ``HilbertOperator`` to be added.
        :type other: HilbertOperator
        :return: A new ``HilbertOperator`` with coefficients given by self + other
        :rtype: HilbertOperator
        """    
        assert isinstance(other, HilbertOperator)
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other: 'HilbertOperator') -> 'HilbertOperator':
        """Subtracts one ``HilbertOperator`` from another. 
        
        :param other: ``HilbertOperator`` to be subtracted.
        :type other: HilbertOperator
        :return: A new ``HilbertOperator`` with coefficients given by self - other
        :rtype: HilbertOperator
        """ 
        assert isinstance(other, HilbertOperator)
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def multiply_state(self, other: 'HilbertState') -> 'HilbertState':
        """Apply the operator to a ``HilbertState`` ket.

        :param other: The state, ketwise.
        :type other: HilbertState
        :return: The new state, ketwise.
        :rtype: HilbertState
        """               
        assert isinstance(other, HilbertState)
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out
        
    def multiply_operator(self, other: 'HilbertOperator') -> 'HilbertOperator':
        """Multiply two ``HilbertOperator`` instances together to get a new one.

        :param other: The other ``HilbertOperator``
        :type other: HilbertOperator
        :return: The product of the two.
        :rtype: HilbertOperator
        """
        assert isinstance(other, HilbertOperator)
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out
        
    def scale(self, other: complex) -> 'HilbertOperator':
        """Scalar multiplication.

        :param other: A scalar.
        :type other: complex
        :return: The resulting scaled operator.
        :rtype: HilbertOperator
        """
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients *= other
        return out
    
    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.coefficients))
        im = str(np.imag(self.coefficients))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_onebody_operator(self, particle_index: int, spin_matrix: np.ndarray, isospin_matrix:np.ndarray=None):
        r"""Applies a one-body / single-particle operator to the ``HilbertOperator``.
        This accounts for the spin-isospin kronecker product, if isospin is used.

        .. math::
            O' = \sigma_{\alpha i} \tau_{\beta i} O

        :param particle_index: Index of particle to apply onebody operator to, starting from 0.
        :type particle_index: int
        :param spin_matrix: The spin part of the operator, a 2x2 matrix
        :type spin_matrix: np.ndarray
        :param isospin_matrix: The isospin part of the operator, a 2x2 matrix, defaults to None
        :type isospin_matrix: numpy.ndarray, optional
        :return: A copy of the ``HilbertOperator`` with the one-body operator applied.
        :rtype: HilbertOperator
        """
        assert type(spin_matrix) == np.ndarray
        assert spin_matrix.shape == (2,2)
        obo = [np.identity(self.n_basis, dtype=complex) for _ in range(self.n_particles)]
        if self.isospin:
            if isospin_matrix is None:
                isospin_matrix = np.identity(2, dtype=complex)
            else:
                assert isospin_matrix.shape == (2,2)        
            op = kronecker_product([isospin_matrix, spin_matrix])
        else:
            op = spin_matrix
        assert op.shape == obo[particle_index].shape
        obo[particle_index] = op
        obo = kronecker_product(obo)
        out = self.copy()
        out.coefficients = np.matmul(obo, out.coefficients, dtype=complex)
        return out
        
    def apply_sigma(self, particle_index: int, dimension: int) -> 'HilbertOperator':
        """ Applies a one-body sigma spin operator.

        :param particle_index: Index of particle, staring from 0.
        :type particle_index: int
        :param dimension: Dimension of sigma operator: 0, 1, 2 = x, y, z
        :type dimension: int
        :return: The resulting ``HilbertOperator``.
        :rtype: HilbertOperator
        """        
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=pauli(dimension), isospin_matrix=np.identity(2, dtype=complex) )

    def apply_tau(self, particle_index: int, dimension: int):
        """ Applies a one-body tau isospin operator.

        :param particle_index: Index of particle, staring from 0.
        :type particle_index: int
        :param dimension: Dimension of tau operator: 0, 1, 2 = x, y, z
        :type dimension: int
        :return: The resulting ``HilbertOperator``.
        :rtype: HilbertOperator
        """        
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=np.identity(2, dtype=complex), isospin_matrix=pauli(dimension) )
                
    def exp(self) -> 'HilbertOperator':
        """Computes the exponential by Pade approximant.

        :return: Exponentiated operator.
        :rtype: HilbertOperator
        """
        out = self.copy()
        out.coefficients = expm(out.coefficients)
        return out

    def zero(self) -> 'HilbertOperator':
        """Multiplies by zero.

        :return: A copy of the ``HilbertOperator`` with all zero coefficients.
        :rtype: HilbertOperator
        """
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out
        
    def dagger(self) -> 'HilbertOperator':
        """Hermitian conjugate.

        :return: The Hermitian conjugate of the original ``HilbertOperator``.
        :rtype: HilbertOperator
        """
        out = self.copy()
        out.coefficients = self.coefficients.conj().T
        return out

    def __mul__(self, other):
        """Defines the ``*`` multiplication operator to be used in place of  ``.multiply_state``, ``.multiply_operator``, and ``.scale``.
        """ 
        if np.isscalar(other): # scalar * op
            return self.copy().scale(other)
        elif isinstance(other, HilbertState):
            if other.ketwise: # op |s>
                return self.multiply_state(other)
            else:
                raise ValueError("Improper multiplication.")
        elif isinstance(other, HilbertOperator):
            return self.multiply_operator(other)
        else:
            raise NotImplementedError("Unsupported multiply.")
        
    def trace(self) -> complex:
        """Compute the trace of this operator

        :return: the trace
        :rtype: complex
        """
        return np.trace(self.coefficients)


# ONE-BODY BASIS CLASSES
        
# numbaspec_productstate = [
#     ('n_particles', int32),
#     ('isospin', boolean),
#     ('n_basis', int8),
#     ('dimension', int32),
#     ('ketwise', boolean),
#     ('coefficients', float64[:,:])
# ]
# @jitclass(numbaspec_productstate)
class ProductState:
    """A spin state in the "Product" basis, a single tensor product of one-body vectors.
    
    States must be defined with a number of particles. 
    If ``isospin`` is False, then the one-body basis is only spin up/down. If True, then it is (spin up/down x isospin up/down).
    ``ketwise`` detemines if it is a bra or a ket.
     
    Tensor product states do not form a proper vector space (e.g. the sum of two is not guaranteed to be a tensor product)
    so methods with ``ProductState`` are restricted. Namely operations + and - do not exist.
    
    The coefficients of the ``ProductState`` are kept in the one-body form and can be projected to the Hilbert basis using the ``to_full_basis`` method.
    """  
    def __init__(self, n_particles: int, coefficients=None, ketwise=True, isospin=True):
        """Instances a new ``ProductState``

        :param n_particles: number of particles
        :type n_particles: int
        :param coefficients: an optional array of coefficients, defaults to None
        :type coefficients: numpy.ndarray, optional
        :param ketwise: True for column vector, False for row vector, defaults to True
        :type ketwise: bool, optional
        :param isospin: True for spin-isospin state, False for spin only, defaults to True
        :type isospin: bool, optional
        :raises ValueError: inconsistency in chosen options
        """   
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.ketwise = ketwise
        
        if coefficients is None:
            if ketwise:
                self.coefficients = np.zeros(shape=(self.n_particles, self.n_basis, 1))
            else:
                self.coefficients = np.zeros(shape=(self.n_particles, 1, self.n_basis))
        else:
            assert type(coefficients)==np.ndarray
            ket_condition = (coefficients.shape == (n_particles, self.n_basis, 1)) and ketwise
            bra_condition = (coefficients.shape == (n_particles, 1, self.n_basis)) and not ketwise
            if not ket_condition and not bra_condition:
                raise ValueError("Inconsistent initialization of state vector. \n\
                                Did you get the shape right?")
            else:
                self.coefficients = coefficients.astype('complex')

    def copy(self):
        """Copies the ``ProductState``.

        :return: a new instance of ``ProductState`` with all the same properties.
        :rtype: ProductState
        """      
        return ProductState(n_particles=self.n_particles, coefficients=self.coefficients.copy(), ketwise=self.ketwise, isospin=self.isospin)

    def to_list(self) -> list:
        """
        :return: A list of one-body vectors
        :rtype: list[numpy.ndarray]
        """
        return [self.coefficients[i] for i in range(self.n_particles)]

    def inner(self, other: 'ProductState') -> complex:
        """Inner product of two ProductState instances. Orientations must be correct.

        :param other: The ket of the inner product.
        :type other: ProductState
        :return: inner product of self (bra) with other (ket)
        :rtype: complex
        """    
        assert isinstance(other, ProductState)
        assert (not self.ketwise) and other.ketwise
        return np.prod([np.dot(self.coefficients[i], other.coefficients[i]) for i in range(self.n_particles)])
        
    def outer(self, other: 'ProductState') -> 'ProductState':
        """Outer product of two ProductState instances, producting a ProductOperator instance. Orientations must be correct.

        :param other: bra part of the outer product
        :type other: ProductState
        :return: Outer product of self (ket) with other (bra)
        :rtype: ProductOperator
        """    
        assert isinstance(other, ProductState)
        assert (self.ketwise) and (not other.ketwise)
        out = ProductOperator(n_particles=self.n_particles, isospin=self.isospin)
        for i in range(self.n_particles):
            out.coefficients[i] = np.matmul(self.coefficients[i], other.coefficients[i], dtype=complex)
        return out

    def multiply_operator(self, other: 'ProductOperator') -> 'ProductState':
        """Multiplies a (bra) ``ProductState`` on a ``ProductOperator``.

        :param other: The operator to multiply onto
        :type other: ProductOperator
        :return: < self | O(other) 
        :rtype: ProductState
        """        
        assert isinstance(other, ProductOperator)
        assert not self.ketwise
        out = self.copy()
        out.coefficients = np.matmul(self.coefficients, other.coefficients, dtype='complex') 
        return out
    
    def dagger(self) -> 'ProductState':
        """Hermitian conjugate.

        :return: The dual ``ProductState``
        :rtype: ProductState
        """  
        out = self.copy()
        out.coefficients = np.transpose(self.coefficients, axes=(0,2,1)).conj()
        out.ketwise = not self.ketwise
        return out

    def __str__(self):
        orient = "ket"
        if not self.ketwise:
            orient = "bra"
        out = f"{self.__class__.__name__} {orient} of {self.n_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{orient} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_full_basis(self) -> 'HilbertState':
        """Projects to the many-body basis.
        
        :return: The Kronecker product of the ``ProductState``. 
        :rtype: HilbertState
        """
        new_coeffs = kronecker_product(self.to_list())
        if self.ketwise:
            new_coeffs = new_coeffs.reshape(self.n_basis ** self.n_particles, 1)
        else:
            new_coeffs = new_coeffs.reshape(1, self.n_basis ** self.n_particles)
        return HilbertState(n_particles=self.n_particles, coefficients=new_coeffs, ketwise=self.ketwise, isospin=self.isospin)

    def normalize(self) -> 'ProductState':
        """Normalize so that the inner product of the state with itself is 1.

        :return: The normalized state.
        :rtype: ProductState
        """
        out = self.copy()
        for i in range(out.n_particles):
            n = np.linalg.norm(out.coefficients[i])
            out.coefficients[i] /= n
        return out

    def scale_one(self, particle_index: int, b: complex) -> 'ProductState':
        """Multiplies a single particle vector by a number.

        :param particle_index: Index of particle, starting from 0.
        :type particle_index: int
        :param b: Scalar
        :type b: complex
        :return: A copy of the ``ProductState`` with the one particle scaled.
        :rtype: ProductState
        """
        assert np.isscalar(b)
        out = self.copy()
        out.coefficients[particle_index] *= b
        return out

    def scale_all(self, b: complex) -> 'ProductState':
        """Scales an A-body state by ``b`` by multiplying each one-body vector by the Ath root of ``b``.

        :param b: scalar
        :type b: complex
        :return: The scaled state
        :rtype: ProductState
        """
        assert np.isscalar(b)
        out = self.copy()
        out.coefficients *= b ** (1 / out.n_particles)
        return out

    def randomize(self, seed:int=None) -> 'ProductState':
        """Randomize coefficients.

        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :return: A copy of the ``ProductState`` with random complex coefficients, normalized.
        :rtype: ProductState
        """        
        rng = np.random.default_rng(seed=seed)
        out = self.copy()
        out.coefficients = rng.standard_normal(size=out.coefficients.shape) + 1.j*rng.standard_normal(size=out.coefficients.shape)
        for i in range(out.n_particles):
            out.coefficients[i] /= np.linalg.norm(out.coefficients[i])
        return out

    def zero(self) -> 'ProductState':
        """Set all coefficients to zero.

        :return: A copy of ``ProductState`` with all coefficients set to zero.
        :rtype: ProductState
        """        
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out

    def generate_basis_states(self) -> list['ProductState']:
        """Makes a list of corresponding basis vectors.

        :return: A list of tensor product states that span the Hilbert space.
        :rtype: list[ProductState]
        """       
        coeffs_list = [np.concatenate(x).reshape(self.n_particles,self.n_basis,1) for x in itertools.product(list(np.eye(self.n_basis)), repeat=self.n_particles)]
        out = [ProductState(self.n_particles, coefficients=c, ketwise=True, isospsin=self.isospin) for c in coeffs_list]
        return out

    def __mul__(self, other):
        """Defines the ``*`` multiplication operator to be used in place of ``.inner`` , ``.outer``, ``.multiply_operator``, and ``.scale``.
        
        """    
        if np.isscalar(other): # scalar * |s>
            return self.copy().scale_all(other)
        elif isinstance(other, ProductState):
            if self.ketwise and not other.ketwise: # |s><s'|
                return self.outer(other)
            elif not self.ketwise and other.ketwise: # <s|s'>
                return self.inner(other)
            else:
                raise ValueError("Improper multiplication.")
        elif isinstance(other, ProductOperator):
            if not self.ketwise:
                return self.multiply_operator(other)
        else:
            raise NotImplementedError("Unsupported multiply.")
        
    def attach_coordinates(self, coordinates: np.ndarray):
        """Adds a new ``.coordinates`` attribute to the ``ProductState``

        :param coordinates: A Numpy array with shape ``(n_particles , 3)`` (e.g. x, y, z)
        :type coordinates: np.ndarray
        """   
        assert coordinates.shape == (self.n_particles, 3)
        self.coordinates = coordinates
        
    def exchange(self, i:int, j:int) -> 'ProductState':
        """Returns a copy of the state with particles i and j swapped.

        :param i: Index of particle one
        :type i: int
        :param j: Index of particle two
        :type j: int
        :return: A copy of the state with particles i and j swapped
        :rtype: ProductState
        """        
        out = self.copy()
        temp = self.coefficients[j].copy()
        out.coefficients[j] = self.coefficients[i].copy()
        out.coefficients[i] = temp
        return out

# numbaspec_productoperator = [
#     ('n_particles', int32),
#     ('isospin', boolean),
#     ('n_basis', int8),
#     ('dimension', int32),
#     ('coefficients', float64[:,:])
# ]
# @jitclass(numbaspec_productoperator)
class ProductOperator:
    """An operator that is a tensor product of one-body operators.
    
    As with ``ProductState`` instances, ``ProductOperator`` instances cannot be added or subtracted.
    """
    def __init__(self, n_particles: int, isospin=True):
        """Instantiates a ``ProductOperator``.

        :param n_particles: Number of particles
        :type n_particles: int
        :param isospin: True for spin-isospin states, False for spin only, defaults to True
        :type isospin: bool, optional
        """
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.coefficients = np.stack(self.n_particles*[np.identity(self.n_basis)], dtype=complex)

    def copy(self) -> 'ProductOperator':
        """Copies the ``ProductOperator``.

        :return: a new instance of ``ProductOperator`` with all the same properties as self.
        :rtype: ProductOperator
        """ 
        out = ProductOperator(n_particles=self.n_particles, isospin=self.isospin)
        for i in range(self.n_particles):
            out.coefficients[i] = self.coefficients[i]
        return out

    def to_list(self) -> list[np.ndarray]:
        """
        :return: A list of one-body operator matrices
        :rtype: list[numpy.ndarray]
        """
        return [self.coefficients[i] for i in range(self.n_particles)]
        
    def multiply_state(self, other: ProductState) -> ProductState:
        """Apply the operator to a ``ProductState`` ket.

        :param other: The state, ketwise.
        :type other: ProductState
        :return: The new state, ketwise.
        :rtype: ProductState
        """                    
        assert isinstance(other, ProductState)
        out = other.copy()
        for i in range(self.n_particles):
                out.coefficients[i] = np.matmul(self.coefficients[i], out.coefficients[i], dtype=complex)
        return out
        
    def multiply_operator(self, other: 'ProductOperator') -> 'ProductOperator':
        """Multiply two ``ProductOperator`` instances together to get a new one.

        :param other: The other ``ProductOperator``
        :type other: ProductOperator
        :return: The product of the two.
        :rtype: ProductOperator
        """
        assert isinstance(other, type(self))
        out = other.copy()
        for i in range(self.n_particles):
                out.coefficients[i] = np.matmul(self.coefficients[i], out.coefficients[i], dtype=complex)
        return out

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        for i, op in enumerate(self.to_list()):
            re = str(np.real(op))
            im = str(np.imag(op))
            out += f"Op {i} Re:\n" + re + f"\nOp {i} Im:\n" + im + "\n"
        return out

    def to_full_basis(self) -> 'HilbertOperator':
        """Projects to the many-body basis.
        
        :return: The Kronecker product of the ``ProductOperator``. 
        :rtype: HilbertOperator
        """
        new_coeffs = kronecker_product(self.to_list())
        new_coeffs = new_coeffs.reshape(self.n_basis ** self.n_particles, self.n_basis ** self.n_particles)
        return HilbertOperator(n_particles=self.n_particles, coefficients=new_coeffs, isospin=self.isospin)
    
    def apply_onebody_operator(self, particle_index: int, spin_matrix: np.ndarray, isospin_matrix:np.ndarray=None) -> 'ProductOperator':
        r"""Applies a one-body / single-particle operator to the ``ProductOperator``.
        This accounts for the spin-isospin kronecker product, if isospin is used.

        .. math::
            O' = \sigma_{\alpha i} \tau_{\beta i} O

        :param particle_index: Index of particle to apply onebody operator to, starting from 0.
        :type particle_index: int
        :param spin_matrix: The spin part of the operator, a 2x2 matrix
        :type spin_matrix: np.ndarray
        :param isospin_matrix: The isospin part of the operator, a 2x2 matrix, defaults to None
        :type isospin_matrix: numpy.ndarray, optional
        :return: A copy of the ``ProductOperator`` with the one-body operator applied.
        :rtype: ProductOperator
        """
        if self.isospin:
            if isospin_matrix is None:
                isospin_matrix = np.identity(2, dtype=complex)
            onebody_matrix = kronecker_product([isospin_matrix, spin_matrix])
        else:
            onebody_matrix = spin_matrix
        out = self.copy()
        out.coefficients[particle_index] = np.matmul(onebody_matrix, out.coefficients[particle_index], dtype=complex)
        return out

    def apply_sigma(self, particle_index: int, dimension: int) -> 'ProductOperator':
        """ Applies a one-body sigma spin operator.

        :param particle_index: Index of particle, staring from 0.
        :type particle_index: int
        :param dimension: Dimension of sigma operator: 0, 1, 2 = x, y, z
        :type dimension: int
        :return: The resulting ``ProductOperator``.
        :rtype: ProductOperator
        """
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))

    def apply_tau(self, particle_index: int, dimension: int) -> 'ProductOperator':
        """ Applies a one-body tau isospin operator.

        :param particle_index: Index of particle, staring from 0.
        :type particle_index: int
        :param dimension: Dimension of tau operator: 0, 1, 2 = x, y, z
        :type dimension: int
        :return: The resulting ``ProductOperator``.
        :rtype: ProductOperator
        """        
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))
        
    def exp_onebody(self, particle_index: int) -> 'ProductOperator':
        """ Applies an exponential function to the corresponding one-body operator.

        :param particle_index: Index of particle, staring from 0.
        :type particle_index: int
        :return: The resulting operator.
        :rtype: ProductOperator
        """
        out = self.copy()
        out[particle_index] = expm(out[particle_index])
        return out

    def scale_one(self, particle_index: int, b: complex) -> 'ProductOperator':
        """Multiplies a single particle operator matrix by a number.

        :param particle_index: Index of particle, starting from 0.
        :type particle_index: int
        :param b: Scalar
        :type b: complex
        :return: A copy of the ``ProductOperator`` with the one-body matrix scaled.
        :rtype: ProductOperator
        """
        assert np.isscalar(b)
        out = self.copy()
        out.coefficients[particle_index] *= b
        return out
        
    def scale_all(self, b) -> 'ProductOperator':
        """Scales an A-body operator by ``b`` by multiplying each one-body matrix by the Ath root of ``b``.

        :param b: scalar
        :type b: complex
        :return: The scaled state
        :rtype: ProductOperator
        """
        assert np.isscalar(b)
        out = self.copy()
        out.coefficients *= b ** (1 / out.n_particles)
        return out

    def zero(self) -> 'ProductOperator':
        """Set all coefficients to zero.

        :return: A copy of ``ProductOperator`` with all coefficients set to zero.
        :rtype: ProductOperator
        """  
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out

    def dagger(self):
        """Hermitian conjugate.

        :return: The dual ``ProductOperator``
        :rtype: ProductOperator
        """  
        out = self.copy()
        out.coefficients = np.transpose(self.coefficients, axes=(0,2,1)).conj()
        return out        
        
    def to_full_basis(self):
        """Projects to the many-body basis.
        
        :return: The Kronecker product of the ``ProductOperator``. 
        :rtype: HilbertOperator
        """
        new_coeffs = kronecker_product(self.to_list())
        out = HilbertOperator(n_particles=self.n_particles,isospin=self.isospin)
        out.coefficients = new_coeffs
        return out


    
    def __mul__(self, other):
        """Defines the ``*`` multiplication operator to be used in place of  ``.multiply_state``, ``.multiply_operator``, and ``.scale``.
        """ 
        if np.isscalar(other): # scalar * op
            return self.copy().scale_all(other)
        elif isinstance(other, ProductState):
            if other.ketwise: # op |s>
                return self.multiply_state(other)
            else:
                raise ValueError("Improper multiplication.")
        elif isinstance(other, ProductOperator):
            return self.multiply_operator(other)
        else:
            raise NotImplementedError("Unsupported multiply.")

    

# COUPLINGS / POTENTIALS

class CouplingArray:
    r"""Base class for coupling arrays.
    
    Set and get are defined like numpy.ndarray objects.
    
    The simplest example would be something like :math:`g_{\alpha i}` for :math:`\alpha=x,y,z` and :math:`i=0 \dots A-1`.
    
    .. code-block:: python
    
        A = 2
        g = Coupling(n_particles=A, shape=(3,A))    #initialize to zeros
        g[0,0] = 1.0  # set an entry by hand 
    
    """    
    def __init__(self, n_particles: int, shape: tuple[int], file=None):
        r"""Base class for coupling arrays.

        :param n_particles: number of particles
        :type n_particles: int
        :param shape: the dimensions of the array, e.g. :math:`A^\sigma_{\alpha i \beta j}` has shape (3,A,3,A)
        :type shape: tuple[int]
        :param file: name of text file to read array from, defaults to None
        :type file: str, optional
        """        
        self.n_particles = n_particles
        self.shape = shape
        self.coefficients = np.zeros(shape=self.shape)
        if file is not None:
            self.read(file)

    def copy(self):
        assert isinstance(self.coefficients, np.ndarray)
        out = CouplingArray(self.n_particles, self.shape)
        out.coefficients = self.coefficients.copy()
        return out

    def __mul__(self, other):
        """Scalar multiplication.

        :param other: _description_
        :type other: _type_
        :return: _description_
        :rtype: _type_
        """        
        assert np.isscalar(other)
        out = self.copy()
        out.coefficients = other * out.coefficients
        return out
    
    def __rmul__(self,other):
        return self.__mul__(other)

    def read(self, filename):
        if self.shape is None:
            raise ValueError("Must define self.shape before reading from file.")
        self.coefficients = read_from_file(filename, shape=self.shape)

    def __getitem__(self, key):
        return self.coefficients[key]

    def __setitem__(self, key, value):
        self.coefficients[key] = value

    def __str__(self):
        return str(self.coefficients)
        
    def generate_random(self,shape,scale=1.0,mean=0.0,seed=0):
        rng = np.random.default_rng(seed=seed)
        coefficients = scale*rng.standard_normal(size=shape)
        coefficients += mean*np.ones_like(coefficients)
        return coefficients
        
        
# I wrote classes for the sigma, sigma-tau, tau, coulomb, spin-orbit coupling arrays, but these are not strictly required for the other parts of spinbox.
# These exist for convenience, and mainly so we can easily generate random couplings with good symmetries (e.g. i=j entries are zero).
# Also, the NuclearPotential class provides a container for "Argonne-like" nuclear interactions, containing an instance of each of these as attributes.  
# In general though, one can instantiate a Coupling, set the shape, then either set the coefficients using an array or use the read() method to read from file. 
# That should be sufficient to use the Propagator and Integrator classes.

class TwoBodyCoupling(CouplingArray):
    r"""The coupling matrix :math:`A_{\alpha i \beta j}`
    
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)

    def symmetrize(self):
        for i in range(self.n_particles):
            self.coefficients[:,i,:,i] = 0.0
            for j in range(self.n_particles):
                for a in range(3):
                    for b in range(3):
                        self.coefficients[a,i,b,j]=self.coefficients[a,j,b,i]
    
    def random(self, scale, mean, seed=0):
        self.coefficients = self.generate_random(self.shape, scale, mean, seed)
        self.symmetrize()
        return self
    

class TwoBodyCouplingIsotropic(CouplingArray):
    """container class for couplings A(i,j) that have no dimensionful dependence
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, file=None):
        shape = (n_particles, n_particles)
        super().__init__(n_particles, shape, file)
        
    def symmetrize(self):
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                self.coefficients[i,j]=self.coefficients[j,i]

    def random(self, scale, mean, seed=0):
        self.coefficients = self.generate_random(self.shape, scale, mean, seed)
        self.symmetrize()
        return self
    

class OneBodyCoupling(CouplingArray):
    """container class for couplings A(a,i)
    for i = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles)
        super().__init__(n_particles, shape, file)
        
    def random(self, scale, mean, seed=0):
        self.coefficients = self.generate_random(self.shape, scale, mean, seed)
        return self
    

class ThreeBodyCoupling(CouplingArray):
    """container class for couplings A(a,i,b,j,c,k)
    for i, j, k = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles, 3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        
    def symmetrize(self):
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                self.coefficients[a,i,b,i,c,i]=0.0
                                self.coefficients[a,i,b,i,c,k]=0.0
                                self.coefficients[a,i,b,j,c,i]=0.0
                                self.coefficients[a,i,b,j,c,j]=0.0
                                self.coefficients[a,i,b,j,c,k]=self.coefficients[a,i,b,k,c,j]
                                self.coefficients[a,i,b,j,c,k]=self.coefficients[a,j,b,i,c,k]
                                self.coefficients[a,i,b,j,c,k]=self.coefficients[a,j,b,k,c,i]
                                self.coefficients[a,i,b,j,c,k]=self.coefficients[a,k,b,i,c,j]
                                self.coefficients[a,i,b,j,c,k]=self.coefficients[a,k,b,j,c,i]

    def random(self, scale, mean, seed=0):
        self.coefficients = self.generate_random(self.shape, scale, mean, seed)
        self.symmetrize()
        return self


class NuclearPotential:
    """container class for Argonne-style NN potential + NNN
    """
    def __init__(self, n_particles):
        self.n_particles = n_particles
        self.sigma = TwoBodyCoupling(n_particles)
        self.sigmatau = TwoBodyCoupling(n_particles)
        self.tau = TwoBodyCouplingIsotropic(n_particles)
        self.coulomb = TwoBodyCouplingIsotropic(n_particles)
        self.spinorbit = OneBodyCoupling(n_particles)
        self.sigma_3b = ThreeBodyCoupling(n_particles)
    
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

    def read_sigma_3b(self, filename):
        self.sigma_3b.read(filename)



# PROPAGATOR CLASSES

class Propagator:
    def __init__(self, n_particles, dt: float, isospin=True, include_prefactors=True):
        self.n_particles = n_particles
        self.dt = dt
        self.deltatau = dt * 1j
        self.symmetry_factor = 0.5
        self.include_prefactors = include_prefactors
        self.isospin = isospin
        self._xyz = [0, 1, 2]
        self._1b_idx = interaction_indices(n_particles, 1)
        self._2b_idx = interaction_indices(n_particles, 2)
        self._3b_idx = interaction_indices(n_particles, 3)
        self._n2 = len(self._2b_idx)
        self._n3 = len(self._3b_idx)
        
        # if not np.imag(self.dt)==0.0:
        #     raise ValueError('Looks like you entered a complex value for dt, which should be real: delta tau = i * dt ')



class ExactPropagator(Propagator):
    r"""The "exact" propagator, calculated in the full Hilbert space.

    .. math::
        \exp \left( - \sum_n  g_n \hat{v}_n  \right)

    where :math:`g_n` is the entire scalar factor (e.g. :math:`\frac{\delta\tau}{2} A^{\sigma}_{i \alpha j \beta}`, note the phase convention)
    and :math:`\hat{v}_n`
    is the 2- or 3-body interaction operator.    

    Note, this calculation must be done in the complete many-body basis; it cannot be restricted to product states.
    
    We use a Pade approximant for the matrix exponential. 
    The LS term can be represented using a linear approximation or the factorization procedure described in Stefano's thesis.
    
    :return: The exact propagator.
    :rtype: HilbertOperator
    """ 
    def __init__(self, n_particles, dt:float, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self.n_particles = n_particles
        self.isospin = isospin
        self._ident = HilbertOperator(n_particles, self.isospin)
        self._sig = [[HilbertOperator(n_particles, self.isospin).apply_sigma(i,a) for a in [0, 1, 2, 3]] for i in range(n_particles)]
        self._tau = [[HilbertOperator(n_particles, self.isospin).apply_tau(i,a) for a in [0, 1, 2, 3]] for i in range(n_particles)]
        self._linear_spinorbit = False # secret parameter to use the linear approximation of LS instead of the factorization

    def force_sigma(self, coupling_array: CouplingArray, i: int, j: int) -> HilbertOperator:
        r"""The two-body sigma force :math:`A^\sigma_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}` .

        :param coupling_array: force coupling array (e.g. :math:`A^{\sigma}_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param i: index of particle i
        :type i: int
        :param j: index of pareticle j
        :type j: int
        :return: The two-body force operator
        :rtype: HilbertOperator
        """        
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                out += self._sig[i][a].multiply_operator(self._sig[j][b]).scale(coupling_array[a, i, b, j])
        return out

    def force_sigmatau(self, coupling_array: CouplingArray, i: int, j: int) -> HilbertOperator:
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    op = HilbertOperator(self.n_particles, self.isospin)
                    op = op.multiply_operator(self._sig[i][a]).multiply_operator(self._tau[i][c])
                    op = op.multiply_operator(self._sig[j][b]).multiply_operator(self._tau[j][c])
                    out += op.scale(coupling_array[a, i, b, j])
        return out

    def force_tau(self, coupling_array:CouplingArray, i:int, j:int) -> HilbertOperator:
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for c in range(3):
            out += self._tau[i][c].multiply_operator(self._tau[j][c]).scale(coupling_array[i, j])
        return out
    
    def force_coulomb(self, coupling_array: CouplingArray, i:int, j:int) -> HilbertOperator:
        out = self._ident + self._tau[i][2] + self._tau[j][2] + self._tau[i][2].multiply_operator(self._tau[j][2])
        out = out.scale(coupling_array[i, j])
        return out

    def force_coulomb_onebody(self, coupling: complex, i: int) -> HilbertOperator:
        """just the one-body part of the expanded coulomb propagator
        for use along with auxiliary field propagators"""
        out =  self._tau[i][2].scale(coupling)
        return out

    def propagator_spinorbit_linear(self, coupling_array:CouplingArray, i: int) -> HilbertOperator:
        # linear approx to LS
        out = HilbertOperator(self.n_particles)
        for a in range(3):
            out = (self._ident - self._sig[i][a].scale(1.j * coupling_array[a, i])).multiply_operator(out) 
        return out
    
    def propagator_spinorbit_onebody(self, coupling_array: CouplingArray, i:int) -> HilbertOperator:
        # one-body part of the LS factorization
        out = HilbertOperator(self.n_particles).zero()
        for a in range(3):
            out += self._sig[i][a].scale(- 1.j * coupling_array[a,i])
        return out.exp()
    
    def propagator_spinorbit_twobody(self, coupling_array: CouplingArray, i:int, j:int) -> HilbertOperator:
        # two-body part of the LS factorization
        out = HilbertOperator(self.n_particles).zero()
        for a in range(3):
            for b in range(3):
                out += self._sig[i][a].multiply_operator(self._sig[j][b]).scale(0.5 * coupling_array[a,i] * coupling_array[b,j])
        return out.exp()

    def force_sigma_3b(self, coupling_array:CouplingArray, i:int, j:int, k:int) -> HilbertOperator:
        # 3-body sigma
        out = HilbertOperator(self.n_particles).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out += self._sig[i][a].multiply_operator(self._sig[j][b]).multiply_operator(self._sig[k][c]).scale(-coupling_array[a, i, b, j, c, k])
        return out

    def propagator_combined(self,
                            potential:NuclearPotential,
                            sigma=False,
                            sigmatau=False,
                            tau=False,
                            coulomb=False,
                            spinorbit=False,
                            sigma_3b=False) -> HilbertOperator: 
        
        pairs_ij = interaction_indices(self.n_particles)
        triples_ijk = interaction_indices(self.n_particles, 3)

        force = HilbertOperator(self.n_particles).zero()
        if sigma:
            for i,j in pairs_ij:
                force += self.force_sigma(potential.sigma, i, j)
        if sigmatau:
            for i,j in pairs_ij:
                force += self.force_sigmatau(potential.sigmatau, i, j)
        if tau:
            for i,j in pairs_ij:
                force += self.force_tau(potential.tau, i, j)
        if coulomb:
            for i,j in pairs_ij:
                force += self.force_coulomb(potential.coulomb, i, j)
        if sigma_3b:
            for i,j,k in triples_ijk:
                force += self.force_sigma_3b(potential.sigma_3b, i, j, k)
        
        prop = force.scale(- self.deltatau * self.symmetry_factor).exp()

        if spinorbit:
            if self._linear_spinorbit:
                for i in range(self.n_particles):
                    prop = self.propagator_spinorbit_linear(potential.spinorbit, i).multiply_operator(prop)
            else:
                for i in range(self.n_particles):
                    prop = self.propagator_spinorbit_onebody(potential.spinorbit, i).multiply_operator(prop)
                    for j in range(self.n_particles):
                        prop = self.propagator_spinorbit_twobody(potential.spinorbit, i, j).multiply_operator(prop)
        
        return prop
    
    
    
class HilbertPropagatorHS(Propagator):
    r""" The two-body propagator applied by Hubbard-Stratonovich transform in the full Hilbert basis
    
    .. math::
            e^{-z \hat{o}_i \hat{o}_j} = e^z\int dx \left[ \frac{1}{\sqrt{2\pi}}e^{-x^2/2} \right] e^{x \sqrt{-z} (\hat{o}_i + \hat{o}_j)}
        
        
    """
    def __init__(self, n_particles:int, dt:float, isospin=True, include_prefactors=True):
        """The two-body propagator applied by Hubbard-Stratonovich

        :param n_particles: number of particles
        :type n_particles: int
        :param dt: time step (real)
        :type dt: float
        :param isospin: if True include isospin, defaults to True
        :type isospin: bool, optional
        :param include_prefactors: whether to include the constant scaling part of the propagator (e.g. if done in another part of the calculation), defaults to True
        :type include_prefactors: bool, optional
        """        
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = HilbertOperator(self.n_particles, isospin=isospin)
        self._sig_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2, 3]] for i in range(self.n_particles)]
        self._tau_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_tau(i,a) for a in [0, 1, 2, 3]] for i in range(self.n_particles)]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = self._n2
        self.n_aux_spinorbit = 9 * self._n2
        
    def onebody(self, coupling: float, operator: HilbertOperator) -> HilbertOperator:
        r"""One-body propagator 
        
        .. math::
            \exp \left[ - z \hat{o} \right]

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`

        :param coupling: scalar
        :type coupling: float
        :param operator: one-body operator
        :type operator: HilbertOperator
        :return: The one-body propagator
        :rtype: HilbertOperator
        """
        z = self.deltatau * self.symmetry_factor * coupling
        return operator.scale(-z).exp()        

    def twobody_sample(self, coupling:float, x: float, operator_i: HilbertOperator, operator_j: HilbertOperator) -> HilbertOperator:
        r"""A sample of the two-body propagator in the integrand of the Hubbard-Stratonovich transform.
        
        .. math::
            \exp(z) \exp ({x \sqrt{-z} \hat{\sigma}_{i \alpha} }) \exp({x \sqrt{-z} \hat{\sigma}_{j \beta} } )

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`

        :param coupling: scalar
        :type coupling: float
        :param x: auxiliary field value
        :type x: float
        :param operator_i: operator on particle i
        :type operator_i: HilbertOperator
        :param operator_j: operator on particle j
        :type operator_j: HilbertOperator
        :return: One sample of the two-body propagator
        :rtype: HilbertOperator
        """        
        z = self.deltatau * self.symmetry_factor * coupling
        arg = csqrt(-z)*x
        if self.include_prefactors:
            prefactor = cexp(z)
        else:
            prefactor = 1.0
        gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
        gj = self._ident.scale(ccosh(arg)) + operator_j.scale(csinh(arg))
        return gi.multiply_operator(gj).scale(prefactor)
                
    def factors_sigma(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                        idx += 1
        return out

    def factors_sigmatau(self, coupling_array: CouplingArray,  aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`A^{\sigma\tau}_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\sigma\tau}_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*27
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """    
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        for c in self._xyz:
                            opi = self._sig_op[i][a].multiply_operator(self._tau_op[i][c])
                            opj = self._sig_op[j][b].multiply_operator(self._tau_op[j][c])
                            out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], opi, opj) )
                            idx += 1
        return out
    
    def factors_tau(self, coupling_array: CouplingArray, aux: list):
        r"""Creates factors of the :math:`\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\tau}_{ij}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*3
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                if not coupling_array[i,j]==0.:
                    out.append( self.twobody_sample(coupling_array[i,j], aux[idx], self._tau_op[i][a], self._tau_op[j][a]) )
                    idx += 1
        return out

    def factors_coulomb(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the Coulomb propagator. 
        
        .. math::
            \exp \left[ -\frac{\delta\tau}{2} \frac{v_{ij}}{4} (1+\tau_{iz}+ \tau_{jz} + \tau_{iz}\tau_{jz} ) \right] 
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`v_C(r_{ij})`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j in self._2b_idx:
                if not coupling_array[i,j]==0.:
                    if self.include_prefactors:
                        z = self.deltatau * self.symmetry_factor * 0.25 * coupling_array[i,j]
                        out.append(HilbertOperator(self.n_particles, self.isospin).scale(cexp(-z)))
                    out.append( self.onebody(0.25*coupling_array[i,j], self._tau_op[i][2]) )
                    out.append( self.onebody(0.25*coupling_array[i,j], self._tau_op[j][2]) )
                    out.append( self.twobody_sample(0.25*coupling_array[i,j], aux[idx], self._tau_op[i][2], self._tau_op[j][2]) )
                    idx += 1
        return out
    
    def factors_spinorbit(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the spin-orbit propagator. 
        
        .. math::
            \exp \left[ - \frac{\delta\tau}{2} v_{LS}(r_{ij}) \mathbf{L}\cdot\mathbf{S} \right]
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`g^\text{LS}_{\alpha i}`)
        :type coupling_array: Coupling
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """ 
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                if not coupling_array[a,i]==0.:
                    z = 1.j*coupling_array[a,i] / (self.deltatau * self.symmetry_factor)
                    out.append( self.onebody(z,  self._sig_op[i][a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i]==0.:
                        if not coupling_array[b,j]==0.:
                            z = - 0.5 * coupling_array[a, i] * coupling_array[b, j] / (self.deltatau * self.symmetry_factor)
                            out.append( self.twobody_sample(z, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                            idx += 1
        if self.include_prefactors:
            prefactor = np.exp( 0.5 * np.sum(coupling_array.coefficients**2))
            out.append(  HilbertOperator(self.n_particles, self.isospin).scale(prefactor) )
        return out



class HilbertPropagatorRBM(Propagator):
    r""" The two-body propagator applied by restricted Boltzmann machine in the Hilbert basis.
    
    .. math::
            e^{-z \hat{o}_i \hat{o}_j} = e^{-|z|}  \sum_{h=0,1} e^{ \tanh^{-1}\left( \sqrt{\tanh (|z|)}\right) (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

    where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
    
    """
    def __init__(self, n_particles:int, dt:float, isospin=True, include_prefactors=True):
        r""" The two-body propagator applied by restricted Boltzmann machine in the Hilbert basis.
        
        .. math::
                e^{-z \hat{o}_i \hat{o}_j} = e^{-|z|}  \sum_{h=0,1} e^{ \tanh^{-1}\left( \sqrt{\tanh (|z|)}\right) (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
        
        """
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = HilbertOperator(self.n_particles, isospin=isospin)
        self._sig_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2, 3]] for i in range(self.n_particles)]
        self._tau_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_tau(i,a) for a in [0, 1, 2, 3]] for i in range(self.n_particles)]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = self._n2
        self.n_aux_spinorbit = 9 * self._n2
        self.n_aux_sigma_3b = 27 * 4 * self._n3

    
    def _a2b_factors(self, coupling:float) -> tuple:
        """Computes coefficients for the two-body RBM

        :param a2: scalar
        :type a2: float
        :return: factors N (normalization), W (weight), and S (sign)
        :rtype: tuple
        """
        z = self.deltatau * self.symmetry_factor * coupling
        n = cexp(-cabs(z))
        w = carctanh(csqrt(ctanh(cabs(z))))
        s = coupling/cabs(coupling)
        return n, w, s
    
    def _a3b_factors(self, coupling:float) -> tuple:
        """Computes coefficients for the three-body RBM

        :param a3: scalar
        :type a3: float
        :return: factors N (normalization), C (bias), W (weight), A1 (one-body), A2 (two-body)
        :rtype: tuple
        """
        a3 = self.deltatau * self.symmetry_factor * coupling
        log = lambda x: np.log(x, dtype=complex)
        if coupling>0:
            x = csqrt(cexp(8*a3) - 1)
            x = csqrt( 2*cexp(4*a3)*( cexp(4*a3)*x + cexp(8*a3) - 1  ) - x)
            x = x + cexp(6*a3) + cexp(2*a3)*csqrt(cexp(8*a3) - 1)
            x = x*2*cexp(2*a3) - 1
            c = 0.5*log(x)
            w = c
            a1 = 0.125*( 6*c - log(cexp(4*c) + 1) + log(2) )
            a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
            top = cexp( 5 * c / 4)
            bottom = 2**(3/8) * csqrt(cexp(2*c) + 1) * (cexp(4*c) + 1)**0.125
            n = top/bottom
        else:
            x = csqrt(1 - cexp(8*a3))
            x = csqrt( 2*(x + 1) - cexp(8*a3) * ( x + 2) )
            x = x + 1 + csqrt(1 - cexp(8*a3))
            c = 0.5 * log(2*cexp(-8*a3)*x - 1)
            w = -c
            a1 = 0.125*( log(0.5*(cexp(4*c) + 1)) - 6*c )
            a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
            top = cexp( c / 4)
            bottom = 2**(3/8) * csqrt(cexp(-2*c) + 1) * (cexp(4*c) + 1)**0.125
            n = top/bottom
        return n, c, w, a1, a2

    def onebody(self, coupling: float, operator: HilbertOperator) -> HilbertOperator:
        r"""A one-body propagator. Hilbert basis.
        
        .. math::
            \exp \left[ - z \hat{o} \right]

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
        
        
        :param coupling: scalar
        :type coupling: complex
        :param operator: one-body operator
        :type operator: HilbertOperator
        :return: The one-body propagator
        :rtype: HilbertOperator
        """
        z = self.deltatau*self.symmetry_factor*coupling
        return operator.scale(-z).exp()

    def twobody_sample(self, coupling: float, h: int, operator_i: HilbertOperator, operator_j: HilbertOperator) -> HilbertOperator:
        r"""A sample of the two-body propagator in the integrand of the RBM transform. Hilbert basis.
        
        .. math::
            e^{-|z|} e^{ W (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

        where :math:`\tanh^{-1}\left( \sqrt{\tanh (|\delta\tau}{2} \times \frac{\text{coupling}|)}\right)`

        :param coupling: scalar
        :type coupling: float
        :param h: auxiliary field value
        :type h: float
        :param operator_i: operator on particle i
        :type operator_i: HilbertOperator
        :param operator_j: operator on particle j
        :type operator_j: HilbertOperator
        :return: One sample of the two-body propagator
        :rtype: HilbertOperator
        """ 
        N, W, S = self._a2b_factors(coupling)
        if self.include_prefactors:
            prefactor = N
        else:
            prefactor = 1.0
        arg = W*(2*h-1)
        gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
        gj = self._ident.scale(ccosh(arg)) - operator_j.scale(S * csinh(arg))
        return gi.multiply_operator(gj).scale(prefactor)

    # def threebody_sample_partial(self, z: float, h_list: list, operator_i: HilbertOperator, operator_j: HilbertOperator, operator_k: HilbertOperator):
    #         """three body propagator sample using three 2-body RBMs"""
    #         N, C, W, A1, A2 = self._a3b_factors(z)
    #         if self.include_prefactors:
    #             prefactor = N*cexp(-h_list[0]*C)
    #         else:
    #             prefactor = 1.0
    #         # one-body factors
    #         arg = A1 - h_list[0]*W
    #         gi = operator_i.scale(arg).exp()
    #         gj = operator_j.scale(arg).exp()
    #         gk = operator_k.scale(arg).exp()
    #         out = gk.multiply_operator(gj).multiply_operator(gi).scale(prefactor)
    #         # two-body factors
    #         out = out.multiply_operator(self.twobody_sample(-A2, h_list[1], operator_i, operator_j))
    #         out = out.multiply_operator(self.twobody_sample(-A2, h_list[2], operator_i, operator_k))
    #         out = out.multiply_operator(self.twobody_sample(-A2, h_list[3], operator_j, operator_k))
    #         return out.scale(prefactor)

    def threebody_sample(self, coupling: float, h_list: list, operator_i:HilbertOperator, operator_j:HilbertOperator, operator_k:HilbertOperator) -> HilbertOperator:
        """A sample of the three-body propagator in the integrand of the RBM transform. Hilbert basis.

        :param coupling: scalar
        :type coupling: float
        :param h_list: auxiliary field values 
        :type h_list: list
        :param operator_i: operator on particle i
        :type operator_i: HilbertOperator
        :param operator_j: operator on particle j
        :type operator_j: HilbertOperator
        :param operator_k: operator on particle k
        :type operator_k: HilbertOperator
        :return: The sample of the three-body RBM propagator
        :rtype: HilbertOperator
        """        
        N, C, W, A1, A2 = self._a3b_factors(coupling)
        if self.include_prefactors:
            prefactor = N*2*cexp(-3*cabs(A2)-h_list[0]*C)
        else:
            prefactor = 1.0
        W2 = carctanh(csqrt(ctanh(cabs(A2))))
        S2 = A2/cabs(A2)
        arg_i = W2*(2*h_list[1] - 1) + W2*(2*h_list[2] - 1) + A1 - h_list[0]*W
        arg_j = W2*S2*(2*h_list[1] -1) + W2*(2*h_list[3] - 1) + A1 - h_list[0]*W
        arg_k = W2*S2*(2*h_list[2] - 1) + W2*S2*(2*h_list[3] - 1) + A1 - h_list[0]*W
        out = operator_i.scale(arg_i).exp() * operator_j.scale(arg_j).exp() * operator_k.scale(arg_k).exp()
        return out.scale(prefactor)

    def factors_sigma(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                        idx += 1
        return out

    def factors_sigmatau(self, coupling_array: CouplingArray,  aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`A^{\sigma\tau}_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\sigma\tau}_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*27
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """  
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        for c in self._xyz:
                            opi = self._sig_op[i][a].multiply_operator(self._tau_op[i][c])
                            opj = self._sig_op[j][b].multiply_operator(self._tau_op[j][c])
                            out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], opi, opj) )
                            idx += 1
        return out
    
    def factors_tau(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\tau}_{ij}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*3
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                if not coupling_array[i,j]==0.:
                    out.append( self.twobody_sample(coupling_array[i,j], aux[idx], self._tau_op[i][a], self._tau_op[j][a]) )
                    idx += 1
        return out

    def factors_coulomb(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the Coulomb propagator. 
        
        .. math::
            \exp \left[ -\frac{\delta\tau}{2} \frac{v_{ij}}{4} (1+\tau_{iz}+ \tau_{jz} + \tau_{iz}\tau_{jz} ) \right] 
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`v_C(r_{ij})`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j in self._2b_idx:
            if not coupling_array[i,j]==0.:
                z = self.deltatau * self.symmetry_factor * coupling_array[i,j] * 0.25
                if self.include_prefactors:
                    out.append(HilbertOperator(self.n_particles, self.isospin).scale(cexp(-z)))
                out.append( self.onebody(0.25*coupling_array[i,j], self._tau_op[i][2]) )
                out.append( self.onebody(0.25*coupling_array[i,j], self._tau_op[j][2]) )
                out.append( self.twobody_sample(0.25*coupling_array[i,j], aux[idx], self._tau_op[i][2], self._tau_op[j][2]) )
                idx += 1
        return out
    
    def factors_spinorbit(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the spin-orbit propagator. 
        
        .. math::
            \exp \left[ - \frac{\delta\tau}{2} v_{LS}(r_{ij}) \mathbf{L}\cdot\mathbf{S} \right]
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`g^\text{LS}_{\alpha i}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """ 
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                if not coupling_array[a,i]==0.:
                    z = 1.j*coupling_array[a,i] / (self.deltatau * self.symmetry_factor)
                    out.append( self.onebody(z,  self._sig_op[i][a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i]==0.:
                        if not coupling_array[b,j]==0.:
                            z = - 0.5 * coupling_array[a, i] * coupling_array[b, j] / (self.deltatau * self.symmetry_factor)
                            out.append( self.twobody_sample(z, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                            idx += 1
        if self.include_prefactors:
            prefactor = np.exp( 0.5 * np.sum(coupling_array.coefficients**2))
            out.append( HilbertOperator(self.n_particles, self.isospin).scale(prefactor) )
        return out

    def factors_sigma_3b(self, coupling_array: CouplingArray, aux: list) -> list[HilbertOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j \gamma k} \sigma_{i \alpha} \sigma_{j \beta} \sigma_{k \gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j \gamma k}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*108
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[HilbertOperator]
        """
        out = []
        idx = 0
        for i,j,k in self._3b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        if not coupling_array[a,i,b,j,c,k]==0.:
                            out.append( self.threebody_sample(coupling_array[a,i,b,j,c,k], aux[idx:idx+4], self._sig_op[i][a], self._sig_op[j][b], self._sig_op[k][c]) )
                        idx += 4
        return out


## PRODUCT STATE PROPAGATORS

class ProductPropagatorHS(Propagator):
    r""" The two-body propagator applied by Hubbard-Stratonovich transform in the product state basis.
    
    .. math::
            e^{-z \hat{o}_i \hat{o}_j} = e^z\int dx \left[ \frac{1}{\sqrt{2\pi}}e^{-x^2/2} \right] e^{x \sqrt{-z} (\hat{o}_i + \hat{o}_j)}
        
        
    """

    def __init__(self, n_particles: int, dt: float, isospin=True, include_prefactors=True):
        """The two-body propagator applied by Hubbard-Stratonovich in the product state basis

        :param n_particles: number of particles
        :type n_particles: int
        :param dt: time step (real)
        :type dt: float
        :param isospin: if True include isospin, defaults to True
        :type isospin: bool, optional
        :param include_prefactors: whether to include the constant scaling part of the propagator (e.g. if done in another part of the calculation), defaults to True
        :type include_prefactors: bool, optional
        """  
        super().__init__(n_particles, dt, isospin, include_prefactors)
        if isospin:
            self._ident = np.identity(4)
            self._sig = [kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2, 3]]
            self._tau = [kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2, 3]]
        else:
            self._ident = np.identity(2)
            self._sig = pauli('list')
            self._tau = None
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, coupling: float, i: int, onebody_matrix: np.ndarray) -> ProductOperator:
        r"""One-body propagator 
        
        .. math::
            \exp \left[ - z \hat{o} \right]

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`

        :param coupling: scalar
        :type coupling: float
        :param i: index of particle i
        :type i: int
        :param onebody_matrix: matrix representation of operator
        :type onebody_matrix: np.ndarray
        :return: The one-body product space propagator
        :rtype: ProductOperator
        """        
        out = ProductOperator(self.n_particles, self.isospin)
        z = self.deltatau * self.symmetry_factor * coupling
        out.coefficients[i] = ccosh(z) * out.coefficients[i] - csinh(z) * onebody_matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, coupling: float, x: float, i: int, j: int, onebody_matrix_i: np.ndarray, onebody_matrix_j: np.ndarray):
        r"""A sample of the two-body propagator in the integrand of the Hubbard-Stratonovich transform.
        
        .. math::
            \exp(z) \exp ({x \sqrt{-z} \hat{\sigma}_{i \alpha} }) \exp({x \sqrt{-z} \hat{\sigma}_{j \beta} } )

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`

        :param coupling: scalar
        :type coupling: float
        :param x: auxiliary field value
        :type x: float
        :param i: index of particle i
        :type i: int
        :param j: index of particle j
        :type j: int
        :param onebody_matrix_i: matrix form of operator on particle i
        :type onebody_matrix_i: np.ndarray
        :param onebody_matrix_j: matrix form of operator on particle j
        :type onebody_matrix_j: np.ndarray
        :return: The sample of the two-body product space propagator
        :rtype: ProductOperator
        """        
        z = self.deltatau * self.symmetry_factor * coupling
        arg = csqrt(-z)*x
        if self.include_prefactors:
            prefactor = cexp(z)
        else:
            prefactor = 1.0
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] + csinh(arg) * onebody_matrix_j @ out.coefficients[j]
        out.coefficients[i] *= csqrt(prefactor)
        out.coefficients[j] *= csqrt(prefactor)
        return out

    def factors_sigma(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], i, j, self._sig[a], self._sig[b]) )
                        idx += 1
        return out

    def factors_sigmatau(self, coupling_array: CouplingArray,  aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`A^{\sigma\tau}_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\sigma\tau}_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*27
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        for c in self._xyz:
                            out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], i, j, self._sig[a] @ self._tau[c], self._sig[b] @ self._tau[c]) )
                            idx += 1
        return out
    
    def factors_tau(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^{\tau}_{ij}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*3
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                if not coupling_array[i,j]==0.:
                    out.append( self.twobody_sample(coupling_array[i,j], aux[idx], i, j, self._tau[a], self._tau[a]) )
        return out

    def factors_coulomb(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the Coulomb propagator. 
        
        .. math::
            \exp \left[ -\frac{\delta\tau}{2} \frac{v_{ij}}{4} (1+\tau_{iz}+ \tau_{jz} + \tau_{iz}\tau_{jz} ) \right] 
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`v_C(r_{ij})`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            if not coupling_array[i,j]==0.:
                if self.include_prefactors:
                    norm_op = ProductOperator(self.n_particles, self.isospin)
                    z = self.deltatau * self.symmetry_factor * 0.25 * coupling_array[i,j]
                    norm_op = norm_op.scale_all(cexp(-z))
                    out.append(norm_op)
                out.append( self.onebody(0.25*coupling_array[i,j], i, self._tau[2]) )
                out.append( self.onebody(0.25*coupling_array[i,j], j, self._tau[2]) )
                out.append( self.twobody_sample(0.25*coupling_array[i,j], aux[idx], i, j, self._tau[2], self._tau[2]) )
                idx += 1
        return out
    
    def factors_spinorbit(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the spin-orbit propagator. 
        
        .. math::
            \exp \left[ - \frac{\delta\tau}{2} v_{LS}(r_{ij}) \mathbf{L}\cdot\mathbf{S} \right]
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`g^\text{LS}_{\alpha i}`)
        :type coupling_array: CouplingArray
        :param aux:  values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                if not coupling_array[a,i]==0.:
                    z = 1.j * coupling_array[a,i] / (self.deltatau * self.symmetry_factor)
                    out.append( self.onebody(z, i, self._sig[a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i]==0.:
                        if not coupling_array[b,j]==0.:
                            z = - 0.5 * coupling_array[a, i] * coupling_array[b, j] / (self.deltatau * self.symmetry_factor)
                            out.append( self.twobody_sample(z, aux[idx], i, j, self._sig[a], self._sig[b]) )
                            idx += 1
        if self.include_prefactors:
            norm_op = ProductOperator(self.n_particles, self.isospin)
            norm_op = norm_op.scale_all(np.exp( 0.5 * np.sum(coupling_array.coefficients**2)) )
            out.append( norm_op )
        return out    


class ProductPropagatorRBM(Propagator):
    r""" The two-body propagator applied by restricted Boltzmann machine in the product state basis.
    
    .. math::
            e^{-z \hat{o}_i \hat{o}_j} = e^{-|z|}  \sum_{h=0,1} e^{ \tanh^{-1}\left( \sqrt{\tanh (|z|)}\right) (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

    where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
    """
    def __init__(self, n_particles:int, dt:float, isospin=True, include_prefactors=True):
        r""" The two-body propagator applied by restricted Boltzmann machine in the product state basis.
        
        .. math::
                e^{-z \hat{o}_i \hat{o}_j} = e^{-|z|}  \sum_{h=0,1} e^{ \tanh^{-1}\left( \sqrt{\tanh (|z|)}\right) (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
        """
        super().__init__(n_particles, dt, isospin, include_prefactors)
        if isospin:
            self._ident = np.identity(4)
            self._sig = [kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2, 3]]
            self._tau = [kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2, 3]]
        else:
            self._ident = np.identity(2)
            self._sig = pauli('list')
            self._tau = None
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = self._n2
        self.n_aux_spinorbit = 9 * self._n2
        self.n_aux_sigma_3b = 27 * 4 * self._n3

    def _a2b_factors(self, coupling:float) -> tuple:
        """Computes coefficients for the two-body RBM

        :param a2: scalar
        :type a2: float
        :return: factors N (normalization), W (weight), and S (sign)
        :rtype: tuple
        """
        z = self.deltatau * self.symmetry_factor * coupling
        n = cexp(-cabs(z))
        w = carctanh(csqrt(ctanh(cabs(z))))
        s = coupling/cabs(coupling)
        return n, w, s
    
    def _a3b_factors(self, coupling:float) -> tuple:
        """Computes coefficients for the three-body RBM

        :param a3: scalar
        :type a3: float
        :return: factors N (normalization), C (bias), W (weight), A1 (one-body), A2 (two-body)
        :rtype: tuple
        """
        a3 = self.deltatau * self.symmetry_factor * coupling
        log = lambda x: np.log(x, dtype=complex)
        if coupling>0:
            x = csqrt(cexp(8*a3) - 1)
            x = csqrt( 2*cexp(4*a3)*( cexp(4*a3)*x + cexp(8*a3) - 1  ) - x)
            x = x + cexp(6*a3) + cexp(2*a3)*csqrt(cexp(8*a3) - 1)
            x = x*2*cexp(2*a3) - 1
            c = 0.5*log(x)
            w = c
            a1 = 0.125*( 6*c - log(cexp(4*c) + 1) + log(2) )
            a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
            top = cexp( 5 * c / 4)
            bottom = 2**(3/8) * csqrt(cexp(2*c) + 1) * (cexp(4*c) + 1)**0.125
            n = top/bottom
        else:
            x = csqrt(1 - cexp(8*a3))
            x = csqrt( 2*(x + 1) - cexp(8*a3) * ( x + 2) )
            x = x + 1 + csqrt(1 - cexp(8*a3))
            c = 0.5 * log(2*cexp(-8*a3)*x - 1)
            w = -c
            a1 = 0.125*( log(0.5*(cexp(4*c) + 1)) - 6*c )
            a2 = 0.125*( 2*c - log(cexp(4*c) + 1) + log(2) )
            top = cexp( c / 4)
            bottom = 2**(3/8) * csqrt(cexp(-2*c) + 1) * (cexp(4*c) + 1)**0.125
            n = top/bottom
        return n, c, w, a1, a2

    def onebody(self, coupling: float, i: int, onebody_matrix: np.ndarray) -> ProductOperator:
        r"""A one-body propagator. Hilbert basis.
        
        .. math::
            \exp \left[ - z \hat{o} \right]

        where :math:`z=\frac{\delta\tau}{2} \times \text{coupling}`
        

        :param coupling: scalar
        :type coupling: float
        :param i: index of particle i
        :type i: int
        :param onebody_matrix: matrix form of one-body operator on i
        :type onebody_matrix: np.ndarray
        :return: _description_
        :rtype: ProductOperator
        """        
        z = self.deltatau * self.symmetry_factor * coupling
        out = ProductOperator(self.n_particles)
        out.coefficients[i] = ccosh(z) * out.coefficients[i] - csinh(z) * onebody_matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, coupling: float, h: int, i: int, j: int, onebody_matrix_i: np.ndarray, onebody_matrix_j: np.ndarray) -> ProductOperator:
        r"""A sample of the two-body propagator in the integrand of the RBM transform. Product basis.
        
        .. math::
            e^{-|z|} e^{ W (2h-1)\left(\hat{o}_i - \frac{z}{|z|}\hat{o}_j\right)}

        where :math:`\tanh^{-1}\left( \sqrt{\tanh (|\delta\tau}{2} \times \frac{\text{coupling}|)}\right)`

        :param coupling: scalar
        :type coupling: float
        :param h: auxiliary field value
        :type h: int
        :param i: index of particle i
        :type i: int
        :param j: index of particle j
        :type j: int
        :param onebody_matrix_i: matrix form of one-body operator on particle i
        :type onebody_matrix_i: np.ndarray
        :param onebody_matrix_j: matrix form of one-body operator on particle j
        :type onebody_matrix_j: np.ndarray
        :return: One sample of the two-body propagator
        :rtype: ProductOperator
        """        
        N, W, S = self._a2b_factors(coupling)
        if self.include_prefactors:
            prefactor = N
        else:
            prefactor = 1.0
        arg = W*(2*h-1)
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] - S * csinh(arg) * onebody_matrix_j @ out.coefficients[j]
        out.coefficients[i] *= csqrt(prefactor)
        out.coefficients[j] *= csqrt(prefactor)
        return out

    def threebody_sample(self, coupling: float, h_list: list, i: int, j: int, k: int, onebody_matrix_i: np.ndarray, onebody_matrix_j: np.ndarray, onebody_matrix_k: np.ndarray) -> ProductOperator:
        """A sample of the three-body propagator in the integrand of the RBM transform. Product basis.

        :param coupling: scalar
        :type coupling: float
        :param h_list: auxiliary field values 
        :type h_list: list
        :param i: index of particle i
        :type i: int
        :param j: index of particle j
        :type j: int
        :param k: index of particle k
        :type k: int
        :param onebody_matrix_i: matrix form of one-body operator on particle i
        :type onebody_matrix_i: np.ndarray
        :param onebody_matrix_j: matrix form of one-body operator on particle j
        :type onebody_matrix_j: np.ndarray
        :param onebody_matrix_k: matrix form of one-body operator on particle k
        :type onebody_matrix_k: np.ndarray
        :return: The sample of the three-body RBM propagator
        :rtype: ProductOperator
        """            
        N, C, W, A1, A2 = self._a3b_factors(coupling)
        if self.include_prefactors:
            prefactor = N*2*cexp(-3*cabs(A2)-h_list[0]*C)
        else:
            prefactor = 1.0
        out = ProductOperator(self.n_particles, self.isospin)
        W2 = carctanh(csqrt(ctanh(cabs(A2))))
        S2 = A2/cabs(A2)
        arg_i = W2*(2*h_list[1] - 1) + W2*(2*h_list[2] - 1) + A1 - h_list[0]*W
        arg_j = W2*S2*(2*h_list[1] -1) + W2*(2*h_list[3] - 1) + A1 - h_list[0]*W
        arg_k = W2*S2*(2*h_list[2] - 1) + W2*S2*(2*h_list[3] - 1) + A1 - h_list[0]*W
        out.coefficients[i] = ccosh(arg_i) * out.coefficients[i] + csinh(arg_i) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg_j) * out.coefficients[j] + csinh(arg_j) * onebody_matrix_j @ out.coefficients[j]
        out.coefficients[k] = ccosh(arg_k) * out.coefficients[k] + csinh(arg_k) * onebody_matrix_k @ out.coefficients[k]
        return out.scale_all(prefactor)

    def factors_sigma(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], i, j, self._sig[a], self._sig[b]) )
                        idx += 1
        return out

    def factors_sigmatau(self, coupling_array: CouplingArray,  aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`A^{\sigma\tau}_{\alpha i \beta j} \sigma_{i \alpha} \sigma_{j \beta}\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array:  force coupling array (e.g. :math:`A^{\sigma\tau}_{\alpha i \beta j}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*27
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i,b,j]==0.:
                        for c in self._xyz:
                            out.append( self.twobody_sample(coupling_array[a,i,b,j], aux[idx], i, j, self._sig[a] @ self._tau[c], self._sig[b] @ self._tau[c]) )
                            idx += 1
        return out
    
    def factors_tau(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`\tau_{i\gamma}\tau_{j\gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array:  force coupling array (e.g. :math:`A^{\tau}_{ij}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*3
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                if not coupling_array[i,j]==0.:
                    out.append( self.twobody_sample(coupling_array[i,j], aux[idx], i, j, self._tau[a], self._tau[a]) )
                    idx += 1
        return out

    def factors_coulomb(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the Coulomb propagator. 
        
        .. math::
            \exp \left[ -\frac{\delta\tau}{2} \frac{v_{ij}}{4} (1+\tau_{iz}+ \tau_{jz} + \tau_{iz}\tau_{jz} ) \right] 
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`v_C(r_{ij})`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs
        :type aux: list
        :return:  The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j in self._2b_idx:
            if not coupling_array[i,j]==0.:
                z = self.deltatau * self.symmetry_factor * 0.25 * coupling_array[i,j]
                if self.include_prefactors:
                    norm_op = ProductOperator(self.n_particles)
                    norm_op = norm_op.scale_all(cexp(-z))
                    out.append(norm_op)
                out.append( self.onebody(0.25 * coupling_array[i,j], i, self._tau[2]) )
                out.append( self.onebody(0.25 * coupling_array[i,j], j, self._tau[2]) )
                out.append( self.twobody_sample(0.25 * coupling_array[i,j], aux[idx], i, j, self._tau[2], self._tau[2]) )
                idx += 1
        return out
    
    
    def factors_spinorbit(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the spin-orbit propagator. 
        
        .. math::
            \exp \left[ - \frac{\delta\tau}{2} v_{LS}(r_{ij}) \mathbf{L}\cdot\mathbf{S} \right]
        
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`g^\text{LS}_{\alpha i}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*9
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                if not coupling_array[a,i]==0.:
                    z = 1.j * coupling_array[a,i] / (self.deltatau * self.symmetry_factor)
                    out.append( self.onebody(z, i, self._sig[a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    if not coupling_array[a,i]==0.:
                        if not coupling_array[b,j]==0.:
                            z = - 0.5 * coupling_array[a, i] * coupling_array[b, j] / (self.deltatau * self.symmetry_factor)
                            out.append( self.twobody_sample(z, aux[idx], i, j, self._sig[a], self._sig[b]) )
                            idx += 1
        if self.include_prefactors:
            norm_op = ProductOperator(self.n_particles)
            norm_op = norm_op.scale_all(np.exp( 0.5 * np.sum(coupling_array.coefficients**2)) )
            out.append( norm_op )
        return out    

    def factors_sigma_3b(self, coupling_array: CouplingArray, aux: list) -> list[ProductOperator]:
        r"""Creates factors of the :math:`A^\sigma_{\alpha i \beta j \gamma k} \sigma_{i \alpha} \sigma_{j \beta} \sigma_{k \gamma}` propagator. 
        The result is a list of (noncommuting) terms so they may be shuffled.

        :param coupling_array: force coupling array (e.g. :math:`A^\sigma_{\alpha i \beta j \gamma k}`)
        :type coupling_array: CouplingArray
        :param aux: values of auxiliary field, length equal to the number of pairs*108
        :type aux: list
        :return: The list of propagator factors
        :rtype: list[ProductOperator]
        """        
        out = []
        idx = 0
        for i,j,k in self._3b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        if not coupling_array[a,i,b,j,c,k]==0.:
                            out.append( self.threebody_sample(coupling_array[a,i,b,j,c,k], aux[idx:idx+4], i, j, k, self._sig[a], self._sig[b], self._sig[c]) )
                            idx += 4
        return out
    



    


class Integrator:
    def __init__(self, potential: NuclearPotential, propagator, isospin=True):
        if type(propagator) in [HilbertPropagatorHS, ProductPropagatorHS]:
            self.method = 'HS'
        elif type(propagator) in [HilbertPropagatorRBM, ProductPropagatorRBM]:
            self.method = 'RBM'
        self.n_particles = potential.n_particles
        self.potential = potential
        self.propagator = propagator
        self.isospin = isospin
        self.is_ready = False

    def setup(self, 
              n_samples, 
              seed=0, 
              mix=True,
              flip_aux=False,
              sigma=False, 
              sigmatau=False, 
              tau=False, 
              coulomb=False, 
              spinorbit=False,
              sigma_3b=False,
              parallel=True,
              n_processes=None):
        
        n_aux = 0
        if sigma:
            n_aux += self.propagator.n_aux_sigma
        if sigmatau:
            n_aux += self.propagator.n_aux_sigmatau
        if tau:
            n_aux += self.propagator.n_aux_tau
        if coulomb:
            n_aux += self.propagator.n_aux_coulomb
        if spinorbit:
            n_aux += self.propagator.n_aux_spinorbit
        if sigma_3b:
            n_aux += self.propagator.n_aux_sigma_3b

        self.sigma = sigma
        self.sigmatau = sigmatau
        self.tau = tau
        self.coulomb = coulomb
        self.spinorbit = spinorbit
        self.sigma_3b = sigma_3b
        self.mix = mix
        self.parallel = parallel
        self.n_processes = n_processes

        self.rng = np.random.default_rng(seed=seed)
        if self.method.lower() == 'hs':
            self.aux_fields_samples = self.rng.standard_normal(size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields_samples = - self.aux_fields_samples
        elif self.method.lower() == 'rbm':
            self.aux_fields_samples = self.rng.integers(0,2,size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields_samples = np.ones_like(self.aux_fields_samples) - self.aux_fields_samples
        self.is_ready = True

    def bracket(self, bra, ket, aux_fields):
        ket_prop = ket.copy()
        idx = 0
        self.prop_list = []
        if self.sigma:
            self.prop_list.extend( self.propagator.factors_sigma(self.potential.sigma, aux_fields[idx : idx + self.propagator.n_aux_sigma] ) )
            idx += self.propagator.n_aux_sigma
        if self.sigmatau:
            self.prop_list.extend( self.propagator.factors_sigmatau(self.potential.sigmatau, aux_fields[idx : idx + self.propagator.n_aux_sigmatau] ) )
            idx += self.propagator.n_aux_sigmatau
        if self.tau:
            self.prop_list.extend( self.propagator.factors_tau(self.potential.tau, aux_fields[idx : idx + self.propagator.n_aux_tau] ) )
            idx += self.propagator.n_aux_tau
        if self.coulomb:
            self.prop_list.extend( self.propagator.factors_coulomb(self.potential.coulomb, aux_fields[idx : idx + self.propagator.n_aux_coulomb] ) )
            idx += self.propagator.n_aux_coulomb
        if self.spinorbit:
            self.prop_list.extend( self.propagator.factors_spinorbit(self.potential.spinorbit, aux_fields[idx : idx + self.propagator.n_aux_spinorbit] ) )
            idx += self.propagator.n_aux_spinorbit
        if self.sigma_3b:
            self.prop_list.extend( self.propagator.factors_sigma_3b(self.potential.sigma_3b, aux_fields[idx : idx + self.propagator.n_aux_sigma_3b] ) )
            idx += self.propagator.n_aux_sigma_3b
        if self.mix:
            self.rng.shuffle(self.prop_list)
        for p in self.prop_list:
            ket_prop = p.multiply_state(ket_prop)
        return bra.inner(ket_prop)

    def run(self, bra, ket):
        if not self.is_ready:
            raise ValueError("Integrator is not ready. Did you run .setup() ?")
        assert (ket.ketwise) and (not bra.ketwise)
        if self.parallel:
            with Pool(processes=self.n_processes) as pool:
                b_array = pool.starmap_async(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields_samples], leave=True)).get()
        else:
            b_array = list(itertools.starmap(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields_samples])))
        b_array = np.array(b_array).flatten()
        return b_array
            
    def exact(self, bra, ket):
        ex = ExactPropagator(self.n_particles, self.propagator.dt, isospin=self.isospin, include_prefactors=self.propagator.include_prefactors)
        g_exact = ex.propagator_combined(self.potential,
                                            self.sigma,
                                            self.sigmatau,
                                            self.tau,
                                            self.coulomb,
                                            self.spinorbit,
                                            self.sigma_3b)
        b_exact = bra.inner(g_exact.multiply_state(ket))
        return b_exact
    
    