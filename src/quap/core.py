# quap
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.1.0'

# 4/23/2024 changelog
#
# replace orientation attribute (a string, 'bra' or 'ket') with new attribute "ketwise"
# if ketwise=True, it's a ket, else, it's a bra
#
# cut down on unnecessary asserts. replace with try/except if necessary.
#
# classes have a .copy() method to be used for __add__, __mul__, etc 
# but otherwise allow for in-place operations.
# e.g., c = a + b , when valid, uses a .copy() method
# but methods like .scalar_mult() and .apply_onebody_operator() need not use .copy()
# e.g., o = Operator(); o.apply_sigma(i,a); 
# this should result in sigma(i,a) * o
# the apply methods should also return the new modified object
#
# ideally every method should return something. if in-place, return self

# imports
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
from scipy.linalg import expm
from functools import reduce
# from dataclasses import dataclass
import itertools
from multiprocessing.pool import Pool
from tqdm import tqdm

###

SAFE = True

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
    

def interaction_indices(n, m = 2):
    """ returns a list of all possible m-plets of n objects (labelled 0 to n-1)
    default: m=2, giving all possible pairs
    for m=1, returns a range(0, n-1)
    """
    if m==1:
        return np.arange(n)
    else:
        return np.array(list(itertools.combinations(range(n), m)))


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


# Hilbert BASIS CLASSES

class HilbertState:
    def __init__(self, n_particles: int, coefficients=None, ketwise=True, isospin=True):
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.dimension = self.n_basis ** self.n_particles
        self.dim = self.dimension
        self.ketwise = ketwise
        self.friendly_operator = HilbertOperator
        
        if coefficients is None:
            if ketwise:
                self.coefficients = np.zeros(shape=(self.dim, 1))
            else:
                self.coefficients = np.zeros(shape=(1, self.dim))
        else: 
            # assert type(coefficients)==np.ndarray
            ket_condition = (coefficients.shape == (self.dim, 1)) and ketwise
            bra_condition = (coefficients.shape == (1, self.dim)) and not ketwise
            if not ket_condition and not bra_condition:
                raise ValueError("Inconsistent initialization of state vector. \n\
                                Did you get the shape right?")
            else:
                self.coefficients = coefficients.astype('complex')

    def copy(self):
        return HilbertState(self.n_particles, self.coefficients.copy(), self.ketwise, isospin=self.isospin)
    
    def __add__(self, other):
        if SAFE: assert type(other) == type(self)
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other):
        if SAFE: assert type(other) == type(self)
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def inner(self, other):
        if SAFE:
            assert type(other) == type(self)
            assert not self.ketwise and other.ketwise
        return np.dot(self.coefficients, other.coefficients) 
        
    def __mul__(self, other):
        return self.inner(other)
    
    def scalar_multiply(self, other):
        if SAFE: assert np.isscalar(other)
        self.coefficients *= other
        return self    
    
    def __rmul__(self, other):
        return self.scalar_multiply(other)
    
    def outer(self, other):
        if SAFE:
            assert type(other) == type(self)
            assert self.ketwise and not other.ketwise
        return np.matmul(self.coefficients, other.coefficients, dtype='complex') 
    
    def dagger(self):
        """ copy-based conjugate transpose """
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

    def randomize(self, seed):
        """ in-place randomize """
        rng = np.random.default_rng(seed=seed)
        self.coefficients = rng.standard_normal(size=self.coefficients.shape)
        self.coefficients /= np.linalg.norm(self.coefficients)
        return self
    
    def zero(self):
        """ in-place zero """
        self.coefficients = np.zeros_like(self.coefficients)
        return self


class HilbertOperator:
    def __init__(self, n_particles: int, isospin=True):
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.dimension = self.n_basis ** self.n_particles
        self.dim = self.dimension
        self.coefficients = np.identity(self.dim, dtype=complex)
        self.friendly_state = HilbertState

    def copy(self):
        out = HilbertOperator(self.n_particles, self.isospin)
        out.coefficients = self.coefficients.copy()
        return out
    
    def __add__(self, other):
        if SAFE: assert type(other) == type(self)
        out = self.copy()
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other):
        if SAFE: assert type(other) == type(self)
        out = self.copy()
        out.coefficients = self.coefficients - other.coefficients
        return out

    def __mul__(self, other: HilbertState):
        """ operator * |ket> """
        if SAFE: 
            assert type(other) == self.friendly_state
            assert other.ketwise
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out
        
    def __matmul__(self, other):
        """ operator_A @ operator_B"""
        if SAFE: assert type(other) == type(self)
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out

    def scalar_multiply(self, other):
        """ c * operator """
        if SAFE: assert np.isscalar(other)
        self.coefficients *= other
        return self    
    
    def __rmul__(self, other):
        """ c * operator """
        return self.scalar_multiply(other)

    def __str__(self):
        out = f"{self.__class__.__name__}\n"
        re = str(np.real(self.coefficients))
        im = str(np.imag(self.coefficients))
        out += "Re=\n" + re + "\nIm:\n" + im
        return out

    def apply_onebody_operator(self, particle_index: int, spin_matrix: np.ndarray, isospin_matrix=None):
        if SAFE:
            assert type(spin_matrix) == np.ndarray
            assert spin_matrix.shape == (2,2)
        obo = [np.identity(self.n_basis, dtype=complex) for _ in range(self.n_particles)]
        if self.isospin:
            if isospin_matrix is None:
                isospin_matrix = np.identity(2, dtype=complex)
            else:
                if SAFE:
                    assert type(spin_matrix) == np.ndarray
                    assert spin_matrix.shape == (2,2)        
            op = repeated_kronecker_product([isospin_matrix, spin_matrix])
        else:
            op = spin_matrix
        if SAFE:
            assert op.shape == obo[particle_index].shape
        obo[particle_index] = op
        obo = repeated_kronecker_product(obo)
        self.coefficients = np.matmul(obo, self.coefficients, dtype=complex)
        return self
        
    def apply_sigma(self, particle_index: int, dimension: int):
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=pauli(dimension), isospin_matrix=np.identity(2, dtype=complex) )

    def apply_tau(self, particle_index: int, dimension: int):
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=np.identity(2, dtype=complex), isospin_matrix=pauli(dimension) )
                
    def exp(self):
        self.coefficients = expm(self.coefficients)
        return self

    def zero(self):
        self.coefficients = np.zeros_like(self.coefficients)
        return self
        
    def dagger(self):
        """ copy-based conj transpose"""
        out = self.copy()
        out.coefficients = self.coefficients.conj().T
        return out



# ONE-BODY BASIS CLASSES
        

class ProductState:
    def __init__(self, n_particles: int, coefficients=None, ketwise=True, isospin=True):
        """an array of single particle spinors

        Orientation must be consistent with array shape!
        The shape of a bra is (A, 2, 1)
        The shape of a ket is (A, 1, 2)

        Args:
            n_particles (int): number of single particle states
            coefficients (np.ndarray): array of complex numbers
            ketwise (bool): True for ket, False for bra

        Raises:
            ValueError: _description_
        """
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.ketwise = ketwise
        self.friendly_operator = ProductOperator
        
        if coefficients is None:
            if ketwise:
                self.coefficients = np.zeros(shape=(self.n_particles, self.n_basis, 1))
            else:
                self.coefficients = np.zeros(shape=(self.n_particles, 1, self.n_basis))
        else:
            if SAFE:
                assert type(coefficients)==np.ndarray
            ket_condition = (coefficients.shape == (n_particles, self.n_basis, 1)) and ketwise
            bra_condition = (coefficients.shape == (n_particles, 1, self.n_basis)) and not ketwise
            if not ket_condition and not bra_condition:
                raise ValueError("Inconsistent initialization of state vector. \n\
                                Did you get the shape right?")
            else:
                self.coefficients = coefficients.astype('complex')

    def __add__(self, other):
        raise SyntaxError('Product states do not add')
    
    def __sub__(self, other):
        raise SyntaxError('Product states do not subtract')

    def copy(self):
        return ProductState(self.n_particles, self.coefficients.copy(), self.ketwise, self.isospin)

    def to_list(self):
        return [self.coefficients[i] for i in range(self.n_particles)]

    def inner(self, other):
        if SAFE:
            assert type(other) == type(self)
            assert (not self.ketwise) and other.ketwise
        return complex(np.prod([np.dot(self.coefficients[i], other.coefficients[i]) for i in range(self.n_particles)]))
        
    def __mul__(self, other):
        return self.inner(other)
        
    def outer(self, other):
        if SAFE:
            assert type(other) == type(self)
            assert (self.ketwise) and (not other.ketwise)
        out = ProductOperator(n_particles=self.n_particles, isospin=self.isospin)
        for i in range(self.n_particles):
            out.coefficients[i] = np.matmul(self.coefficients[i], other.coefficients[i], dtype=complex)
        return out

    def dagger(self):
        """ copy_based conj transpose"""
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

    def to_manybody_basis(self):
        """project the NxA TP state into the full N^A MB basis"""
        new_coeffs = repeated_kronecker_product(self.to_list())
        if self.ketwise:
            new_coeffs = new_coeffs.reshape(self.n_basis ** self.n_particles, 1)
        else:
            new_coeffs = new_coeffs.reshape(1, self.n_basis ** self.n_particles)
        return HilbertState(self.n_particles, new_coeffs, self.ketwise, self.isospin)

    def normalize(self):
        for i in range(self.n_particles):
            n = np.linalg.norm(self.coefficients[i])
            self.coefficients[i] /= n
        return self

    def scalar_mult(self, particle_index: int, b):
        if SAFE: assert np.isscalar(b)
        self.coefficients[particle_index] *= b
        return self

    def spread_scalar_mult(self, b):
        if SAFE: assert np.isscalar(b)
        self.coefficients *= b ** (1 / self.n_particles)
        return self

    def randomize(self, seed):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = rng.standard_normal(size=self.coefficients.shape)
        for i in range(self.n_particles):
            self.coefficients[i] /= np.linalg.norm(self.coefficients[i])
        return self

    def zero(self):
        self.coefficients = np.zeros_like(self.coefficients)
        return self




class ProductOperator:
    def __init__(self, n_particles: int, isospin=True):
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.coefficients = np.stack(self.n_particles*[np.identity(self.n_basis)], dtype=complex)    
        self.friendly_state = ProductState

    def __add__(self, other):
        raise SyntaxError('Product operators do not add')
    
    def __sub__(self, other):
        raise SyntaxError('Product operators do not subtract')

    def copy(self):
        out = ProductOperator(self.n_particles, self.isospin)
        for i in range(self.n_particles):
            out.coefficients[i] = self.coefficients[i]
        return out

    def to_list(self):
        return [self.coefficients[i] for i in range(self.n_particles)]
    
    def __mul__(self, other: ProductState):
        """ operator * |ket> """
        if SAFE:
            assert type(other) == self.friendly_state
            assert other.ketwise
        out = other.copy()
        for i in range(self.n_particles):
                out.coefficients[i] = np.matmul(self.coefficients[i], out.coefficients[i], dtype=complex)
        return out
        
    def __matmul__(self, other):
        """ operator_A @ operator_B"""
        if SAFE: assert type(other) == type(self)
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

    def apply_onebody_operator(self, particle_index: int, spin_matrix: np.ndarray, isospin_matrix=None):
        if self.isospin:
            if isospin_matrix is None:
                isospin_matrix = np.identity(2, dtype=complex)
            onebody_matrix = repeated_kronecker_product([isospin_matrix, spin_matrix])
        else:
            onebody_matrix = spin_matrix
        self.coefficients[particle_index] = np.matmul(onebody_matrix, self.coefficients[particle_index], dtype=complex)
        return self


    def apply_sigma(self, particle_index, dimension):
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))

    def apply_tau(self, particle_index, dimension):
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))

    def scalar_mult(self, particle_index: int, b):
        if SAFE: assert np.isscalar(b)
        self.coefficients[particle_index] *= b
        return self
        
    def scale_all(self, b):
        if SAFE: assert np.isscalar(b)
        self.coefficients *= b ** (1 / self.n_particles)
        return self

    def zero(self):
        self.coefficients = np.zeros_like(self.coefficients)
        return self

    def dagger(self):
        """ copy-based conj transpose"""
        out = self.copy()
        out.coefficients = np.transpose(self.coefficients, axes=(0,2,1)).conj()
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
        if SAFE: assert isinstance(self.coefficients, np.ndarray)
        out = Coupling(self.n_particles, self.shape)
        out.coefficients = self.coefficients.copy()
        return out

    def __mul__(self, other):
        if SAFE: assert np.isscalar(other)
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
    
        
class SigmaCoupling(Coupling):
    """container class for couplings A^sigma (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for a in range(3):
                    for b in range(3):
                        assert self.coefficients[a,i,b,j]==self.coefficients[a,j,b,i]
    
    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        for i in range(self.n_particles):
            self.coefficients[:,i,:,i] = 0.0
            for j in range(i):
                for a in range(3):
                    for b in range(3):
                        self.coefficients[a,i,b,j]=self.coefficients[a,j,b,i]     
        return self
    
class SigmaTauCoupling(Coupling):
    """container class for couplings A ^ sigma tau (a,i,b,j)
    for i, j = 0 .. n_particles - 1
    and a, b = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()
        
    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            self.coefficients[:,i,:,i] = 0.0
            for j in range(self.n_particles):
                for a in range(3):
                    for b in range(3):
                        assert self.coefficients[a,i,b,j]==self.coefficients[a,j,b,i]
        
    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        for i in range(self.n_particles):
            for j in range(i):
                for a in range(3):
                    for b in range(3):
                        self.coefficients[a,i,b,j]=self.coefficients[a,j,b,i]
        return self
    

class TauCoupling(Coupling):
    """container class for couplings A^tau (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, file=None):
        shape = (n_particles, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                assert self.coefficients[i,j]==self.coefficients[j,i]

    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        for i in range(self.n_particles):
            self.coefficients[i,i] = 0.0
            for j in range(i):
                self.coefficients[i,j]=self.coefficients[j,i]    
        return self


class CoulombCoupling(Coupling):
    """container class for couplings V^coul (i,j)
    for i, j = 0 .. n_particles - 1
    """
    def __init__(self, n_particles, file=None):
        shape = (n_particles, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                assert self.coefficients[i,j]==self.coefficients[j,i]

    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        for i in range(self.n_particles):
            self.coefficients[i,i] = 0.0
            for j in range(i):
                self.coefficients[i,j]=self.coefficients[j,i]    
        return self



class SpinOrbitCoupling(Coupling):
    """container class for couplings g_LS (a,i)
    for i = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        # gLS is a vector. no symmetry to validate.   

    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        return self


class ThreeBodyCoupling(Coupling):
    """container class for couplings A(a,i,b,j,c,k)
    for i, j, k = 0 .. n_particles - 1
    and a = 0, 1, 2  (x, y, z)
    """
    def __init__(self, n_particles, file=None):
        shape = (3, n_particles, 3, n_particles, 3, n_particles)
        super().__init__(n_particles, shape, file)
        if SAFE and (file is not None):
            self.validate()

    def validate(self):
        assert self.coefficients.shape==self.shape
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                for k in range(self.n_particles):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                assert self.coefficients[a,i,b,i,c,i]==0.0
                                assert self.coefficients[a,i,b,i,c,k]==0.0
                                assert self.coefficients[a,i,b,j,c,i]==0.0
                                assert self.coefficients[a,i,b,j,c,j]==0.0
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,i,b,k,c,j]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,j,b,i,c,k]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,j,b,k,c,i]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,k,b,i,c,j]
                                assert self.coefficients[a,i,b,j,c,k]==self.coefficients[a,k,b,j,c,i]

    def generate(self, scale, seed=0):
        rng = np.random.default_rng(seed=seed)
        self.coefficients = scale*rng.standard_normal(size=self.shape)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                self.coefficients[:,i,:,i,:,i] = 0.0
                self.coefficients[:,i,:,i,:,j] = 0.0
                self.coefficients[:,i,:,j,:,i] = 0.0    
                self.coefficients[:,i,:,j,:,j] = 0.0    
        return self


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

class Propagator:
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        self.n_particles = n_particles
        self.dt = dt
        self.include_prefactors = include_prefactors
        self.isospin = isospin
        self._xyz = [0, 1, 2]
        self._1b_idx = interaction_indices(n_particles, 1)
        self._2b_idx = interaction_indices(n_particles, 2)
        self._3b_idx = interaction_indices(n_particles, 3)
        self._n2 = len(self._2b_idx)
        self._n3 = len(self._3b_idx)


class HilbertPropagatorHS(Propagator):
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)        
        self._ident = HilbertOperator(self.n_particles, isospin=isospin)
        self._sig_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[HilbertOperator(self.n_particles, isospin=isospin).apply_tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = 1 * self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, k, operator):
        """exp (- k opi)"""
        return (- k * operator).exp()        

    def twobody_sample(self, k: complex, x: complex, operator_i, operator_j):
        """ exp( x sqrt( -k ) opi ) * exp( x sqrt( -k ) opj ) """
        arg = csqrt(-k)*x
        if self.include_prefactors:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        gi = ccosh(arg) * self._ident + csinh(arg) * operator_i
        gj = ccosh(arg) * self._ident + csinh(arg) * operator_j
        return prefactor * gi @ gj
                
    def factors_sigma(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                    idx += 1
        return out

    def factors_sigmatau(self, potential: ArgonnePotential,  aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a] @ self._tau_op[i][c], self._sig_op[j][b] @ self._tau_op[j][c]) )
                        idx += 1
        return out
    
    def factors_tau(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    out.append( self.twobody_sample(k, aux[idx], self._tau_op[i][a], self._tau_op[j][a]) )
        return out

    def factors_coulomb(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
                k = 0.125 * self.dt * potential.coulomb[i,j]
                if self.include_prefactors:
                    out.append(cexp(-k) * HilbertOperator(self.n_particles, self.isospin))
                out.append( self.onebody(k, self._tau_op[i][2]) )
                out.append( self.onebody(k, self._tau_op[j][2]) )
                out.append( self.twobody_sample(k, aux[idx], self._tau_op[i][2], self._tau_op[j][2]) )
                idx += 1
        return out
    
    def factors_spinorbit(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                k = 1.j * potential.spinorbit[a,i]
                out.append( self.onebody(k,  self._sig_op[i][a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                    idx += 1
        if self.include_prefactors:
            out.append( np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * HilbertOperator(self.n_particles, self.isospin) )
        return out



class HilbertPropagatorRBM(Propagator):
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = HilbertOperator(self.n_particles)
        self._sig_op = [[HilbertOperator(self.n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[HilbertOperator(self.n_particles).apply_tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = 1 * self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, k, operator):
        """exp (- k opi)"""
        return (- k * operator).exp()        

    def twobody_sample(self, k, h, operator_i, operator_j):
        if self.include_prefactors:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        sgn = k/abs(k)
        gi = ccosh(arg) * self._ident + csinh(arg) * operator_i
        gj = ccosh(arg) * self._ident - sgn*csinh(arg) * operator_j
        return prefactor * gi @ gj

    def factors_sigma(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                    idx += 1
        return out

    def factors_sigmatau(self, potential: ArgonnePotential,  aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a]@self._tau_op[i][c], self._sig_op[j][b]@self._tau_op[j][c]) )
                        idx += 1
        return out
    
    def factors_tau(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    out.append( self.twobody_sample(k, aux[idx], self._tau_op[i][a], self._tau_op[j][a]) )
        return out

    def factors_coulomb(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
                k = 0.125 * self.dt * potential.coulomb[i,j]
                if self.include_prefactors:
                    out.append(cexp(-k) * HilbertOperator(self.n_particles, self.isospin))
                out.append( self.onebody(k, self._tau_op[i][2]) )
                out.append( self.onebody(k, self._tau_op[j][2]) )
                out.append( self.twobody_sample(k, aux[idx], self._tau_op[i][2], self._tau_op[j][2]) )
                idx += 1
        return out
    
    def factors_spinorbit(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                k = 1.j * potential.spinorbit[a,i]
                out.append( self.onebody(k,  self._sig_op[i][a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
                    idx += 1
        if self.include_prefactors:
            out.append( np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) * HilbertOperator(self.n_particles, self.isospin) )
        return out


class ProductPropagatorHS(Propagator):
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = np.identity(4)
        self._sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        self._tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = 1 * self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, k, i, matrix):
        """exp (- k opi) * |ket> """
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(k) * out.coefficients[i] - csinh(k) * matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, k, x, i, j, operator_i, operator_j):
        """exp ( sqrt( -kx ) opi opj) * |ket>  """
        arg = csqrt(-k)*x
        if self.include_prefactors:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * operator_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] + csinh(arg) * operator_j @ out.coefficients[j]
        out.coefficients[i] *= csqrt(prefactor)
        out.coefficients[j] *= csqrt(prefactor)
        return out

    def factors_sigma(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a], self._sig[b]) )
                    idx += 1
        return out

    def factors_sigmatau(self, potential: ArgonnePotential,  aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a]@self._tau[c], self._sig[b]@self._tau[c]) )
                        idx += 1
        return out
    
    def factors_tau(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._tau[a], self._tau[a]) )
        return out

    def factors_coulomb(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
                k = 0.125 * self.dt * potential.coulomb[i,j]
                if self.include_prefactors:
                    norm_op = ProductOperator(self.n_particles, self.isospin)
                    norm_op = norm_op.spread_scalar_mult(cexp(-k))
                    out.append(norm_op)
                out.append( self.onebody(k, i, self._tau[2]) )
                out.append( self.onebody(k, j, self._tau[2]) )
                out.append( self.twobody_sample(k, aux[idx], i, j, self._tau[2], self._tau[2]) )
                idx += 1
        return out
    
    def factors_spinorbit(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                k = 1.j * potential.spinorbit[a,i]
                out.append( self.onebody(k, i, self._sig[a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a], self._sig[b]) )
                    idx += 1
        if self.include_prefactors:
            norm_op = ProductOperator(self.n_particles, self.isospin)
            norm_op = norm_op.spread_scalar_mult(np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) )
            out.append( norm_op )
        return out    


class ProductPropagatorRBM(Propagator):
    """ exp( - k op_i op_j )
    seed determines mixing
    """
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = np.identity(4)
        self._sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        self._tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = 1 * self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, k, i, matrix):
        """exp (- k opi) * |ket> """
        out = ProductOperator(self.n_particles)
        out.coefficients[i] = ccosh(k) * out.coefficients[i] - csinh(k) * matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, k, h, i, j, operator_i, operator_j):
        if self.include_prefactors:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * operator_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] - np.sign(k) * csinh(arg) * operator_j @ out.coefficients[j]
        out.coefficients[i] *= csqrt(prefactor)
        out.coefficients[j] *= csqrt(prefactor)
        return out

    def factors_sigma(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = 0.5 * self.dt * potential.sigma[a,i,b,j]
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a], self._sig[b]) )
                    idx += 1
        return out

    def factors_sigmatau(self, potential: ArgonnePotential,  aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    for c in self._xyz:
                        k = 0.5 * self.dt * potential.sigmatau[a,i,b,j]
                        out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a]@self._tau[c], self._sig[b]@self._tau[c]) )
                        idx += 1
        return out
    
    def factors_tau(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
            for a in self._xyz:
                    k = 0.5 * self.dt * potential.tau[i,j]
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._tau[a], self._tau[a]) )
        return out

    def factors_coulomb(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i,j in self._2b_idx:
                k = 0.125 * self.dt * potential.coulomb[i,j]
                if self.include_prefactors:
                    norm_op = ProductOperator(self.n_particles)
                    norm_op = norm_op.scale_all(cexp(-k))
                    out.append(norm_op)
                out.append( self.onebody(k, i, self._tau[2]) )
                out.append( self.onebody(k, j, self._tau[2]) )
                out.append( self.twobody_sample(k, aux[idx], i, j, self._tau[2], self._tau[2]) )
                idx += 1
        return out
    
    
    def factors_spinorbit(self, potential: ArgonnePotential, aux: list):
        out = []
        idx = 0
        for i in self._1b_idx:
            for a in self._xyz:
                k = 1.j * potential.spinorbit[a,i]
                out.append( self.onebody(k, i, self._sig[a])  )
        for i,j in self._2b_idx:
            for a in self._xyz:
                for b in self._xyz:
                    k = - 0.5 * potential.spinorbit[a, i] * potential.spinorbit[b, j] 
                    out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a], self._sig[b]) )
                    idx += 1
        if self.include_prefactors:
            norm_op = ProductOperator(self.n_particles)
            norm_op = norm_op.scale_all(np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) )
            out.append( norm_op )
        return out    


class ExactGFMC:
    """the "exact" propagator calculation must be done in the complete many-body basis
    we use Pade approximants for matrix exponentials
    the LS term can be represented using a linear approximation or the factorization procedure described in Stefano's thesis
    """
    def __init__(self, n_particles, isospin=True):
        self.n_particles = n_particles
        self.isospin = isospin
        self.ident = HilbertOperator(n_particles, self.isospin)
        self.sig = [[HilbertOperator(n_particles, self.isospin).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(n_particles)]
        self.tau = [[HilbertOperator(n_particles, self.isospin).apply_tau(i,a) for a in [0, 1, 2]] for i in range(n_particles)]

    def g_pade_sig(self, dt, asig, i, j):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                out += asig[a, i, b, j] * self.sig[i][a] * self.sig[j][b]
        out = -0.5 * dt * out
        return out.exp()


    def g_pade_sigtau(self, dt, asigtau, i, j):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out += asigtau[a, i, b, j] * self.sig[i][a] * self.sig[j][b] * self.tau[i][c] * self.tau[j][c]
        out = -0.5 * dt * out
        return out.exp()


    def g_pade_tau(self, dt, atau, i, j):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for c in range(3):
            out += atau[i, j] * self.tau[i][c] * self.tau[j][c]
        out = -0.5 * dt * out
        return out.exp()


    def g_pade_coul(self, dt, v, i, j):
        out = self.ident + self.tau[i][2] + self.tau[j][2] + self.tau[i][2] * self.tau[j][2]
        out = -0.125 * v[i, j] * dt * out
        return out.exp()


    def g_coulomb_onebody(self, dt, v, i):
        """just the one-body part of the expanded coulomb propagator
        for use along with auxiliary field propagators"""
        out = - 0.125 * v * dt * self.tau[i][2]
        return out.exp()


    def g_ls_linear(self, gls, i):
        # linear approx to LS
        out = HilbertOperator(self.n_particles)
        for a in range(3):
            out = (self.ident - 1.j * gls[a, i] * self.sig[i][a]) * out 
        return out
    

    def g_ls_onebody(self, gls, i, a):
        # one-body part of the LS propagator factorization
        out = - 1.j * gls[a,i] * self.sig[i][a]
        return out.exp()


    def g_ls_twobody(self, gls, i, j, a, b):
        # two-body part of the LS propagator factorization
        out = 0.5 * gls[a,i] * gls[b,j] * self.sig[i][a] * self.sig[j][b]
        return out.exp()


    def g_pade_sig_3b(self, dt, asig3b, i, j, k):
        # 3-body sigma
        out = HilbertOperator(self.n_particles).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out += asig3b[a, i, b, j, c, k] * self.sig[i][a] * self.sig[j][b] * self.sig[k][c]
        out = -0.5 * dt * out
        return out.exp()


    def make_g_exact(self, dt, potential, controls):
        # compute exact bracket
        g_exact = self.ident.copy()
        pairs_ij = interaction_indices(self.n_particles)
        for i,j in pairs_ij:
            if controls['sigma']:
                g_exact = self.g_pade_sig(dt, potential.sigma, i, j) * g_exact
            if controls['sigmatau']:
                g_exact = self.g_pade_sigtau(dt, potential.sigmatau, i, j) * g_exact 
            if controls['tau']:
                g_exact = self.g_pade_tau(dt, potential.tau, i, j) * g_exact
            if controls['coulomb']:
                g_exact = self.g_pade_coul(dt, potential.coulomb, i, j) * g_exact
        if controls['spinorbit']:
            # linear approximation
            # for i in range(self.n_particles):
            #     g_exact = self.g_ls_linear(potential.spinorbit, i) * g_exact
            # factorized into one- and two-body
            for i in range(self.n_particles):
                for a in range(3):
                    g_exact = self.g_ls_onebody(potential.spinorbit, i, a) * g_exact
            for i in range(self.n_particles):
                for j in range(self.n_particles):
                    for a in range(3):
                        for b in range(3):
                            g_exact = self.g_ls_twobody(potential.spinorbit, i, j, a, b) * g_exact
        return g_exact
    


class Integrator:
    def __init__(self, potential: ArgonnePotential, propagator):
        if type(propagator) in [HilbertPropagatorHS, ProductPropagatorHS]:
            self.method = 'HS'
        elif type(propagator) in [HilbertPropagatorRBM, ProductPropagatorRBM]:
            self.method = 'RBM'
        self.n_particles = potential.n_particles
        self.potential = potential
        self.propagator = propagator
        self.is_ready = False

    def setup(self, n_samples, seed=1729, mix=True, flip_aux=False,
              sigma=False, sigmatau=False, tau=False, coulomb=False, spinorbit=False):
        self.controls={"seed":seed, "mix":mix, "sigma":sigma, "sigmatau":sigmatau, 
                       "tau":tau, "coulomb":coulomb, "spinorbit":spinorbit}
        
        n_aux = 0
        if self.controls['sigma']:
            n_aux += self.propagator.n_aux_sigma
        if self.controls['sigmatau']:
            n_aux += self.propagator.n_aux_sigmatau
        if self.controls['tau']:
            n_aux += self.propagator.n_aux_tau
        if self.controls['coulomb']:
            n_aux += self.propagator.n_aux_coulomb
        if self.controls['spinorbit']:
            n_aux += self.propagator.n_aux_spinorbit

        self.rng = np.random.default_rng(seed=self.controls["seed"])
        if self.method=='HS':
            self.aux_fields = self.rng.standard_normal(size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields = - self.aux_fields
        elif self.method=='RBM':
            self.aux_fields = self.rng.integers(0,2,size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields = np.ones_like(self.aux_fields) - self.aux_fields
        self.is_ready = True
        return self.controls

    def bracket(self, bra, ket, aux_fields):
        ket_prop = ket.copy()
        idx = 0
        self.prop_list = []
        if self.controls['sigma']:
            self.prop_list.extend( self.propagator.factors_sigma(self.potential, aux_fields[idx : idx + self.propagator.n_aux_sigma] ) )
            idx += self.propagator.n_aux_sigma
        if self.controls['sigmatau']:
            self.prop_list.extend( self.propagator.factors_sigmatau(self.potential, aux_fields[idx : idx + self.propagator.n_aux_sigmatau] ) )
            idx += self.propagator.n_aux_sigmatau
        if self.controls['tau']:
            self.prop_list.extend( self.propagator.factors_tau(self.potential, aux_fields[idx : idx + self.propagator.n_aux_tau] ) )
            idx += self.propagator.n_aux_tau
        if self.controls['coulomb']:
            self.prop_list.extend( self.propagator.factors_coulomb(self.potential, aux_fields[idx : idx + self.propagator.n_aux_coulomb] ) )
            idx += self.propagator.n_aux_coulomb
        if self.controls['spinorbit']:
            self.prop_list.extend( self.propagator.factors_spinorbit(self.potential, aux_fields[idx : idx + self.propagator.n_aux_spinorbit] ) )
            idx += self.propagator.n_aux_spinorbit
        if self.controls["mix"]:
            self.rng.shuffle(self.prop_list)
        for p in self.prop_list:
            ket_prop = p * ket_prop
        return bra * ket_prop

    def run(self, bra, ket, parallel=True, n_processes=None):
        if not self.is_ready:
            raise ValueError("Integrator is not ready. Did you run .setup() ?")
        assert (ket.ketwise) and (not bra.ketwise)
        if parallel:
            with Pool(processes=n_processes) as pool:
                b_array = pool.starmap_async(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields], leave=True)).get()
        else:
            b_array = list(itertools.starmap(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields])))
        b_array = np.array(b_array).flatten()
        return b_array
            
    def exact(self, bra, ket):
        ex = ExactGFMC(self.n_particles)
        g_exact = ex.make_g_exact(self.propagator.dt, self.potential, self.controls)
        b_exact = bra * g_exact * ket
        return b_exact