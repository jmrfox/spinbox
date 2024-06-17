# quap
# a quantum mechanics playground
# jordan m r fox 2024

__version__ = '0.1.0'

# 5/17 changelog
# 
# *** classes are now isospin dependent
#   rather than having one isospinless version and another isospinful version
#   the class takes isospin as a bool
# *** NOTHING is done in-place anymore
#   every method will instatiate a self.copy(), do operations on that, then return that
#    in-place operations are far too unintuitive and inevitably lead to bugs
# *** state and operator operations are now relegated to named methods only
#   the one exception is + and - for the Hilbert classes
#   but each type of multiplication should have its own method
#   if the user wants to alias those, they are free to do so
# *** introduced SAFE mode for doing assert checks and stuff
#

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
        
    def outer(self, other):
        if SAFE:
            assert type(other) == type(self)
            assert self.ketwise and not other.ketwise
        return np.matmul(self.coefficients, other.coefficients, dtype='complex') 
    
    def scale(self, other):
        if SAFE: assert np.isscalar(other)
        out = self.copy()
        out.coefficients *= other
        return out

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

    def randomize(self, seed=0):
        """ randomize """
        rng = np.random.default_rng(seed=seed)
        out = self.copy()
        out.coefficients = rng.standard_normal(size=out.coefficients.shape) + 1.j*rng.standard_normal(size=out.coefficients.shape)
        out.coefficients /= np.linalg.norm(out.coefficients)
        return out
    
    def zero(self):
        """ zero """
        out = self.copy()
        out.coefficients = np.zeros_like(self.coefficients)
        return out
    
    def entropy(self):
         return - np.sum(self.coefficients * np.log(self.coefficients))
     
    def generate_basis_states(self):
        coeffs_list = list(np.eye(self.n_basis**self.n_particles))
        out = [HilbertState(self.n_particles, coefficients=c, ketwise=True, isospsin=self.isospin) for c in coeffs_list]
        return out
    
    def nearby_product_state(self, seed: int, maxiter=100):
        from scipy.optimize import minimize, NonlinearConstraint
        fit = ProductState(self.n_particles, isospin=self.isospin).randomize(seed)
        shape = fit.coefficients.shape
        n_coeffs = len(fit.coefficients.flatten())
        n_params = 2*n_coeffs
        def x_to_coef(x):
            return x[:n_params//2].reshape(shape) + 1.j*x[n_params//2:].reshape(shape)   
        def loss(x):
            fit.coefficients = x_to_coef(x)
            overlap = self.dagger().inner(fit.to_manybody_basis())
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
    
    def nearest_product_state(self, seeds: list, maxiter=100):
        overlap = 0.
        for seed in seeds:
            this_fit, _ = self.nearby_product_state(seed, maxiter=maxiter)
            this_overlap = this_fit.to_manybody_basis().dagger().inner(self)
            if this_overlap>overlap:
                overlap = this_overlap
                fit = this_fit
        return fit
    


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

    def multiply_state(self, other):
        if SAFE: assert isinstance(other, self.friendly_state)
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out
        
    def multiply_operator(self, other):
        if SAFE: assert isinstance(other, type(self))
        out = other.copy()
        out.coefficients = np.matmul(self.coefficients, out.coefficients, dtype=complex)
        return out
        
    def scale(self, other):
        """ c * operator """
        if SAFE: assert np.isscalar(other)
        out = self.copy()
        out.coefficients *= other
        return out
    
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
        out = self.copy()
        out.coefficients = np.matmul(obo, out.coefficients, dtype=complex)
        return out
        
    def apply_sigma(self, particle_index: int, dimension: int):
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=pauli(dimension), isospin_matrix=np.identity(2, dtype=complex) )

    def apply_tau(self, particle_index: int, dimension: int):
        return self.apply_onebody_operator(particle_index=particle_index, spin_matrix=np.identity(2, dtype=complex), isospin_matrix=pauli(dimension) )
                
    def exp(self):
        out = self.copy()
        out.coefficients = expm(out.coefficients)
        return out

    def zero(self):
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out
        
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
        The shape of a bra is (A, n_basis, 1)
        The shape of a ket is (A, 1, n_basis)
        n_basis = 2 for spin, 4 for spin-isospin

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

    def copy(self):
        return ProductState(self.n_particles, self.coefficients.copy(), self.ketwise, self.isospin)

    def to_list(self):
        return [self.coefficients[i] for i in range(self.n_particles)]

    def inner(self, other):
        if SAFE:
            if isinstance(other, type(self)):
                assert (not self.ketwise) and other.ketwise
            else:
                raise TypeError("Bad multiply.")
        return np.prod([np.dot(self.coefficients[i], other.coefficients[i]) for i in range(self.n_particles)])
        
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
        out = self.copy()
        for i in range(out.n_particles):
            n = np.linalg.norm(out.coefficients[i])
            out.coefficients[i] /= n
        return out

    def scale_one(self, particle_index: int, b):
        if SAFE: assert np.isscalar(b)
        out = self.copy()
        out.coefficients[particle_index] *= b
        return out

    def scale_all(self, b):
        if SAFE: assert np.isscalar(b)
        out = self.copy()
        out.coefficients *= b ** (1 / out.n_particles)
        return out

    def randomize(self, seed=0):
        rng = np.random.default_rng(seed=seed)
        out = self.copy()
        out.coefficients = rng.standard_normal(size=out.coefficients.shape) + 1.j*rng.standard_normal(size=out.coefficients.shape)
        for i in range(out.n_particles):
            out.coefficients[i] /= np.linalg.norm(out.coefficients[i])
        return out

    def zero(self):
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out

    def generate_basis_states(self):
        coeffs_list = [np.concatenate(x).reshape(self.n_particles,self.n_basis,1) for x in itertools.product(list(np.eye(self.n_basis)), repeat=self.n_particles)]
        out = [ProductState(self.n_particles, coefficients=c, ketwise=True, isospsin=self.isospin) for c in coeffs_list]
        return out



class ProductOperator:
    def __init__(self, n_particles: int, isospin=True):
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2*isospin
        self.coefficients = np.stack(self.n_particles*[np.identity(self.n_basis)], dtype=complex)
        self.friendly_state = ProductState

    def copy(self):
        out = ProductOperator(self.n_particles, self.isospin)
        for i in range(self.n_particles):
            out.coefficients[i] = self.coefficients[i]
        return out

    def to_list(self):
        return [self.coefficients[i] for i in range(self.n_particles)]
        
    def multiply_state(self, other):
        if SAFE: assert isinstance(other, self.friendly_state)
        out = other.copy()
        for i in range(self.n_particles):
                out.coefficients[i] = np.matmul(self.coefficients[i], out.coefficients[i], dtype=complex)
        return out
        
    def multiply_operator(self, other):
        if SAFE: assert isinstance(other, type(self))
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
        out = self.copy()
        out.coefficients[particle_index] = np.matmul(onebody_matrix, out.coefficients[particle_index], dtype=complex)
        return out

    def apply_sigma(self, particle_index, dimension):
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=np.identity(2, dtype=complex),
                                          spin_matrix=pauli(dimension))

    def apply_tau(self, particle_index, dimension):
        return self.apply_onebody_operator(particle_index=particle_index,
                                          isospin_matrix=pauli(dimension),
                                          spin_matrix=np.identity(2, dtype=complex))

    def scale_one(self, particle_index: int, b):
        if SAFE: assert np.isscalar(b)
        out = self.copy()
        out.coefficients[particle_index] *= b
        return out
        
    def scale_all(self, b):
        if SAFE: assert np.isscalar(b)
        out = self.copy()
        out.coefficients *= b ** (1 / out.n_particles)
        return out

    def zero(self):
        out = self.copy()
        out.coefficients = np.zeros_like(out.coefficients)
        return out

    def dagger(self):
        """ conj transpose"""
        out = self.copy()
        out.coefficients = np.transpose(self.coefficients, axes=(0,2,1)).conj()
        return out        
        
    def to_manybody_basis(self):
        """project the product operator into the full many-body configuration basis"""
        new_coeffs = repeated_kronecker_product(self.to_list())
        return HilbertOperator(self.n_particles, new_coeffs, self.isospin)

    

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

    def onebody(self, k: complex, operator: HilbertOperator):
        """exp (- k opi)"""
        return operator.scale(-k).exp()        

    def twobody_sample(self, k: complex, x: float, operator_i: HilbertOperator, operator_j: HilbertOperator):
        """ exp( x sqrt( -k ) opi ) * exp( x sqrt( -k ) opj ) """
        arg = csqrt(-k)*x
        if self.include_prefactors:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
        gj = self._ident.scale(ccosh(arg)) + operator_j.scale(csinh(arg))
        return gi.multiply_operator(gj).scale(prefactor)
                
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
                        opi = self._sig_op[i][a].multiply_operator(self._tau_op[i][c])
                        opj = self._sig_op[j][b].multiply_operator(self._tau_op[j][c])
                        out.append( self.twobody_sample(k, aux[idx], opi, opj) )
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
                    out.append(HilbertOperator(self.n_particles, self.isospin).scale(cexp(-k)))
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
            prefactor = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2))
            out.append(  HilbertOperator(self.n_particles, self.isospin).scale(prefactor) )
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

    def onebody(self, k: complex, operator: HilbertOperator):
        """exp (- k opi)"""
        return operator.scale(-k).exp()

    def twobody_sample(self, k: float, h: int, operator_i: HilbertOperator, operator_j: HilbertOperator):
        if self.include_prefactors:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
        gj = self._ident.scale(ccosh(arg)) - operator_j.scale(np.sign(k) * csinh(arg))
        return gi.multiply_operator(gj).scale(prefactor)

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
                        opi = self._sig_op[i][a].multiply_operator(self._tau_op[i][c])
                        opj = self._sig_op[j][b].multiply_operator(self._tau_op[j][c])
                        out.append( self.twobody_sample(k, aux[idx], opi, opj) )
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
                    out.append(HilbertOperator(self.n_particles, self.isospin).scale(cexp(-k)))
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
            prefactor = np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2))
            out.append( HilbertOperator(self.n_particles, self.isospin).scale(prefactor) )
        return out


class HilbertPropagatorRBM3(Propagator):
    """ exp( - z op_i op_j op_k )"""
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = HilbertOperator(self.n_particles)
        self._sig_op = [[HilbertOperator(self.n_particles).apply_sigma(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self._tau_op = [[HilbertOperator(self.n_particles).apply_tau(i,a) for a in [0, 1, 2]] for i in range(self.n_particles)]
        self.n_aux_sigma = 9 * self._n3

    def _a2b_factors(z):
        n = cexp(-abs(z))/2.
        w = carctanh(csqrt(ctanh(abs(z))))
        s = z/abs(z)
        return n, w, s

    def _a3b_factors(a3):
        log = lambda x: np.log(x, dtype=complex)
        if a3>0:
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
        
    def onebody(self, z: complex, operator: HilbertOperator):
        """exp (- z opi)"""
        return operator.scale(-z).exp()

    def twobody_sample(self, z: float, h: int, operator_i: HilbertOperator, operator_j: HilbertOperator):
        if self.include_prefactors:
            prefactor = cexp(-abs(z))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(z))))
        arg = W*(2*h-1)
        gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
        gj = self._ident.scale(ccosh(arg)) - operator_j.scale(np.sign(z) * csinh(arg))
        return gi.multiply_operator(gj).scale(prefactor)

    def threebody_sample(self, z: float, h_list: list, operator_i: HilbertOperator, operator_j: HilbertOperator, operator_k: HilbertOperator):
            N, C, W, A1, A2 = self._a3b_factors(0.5 * self.dt * z)
            if self.include_prefactors:
                prefactor = N*cexp(-h_list[0]*C)
            else:
                prefactor = 1.0
            # one-body factors
            arg = A1 - h_list[0]*W
            gi = self._ident.scale(ccosh(arg)) + operator_i.scale(csinh(arg))
            gj = self._ident.scale(ccosh(arg)) + operator_j.scale(csinh(arg))
            gk = self._ident.scale(ccosh(arg)) + operator_k.scale(csinh(arg))
            out = gk.multiply_operator(gj).multiply_operator(gi).scale(prefactor)
            # two-body factors
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[1], operator_i, operator_j))
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[2], operator_i, operator_k))
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[3], operator_j, operator_k))
            return out.scale(prefactor)

    # def factors_sigma(self, potential: ArgonnePotential, aux: list):
    #     out = []
    #     idx = 0
    #     for i,j in self._2b_idx:
    #         for a in self._xyz:
    #             for b in self._xyz:
    #                 k = 0.5 * self.dt * potential.sigma[a,i,b,j]
    #                 out.append( self.twobody_sample(k, aux[idx], self._sig_op[i][a], self._sig_op[j][b]) )
    #                 idx += 1
    #     return out


class ProductPropagatorHS(Propagator):
    """ exp( - k op_i op_j )"""
    def __init__(self, n_particles: int, dt: float, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        
        if isospin:
            self._ident = np.identity(4)
            self._sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
            self._tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
        else:
            self._ident = np.identity(2)
            self._sig = pauli('list')
            self._tau = None
        self.n_aux_sigma = 9 * self._n2
        self.n_aux_sigmatau = 27 * self._n2
        self.n_aux_tau = 3 * self._n2
        self.n_aux_coulomb = 1 * self._n2
        self.n_aux_spinorbit = 9 * self._n2

    def onebody(self, k: complex, i: int, onebody_matrix: np.ndarray):
        """exp (- k opi) * |ket> """
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(k) * out.coefficients[i] - csinh(k) * onebody_matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, k: complex, x: float, i: int, j: int, onebody_matrix_i: np.ndarray, onebody_matrix_j: np.ndarray):
        """exp ( sqrt( -kx ) opi opj) * |ket>  """
        arg = csqrt(-k)*x
        if self.include_prefactors:
            prefactor = cexp(k)
        else:
            prefactor = 1.0
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] + csinh(arg) * onebody_matrix_j @ out.coefficients[j]
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
                        out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a] @ self._tau[c], self._sig[b] @ self._tau[c]) )
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
            norm_op = ProductOperator(self.n_particles, self.isospin)
            norm_op = norm_op.scale_all(np.exp( 0.5 * np.sum(potential.spinorbit.coefficients**2)) )
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

    def onebody(self, k: complex, i: int, onebody_matrix: np.ndarray):
        """exp (- k opi) * |ket> """
        out = ProductOperator(self.n_particles)
        out.coefficients[i] = ccosh(k) * out.coefficients[i] - csinh(k) * onebody_matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, k: complex, h: int, i: int, j: int, onebody_matrix_i, onebody_matrix_j):
        if self.include_prefactors:
            prefactor = cexp(-abs(k))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(k))))
        arg = W*(2*h-1)
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] - np.sign(k) * csinh(arg) * onebody_matrix_j @ out.coefficients[j]
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
                        out.append( self.twobody_sample(k, aux[idx], i, j, self._sig[a] @ self._tau[c], self._sig[b] @ self._tau[c]) )
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


class ProductPropagatorRBM3(Propagator):
    """ exp( - k op_i op_j op_k )
    seed determines mixing
    """
    def __init__(self, n_particles, dt, isospin=True, include_prefactors=True):
        super().__init__(n_particles, dt, isospin, include_prefactors)
        self._ident = np.identity(4)
        self._sig = [repeated_kronecker_product([np.identity(2), pauli(a)]) for a in [0, 1, 2]]
        self._tau = [repeated_kronecker_product([pauli(a), np.identity(2)]) for a in [0, 1, 2]]
        self.n_aux_sigma = 9 * self._n3

    def onebody(self, z: complex, i: int, onebody_matrix: np.ndarray):
        """exp (- k opi) * |ket> """
        out = ProductOperator(self.n_particles)
        out.coefficients[i] = ccosh(z) * out.coefficients[i] - csinh(z) * onebody_matrix @ out.coefficients[i]
        return out
    
    def twobody_sample(self, z: complex, h: int, i: int, j: int, onebody_matrix_i, onebody_matrix_j):
        if self.include_prefactors:
            prefactor = cexp(-abs(z))
        else:
            prefactor = 1.0
        W = carctanh(csqrt(ctanh(abs(z))))
        arg = W*(2*h-1)
        out = ProductOperator(self.n_particles, self.isospin)
        out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
        out.coefficients[j] = ccosh(arg) * out.coefficients[j] - np.sign(z) * csinh(arg) * onebody_matrix_j @ out.coefficients[j]
        out.coefficients[i] *= csqrt(prefactor)
        out.coefficients[j] *= csqrt(prefactor)
        return out
    

    def threebody_sample(self, z: float, h_list: list, i: int, j: int, k: int, onebody_matrix_i, onebody_matrix_j, onebody_matrix_k):
            N, C, W, A1, A2 = self._a3b_factors(0.5 * self.dt * z)
            if self.include_prefactors:
                prefactor = N*cexp(-h_list[0]*C)
            else:
                prefactor = 1.0
            out = ProductOperator(self.n_particles, self.isospin)
            # one-body factors
            arg = A1 - h_list[0]*W
            out.coefficients[i] = ccosh(arg) * out.coefficients[i] + csinh(arg) * onebody_matrix_i @ out.coefficients[i]
            out.coefficients[j] = ccosh(arg) * out.coefficients[j] + csinh(arg) * onebody_matrix_j @ out.coefficients[j]
            out.coefficients[k] = ccosh(arg) * out.coefficients[k] + csinh(arg) * onebody_matrix_k @ out.coefficients[k]
            # two-body factors
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[1], onebody_matrix_i, onebody_matrix_j))
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[2], onebody_matrix_i, onebody_matrix_k))
            out = out.multiply_operator(self.twobody_sample(-A2, h_list[3], onebody_matrix_j, onebody_matrix_k))
            return out.scale(prefactor)

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
        
        self.linear_spinorbit = False # secret parameter to use the linear approximation of LS instead of the factorization

    def g_pade_sig(self, dt: float, asig: SigmaCoupling, i: int, j: int):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                out += self.sig[i][a].multiply_operator(self.sig[j][b]).scale(asig[a, i, b, j])
        out = out.scale(-0.5 * dt)
        return out.exp()


    def g_pade_sigtau(self, dt: float, asigtau: SigmaTauCoupling, i: int, j: int):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    op = HilbertOperator(self.n_particles, self.isospin)
                    op = op.multiply_operator(self.sig[i][a]).multiply_operator(self.tau[i][c])
                    op = op.multiply_operator(self.sig[j][b]).multiply_operator(self.tau[j][c])
                    out += op.scale(asigtau[a, i, b, j])
        out = out.scale(-0.5 * dt)
        return out.exp()


    def g_pade_tau(self, dt, atau, i, j):
        out = HilbertOperator(self.n_particles, self.isospin).zero()
        for c in range(3):
            out += self.tau[i][c].multiply_operator(self.tau[j][c]).scale(atau[i, j])
        out = out.scale(-0.5 * dt)
        return out.exp()


    def g_pade_coul(self, dt, v, i, j):
        out = self.ident + self.tau[i][2] + self.tau[j][2] + self.tau[i][2].multiply_operator(self.tau[j][2])
        out = out.scale(-0.125 * v[i, j] * dt)
        return out.exp()


    def g_coulomb_onebody(self, dt, v, i):
        """just the one-body part of the expanded coulomb propagator
        for use along with auxiliary field propagators"""
        out =  self.tau[i][2].scale(- 0.125 * v * dt)
        return out.exp()


    def g_ls_linear(self, gls, i):
        # linear approx to LS
        out = HilbertOperator(self.n_particles)
        for a in range(3):
            out = (self.ident - self.sig[i][a].scale(1.j * gls[a, i])).multiply_operator(out) 
        return out
    

    def g_ls_onebody(self, gls, i, a):
        # one-body part of the LS propagator factorization
        out = self.sig[i][a].scale(- 1.j * gls[a,i])
        return out.exp()


    def g_ls_twobody(self, gls, i, j, a, b):
        # two-body part of the LS propagator factorization
        out = self.sig[i][a].multiply_operator(self.sig[j][b]).scale(0.5 * gls[a,i] * gls[b,j])
        return out.exp()


    def g_pade_sig_3b(self, dt, asig3b, i, j, k):
        # 3-body sigma
        out = HilbertOperator(self.n_particles).zero()
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    out += self.sig[i][a].multiply_operator(self.sig[j][b]).multiply_operator(self.sig[k][c]).scale(asig3b[a, i, b, j, c, k])
        out = out.scale(-0.5 * dt)
        return out.exp()


    def make_g_exact(self, dt, potential,
                     sigma,
                     sigmatau,
                     tau,
                     coulomb,
                     spinorbit):
        # compute exact bracket
        g_exact = self.ident.copy()
        pairs_ij = interaction_indices(self.n_particles)
        for i,j in pairs_ij:
            if sigma:
                g_exact = self.g_pade_sig(dt, potential.sigma, i, j).multiply_operator(g_exact)
            if sigmatau:
                g_exact = self.g_pade_sigtau(dt, potential.sigmatau, i, j).multiply_operator(g_exact)
            if tau:
                g_exact = self.g_pade_tau(dt, potential.tau, i, j).multiply_operator(g_exact)
            if coulomb:
                g_exact = self.g_pade_coul(dt, potential.coulomb, i, j).multiply_operator(g_exact)
        if spinorbit:
            if self.linear_spinorbit:
                for i in range(self.n_particles):
                    g_exact = self.g_ls_linear(potential.spinorbit, i) * g_exact
            else:
                for i in range(self.n_particles):
                    for a in range(3):
                        g_exact = self.g_ls_onebody(potential.spinorbit, i, a).multiply_operator(g_exact)
                for i in range(self.n_particles):
                    for j in range(self.n_particles):
                        for a in range(3):
                            for b in range(3):
                                g_exact = self.g_ls_twobody(potential.spinorbit, i, j, a, b).multiply_operator(g_exact)
        return g_exact
    


class Integrator:
    def __init__(self, potential: ArgonnePotential, propagator, isospin=True):
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

        self.sigma = sigma
        self.sigmatau = sigmatau
        self.tau = tau
        self.coulomb = coulomb
        self.spinorbit = spinorbit
        self.mix = mix
        self.parallel = parallel
        self.n_processes = n_processes

        self.rng = np.random.default_rng(seed=seed)
        if self.method=='HS':
            self.aux_fields = self.rng.standard_normal(size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields = - self.aux_fields
        elif self.method=='RBM':
            self.aux_fields = self.rng.integers(0,2,size=(n_samples,n_aux))
            if flip_aux:
                self.aux_fields = np.ones_like(self.aux_fields) - self.aux_fields
        self.is_ready = True

    def bracket(self, bra, ket, aux_fields):
        ket_prop = ket.copy()
        idx = 0
        self.prop_list = []
        if self.sigma:
            self.prop_list.extend( self.propagator.factors_sigma(self.potential, aux_fields[idx : idx + self.propagator.n_aux_sigma] ) )
            idx += self.propagator.n_aux_sigma
        if self.sigmatau:
            self.prop_list.extend( self.propagator.factors_sigmatau(self.potential, aux_fields[idx : idx + self.propagator.n_aux_sigmatau] ) )
            idx += self.propagator.n_aux_sigmatau
        if self.tau:
            self.prop_list.extend( self.propagator.factors_tau(self.potential, aux_fields[idx : idx + self.propagator.n_aux_tau] ) )
            idx += self.propagator.n_aux_tau
        if self.coulomb:
            self.prop_list.extend( self.propagator.factors_coulomb(self.potential, aux_fields[idx : idx + self.propagator.n_aux_coulomb] ) )
            idx += self.propagator.n_aux_coulomb
        if self.spinorbit:
            self.prop_list.extend( self.propagator.factors_spinorbit(self.potential, aux_fields[idx : idx + self.propagator.n_aux_spinorbit] ) )
            idx += self.propagator.n_aux_spinorbit
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
                b_array = pool.starmap_async(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields], leave=True)).get()
        else:
            b_array = list(itertools.starmap(self.bracket, tqdm([(bra, ket, aux) for aux in self.aux_fields])))
        b_array = np.array(b_array).flatten()
        return b_array
            
    def exact(self, bra, ket):
        ex = ExactGFMC(self.n_particles, isospin=self.isospin)
        g_exact = ex.make_g_exact(self.propagator.dt, 
                                  self.potential,
                                  self.sigma,
                                  self.sigmatau,
                                  self.tau,
                                  self.coulomb,
                                  self.spinorbit)
        b_exact = bra.inner(g_exact.multiply_state(ket))
        return b_exact