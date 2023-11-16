# quap
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.3'

# this version has isospin

# CONSTANTS
n_basis_spin = 2  # currently only spin up / spin down

# imports
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng

rng = default_rng()

from scipy.linalg import expm
from numpy.linalg import matrix_power
from scipy.special import spherical_jn, sph_harm

from functools import reduce

# functions

def imat():
    return np.identity(n_basis_spin)

# makes a random spinor
def read_sp(filename):
    """
    special-use function for loading spinor coefficients from a fortran output file
    each line of fortran output is a tuple "(a,b)" indicating a complex-valued coefficient a+ib
    the coefficients typically correspond to spin-isospin states, so each fermion has 4 coefficients
    thus an A-body state will have 4A coefficients
    :param filename: file path to read from
    :return: a numpy.ndarray(dtype=complex)
    """
    def r2c(x):
        y = float(x[0])+1j*float(x[1])
        return y
    l=[]
    with open(filename,'r') as f:
        for line in f:
            l.append(line.strip("()\n").split(','))
    array = np.array(l)
    sp = np.array([r2c(x) for x in array])
    return sp

def prod(l : list):
    """compute the product of items in a list
    this is a convenience function to help write products nicely
    example: s1 * s2 |psi> can be expressed as prod([s1,s2,psi])"""
    lrev = l[::-1]
    out = lrev[0]
    for x in lrev[1:]:
        out = x * out
    return out

def pauli(arg):
    """
    pauli matrices
    :param arg: any of x or 0, y or 1, z or 2, or 'list' which returns the list of all three
    :return: pauli matrix (numpy.ndarray), or list of pauli matrices (list)
    """
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


def spinor(state, orientation):
    """convenience function for making spinors"""
    assert state in ['up', 'down', 'random', 'max']
    assert orientation in ['bra', 'ket']
    if state == 'up':
        sp = np.array([1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1], dtype=complex)
    elif state == 'random':
        sp = rng.uniform(-1, 1, 2) + 1j * rng.uniform(-1, 1, 2)
        sp = sp / np.linalg.norm(sp)
    elif state == 'max':
        sp = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    if orientation == 'ket':
        return sp.reshape((2, 1))
    elif orientation == 'bra':
        return sp.reshape((1, 2))


def repeated_kronecker_product(matrices):
    """
    returns the tensor/kronecker product of a list of arrays
    :param matrices: a list of numpy.ndarray's
    :return: the product numpy.ndarray
    """
    return reduce(np.kron, matrices)


def pmat(x, heatmap=False, lims=None, print_zeros=False):
    """print and/or plot a complex matrix
    heatmat: boolean, plot a heatmap
    lims: boolean, limits on colormap if plotting
    print_zeros: boolean, whether to print either Re/Im parts if all zero"""
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


class CoordinateState(State):
    """
    base class for states with spatial coordinates
    requirements:
     1) integer number of particles, A
     2) has an ``orientation'' -> bra or ket
     3) choice of coordinate space: 1Dradial, etc

     options:
     1Dradial : spherical bessel functions, options = {'n':n}
     3Dspherical: product of spherical bessel function and spherical harmonic, options = {'n':n,'l':l,'m':m}
    """
    def __init__(self, num_particles: int, orientation: str, coordinates: str, options: dict):
        super().__init__(num_particles, orientation)
        if coordinates == '1Dradial':
            self.n = options['n']
            self.psi = lambda r: spherical_jn(self.n, r)
        elif coordinates == '3Dspherical':
            self.n = options['n']
            self.l = options['l']
            self.m = options['m']
            self.psi = lambda r, theta, phi: spherical_jn(self.n, r) * sph_harm(self.m, self.l, theta, phi)
        else:
            raise ValueError(f'Option not supported: {coordinates}')

class SpinState(State):
    """
    base class for all spin states
    requirements:
     1) has an integer number of particles, A
     2) has an ``orientation'' -> bra or ket
     3) has a well-defined number of eigenstates for each particle (e.g. up, down -> 2)
    """

    def __init__(self, num_particles: int, orientation: str, n_basis_spin=n_basis_spin):
        super().__init__(num_particles, orientation)
        assert isinstance(n_basis_spin, int)
        self.n_basis_spin = n_basis_spin

class OneBodyBasisSpinState(SpinState):
    """
    an array of single particle spinors
    
    num_particles: number of single particle states
    n_basis_spin: number of eigenstates in each SPS, i.e. dimension 
    coefficients: list or array of numbers
    orientation: 'bra' or 'ket'
    """

    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray, n_basis_spin=n_basis_spin):
        super().__init__(num_particles, orientation, n_basis_spin=n_basis_spin)

        self.dim = self.num_particles * self.n_basis_spin
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')

    def __add__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return OneBodyBasisSpinState(self.num_particles, self.orientation, self.coefficients)

    def to_list(self):
        if self.orientation == 'ket':
            return [self.coefficients[i*self.n_basis_spin:(i+1)*self.n_basis_spin, 0] for i in range(self.num_particles)]
        elif self.orientation == 'bra':
            return [self.coefficients[0, i*self.n_basis_spin:(i+1)*self.n_basis_spin] for i in range(self.num_particles)]

    def __mul__(self, other):
        """
        target must be another TP state
        if multiplying by another TP state, orientations have to be the opposite (either inner or outer product)
        """
        if isinstance(other, OneBodyBasisSpinState):
            if self.orientation == 'bra':  # inner product
                assert other.orientation == 'ket'
                c0 = self.to_list()
                c1 = other.to_list()
                return np.prod(
                    [np.dot(c0[i], c1[i]) for i in range(self.num_particles)])
            elif self.orientation == 'ket':  # outer product
                assert other.orientation == 'bra'
                out = OneBodyBasisSpinOperator(num_particles=self.num_particles, n_basis_spin=self.n_basis_spin)
                for i in range(self.num_particles):
                    idx_i = i * self.n_basis_spin
                    idx_f = (i + 1) * self.n_basis_spin
                    out.matrix[idx_i:idx_f, idx_i:idx_f] = self.coefficients[idx_i:idx_f, 0:1] @ other.coefficients[0:1, idx_i:idx_f]
                return out
        else:
            raise ValueError('TP state can only multiply another TP state')


    def transpose(self):
        if self.orientation == 'bra':
            out = OneBodyBasisSpinState(self.num_particles, 'ket', self.coefficients.T)
            return out
        elif self.orientation == 'ket':
            out = OneBodyBasisSpinState(self.num_particles, 'bra', self.coefficients.T)
            return out

    def __str__(self):
        out = f"Tensor product {self.orientation} of {self.num_particles} particles: \n"
        for i, ci in enumerate(self.to_list()):
            out += f"{self.orientation} #{i}:\n"
            out += str(ci) + "\n"
        return out

    def to_many_body_state(self):
        """project the NxA TP state into the full N^A MB basis"""
        # if len(self.coefficients) == 1:
        #     sp_mb = self.coefficients
        # else:
        #     sp_mb = np.kron(self.coefficients[0], self.coefficients[1])
        #     for i in range(2, self.num_particles):
        #         sp_mb = np.kron(sp_mb, self.coefficients[i])
        sp_mb = repeated_kronecker_product(self.to_list())
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(self.n_basis_spin ** self.num_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, self.n_basis_spin ** self.num_particles)
        return ManyBodySpinState(self.num_particles, self.orientation, sp_mb)

    def normalize(self):
        if self.orientation == 'ket':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[i:i+self.n_basis_spin, 0])
                self.coefficients[i:i+self.n_basis_spin, 0] /= n
        if self.orientation == 'bra':
            for i in range(self.num_particles):
                n = np.linalg.norm(self.coefficients[0, i:i + self.n_basis_spin])
                self.coefficients[0, i:i + self.n_basis_spin] /= n

    def scalar_mult(self, x, particle_index=0):
        out = self.copy()
        if self.orientation == 'ket':
            out.coefficients[particle_index*self.n_basis_spin:(particle_index+1)*self.n_basis_spin, 0] *= x
        elif self.orientation == 'bra':
            out.coefficients[0, particle_index*self.n_basis_spin:(particle_index+1)*self.n_basis_spin] *= x
        return out

class ManyBodySpinState(SpinState):
    def __init__(self, num_particles: int, orientation: str, coefficients: np.ndarray, n_basis_spin=n_basis_spin):
        super().__init__(num_particles, orientation, n_basis_spin=n_basis_spin)

        self.dim = self.n_basis_spin ** self.num_particles
        assert type(coefficients) == np.ndarray
        ket_condition = (coefficients.shape == (self.dim, 1)) and (orientation == 'ket')
        bra_condition = (coefficients.shape == (1, self.dim)) and (orientation == 'bra')
        if not ket_condition and not bra_condition:
            raise ValueError('Inconsistent initialization of state vector')
        else:
            self.coefficients = coefficients.astype('complex')

    def __add__(self, other):
        return ManyBodySpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return ManyBodySpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return ManyBodySpinState(self.num_particles, self.orientation, self.coefficients)

    def __mul__(self, other):
        if isinstance(other, ManyBodySpinState):
            if self.orientation == 'bra':  # inner product
                assert other.orientation == 'ket'
                out = np.dot(self.coefficients.flatten(), other.coefficients.flatten())
            elif self.orientation == 'ket':  # outer product
                assert other.orientation == 'bra'
                c = np.outer(self.coefficients.flatten(), other.coefficients.flatten())
                out = ManyBodySpinOperator(self.num_particles)
                out.matrix = c
            return out
        else:
            raise TypeError('ManyBodySpinState can only multiply another MB state')

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.coefficients *= other
        else:
            raise TypeError(f'Not supported: {type(other)} * ManyBodySpinState')
        return out

    def transpose(self):
        if self.orientation == 'bra':
            out = ManyBodySpinState(self.num_particles, 'ket', self.coefficients.T)
            return out
        elif self.orientation == 'ket':
            out = ManyBodySpinState(self.num_particles, 'bra', self.coefficients.T)
            return out

    def __str__(self):
        out = [f'Many-body {self.orientation} of {self.num_particles} particles:']
        out += [str(self.coefficients)]
        return "\n".join(out)


# class ProductState(State):
#     """Product of coordinate state and spin state"""
#     def __init__(self, num_particles: int, orientation: str):
#         super().__init__(num_particles, orientation)

class ProductState(State):
    """Product of coordinate state and spin state"""
    def __init__(self, coordinate_state, spin_state):
        self.psi_coord = coordinate_state
        self.psi_spin = spin_state

class SpinOperator:
    """
    base class for spin operators
    """

    def __init__(self, num_particles: int, n_basis_spin=n_basis_spin):
        assert isinstance(num_particles, int)
        self.num_particles = num_particles
        assert isinstance(n_basis_spin, int)
        self.n_basis_spin = n_basis_spin


class OneBodyBasisSpinOperator(SpinOperator):
    def __init__(self, num_particles: int, n_basis_spin=n_basis_spin):
        super().__init__(num_particles, n_basis_spin=n_basis_spin)
        self.dim = num_particles * n_basis_spin
        self.matrix = np.identity(self.dim, dtype='complex')

    def __add__(self, other):
        assert isinstance(other, OneBodyBasisSpinOperator)
        out = OneBodyBasisSpinOperator(self.num_particles)
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, OneBodyBasisSpinOperator)
        out = OneBodyBasisSpinOperator(self.num_particles)
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = OneBodyBasisSpinOperator(self.num_particles)
        out.matrix = self.matrix
        return out

    def __mul__(self, other):
        if isinstance(other, OneBodyBasisSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = self.matrix @ out.coefficients
            return out
        elif isinstance(other, OneBodyBasisSpinOperator):
            out = other.copy()
            out.matrix = self.matrix @ other.matrix
            return out
        else:
            raise ValueError('OneBodyBasisSpinOperator must multiply a OneBodyBasisSpinState, or OneBodyBasisSpinOperator')

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.matrix *= other
            return out
        else:
            raise ValueError('rmul for TPSO not set up')

    # def to_many_body_operator(self):
    #     """project the NxA TP state into the full N^A MB basis"""
    #     # if len(self.coefficients) == 1:
    #     #     sp_mb = self.coefficients
    #     # else:
    #     #     sp_mb = np.kron(self.coefficients[0], self.coefficients[1])
    #     #     for i in range(2, self.num_particles):
    #     #         sp_mb = np.kron(sp_mb, self.coefficients[i])
    #     sp_mb = repeated_kronecker_product(self.coefficients)
    #     out = ManyBodySpinOperator(self.num_particles)
    #     out.coefficients = sp_mb
    #     return out

    def __str__(self):
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        return "Re=\n"+re+"\nIm:\n"+im

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert matrix.shape == (self.n_basis_spin, self.n_basis_spin)
        idx_i = particle_index * self.n_basis_spin
        idx_f = (particle_index + 1) * self.n_basis_spin
        self.matrix[idx_i:idx_f, idx_i:idx_f] = matrix @ self.matrix[idx_i:idx_f, idx_i:idx_f]

    def sigma(self, particle_index, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        self.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return self

    def scalar_mult(self, particle_index, b):
        assert np.isscalar(b)
        self.apply_one_body_operator(particle_index=particle_index, matrix=b * imat())
        return self

    def exchange(self, particle_a: int, particle_b: int):
        idx_ai = particle_a * self.n_basis_spin
        idx_af = (particle_a + 1) * self.n_basis_spin
        idx_bi = particle_b * self.n_basis_spin
        idx_bf = (particle_b + 1) * self.n_basis_spin
        temp = self.matrix.copy()
        self.matrix[idx_ai:idx_af, idx_ai:idx_af] = temp[idx_ai:idx_af, idx_bi:idx_bf]
        self.matrix[idx_ai:idx_af, idx_bi:idx_bf] = temp[idx_ai:idx_af, idx_ai:idx_af]
        self.matrix[idx_bi:idx_bf, idx_bi:idx_bf] = temp[idx_bi:idx_bf, idx_ai:idx_af]
        self.matrix[idx_bi:idx_bf, idx_ai:idx_af] = temp[idx_bi:idx_bf, idx_bi:idx_bf]
        return self


class ManyBodySpinOperator(SpinOperator):
    def __init__(self, num_particles: int, n_basis_spin=n_basis_spin):
        super().__init__(num_particles, n_basis_spin=n_basis_spin)
        self.matrix = np.identity(n_basis_spin ** num_particles)

    def __add__(self, other):
        assert isinstance(other, ManyBodySpinOperator)
        out = ManyBodySpinOperator(self.num_particles)
        out.matrix = self.matrix + other.matrix
        return out

    def __sub__(self, other):
        assert isinstance(other, ManyBodySpinOperator)
        out = ManyBodySpinOperator(self.num_particles)
        out.matrix = self.matrix - other.matrix
        return out

    def copy(self):
        out = ManyBodySpinOperator(self.num_particles)
        out.matrix = self.matrix
        return out

    def __mul__(self, other):
        if isinstance(other, ManyBodySpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = self.matrix @ out.coefficients
            return out
        elif isinstance(other, ManyBodySpinOperator):
            out = other.copy()
            out.matrix = self.matrix @ other.matrix
            return out
        else:
            raise ValueError('ManyBodySpinOperator must multiply a ManyBodySpinState, or ManyBodySpinOperator')

    def __rmul__(self, other):
        if np.isscalar(other):
            out = self.copy()
            out.matrix = other * out.matrix
        else:
            raise ValueError('Type not supported in __rmul__:', type(other))
        return out

    def __str__(self):
        re = str(np.real(self.matrix))
        im = str(np.imag(self.matrix))
        return "Re=\n"+re+"\nIm:\n"+im

    def apply_one_body_operator(self, particle_index: int, matrix: np.ndarray):
        assert type(matrix) == np.ndarray and matrix.shape == (self.n_basis_spin, self.n_basis_spin)
        obo = [imat() for _ in range(self.num_particles)]
        obo[particle_index] = matrix
        obo = repeated_kronecker_product(obo)
        self.matrix = obo @ self.matrix

    def sigma(self, particle_index: int, dimension):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        self.apply_one_body_operator(particle_index=particle_index, matrix=pauli(dimension))
        return self

    def scalar_mult(self, particle_index: int, b):
        """
        multiply one particle by a scalar.
        To multiply the whole state by a scalar, just do b * ManyBodySpinState.
        :param particle_index:
        :param b:
        :return:
        """
        assert np.isscalar(b)
        self.apply_one_body_operator(particle_index=particle_index, matrix=b * imat())
        return self

    def exchange(self, particle_a: int, particle_b: int):
        P_1 = ManyBodySpinOperator(num_particles=self.num_particles, n_basis_spin=self.n_basis_spin)
        P_x = P_1.copy().sigma(particle_a, 'x').sigma(particle_b, 'x')
        P_y = P_1.copy().sigma(particle_a, 'y').sigma(particle_b, 'y')
        P_z = P_1.copy().sigma(particle_a, 'z').sigma(particle_b, 'z')
        P = (P_x + P_y + P_z + P_1)
        return 0.5 * P * self

    def exponentiate(self):
        out = self.copy()
        out.matrix = expm(out.matrix)
        return out

    def zeros(self):
        out = self.copy()
        out.matrix = np.zeros_like(out.matrix)
        return out

def random_bra_ket():
    coeffs_ket = np.concatenate([spinor('random', 'ket'), spinor('random', 'ket')], axis=0)
    coeffs_bra = np.concatenate([spinor('random', 'bra'), spinor('random', 'bra')], axis=1)
    ket = OneBodyBasisSpinState(2, 'ket', coeffs_ket)
    bra = OneBodyBasisSpinState(2, 'bra', coeffs_bra)
    return bra, ket


def test_states():
    print('TESTING STATES')
    print('INITIALIZING TENSOR PRODUCTS')
    coeffs_uu = np.concatenate([spinor('up', 'ket'), spinor('up', 'ket')], axis=0)
    coeffs_ud = np.concatenate([spinor('up', 'ket'), spinor('down', 'ket')], axis=0)
    sp_uu = OneBodyBasisSpinState(2, 'ket', coeffs_uu)
    sp_ud = OneBodyBasisSpinState(2, 'ket', coeffs_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('INNER PRODUCTS')
    print("<uu|uu> = \n", sp_uu.transpose() * sp_uu)
    print("<ud|ud> = \n", sp_ud.transpose() * sp_ud)
    print("<uu|ud> = \n", sp_uu.transpose() * sp_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.transpose())
    print("|ud><ud| = \n", sp_ud * sp_ud.transpose())
    print("|uu><ud| = \n", sp_uu * sp_ud.transpose())
    print('TO MANYBODY')
    sp_uu = sp_uu.to_many_body_state()
    sp_ud = sp_ud.to_many_body_state()
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('INNER PRODUCTS')
    print("<uu|uu> = \n", sp_uu.transpose() * sp_uu)
    print("<ud|ud> = \n", sp_ud.transpose() * sp_ud)
    print("<uu|ud> = \n", sp_uu.transpose() * sp_ud)
    print('OUTER PRODUCTS')
    print("|uu><uu| = \n", sp_uu * sp_uu.transpose())
    print("|ud><ud| = \n", sp_ud * sp_ud.transpose())
    print("|uu><ud| = \n", sp_uu * sp_ud.transpose())

    print('RANDOM TENSOR PRODUCTS')
    coeffs_0 = np.concatenate([spinor('random', 'ket'), spinor('random', 'ket')], axis=0)
    coeffs_1 = np.concatenate([spinor('random', 'ket'), spinor('random', 'ket')], axis=0)
    sp_0 = OneBodyBasisSpinState(2, 'ket', coeffs_0)
    sp_1 = OneBodyBasisSpinState(2, 'ket', coeffs_1)
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.transpose() * sp_0)
    print("<1|1> = \n", sp_1.transpose() * sp_1)
    print("<0|1> = \n", sp_0.transpose() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.transpose())
    print("|1><1| = \n", sp_1 * sp_1.transpose())
    print("|0><1| = \n", sp_0 * sp_1.transpose())
    print('TO MANYBODY')
    sp_0 = sp_0.to_many_body_state()
    sp_1 = sp_1.to_many_body_state()
    print("|0> = \n", sp_0)
    print("|1> = \n", sp_1)
    print('INNER PRODUCTS')
    print("<0|0> = \n", sp_0.transpose() * sp_0)
    print("<1|1> = \n", sp_1.transpose() * sp_1)
    print("<0|1> = \n", sp_0.transpose() * sp_1)
    print('OUTER PRODUCTS')
    print("|0><0| = \n", sp_0 * sp_0.transpose())
    print("|1><1| = \n", sp_1 * sp_1.transpose())
    print("|0><1| = \n", sp_0 * sp_1.transpose())
    print('DONE TESTING STATES')


def test_operators():
    print('TESTING OPERATORS')
    print('TENSOR PRODUCT STATES')
    coeffs_uu = np.concatenate([spinor('up', 'ket'), spinor('up', 'ket')], axis=0)
    coeffs_ud = np.concatenate([spinor('up', 'ket'), spinor('down', 'ket')], axis=0)
    sp_uu = OneBodyBasisSpinState(2, 'ket', coeffs_uu)
    sp_ud = OneBodyBasisSpinState(2, 'ket', coeffs_ud)
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('TENSOR PRODUCT OPERATORS')
    sigx0 = OneBodyBasisSpinOperator(2).sigma(0, 'x')
    sigy0 = OneBodyBasisSpinOperator(2).sigma(0, 'y')
    sigz0 = OneBodyBasisSpinOperator(2).sigma(0, 'z')
    print("sigx0 = \n", sigx0)
    print("sigy0 = \n", sigy0)
    print("sigz0 = \n", sigz0)
    print("sigx0 |uu> = \n", sigx0 * sp_uu)
    print("sigy0 |uu> = \n", sigy0 * sp_uu)
    print("sigz0 |uu> = \n", sigz0 * sp_uu)
    print("sigx0 |ud> = \n", sigx0 * sp_ud)
    print("sigy0 |ud> = \n", sigy0 * sp_ud)
    print("sigz0 |ud> = \n", sigz0 * sp_ud)
    print('EXCHANGE P(i,j)')
    P01 = OneBodyBasisSpinOperator(2).exchange(0, 1)
    print('P(0,1) = \n', P01)
    print("|ud> = \n", sp_ud)
    print("P(0,1) |ud> = \n", P01 * sp_ud)
    print("SCALAR MULTIPLICATION")
    five0 = OneBodyBasisSpinOperator(2).scalar_mult(0, 5)
    three1 = OneBodyBasisSpinOperator(2).scalar_mult(1, 3)
    print("5(0) = \n", five0)
    print("3(1) = \n", three1)
    print("5(0) |uu> = \n", five0 * sp_uu)
    print("3(1) |ud> = \n", three1 * sp_ud)

    print('MANYBODY STATES')
    sp_uu = sp_uu.to_many_body_state()
    sp_ud = sp_ud.to_many_body_state()
    print("|uu> = \n", sp_uu)
    print("|ud> = \n", sp_ud)
    print('MANYBODY OPERATORS')
    sigx0 = ManyBodySpinOperator(2).sigma(0, 'x')
    sigy0 = ManyBodySpinOperator(2).sigma(0, 'y')
    sigz0 = ManyBodySpinOperator(2).sigma(0, 'z')
    print("sigx0 = \n", sigx0)
    print("sigy0 = \n", sigy0)
    print("sigz0 = \n", sigz0)
    print("sigx0 |uu> = \n", sigx0 * sp_uu)
    print("sigy0 |uu> = \n", sigy0 * sp_uu)
    print("sigz0 |uu> = \n", sigz0 * sp_uu)
    print("sigx0 |ud> = \n", sigx0 * sp_ud)
    print("sigy0 |ud> = \n", sigy0 * sp_ud)
    print("sigz0 |ud> = \n", sigz0 * sp_ud)
    print('EXCHANGE P(i,j)')
    P01 = ManyBodySpinOperator(2).exchange(0, 1)
    print('P(0,1) = \n', P01)
    print("|ud> = \n", sp_ud)
    print("P(0,1) |ud> = \n", P01 * sp_ud)
    print("SCALAR MULTIPLICATION")
    five0 = ManyBodySpinOperator(2).scalar_mult(0, 5)
    three1 = ManyBodySpinOperator(2).scalar_mult(1, 3)
    print("5(0) = \n", five0)
    print("3(1) = \n", three1)
    print("5(0) |uu> = \n", five0 * sp_uu)
    print("3(1) |ud> = \n", three1 * sp_ud)

    print('TENSOR PRODUCT TEST: sigma(i) dot sigma(j) = 2P(i,j) - 1')
    bra, ket = random_bra_ket()
    sigx01 = OneBodyBasisSpinOperator(2).sigma(0, 'x').sigma(1, 'x')
    sigy01 = OneBodyBasisSpinOperator(2).sigma(0, 'y').sigma(1, 'y')
    sigz01 = OneBodyBasisSpinOperator(2).sigma(0, 'z').sigma(1, 'z')
    P01 = OneBodyBasisSpinOperator(2).exchange(0, 1)
    lhs = bra * (sigx01 * ket) + bra * (sigy01 * ket) + bra * (sigz01 * ket)
    rhs = 2 * (bra * (P01 * ket)) - bra * ket
    print("sigma(i) dot sigma(j) = \n", lhs)
    print("2P(i,j) - 1 \n", rhs)

    print('MANY BODY TEST: sigma(i) dot sigma(j) = 2P(i,j) - 1')
    bra, ket = random_bra_ket()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    sigx01 = ManyBodySpinOperator(2).sigma(0, 'x').sigma(1, 'x')
    sigy01 = ManyBodySpinOperator(2).sigma(0, 'y').sigma(1, 'y')
    sigz01 = ManyBodySpinOperator(2).sigma(0, 'z').sigma(1, 'z')
    P01 = ManyBodySpinOperator(2).exchange(0, 1)
    lhs = prod([bra, sigx01, ket]) + prod([bra, sigy01, ket]) + prod([bra, sigz01, ket])
    rhs = 2 * (bra * (P01 * ket)) - bra * ket
    print("sigma(i) dot sigma(j) = \n", lhs)
    print("2P(i,j) - 1 \n", rhs)

    print('DONE TESTING OPERATORS')



def test_propagator_easy():
    print('TESTING PROPAGATOR (EASY)')
    # bra, ket = random_bra_ket()
    coeffs_bra = np.concatenate([spinor('max', 'bra'), spinor('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor('max', 'ket'), spinor('max', 'ket')], axis=0)
    bra = OneBodyBasisSpinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinState(2, 'ket', coeffs_ket).to_many_body_state()
    dt = 0.01
    A = 1.0
    Sx = ManyBodySpinOperator(2).sigma(0, 'x').sigma(1, 'x')
    Gx = (-dt * A * Sx).exponentiate()
    print("Gx = \n", Gx)
    print("<s| = \n", bra)
    print("|s> = \n", ket)
    print("<s|Gx|s> = \n", prod([bra, Gx, ket]))

def test_product_states():
    import matplotlib.pyplot as plt
    state_x = CoordinateState(2,'ket','1Dradial',{'n':5})
    domain = np.linspace(0,3)
    y = state_x.psi(domain)
    plt.plot(domain,y)
    plt.show()



if __name__ == "__main__":
    print('BEGIN MODULE TEST')
    # test_states()
    # test_operators()
    # test_propagator_easy()
    # test_product_states()
    x = read_sp('data\sp.dat')
    print(x)

    print('MODULE TEST COMPLETE')
