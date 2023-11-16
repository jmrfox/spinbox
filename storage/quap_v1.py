# quap
# a quantum mechanics playground
# jordan fox 2023

__version__ = '0.1'

# dev diary
# 9 / 12
# MB and TP state classes done
# don't put more than is necessary in parent classes (init, str)
# subclasses are basis specific
# operators -> TP operator, MB operator
# basis generally defines how multiplication works
# States need modifications:
# 1) should be explicitly BRA or KET (method returning row/col vec?)
# 2) do not deduce operations, if user writes them wrong they must fail
# TP states should be lists or numpy arrays, not arrays themselves


# CONSTANTS
NUM_BASIS = 2  # currently only spin up / spin down

# imports
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng

rng = default_rng(seed=1312)

from scipy.linalg import expm
from numpy.linalg import matrix_power

from functools import reduce


# my functions

def imat():
    return np.identity(NUM_BASIS)


def pauli(arg):
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
    assert state in ['up', 'down', 'random']
    assert orientation in ['bra', 'ket']
    if state == 'up':
        sp = np.array([1, 0], dtype=complex)
    elif state == 'down':
        sp = np.array([0, 1], dtype=complex)
    elif state == 'random':
        sp = rng.uniform(-1, 1, 2) + 1j * rng.uniform(-1, 1, 2)
        sp = sp / np.linalg.norm(sp)
    if orientation == 'ket':
        return sp.reshape((2, 1))
    elif orientation == 'bra':
        return sp.reshape((1, 2))


def repeated_kronecker_product(matrices):
    """
    returns the tensor/kronecker product of a list of arrays
    :param matrices:
    :return:
    """
    return reduce(np.kron, matrices)


def pmat(x, heatmap=False, lims=None, print_zeros=False):
    """print and/or plot a complex matrix
    heatmat: plot a heatmap
    lims: if plotting, limits on colormap
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

class SpinState:
    """
    base class for all states
    requirements for all states:
     1) has an integer number of particles, A
     2) has a well-defined number of eigenstates for each particle (e.g. up, down -> 2)
     3) has an ``orientation'' -> bra or ket
    """

    def __init__(self, num_particles: int, orientation: str, num_basis=NUM_BASIS):
        assert isinstance(num_particles, int)
        self.num_particles = num_particles
        assert isinstance(num_basis, int)
        self.num_basis = num_basis
        assert orientation in ['bra', 'ket']
        self.orientation = orientation


class TensorProductSpinState(SpinState):
    """
    tensor product of single particle states
    
    num_particles: number of single particle states in tensor product
    num_basis: number of eigenstates in each SPS, i.e. dimension 
    coefficients: list or array of numbers
    orientation: 'bra' or 'ket' specifying orientation
    """

    def __init__(self, num_particles: int, orientation: str, coefficients: list, num_basis=NUM_BASIS):
        super().__init__(num_particles, orientation, num_basis=num_basis)

        try:
            c = [np.array(x, dtype=complex) for x in coefficients]
        except:
            raise ValueError('Could not convert coefficients to a list of complex arrays')

        assert len(coefficients) == num_particles
        if orientation == 'bra':
            for ci in c:
                assert ci.shape == (1, num_basis)
        elif orientation == 'ket':
            for ci in c:
                assert ci.shape == (num_basis, 1)

        self.coefficients = c

    def __add__(self, other):
        return TensorProductSpinState(self.num_particles, self.orientation, self.coefficients + other.coefficients)

    def __sub__(self, other):
        return TensorProductSpinState(self.num_particles, self.orientation, self.coefficients - other.coefficients)

    def copy(self):
        return TensorProductSpinState(self.num_particles, self.orientation, self.coefficients)

    def __mul__(self, other):
        """
        target must be another TP state
        no scalar multiplication is defined! to multiply by scalar b, define operator bI_i and multiply with that.
        if multiplying by another TP state, orientations have to be the opposite (either inner or outer product)
        """
        if isinstance(other, TensorProductSpinState):
            if self.orientation == 'bra':  # inner product
                assert other.orientation == 'ket'
                return np.prod(
                    [np.dot(self.coefficients[i], other.coefficients[i]) for i in range(self.num_particles)])
            elif self.orientation == 'ket':  # outer product
                assert other.orientation == 'bra'
                c = [np.outer(self.coefficients[i], other.coefficients[i]) for i in range(self.num_particles)]
                out = TensorProductSpinOperator(self.num_particles, num_basis=self.num_basis)
                out.coefficients = c
                return out
        else:
            raise ValueError('TP state can only multiply another TP state')

    def transpose(self):
        if self.orientation == 'bra':
            out = TensorProductSpinState(self.num_particles, 'ket', [ci.T for ci in self.coefficients])
            return out
        elif self.orientation == 'ket':
            out = TensorProductSpinState(self.num_particles, 'bra', [ci.T for ci in self.coefficients])
            return out

    def __str__(self):
        out = f"Tensor product {self.orientation} of {self.num_particles} particles: \n"
        for i, ci in enumerate(self.coefficients):
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
        sp_mb = repeated_kronecker_product(self.coefficients)
        if self.orientation == 'ket':
            sp_mb = sp_mb.reshape(self.num_basis ** self.num_particles, 1)
        elif self.orientation == 'bra':
            sp_mb = sp_mb.reshape(1, self.num_basis ** self.num_particles)
        return ManyBodySpinState(self.num_particles, self.orientation, sp_mb)

    def normalize(self):
        for i in range(self.num_particles):
            self.coefficients[i] = self.coefficients[i] / np.linalg.norm(self.coefficients[i])

    def randomize(self):
        out = self.copy()
        for i, ci in enumerate(out.coefficients):
            out.coefficients[i] = rng.uniform(-1, 1, ci.shape)
        out.normalize()
        return out

    def exchange(self, i, j):
        out = self.copy()
        out.coefficients[i] = self.coefficients[j]
        out.coefficients[j] = self.coefficients[i]
        return out

class ManyBodySpinState(SpinState):
    def __init__(self, num_particles: int, orientation: str, coefficients, num_basis=NUM_BASIS):
        super().__init__(num_particles, orientation, num_basis=num_basis)

        try:
            c = np.array(coefficients, dtype=complex)
        except:
            raise ValueError('Could not convert coefficients to a complex array')

        self.dim = num_basis ** num_particles
        if orientation == 'bra':
            assert c.shape == (1, self.dim)
        elif orientation == 'ket':
            assert c.shape == (self.dim, 1)

        self.coefficients = c

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
                out.coefficients = c
            return out
        else:
            raise TypeError('ManyBodySpinState can only multiply another MB state')

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


class SpinOperator:
    """
    base class for spin operators
    """

    def __init__(self, num_particles: int, num_basis=NUM_BASIS):
        assert isinstance(num_particles, int)
        self.num_particles = num_particles
        assert isinstance(num_basis, int)
        self.num_basis = num_basis


class TensorProductSpinOperator(SpinOperator):
    def __init__(self, num_particles: int, num_basis=NUM_BASIS):
        super().__init__(num_particles, num_basis=num_basis)
        self.coefficients = [np.identity(num_basis) for _ in range(num_particles)]

    def __add__(self, other):
        assert isinstance(other, TensorProductSpinOperator)
        out = TensorProductSpinOperator(self.num_particles)
        for i in range(self.num_particles):
            out.coefficients[i] = self.coefficients[i] + other.coefficients[i]
        return out

    def __sub__(self, other):
        assert isinstance(other, TensorProductSpinOperator)
        out = TensorProductSpinOperator(self.num_particles)
        for i in range(self.num_particles):
            out.coefficients[i] = self.coefficients[i] - other.coefficients[i]
        return out

    def copy(self):
        out = TensorProductSpinOperator(self.num_particles)
        for i in range(self.num_particles):
            out.coefficients[i] = self.coefficients[i]
        return out

    def __mul__(self, other):
        if isinstance(other, TensorProductSpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            for i in range(self.num_particles):
                out.coefficients[i] = self.coefficients[i] @ out.coefficients[i]
            return out
        elif isinstance(other, TensorProductSpinOperator):
            out = other.copy()
            for i in range(self.num_particles):
                out.coefficients[i] = self.coefficients[i] @ other.coefficients[i]
            return out
        else:
            raise ValueError('TensorProductSpinOperator must multiply a TensorProductSpinState, or TensorProductSpinOperator')

    def to_many_body_operator(self):
        """project the NxA TP state into the full N^A MB basis"""
        # if len(self.coefficients) == 1:
        #     sp_mb = self.coefficients
        # else:
        #     sp_mb = np.kron(self.coefficients[0], self.coefficients[1])
        #     for i in range(2, self.num_particles):
        #         sp_mb = np.kron(sp_mb, self.coefficients[i])
        sp_mb = repeated_kronecker_product(self.coefficients)
        out = ManyBodySpinOperator(self.num_particles)
        out.coefficients = sp_mb
        return out

    def __str__(self):
        out = f"Tensor product operator: \n"
        for i, ci in enumerate(self.coefficients):
            out += f"Op #{i}:\n"
            out += str(ci) + "\n"
        return out

    def sigma(self, dimension, particle_index):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = TensorProductSpinOperator(self.num_particles)
        out.coefficients[particle_index] = pauli(dimension) @ out.coefficients[particle_index]
        return out

    def scalar_mult(self, particle_index, b):
        out = TensorProductSpinOperator(self.num_particles)
        out.coefficients[particle_index] = b * out.coefficients[particle_index]
        return out


class ManyBodySpinOperator(SpinOperator):
    def __init__(self, num_particles: int, num_basis=NUM_BASIS):
        super().__init__(num_particles, num_basis=num_basis)
        self.coefficients = np.identity(num_basis ** num_particles)

    def __add__(self, other):
        assert isinstance(other, ManyBodySpinOperator)
        out = ManyBodySpinOperator(self.num_particles)
        out.coefficients = self.coefficients + other.coefficients
        return out

    def __sub__(self, other):
        assert isinstance(other, ManyBodySpinOperator)
        out = ManyBodySpinOperator(self.num_particles)
        out.coefficients = self.coefficients - other.coefficients
        return out

    def copy(self):
        out = ManyBodySpinOperator(self.num_particles)
        out.coefficients = self.coefficients
        return out

    def __mul__(self, other):
        if isinstance(other, ManyBodySpinState):
            assert other.orientation == 'ket'
            out = other.copy()
            out.coefficients = self.coefficients @ out.coefficients
            return out
        elif isinstance(other, ManyBodySpinOperator):
            out = other.copy()
            out.coefficients = self.coefficients @ other.coefficients
            return out
        else:
            raise ValueError('ManyBodySpinOperator must multiply a ManyBodySpinState, or ManyBodySpinOperator')

    def __str__(self):
        return str(self.coefficients)

    def sigma(self, dimension, particle_index):
        assert (dimension in ['x', 'y', 'z']) or (dimension in [0, 1, 2])
        out = self.copy()
        matrices = [imat()]*self.num_particles
        matrices[particle_index] = pauli(dimension)
        out.coefficients = repeated_kronecker_product(matrices) @ self.coefficients
        return out

    def scalar_mult(self, b):
        out = self.coefficients = b * self.coefficients


if __name__ == "__main__":
    test_states = False
    test_operators = True

    print('BEGIN MODULE TEST')
    if test_states:
        print('INITIALIZING TENSOR PRODUCTS')
        sp_uu = TensorProductSpinState(2, 'ket', [spinor('up', 'ket'), spinor('up', 'ket')])
        sp_ud = TensorProductSpinState(2, 'ket', [spinor('up', 'ket'), spinor('down', 'ket')])
        print("|uu> = \n", sp_uu)
        print("|ud> = \n", sp_ud)
        print('INNER PRODUCTS')
        print("<uu|uu> = \n", sp_uu.transpose() * sp_uu)
        print("<1|1> = \n", sp_ud.transpose() * sp_ud)
        print("<uu|ud> = \n", sp_uu.transpose() * sp_ud)
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
        sp_0 = TensorProductSpinState(2, 'ket', [spinor('random', 'ket'), spinor('random', 'ket')])
        sp_1 = TensorProductSpinState(2, 'ket', [spinor('random', 'ket'), spinor('random', 'ket')])
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

    if test_operators:
        # print("pauli x = \n",pauli('x'))
        # print("pauli y = \n",pauli('y'))
        # print("pauli z = \n",pauli('z'))
        # print("pauli x * pauli y = \n",pauli('x') @ pauli('y'))
        print('TENSOR PRODUCT STATES')
        sp_uu = TensorProductSpinState(2, 'ket', [spinor('up', 'ket'), spinor('up', 'ket')])
        sp_ud = TensorProductSpinState(2, 'ket', [spinor('up', 'ket'), spinor('down', 'ket')])
        print("|uu> = \n", sp_uu)
        print("|ud> = \n", sp_ud)
        print('TENSOR PRODUCT OPERATORS')
        sigx0 = TensorProductSpinOperator(2).sigma('x', 0)
        sigy0 = TensorProductSpinOperator(2).sigma('y', 0)
        sigz0 = TensorProductSpinOperator(2).sigma('z', 0)
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
        print("|ud> = \n", sp_ud)
        print("P(0,1) |ud> = \n", sp_ud.exchange(0, 1))
        print("SCALAR MULTIPLICATION")
        five0 = TensorProductSpinOperator(2).scalar_mult(0, 5)
        three1 = TensorProductSpinOperator(2).scalar_mult(1, 3)
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
        sigx0 = ManyBodySpinOperator(2).sigma('x', 0)
        sigy0 = ManyBodySpinOperator(2).sigma('y', 0)
        sigz0 = ManyBodySpinOperator(2).sigma('z', 0)
        print("sigx0 = \n", sigx0)
        print("sigy0 = \n", sigy0)
        print("sigz0 = \n", sigz0)
        print("sigx0 |uu> = \n", sigx0 * sp_uu)
        print("sigy0 |uu> = \n", sigy0 * sp_uu)
        print("sigz0 |uu> = \n", sigz0 * sp_uu)
        print("sigx0 |ud> = \n", sigx0 * sp_ud)
        print("sigy0 |ud> = \n", sigy0 * sp_ud)
        print("sigz0 |ud> = \n", sigz0 * sp_ud)

        print('sigma(i) dot sigma(j) = 2P(i,j) - 1')
        ket = TensorProductSpinState(2, 'ket', [spinor('random', 'ket'), spinor('random', 'ket')])
        bra = TensorProductSpinState(2, 'bra', [spinor('random', 'bra'), spinor('random', 'bra')])
        sigx01 = TensorProductSpinOperator(2).sigma('x', 0).sigma('x', 1)
        sigy01 = TensorProductSpinOperator(2).sigma('y', 0).sigma('y', 1)
        sigz01 = TensorProductSpinOperator(2).sigma('z', 0).sigma('z', 1)
        lhs = bra*(sigx01*ket) + bra*(sigy01*ket) + bra*(sigz01*ket)
        rhs = 2*(bra*(ket.exchange(0, 1))) - bra*ket
        print("sigma(i) dot sigma(j) = \n", lhs)
        print("2P(i,j) - 1 \n", rhs)

    print('TEST COMPLETE')
