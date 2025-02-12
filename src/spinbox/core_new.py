# spinbox
# tools for many-body spin systems in a Monte Carlo context
# jordan m r fox 2024

__version__ = "0.1.0"

import sys
import numpy as np
np.set_printoptions(linewidth=200, threshold=sys.maxsize)
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
from functools import reduce
import itertools
from multiprocessing.pool import Pool
from tqdm import tqdm

def csqrt(x):
    """Complex square root

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex square root of x
    :rtype: complex or np.ndarray
    """
    return np.sqrt(x, dtype=complex)


def ccos(x):
    """Complex cosine

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex cosine of x
    :rtype: complex or np.ndarray
    """
    return np.cos(x, dtype=complex)


def csin(x):
    """Complex sine

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex sine of x
    :rtype: complex or np.ndarray
    """
    return np.sin(x, dtype=complex)


def cexp(x):
    """Complex exponential

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex exponential of x
    :rtype: complex or np.ndarray
    """
    return np.exp(x, dtype=complex)


def ccosh(x):
    """Complex hyperbolic cosine

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex hyperbolic cosine of x
    :rtype: complex or np.ndarray
    """
    return np.cosh(x, dtype=complex)


def csinh(x):
    """Complex hyperbolic sine

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex hyperbolic sine of x
    :rtype: complex or np.ndarray
    """
    return np.sinh(x, dtype=complex)


def ctanh(x):
    """Complex hyperbolic tangent

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex hyperbolic tangent of x
    :rtype: complex or np.ndarray
    """
    return np.tanh(x, dtype=complex)


def carctanh(x):
    """Complex inverse hyperbolic tangent

    :param x: Input value
    :type x: float or np.ndarray
    :return: Complex inverse hyperbolic tangent of x
    :rtype: complex or np.ndarray
    """
    return np.arctanh(x, dtype=complex)


def interaction_indices(n: int, m=2) -> list:
    """Returns a list of all possible m-plets of n objects (labelled 0 to n-1)

    :param n: Number of items
    :type n: int
    :param m: Size of tuplet, defaults to 2
    :type m: int, optional
    :return: List of possible m-plets of n items
    :rtype: list
    """
    if m == 1:
        return np.arange(n)
    else:
        return np.array(list(itertools.combinations(range(n), m)))


def read_from_file(filename: str, complex=False, shape=None, order="F") -> np.ndarray:
    """Read numbers from a text file

    :param filename: Input file name
    :type filename: str
    :param complex: Complex entries, defaults to False
    :type complex: bool, optional
    :param shape: Shape of output array, defaults to None
    :type shape: tuple, optional
    :param order: 'F' for columns first, otherwise use 'C', defaults to 'F'
    :type order: str, optional
    :return: Numpy array
    :rtype: np.ndarray
    """

    def tuple_to_complex(x):
        y = float(x[0]) + 1j * float(x[1])
        return y

    c = np.loadtxt(filename)
    if complex:
        sp = np.array([tuple_to_complex(x) for x in c], dtype="complex")
    else:
        sp = np.array(c)

    if shape is not None:
        sp = sp.reshape(shape, order=order)

    return sp


def pauli(arg) -> np.ndarray:
    """Pauli matrix x, y, z, or a list of all three

    :param arg: 0 or 'x' for Pauli x, 1 or 'y' for Pauli y, 2 or 'z' for Pauli z, 'list' for a list of x, y ,z
    :type arg: int or str
    :raises ValueError: Option not found
    :return: Pauli matrix or list
    :rtype: np.ndarray
    """
    if arg in [0, "x"]:
        out = np.array([[0, 1], [1, 0]], dtype=complex)
    elif arg in [1, "y"]:
        out = np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif arg in [2, "z"]:
        out = np.array([[1, 0], [0, -1]], dtype=complex)
    elif arg in [3, "i"]:
        out = np.array([[1, 0], [0, 1]], dtype=complex)
    elif arg in ["list"]:
        out = [
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ]
    else:
        raise ValueError(f"No option: {arg}")
    return out


# def rlprod(factors:list):
#     """Multiplies a list of factors (left-to-right) by associating pairs right-to-left
#     e.g.      ABCx = A(B(Cx))

#     Args:
#         factors (list): things to be multiplied

#     """
#     out = factors[-1]
#     for x in reversed(factors[:-1]):
#         out = x * out
#     return out


def kronecker_product(matrices: list) -> np.ndarray:
    """Returns the Kronecker (i.e. tensor) product of a list of arrays

    :param matrices: List of matrix factors
    :type matrices: list
    :return: Kronecker product of input list
    :rtype: np.ndarray
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
    ``ketwise`` determines if it is a bra or a ket.
    """

    def __init__(self, n_particles: int, coefficients=None, ketwise=True, isospin=True):
        """Instantiate a new ``HilbertState``

        :param n_particles: Number of particles
        :type n_particles: int
        :param coefficients: An optional array of coefficients, defaults to None
        :type coefficients: np.ndarray, optional
        :param ketwise: True for column vector, False for row vector, defaults to True
        :type ketwise: bool, optional
        :param isospin: True for spin-isospin state, False for spin only, defaults to True
        :type isospin: bool, optional
        :raises ValueError: Inconsistency in chosen options
        """
        self.n_particles = n_particles
        self.isospin = isospin
        self.n_basis = 2 + 2 * isospin
        self.dimension = self.n_basis**self.n_particles
        self.ketwise = ketwise

        if coefficients is None:
            if ketwise:
                self.coefficients = np.zeros(shape=(self.dimension, 1))
            else:
                self.coefficients = np.zeros(shape=(1, self.dimension))
        else:
            ket_condition = (coefficients.shape == (self.dimension, 1)) and ketwise
            bra_condition = (coefficients.shape == (1, self.dimension)) and not ketwise
            if not ket_condition and not bra_condition:
                raise ValueError(
                    "Inconsistent initialization of state vector. \n\
                                Did you get the shape right?"
                )
            else:
                self.coefficients = coefficients.astype("complex")

    def copy(self):
        """Copies the ``HilbertState``.

        :return: A new instance of ``HilbertState`` with all the same properties.
        :rtype: HilbertState
        """
        return HilbertState(
            n_particles=self.n_particles,
            coefficients=self.coefficients.copy(),
            ketwise=self.ketwise,
            isospin=self.isospin,
        )

    def __add__(self, other: "HilbertState") -> "HilbertState":
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

    def __sub__(self, other: "HilbertState") -> "HilbertState":
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

    def scale(self, other: complex) -> "HilbertState":
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

    def inner(self, other: "HilbertState") -> complex:
        """Inner product of two HilbertState instances. Orientations must be correct.

        :param other: The ket of the inner product.
        :type other: HilbertState
        :return: Inner product of self (bra) with other (ket)
        :rtype: complex
        """
        assert isinstance(other, HilbertState)
        assert not self.ketwise and other.ketwise
        return np.dot(self.coefficients, other.coefficients)

    def outer(self, other: "HilbertState") -> "HilbertOperator":
        """Outer product of two HilbertState instances, producing a HilbertOperator instance. Orientations must be correct.

        :param other: Bra part of the outer product
        :type other: HilbertState
        :return: Outer product of self (ket) with other (bra)
        :rtype: HilbertOperator
        """
        assert isinstance(other, HilbertState)
        assert self.ketwise and not other.ketwise
        out = HilbertOperator(n_particles=self.n_particles, isospin=self.isospin)
        out.coefficients = np.matmul(
            self.coefficients, other.coefficients, dtype="complex"
        )
        return out

    def multiply_operator(self, other: "HilbertOperator") -> "HilbertState":
        """Multiplies a (bra) ``HilbertState`` on a ``HilbertOperator``.

        :param other: The operator.
        :type other: HilbertOperator
        :return: < self| O(other)
        :rtype: HilbertState
        """
        assert isinstance(other, HilbertOperator)
        assert not self.ketwise
        out = self.copy()
        out.coefficients = np.matmul(
            self.coefficients, other.coefficients, dtype="complex"
        )
        return out

    def dagger(self) -> "HilbertState":
        """Hermitian conjugate.

        :return: The dual ``HilbertState``
        :rtype: HilbertState
        """
        out = self.copy()
        out.coefficients = self.coefficients.conj().T
        out.ketwise = not self.ketwise
        return out

    def __str__(self):
        orient = "ket"
        if not self.ketwise:
            orient = "bra"
        out = [f"{self.__class__.__name__} {orient} of {self.n_particles} particles:"]
        out += [str(self.coefficients)]
        return "\n".join(out)

    def random(self, seed: int = None) -> "HilbertState":
        """Random coefficients.

        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :return: A copy of the ``HilbertState`` with random complex coefficients, normalized.
        :rtype: HilbertState
        """
        rng = np.random.default_rng(seed=seed)
        out = self.copy()
        out.coefficients = rng.standard_normal(
            size=out.coefficients.shape
        ) + 1.0j * rng.standard_normal(size=out.coefficients.shape)
        out.coefficients /= np.linalg.norm(out.coefficients)
        return out

    def zero(self) -> "HilbertState":
        """Set all coefficients to zero.

        :return: A copy of ``HilbertState`` with all coefficients set to zero.
        :rtype: HilbertState
        """
        out = self.copy()
        out.coefficients = np.zeros_like(self.coefficients)
        return out

    def density(self) -> "HilbertOperator":
        """Calculate the density matrix for this state as an operator.

        :return: The density matrix
        :rtype: HilbertOperator
        """
        if self.ketwise:
            dens = self.outer(self.copy().dagger())
        else:
            dens = self.dagger().outer(self.copy())
        return dens

    def generate_basis_states(self) -> list:
        """Makes a list of corresponding basis vectors.

        :return: A list of tensor product states that span the Hilbert space.
        :rtype: list
        """
        coeffs_list = list(np.eye(self.n_basis**self.n_particles))
        out = [
            HilbertState(
                self.n_particles, coefficients=c, ketwise=True, isospsin=self.isospin
            )
            for c in coeffs_list
        ]
        return out

    def nearby_product_state(self, seed: int = None, maxiter=100):
        """Finds a ``ProductState`` that has a large overlap with the ``HilbertState``.

        :param seed: RNG seed, defaults to None
        :type seed: int, optional
        :param maxiter: Maximum iterations to do in optimization, defaults to 100
        :type maxiter: int, optional
        :return: A tuple: (fitted ``ProductState``, optimization result)
        :rtype: (ProductState, scipy.OptimizeResult)
        """
        from scipy.optimize import minimize, NonlinearConstraint

        fit = ProductState(self.n_particles, isospin=self.isospin).random(seed)
        shape = fit.coefficients.shape
        n_coeffs = len(fit.coefficients.flatten())
        n_params = 2 * n_coeffs

        def x_to_coef(x):
            return x[: n_params // 2].reshape(shape) + 1.0j * x[
                n_params // 2 :
            ].reshape(shape)

        def loss(x):
            fit.coefficients = x_to_coef(x)
            overlap = self.dagger().inner(fit.to_full_basis())
            return (1 - np.real(overlap)) ** 2 + np.imag(overlap) ** 2

        def norm(x):
            fit.coefficients = x_to_coef(x)
            return fit.dagger().inner(fit)

        start = np.concatenate(
            [np.real(fit.coefficients).flatten(), np.imag(fit.coefficients).flatten()]
        )
        print(start)
        normalize = NonlinearConstraint(norm, 1.0, 1.0)
        result = minimize(
            loss,
            x0=start,
            constraints=normalize,
            options={"maxiter": maxiter, "disp": True},
            tol=10**-15,
        )
        fit.coefficients = x_to_coef(result.x)
        return fit, result

    def nearest_product_state(self, seeds: list[int], maxiter=100) -> "ProductState":
        """Does ``self.nearby_product_state`` for a list of seeds and returns the result maximizing overlap

        :param seeds: List of RNG seeds
        :type seeds: list[int]
        :param maxiter: Maximum iterations to do in optimization, defaults to 100
        :type maxiter: int, optional
        :return: Fitted ``ProductState``
        :rtype: ProductState
        """
        overlap = 0.0
        for seed in seeds:
            this_fit, _ = self.nearby_product_state(seed, maxiter=maxiter)
            this_overlap = this_fit.to_full_basis().
