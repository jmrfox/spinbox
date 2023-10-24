from quap import *

num_particles = 2

id = OneBodyBasisSpinOperator(num_particles)
sigx0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'x')
sigy0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'y')
sigz0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'z')
sigx1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'x')
sigy1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'y')
sigz1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'z')
# sigx01 = sigx0 * sigx1
# sigy01 = sigy0 * sigy1
# sigz01 = sigz0 * sigz1

sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]


# def make_spinor(states):
#     """makes a 2*npart vector of single-particle spinor coefficients from input string
#     u = up , d = down
#     """
#     n = len(states)
#     c = np.zeros(2*n, dtype=complex)
#     for i in range(n):
#         if states[i] == 'u':
#             c[2*i] = 1
#         elif states[i] == 'd':
#             c[2*i+1] = 1
#         else:
#             raise ValueError('Unknown character')
#     return c
#
#
# def normalize_spinor(sp):
#     n = sp.shape[-1]
#     for i in range(n):
#         sp[:, i] = sp[:, i] / np.linalg.norm(sp[:, i])
#     return sp
#
#
# def twobody_spinor(states):
#     if states in ['uu', 'ud', 'du', 'dd']:
#         c = make_spinor(states)
#     elif states == 'Q':
#         # print('Q is an equal mixture of uu, ud, du, dd')
#         c = normalize_spinor(make_spinor('uu') + make_spinor('dd'))
#     elif states == 'R':
#         # print('R is an asymmetric mixture. See function definition.')
#         # c = normalize_spinor(np.sqrt(0.75)*make_spinor('ud') - np.sqrt(0.25)*make_spinor('du'))
#         q1 = 0.5
#         q2 = 0.25
#         c1u = np.sqrt(q1)
#         c1d = -np.sqrt(1 - q1)
#         c2u = np.sqrt(q2)
#         c2d = np.sqrt(1 - q2)
#         c = np.array([c1u, c2u, c1d, c2d], dtype=complex)
#     else:
#         raise ValueError('Unknown character')
#     return c


# def make_state(s, orientation, quiet=True):
#     """s is a string like uu is up up, ud is up down"""
#     p = twobody_spinor(s)
#     if orientation == 'ket':
#         return OneBodyBasisSpinState(num_particles=2, orientation=orientation, coefficients=p.reshape((4, 1)))
#     elif orientation == 'bra':
#         return OneBodyBasisSpinState(num_particles=2, orientation=orientation, coefficients=p.reshape((1, 4)))
#     else:
#         raise ValueError('Invalid orientation')

def make_test_states():
    coeffs_bra = np.concatenate([spinor2('max', 'bra'), spinor2('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor2('up', 'ket'), spinor2('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinState(2, 'bra', coeffs_bra)
    ket = OneBodyBasisSpinState(2, 'ket', coeffs_ket)
    return bra, ket

def get_Amat(diag=False):
    """returns Amat for testing"""
    Amat = np.zeros((3, 3))
    Amat[0, 0] = 10.0
    Amat[1, 1] = -20.0
    Amat[2, 2] = 30.0
    if not diag:
        Amat[0, 1] = 5.0
        Amat[0, 2] = -6.0
        Amat[1, 2] = 7.0
        Amat[1, 0] = Amat[0, 1]
        Amat[2, 0] = Amat[0, 2]
        Amat[2, 1] = Amat[1, 2]
    return Amat


def Bexact_1d(bra, ket, opi, opj, dt, A):
    k = 0.5 * dt * A
    sp = ket.copy()
    return np.cosh(k) * bra * ket - np.sinh(k) * bra * opi * opj * ket


def test_brackets_1():
    print('< uu | cosh - sinh sigx1 sigx2 | R >')
    dt = 0.01
    Amat = get_Amat()
    kx = 0.5 * dt * Amat[0, 0]

    # sum brackets
    bra, ket = make_test_states()
    b1 = np.cosh(kx) * (bra * ket) - np.sinh(kx) * (bra * sigx0 * sigx1 * ket)
    print('sum brackets: ', b1)

    # sum wavefunctions
    bra, ket = make_test_states()
    ket = np.cosh(kx) * ket - np.sinh(kx) * sigx0 * sigx1 * ket
    b2 = bra * ket
    print('sum wavefunctions: ', b2)


if __name__ == "__main__":
    test_brackets_1()