import nuctest as nt
from quap import *

one = OneBodyBasisSpinIsospinOperator(2)
sigx0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
sigy0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
sigz0 = OneBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
sigx1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'x')
sigy1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'y')
sigz1 = OneBodyBasisSpinIsospinOperator(2).sigma(1, 'z')
sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]
taux0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'x')
tauy0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'y')
tauz0 = OneBodyBasisSpinIsospinOperator(2).tau(0, 'z')
taux1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'x')
tauy1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'y')
tauz1 = OneBodyBasisSpinIsospinOperator(2).tau(1, 'z')
tau0vec = [taux0, tauy0, tauz0]
tau1vec = [taux1, tauy1, tauz1]


def Ggauss_1d_sample(dt: float, A: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.copy().scalar_mult(i, np.cosh(k * x)).scalar_mult(j, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x)) * opj.scalar_mult(j, np.sinh(k * x))
    return out.spread_scalar_mult(norm)


def gaussian_test():
    coeffs = read_sp('../data/h2_sp.dat').reshape(8, 1)
    print(coeffs)
    sp_i = OneBodyBasisSpinIsospinState(2, 'ket', coeffs)
    print(f"initial = {sp_i}")
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    x = 1.0
    sp_f = sp_i.copy()
    idx = [0, 1, 2]
    for a in idx:
        for b in idx:
            sp_f = Ggauss_1d_sample(nt.dt, Asig[a, b], x, 0, 1, sig0vec[a], sig1vec[b]) * sp_f
    print(f"final = {sp_f}")


def Grbm_sample(dt, A, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(A) * np.sinh(arg))
    return out.spread_scalar_mult(norm)


def rbm_test():
    bra, ket = nt.make_test_states()
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    h = 1.0
    idx = [0, 1, 2]
    for a in idx:
        for b in idx:
            ket = Grbm_sample(nt.dt, Asig[a, b], h, 0, 1, sig0vec[a], sig1vec[b]) * ket
    b = bra * ket
    print(f"final = {b}")


if __name__ == "__main__":
    rbm_test()
    print('DONE')
