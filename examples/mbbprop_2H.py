import numpy as np

from quap import *

one = ManyBodyBasisSpinIsospinOperator(2)
sigx0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'x')
sigy0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'y')
sigz0 = ManyBodyBasisSpinIsospinOperator(2).sigma(0, 'z')
sigx1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'x')
sigy1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'y')
sigz1 = ManyBodyBasisSpinIsospinOperator(2).sigma(1, 'z')
sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]
taux0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'x')
tauy0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'y')
tauz0 = ManyBodyBasisSpinIsospinOperator(2).tau(0, 'z')
taux1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'x')
tauy1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'y')
tauz1 = ManyBodyBasisSpinIsospinOperator(2).tau(1, 'z')
tau0vec = [taux0, tauy0, tauz0]
tau1vec = [taux1, tauy1, tauz1]


def make_test_states():
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket).to_many_body_state()
    return bra, ket


def random_A_matrices():
    """this makes the same random matrices every time
    for 2-particle systems, we only need the i=1, j=2 slice of the matrices
    so, Asig[a,b] , Asigtau[a,b,c] , Atau[c] for a,b,c = 1,2,3
    """
    rng = np.random.default_rng(1312)
    spread = 10
    Asig = spread * rng.standard_normal(size=(3, 3))
    Asigtau = spread * rng.standard_normal(size=(3, 3, 3))
    Atau = spread * rng.standard_normal(size=3)
    return Asig, Asigtau, Atau

def make_A():
    a = 3.14
    Asig = a * np.ones((3,3))
    Asigtau = 0. * np.ones((3, 3, 3))
    Atau = 0. * np.ones(3)
    return Asig, Asigtau, Atau

def Gpade_sigma(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            out += Amat[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def Gpade_sigmatau(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for a in range(3):
        for b in range(3):
            for c in range(3):
                out += Amat[a, b, c] * sig0vec[a] * sig1vec[b] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()

def Gpade_tau(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += Amat[c] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()

def test_bracket():
    # exact bracket over sigma, sigma-tau, tau
    Asig, Asigtau, Atau = random_A_matrices()
    dt = 0.01
    bra, ket = make_test_states()
    Gsig = Gpade_sigma(dt, Asig)
    Gsigtau = Gpade_sigmatau(dt, Asigtau)
    Gtau = Gpade_tau(dt, Atau)
    Bexact = bra * Gtau * Gsigtau * Gsig * ket
    print(f"B exact = {Bexact}")


if __name__ == "__main__":
    coeffs = read_sp('../data/h2_sp.dat').reshape(8, 1)
    print(coeffs)
    sp_i = OneBodyBasisSpinIsospinState(2, 'ket', coeffs).to_many_body_state()
    print(f"initial = {sp_i}")
    Asig, Asigtau, Atau = make_A()
    dt = 0.00001
    Gsig = Gpade_sigma(dt, Asig)
    sp_f = Gsig * sp_i
    print(f"final = {sp_f}")

    print('DONE')