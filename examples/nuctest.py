from quap import *

dt = 0.001

def make_test_states():
    """returns one body basis spin-isospin states for testing"""
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra)
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket)
    return bra, ket


def make_A_matrices(random=False):
    if random:
        def random_A_matrices():
            """this makes the same random matrices every time
            for 2-particle systems, we only need the i=1, j=2 slice of the matrices
            so, Asig[a,b] , Asigtau[a,b,c] , Atau[c] for a,b,c = 1,2,3
            """
            rng = np.random.default_rng(2023)
            spread = 10
            Asig = spread * rng.standard_normal(size=(3, 3))
            Asigtau = spread * rng.standard_normal(size=(3, 3, 3))
            Atau = spread * rng.standard_normal(size=3)
            return Asig, Asigtau, Atau
        return random_A_matrices()
    else:
        a = 3.14
        Asig = a * np.ones((3,3))
        Asigtau = a * np.ones((3, 3, 3))
        Atau = a * np.ones(3)
        return Asig, Asigtau, Atau

def make_potentials(random=False):
    Asig, Asigtau, Atau = make_A_matrices(random=random)
    Vcoul = 1.0
    return Asig, Asigtau, Atau, Vcoul