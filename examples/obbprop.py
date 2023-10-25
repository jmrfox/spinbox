from quap import *

num_particles = 2

one = OneBodyBasisSpinOperator(num_particles)
sigx0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'x')
sigy0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'y')
sigz0 = OneBodyBasisSpinOperator(num_particles).sigma(0, 'z')
sigx1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'x')
sigy1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'y')
sigz1 = OneBodyBasisSpinOperator(num_particles).sigma(1, 'z')

sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]


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


def test_brackets_0():
    print('< max | cosh - sinh sigx0 | ud >')
    dt = 0.01
    Amat = get_Amat()
    k = 0.5 * dt * Amat[0, 0]
    # sum brackets
    bra, ket = make_test_states()
    b1 = np.cosh(k) * bra * ket - np.sinh(k) * bra * sigx0 * ket
    print(b1)


def Gpade_sigma_mb(dt: float, Amat: np.ndarray):
    sigx0 = ManyBodyBasisSpinOperator(num_particles).sigma(0, 'x')
    sigy0 = ManyBodyBasisSpinOperator(num_particles).sigma(0, 'y')
    sigz0 = ManyBodyBasisSpinOperator(num_particles).sigma(0, 'z')
    sigx1 = ManyBodyBasisSpinOperator(num_particles).sigma(1, 'x')
    sigy1 = ManyBodyBasisSpinOperator(num_particles).sigma(1, 'y')
    sigz1 = ManyBodyBasisSpinOperator(num_particles).sigma(1, 'z')
    sig0vec = [sigx0, sigy0, sigz0]
    sig1vec = [sigx1, sigy1, sigz1]

    out = ManyBodyBasisSpinOperator(num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += Amat[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def Ggauss_1d_sample(dt: float, A: float, x, i: int, j: int, opi: OneBodyBasisSpinOperator, opj: OneBodyBasisSpinOperator):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    gi = one.scalar_mult(i, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x))
    gj = one.scalar_mult(j, np.cosh(k * x)) + opj.scalar_mult(j, np.sinh(k * x))
    return norm * gi * gj


def gaussian_brackets(n_samples=100, mix=False):
    print('HS brackets')
    dt = 0.01
    Amat = get_Amat(diag=False)

    bra, ket = make_test_states()

    # correct answer via pade
    b_exact = bra.to_many_body_state() * Gpade_sigma_mb(dt, Amat) * ket.to_many_body_state()

    bra, ket = make_test_states()

    b_list = []
    x_set = rng.standard_normal(n_samples * 9)  # different x for each x,y,z
    # mix = False
    n = 0
    for i in range(n_samples):
        ket_p = ket.copy()
        ket_m = ket.copy()
        idx = [0, 1, 2]
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = Ggauss_1d_sample(dt, Amat[a, b], x_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Ggauss_1d_sample(dt, Amat[a, b], -x_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    # plt.figure(figsize=(5, 3))
    # plt.hist(np.real(b_list), label='Re', alpha=0.6, bins=20)
    # plt.hist(np.imag(b_list), label='Im', alpha=0.6, bins=20)
    # plt.title(f'<G(gauss)>')
    # plt.legend()
    # plt.show()

    b_gauss = np.mean(b_list)
    print('exact = ', b_exact)
    print('gauss = ', b_gauss)
    print('error = ', b_exact - b_gauss)


def Grbm_1d_sample(dt, A, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * one + np.sinh(arg) * opi
    qj = np.cosh(arg) * one - np.sign(A) * np.sinh(arg) * opj
    return norm * qi * qj


def rbm_brackets(n_samples=100, mix=False):
    print('RBM brackets')
    dt = 0.01
    Amat = get_Amat()

    bra, ket = make_test_states()

    # correct answer via pade
    b_exact = bra.to_many_body_state() * Gpade_sigma_mb(dt, Amat) * ket.to_many_body_state()

    # make population of identical wfns
    # n_samples = 1000
    b_list = []
    h_set = rng.integers(0, 2, n_samples * 9)
    # mix = True
    n = 0
    idx = [0, 1, 2]
    for i in range(n_samples):
        ket_p = ket.copy()
        ket_m = ket.copy()
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = 2 * Grbm_1d_sample(dt, Amat[a, b], h_set[n], sig0vec[a], sig1vec[b]) * ket_p
                ket_m = 2 * Grbm_1d_sample(dt, Amat[a, b], 1 - h_set[n], sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    b_rbm = np.mean(b_list)
    print('exact = ', b_exact)
    print('rbm = ', b_rbm)
    print('error = ', b_exact - b_rbm)



if __name__ == "__main__":
    print('ONE-BODY BASIS PROPAGATORS')
    test_brackets_0()
    gaussian_brackets()
    # rbm_brackets()
    print('ONE-BODY BASIS PROPAGATORS DONE')

