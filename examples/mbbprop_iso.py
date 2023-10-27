from quap import *
from tqdm import tqdm

num_particles = 2

one = ManyBodyBasisSpinIsospinOperator(num_particles)
sigx0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'x')
sigy0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'y')
sigz0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'z')
sigx1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'x')
sigy1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'y')
sigz1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'z')
# sigx01 = sigx0 * sigx1
# sigy01 = sigy0 * sigy1
# sigz01 = sigz0 * sigz1

sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]


def make_test_states():
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra).to_many_body_state()
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket).to_many_body_state()
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
    print(f'summing brackets : {b1}')

    # sum wavefunctions
    bra, ket = make_test_states()
    ket = np.cosh(k) * ket - np.sinh(k) * sigx0 * ket
    b2 = bra * ket
    print(f'summing wavefunctions : {b2}')

def test_brackets_1():
    print('< uu | cosh - sinh sigx1 sigx2 | R >')
    dt = 0.01
    Amat = get_Amat()
    k = 0.5 * dt * Amat[0, 0]
    # sum brackets
    bra, ket = make_test_states()
    b1 = np.cosh(k) * bra * ket - np.sinh(k) * bra * sigx0 * sigx1 * ket
    print(b1)

    # sum wavefunctions
    bra, ket = make_test_states()
    ket = np.cosh(k) * ket - np.sinh(k) * sigx0 * sigx1 * ket
    b2 = bra * ket
    print(b2)


def test_brackets_2():
    print('< uu | (cosh - sinh sigx1 sigx2)(cosh - sinh sigy1 sigy2) | R >')
    dt = 0.01
    Amat = get_Amat()
    K = 0.5 * dt * Amat

    bra, ket = make_test_states()
    b1 = (np.cosh(K[0, 0]) * np.cosh(K[1, 1]) * bra * ket
          - np.cosh(K[0, 0]) * np.sinh(K[1, 1]) * prod([bra, sigy0, sigy1, ket])
          - np.cosh(K[1, 1]) * np.sinh(K[0, 0]) * prod([bra, sigx0, sigx1, ket])
          + np.sinh(K[0, 0]) * np.sinh(K[1, 1]) * prod([bra, sigx0, sigx1, sigy0, sigy1, ket]))
    print(b1)

    bra, ket = make_test_states()
    ket = np.cosh(K[0, 0]) * ket - np.sinh(K[0, 0]) * prod([sigx0, sigx1, ket])
    ket = np.cosh(K[1, 1]) * ket - np.sinh(K[1, 1]) * prod([sigy0, sigy1, ket])
    b2 = bra * ket
    print(b2)


def Gpade_sigma(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += Amat[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def Bpade_sigma(bra, ket, dt, Amat):
    out = 0
    Kmat = -0.5 * dt * Amat
    for a in range(3):
        for b in range(3):
            out += np.cosh(Kmat[a, b]) * bra * ket + np.sinh(Kmat[a, b]) * bra * sig0vec[a] * sig1vec[b] * ket
    return out

def Ggauss_1d_sample(dt, A, x, opi, opj):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    gi = np.cosh(k * x) * one + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * one + np.sinh(k * x) * opj
    return norm * gi * gj

def Bgauss_1d_sample(bra, ket, dt, A, x, opi, opj):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    bi = np.cosh(k * x) * bra * ket + np.sinh(k * x) * bra * opi * ket
    bj = np.cosh(k * x) * bra * ket + np.sinh(k * x) * bra * opj * ket
    return norm * bi * bj

def test_gaussian_sample():
    A = get_Amat()[0, 0]
    dt = 0.01
    bra, ket = make_test_states()
    x = 1.0
    out = bra * Ggauss_1d_sample(dt, A, x, sigx0, sigx1) * ket
    print(f'<Gx> = {out}')
    # print(f'Bx = {Bgauss_1d_sample(bra, ket, dt, A, x, sigx0, sigx1)}')

def gaussian_brackets(n_samples=100, mix=False, plot=False):
    print('HS brackets')
    dt = 0.01
    Amat = get_Amat(diag=False)

    bra, ket = make_test_states()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(dt, Amat) * ket

    b_list = []
    x_set = rng.standard_normal(n_samples * 9)  # different x for each x,y,z
    n = 0
    for i in tqdm(range(n_samples)):
        ket_p = ket.copy()
        ket_m = ket.copy()
        idx = [0, 1, 2]
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = Ggauss_1d_sample(dt, Amat[a, b], x_set[n], sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Ggauss_1d_sample(dt, Amat[a, b], -x_set[n], sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plt.figure(figsize=(5, 3))
        plt.hist(np.real(b_list), label='Re', alpha=0.6, bins=20)
        plt.hist(np.imag(b_list), label='Im', alpha=0.6, bins=20)
        plt.title(f'<G(gauss)>')
        plt.legend()
        plt.show()

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

def test_rbm_sample():
    A = get_Amat()[0, 0]
    dt = 0.01
    bra, ket = make_test_states()
    out = bra * Grbm_1d_sample(dt, A, 0, sigx0, sigx1) * ket + bra * Grbm_1d_sample(dt, A, 1, sigx0, sigx1) * ket
    print(f'<Gx> = {out}')

def rbm_brackets(n_samples=100, mix=False):
    print('RBM brackets')
    dt = 0.01
    Amat = get_Amat()

    bra, ket = make_test_states()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(dt, Amat) * ket

    # make population of identical wfns
    b_list = []
    h_set = rng.integers(0, 2, n_samples * 9)
    n = 0
    idx = [0, 1, 2]
    for i in tqdm(range(n_samples)):
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

    if plot:
        plt.figure(figsize=(5, 3))
        plt.hist(np.real(b_list), label='Re', alpha=0.6, bins=20)
        plt.hist(np.imag(b_list), label='Im', alpha=0.6, bins=20)
        plt.title(f'<G(rbm)>')
        plt.legend()
        plt.show()

    b_rbm = np.mean(b_list)
    print('exact = ', b_exact)
    print('rbm = ', b_rbm)
    print('error = ', b_exact - b_rbm)


if __name__ == '__main__':
    test_brackets_0()
    test_brackets_1()
    test_brackets_2()

    n_samples = 1000
    mix = False

    test_gaussian_sample()
    gaussian_brackets(n_samples=n_samples, mix=mix)

    test_rbm_sample()
    rbm_brackets(n_samples=n_samples, mix=mix)
