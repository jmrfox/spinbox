from quap import *
from tqdm import tqdm

# do some tests for 2 particles in the spin-isospin one-body basis
# still only testing spin operators, no taus

num_particles = 2

one = OneBodyBasisSpinIsospinOperator(num_particles)
sigx0 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'x')
sigy0 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'y')
sigz0 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'z')
sigx1 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'x')
sigy1 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'y')
sigz1 = OneBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'z')

sig0vec = [sigx0, sigy0, sigz0]
sig1vec = [sigx1, sigy1, sigz1]

global_seed = 17
rng = default_rng(seed=global_seed)

def make_test_states(rng=None):
    """returns one body basis spin-isospin states for testing"""
    bra, ket = random_spinisospin_bra_ket(2, bra_seed=global_seed, ket_seed=global_seed+1)
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    return bra, ket

def make_test_states():
    coeffs_bra = np.concatenate([spinor4('max', 'bra'), spinor4('max', 'bra')], axis=1)
    coeffs_ket = np.concatenate([spinor4('up', 'ket'), spinor4('down', 'ket')], axis=0)
    bra = OneBodyBasisSpinIsospinState(2, 'bra', coeffs_bra)
    ket = OneBodyBasisSpinIsospinState(2, 'ket', coeffs_ket)
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
    sigx0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'x')
    sigy0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'y')
    sigz0 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(0, 'z')
    sigx1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'x')
    sigy1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'y')
    sigz1 = ManyBodyBasisSpinIsospinOperator(num_particles).sigma(1, 'z')
    sig0vec = [sigx0, sigy0, sigz0]
    sig1vec = [sigx1, sigy1, sigz1]

    out = ManyBodyBasisSpinIsospinOperator(num_particles).zeros()
    for a in range(3):
        for b in range(3):
            out += Amat[a, b] * sig0vec[a] * sig1vec[b]
    out = -0.5 * dt * out
    return out.exponentiate()


def Ggauss_1d_sample(dt: float, A: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.copy().scalar_mult(i, np.cosh(k*x)).scalar_mult(j, np.cosh(k*x)) + opi.scalar_mult(i, np.sinh(k*x)) * opj.scalar_mult(j, np.sinh(k*x))
    return out.spread_scalar_mult(norm)

def test_gaussian_sample():
    A = get_Amat()[0, 0]
    dt = 0.01
    bra, ket = make_test_states()
    x = 1.0
    out = bra * Ggauss_1d_sample(dt, A, x, 0, 1, sigx0, sigx1) * ket
    print(f'<Gx> = {out}')

def gaussian_brackets(n_samples=100, mix=False, plot=False):
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
    for i in tqdm(range(n_samples)):
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


def Grbm_1d_sample(dt, A, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    # qi = np.cosh(arg) * one + np.sinh(arg) * opi
    # qj = np.cosh(arg) * one - np.sign(A) * np.sinh(arg) * opj
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(A)*np.sinh(arg))
    return out.spread_scalar_mult(norm)

def test_rbm_sample():
    A = get_Amat()[0, 0]
    dt = 0.01
    bra, ket = make_test_states()
    x = 1.0
    out = bra * Grbm_1d_sample(dt, A, x, 0, 1, sigx0, sigx1) * ket
    print(f'<Gx> = {out}')


def rbm_brackets(n_samples=100, mix=False, plot=False):
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
    for i in tqdm(range(n_samples)):
        ket_p = ket.copy()
        ket_m = ket.copy()
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = 2 * (Grbm_1d_sample(dt, Amat[a, b], h_set[n],0, 1, sig0vec[a], sig1vec[b]) * ket_p)
                ket_m = 2 * (Grbm_1d_sample(dt, Amat[a, b], 1 - h_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m)
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



if __name__ == "__main__":
    print('ONE-BODY BASIS PROPAGATORS')
    test_brackets_0()

    n_samples = 10000
    mix = False
    test_gaussian_sample()
    gaussian_brackets(n_samples=n_samples, mix=mix)
    test_rbm_sample()
    rbm_brackets(n_samples=n_samples, mix=mix)
    print('ONE-BODY BASIS PROPAGATORS DONE')

