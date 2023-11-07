import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats


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

def plot_samples(X, range, filename, title):
    plt.figure(figsize=(5, 3))
    plt.hist(np.real(X), label='Re', alpha=0.5, bins=30, range=range, color='red')
    plt.hist(np.imag(X), label='Im', alpha=0.5, bins=30, range=range, color='blue')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

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

def Gpade_coul(dt, Amat):
    out = ManyBodyBasisSpinIsospinOperator(2).zeros()
    for c in range(3):
        out += Amat[c] * tau0vec[c] * tau1vec[c]
    out = -0.5 * dt * out
    return out.exponentiate()

def test_bracket_1():
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    Gsig = Gpade_sigma(nt.dt, Asig)
    b = bra * Gsig * ket
    print(f"< Gsig > = {b}")


def test_bracket_2():
    # exact bracket over sigma, sigma-tau, tau
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    Gsig = Gpade_sigma(nt.dt, Asig)
    Gsigtau = Gpade_sigmatau(nt.dt, Asigtau)
    Gtau = Gpade_tau(nt.dt, Atau)
    b = bra * Gtau * Gsigtau * Gsig * ket
    print(f"< Gsig Gsigtau Gtau > = {b}")


def Ggauss_sample(dt, A, x, opi, opj):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    gi = np.cosh(k * x) * one + np.sinh(k * x) * opi
    gj = np.cosh(k * x) * one + np.sinh(k * x) * opj
    return norm * gi * gj


def Bgauss_sample(bra, ket, dt, A, x, opi, opj):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    bi = np.cosh(k * x) * bra * ket + np.sinh(k * x) * bra * opi * ket
    bj = np.cosh(k * x) * bra * ket + np.sinh(k * x) * bra * opj * ket
    return norm * bi * bj


def test_gaussian_sample():
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    x = 1.0
    out = bra * Ggauss_sample(nt.dt, Asig[0, 0], x, sigx0, sigx1) * ket
    print(f'gaussian test = {out}')


def gaussian_brackets_1(n_samples=100, mix=False, plot=False):
    print('HS brackets with sig, sigtau, tau')
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(nt.dt, Asig) * Gpade_sigmatau(nt.dt, Asigtau) * Gpade_tau(nt.dt, Atau) * ket

    b_list = []
    x_set = rng.standard_normal(n_samples * 40)  # different x for each x,y,z
    n = 0
    for i in tqdm(range(n_samples)):
        ket_p = ket.copy()
        ket_m = ket.copy()
        idx = [0, 1, 2]
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = Ggauss_sample(nt.dt, Asig[a, b], x_set[n], sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Ggauss_sample(nt.dt, Asig[a, b], -x_set[n], sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        for a in idx:
            for b in idx:
                for c in idx:
                    ket_p = Ggauss_sample(nt.dt, Asigtau[a, b, c], x_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                    ket_m = Ggauss_sample(nt.dt, Asigtau[a, b, c], -x_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                    n += 1
        for c in idx:
            ket_p = Ggauss_sample(nt.dt, Atau[c], x_set[n], tau0vec[c], tau1vec[c]) * ket_p
            ket_m = Ggauss_sample(nt.dt, Atau[c], -x_set[n], tau0vec[c], tau1vec[c]) * ket_m
            n += 1

        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plot_samples(b_list, range=(-5, 8), filename='gaussian_brackets_mb_1.pdf', title=f'<G(gauss)>')

    b_gauss = np.mean(b_list)
    print('exact = ', b_exact)
    print('gauss = ', b_gauss)
    print('error = ', b_exact - b_gauss)


def gaussian_brackets_2(n_samples=100, mix=False, plot=False):
    print('HS brackets with sig, sigtau, tau, coulomb')
    Asig, Asigtau, Atau, Vcoul = nt.make_potentials(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(nt.dt, Asig) * Gpade_sigmatau(nt.dt, Asigtau) * Gpade_tau(nt.dt, Atau) * ket

    b_list = []
    x_set = rng.standard_normal(n_samples * 40)  # different x for each x,y,z
    n = 0
    for i in tqdm(range(n_samples)):
        ket_p = ket.copy()
        ket_m = ket.copy()
        idx = [0, 1, 2]
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = Ggauss_sample(nt.dt, Asig[a, b], x_set[n], sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Ggauss_sample(nt.dt, Asig[a, b], -x_set[n], sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        for a in idx:
            for b in idx:
                for c in idx:
                    ket_p = Ggauss_sample(nt.dt, Asigtau[a, b, c], x_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                    ket_m = Ggauss_sample(nt.dt, Asigtau[a, b, c], -x_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                    n += 1
        for c in idx:
            ket_p = Ggauss_sample(nt.dt, Atau[c], x_set[n], tau0vec[c], tau1vec[c]) * ket_p
            ket_m = Ggauss_sample(nt.dt, Atau[c], -x_set[n], tau0vec[c], tau1vec[c]) * ket_m
            n += 1

        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plot_samples(b_list, range=(-5, 8), filename='gaussian_brackets_mb_1.pdf', title=f'<G(gauss)>')

    b_gauss = np.mean(b_list)
    print('exact = ', b_exact)
    print('gauss = ', b_gauss)
    print('error = ', b_exact - b_gauss)


def Grbm_sample(dt, A, h, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    qi = np.cosh(arg) * one + np.sinh(arg) * opi
    qj = np.cosh(arg) * one - np.sign(A) * np.sinh(arg) * opj
    return 2 * norm * qi * qj


def test_rbm_sample():
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()
    out = 0.5 * (bra * Grbm_sample(nt.dt, Asig[0, 0], 0, sigx0, sigx1) * ket + bra * Grbm_sample(nt.dt, Asig[0, 0], 1, sigx0, sigx1) * ket)
    print(f'rbm test = {out}')


def rbm_brackets_1(n_samples=100, mix=False, plot=False):
    print('RBM brackets')
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    bra = bra.to_many_body_state()
    ket = ket.to_many_body_state()

    # correct answer via pade
    b_exact = bra * Gpade_sigma(nt.dt, Asig) * Gpade_sigmatau(nt.dt, Asigtau) * Gpade_tau(nt.dt, Atau) * ket

    # make population of identical wfns
    b_list = []
    h_set = rng.integers(0, 2, n_samples * 40)
    n = 0
    idx = [0, 1, 2]
    for _ in tqdm(range(n_samples)):
        ket_p = ket.copy()
        ket_m = ket.copy()
        if mix:
            rng.shuffle(idx)
        for a in idx:
            for b in idx:
                ket_p = Grbm_sample(nt.dt, Asig[a, b], h_set[n], sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Grbm_sample(nt.dt, Asig[a, b], 1 - h_set[n], sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        for a in idx:
            for b in idx:
                for c in idx:
                    ket_p = Grbm_sample(nt.dt, Asigtau[a, b, c], h_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_p
                    ket_m = Grbm_sample(nt.dt, Asigtau[a, b, c], 1 - h_set[n], sig0vec[a] * tau0vec[c], sig1vec[b] * tau1vec[c]) * ket_m
                    n += 1
        for c in idx:
            ket_p = Grbm_sample(nt.dt, Atau[c], h_set[n], tau0vec[c], tau1vec[c]) * ket_p
            ket_m = Grbm_sample(nt.dt, Atau[c], 1 - h_set[n], tau0vec[c], tau1vec[c]) * ket_m
            n += 1
        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plot_samples(b_list, range=(-5, 8), filename='rbm_brackets_mb_1.pdf', title=f'<G(rbm)>')

    b_rbm = np.mean(b_list)
    print('exact = ', b_exact)
    print('rbm = ', b_rbm)
    print('error = ', b_exact - b_rbm)


if __name__ == "__main__":
    # test_bracket_1()
    # test_bracket_2()
    # test_gaussian_sample()
    # test_rbm_sample()

    n_samples = 5000
    with Profile() as profile:
        gaussian_brackets_1(n_samples=n_samples, plot=True)
        rbm_brackets_1(n_samples=n_samples, plot=True)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()
    print('DONE')
