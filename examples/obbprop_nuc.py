import matplotlib.pyplot as plt

import nuctest as nt
from quap import *
from tqdm import tqdm
from cProfile import Profile
from pstats import SortKey, Stats

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

def plot_samples(X, range, filename, title):
    plt.figure(figsize=(5, 3))
    plt.hist(np.real(X), label='Re', alpha=0.5, bins=30, range=range, color='red')
    plt.hist(np.imag(X), label='Im', alpha=0.5, bins=30, range=range, color='blue')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

def Ggauss_sample(dt: float, A: float, x, i: int, j: int, opi: OneBodyBasisSpinIsospinOperator, opj: OneBodyBasisSpinIsospinOperator):
    k = np.sqrt(-0.5 * dt * A, dtype=complex)
    norm = np.exp(0.5 * dt * A)
    op = OneBodyBasisSpinIsospinOperator(2)
    out = op.copy().scalar_mult(i, np.cosh(k * x)).scalar_mult(j, np.cosh(k * x)) + opi.scalar_mult(i, np.sinh(k * x)) * opj.scalar_mult(j, np.sinh(k * x))
    return out.spread_scalar_mult(norm)

def test_gaussian_sample():
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()
    x = 1.0
    out = bra * Ggauss_sample(nt.dt, Asig[0, 0], x, 0, 1, sigx0, sigx1) * ket
    print(f'gaussian test = {out}')

def gaussian_brackets_1(n_samples=100, mix=False, plot=False):
    print('HS brackets')
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()

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
                ket_p = Ggauss_sample(nt.dt, Asig[a, b], x_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Ggauss_sample(nt.dt, Asig[a, b], -x_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        for a in idx:
            for b in idx:
                for c in idx:
                    ket_p = Ggauss_sample(nt.dt, Asigtau[a, b, c], x_set[n], 0, 1, sig0vec[a]*tau0vec[c], sig1vec[b]*tau1vec[c]) * ket_p
                    ket_m = Ggauss_sample(nt.dt, Asigtau[a, b, c], -x_set[n], 0, 1, sig0vec[a]*tau0vec[c], sig1vec[b]*tau1vec[c]) * ket_m
                    n += 1
        for c in idx:
            ket_p = Ggauss_sample(nt.dt, Atau[c], x_set[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
            ket_m = Ggauss_sample(nt.dt, Atau[c], -x_set[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
            n += 1

        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plot_samples(b_list, range=(-5, 8), filename='gaussian_brackets_ob_1.pdf', title=f'<G(gauss)>')

    b_gauss = np.mean(b_list)
    print('HS < Gsig Gsigtau Gtau > = ', b_gauss)


def Grbm_sample(dt, A, h, i, j, opi, opj):
    norm = np.exp(-0.5 * dt * np.abs(A)) / 2
    W = np.arctanh(np.sqrt(np.tanh(0.5 * dt * np.abs(A))))
    arg = W * (2 * h - 1)
    out = one.scalar_mult(i, np.cosh(arg)).scalar_mult(j, np.cosh(arg)) + opi.scalar_mult(i, np.sinh(arg)) * opj.scalar_mult(j, -np.sign(A) * np.sinh(arg))
    return out.spread_scalar_mult(2*norm)
    # return out.scalar_mult(0, 2*norm)


def test_rbm_sample():
    bra, ket = nt.make_test_states()
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    out = 0.5 * (bra * Grbm_sample(nt.dt, Asig[0, 0], 0, 0, 1, sigx0, sigx1) * ket + bra * Grbm_sample(nt.dt, Asig[0, 0], 1, 0, 1, sigx0, sigx1) * ket)
    print(f"rbm test = {out}")




def rbm_brackets_1(n_samples=100, mix=False, plot=False):
    print('RBM brackets')
    Asig, Asigtau, Atau = nt.make_A_matrices(random=True)
    bra, ket = nt.make_test_states()

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
                ket_p = Grbm_sample(nt.dt, Asig[a, b], h_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_p
                ket_m = Grbm_sample(nt.dt, Asig[a, b], 1-h_set[n], 0, 1, sig0vec[a], sig1vec[b]) * ket_m
                n += 1
        for a in idx:
            for b in idx:
                for c in idx:
                    ket_p = Grbm_sample(nt.dt, Asigtau[a, b, c], h_set[n], 0, 1, sig0vec[a]*tau0vec[c], sig1vec[b]*tau1vec[c]) * ket_p
                    ket_m = Grbm_sample(nt.dt, Asigtau[a, b, c], 1-h_set[n], 0, 1, sig0vec[a]*tau0vec[c], sig1vec[b]*tau1vec[c]) * ket_m
                    n += 1
        for c in idx:
            ket_p = Grbm_sample(nt.dt, Atau[c], h_set[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_p
            ket_m = Grbm_sample(nt.dt, Atau[c], 1-h_set[n], 0, 1, tau0vec[c], tau1vec[c]) * ket_m
            n += 1
        b_list.append(bra * ket_p)
        b_list.append(bra * ket_m)

    if plot:
        plot_samples(b_list, range=(-5,8), filename='rbm_brackets_ob_1.pdf', title=f'<G(rbm)>')

    b_rbm = np.mean(b_list)
    print('HS < Gsig Gsigtau Gtau > = ', b_rbm)



if __name__ == "__main__":
    # test_gaussian_sample()
    # test_rbm_sample()

    n_samples = 5000
    with Profile() as profile:
        gaussian_brackets_1(n_samples=n_samples, plot=True)
        rbm_brackets_1(n_samples=n_samples, plot=True)
        # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

    print('DONE')
